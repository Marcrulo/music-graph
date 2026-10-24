"""Do the expensive features add information? Across-track variance analysis of benchmark outputs.

Reads every data/bench/track_<id>.json written by benchmark_all_features.py and, per catalogue
feature (and per costly block of features sharing a prerequisite such as demucs):

  cost     seconds per minute of audio, including prerequisites (e.g. vocal features pay for demucs + crepe)
  unique   1 - cross-validated R^2 (clipped to [0, 1]) of predicting the feature's principal components from all *other*
           features (ridge on a PCA of the rest). 0 = fully redundant, 1 = nothing else predicts it.
  share    its share of the variance in a global PCA where every feature has equal total weight
  eff_dim  participation ratio of the feature's own PCA spectrum (how many directions it really uses)

Columns are rank-transformed to normal scores first. With few tracks R^2 is underestimated, so `unique` is biased upwards; a random-noise control row
shows the ceiling. Needs >= 10 tracks to say anything, ideally 50+.

Usage: uv run python experiments/feature_variance.py [--dir data/bench] [--top 40]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

DEMUCS = "7. separation/Stem features (vocals, drums, bass, other)"
CREPE = "7. separation/Speechiness"  # its time includes the torchcrepe pass the vocal features reuse

# features whose cost includes another feature's time (shared intermediates computed on first use)
PREREQS = {
    "7. separation/Vocal activity curve": [DEMUCS],
    "7. separation/Vocal presence / instrumentalness": [DEMUCS],
    "7. separation/Speechiness": [DEMUCS],
    "7. separation/Stem energy balance": [DEMUCS],
    "7. separation/Per-stem feature sets": [DEMUCS],
    "4. harmony/Bass line pitch / root motion": [DEMUCS],
    "8. semantic/Language of vocals": [DEMUCS],
    "16. non-audio/Lyrics embeddings, lyric themes": [],
    **{f"15. vocal/{n}": [DEMUCS, CREPE] for n in [
        "Vocal pitch range, median pitch", "Vibrato rate and extent", "Breathiness, jitter, shimmer, HNR",
        "Rap vs. sung ratio", "Syllable / word rate", "Harmony vocals / choir presence"]},
    "11. psychoacoustic/Sharpness": ["11. psychoacoustic/Zwicker loudness (sones), specific loudness"],
    "5. rhythm/Meter / time signature": ["5. rhythm/Beat and downbeat positions"],
    "5. rhythm/Rhythmic density per bar": ["5. rhythm/Beat and downbeat positions"],
}

# blocks: what we'd lose by dropping one costly component entirely
BLOCKS = {
    "[block] everything needing demucs": lambda k: k == DEMUCS or DEMUCS in PREREQS.get(k, []),
    "[block] mosqito psychoacoustics": lambda k: k.startswith("11. psychoacoustic/") and any(
        s in k for s in ("Zwicker", "Sharpness", "Roughness", "Tonality")),
    "[block] madmom downbeats": lambda k: k == "5. rhythm/Beat and downbeat positions"
                                          or "5. rhythm/Beat and downbeat positions" in PREREQS.get(k, []),
    "[block] all learned embeddings (9.)": lambda k: k.startswith("9. learned/"),
}

SKIP = {"16. non-audio/Artist", "16. non-audio/Title"}  # identity, not features


def numeric_leaves(x, path=""):
    """Flatten a feature's value dict into (name, float) pairs; strings and ragged 2D data are skipped."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numeric_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(x, bool):
        yield path, float(x)
    elif isinstance(x, (int, float)):
        yield path, float(x)
    elif isinstance(x, list) and x and not isinstance(x[0], (list, dict, str)):
        for i, v in enumerate(x):
            if isinstance(v, (int, float)):
                yield f"{path}[{i}]", float(v)


def rank_gauss(X):
    """Per column: ranks -> normal scores, so a few extreme tracks can't dominate a PCA or a fit."""
    from scipy.stats import norm, rankdata

    R = rankdata(X, axis=0)
    Z = norm.ppf((R - 0.5) / len(X))
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-12)


def load(bench_dir: str):
    files = sorted(f for f in glob.glob(os.path.join(bench_dir, "track_*.json"))
                   if re.fullmatch(r"track_\d+\.json", os.path.basename(f)))
    tracks = [json.load(open(f)) for f in files]
    feats, costs = {}, {}
    for t in tracks:
        minutes = t["duration_s"] / 60
        for r in t["results"]:
            if r["status"] == "ok" and r["cat"] not in ("shared", "model load"):
                costs.setdefault(f"{r['cat']}/{r['feature']}", []).append(r["seconds"] / minutes)
        for k, v in t["values"].items():
            if k not in SKIP:
                feats.setdefault(k, []).append(dict(numeric_leaves(v)))
    n = len(tracks)
    mats = {}
    for k, rows in feats.items():
        if len(rows) != n:
            continue  # failed on some track
        cols = sorted(set.intersection(*(set(r) for r in rows)))
        if not cols:
            continue
        X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
        X[~np.isfinite(X)] = np.nan
        X = X[:, ~np.isnan(X).all(0)]
        X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
        X = X[:, X.std(0) > 1e-9 * (np.abs(X).mean(0) + 1e-12)]
        if X.shape[1]:
            mats[k] = rank_gauss(X)
    cost = {k: float(np.mean(v)) for k, v in costs.items()}
    return n, mats, cost


def cost_with_prereqs(k, cost):
    return cost.get(k, 0.0) + sum(cost.get(p, 0.0) for p in PREREQS.get(k, []) if p != k)


def block_cost(members, cost):
    paid = set(members) | {p for m in members for p in PREREQS.get(m, [])}
    return sum(cost.get(p, 0.0) for p in paid)


def weighted(mats, keys):
    """Concatenate standardized features, each scaled to equal total variance."""
    return np.hstack([mats[k] / np.sqrt(mats[k].shape[1]) for k in keys])


def unique_variance(Y, X_rest, n_folds, seed=0):
    """1 - CV R^2 of Y's principal components (90% var, <=10) predicted from X_rest."""
    n = len(Y)
    pca_y = PCA(n_components=min(Y.shape[1], n - 2, 10)).fit(Y)
    k = int(np.searchsorted(np.cumsum(pca_y.explained_variance_ratio_), 0.9) + 1)
    T = pca_y.transform(Y)[:, :k]
    Z = PCA(n_components=min(X_rest.shape[1], n - 2, 50)).fit_transform(X_rest)
    cv = KFold(n_folds, shuffle=True, random_state=seed)
    # PCA of the predictors inside the folds would be stricter; with ridge the leak is small
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 25)), Z, T, cv=cv)
    sse, sst = ((T - pred) ** 2).sum(), ((T - T.mean(0)) ** 2).sum()
    return float(np.clip(sse / sst, 0, 1)), k  # >1 = worse than predicting the mean


def eff_dim(X):
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "bench"))
    ap.add_argument("--top", type=int, default=40, help="rows of the per-feature table (most expensive first)")
    ap.add_argument("--csv", default=None, help="write the full table here (default: <dir>/feature_variance.csv)")
    args = ap.parse_args()

    n, mats, cost = load(args.dir)
    print(f"{n} tracks, {len(mats)} features with variance, {sum(m.shape[1] for m in mats.values())} dims")
    if n < 10:
        print("need at least 10 tracks for this analysis (ideally 50+)")
        return
    folds = min(5, n)
    keys = sorted(mats)

    # global equal-weight PCA: each feature's share of the variance in the components covering 80%
    G = weighted(mats, keys)
    pca = PCA(n_components=min(n - 1, G.shape[1])).fit(G)
    m = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.8) + 1)
    contrib = (pca.components_[:m] ** 2 * pca.explained_variance_[:m, None]).sum(0)
    share, i = {}, 0
    for k in keys:
        d = mats[k].shape[1]
        share[k] = float(contrib[i:i + d].sum() / contrib.sum())
        i += d
    print(f"global PCA: {m} components cover 80% of equal-weight variance")

    rows = []
    for k in keys:
        rest = [o for o in keys if o != k and o not in PREREQS.get(k, []) and k not in PREREQS.get(o, [])]
        u, kc = unique_variance(mats[k], weighted(mats, rest), folds)
        rows.append({"feature": k, "dims": mats[k].shape[1], "cost_s_per_min": cost_with_prereqs(k, cost),
                     "own_cost": cost.get(k, 0.0), "unique": u, "pcs": kc, "share": share[k], "eff_dim": eff_dim(mats[k])})
    for name, sel in BLOCKS.items():
        members = [k for k in keys if sel(k)]
        if not members:
            continue
        Y = weighted(mats, members)
        u, kc = unique_variance(Y, weighted(mats, [k for k in keys if k not in members]), folds)
        rows.append({"feature": name, "dims": Y.shape[1], "cost_s_per_min": block_cost(members, cost), "own_cost": np.nan,
                     "unique": u, "pcs": kc, "share": sum(share[k] for k in members), "eff_dim": eff_dim(Y),
                     "members": len(members)})
    rng = np.random.default_rng(0)
    u, kc = unique_variance(rng.standard_normal((n, 20)), G, folds)
    rows.append({"feature": "[control] random noise", "dims": 20, "cost_s_per_min": 0.0, "own_cost": 0.0,
                 "unique": u, "pcs": kc, "share": np.nan, "eff_dim": np.nan})

    total_cost = sum(cost.values())
    print(f"total per-track compute: {total_cost:.0f} s per minute of audio\n")
    hdr = f"{'cost s/min':>10} {'%cost':>6} {'unique':>7} {'share%':>7} {'effdim':>6} {'dims':>6}  feature"
    print("Costly blocks and control\n" + hdr)
    for r in rows:
        if r["feature"].startswith("["):
            print(f"{r['cost_s_per_min']:>10.1f} {100 * r['cost_s_per_min'] / total_cost:>6.1f} {r['unique']:>7.2f} "
                  f"{100 * r['share']:>7.2f} {r['eff_dim']:>6.1f} {r['dims']:>6}  {r['feature']}")
    print(f"\nPer feature, most expensive first (top {args.top})\n" + hdr)
    feats = sorted((r for r in rows if not r["feature"].startswith("[")), key=lambda r: -r["cost_s_per_min"])
    for r in feats[: args.top]:
        print(f"{r['cost_s_per_min']:>10.1f} {100 * r['cost_s_per_min'] / total_cost:>6.1f} {r['unique']:>7.2f} "
              f"{100 * r['share']:>7.2f} {r['eff_dim']:>6.1f} {r['dims']:>6}  {r['feature']}")

    import pandas as pd

    path = args.csv or os.path.join(args.dir, "feature_variance.csv")
    pd.DataFrame(rows).sort_values("cost_s_per_min", ascending=False).to_csv(path, index=False)
    print(f"\nfull table: {path}")


if __name__ == "__main__":
    main()
