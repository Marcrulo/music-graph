import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config, db, features

MIN_TRACKS = 5
N_CLUSTERS = 12


MAX_BLOCK_DIMS = 32


def blocks(con) -> tuple[list[int], dict[str, np.ndarray]]:
    """Feature blocks for tracks that have every extractor output.

    Each model / extractor is a block; `dsp` and `stems` are split by their category prefix
    ('spectral/...', 'vocal/...') so one big extractor doesn't outweigh the others.
    """
    from .extract import all_outputs

    tables = {}
    for extractor, version in all_outputs():
        ids, names, X = db.load_features(con, extractor, version)
        tables[extractor] = (dict(zip(ids, range(len(ids)))), names, X)
    common = sorted(set.intersection(*(set(t[0]) for t in tables.values()))) if tables else []
    out = {}
    for extractor, (index, names, X) in tables.items():
        if not common:
            break
        X = X[[index[i] for i in common]]
        groups = [n.split("/", 1)[0] if extractor in ("dsp", "stems") else "" for n in names]
        for g in dict.fromkeys(groups):
            cols = [j for j, gg in enumerate(groups) if gg == g]
            out[f"{extractor}/{g}" if g else extractor] = X[:, cols]
    return common, out


def block_embedding(B: dict[str, np.ndarray]) -> np.ndarray:
    """Standardize each block, reduce it to <= MAX_BLOCK_DIMS PCs (95% var), scale to unit total
    variance (every block weighs the same), concatenate."""
    parts = []
    for X in B.values():
        X = np.nan_to_num(X.astype(np.float64))
        X = X[:, X.std(0) > 1e-9]
        if X.shape[1] == 0:
            continue
        Xs = StandardScaler().fit_transform(X)
        k = min(Xs.shape[1], len(Xs) - 1, MAX_BLOCK_DIMS)
        pca = PCA(n_components=k, svd_solver="full").fit(Xs)
        k95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
        Z = pca.transform(Xs)[:, :k95]
        parts.append(Z / np.sqrt(Z.var(0).sum() + 1e-12))
    return np.hstack(parts)


def build(con) -> dict:
    ids, B = blocks(con)
    if len(ids) >= MIN_TRACKS:
        X = block_embedding(B)
        source = f"{len(B)} equal-weight blocks"
    else:  # new extractors not run yet: fall back to the old handcrafted vector
        ids, names, X = db.load_features(con, features.EXTRACTOR, features.VERSION)
        if len(ids) < MIN_TRACKS:
            raise RuntimeError(f"need at least {MIN_TRACKS} tracks with features, have {len(ids)}")
        X = StandardScaler().fit_transform(X)
        source = f"{features.EXTRACTOR} v{features.VERSION}"

    pca = PCA(n_components=0.95, svd_solver="full").fit(X)
    Zp = pca.transform(X)
    db.replace_embeddings(con, "pca", ids, Zp)

    import umap  # slow import, keep it local

    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(ids) - 1), metric="cosine", random_state=42)
    Z2 = reducer.fit_transform(Zp)
    db.replace_embeddings(con, "umap2d", ids, Z2)

    from sklearn.cluster import KMeans

    clusters = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(Zp)
    html = write_map(con, ids, Z2, clusters)
    return {"n": len(ids), "dims": X.shape[1], "pca_dims": Zp.shape[1], "map": html, "source": source}


def write_map(con, ids, Z2, clusters):
    import plotly.express as px

    rows = []
    for tid, (x, y), c in zip(ids, Z2, clusters):
        t = db.get_track(con, tid)
        rows.append({"id": tid, "x": x, "y": y, "title": t["title"], "artist": t["artist"], "cluster": f"c{c}"})
    df = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.str[1:].astype(int))
    fig = px.scatter(
        df, x="x", y="y", color="cluster", hover_name="title",
        hover_data={"artist": True, "id": True, "x": False, "y": False},
        title="Music graph: audio feature embedding (UMAP)",
    )
    fig.update_traces(marker={"size": 10})
    path = config.MAPS_DIR / "embedding.html"
    fig.write_html(path)
    return path


def similar(con, track_id: int, k: int = 5) -> list[tuple[int, float]]:
    ids, Z = db.load_embeddings(con, "pca")
    if track_id not in ids:
        raise RuntimeError(f"track {track_id} has no embedding; run `mg embed`")
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sims = Zn @ Zn[ids.index(track_id)]
    order = [i for i in np.argsort(-sims) if ids[i] != track_id][:k]
    return [(ids[i], float(sims[i])) for i in order]
