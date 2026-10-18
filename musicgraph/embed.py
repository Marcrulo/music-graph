import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config, db, features

MIN_TRACKS = 5
N_CLUSTERS = 12


def build(con) -> dict:
    ids, names, X = db.load_features(con, features.EXTRACTOR, features.VERSION)
    if len(ids) < MIN_TRACKS:
        raise RuntimeError(f"need at least {MIN_TRACKS} tracks with features, have {len(ids)}")

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xs)
    Zp = pca.transform(Xs)
    db.replace_embeddings(con, "pca", ids, Zp)

    import umap  # slow import, keep it local

    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(ids) - 1), metric="cosine", random_state=42)
    Z2 = reducer.fit_transform(Zp)
    db.replace_embeddings(con, "umap2d", ids, Z2)

    from sklearn.cluster import KMeans

    clusters = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(Zp)
    html = write_map(con, ids, Z2, clusters)
    return {"n": len(ids), "dims": X.shape[1], "pca_dims": Zp.shape[1], "map": html}


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
        title="Music graph: handcrafted feature embedding (UMAP)",
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
