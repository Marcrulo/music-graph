
import typer

from . import config, db, embed, features, spectrogram
from .sources import jamendo

app = typer.Typer(no_args_is_help=True, help="Music graph: build a track database, spectrograms, features and embeddings.")


@app.command()
def fetch(
    tags: str = typer.Option("rock,electronic,jazz,classical,hiphop",
                             help="Comma-separated Jamendo search tags (steer sampling only; not stored)"),
    per_tag: int = typer.Option(10, help="New tracks to download per search tag"),
    max_per_artist: int = typer.Option(2, help="Max tracks per artist across the DB"),
):
    """Download new full-length tracks from Jamendo."""
    con = db.connect()
    total = 0
    for tag in [t.strip() for t in tags.split(",") if t.strip()]:
        total += jamendo.fetch(con, tag, per_tag, max_per_artist)
    typer.echo(f"added {total} tracks")


@app.command()
def spectro(force: bool = typer.Option(False, help="Recompute all")):
    """Compute log-mel spectrograms for tracks that lack one."""
    con = db.connect()
    todo = con.execute("SELECT * FROM tracks ORDER BY id").fetchall() if force else db.tracks_missing(con, "spectrograms")
    for t in todo:
        typer.echo(f"  #{t['id']} {t['artist']} - {t['title']}")
        try:
            y = spectrogram.load_audio(t["audio_path"])
        except Exception as e:
            typer.echo(f"    failed to decode: {e}")
            continue
        path, n_frames = spectrogram.compute_and_save(t["id"], y)
        db.upsert_spectrogram(con, t["id"], path.relative_to(config.DATA_DIR), n_frames)
    typer.echo(f"computed {len(todo)} spectrograms")


@app.command("features")
def features_cmd(force: bool = typer.Option(False, help="Recompute all")):
    """Extract handcrafted features for tracks that have a spectrogram."""
    con = db.connect()
    if force:
        todo = con.execute("SELECT t.* FROM tracks t JOIN spectrograms s ON s.track_id = t.id ORDER BY t.id").fetchall()
    else:
        todo = [
            t for t in db.tracks_missing(con, "features", "x.extractor = ? AND x.version = ?",
                                         (features.EXTRACTOR, features.VERSION))
            if spectrogram.spec_path(t["id"]).exists()
        ]
    for t in todo:
        typer.echo(f"  #{t['id']} {t['artist']} - {t['title']}")
        y = spectrogram.load_audio(t["audio_path"])
        names, vec = features.extract(y, spectrogram.load(t["id"]))
        db.upsert_features(con, t["id"], features.EXTRACTOR, features.VERSION, names, vec)
    typer.echo(f"extracted features for {len(todo)} tracks")


@app.command("embed")
def embed_cmd():
    """Fit PCA + UMAP over all tracks and write the interactive map."""
    con = db.connect()
    info = embed.build(con)
    typer.echo(f"{info['n']} tracks, {info['dims']} features -> {info['pca_dims']} PCA dims")
    typer.echo(f"map: {info['map']}")


@app.command()
def run(
    tags: str = typer.Option("rock,electronic,jazz,classical,hiphop"),
    per_tag: int = typer.Option(0, help="New tracks per tag to fetch first (0 = skip fetching)"),
):
    """Run every stage in order."""
    if per_tag:
        fetch(tags=tags, per_tag=per_tag)
    spectro(force=False)
    features_cmd(force=False)
    embed_cmd()


@app.command()
def stats():
    """Counts per genre and per pipeline stage."""
    con = db.connect()
    q = lambda sql, *p: con.execute(sql, p).fetchone()[0]
    typer.echo(f"tracks:       {q('SELECT COUNT(*) FROM tracks')}")
    typer.echo(f"spectrograms: {q('SELECT COUNT(*) FROM spectrograms')}")
    typer.echo(f"features:     {q('SELECT COUNT(*) FROM features WHERE extractor = ? AND version = ?', features.EXTRACTOR, features.VERSION)}")
    typer.echo(f"embedded:     {q('SELECT COUNT(*) FROM embeddings WHERE method = ?', 'pca')}")
    hours = (q("SELECT SUM(duration_s) FROM tracks") or 0) / 3600
    typer.echo(f"total audio:  {hours:.1f} h")
    typer.echo(f"with lyrics:  {q('SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL')}")
    typer.echo(f"artists:      {q('SELECT COUNT(DISTINCT artist) FROM tracks')}")


@app.command()
def lyrics():
    """Backfill lyrics for tracks that don't have them yet."""
    typer.echo(f"found lyrics for {jamendo.backfill_lyrics(db.connect())} tracks")


@app.command()
def similar(track_id: int, k: int = typer.Option(5, "-k")):
    """Nearest tracks in feature space (cosine on PCA embedding)."""
    con = db.connect()
    t = db.get_track(con, track_id)
    typer.echo(f"#{track_id} {t['artist']} - {t['title']}")
    for tid, s in embed.similar(con, track_id, k):
        o = db.get_track(con, tid)
        typer.echo(f"  {s:+.3f}  #{tid} {o['artist']} - {o['title']}")


@app.command()
def show(track_id: int):
    """Save a PNG of the track's spectrogram."""
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    con = db.connect()
    t = db.get_track(con, track_id)
    M = spectrogram.load(track_id)
    fig, ax = plt.subplots(figsize=(14, 4))
    img = librosa.display.specshow(M, sr=config.SR, hop_length=config.HOP, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"#{track_id} {t['artist']} - {t['title']}")
    path = config.MAPS_DIR / f"spec_{track_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    typer.echo(str(path))


if __name__ == "__main__":
    app()
