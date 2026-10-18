import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    duration_s  REAL,
    lyrics      TEXT,          -- only non-audio content besides artist/title
    license_url TEXT,
    audio_path  TEXT NOT NULL,
    sha256      TEXT,
    added_at    TEXT,
    UNIQUE (source, source_id)
);
CREATE TABLE IF NOT EXISTS spectrograms (
    track_id   INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    sr         INTEGER,
    n_mels     INTEGER,
    hop        INTEGER,
    n_frames   INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS features (
    track_id  INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    extractor TEXT NOT NULL,
    version   INTEGER NOT NULL,
    names     TEXT NOT NULL,   -- JSON list, one name per dimension
    vector    BLOB NOT NULL,   -- float32
    created_at TEXT,
    PRIMARY KEY (track_id, extractor, version)
);
CREATE TABLE IF NOT EXISTS embeddings (
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    method   TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL,    -- float32
    created_at TEXT,
    PRIMARY KEY (track_id, method)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode = WAL")  # readers don't block the pipeline's writes
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def _migrate(con) -> None:
    """Everything except artist/title/lyrics must come from the audio: no source labels."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(tracks)")}
    for c in ("genres", "tags"):
        if c in cols:
            con.execute(f"ALTER TABLE tracks DROP COLUMN {c}")
    if "lyrics" not in cols:
        con.execute("ALTER TABLE tracks ADD COLUMN lyrics TEXT")
    con.commit()


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def has_track(con, source: str, source_id: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM tracks WHERE source = ? AND source_id = ?", (source, source_id)
    ).fetchone()
    return row is not None


def insert_track(con, **t) -> int:
    cur = con.execute(
        """INSERT INTO tracks (source, source_id, title, artist, album, duration_s,
               lyrics, license_url, audio_path, sha256, added_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            t["source"], t["source_id"], t.get("title"), t.get("artist"), t.get("album"),
            t.get("duration_s"), t.get("lyrics"),
            t.get("license_url"), t["audio_path"], t.get("sha256"), now(),
        ),
    )
    con.commit()
    return cur.lastrowid


def get_track(con, track_id: int):
    return con.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()


def tracks_missing(con, table: str, extra_where: str = "", params=()) -> list:
    """Tracks with no row in `table` (optionally restricted by extra_where on that table)."""
    cond = f"AND {extra_where}" if extra_where else ""
    return con.execute(
        f"""SELECT t.* FROM tracks t
            WHERE NOT EXISTS (SELECT 1 FROM {table} x WHERE x.track_id = t.id {cond})
            ORDER BY t.id""",
        params,
    ).fetchall()


def upsert_spectrogram(con, track_id, path, n_frames) -> None:
    con.execute(
        """INSERT OR REPLACE INTO spectrograms (track_id, path, sr, n_mels, hop, n_frames, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (track_id, str(path), config.SR, config.N_MELS, config.HOP, n_frames, now()),
    )
    con.commit()


def upsert_features(con, track_id, extractor, version, names, vector) -> None:
    con.execute(
        """INSERT OR REPLACE INTO features (track_id, extractor, version, names, vector, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (track_id, extractor, version, json.dumps(names), to_blob(vector), now()),
    )
    con.commit()


def load_features(con, extractor, version):
    """Returns (track_ids, names, matrix) for one extractor version."""
    rows = con.execute(
        "SELECT track_id, names, vector FROM features WHERE extractor = ? AND version = ? ORDER BY track_id",
        (extractor, version),
    ).fetchall()
    if not rows:
        return [], [], np.empty((0, 0), dtype=np.float32)
    names = json.loads(rows[0]["names"])
    ids = [r["track_id"] for r in rows]
    X = np.stack([from_blob(r["vector"]) for r in rows])
    return ids, names, X


def replace_embeddings(con, method, ids, Z) -> None:
    con.execute("DELETE FROM embeddings WHERE method = ?", (method,))
    con.executemany(
        "INSERT INTO embeddings (track_id, method, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
        [(tid, method, Z.shape[1], to_blob(z), now()) for tid, z in zip(ids, Z)],
    )
    con.commit()


def load_embeddings(con, method):
    rows = con.execute(
        "SELECT track_id, vector FROM embeddings WHERE method = ? ORDER BY track_id", (method,)
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    return [r["track_id"] for r in rows], np.stack([from_blob(r["vector"]) for r in rows])
