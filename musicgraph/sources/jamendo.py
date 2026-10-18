"""Full-length Creative Commons tracks from the Jamendo API (v3.0).

Search tags only steer *which* tracks get sampled; no Jamendo labels are stored.
Besides artist/title we keep only lyrics; everything else is derived from audio.
"""

import hashlib
import time
from pathlib import Path

import requests

from .. import config, db

API = "https://api.jamendo.com/v3.0/tracks/"
SOURCE = "jamendo"
PAUSE_S = 0.5
EMPTY_RETRIES = 3  # the API intermittently returns empty pages
MAX_OFFSET = 5000


def search(tag: str, limit: int, offset: int = 0) -> list[dict]:
    if not config.JAMENDO_CLIENT_ID:
        raise RuntimeError("JAMENDO_CLIENT_ID is not set (see .env.example)")
    params = {
        "client_id": config.JAMENDO_CLIENT_ID,
        "format": "json",
        "tags": tag,
        "limit": limit,
        "offset": offset,
        "audiodownload_allowed": "true",
        "audiodlformat": "mp32",
        "include": "lyrics",
        "order": "popularity_total",
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    status = body.get("headers", {})
    if status.get("status") != "success":
        raise RuntimeError(f"Jamendo error: {status.get('error_message') or status}")
    return body["results"]


def download(url: str, dest: Path) -> str:
    """Stream url to dest; returns sha256. Writes to a temp file so partial downloads never count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                h.update(chunk)
    tmp.rename(dest)
    return h.hexdigest()


def artist_count(con, artist: str) -> int:
    return con.execute("SELECT COUNT(*) FROM tracks WHERE source = ? AND artist = ?", (SOURCE, artist)).fetchone()[0]


def fetch(con, tag: str, n_new: int, max_per_artist: int = 2, log=print) -> int:
    """Download up to n_new tracks for `tag` that aren't already in the DB,
    keeping at most max_per_artist tracks per artist across the whole DB."""
    added, offset, page = 0, 0, min(max(n_new * 3, 50), 200)
    while added < n_new:
        for _ in range(EMPTY_RETRIES):
            results = search(tag, limit=page, offset=offset)
            if results:
                break
            time.sleep(2)
        if not results or offset >= MAX_OFFSET:
            log(f"  [{tag}] no more results (offset {offset})")
            break
        offset += len(results)
        for t in results:
            if added >= n_new:
                break
            sid = str(t["id"])
            if db.has_track(con, SOURCE, sid) or not t.get("audiodownload"):
                continue
            if artist_count(con, t["artist_name"]) >= max_per_artist:
                continue
            dest = config.AUDIO_DIR / SOURCE / f"{sid}.mp3"
            log(f"  [{tag}] {t['artist_name']} - {t['name']}")
            try:
                sha = download(t["audiodownload"], dest)
            except requests.RequestException as e:
                log(f"    download failed: {e}")
                continue
            db.insert_track(
                con,
                source=SOURCE,
                source_id=sid,
                title=t.get("name"),
                artist=t.get("artist_name"),
                album=t.get("album_name"),
                duration_s=t.get("duration"),
                lyrics=(t.get("lyrics") or "").strip() or None,
                license_url=t.get("license_ccurl"),
                audio_path=str(dest.relative_to(config.DATA_DIR)),
                sha256=sha,
            )
            added += 1
            time.sleep(PAUSE_S)
        time.sleep(PAUSE_S)
    return added


def backfill_lyrics(con, log=print) -> int:
    """Fetch lyrics for tracks downloaded before lyrics were stored."""
    rows = con.execute("SELECT id, source_id FROM tracks WHERE source = ? AND lyrics IS NULL", (SOURCE,)).fetchall()
    found = 0
    for i in range(0, len(rows), 50):
        batch = {r["source_id"]: r["id"] for r in rows[i:i + 50]}
        r = requests.get(API, params={
            "client_id": config.JAMENDO_CLIENT_ID, "format": "json", "limit": 50,
            "id": " ".join(batch), "include": "lyrics",
        }, timeout=30)
        r.raise_for_status()
        for t in r.json()["results"]:
            lyrics = (t.get("lyrics") or "").strip()
            if lyrics and str(t["id"]) in batch:
                con.execute("UPDATE tracks SET lyrics = ? WHERE id = ?", (lyrics, batch[str(t["id"])]))
                found += 1
        con.commit()
        time.sleep(PAUSE_S)
    return found
