"""Feature extraction stages. Each stage loads its models once, then processes every track
that is missing one of its outputs (extractor, version) - incremental and resumable.

  dsp     signal-processing features on the full track            -> dsp
  stems   demucs on fixed excerpts: vocal / bass / stem features    -> stems (+ transcripts table)
  models  pretrained embeddings and classifier heads on excerpts    -> effnet maest musicnn clap mert muq semantic
  lyrics  sentence embedding of lyrics or the transcript            -> lyrics
"""

import gc
import time
import traceback

STAGE_ORDER = ["dsp", "stems", "models", "lyrics"]


def stage_class(name):
    if name == "dsp":
        from .dsp import Stage
    elif name == "stems":
        from .stems import Stage
    elif name == "models":
        from .models import Stage
    elif name == "lyrics":
        from .lyrics import Stage
    else:
        raise ValueError(f"unknown stage {name!r}; choose from {STAGE_ORDER}")
    return Stage


def all_outputs() -> list[tuple[str, int]]:
    return [o for n in STAGE_ORDER for o in stage_class(n).outputs]


def release_memory():
    """Return freed heap pages to the OS between tracks (glibc otherwise keeps them)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def tracks_missing(con, outputs, force=False, track_ids=None) -> list:
    if force:
        rows = con.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    else:
        cond = " OR ".join("(x.extractor = ? AND x.version = ?)" for _ in outputs)
        params = [v for o in outputs for v in o]
        rows = con.execute(
            f"""SELECT t.* FROM tracks t
                WHERE (SELECT COUNT(DISTINCT x.extractor) FROM features x WHERE x.track_id = t.id AND ({cond})) < ?
                ORDER BY t.id""", (*params, len(outputs))).fetchall()
    return [r for r in rows if track_ids is None or r["id"] in track_ids]


def run(con, stages=None, force=False, limit=None, track_ids=None, log=print) -> dict:
    import torch  # noqa: F401  must be imported before essentia/TensorFlow or a later torch import breaks

    from .. import db
    from .audio import Audio

    done = {}
    for name in stages or STAGE_ORDER:
        stage = stage_class(name)()
        todo = tracks_missing(con, stage.outputs, force, track_ids)
        if hasattr(stage, "ready"):
            todo = [t for t in todo if stage.ready(con, t)]
        todo = todo[:limit] if limit else todo
        if not todo:
            log(f"[{name}] nothing to do")
            continue
        log(f"[{name}] {len(todo)} tracks; loading models")
        t0 = time.perf_counter()
        stage.load()
        log(f"[{name}] models loaded in {time.perf_counter() - t0:.0f}s")
        ok = 0
        for i, t in enumerate(todo, 1):
            t1 = time.perf_counter()
            try:
                if getattr(stage, "needs_audio", True):
                    a = Audio(t)
                    out = stage.process(a)
                    del a
                else:
                    out = stage.process_text(con, t)
                if "_transcript" in out:
                    db.upsert_transcript(con, t["id"], out.pop("_transcript"), "whisper-tiny")
                for (extractor, version) in stage.outputs:
                    names, vec = out[extractor]
                    db.upsert_features(con, t["id"], extractor, version, names, vec)
                ok += 1
                log(f"[{name}] {i}/{len(todo)} #{t['id']} {t['artist']} - {t['title']}: {time.perf_counter() - t1:.0f}s")
            except Exception as e:
                log(f"[{name}] {i}/{len(todo)} #{t['id']} FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc(limit=3))
            release_memory()
        done[name] = ok
        del stage
        release_memory()
    return done
