# CLAUDE.md

Guidance for AI coding agents working in this repo.

## Project
`musicgraph`: a local database of full-length Creative Commons tracks (Jamendo), with
log-mel spectrograms, extracted audio features, and embeddings. CLI: `uv run mg --help`.
Pipeline stages (fetch -> spectro -> features -> embed) are incremental; see README.md.
Feature stages (`musicgraph/extract/`: dsp, stems, models, lyrics) need `uv run --group full`.

## Setup / new machine
- Run `scripts/install.sh` (add `--warm` to pre-download all model weights). It checks prerequisites
  (uv, git, C compiler), runs `uv sync --group full`, creates `.env` from `.env.example`, downloads the
  Essentia models (`mg setup`) and reports what data is present. Keep it working when you add
  dependencies, model files or setup steps; `mg setup` is where new model downloads belong.
- Nothing under `data/` is in git: audio (~1.6 GB), `music.db`, spectrograms, Essentia models, bench logs.
  A collection moves by copying `data/music.db` + `data/audio/` together (DB paths are relative to `data/`);
  spectrograms and features can be recomputed. If they are missing, say so; don't assume the data exists.
- Feature extraction is heavy (~0.75x realtime per track on CPU, ~5-6 GB RAM per process). Ask before
  starting long or parallel runs on someone's machine.

## Core principle: audio only
Everything in the dataset must be derivable from the audio itself, so it scales to any
audio collection. The only non-audio data kept are **artist, title and lyrics**. Never store
or train on source labels (Jamendo genres, instrument/mood tags, etc.). Search tags may steer
*which* tracks get sampled, but are not stored. Evaluate embeddings without labels
(e.g. same-artist retrieval, cluster inspection, listening).

## Feature catalogue: `music_audio_features.md`
- This is the reference for **all feature and embedding types**, grouped by category.
  Consult it when choosing or designing new features/embeddings.
- When you add or change an extractor or embedding method, **update its
  "Implementation Status" table** (name, version, dims, contents) and add any new
  feature types to the matching category (or a new category).

## Conventions
- Feature extractors are versioned (`EXTRACTOR`, `VERSION` in the module); bump `VERSION`
  when the output changes so old and new vectors never mix in the `features` table.
- Store dimension names alongside vectors so features stay interpretable.
- `data/` (audio, spectrograms, DB, maps) and `.env` (JAMENDO_CLIENT_ID) are gitignored.
- Use `uv` (Python 3.11 venv); don't use the system anaconda Python.
