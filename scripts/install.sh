#!/usr/bin/env bash
# Set up musicgraph on a new machine: Python env, config, model files, and a check of the data.
#
#   scripts/install.sh            # env + Essentia models
#   scripts/install.sh --warm     # also pre-download Hugging Face / demucs weights (~3 GB, loads every model once)
#
# Not in git (data/ is gitignored): the audio, music.db, spectrograms and model files.
# To keep an existing collection, copy data/music.db and data/audio/ from the old machine
# (paths in the DB are relative to data/), e.g.
#   rsync -a --info=progress2 OLDPC:music-graph/data/{music.db,audio} data/
# Spectrograms and features can then be recomputed with `uv run --group full mg run`.
set -euo pipefail
cd "$(dirname "$0")/.."

WARM=""
[[ "${1:-}" == "--warm" ]] && WARM="--warm"

missing=()
command -v uv >/dev/null || missing+=("uv (curl -LsSf https://astral.sh/uv/install.sh | sh)")
command -v git >/dev/null || missing+=("git")
command -v cc >/dev/null || command -v gcc >/dev/null || missing+=("a C compiler (madmom builds Cython extensions; e.g. apt install build-essential)")
if ((${#missing[@]})); then
    printf 'Missing prerequisites:\n'; printf '  - %s\n' "${missing[@]}"; exit 1
fi

echo "== Python environment (uv, Python $(cat .python-version), group 'full', CPU torch)"
uv sync --group full

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "== Created .env: set JAMENDO_CLIENT_ID (https://devportal.jamendo.com) to fetch tracks"
fi

echo "== Model files"
uv run --group full mg setup $WARM

echo "== Data"
uv run --group full mg stats
if [[ ! -f data/music.db ]] || [[ -z "$(ls -A data/audio 2>/dev/null)" ]]; then
    cat <<'EOF'
No audio collection here yet. Either copy data/music.db and data/audio/ from the old machine
(see the top of this script), or start a new one:
  uv run mg fetch --per-tag 10
EOF
fi
echo "== Done. Next: uv run --group full mg run   (spectrograms, features, embedding; resumable)"
