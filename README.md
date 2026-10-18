# music-graph

Local database of full-length Creative Commons tracks (Jamendo): log-mel spectrograms,
handcrafted audio features, and an embedding map.

## Setup
```bash
uv sync
cp .env.example .env   # add your JAMENDO_CLIENT_ID from https://devportal.jamendo.com
```

## Usage
```bash
uv run mg fetch --tags rock,jazz --per-tag 3   # download new tracks
uv run mg spectro                              # log-mel spectrograms -> data/spectrograms/
uv run mg features                             # ~300 named features per track
uv run mg embed                                # PCA + UMAP, writes data/maps/embedding.html
uv run mg similar 1 -k 5                       # nearest tracks
uv run mg show 1                               # spectrogram PNG
uv run mg stats
uv run mg run --per-tag 10                     # fetch + all stages
```
Every stage is incremental: it only processes tracks that are missing that stage.
