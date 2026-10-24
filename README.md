# music-graph

Local database of full-length Creative Commons tracks (Jamendo): log-mel spectrograms,
handcrafted audio features, and an embedding map.

## Setup
```bash
scripts/install.sh            # uv env (group "full"), .env, Essentia models; --warm also caches HF weights
```
Then add your `JAMENDO_CLIENT_ID` (https://devportal.jamendo.com) to `.env`. `data/` (audio, DB,
spectrograms, models) is not in git: to move a collection, copy `data/music.db` and `data/audio/`.

## Usage
```bash
uv run mg fetch --tags rock,jazz --per-tag 3   # download new tracks
uv run mg spectro                              # log-mel spectrograms -> data/spectrograms/
uv run --group full mg features                # audio features + model embeddings (see below)
uv run mg embed                                # PCA + UMAP, writes data/maps/embedding.html
uv run mg similar 1 -k 5                       # nearest tracks
uv run mg show 1                               # spectrogram PNG
uv run mg stats
uv run mg run --per-tag 10                     # fetch + all stages
```
Every stage is incremental: it only processes tracks that are missing that stage.

## Features
`mg features` runs four stages (`--stage dsp,stems,models,lyrics`), each loading its models once:
`dsp` (signal-processing features on the full track), `stems` (demucs on three 30 s excerpts: vocal,
bass and stem features, Whisper transcript), `models` (EffNet, MAEST, MusiCNN, CLAP, MERT, MuQ embeddings
and Essentia classifier heads, on the same excerpts) and `lyrics` (sentence embedding of lyrics or
transcript). About 0.75x realtime on CPU for a 5-minute track. The Essentia models must be in
`data/models/essentia/` (from https://essentia.upf.edu/models/); missing files are listed on start.
See `music_audio_features.md` for what each extractor contains.
