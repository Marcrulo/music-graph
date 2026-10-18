# TODO

## Feature extraction speed (from `experiments/benchmark_all_features.py`, 83 s track: ~927 s compute, 11x realtime)
- [ ] Speechiness / vocal pitch: switch torchcrepe `full` -> `tiny`/`small` (216 s, ~10-20x faster expected)
- [ ] Psychoacoustics (mosqito, single-threaded, 466 s total):
  - [ ] Sharpness from Zwicker specific loudness instead of recomputing (saves ~145 s)
  - [ ] Roughness (and maybe loudness) on a ~30 s excerpt
  - [ ] Decide whether to keep them at all (overlap with loudness/spectral features)
- [ ] Language of vocals: run Whisper only on vocal-active segments (83 s; likely hallucinating on silence)
- [ ] Process tracks in parallel (8 cores; ~4 workers), free models between stages (avoid swapping)
- [ ] MuQ: incompatible with transformers v5 -> separate env with transformers<5, or drop

## Pipeline
- [ ] Decide feature set for the real pipeline (proposal: everything except psychoacoustics as `full` v1)
- [ ] Turn benchmark functions into versioned extractors stored in the `features` table
- [ ] Run on all 300 tracks (estimated ~8-10 h after speed fixes, 4 workers)
- [ ] Embedding design session: per-category weighting/reduction so timbre doesn't dominate
- [ ] Segment-level features -> songs as trajectories ("soundtrack" view)
- [ ] Remove/limit over-represented artist (10 classical tracks by one artist predate the per-artist cap)
