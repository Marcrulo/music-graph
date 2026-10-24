# TODO

## Feature extraction speed (`experiments/benchmark_all_features.py`)
Before: 83 s track ~927 s compute (11x realtime). After fixes: 83 s track 313 s (3.8x), 314 s track 933 s (3.0x), single worker.
- [x] Vocal pitch: torchcrepe `full` -> `tiny` (200 s -> 11 s; 99.7% of co-voiced frames within 50 cents)
- [x] Psychoacoustics: sharpness from Zwicker specific loudness; loudness/roughness/tonality on 3x10 s excerpts
      (20/50/80%) -> fixed ~125-155 s per track instead of ~5.6x realtime
  - [ ] Decide whether to keep them at all (run `experiments/feature_variance.py` on a multi-track bench)
- [x] Language of vocals: Whisper only on vocal-active segments; skipped below 5 s of vocals (instrumental);
      language via direct detection (+prob); per-30 s repetition check; max 128 tokens/piece (126 s -> ~20 s on 5 min)
- [x] Parallel: `benchmark_all_features.py <ids...> --workers N` (subprocess per track, threads split, heavy-stage
      lock for demucs/learned models, start only with >=4 GB free, kill+requeue newest worker below 1 GB)
  - [ ] Untested beyond one worker. 4 workers froze the 15 GB machine (all hit demucs at once, before the guards).
        Single worker on a 5 min track now peaks ~5.9 GB RSS -> at most 2 workers here
- [x] Memory: Encodec/CLAP/OpenL3 chunked, malloc_trim between categories, stems freed after last user
      (peak 8.2 -> 5.9 GB on 314 s track). RSS still creeps ~1 GB per stage; 2-process isolation would cap it
- [x] MuQ: transformers-v5 shim in `load_muq()` (outputs bit-identical to transformers 4.57)
- [ ] Remaining hotspots (314 s track): demucs 190 s, MERT+MuQ 112 s, roughness 103 s, madmom downbeats 44 s

## Pipeline (`musicgraph/extract/`, `uv run --group full mg features`)
- [x] Feature set: dsp + stems + models (effnet, maest, musicnn, semantic, clap, mert, muq) + lyrics.
      Dropped for speed/redundancy: mosqito psychoacoustics, madmom downbeats/meter, VGGish/YAMNet/PANNs,
      AST, OpenL3, Encodec, the Whisper language label, k-means codebook, trajectories
- [x] Versioned extractors in the `features` table; stage-major (models loaded once per stage); demucs and
      all model stages on 3x30 s excerpts (20/50/80%); Whisper transcript in `transcripts` table
- [x] `mg embed` builds equal-weight blocks (each extractor / dsp+stems category) -> PCA -> UMAP
- [ ] Run on all 300 tracks. 5:14 track: dsp 79 s + stems ~95 s + models 61 s ~= 235 s (was 880 s).
      Estimated ~18 h single process for the current 19.2 h of audio; resumable (rerun skips finished tracks)
- [ ] Parallelism for the stage runner (one process per stage or per track subset; RAM: ~5-6 GB per process)
- [ ] Embedding design session: per-category weighting/reduction so timbre doesn't dominate
- [ ] Run `experiments/feature_variance.py` (needs a multi-track bench) to check what the dropped features added
- [ ] Remaining hotspots: demucs (~65 s of the stems stage), dsp full-track (79 s on 5 min: HPSS, Melodia,
      basic-pitch, CQT/VQT)
- [x] Essentia model files: `mg setup` downloads them; `scripts/install.sh` sets up a new machine
- [ ] Segment-level features -> songs as trajectories ("soundtrack" view)
- [ ] Remove/limit over-represented artist (10 classical tracks by one artist predate the per-artist cap)
