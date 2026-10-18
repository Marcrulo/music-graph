# Music Audio Features

A reference of features that can be automatically extracted from music audio or spectrograms, grouped by category.

Common libraries: **librosa**, **Essentia**, **madmom**, **openSMILE**, **torchaudio**, **nnAudio**, **pyloudnorm**, **Kymatio** (scattering), **Demucs** (stems), **basic-pitch** / **CREPE** (pitch), **mosqito** (psychoacoustics).

---

## 1. Low-Level Spectral Features (frame-wise)

| Feature | What it captures |
|---|---|
| Spectral centroid | "Center of mass" of the spectrum; correlates with brightness |
| Spectral bandwidth / spread | How wide the spectrum is around the centroid |
| Spectral rolloff | Frequency below which a set percentage (e.g., 85%) of energy lies |
| Spectral flatness | Tonal vs. noise-like content (Wiener entropy) |
| Spectral contrast | Peak-to-valley energy difference across sub-bands |
| Spectral flux | Frame-to-frame change in the spectrum; useful for onsets |
| Spectral skewness / kurtosis | Shape statistics of the spectral distribution |
| Spectral entropy | How evenly energy is spread across frequencies |
| Spectral slope / decrease | Overall tilt of the spectrum |
| Spectral crest | Peak-to-mean ratio of the spectrum; tonal peakiness |
| Spectral irregularity | Jaggedness between adjacent spectral bins |
| Band energy ratios | Energy share in sub-bass / bass / low-mid / mid / high-mid / air bands |
| High-frequency content (HFC) | Weighted high-frequency energy; emphasizes percussive attacks |

## 2. Cepstral and Timbral Features

| Feature | What it captures |
|---|---|
| MFCCs (+ delta, delta-delta) | Compact timbre descriptor and its dynamics |
| Mel spectrogram / log-mel bands | Perceptually scaled spectrogram; standard deep learning input |
| Bark / ERB / gammatone bands, GFCCs | Alternative perceptual frequency scales |
| Constant-Q transform (CQT) | Log-frequency spectrogram aligned to musical pitches |
| Linear predictive coefficients (LPC) | Spectral envelope via an all-pole model |
| Variable-Q transform (VQT) | CQT variant with better time resolution in the low end |
| Odd-to-even harmonic ratio | Hollow (clarinet-like) vs. full (brass/strings) timbres |
| Spectral envelope / formants | Resonance structure; vocal and instrument body |

## 3. Energy and Dynamics

| Feature | What it captures |
|---|---|
| RMS energy | Frame-level loudness proxy |
| Zero-crossing rate | Noisiness and percussiveness |
| Loudness (EBU R128 / LUFS), loudness range | Perceived loudness and its variation |
| Dynamic range, crest factor | Peak vs. average level; compression amount |
| Attack time, decay, temporal centroid | Shape of the note or event envelope |
| True peak, clipping ratio | Inter-sample peaks; mastering hot-ness and distortion |
| Energy contour shape | Build-ups, drops, fade-ins/outs over the track |
| Peak-to-loudness ratio (PLR) | Headroom; "loudness war" indicator |

## 4. Harmony and Pitch

| Feature | What it captures |
|---|---|
| Chroma (STFT, CQT, CENS) | Energy per pitch class |
| Harmonic pitch class profile (HPCP) | Robust chroma variant for tonal analysis |
| Tonnetz / tonal centroid | Harmonic relationships in a 6-D tonal space |
| Key and mode | Estimated key, major/minor, with confidence |
| Chord sequence | Time-stamped chord labels |
| Fundamental frequency (f0) | Pitch track via YIN, pYIN, CREPE |
| Predominant melody | Main melodic line in polyphonic audio |
| Tuning frequency | Deviation from A440 |
| Inharmonicity, harmonic-to-noise ratio | Purity of harmonic content |
| Tristimulus | Relative strength of low, mid, and high harmonics |
| Dissonance / roughness | Perceived sensory dissonance |
| Key-normalized chroma | Chroma rotated to the estimated tonic; comparable across keys |
| Key clarity / tonalness | How strongly a single key is implied (atonal vs. tonal) |
| Chord change rate, harmonic rhythm | How often harmony moves |
| Chord vocabulary statistics | Share of major/minor/7th/extended/diminished chords; chord-transition histogram |
| Pitch range, melodic intervals | Melodic span and step vs. leap distribution |
| Polyphony estimate | Number of simultaneous notes (via multi-pitch, e.g. basic-pitch) |
| Bass line pitch / root motion | Harmonic foundation from the low register |

## 5. Rhythm and Tempo

| Feature | What it captures |
|---|---|
| Onset strength envelope, onset times | Where new events start |
| Onset rate | Events per second |
| Tempo (BPM) and confidence | Global or local tempo estimate |
| Beat and downbeat positions | Beat grid and bar starts |
| Tempogram | Tempo content over time |
| Beat histogram, rhythm patterns | Distribution of periodicities |
| Meter / time signature | Beats per bar |
| Pulse clarity, syncopation | Strength and regularity of the beat |
| Danceability | Suitability for dancing based on rhythm stability |
| Tempo stability / drift | Click-tracked vs. live, rubato |
| Swing ratio, microtiming | Groove feel; deviation of onsets from the straight grid |
| Onset patterns per band | Kick / snare / hi-hat style rhythm separated by frequency band |
| Fourier tempogram / cyclic tempogram | Tempo-invariant rhythm descriptors (octave-folded) |
| Rhythmic density per bar | Event count per bar, its variation |

## 6. Structure

| Feature | What it captures |
|---|---|
| Self-similarity / recurrence matrix | Repetition across the track |
| Novelty curve | Points of significant change |
| Segment boundaries and labels | Sections such as intro, verse, chorus |
| Duration, intro/outro detection | Track-level layout |
| Silence ratio | Proportion of near-silent frames |
| Repetitiveness score | Fraction of the track that is near-repeated material |
| Number / mean length of sections | Form complexity |
| Chorus / drop / climax detection | Peak-energy or most-repeated section |
| Loop detection | Exact repetition typical of electronic / sample-based music |

## 7. Source Separation Derived

| Feature | What it captures |
|---|---|
| Harmonic/percussive separation (HPSS) ratio | Balance of tonal vs. percussive content |
| Stem features (vocals, drums, bass, other) | Per-instrument features via Demucs, Spleeter |
| Vocal presence / instrumentalness | Whether vocals are present |
| Speechiness | Presence of spoken word |
| Stem energy balance | Relative loudness of vocals / drums / bass / other |
| Per-stem feature sets | Any category above computed on a single stem (e.g. drum-only rhythm, vocal-only pitch) |
| Vocal activity curve | Where vocals appear over time; vocal ratio |

## 8. High-Level / Semantic (usually model-based)

| Feature | What it captures |
|---|---|
| Genre and subgenre | Stylistic category |
| Mood and emotion (valence, arousal) | Affective character |
| Energy, acousticness, liveness | Descriptors similar to Spotify audio features |
| Instrument recognition / tags | Which instruments are present |
| Vocal gender, singing technique | Vocal characteristics |
| Era / decade estimation | Approximate production period |
| Production / audio quality | Mix and recording characteristics |
| Language of vocals | Spoken/sung language identification |
| Context / use-case tags | Workout, study, sleep, party, film-score |
| Approachability / engagement | Mainstream vs. niche character (Essentia models) |

## 9. Learned Embeddings

| Model | Notes |
|---|---|
| VGGish, YAMNet, PANNs | General audio event models |
| OpenL3 | Self-supervised audio-visual embeddings |
| MusiCNN | Music tagging model |
| MERT, MuQ | Self-supervised music representation models |
| CLAP | Joint audio-text embeddings; enables text queries |
| Encodec / Jukebox latents | Codec or generative model representations |
| Essentia Discogs-EffNet / MAEST | Trained on Discogs styles; strong for genre/style similarity |
| BEATs, AST, HTS-AT | Audio transformers (AudioSet) |
| DAC / SoundStream latents | Neural codec latents, time-resolved |
| Contrastive / triplet models trained here | Our own embedding, trained on this database's features or spectrograms |

Useful for similarity search, clustering, recommendation, and downstream classifiers.

## 10. Summary Statistics (frame-wise to track-level)

Frame-wise features are typically aggregated into a fixed-length track vector using:

- Mean, standard deviation, min, max
- Percentiles (e.g., 10th, 50th, 90th)
- Skewness and kurtosis
- Mean and variance of deltas
- Statistics computed per segment or per beat (beat-synchronous features)
- Covariance / Gaussian summary (mean + covariance of frame features)
- Bag-of-frames histograms (k-means codebook over frames)
- Temporal pooling with attention or learned weights
- Trajectories: keep per-segment vectors instead of collapsing (song as a path)

## 11. Psychoacoustic

| Feature | What it captures |
|---|---|
| Zwicker loudness (sones), specific loudness | Perceptual loudness per critical band |
| Sharpness | Perceived high-frequency "edge" |
| Roughness | Fast amplitude modulation (15–300 Hz) heard as harshness |
| Fluctuation strength | Slow modulation (~4 Hz) heard as wobble/pulsing |
| Tonality (psychoacoustic) | Prominence of tonal components over noise |
| Warmth / boominess | Low-mid emphasis heuristics |

## 12. Modulation and Texture

| Feature | What it captures |
|---|---|
| Modulation spectrum | Energy at amplitude-modulation rates per frequency band |
| Scattering transform | Stable multi-scale time-frequency texture (Kymatio) |
| Spectro-temporal receptive fields (2D Gabor) | Ripples in time and frequency; auditory-cortex-inspired texture |
| Texture statistics (McDermott-Simoncelli) | Marginal and correlation statistics of cochlear envelopes |
| Spectral flux variance, stationarity | How static vs. evolving the sound is |

## 13. Stereo and Spatial

Requires keeping the stereo signal (currently audio is loaded mono).

| Feature | What it captures |
|---|---|
| Stereo width, mid/side energy ratio | How wide the mix is |
| Inter-channel correlation | Mono compatibility, phase issues |
| Panning distribution per band | Where instruments sit left-to-right |
| Inter-channel level / time differences | Spatial cues |
| Reverb amount, estimated RT60, direct-to-reverberant ratio | Room size and wetness |

## 14. Production and Recording

| Feature | What it captures |
|---|---|
| Spectral balance / tonal curve | Mix EQ shape compared to reference curves |
| Compression / limiting indicators | Low crest factor, flattened loudness range |
| Sub-bass presence | Energy below ~60 Hz (EDM, hip-hop vs. acoustic) |
| Noise floor, hiss, hum (50/60 Hz) | Lo-fi or analog character |
| Codec artifacts, effective bandwidth | Low-pass cutoff from lossy encoding (e.g. 16 kHz shelf) |
| Saturation / distortion estimate | Harmonic distortion amount |
| Live vs. studio cues | Audience noise, tempo drift, room sound |

## 15. Vocal-Specific

Best computed on a separated vocal stem.

| Feature | What it captures |
|---|---|
| Vocal pitch range, median pitch | Register of the singer |
| Vibrato rate and extent | Singing style |
| Breathiness, jitter, shimmer, HNR | Voice quality |
| Rap vs. sung ratio | Pitched vs. rhythmic-speech delivery |
| Syllable / word rate | Lyrical density |
| Harmony vocals / choir presence | Vocal layering |

## 16. Allowed Non-Audio Data

Project rule: everything is derived from audio, except **artist, title and lyrics**.
Source labels (genre/instrument/mood tags) are never stored or used.

| Feature | Notes |
|---|---|
| Artist | Weak supervision / evaluation (same-artist retrieval) |
| Title | Display only |
| Lyrics embeddings, lyric themes | Text models on lyrics, when available (Jamendo lyrics for ~1/4 of tracks) |

---

## Implementation Status

What `musicgraph` currently extracts and stores (DB table `features`/`embeddings`).

| Extractor / method | Version | Dims | Contents |
|---|---|---|---|
| `handcrafted` | 1 | 294 | MFCC 0–19 + deltas, chroma (12), tonnetz (6), spectral centroid / bandwidth / rolloff / flatness / contrast (7), ZCR, RMS, onset strength, each as mean/std/p10/p90; plus tempo, beat strength mean/std, beat regularity, onset rate, duration |
| `pca` (embedding) | – | ~30 | StandardScaler → PCA (95% variance) of `handcrafted` v1 |
| `umap2d` (embedding) | – | 2 | UMAP (cosine) of `pca`, for the map; coloured by KMeans (12) clusters on `pca` |
