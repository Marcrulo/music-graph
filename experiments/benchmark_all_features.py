"""Compute every feature listed in music_audio_features.md for one track and time it.

Each catalogue row is one timed function returning a dict of track-level values
(what we'd store). Shared intermediates (decoding, STFT, stems, backbone embeddings)
and one-off model loading are timed separately, so per-feature times are marginal.

Usage: uv run --group full python experiments/benchmark_all_features.py [track_id]
"""

import gc
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import torch  # import before essentia/TF, avoids a broken partial torch import later
import librosa
import numpy as np
import scipy.signal
import scipy.stats
import soundfile as sf

from musicgraph import config, db

ESS_MODELS = config.DATA_DIR / "models" / "essentia"
RESULTS: list[dict] = []
VALUES: dict[str, dict] = {}


# ---------------------------------------------------------------- bench helpers

def n_values(out) -> int:
    if out is None:
        return 0
    if isinstance(out, dict):
        return sum(n_values(v) for v in out.values())
    if isinstance(out, (str, bytes)):
        return 1
    return int(np.size(out))


def run(cat: str, name: str, fn, *args, shared=False, **kw):
    t0 = time.perf_counter()
    try:
        out, status, note = fn(*args, **kw), "ok", ""
    except Exception as e:  # record and keep going
        out, status = None, "fail"
        note = f"{type(e).__name__}: {e}".replace("\n", " ")[:160]
        if os.environ.get("BENCH_TRACEBACK"):
            traceback.print_exc()
    dt = time.perf_counter() - t0
    dims = None if shared else n_values(out)
    RESULTS.append({"cat": cat, "feature": name, "status": status, "seconds": round(dt, 3), "dims": dims, "note": note})
    if status == "ok" and not shared:
        VALUES[f"{cat}/{name}"] = out
    print(f"  {status:<4} {dt:7.2f}s  [{cat}] {name}" + (f"  dims={dims}" if dims else "") + (f"  -- {note}" if note else ""), flush=True)
    return out


def na(cat: str, name: str, note: str):
    RESULTS.append({"cat": cat, "feature": name, "status": "n/a", "seconds": 0.0, "dims": 0, "note": note})
    print(f"  n/a            [{cat}] {name}  -- {note}", flush=True)


_MODELS = {}


def model(name: str, loader):
    """Load once; loading time is reported under 'model load'."""
    if name not in _MODELS:
        _MODELS[name] = run("model load", name, loader, shared=True)
    if _MODELS[name] is None:
        raise RuntimeError(f"model {name} failed to load")
    return _MODELS[name]


def stats(M, prefix="") -> dict:
    """mean/std/p10/p90 per row, the summary used by the current pipeline."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    return {f"{prefix}mean": M.mean(1), f"{prefix}std": M.std(1),
            f"{prefix}p10": np.percentile(M, 10, axis=1), f"{prefix}p90": np.percentile(M, 90, axis=1)}


def db_(x, floor=1e-10):
    return 10 * np.log10(np.maximum(x, floor))


# ---------------------------------------------------------------- shared intermediates

class A:
    """Audio and intermediates shared by many features."""


def load_shared(track) -> A:
    a = A()
    a.path = str(config.DATA_DIR / track["audio_path"])
    a.sr = config.SR
    a.stereo44 = run("shared", "decode stereo 44.1k", lambda: librosa.load(a.path, sr=44100, mono=False)[0], shared=True)
    if a.stereo44.ndim == 1:
        a.stereo44 = np.stack([a.stereo44, a.stereo44])
    a.mono44 = a.stereo44.mean(0).astype(np.float32)
    a.y = run("shared", "resample mono 22.05k", lambda: librosa.resample(a.mono44, orig_sr=44100, target_sr=a.sr), shared=True)
    a.y16 = run("shared", "resample mono 16k", lambda: librosa.resample(a.mono44, orig_sr=44100, target_sr=16000).astype(np.float32), shared=True)
    a.dur = len(a.y) / a.sr
    a.S = run("shared", "STFT magnitude", lambda: np.abs(librosa.stft(a.y, n_fft=config.N_FFT, hop_length=config.HOP)), shared=True)
    a.P = a.S ** 2
    a.freqs = librosa.fft_frequencies(sr=a.sr, n_fft=config.N_FFT)
    a.mel_db = run("shared", "log-mel", lambda: librosa.power_to_db(
        librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=config.N_MELS), ref=np.max), shared=True)
    a.onset_env = run("shared", "onset envelope", lambda: librosa.onset.onset_strength(S=a.mel_db, sr=a.sr), shared=True)
    a.fps = a.sr / config.HOP

    def _beats():
        tempo, beats = librosa.beat.beat_track(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP)
        return float(np.atleast_1d(tempo)[0]), beats
    a.tempo, a.beats = run("shared", "beat tracking (librosa)", _beats, shared=True)
    a.beat_times = librosa.frames_to_time(a.beats, sr=a.sr, hop_length=config.HOP)
    a.yh, a.yp = run("shared", "HPSS", lambda: librosa.effects.hpss(a.y), shared=True)
    a.chroma = run("shared", "chroma STFT", lambda: librosa.feature.chroma_stft(S=a.P, sr=a.sr), shared=True)
    a.mfcc = librosa.feature.mfcc(S=a.mel_db, n_mfcc=20)
    a.rms = librosa.feature.rms(S=a.S, frame_length=config.N_FFT)[0]

    import essentia.standard as es

    def _ess_spectra():
        w, spec = es.Windowing(type="blackmanharris62"), es.Spectrum()
        frames = list(es.FrameGenerator(a.mono44, frameSize=2048, hopSize=1024, startFromZero=True))
        return frames, [spec(w(f)) for f in frames]
    a.ess_frames, a.ess_spectra = run("shared", "essentia frames+spectra 44.1k", _ess_spectra, shared=True)

    def _ess_peaks():
        pk = es.SpectralPeaks(orderBy="frequency", minFrequency=20, maxFrequency=8000, maxPeaks=60,
                              magnitudeThreshold=1e-5, sampleRate=44100)
        yin = es.PitchYinFFT(frameSize=2048, sampleRate=44100)
        return [pk(s) for s in a.ess_spectra], [yin(s) for s in a.ess_spectra]
    a.ess_peaks, a.ess_pitch = run("shared", "essentia spectral peaks + YinFFT pitch", _ess_peaks, shared=True)
    return a


def demucs_stems(a):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    m = model("demucs htdemucs", lambda: get_model("htdemucs").eval())
    wav = torch.tensor(a.stereo44, dtype=torch.float32)
    ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(m, wav_n[None], device="cpu", split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    return {name: out[i].numpy() for i, name in enumerate(m.sources)}  # (2, N) each


def effnet_embeddings(a):
    import essentia.standard as es

    m = model("essentia discogs-effnet", lambda: es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(ESS_MODELS / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"))
    return m(a.y16)


def musicnn_embeddings(a):
    import essentia.standard as es

    m = model("essentia msd-musicnn", lambda: es.TensorflowPredictMusiCNN(
        graphFilename=str(ESS_MODELS / "msd-musicnn-1.pb"), output="model/dense/BiasAdd"))
    return m(a.y16)


def ess_head(name: str, emb):
    """Apply an Essentia classification head (from its JSON schema) to backbone embeddings."""
    import essentia.standard as es

    meta = json.load(open(ESS_MODELS / f"{name}.json"))
    inp = meta["schema"]["inputs"][0]["name"]
    outs = meta["schema"]["outputs"]
    out = next((o["name"] for o in outs if o.get("output_purpose") == "predictions"), outs[0]["name"])
    m = model(f"essentia head {name}", lambda: es.TensorflowPredict2D(
        graphFilename=str(ESS_MODELS / f"{name}.pb"), input=inp, output=out))
    p = np.asarray(m(emb)).mean(0)
    classes = meta.get("classes") or [f"c{i}" for i in range(len(p))]
    return dict(zip(classes, p.astype(float)))


# ---------------------------------------------------------------- 1. spectral

def f_spectral(a):
    S, P, fr = a.S, a.P, a.freqs
    Pn = P / (P.sum(0, keepdims=True) + 1e-12)
    out = {}

    def add(name, fn):
        out[name] = fn
    add("Spectral centroid", lambda: stats(librosa.feature.spectral_centroid(S=S, sr=a.sr)))
    add("Spectral bandwidth / spread", lambda: stats(librosa.feature.spectral_bandwidth(S=S, sr=a.sr)))
    add("Spectral rolloff", lambda: stats(librosa.feature.spectral_rolloff(S=S, sr=a.sr)))
    add("Spectral flatness", lambda: stats(librosa.feature.spectral_flatness(S=S)))
    add("Spectral contrast", lambda: stats(librosa.feature.spectral_contrast(S=S, sr=a.sr)))
    add("Spectral flux", lambda: stats(np.sqrt((np.diff(S, axis=1).clip(0) ** 2).sum(0))))

    def skew_kurt():
        mu = (fr[:, None] * Pn).sum(0)
        var = (((fr[:, None] - mu) ** 2) * Pn).sum(0)
        sk = (((fr[:, None] - mu) ** 3) * Pn).sum(0) / (var ** 1.5 + 1e-12)
        ku = (((fr[:, None] - mu) ** 4) * Pn).sum(0) / (var ** 2 + 1e-12)
        return {**stats(sk, "skew_"), **stats(ku, "kurt_")}
    add("Spectral skewness / kurtosis", skew_kurt)
    add("Spectral entropy", lambda: stats(-(Pn * np.log2(Pn + 1e-12)).sum(0) / np.log2(len(fr))))

    def slope_decrease():
        slope = np.polyfit(fr, S, 1)[0]
        k = np.arange(1, S.shape[0])
        dec = ((S[1:] - S[:1]) / k[:, None]).sum(0) / (S[1:].sum(0) + 1e-12)
        return {**stats(slope, "slope_"), **stats(dec, "decrease_")}
    add("Spectral slope / decrease", slope_decrease)
    add("Spectral crest", lambda: stats(S.max(0) / (S.mean(0) + 1e-12)))
    add("Spectral irregularity", lambda: stats((np.diff(S, axis=0) ** 2).sum(0) / ((S ** 2).sum(0) + 1e-12)))

    def band_ratios():
        bands = {"sub": (20, 60), "bass": (60, 250), "lowmid": (250, 500), "mid": (500, 2000),
                 "highmid": (2000, 4000), "presence": (4000, 6000), "air": (6000, 11025)}
        tot = P.sum(0) + 1e-12
        return {k: stats(P[(fr >= lo) & (fr < hi)].sum(0) / tot)["mean"] for k, (lo, hi) in bands.items()}
    add("Band energy ratios", band_ratios)
    add("High-frequency content (HFC)", lambda: stats((np.arange(len(fr))[:, None] * P).sum(0)))
    return out


# ---------------------------------------------------------------- 2. timbre

def f_timbre(a):
    import essentia.standard as es
    out = {}

    def mfcc():
        d1, d2 = librosa.feature.delta(a.mfcc), librosa.feature.delta(a.mfcc, order=2)
        return {**stats(a.mfcc, "mfcc_"), **stats(d1, "d1_"), **stats(d2, "d2_")}
    out["MFCCs (+ delta, delta-delta)"] = mfcc
    out["Mel spectrogram / log-mel bands"] = lambda: stats(a.mel_db)

    def gfcc_bark_erb():
        g = es.GFCC(sampleRate=44100, inputSize=1025, numberCoefficients=13)
        bark = es.BarkBands(sampleRate=44100)
        G = np.array([g(s)[1] for s in a.ess_spectra]).T
        B = np.array([bark(s) for s in a.ess_spectra]).T
        return {**stats(G, "gfcc_"), **stats(db_(B), "bark_")}
    out["Bark / ERB / gammatone bands, GFCCs"] = gfcc_bark_erb
    out["Constant-Q transform (CQT)"] = lambda: stats(librosa.amplitude_to_db(
        np.abs(librosa.cqt(a.y, sr=a.sr, hop_length=config.HOP, n_bins=84)), ref=np.max))

    def lpc():
        frames = librosa.util.frame(a.y, frame_length=1024, hop_length=1024).T
        frames = frames[np.abs(frames).max(1) > 1e-3]
        L = np.array([librosa.lpc(f * np.hanning(len(f)), order=12) for f in frames])
        a.lpc_frames = L
        return stats(L[:, 1:].T)
    out["Linear predictive coefficients (LPC)"] = lpc
    out["Variable-Q transform (VQT)"] = lambda: stats(librosa.amplitude_to_db(
        np.abs(librosa.vqt(a.y, sr=a.sr, hop_length=config.HOP, n_bins=84, gamma=None)), ref=np.max))

    def odd_even():
        hp, oer = es.HarmonicPeaks(), es.OddToEvenHarmonicEnergyRatio()
        vals = []
        for (f, m), (p, c) in zip(a.ess_peaks, a.ess_pitch):
            if p > 0 and c > 0.5 and len(f):
                hf, hm = hp(f, m, p)
                vals.append(oer(hf, hm))
        return stats(np.array(vals) if vals else np.zeros(1))
    out["Odd-to-even harmonic ratio"] = odd_even

    def formants():
        L = getattr(a, "lpc_frames", None)
        if L is None:
            lpc()
            L = a.lpc_frames
        F = []
        for coefs in L[:: max(1, len(L) // 400)]:
            r = np.roots(coefs)
            r = r[np.imag(r) > 0]
            f = np.sort(np.angle(r) * a.sr / (2 * np.pi))
            f = f[f > 90][:3]
            if len(f) == 3:
                F.append(f)
        return stats(np.array(F).T)
    out["Spectral envelope / formants"] = formants
    return out


# ---------------------------------------------------------------- 3. energy & dynamics

def f_energy(a):
    import essentia.standard as es
    out = {}
    out["RMS energy"] = lambda: stats(a.rms)
    out["Zero-crossing rate"] = lambda: stats(librosa.feature.zero_crossing_rate(a.y, frame_length=config.N_FFT, hop_length=config.HOP))

    def lufs():
        mom, short, integrated, lra = es.LoudnessEBUR128(sampleRate=44100)(a.stereo44.T.astype(np.float32))
        a.short_lufs, a.lufs, a.lra = np.asarray(short), float(integrated), float(lra)
        return {"integrated_lufs": a.lufs, "loudness_range_lu": a.lra, **stats(a.short_lufs, "short_term_")}
    out["Loudness (EBU R128 / LUFS), loudness range"] = lufs

    def dyn():
        rms_db = db_(a.rms ** 2)
        peak, rms = np.abs(a.mono44).max(), np.sqrt((a.mono44 ** 2).mean())
        return {"dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
                "crest_factor_db": float(20 * np.log10(peak / (rms + 1e-12)))}
    out["Dynamic range, crest factor"] = dyn

    def envelope_shape():
        on = librosa.onset.onset_detect(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP)
        att, dec, tc = [], [], []
        for s, e in zip(on[:-1], on[1:]):
            seg = a.rms[s:e]
            if len(seg) < 3:
                continue
            pk = int(seg.argmax())
            att.append(pk / a.fps)
            below = np.where(seg[pk:] < seg[pk] * 0.5)[0]
            dec.append((below[0] if len(below) else len(seg) - pk) / a.fps)
            tc.append((np.arange(len(seg)) * seg).sum() / (seg.sum() + 1e-12) / len(seg))
        return {"attack_s": stats(att)["mean"], "decay_s": stats(dec)["mean"], "temporal_centroid": stats(tc)["mean"]}
    out["Attack time, decay, temporal centroid"] = envelope_shape

    def true_peak():
        tp = es.TruePeakDetector(oversamplingFactor=4, sampleRate=44100)
        peaks = [float(np.max(np.abs(tp(ch.astype(np.float32))[1]))) for ch in a.stereo44]
        a.true_peak_db = 20 * np.log10(max(peaks) + 1e-12)
        return {"true_peak_dbtp": a.true_peak_db, "clipping_ratio": float((np.abs(a.stereo44) > 0.999).mean())}
    out["True peak, clipping ratio"] = true_peak

    def contour():
        sm = scipy.signal.savgol_filter(a.rms, 101 if len(a.rms) > 101 else 5, 2)
        bins = np.array([b.mean() for b in np.array_split(sm, 20)])
        bins = bins / (bins.max() + 1e-12)
        return {"contour20": bins, "slope": float(np.polyfit(np.arange(20), bins, 1)[0]),
                "argmax_pos": float(bins.argmax() / 19), "n_buildups": int((np.diff(np.sign(np.diff(bins))) < 0).sum())}
    out["Energy contour shape"] = contour

    def plr():
        if not hasattr(a, "lufs"):
            lufs()
        if not hasattr(a, "true_peak_db"):
            true_peak()
        return {"plr_db": a.true_peak_db - a.lufs}
    out["Peak-to-loudness ratio (PLR)"] = plr
    return out


# ---------------------------------------------------------------- 4. harmony & pitch

KEYS = "C C# D D# E F F# G G# A A# B".split()


def f_harmony(a):
    import essentia.standard as es
    out = {}

    def chroma():
        cqt = librosa.feature.chroma_cqt(y=a.yh, sr=a.sr, hop_length=config.HOP)
        cens = librosa.feature.chroma_cens(C=None, y=a.yh, sr=a.sr, hop_length=config.HOP)
        return {**stats(a.chroma, "stft_"), **stats(cqt, "cqt_"), **stats(cens, "cens_")}
    out["Chroma (STFT, CQT, CENS)"] = chroma

    def hpcp():
        h = es.HPCP(sampleRate=44100, size=36)
        H = np.array([h(f, m) for f, m in a.ess_peaks]).T
        return stats(H)
    out["Harmonic pitch class profile (HPCP)"] = hpcp
    out["Tonnetz / tonal centroid"] = lambda: stats(librosa.feature.tonnetz(chroma=a.chroma))

    def key():
        k, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=44100)(a.mono44)
        a.key_idx, a.key_scale, a.key_strength = KEYS.index(k.replace("Bb", "A#").replace("Eb", "D#").replace("Ab", "G#").replace("Db", "C#").replace("Gb", "F#")), scale, strength
        return {"key": k, "mode": scale, "strength": float(strength)}
    out["Key and mode"] = key

    def chords():
        r = es.TonalExtractor()(a.mono44)
        names = [o for o in es.TonalExtractor().outputNames()]
        d = dict(zip(names, r))
        a.chords = list(d["chords_progression"])
        a.chords_strength = np.asarray(d["chords_strength"])
        return {"n_frames": len(a.chords), "chords_changes_rate": float(d["chords_changes_rate"]),
                "chords_number_rate": float(d["chords_number_rate"]), "chords_histogram": np.asarray(d["chords_histogram"])}
    out["Chord sequence"] = chords

    def f0():
        p = np.array([p for p, c in a.ess_pitch])
        c = np.array([c for p, c in a.ess_pitch])
        v = p[(c > 0.5) & (p > 0)]
        return {"voiced_ratio": float(len(v) / len(p)), **stats(librosa.hz_to_midi(v) if len(v) else np.zeros(1), "midi_")}
    out["Fundamental frequency (f0)"] = f0

    def melody():
        eq = es.EqualLoudness(sampleRate=44100)(a.mono44)
        pitch, conf = es.PredominantPitchMelodia(frameSize=2048, hopSize=256, sampleRate=44100)(eq)
        a.melody = np.asarray(pitch)
        v = a.melody[a.melody > 0]
        return {"voiced_ratio": float(len(v) / len(a.melody)), **stats(librosa.hz_to_midi(v) if len(v) else np.zeros(1))}
    out["Predominant melody"] = melody
    out["Tuning frequency"] = lambda: {"tuning_cents": float(librosa.estimate_tuning(S=a.S, sr=a.sr) * 100)}

    def inharm_hnr():
        hp, inh = es.HarmonicPeaks(), es.Inharmonicity()
        vals = []
        for (f, m), (p, c) in zip(a.ess_peaks, a.ess_pitch):
            if p > 0 and c > 0.5 and len(f):
                hf, hm = hp(f, m, p)
                if hf[0] > 0:
                    try:
                        vals.append(inh(hf, hm))
                    except Exception:
                        pass
        hnr = db_((a.yh ** 2).mean()) - db_(((a.y - a.yh - a.yp) ** 2).mean() + 1e-12)
        return {"inharmonicity": stats(vals if vals else [0])["mean"], "hnr_db_approx": float(hnr)}
    out["Inharmonicity, harmonic-to-noise ratio"] = inharm_hnr

    def tristimulus():
        hp, tri = es.HarmonicPeaks(), es.Tristimulus()
        T = []
        for (f, m), (p, c) in zip(a.ess_peaks, a.ess_pitch):
            if p > 0 and c > 0.5 and len(f):
                hf, hm = hp(f, m, p)
                T.append(tri(hf, hm))
        return stats(np.array(T).T if T else np.zeros((3, 1)))
    out["Tristimulus"] = tristimulus

    def dissonance():
        d = es.Dissonance()
        return stats([d(f, m) for f, m in a.ess_peaks if len(f) > 1])
    out["Dissonance / roughness"] = dissonance

    def key_norm_chroma():
        if not hasattr(a, "key_idx"):
            key()
        return {"chroma_rel_mean": np.roll(a.chroma.mean(1), -a.key_idx), "mode_major": float(a.key_scale == "major")}
    out["Key-normalized chroma"] = key_norm_chroma

    def key_clarity():
        if not hasattr(a, "key_strength"):
            key()
        prof = a.chroma.mean(1)
        return {"key_strength": float(a.key_strength), "chroma_entropy": float(scipy.stats.entropy(prof / prof.sum()))}
    out["Key clarity / tonalness"] = key_clarity

    def harmonic_rhythm():
        if not hasattr(a, "chords"):
            chords()
        ch = np.array(a.chords)
        changes = np.flatnonzero(ch[1:] != ch[:-1])
        sec_per_frame = a.dur / max(len(ch), 1)
        seg_len = np.diff(np.r_[0, changes + 1, len(ch)]) * sec_per_frame
        return {"chord_changes_per_min": len(changes) / a.dur * 60, "mean_chord_dur_s": float(seg_len.mean())}
    out["Chord change rate, harmonic rhythm"] = harmonic_rhythm

    def chord_vocab():
        if not hasattr(a, "chords"):
            chords()
        ch = np.array(a.chords)
        seq = ch[np.r_[True, ch[1:] != ch[:-1]]]
        minor = np.mean([c.endswith("m") for c in seq])
        trans = list(zip(seq[:-1], seq[1:]))
        _, cnt = np.unique(np.array(trans, dtype=str), axis=0, return_counts=True) if trans else (None, np.ones(1))
        return {"minor_ratio": float(minor), "n_unique_chords": len(set(seq)),
                "transition_entropy": float(scipy.stats.entropy(cnt))}
    out["Chord vocabulary statistics"] = chord_vocab

    def melodic():
        if not hasattr(a, "melody"):
            melody()
        midi = librosa.hz_to_midi(a.melody[a.melody > 0])
        steps = np.abs(np.diff(np.round(midi)))
        steps = steps[(steps > 0) & (steps < 13)]
        return {"range_semitones": float(np.percentile(midi, 95) - np.percentile(midi, 5)),
                "interval_hist": np.bincount(steps.astype(int), minlength=13)[1:13] / max(len(steps), 1)}
    out["Pitch range, melodic intervals"] = melodic

    def polyphony():
        from basic_pitch.inference import predict

        _, _, notes = predict(a.path)
        a.notes = notes
        t = np.arange(0, a.dur, 0.05)
        active = np.zeros_like(t)
        for s, e, *_ in notes:
            active[(t >= s) & (t < e)] += 1
        return {"n_notes": len(notes), "notes_per_s": len(notes) / a.dur, **stats(active[active > 0] if active.any() else [0], "poly_")}
    out["Polyphony estimate"] = polyphony

    def bass_line():
        stems = getattr(a, "stems", None)
        src = stems["bass"].mean(0) if stems else a.mono44
        yb = librosa.resample(src, orig_sr=44100, target_sr=8000)
        f0, voiced, _ = librosa.pyin(yb, fmin=30, fmax=300, sr=8000, frame_length=1024, hop_length=256)
        f = f0[voiced & ~np.isnan(f0)]
        midi = np.round(librosa.hz_to_midi(f)) if len(f) else np.zeros(1)
        motion = np.abs(np.diff(midi))
        motion = motion[(motion > 0) & (motion < 13)]
        return {"from_stem": bool(stems), "pc_hist": np.bincount((midi % 12).astype(int), minlength=12) / len(midi),
                "root_motion_hist": np.bincount(motion.astype(int), minlength=13)[1:13] / max(len(motion), 1)}
    out["Bass line pitch / root motion"] = bass_line
    return out


# ---------------------------------------------------------------- 5. rhythm

def f_rhythm(a):
    import essentia.standard as es
    out = {}

    def onsets():
        on = librosa.onset.onset_detect(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP, units="time")
        a.onset_times = on
        ioi = np.diff(on)
        return {**stats(a.onset_env, "env_"), **stats(ioi if len(ioi) else [0], "ioi_")}
    out["Onset strength envelope, onset times"] = onsets
    out["Onset rate"] = lambda: {"onset_rate": len(librosa.onset.onset_detect(onset_envelope=a.onset_env, sr=a.sr)) / a.dur}

    def tempo():
        bpm, beats, conf, est, intervals = es.RhythmExtractor2013(method="multifeature")(a.mono44)
        a.bpm_intervals = intervals
        tc = model("essentia TempoCNN", lambda: es.TempoCNN(graphFilename=str(ESS_MODELS / "deeptemp-k16-3.pb")))
        y11 = librosa.resample(a.mono44, orig_sr=44100, target_sr=11025).astype(np.float32)
        g, local, _ = tc(y11)
        return {"bpm_essentia": float(bpm), "confidence": float(conf), "bpm_tempocnn": float(g),
                "bpm_librosa": a.tempo, "local_tempo_std": float(np.std(local))}
    out["Tempo (BPM) and confidence"] = tempo

    def downbeats():
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

        act = RNNDownBeatProcessor()(a.path)
        db = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        a.db_times = db[db[:, 1] == 1, 0]
        a.beats_per_bar = int(db[:, 1].max())
        return {"n_beats": len(db), "n_downbeats": len(a.db_times), "beats_per_bar": a.beats_per_bar}
    out["Beat and downbeat positions"] = downbeats

    def tempogram():
        T = librosa.feature.tempogram(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP)
        a.tempogram = T
        return stats(T[1:100])  # first 100 lags
    out["Tempogram"] = tempogram

    def beat_hist():
        if not hasattr(a, "bpm_intervals"):
            tempo()
        r = es.BpmHistogramDescriptors()(a.bpm_intervals)
        names = es.BpmHistogramDescriptors().outputNames()
        d = {n: np.asarray(v) for n, v in zip(names, r)}
        pattern = beat_pattern(a, a.onset_env, 16)
        return {**{k: (float(v) if v.ndim == 0 else v) for k, v in d.items() if k != "histogram"}, "pattern16": pattern}
    out["Beat histogram, rhythm patterns"] = beat_hist

    def meter():
        if not hasattr(a, "beats_per_bar"):
            downbeats()
        return {"beats_per_bar": a.beats_per_bar}
    out["Meter / time signature"] = meter

    def pulse_sync():
        if not hasattr(a, "tempogram"):
            tempogram()
        ac = a.tempogram.mean(1)
        clarity = float(ac[1:].max() / (ac[1:].mean() + 1e-12))
        pat = beat_pattern(a, a.onset_env, 4)  # per-beat 16ths
        sync = float((pat[1] + pat[3]) / (pat.sum() + 1e-12))
        return {"pulse_clarity": clarity, "syncopation": sync}
    out["Pulse clarity, syncopation"] = pulse_sync
    out["Danceability"] = lambda: dict(zip(["danceability", "dfa_mean"], [float(v) if np.ndim(v) == 0 else float(np.mean(v)) for v in es.Danceability(sampleRate=44100)(a.mono44)]))

    def stability():
        ibi = np.diff(a.beat_times)
        return {"ibi_cv": float(ibi.std() / (ibi.mean() + 1e-12)),
                "tempo_drift_bpm_per_min": float(np.polyfit(a.beat_times[1:], 60 / ibi, 1)[0] * 60) if len(ibi) > 2 else 0.0}
    out["Tempo stability / drift"] = stability

    def swing():
        if not hasattr(a, "onset_times"):
            onsets()
        bt, on = a.beat_times, a.onset_times
        ratios, dev = [], []
        for b0, b1 in zip(bt[:-1], bt[1:]):
            L = b1 - b0
            inside = on[(on > b0 + 0.3 * L) & (on < b0 + 0.8 * L)]
            if len(inside):
                pos = (inside[0] - b0) / L
                ratios.append(pos / (1 - pos))
            rel = (on[(on >= b0) & (on < b1)] - b0) / L * 4
            dev += list(rel - np.round(rel))
        return {"swing_ratio": stats(ratios if ratios else [1])["mean"], "microtiming_std": float(np.std(dev)) if dev else 0.0}
    out["Swing ratio, microtiming"] = swing

    def band_patterns():
        mel = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=64)
        mf = librosa.mel_frequencies(64, fmax=a.sr / 2)
        res = {}
        for name, (lo, hi) in {"low": (0, 150), "mid": (150, 2000), "high": (4000, 11025)}.items():
            env = librosa.onset.onset_strength(S=librosa.power_to_db(mel[(mf >= lo) & (mf < hi)]), sr=a.sr)
            res[name] = beat_pattern(a, env, 16)
        return res
    out["Onset patterns per band"] = band_patterns

    def fourier_cyclic():
        F = np.abs(librosa.feature.fourier_tempogram(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP))
        bpms = librosa.fourier_tempo_frequencies(sr=a.sr, hop_length=config.HOP)
        prof = F.mean(1)
        ok = bpms > 30
        cyc_bins = (np.log2(bpms[ok] / 60) % 1 * 24).astype(int)
        cyclic = np.bincount(cyc_bins, weights=prof[ok], minlength=24)
        return {"fourier_profile_32": np.interp(np.geomspace(30, 480, 32), bpms, prof), "cyclic24": cyclic / (cyclic.sum() + 1e-12)}
    out["Fourier tempogram / cyclic tempogram"] = fourier_cyclic

    def density():
        if not hasattr(a, "db_times"):
            downbeats()
        if not hasattr(a, "onset_times"):
            onsets()
        counts = np.histogram(a.onset_times, bins=a.db_times)[0] if len(a.db_times) > 1 else np.zeros(1)
        return stats(counts)
    out["Rhythmic density per bar"] = density
    return out


def beat_pattern(a, env, subdiv):
    """Mean onset envelope at `subdiv` positions per bar (4 beats) or per beat (subdiv<=4)."""
    bt = a.beats
    per = subdiv if subdiv <= 4 else subdiv // 4
    pts = []
    for b0, b1 in zip(bt[:-1], bt[1:]):
        pts.append([env[min(int(b0 + (b1 - b0) * i / per), len(env) - 1)] for i in range(per)])
    P = np.array(pts) if pts else np.zeros((1, per))
    if subdiv > 4:
        n = len(P) // 4 * 4
        P = P[:n].reshape(-1, subdiv) if n else np.zeros((1, subdiv))
    m = P.mean(0)
    return m / (m.max() + 1e-12)


# ---------------------------------------------------------------- 6. structure

def f_structure(a):
    out = {}

    def beat_feats():
        F = np.vstack([librosa.util.normalize(a.mfcc[1:13], axis=1), a.chroma])
        a.bsync = librosa.util.sync(F, a.beats, aggregate=np.median)
        return a.bsync

    def ssm():
        beat_feats()
        R = librosa.segment.recurrence_matrix(a.bsync, mode="affinity", sym=True, width=4)
        a.R = R
        lag = librosa.segment.recurrence_to_lag(R, pad=False)
        prof = lag.mean(1)
        return {"lag_profile_64": np.interp(np.linspace(0, len(prof) - 1, 64), np.arange(len(prof)), prof),
                "density": float((R > 0).mean())}
    out["Self-similarity / recurrence matrix"] = ssm

    def novelty():
        if not hasattr(a, "R"):
            ssm()
        S = np.corrcoef(a.bsync.T)
        k = 16
        g = np.outer(scipy.signal.windows.gaussian(2 * k, k / 2), scipy.signal.windows.gaussian(2 * k, k / 2))
        kern = g * np.kron([[1, -1], [-1, 1]], np.ones((k, k)))
        Sp = np.pad(S, k, mode="constant")
        nov = np.array([(Sp[i:i + 2 * k, i:i + 2 * k] * kern).sum() for i in range(len(S))])
        a.novelty = nov
        peaks = scipy.signal.find_peaks(nov, distance=8, prominence=np.std(nov))[0]
        return {**stats(nov), "n_peaks": len(peaks)}
    out["Novelty curve"] = novelty

    def segments():
        if not hasattr(a, "bsync"):
            beat_feats()
        k = int(np.clip(a.dur / 20, 2, 12))
        bounds = librosa.segment.agglomerative(a.bsync, k)
        seg_feats = np.array([a.bsync[:, s:e].mean(1) for s, e in zip(bounds, list(bounds[1:]) + [a.bsync.shape[1]])])
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(n_clusters=max(2, k // 2)).fit_predict(seg_feats)
        bt = np.r_[0, a.beat_times]
        a.seg_times = bt[np.clip(bounds, 0, len(bt) - 1)]
        a.seg_labels = labels
        return {"n_segments": len(bounds), "n_labels": len(set(labels)), "label_seq": labels}
    out["Segment boundaries and labels"] = segments

    def intro_outro():
        sm = scipy.signal.medfilt(a.rms, 43)
        thr = 0.5 * np.median(sm)
        above = np.flatnonzero(sm > thr)
        return {"duration_s": a.dur, "intro_s": above[0] / a.fps if len(above) else 0.0,
                "outro_s": (len(sm) - above[-1]) / a.fps if len(above) else 0.0}
    out["Duration, intro/outro detection"] = intro_outro
    out["Silence ratio"] = lambda: {"silence_ratio": float((db_(a.rms ** 2) < -60).mean())}

    def repetitiveness():
        if not hasattr(a, "R"):
            ssm()
        R = a.R.copy()
        n = len(R)
        for d in range(-8, 9):
            idx = np.arange(max(0, -d), min(n, n - d))
            R[idx, idx + d] = 0
        return {"repeated_fraction": float((R.max(1) > 0).mean())}
    out["Repetitiveness score"] = repetitiveness

    def sections():
        if not hasattr(a, "seg_times"):
            segments()
        L = np.diff(np.r_[a.seg_times, a.dur])
        return {"n_sections": len(L), "mean_len_s": float(L.mean()), "std_len_s": float(L.std())}
    out["Number / mean length of sections"] = sections

    def climax():
        if not hasattr(a, "seg_times"):
            segments()
        ends = np.r_[a.seg_times[1:], a.dur]
        energy = [a.rms[int(s * a.fps):int(e * a.fps)].mean() for s, e in zip(a.seg_times, ends)]
        counts = np.bincount(a.seg_labels)
        score = np.array(energy) * counts[a.seg_labels]
        i = int(score.argmax())
        return {"climax_pos": float(a.seg_times[i] / a.dur), "climax_energy_ratio": float(energy[i] / (np.mean(energy) + 1e-12))}
    out["Chorus / drop / climax detection"] = climax

    def loops():
        if not hasattr(a, "bsync"):
            beat_feats()
        X = a.bsync - a.bsync.mean(1, keepdims=True)
        res = {}
        for bars in (1, 2, 4, 8):
            lag = bars * 4
            if X.shape[1] > lag + 4:
                c = [np.corrcoef(X[:, i], X[:, i + lag])[0, 1] for i in range(X.shape[1] - lag)]
                res[f"loop_corr_{bars}bar"] = float(np.nanmean(c))
        return res
    out["Loop detection"] = loops
    return out


# ---------------------------------------------------------------- 7. source separation

def f_separation(a):
    out = {}
    out["Harmonic/percussive separation (HPSS) ratio"] = lambda: {
        "harmonic_ratio": float((a.yh ** 2).sum() / ((a.yh ** 2).sum() + (a.yp ** 2).sum() + 1e-12))}

    def stems():
        a.stems = demucs_stems(a)
        return {k: float(np.sqrt((v ** 2).mean())) for k, v in a.stems.items()}
    out["Stem features (vocals, drums, bass, other)"] = stems

    def need():
        if getattr(a, "stems", None) is None:
            raise RuntimeError("stems unavailable")
        return a.stems

    def vocal_curve():
        v = need()["vocals"].mean(0)
        r = librosa.feature.rms(y=v, frame_length=4096, hop_length=2048)[0]
        mix_r = librosa.feature.rms(y=a.mono44, frame_length=4096, hop_length=2048)[0]
        active = db_(r ** 2) > db_(mix_r ** 2) - 15
        a.vocal_active = active
        return {"vocal_ratio": float(active.mean()), "curve20": np.array([b.mean() for b in np.array_split(active, 20)])}
    out["Vocal activity curve"] = vocal_curve

    def vocal_presence():
        if not hasattr(a, "vocal_active"):
            vocal_curve()
        e = {k: (v ** 2).mean() for k, v in need().items()}
        return {"vocal_energy_share": float(e["vocals"] / sum(e.values())), "instrumentalness": float(1 - a.vocal_active.mean())}
    out["Vocal presence / instrumentalness"] = vocal_presence

    def speechiness():
        if not hasattr(a, "vocal_pitch"):
            vocal_crepe(a)
        p, per, act = a.vocal_pitch, a.vocal_periodicity, a.vocal_frames_active
        if not act.any():
            return {"speechiness": 0.0}
        cents = 1200 * np.log2(np.maximum(p, 1) / 440)
        jumpy = np.abs(np.diff(cents, prepend=cents[0])) > 50
        return {"speechiness": float(((per < 0.5) | jumpy)[act].mean())}
    out["Speechiness"] = speechiness

    def balance():
        e = {k: (v ** 2).mean() for k, v in need().items()}
        tot = sum(e.values())
        return {k: float(v / tot) for k, v in e.items()}
    out["Stem energy balance"] = balance

    def per_stem():
        res = {}
        for k, v in need().items():
            m = librosa.resample(v.mean(0), orig_sr=44100, target_sr=a.sr)
            S = np.abs(librosa.stft(m, n_fft=config.N_FFT, hop_length=config.HOP))
            mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=a.sr), ref=np.max)
            res[k] = np.concatenate([stats(librosa.feature.spectral_centroid(S=S, sr=a.sr))["mean"],
                                     stats(librosa.feature.mfcc(S=mel, n_mfcc=13))["mean"],
                                     [len(librosa.onset.onset_detect(y=m, sr=a.sr)) / a.dur]])
        return res
    out["Per-stem feature sets"] = per_stem
    return out


def vocal_crepe(a):
    import torch
    import torchcrepe

    v = librosa.resample(a.stems["vocals"].mean(0), orig_sr=44100, target_sr=16000)
    with torch.no_grad():
        p, per = torchcrepe.predict(torch.tensor(v)[None], 16000, hop_length=160, fmin=60, fmax=1000, model="full",
                                    return_periodicity=True, batch_size=1024, device="cpu")
    a.vocal_pitch, a.vocal_periodicity = p[0].numpy(), per[0].numpy()
    r = librosa.feature.rms(y=v, frame_length=1024, hop_length=160)[0][: len(a.vocal_pitch)]
    a.vocal_frames_active = np.zeros(len(a.vocal_pitch), bool)
    a.vocal_frames_active[: len(r)] = db_(r ** 2) > db_(np.max(r) ** 2) - 35
    a.vocal_voiced = a.vocal_frames_active & (a.vocal_periodicity > 0.5)


# ---------------------------------------------------------------- 8. semantic (Essentia heads on Discogs-EffNet)

def f_semantic(a):
    out = {}

    def emb():
        if not hasattr(a, "effnet"):
            a.effnet = effnet_embeddings(a)
        return a.effnet

    def heads(*names):
        return lambda: {n: ess_head(f"{n}-discogs-effnet-1", emb()) for n in names}
    out["Genre and subgenre"] = heads("genre_discogs400", "mtg_jamendo_genre")

    def mood():
        res = heads("mood_happy", "mood_sad", "mood_relaxed", "mood_aggressive", "mood_party", "mtg_jamendo_moodtheme")()
        if not hasattr(a, "musicnn"):
            a.musicnn = musicnn_embeddings(a)
        res["deam_arousal_valence"] = ess_head("deam-msd-musicnn-2", a.musicnn)
        res["emomusic_arousal_valence"] = ess_head("emomusic-msd-musicnn-2", a.musicnn)
        return res
    out["Mood and emotion (valence, arousal)"] = mood
    out["Energy, acousticness, liveness"] = lambda: {**heads("mood_acoustic", "mood_electronic", "nsynth_acoustic_electronic")(),
                                                     "liveness": "no open model"}
    out["Instrument recognition / tags"] = heads("mtg_jamendo_instrument")
    out["Vocal gender, singing technique"] = heads("gender", "voice_instrumental")
    out["Production / audio quality"] = heads("nsynth_bright_dark", "nsynth_reverb", "timbre")

    def language():
        from transformers import pipeline

        asr = model("whisper-tiny", lambda: pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu"))
        src = a.stems["vocals"].mean(0) if getattr(a, "stems", None) else a.mono44
        y16 = librosa.resample(src, orig_sr=44100, target_sr=16000)
        r = asr({"raw": y16, "sampling_rate": 16000}, chunk_length_s=30, return_language=True,
                generate_kwargs={"task": "transcribe"})
        a.transcript = r["text"]
        langs = [c.get("language") for c in r.get("chunks", []) if c.get("language")]
        return {"language": max(set(langs), key=langs.count) if langs else "?", "n_words": len(r["text"].split())}
    out["Language of vocals"] = language
    out["Context / use-case tags"] = lambda: {"moodtheme": ess_head("mtg_jamendo_moodtheme-discogs-effnet-1", emb())}
    out["Approachability / engagement"] = heads("approachability_regression", "engagement_regression")
    return out


# ---------------------------------------------------------------- 9. learned embeddings

def chunks(y, sr, sec=10.0):
    n = int(sr * sec)
    return [y[i:i + n] for i in range(0, max(len(y) - n // 2, 1), n)]


def f_learned(a):
    import essentia.standard as es
    out = {}

    def vggish_yamnet_panns():
        res = {}
        vg = model("essentia vggish", lambda: es.TensorflowPredictVGGish(graphFilename=str(ESS_MODELS / "audioset-vggish-3.pb"), output="model/vggish/embeddings"))
        res["vggish"] = np.asarray(vg(a.y16)).mean(0)
        ym = model("essentia yamnet", lambda: es.TensorflowPredictVGGish(graphFilename=str(ESS_MODELS / "audioset-yamnet-1.pb"), input="melspectrogram", output="activations"))
        res["yamnet_classes"] = np.asarray(ym(a.y16)).mean(0)
        from panns_inference import AudioTagging

        at = model("PANNs Cnn14", lambda: AudioTagging(checkpoint_path=None, device="cpu"))
        y32 = librosa.resample(a.mono44, orig_sr=44100, target_sr=32000)
        clip, emb = at.inference(y32[None])
        res["panns_embedding"], res["panns_classes"] = emb[0], clip[0]
        return res
    out["VGGish, YAMNet, PANNs"] = vggish_yamnet_panns
    out["OpenL3"] = lambda: openl3(a)
    out["MusiCNN"] = lambda: {"musicnn": np.asarray(getattr(a, "musicnn", None) if hasattr(a, "musicnn") else musicnn_embeddings(a)).mean(0)}

    def mert_muq():
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        res = {}
        captured = []
        proc = model("MERT processor", lambda: Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True))
        m = model("MERT-v1-95M", lambda: AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval())
        # transformers v5 drops hidden_states for this remote-code model: capture layer outputs with hooks
        hooks = [l.register_forward_hook(lambda mod, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
                 for l in m.encoder.layers]
        y24 = librosa.resample(a.mono44, orig_sr=44100, target_sr=24000)
        layers = []
        with torch.no_grad():
            for c in chunks(y24, 24000):
                inp = proc(c, sampling_rate=24000, return_tensors="pt")
                captured.clear()
                m(**inp)
                layers.append(torch.stack(captured).squeeze(1).mean(1).numpy())  # (12, 768)
        for h in hooks:
            h.remove()
        res["mert_layers_mean"] = np.mean(layers, 0).mean(0)
        try:
            from muq import MuQ

            mq = model("MuQ-large", lambda: MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").eval())
            y24t = torch.tensor(y24, dtype=torch.float32)
            embs = []
            with torch.no_grad():
                for c in chunks(y24t, 24000):
                    embs.append(mq(c[None], output_hidden_states=False).last_hidden_state.mean(1)[0].numpy())
            res["muq"] = np.mean(embs, 0)
        except Exception as e:
            res["muq_error"] = f"{type(e).__name__}: {e}"[:100]
        return res
    out["MERT, MuQ"] = mert_muq

    def clap():
        import torch
        from transformers import ClapModel, ClapProcessor

        proc = model("CLAP processor", lambda: ClapProcessor.from_pretrained("laion/clap-htsat-unfused"))
        m = model("CLAP htsat-unfused", lambda: ClapModel.from_pretrained("laion/clap-htsat-unfused").eval())
        y48 = librosa.resample(a.mono44, orig_sr=44100, target_sr=48000)
        with torch.no_grad():
            cs = chunks(y48, 48000)
            try:
                inp = proc(audio=cs, sampling_rate=48000, return_tensors="pt")
            except TypeError:
                inp = proc(audios=cs, sampling_rate=48000, return_tensors="pt")
            e = m.get_audio_features(**inp)
            e = getattr(e, "pooler_output", e)
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            prompts = ["strings", "piano", "brass", "woodwinds", "acoustic guitar", "electric guitar",
                       "drums", "synthesizer", "singing voice", "choir"]
            t = proc(text=[f"music with {p}" for p in prompts], return_tensors="pt", padding=True)
            te = m.get_text_features(**t)
            te = torch.nn.functional.normalize(getattr(te, "pooler_output", te), dim=-1)
            sims = (te @ torch.nn.functional.normalize(e, dim=0)).numpy()
        return {"clap": e.numpy(), "zero_shot_instruments": dict(zip(prompts, sims.astype(float)))}
    out["CLAP"] = clap

    def encodec():
        import torch
        from transformers import AutoProcessor, EncodecModel

        proc = model("Encodec processor", lambda: AutoProcessor.from_pretrained("facebook/encodec_24khz"))
        m = model("Encodec 24khz", lambda: EncodecModel.from_pretrained("facebook/encodec_24khz").eval())
        y24 = librosa.resample(a.mono44, orig_sr=44100, target_sr=24000)
        inp = proc(raw_audio=y24, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            z = m.encoder(inp["input_values"])[0]  # (128, T)
        return {"latent_mean": z.mean(1).numpy(), "latent_std": z.std(1).numpy()}
    out["Encodec / Jukebox latents"] = encodec

    def effnet_maest():
        if not hasattr(a, "effnet"):
            a.effnet = effnet_embeddings(a)
        meta = json.load(open(ESS_MODELS / "discogs-maest-30s-pw-2.json"))
        outs = meta["schema"]["outputs"]
        emb_out = next((o["name"] for o in outs if o.get("output_purpose") == "embeddings"), outs[0]["name"])
        mm = model("essentia MAEST-30s", lambda: es.TensorflowPredictMAEST(graphFilename=str(ESS_MODELS / "discogs-maest-30s-pw-2.pb"), output=emb_out))
        e = np.asarray(mm(a.y16))
        return {"effnet": np.asarray(a.effnet).mean(0), "maest": e.reshape(-1, e.shape[-1]).mean(0)}
    out["Essentia Discogs-EffNet / MAEST"] = effnet_maest

    def ast():
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        fe = model("AST extractor", lambda: ASTFeatureExtractor.from_pretrained(name))
        m = model("AST audioset", lambda: ASTForAudioClassification.from_pretrained(name).eval())
        embs, logits = [], []
        with torch.no_grad():
            for c in chunks(a.y16, 16000):
                o = m(**fe(c, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True)
                embs.append(o.hidden_states[-1][0].mean(0).numpy())
                logits.append(torch.sigmoid(o.logits[0]).numpy())
        return {"ast_embedding": np.mean(embs, 0), "ast_classes": np.mean(logits, 0)}
    out["BEATs, AST, HTS-AT"] = ast
    return out


def openl3(a):
    """Essentia OpenL3 (music, mel128, 512-d), following the Essentia model docs."""
    import essentia.standard as es
    from essentia import Pool

    sr, frame, hop = 48000, 2048, 242
    y48 = librosa.resample(a.mono44, orig_sr=44100, target_sr=sr).astype(np.float32)
    w, spec = es.Windowing(size=frame, normalized=False), es.Spectrum(size=frame)
    mb = es.MelBands(highFrequencyBound=sr / 2, inputSize=frame // 2 + 1, log=False, lowFrequencyBound=0,
                     normalize="unit_tri", numberBands=128, sampleRate=sr, type="magnitude",
                     warpingFormula="slaneyMel", weighting="linear")
    patches = []
    for chunk in es.FrameGenerator(y48, frameSize=sr, hopSize=sr, startFromZero=True):
        M = np.array([mb(spec(w(f))) for f in es.FrameGenerator(chunk, frameSize=frame, hopSize=hop, startFromZero=True, validFrameThresholdRatio=0.5)])
        M = 10 * np.log10(np.maximum(1e-10, M))
        M = np.maximum(M, M.max() - 80) - M.max()
        x = np.zeros((199, 128), np.float32)
        x[: min(199, len(M))] = M[:199]
        patches.append(x)
    batch = np.transpose(np.array(patches)[:, None], [0, 3, 2, 1])  # (n, 128, 199, 1)
    m = model("essentia openl3", lambda: es.TensorflowPredict(graphFilename=str(ESS_MODELS / "openl3-music-mel128-emb512-3.pb"),
                                                               inputs=["melspectrogram"], outputs=["embeddings"], squeeze=False))
    pool = Pool()
    pool.set("melspectrogram", batch)
    e = np.asarray(m(pool)["embeddings"]).squeeze()
    return {"openl3": e.reshape(-1, e.shape[-1]).mean(0)}


# ---------------------------------------------------------------- 10. summary statistics

def f_summary(a):
    M = a.mfcc
    out = {}
    out["Mean, standard deviation, min, max"] = lambda: {"mean": M.mean(1), "std": M.std(1), "min": M.min(1), "max": M.max(1)}
    out["Percentiles (e.g., 10th, 50th, 90th)"] = lambda: {f"p{q}": np.percentile(M, q, axis=1) for q in (10, 50, 90)}
    out["Skewness and kurtosis"] = lambda: {"skew": scipy.stats.skew(M, axis=1), "kurt": scipy.stats.kurtosis(M, axis=1)}
    out["Mean and variance of deltas"] = lambda: {"dmean": np.diff(M).mean(1), "dvar": np.diff(M).var(1)}
    out["Statistics computed per segment or per beat (beat-synchronous features)"] = lambda: {
        "beat_sync_std_mean": librosa.util.sync(M, a.beats).std(1)}
    out["Covariance / Gaussian summary (mean + covariance of frame features)"] = lambda: {
        "mean": M.mean(1), "cov_upper": np.cov(M)[np.triu_indices(len(M))]}

    def bag():
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=64, random_state=0, n_init=3).fit(M.T)  # demo codebook on this track
        return {"bof_hist64": np.bincount(km.labels_, minlength=64) / M.shape[1]}
    out["Bag-of-frames histograms (k-means codebook over frames)"] = bag

    def traj():
        seg = int(10 * a.fps)
        return {"segments_mfcc_mean": np.array([M[:, i:i + seg].mean(1) for i in range(0, M.shape[1], seg)])}
    out["Trajectories: keep per-segment vectors instead of collapsing (song as a path)"] = traj
    return out


# ---------------------------------------------------------------- 11. psychoacoustic (mosqito, 48 kHz)

def f_psycho(a):
    out = {}
    sig = lambda: librosa.resample(a.mono44, orig_sr=44100, target_sr=48000)

    def zw():
        from mosqito.sq_metrics import loudness_zwtv

        N, N_spec, bark, t = loudness_zwtv(sig(), 48000, field_type="free")
        a.N_spec = N_spec
        return {**stats(N, "sones_"), "specific_mean": np.asarray(N_spec).mean(1)}
    out["Zwicker loudness (sones), specific loudness"] = zw

    def sharp():
        from mosqito.sq_metrics import sharpness_din_tv

        S, t = sharpness_din_tv(sig(), 48000, weighting="din", field_type="free", skip=0.5)
        return stats(S)
    out["Sharpness"] = sharp

    def rough():
        from mosqito.sq_metrics import roughness_dw

        R, R_spec, bark, t = roughness_dw(sig(), 48000, overlap=0.5)
        return stats(R)
    out["Roughness"] = rough

    def fluct():
        # fluctuation strength proxy: modulation energy around 4 Hz in band envelopes
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=24) ** 0.3
        F = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1))
        mf = np.fft.rfftfreq(env.shape[1], 1 / a.fps)
        band = (mf > 1) & (mf < 8)
        return {"fluct_4hz_ratio": float(F[:, band].sum() / (F[:, mf > 0.5].sum() + 1e-12))}
    out["Fluctuation strength"] = fluct

    def tonality():
        from mosqito.sq_metrics import pr_ecma_st

        x = sig()[: 48000 * 10]  # stationary metric: a 10 s excerpt
        t_pr, pr, prom, freqs = pr_ecma_st(x, 48000, prominence=True)
        return {"total_prominence_ratio": float(t_pr), "n_prominent_tones": int(np.sum(prom))}
    out["Tonality (psychoacoustic)"] = tonality

    def warmth():
        fr, P = a.freqs, a.P.mean(1)
        tot = P.sum() + 1e-12
        return {"warmth_100_500": float(P[(fr >= 100) & (fr < 500)].sum() / tot),
                "boominess_60_200": float(P[(fr >= 60) & (fr < 200)].sum() / tot)}
    out["Warmth / boominess"] = warmth
    return out


# ---------------------------------------------------------------- 12. modulation & texture

def f_texture(a):
    out = {}

    def modspec():
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=32) ** 0.3
        F = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1))
        mf = np.fft.rfftfreq(env.shape[1], 1 / a.fps)
        edges = [0.5, 1, 2, 4, 8, 16, 21]
        return {"mod_power": np.array([[F[b, (mf >= lo) & (mf < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                                       for b in range(32)]).ravel()}
    out["Modulation spectrum"] = modspec

    def scattering():
        import scipy.special

        if not hasattr(scipy.special, "sph_harm"):  # removed in scipy 1.17; kymatio only needs it for 3D
            scipy.special.sph_harm = None
        from kymatio.torch import Scattering1D

        N = 2 ** 16
        sc = model("kymatio Scattering1D", lambda: Scattering1D(J=10, shape=N, Q=8))
        n = len(a.y) // N
        x = torch.tensor(a.y[: n * N].reshape(n, N), dtype=torch.float32)
        with torch.no_grad():
            Sx = sc(x)
        return {"scattering_logmean": torch.log1p(Sx).mean((0, 2)).numpy()}
    out["Scattering transform"] = scattering

    def strf():
        M = a.mel_db
        res = []
        for rate in (2, 4, 8, 16):          # Hz along time
            for scale in (0.25, 0.5, 1, 2):  # cycles per octave (~16 mel bins/octave)
                t = np.arange(-16, 16) / a.fps
                f = np.arange(-16, 16) / 16
                g = np.outer(np.exp(-(f / (1 / scale)) ** 2) * np.cos(2 * np.pi * scale * f),
                             np.exp(-(t / (1 / rate)) ** 2) * np.cos(2 * np.pi * rate * t))
                res.append(np.abs(scipy.signal.fftconvolve(M, g, mode="same")).mean())
        return {"strf16": np.array(res)}
    out["Spectro-temporal receptive fields (2D Gabor)"] = strf

    def mcdermott():
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=32) ** 0.3
        mu = env.mean(1)
        marg = np.stack([mu, env.var(1) / (mu ** 2 + 1e-12), scipy.stats.skew(env, axis=1), scipy.stats.kurtosis(env, axis=1)])
        C = np.corrcoef(env)[np.triu_indices(32, 1)]
        return {"marginals": marg.ravel(), "band_corr": C, **modspec()}
    out["Texture statistics (McDermott-Simoncelli)"] = mcdermott

    def stationarity():
        flux = np.sqrt((np.diff(a.S, axis=1).clip(0) ** 2).sum(0))
        blocks = [a.mel_db[:, i:i + int(5 * a.fps)].mean(1) for i in range(0, a.mel_db.shape[1], int(5 * a.fps))]
        return {"flux_var": float(flux.var()), "stationarity": float(np.mean(np.linalg.norm(np.diff(blocks, axis=0), axis=1)))}
    out["Spectral flux variance, stationarity"] = stationarity
    return out


# ---------------------------------------------------------------- 13. stereo

def f_stereo(a):
    L, R = a.stereo44
    out = {}

    def width():
        M, S = (L + R) / 2, (L - R) / 2
        return {"side_mid_ratio_db": float(db_((S ** 2).mean()) - db_((M ** 2).mean()))}
    out["Stereo width, mid/side energy ratio"] = width
    out["Inter-channel correlation"] = lambda: {"corr": float(np.corrcoef(L, R)[0, 1])}

    def panning():
        SL, SR = np.abs(librosa.stft(L, n_fft=2048, hop_length=2048)), np.abs(librosa.stft(R, n_fft=2048, hop_length=2048))
        fr = librosa.fft_frequencies(sr=44100, n_fft=2048)
        pan = (SR - SL) / (SR + SL + 1e-9)
        res = {}
        for name, (lo, hi) in {"low": (20, 250), "mid": (250, 2000), "high": (2000, 8000), "air": (8000, 20000)}.items():
            sel = (fr >= lo) & (fr < hi)
            w = (SL + SR)[sel].ravel()
            res[name] = np.histogram(pan[sel].ravel(), bins=7, range=(-1, 1), weights=w)[0] / (w.sum() + 1e-12)
        return res
    out["Panning distribution per band"] = panning

    def ild_itd():
        ild = db_((L ** 2).mean()) - db_((R ** 2).mean())
        seg = slice(0, min(len(L), 44100 * 20))
        c = scipy.signal.correlate(L[seg], R[seg], mode="full", method="fft")
        lags = np.arange(-len(L[seg]) + 1, len(L[seg]))
        win = np.abs(lags) <= 44
        return {"ild_db": float(ild), "itd_ms": float(lags[win][c[win].argmax()] / 44.1)}
    out["Inter-channel level / time differences"] = ild_itd

    def reverb():
        # blind decay estimate: energy decay slope after strong offsets
        e = db_(librosa.feature.rms(y=a.mono44, frame_length=1024, hop_length=256)[0] ** 2)
        fps = 44100 / 256
        slopes = []
        for p in scipy.signal.find_peaks(e, prominence=10, distance=int(0.3 * fps))[0]:
            seg = e[p:p + int(0.5 * fps)]
            if len(seg) > 10:
                slopes.append(np.polyfit(np.arange(len(seg)) / fps, seg, 1)[0])
        s = np.median(slopes) if slopes else np.nan
        return {"rt60_est_s": float(-60 / s) if s and s < 0 else float("nan"), "n_events": len(slopes)}
    out["Reverb amount, estimated RT60, direct-to-reverberant ratio"] = reverb
    return out


# ---------------------------------------------------------------- 14. production

def f_production(a):
    out = {}
    S44 = None

    def lt_spectrum():
        nonlocal S44
        if S44 is None:
            S44 = np.mean([s ** 2 for s in a.ess_spectra], axis=0)
        return S44, np.linspace(0, 22050, len(S44))

    def balance():
        P, fr = lt_spectrum()
        centers = 1000 * 2 ** (np.arange(-15, 14) / 3)
        lev = np.array([db_(P[(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6))].mean()) for c in centers])
        fit = np.polyfit(np.log2(centers), lev, 1)
        return {"third_octave_rel": lev - np.polyval(fit, np.log2(centers)), "tilt_db_per_oct": float(fit[0])}
    out["Spectral balance / tonal curve"] = balance

    def compression():
        crest = 20 * np.log10(np.abs(a.mono44).max() / (np.sqrt((a.mono44 ** 2).mean()) + 1e-12))
        r = a.rms
        return {"crest_db": float(crest), "rms_p95_p50_db": float(db_(np.percentile(r, 95) ** 2) - db_(np.percentile(r, 50) ** 2))}
    out["Compression / limiting indicators"] = compression

    def sub_bass():
        P, fr = lt_spectrum()
        return {"sub_bass_ratio": float(P[(fr > 20) & (fr < 60)].sum() / P.sum())}
    out["Sub-bass presence"] = sub_bass

    def noise():
        P, fr = lt_spectrum()
        rms_db = db_(a.rms ** 2)
        hum = {}
        for f0 in (50, 60):
            peak = np.mean([P[np.argmin(np.abs(fr - k * f0))] for k in (1, 2, 3)])
            around = np.mean([P[(np.abs(fr - k * f0) > 5) & (np.abs(fr - k * f0) < 20)].mean() for k in (1, 2, 3)])
            hum[f"hum_{f0}hz_db"] = float(db_(peak) - db_(around))
        quiet = np.array(a.ess_spectra)[np.argsort([s.sum() for s in a.ess_spectra])[: max(1, len(a.ess_spectra) // 10)]]
        hiss = db_((quiet[:, fr > 12000] ** 2).mean())
        return {"noise_floor_db": float(np.percentile(rms_db, 5)), "hiss_db": float(hiss), **hum}
    out["Noise floor, hiss, hum (50/60 Hz)"] = noise

    def codec():
        P, fr = lt_spectrum()
        Pd = db_(P)
        above = np.flatnonzero(Pd > Pd.max() - 90)
        return {"effective_bandwidth_hz": float(fr[above[-1]]) if len(above) else 0.0}
    out["Codec artifacts, effective bandwidth"] = codec

    def distortion():
        x = a.mono44
        return {"waveform_kurtosis": float(scipy.stats.kurtosis(x)),
                "near_full_scale_ratio": float((np.abs(x) > 0.95 * np.abs(x).max()).mean())}
    out["Saturation / distortion estimate"] = distortion

    def live():
        ibi = np.diff(a.beat_times)
        return {"ibi_cv": float(ibi.std() / ibi.mean()), "note": "heuristic cues only; no open liveness model"}
    out["Live vs. studio cues"] = live
    return out


# ---------------------------------------------------------------- 15. vocal

def f_vocal(a):
    out = {}

    def need():
        if getattr(a, "stems", None) is None:
            raise RuntimeError("stems unavailable")
        if not hasattr(a, "vocal_pitch"):
            vocal_crepe(a)
        return a.vocal_pitch[a.vocal_voiced]

    def pitch_range():
        v = need()
        midi = librosa.hz_to_midi(v) if len(v) else np.zeros(1)
        return {"median_midi": float(np.median(midi)), "range_semitones": float(np.percentile(midi, 95) - np.percentile(midi, 5)),
                "voiced_seconds": len(v) / 100}
    out["Vocal pitch range, median pitch"] = pitch_range

    def vibrato():
        need()
        cents = 1200 * np.log2(np.maximum(a.vocal_pitch, 1) / 440)
        runs = np.split(np.arange(len(cents)), np.flatnonzero(np.diff(a.vocal_voiced.astype(int)) != 0) + 1)
        rates, ext = [], []
        for r in runs:
            if len(r) >= 50 and a.vocal_voiced[r[0]]:
                c = scipy.signal.detrend(cents[r])
                F = np.abs(np.fft.rfft(c * np.hanning(len(c))))
                f = np.fft.rfftfreq(len(c), 0.01)
                band = (f >= 4) & (f <= 8)
                if band.any():
                    rates.append(f[band][F[band].argmax()])
                    ext.append(c.std() * 2)
        return {"vibrato_rate_hz": stats(rates or [0])["mean"], "vibrato_extent_cents": stats(ext or [0])["mean"]}
    out["Vibrato rate and extent"] = vibrato

    def voice_quality():
        v = need()
        if len(v) < 3:
            return {"jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0}
        T = 1 / v
        jitter = np.mean(np.abs(np.diff(T))) / np.mean(T)
        voc = librosa.resample(a.stems["vocals"].mean(0), orig_sr=44100, target_sr=16000)
        amp = librosa.feature.rms(y=voc, frame_length=640, hop_length=160)[0][: len(a.vocal_voiced)]
        amp = amp[a.vocal_voiced[: len(amp)]]
        shimmer = np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + 1e-12)
        per = a.vocal_periodicity[a.vocal_voiced].clip(1e-3, 1 - 1e-3)
        return {"jitter": float(jitter), "shimmer": float(shimmer), "hnr_db": float(np.mean(10 * np.log10(per / (1 - per))))}
    out["Breathiness, jitter, shimmer, HNR"] = voice_quality

    def rap_sung():
        need()
        act = a.vocal_frames_active
        return {"sung_ratio": float(a.vocal_voiced.sum() / max(act.sum(), 1))}
    out["Rap vs. sung ratio"] = rap_sung

    def syllables():
        need()
        voc = librosa.resample(a.stems["vocals"].mean(0), orig_sr=44100, target_sr=a.sr)
        on = librosa.onset.onset_detect(y=voc, sr=a.sr, units="time")
        active_s = a.vocal_frames_active.sum() / 100
        res = {"vocal_onsets_per_s": len(on) / max(active_s, 1e-3)}
        if hasattr(a, "transcript"):
            res["words_per_s"] = len(a.transcript.split()) / max(active_s, 1e-3)
        return res
    out["Syllable / word rate"] = syllables

    def layering():
        need()
        voc = librosa.resample(a.stems["vocals"].mean(0), orig_sr=44100, target_sr=a.sr)
        C = librosa.feature.chroma_cqt(y=voc, sr=a.sr)
        act = C.max(0) > 0.5
        n_peaks = (C > 0.6 * C.max(0, keepdims=True)).sum(0)
        return {"mean_simultaneous_pcs": float(n_peaks[act].mean()) if act.any() else 0.0}
    out["Harmony vocals / choir presence"] = layering
    return out


# ---------------------------------------------------------------- 16. allowed non-audio

def f_meta(a, track):
    out = {"Artist": lambda: {"artist": track["artist"]}, "Title": lambda: {"title": track["title"]}}

    def lyrics():
        from sentence_transformers import SentenceTransformer

        text = track["lyrics"] or getattr(a, "transcript", "")
        if not text:
            raise RuntimeError("no lyrics and no transcript")
        m = model("MiniLM-L6 (lyrics)", lambda: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"))
        return {"source": "lyrics" if track["lyrics"] else "whisper transcript", "embedding": m.encode(text)}
    out["Lyrics embeddings, lyric themes"] = lyrics
    return out


# ---------------------------------------------------------------- main

NOT_AVAILABLE = {
    ("8. semantic", "Era / decade estimation"): "no open pretrained model",
    ("9. learned", "Encodec / Jukebox latents [Jukebox]"): "Jukebox is 5B params; infeasible on CPU",
    ("9. learned", "DAC / SoundStream latents"): "descript-audio-codec has a dependency conflict; no SoundStream release",
    ("9. learned", "Contrastive / triplet models trained here"): "requires training on our DB first",
    ("10. summary", "Temporal pooling with attention or learned weights"): "requires training",
}


def main():
    track_id = int(sys.argv[1]) if len(sys.argv) > 1 else 189
    con = db.connect()
    track = db.get_track(con, track_id)
    print(f"Track #{track_id}: {track['artist']} - {track['title']} ({track['duration_s']:.0f}s)\n", flush=True)
    t_all = time.perf_counter()
    a = load_shared(track)

    groups = [
        ("1. spectral", lambda: f_spectral(a)), ("2. timbre", lambda: f_timbre(a)), ("3. energy", lambda: f_energy(a)),
        ("7. separation", lambda: f_separation(a)),  # stems early: harmony/vocal reuse them
        ("4. harmony", lambda: f_harmony(a)), ("5. rhythm", lambda: f_rhythm(a)), ("6. structure", lambda: f_structure(a)),
        ("8. semantic", lambda: f_semantic(a)), ("9. learned", lambda: f_learned(a)), ("10. summary", lambda: f_summary(a)),
        ("11. psychoacoustic", lambda: f_psycho(a)), ("12. texture", lambda: f_texture(a)), ("13. stereo", lambda: f_stereo(a)),
        ("14. production", lambda: f_production(a)), ("15. vocal", lambda: f_vocal(a)), ("16. non-audio", lambda: f_meta(a, track)),
    ]
    only = [c.strip() for c in os.environ.get("BENCH_ONLY", "").split(",") if c.strip()]
    for cat, make in groups:
        if only and cat.split(".")[0] not in only:
            continue
        print(f"\n== {cat}", flush=True)
        for name, fn in make().items():
            run(cat, name, fn)
        _MODELS.clear()  # free models between categories; keeping all resident caused swapping
        gc.collect()
        for (c, n), note in NOT_AVAILABLE.items():
            if c == cat:
                na(c, n, note)
    total = time.perf_counter() - t_all

    # ---- report
    print("\n" + "=" * 78)
    by_cat = {}
    for r in RESULTS:
        d = by_cat.setdefault(r["cat"], {"seconds": 0.0, "ok": 0, "fail": 0, "n/a": 0, "dims": 0})
        d["seconds"] += r["seconds"]
        d[r["status"]] += 1
        d["dims"] += r["dims"] or 0
    print(f"{'category':<22}{'seconds':>9}{'ok':>5}{'fail':>6}{'n/a':>5}{'dims':>8}")
    for c, d in by_cat.items():
        print(f"{c:<22}{d['seconds']:>9.1f}{d['ok']:>5}{d['fail']:>6}{d['n/a']:>5}{d['dims']:>8}")
    load = by_cat.get("model load", {}).get("seconds", 0)
    print(f"\ntrack duration {a.dur:.0f}s | wall total {total:.0f}s | model loading {load:.0f}s | "
          f"per-track compute ~{total - load:.0f}s ({(total - load) / a.dur:.2f}x realtime)")
    print("slowest:")
    for r in sorted((r for r in RESULTS if r["cat"] != "model load"), key=lambda r: -r["seconds"])[:12]:
        print(f"  {r['seconds']:7.1f}s  [{r['cat']}] {r['feature']}")

    out_dir = config.DATA_DIR / "bench"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / (f"track_{track_id}" + (f"_only{'-'.join(only)}" if only else "") + ".json")
    json.dump({"track_id": track_id, "duration_s": a.dur, "wall_s": total, "results": RESULTS,
               "values": VALUES}, open(path, "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    print(f"\nresults: {path}")


if __name__ == "__main__":
    main()
