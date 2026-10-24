"""Signal-processing features on the full track (catalogue sections 1-6, 10-14).

Ported from experiments/benchmark_all_features.py. Left out for speed (see TODO.md):
mosqito psychoacoustics (Zwicker loudness, sharpness, roughness, tonality), madmom
downbeats / meter, the per-track k-means codebook and per-segment trajectories.
Stem-based features live in `stems.py`.
"""

import librosa
import numpy as np
import scipy.signal
import scipy.stats

from .. import config
from .audio import Audio, db_
from .common import run_features, stats

EXTRACTOR = "dsp"
VERSION = 1

KEYS = "C C# D D# E F F# G G# A A# B".split()
FLATS = {"Bb": "A#", "Eb": "D#", "Ab": "G#", "Db": "C#", "Gb": "F#"}


class Stage:
    name = "dsp"
    outputs = [(EXTRACTOR, VERSION)]

    def load(self):
        import essentia.standard as es
        import scipy.special

        if not hasattr(scipy.special, "sph_harm"):  # removed in scipy 1.17; kymatio only needs it for 3D
            scipy.special.sph_harm = None
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        from kymatio.torch import Scattering1D

        self.tempocnn = es.TempoCNN(graphFilename=str(config.DATA_DIR / "models" / "essentia" / "deeptemp-k16-3.pb"))
        self.basic_pitch = Model(ICASSP_2022_MODEL_PATH)
        self.scattering = Scattering1D(J=10, shape=2 ** 16, Q=8)

    def process(self, a: Audio) -> dict:
        groups = {
            "spectral": f_spectral(a), "timbre": f_timbre(a), "energy": f_energy(a),
            "harmony": f_harmony(a, self), "rhythm": f_rhythm(a, self), "structure": f_structure(a),
            "summary": f_summary(a), "psycho": f_psycho(a), "texture": f_texture(a, self),
            "stereo": f_stereo(a), "production": f_production(a),
        }
        return {EXTRACTOR: run_features(groups, EXTRACTOR)}


# ---------------------------------------------------------------- 1. spectral

def f_spectral(a):
    S, P, fr = a.S, a.P, a.freqs
    Pn = P / (P.sum(0, keepdims=True) + 1e-12)
    out = {
        "centroid": lambda: stats(librosa.feature.spectral_centroid(S=S, sr=a.sr)),
        "bandwidth": lambda: stats(librosa.feature.spectral_bandwidth(S=S, sr=a.sr)),
        "rolloff": lambda: stats(librosa.feature.spectral_rolloff(S=S, sr=a.sr)),
        "flatness": lambda: stats(librosa.feature.spectral_flatness(S=S)),
        "contrast": lambda: stats(librosa.feature.spectral_contrast(S=S, sr=a.sr)),
        "flux": lambda: stats(np.sqrt((np.diff(S, axis=1).clip(0) ** 2).sum(0))),
    }

    def skew_kurt():
        mu = (fr[:, None] * Pn).sum(0)
        var = (((fr[:, None] - mu) ** 2) * Pn).sum(0)
        sk = (((fr[:, None] - mu) ** 3) * Pn).sum(0) / (var ** 1.5 + 1e-12)
        ku = (((fr[:, None] - mu) ** 4) * Pn).sum(0) / (var ** 2 + 1e-12)
        return {**stats(sk, "skew_"), **stats(ku, "kurt_")}
    out["skewness_kurtosis"] = skew_kurt
    out["entropy"] = lambda: stats(-(Pn * np.log2(Pn + 1e-12)).sum(0) / np.log2(len(fr)))

    def slope_decrease():
        slope = np.polyfit(fr, S, 1)[0]
        k = np.arange(1, S.shape[0])
        dec = ((S[1:] - S[:1]) / k[:, None]).sum(0) / (S[1:].sum(0) + 1e-12)
        return {**stats(slope, "slope_"), **stats(dec, "decrease_")}
    out["slope_decrease"] = slope_decrease
    out["crest"] = lambda: stats(S.max(0) / (S.mean(0) + 1e-12))
    out["irregularity"] = lambda: stats((np.diff(S, axis=0) ** 2).sum(0) / ((S ** 2).sum(0) + 1e-12))

    def band_ratios():
        bands = {"sub": (20, 60), "bass": (60, 250), "lowmid": (250, 500), "mid": (500, 2000),
                 "highmid": (2000, 4000), "presence": (4000, 6000), "air": (6000, 11025)}
        tot = P.sum(0) + 1e-12
        return {k: float((P[(fr >= lo) & (fr < hi)].sum(0) / tot).mean()) for k, (lo, hi) in bands.items()}
    out["band_energy_ratios"] = band_ratios
    out["hfc"] = lambda: stats((np.arange(len(fr))[:, None] * P).sum(0))
    return out


# ---------------------------------------------------------------- 2. timbre

def f_timbre(a):
    import essentia.standard as es
    out = {}

    def mfcc():
        d1, d2 = librosa.feature.delta(a.mfcc), librosa.feature.delta(a.mfcc, order=2)
        return {**stats(a.mfcc, "mfcc_"), **stats(d1, "d1_"), **stats(d2, "d2_")}
    out["mfcc"] = mfcc
    out["logmel"] = lambda: stats(a.mel_db)

    def gfcc_bark():
        g = es.GFCC(sampleRate=44100, inputSize=1025, numberCoefficients=13)
        bark = es.BarkBands(sampleRate=44100)
        G = np.array([g(s)[1] for s in a.ess_spectra]).T
        B = np.array([bark(s) for s in a.ess_spectra]).T
        return {**stats(G, "gfcc_"), **stats(db_(B), "bark_")}
    out["gfcc_bark"] = gfcc_bark
    out["cqt"] = lambda: stats(librosa.amplitude_to_db(
        np.abs(librosa.cqt(a.y, sr=a.sr, hop_length=config.HOP, n_bins=84)), ref=np.max))
    out["vqt"] = lambda: stats(librosa.amplitude_to_db(
        np.abs(librosa.vqt(a.y, sr=a.sr, hop_length=config.HOP, n_bins=84, gamma=None)), ref=np.max))

    def lpc_frames():
        if not hasattr(a, "lpc_frames"):
            frames = librosa.util.frame(a.y, frame_length=1024, hop_length=1024).T
            frames = frames[np.abs(frames).max(1) > 1e-3]
            a.lpc_frames = np.array([librosa.lpc(f * np.hanning(len(f)), order=12) for f in frames]) \
                if len(frames) else np.ones((1, 13))
        return a.lpc_frames
    out["lpc"] = lambda: stats(lpc_frames()[:, 1:].T)

    def odd_even():
        hp, oer = es.HarmonicPeaks(), es.OddToEvenHarmonicEnergyRatio()
        vals = [oer(*hp(f, m, p)) for (f, m), (p, c) in zip(a.ess_peaks, a.ess_pitch) if p > 0 and c > 0.5 and len(f)]
        return stats(np.array(vals) if vals else np.zeros(1))
    out["odd_even_ratio"] = odd_even

    def formants():
        L = lpc_frames()
        F = []
        for coefs in L[:: max(1, len(L) // 400)]:
            r = np.roots(coefs)
            f = np.sort(np.angle(r[np.imag(r) > 0]) * a.sr / (2 * np.pi))
            f = f[f > 90][:3]
            if len(f) == 3:
                F.append(f)
        return stats(np.array(F).T if F else np.zeros((3, 1)))
    out["formants"] = formants
    return out


# ---------------------------------------------------------------- 3. energy & dynamics

def true_peak_db(stereo, factor=4, block=44100 * 10):
    """ITU-style true peak via 4x polyphase oversampling, in 10 s blocks (essentia's detector took ~25 s)."""
    peak = 0.0
    for ch in stereo:
        for i in range(0, len(ch), block):
            seg = ch[max(0, i - 64):i + block + 64]
            peak = max(peak, float(np.abs(scipy.signal.resample_poly(seg, factor, 1)).max()))
    return 20 * np.log10(peak + 1e-12)


def f_energy(a):
    import essentia.standard as es
    out = {"rms": lambda: stats(a.rms),
           "zcr": lambda: stats(librosa.feature.zero_crossing_rate(a.y, frame_length=config.N_FFT, hop_length=config.HOP))}

    def lufs():
        mom, short, integrated, lra = es.LoudnessEBUR128(sampleRate=44100)(a.stereo44.T.astype(np.float32))
        a.lufs = float(integrated)
        return {"integrated_lufs": a.lufs, "loudness_range_lu": float(lra), **stats(np.asarray(short), "short_term_")}
    out["loudness_ebu_r128"] = lufs

    def dyn():
        rms_db = db_(a.rms ** 2)
        peak, rms = np.abs(a.mono44).max(), np.sqrt((a.mono44 ** 2).mean())
        return {"dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
                "crest_factor_db": float(20 * np.log10(peak / (rms + 1e-12)))}
    out["dynamic_range_crest"] = dyn

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
    out["attack_decay_centroid"] = envelope_shape

    def true_peak():
        a.true_peak_db = true_peak_db(a.stereo44)
        return {"true_peak_dbtp": a.true_peak_db, "clipping_ratio": float((np.abs(a.stereo44) > 0.999).mean())}
    out["true_peak_clipping"] = true_peak

    def contour():
        sm = scipy.signal.savgol_filter(a.rms, 101 if len(a.rms) > 101 else 5, 2)
        bins = np.array([b.mean() for b in np.array_split(sm, 20)])
        bins = bins / (bins.max() + 1e-12)
        return {"contour20": bins, "slope": float(np.polyfit(np.arange(20), bins, 1)[0]),
                "argmax_pos": float(bins.argmax() / 19), "n_buildups": int((np.diff(np.sign(np.diff(bins))) < 0).sum())}
    out["energy_contour"] = contour

    def plr():
        if not hasattr(a, "lufs"):
            lufs()
        if not hasattr(a, "true_peak_db"):
            true_peak()
        return {"plr_db": a.true_peak_db - a.lufs}
    out["plr"] = plr
    return out


# ---------------------------------------------------------------- 4. harmony & pitch

def _harmonic_frames(a, fn):
    import essentia.standard as es

    hp = es.HarmonicPeaks()
    out = []
    for (f, m), (p, c) in zip(a.ess_peaks, a.ess_pitch):
        if p > 0 and c > 0.5 and len(f):
            hf, hm = hp(f, m, p)
            try:
                out.append(fn(hf, hm))
            except Exception:
                pass
    return out


def f_harmony(a, stage):
    import essentia.standard as es
    out = {}

    def chroma():
        cqt = librosa.feature.chroma_cqt(y=a.yh, sr=a.sr, hop_length=config.HOP)
        cens = librosa.feature.chroma_cens(y=a.yh, sr=a.sr, hop_length=config.HOP)
        return {**stats(a.chroma, "stft_"), **stats(cqt, "cqt_"), **stats(cens, "cens_")}
    out["chroma"] = chroma
    def hpcp():
        h = es.HPCP(sampleRate=44100, size=36)
        return stats(np.array([h(f, m) for f, m in a.ess_peaks]).T)
    out["hpcp"] = hpcp
    out["tonnetz"] = lambda: stats(librosa.feature.tonnetz(chroma=a.chroma))

    def key():
        if not hasattr(a, "key_idx"):
            k, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=44100)(a.mono44)
            a.key_idx, a.key_major, a.key_strength = KEYS.index(FLATS.get(k, k)), scale == "major", float(strength)
        return {"key_onehot": np.eye(12)[a.key_idx], "major": float(a.key_major), "strength": a.key_strength}
    out["key_mode"] = key

    def chords():
        if not hasattr(a, "chords"):
            tx = es.TonalExtractor()
            d = dict(zip(tx.outputNames(), tx(a.mono44)))
            a.chords = list(d["chords_progression"])
            a.chord_stats = {"changes_rate": float(d["chords_changes_rate"]), "number_rate": float(d["chords_number_rate"]),
                             "histogram": np.asarray(d["chords_histogram"])}
        return a.chord_stats
    out["chords"] = chords

    def f0():
        p = np.array([p for p, c in a.ess_pitch])
        c = np.array([c for p, c in a.ess_pitch])
        v = p[(c > 0.5) & (p > 0)]
        return {"voiced_ratio": float(len(v) / max(len(p), 1)), **stats(librosa.hz_to_midi(v) if len(v) else np.zeros(1), "midi_")}
    out["f0"] = f0

    def melody():
        if not hasattr(a, "melody"):
            eq = es.EqualLoudness(sampleRate=44100)(a.mono44)
            a.melody = np.asarray(es.PredominantPitchMelodia(frameSize=2048, hopSize=256, sampleRate=44100)(eq)[0])
        v = a.melody[a.melody > 0]
        return {"voiced_ratio": float(len(v) / len(a.melody)), **stats(librosa.hz_to_midi(v) if len(v) else np.zeros(1))}
    out["predominant_melody"] = melody
    out["tuning"] = lambda: {"tuning_cents": float(librosa.estimate_tuning(S=a.S, sr=a.sr) * 100)}

    def inharm_hnr():
        inh = es.Inharmonicity()
        vals = _harmonic_frames(a, lambda hf, hm: inh(hf, hm) if hf[0] > 0 else None)
        vals = [v for v in vals if v is not None]
        hnr = db_((a.yh ** 2).mean()) - db_(((a.y - a.yh - a.yp) ** 2).mean() + 1e-12)
        return {"inharmonicity": stats(vals or [0])["mean"], "hnr_db_approx": float(hnr)}
    out["inharmonicity_hnr"] = inharm_hnr

    def tristimulus():
        T = _harmonic_frames(a, es.Tristimulus())
        return stats(np.array(T).T if T else np.zeros((3, 1)))
    out["tristimulus"] = tristimulus
    def dissonance():
        d = es.Dissonance()
        return stats([d(f, m) for f, m in a.ess_peaks if len(f) > 1] or [0])
    out["dissonance"] = dissonance

    def key_norm_chroma():
        key()
        return {"chroma_rel_mean": np.roll(a.chroma.mean(1), -a.key_idx)}
    out["key_normalized_chroma"] = key_norm_chroma

    def key_clarity():
        key()
        prof = a.chroma.mean(1)
        return {"chroma_entropy": float(scipy.stats.entropy(prof / prof.sum()))}
    out["key_clarity"] = key_clarity

    def harmonic_rhythm():
        chords()
        ch = np.array(a.chords)
        changes = np.flatnonzero(ch[1:] != ch[:-1])
        seg_len = np.diff(np.r_[0, changes + 1, len(ch)]) * a.dur / max(len(ch), 1)
        return {"chord_changes_per_min": len(changes) / a.dur * 60, "mean_chord_dur_s": float(seg_len.mean())}
    out["harmonic_rhythm"] = harmonic_rhythm

    def chord_vocab():
        chords()
        ch = np.array(a.chords)
        seq = ch[np.r_[True, ch[1:] != ch[:-1]]]
        trans = list(zip(seq[:-1], seq[1:]))
        cnt = np.unique(np.array(trans, dtype=str), axis=0, return_counts=True)[1] if trans else np.ones(1)
        return {"minor_ratio": float(np.mean([c.endswith("m") for c in seq])), "n_unique_chords": len(set(seq)),
                "transition_entropy": float(scipy.stats.entropy(cnt))}
    out["chord_vocabulary"] = chord_vocab

    def melodic():
        melody()
        v = a.melody[a.melody > 0]
        midi = librosa.hz_to_midi(v) if len(v) else np.zeros(1)
        steps = np.abs(np.diff(np.round(midi)))
        steps = steps[(steps > 0) & (steps < 13)]
        return {"range_semitones": float(np.percentile(midi, 95) - np.percentile(midi, 5)),
                "interval_hist": np.bincount(steps.astype(int), minlength=13)[1:13] / max(len(steps), 1)}
    out["pitch_range_intervals"] = melodic

    def polyphony():
        from basic_pitch.inference import predict

        notes = predict(a.path, stage.basic_pitch)[2]
        t = np.arange(0, a.dur, 0.05)
        active = np.zeros_like(t)
        for s, e, *_ in notes:
            active[(t >= s) & (t < e)] += 1
        return {"notes_per_s": len(notes) / a.dur, **stats(active[active > 0] if active.any() else [0], "poly_")}
    out["polyphony"] = polyphony
    return out


# ---------------------------------------------------------------- 5. rhythm

def beat_pattern(a, env, subdiv):
    """Mean onset envelope at `subdiv` positions per bar (4 beats) or per beat (subdiv<=4)."""
    bt = a.beats
    per = subdiv if subdiv <= 4 else subdiv // 4
    pts = [[env[min(int(b0 + (b1 - b0) * i / per), len(env) - 1)] for i in range(per)] for b0, b1 in zip(bt[:-1], bt[1:])]
    P = np.array(pts) if pts else np.zeros((1, per))
    if subdiv > 4:
        n = len(P) // 4 * 4
        P = P[:n].reshape(-1, subdiv) if n else np.zeros((1, subdiv))
    m = P.mean(0)
    return m / (m.max() + 1e-12)


def f_rhythm(a, stage):
    import essentia.standard as es
    out = {}

    def onset_times():
        if not hasattr(a, "onset_times"):
            a.onset_times = librosa.onset.onset_detect(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP, units="time")
        return a.onset_times

    def onsets():
        ioi = np.diff(onset_times())
        return {**stats(a.onset_env, "env_"), **stats(ioi if len(ioi) else [0], "ioi_"), "onset_rate": len(onset_times()) / a.dur}
    out["onsets"] = onsets

    def tempo():
        if not hasattr(a, "bpm_intervals"):
            bpm, beats, conf, est, a.bpm_intervals = es.RhythmExtractor2013(method="multifeature")(a.mono44)
            g, local, _ = stage.tempocnn(a.resampled(11025))
            a.tempo_stats = {"bpm_essentia": float(bpm), "confidence": float(conf), "bpm_tempocnn": float(g),
                             "bpm_librosa": a.tempo, "local_tempo_std": float(np.std(local))}
        return a.tempo_stats
    out["tempo"] = tempo

    def tempogram():
        if not hasattr(a, "tempogram"):
            a.tempogram = librosa.feature.tempogram(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP)
        return stats(a.tempogram[1:100])
    out["tempogram"] = tempogram

    def beat_hist():
        tempo()
        bh = es.BpmHistogramDescriptors()
        d = {n: np.asarray(v) for n, v in zip(bh.outputNames(), bh(a.bpm_intervals))}
        return {**{k: float(v) for k, v in d.items() if k != "histogram" and v.ndim == 0},
                "pattern16": beat_pattern(a, a.onset_env, 16)}
    out["beat_histogram_pattern"] = beat_hist

    def pulse_sync():
        tempogram()
        ac = a.tempogram.mean(1)
        pat = beat_pattern(a, a.onset_env, 4)
        return {"pulse_clarity": float(ac[1:].max() / (ac[1:].mean() + 1e-12)),
                "syncopation": float((pat[1] + pat[3]) / (pat.sum() + 1e-12))}
    out["pulse_syncopation"] = pulse_sync

    def danceability():
        d, dfa = es.Danceability(sampleRate=44100)(a.mono44)
        return {"danceability": float(d), "dfa_mean": float(np.mean(dfa))}
    out["danceability"] = danceability

    def stability():
        ibi = np.diff(a.beat_times)
        return {"ibi_cv": float(ibi.std() / (ibi.mean() + 1e-12)) if len(ibi) else 0.0,
                "tempo_drift_bpm_per_min": float(np.polyfit(a.beat_times[1:], 60 / ibi, 1)[0] * 60) if len(ibi) > 2 else 0.0}
    out["tempo_stability"] = stability

    def swing():
        bt, on = a.beat_times, onset_times()
        ratios, dev = [], []
        for b0, b1 in zip(bt[:-1], bt[1:]):
            L = b1 - b0
            inside = on[(on > b0 + 0.3 * L) & (on < b0 + 0.8 * L)]
            if len(inside):
                pos = (inside[0] - b0) / L
                ratios.append(pos / (1 - pos))
            rel = (on[(on >= b0) & (on < b1)] - b0) / L * 4
            dev += list(rel - np.round(rel))
        return {"swing_ratio": stats(ratios or [1])["mean"], "microtiming_std": float(np.std(dev)) if dev else 0.0}
    out["swing_microtiming"] = swing

    def band_patterns():
        mel = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=64)
        mf = librosa.mel_frequencies(64, fmax=a.sr / 2)
        return {name: beat_pattern(a, librosa.onset.onset_strength(S=librosa.power_to_db(mel[(mf >= lo) & (mf < hi)]), sr=a.sr), 16)
                for name, (lo, hi) in {"low": (0, 150), "mid": (150, 2000), "high": (4000, 11025)}.items()}
    out["band_onset_patterns"] = band_patterns

    def fourier_cyclic():
        F = np.abs(librosa.feature.fourier_tempogram(onset_envelope=a.onset_env, sr=a.sr, hop_length=config.HOP))
        bpms = librosa.fourier_tempo_frequencies(sr=a.sr, hop_length=config.HOP)
        prof = F.mean(1)
        ok = bpms > 30
        cyclic = np.bincount((np.log2(bpms[ok] / 60) % 1 * 24).astype(int), weights=prof[ok], minlength=24)
        return {"fourier_profile_32": np.interp(np.geomspace(30, 480, 32), bpms, prof), "cyclic24": cyclic / (cyclic.sum() + 1e-12)}
    out["fourier_cyclic_tempogram"] = fourier_cyclic

    def density():
        # onsets per 4-beat group (librosa beats; madmom downbeats were dropped for speed)
        bars = a.beat_times[::4]
        return stats(np.histogram(onset_times(), bins=bars)[0] if len(bars) > 1 else np.zeros(1))
    out["density_per_bar"] = density
    return out


# ---------------------------------------------------------------- 6. structure

def f_structure(a):
    from sklearn.cluster import AgglomerativeClustering

    out = {}

    def bsync():
        if not hasattr(a, "bsync"):
            F = np.vstack([librosa.util.normalize(a.mfcc[1:13], axis=1), a.chroma])
            a.bsync = librosa.util.sync(F, a.beats, aggregate=np.median)
        return a.bsync

    def recurrence():
        if not hasattr(a, "R"):
            a.R = librosa.segment.recurrence_matrix(bsync(), mode="affinity", sym=True, width=4)
        return a.R

    def ssm():
        prof = librosa.segment.recurrence_to_lag(recurrence(), pad=False).mean(1)
        return {"lag_profile_64": np.interp(np.linspace(0, len(prof) - 1, 64), np.arange(len(prof)), prof),
                "density": float((recurrence() > 0).mean())}
    out["self_similarity"] = ssm

    def novelty():
        S = np.corrcoef(bsync().T)
        k = 16
        g = np.outer(scipy.signal.windows.gaussian(2 * k, k / 2), scipy.signal.windows.gaussian(2 * k, k / 2))
        kern = g * np.kron([[1, -1], [-1, 1]], np.ones((k, k)))
        Sp = np.pad(np.nan_to_num(S), k, mode="constant")
        nov = np.array([(Sp[i:i + 2 * k, i:i + 2 * k] * kern).sum() for i in range(len(S))])
        return {**stats(nov), "n_peaks_per_min": len(scipy.signal.find_peaks(nov, distance=8, prominence=np.std(nov))[0]) / a.dur * 60}
    out["novelty"] = novelty

    def segments():
        if not hasattr(a, "seg_times"):
            B = bsync()
            k = int(np.clip(a.dur / 20, 2, 12))
            bounds = librosa.segment.agglomerative(B, k)
            feats = np.array([B[:, s:e].mean(1) for s, e in zip(bounds, list(bounds[1:]) + [B.shape[1]])])
            a.seg_labels = AgglomerativeClustering(n_clusters=max(2, k // 2)).fit_predict(feats)
            bt = np.r_[0, a.beat_times]
            a.seg_times = bt[np.clip(bounds, 0, len(bt) - 1)]
        return a.seg_times

    def sections():
        L = np.diff(np.r_[segments(), a.dur])
        return {"n_labels": len(set(a.seg_labels)), "mean_len_s": float(L.mean()), "std_len_s": float(L.std())}
    out["sections"] = sections

    def intro_outro():
        sm = scipy.signal.medfilt(a.rms, 43)
        above = np.flatnonzero(sm > 0.5 * np.median(sm))
        return {"duration_s": a.dur, "intro_s": above[0] / a.fps if len(above) else 0.0,
                "outro_s": (len(sm) - above[-1]) / a.fps if len(above) else 0.0}
    out["duration_intro_outro"] = intro_outro
    out["silence_ratio"] = lambda: {"silence_ratio": float((db_(a.rms ** 2) < -60).mean())}

    def repetitiveness():
        R = recurrence().copy()
        n = len(R)
        for d in range(-8, 9):
            idx = np.arange(max(0, -d), min(n, n - d))
            R[idx, idx + d] = 0
        return {"repeated_fraction": float((R.max(1) > 0).mean())}
    out["repetitiveness"] = repetitiveness

    def climax():
        ends = np.r_[segments()[1:], a.dur]
        energy = [a.rms[int(s * a.fps):max(int(e * a.fps), int(s * a.fps) + 1)].mean() for s, e in zip(a.seg_times, ends)]
        score = np.array(energy) * np.bincount(a.seg_labels)[a.seg_labels]
        i = int(score.argmax())
        return {"climax_pos": float(a.seg_times[i] / a.dur), "climax_energy_ratio": float(energy[i] / (np.mean(energy) + 1e-12))}
    out["climax"] = climax

    def loops():
        X = bsync() - bsync().mean(1, keepdims=True)
        res = {}
        for bars in (1, 2, 4, 8):
            lag = bars * 4
            c = [np.corrcoef(X[:, i], X[:, i + lag])[0, 1] for i in range(max(X.shape[1] - lag, 0))]
            res[f"loop_corr_{bars}bar"] = float(np.nanmean(c)) if c else 0.0
        return res
    out["loops"] = loops
    return out


# ---------------------------------------------------------------- 10. summary statistics (on MFCCs)

def f_summary(a):
    M = a.mfcc
    return {
        "min_max": lambda: {"min": M.min(1), "max": M.max(1)},
        "median": lambda: {"p50": np.percentile(M, 50, axis=1)},
        "skew_kurt": lambda: {"skew": scipy.stats.skew(M, axis=1), "kurt": scipy.stats.kurtosis(M, axis=1)},
        "delta_mean_var": lambda: {"dmean": np.diff(M).mean(1), "dvar": np.diff(M).var(1)},
        "beat_sync_std": lambda: {"std": librosa.util.sync(M, a.beats).std(1)},
        "covariance": lambda: {"cov_upper": np.cov(M)[np.triu_indices(len(M), 1)]},
    }


# ---------------------------------------------------------------- 11. psychoacoustic (cheap proxies only)

def f_psycho(a):
    def fluct():
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=24) ** 0.3
        F = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1))
        mf = np.fft.rfftfreq(env.shape[1], 1 / a.fps)
        return {"fluct_4hz_ratio": float(F[:, (mf > 1) & (mf < 8)].sum() / (F[:, mf > 0.5].sum() + 1e-12))}

    def warmth():
        fr, P = a.freqs, a.P.mean(1)
        tot = P.sum() + 1e-12
        return {"warmth_100_500": float(P[(fr >= 100) & (fr < 500)].sum() / tot),
                "boominess_60_200": float(P[(fr >= 60) & (fr < 200)].sum() / tot)}
    return {"fluctuation_strength": fluct, "warmth_boominess": warmth}


# ---------------------------------------------------------------- 12. modulation & texture

def f_texture(a, stage):
    def modspec():
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=32) ** 0.3
        F = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1))
        mf = np.fft.rfftfreq(env.shape[1], 1 / a.fps)
        edges = [0.5, 1, 2, 4, 8, 16, 21]
        return {"mod_power": np.array([[F[b, (mf >= lo) & (mf < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                                       for b in range(32)]).ravel()}

    def scattering():
        N = 2 ** 16
        blocks = [x[i:i + N] for x in a.excerpts(a.sr) for i in range(0, len(x) - N + 1, N)]  # excerpts: fixed cost
        if not blocks:
            blocks = [np.pad(a.y[:N], (0, max(0, N - len(a.y))))]
        import torch

        with torch.no_grad():
            Sx = stage.scattering(torch.tensor(np.stack(blocks), dtype=torch.float32))
        return {"scattering_logmean": torch.log1p(Sx).mean((0, 2)).numpy()}

    def strf():
        res = []
        for rate in (2, 4, 8, 16):
            for scale in (0.25, 0.5, 1, 2):
                t = np.arange(-16, 16) / a.fps
                f = np.arange(-16, 16) / 16
                g = np.outer(np.exp(-(f * scale) ** 2) * np.cos(2 * np.pi * scale * f),
                             np.exp(-(t * rate) ** 2) * np.cos(2 * np.pi * rate * t))
                res.append(np.abs(scipy.signal.fftconvolve(a.mel_db, g, mode="same")).mean())
        return {"strf16": np.array(res)}

    def mcdermott():
        env = librosa.feature.melspectrogram(S=a.P, sr=a.sr, n_mels=32) ** 0.3
        mu = env.mean(1)
        marg = np.stack([mu, env.var(1) / (mu ** 2 + 1e-12), scipy.stats.skew(env, axis=1), scipy.stats.kurtosis(env, axis=1)])
        return {"marginals": marg.ravel(), "band_corr": np.nan_to_num(np.corrcoef(env))[np.triu_indices(32, 1)]}

    def stationarity():
        flux = np.sqrt((np.diff(a.S, axis=1).clip(0) ** 2).sum(0))
        step = int(5 * a.fps)
        blocks = [a.mel_db[:, i:i + step].mean(1) for i in range(0, a.mel_db.shape[1], step)]
        return {"flux_var": float(flux.var()),
                "stationarity": float(np.mean(np.linalg.norm(np.diff(blocks, axis=0), axis=1))) if len(blocks) > 1 else 0.0}

    return {"modulation_spectrum": modspec, "scattering": scattering, "strf_gabor": strf,
            "texture_stats": mcdermott, "flux_var_stationarity": stationarity}


# ---------------------------------------------------------------- 13. stereo

def f_stereo(a):
    L, R = a.stereo44

    def width():
        M, S = (L + R) / 2, (L - R) / 2
        return {"side_mid_ratio_db": float(db_((S ** 2).mean()) - db_((M ** 2).mean()))}

    def corr():
        c = np.corrcoef(L, R)[0, 1]
        return {"corr": float(c) if np.isfinite(c) else 1.0}

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

    def ild_itd():
        seg = slice(0, min(len(L), 44100 * 20))
        c = scipy.signal.correlate(L[seg], R[seg], mode="full", method="fft")
        lags = np.arange(-len(L[seg]) + 1, len(L[seg]))
        win = np.abs(lags) <= 44
        return {"ild_db": float(db_((L ** 2).mean()) - db_((R ** 2).mean())), "itd_ms": float(lags[win][c[win].argmax()] / 44.1)}

    def reverb():
        e = db_(librosa.feature.rms(y=a.mono44, frame_length=1024, hop_length=256)[0] ** 2)
        fps = 44100 / 256
        slopes = []
        for p in scipy.signal.find_peaks(e, prominence=10, distance=int(0.3 * fps))[0]:
            seg = e[p:p + int(0.5 * fps)]
            if len(seg) > 10:
                slopes.append(np.polyfit(np.arange(len(seg)) / fps, seg, 1)[0])
        s = np.median(slopes) if slopes else 0.0
        return {"rt60_est_s": float(min(-60 / s, 10.0)) if s < 0 else 10.0, "events_per_min": len(slopes) / a.dur * 60}

    return {"width": width, "correlation": corr, "panning": panning, "ild_itd": ild_itd, "reverb": reverb}


# ---------------------------------------------------------------- 14. production

def f_production(a):
    cache = {}

    def lt_spectrum():
        if "P" not in cache:
            cache["P"] = np.mean([s ** 2 for s in a.ess_spectra], axis=0)
        P = cache["P"]
        return P, np.linspace(0, 22050, len(P))

    def balance():
        P, fr = lt_spectrum()
        centers = 1000 * 2 ** (np.arange(-15, 14) / 3)
        lev = np.array([db_(P[(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6))].mean()) for c in centers])
        fit = np.polyfit(np.log2(centers), lev, 1)
        return {"third_octave_rel": lev - np.polyval(fit, np.log2(centers)), "tilt_db_per_oct": float(fit[0])}

    def compression():
        r = a.rms
        return {"rms_p95_p50_db": float(db_(np.percentile(r, 95) ** 2) - db_(np.percentile(r, 50) ** 2))}

    def noise():
        P, fr = lt_spectrum()
        hum = {}
        for f0 in (50, 60):
            peak = np.mean([P[np.argmin(np.abs(fr - k * f0))] for k in (1, 2, 3)])
            around = np.mean([P[(np.abs(fr - k * f0) > 5) & (np.abs(fr - k * f0) < 20)].mean() for k in (1, 2, 3)])
            hum[f"hum_{f0}hz_db"] = float(db_(peak) - db_(around))
        quiet = np.array(a.ess_spectra)[np.argsort([s.sum() for s in a.ess_spectra])[: max(1, len(a.ess_spectra) // 10)]]
        return {"noise_floor_db": float(np.percentile(db_(a.rms ** 2), 5)), "hiss_db": float(db_((quiet[:, fr > 12000] ** 2).mean())), **hum}

    def codec():
        P, fr = lt_spectrum()
        Pd = db_(P)
        above = np.flatnonzero(Pd > Pd.max() - 90)
        return {"effective_bandwidth_hz": float(fr[above[-1]]) if len(above) else 0.0}

    def distortion():
        x = a.mono44
        return {"waveform_kurtosis": float(scipy.stats.kurtosis(x)),
                "near_full_scale_ratio": float((np.abs(x) > 0.95 * np.abs(x).max()).mean())}

    return {"spectral_balance": balance, "compression": compression, "noise_hiss_hum": noise,
            "effective_bandwidth": codec, "saturation": distortion}
