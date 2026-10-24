"""Source-separation features (catalogue sections 7, 15, bass line from 4) on the fixed excerpts.

demucs separates the excerpts (see audio.EXCERPT_*), so cost no longer grows with track
length. Vocal pitch uses torchcrepe `tiny`; Whisper transcribes the vocal-active parts of
the excerpts (the transcript is a lyrics fallback, stored in the `transcripts` table).
"""

import gzip

import librosa
import numpy as np
import scipy.signal

from .. import config
from .audio import Audio, db_
from .common import run_features, stats

EXTRACTOR = "stems"
VERSION = 1

MIN_VOCAL_S = 5.0          # less detected vocal than this counts as instrumental (stem bleed)
WHISPER_MODEL = "openai/whisper-tiny"


class Stage:
    name = "stems"
    outputs = [(EXTRACTOR, VERSION)]

    def load(self):
        from demucs.pretrained import get_model
        from transformers import pipeline

        self.demucs = get_model("htdemucs").eval()
        self.asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device="cpu")

    def separate(self, a: Audio) -> dict:
        import torch
        from demucs.apply import apply_model

        parts = {s: [] for s in self.demucs.sources}
        for x in a.stereo_excerpts():
            wav = torch.tensor(x, dtype=torch.float32)
            ref = wav.mean(0)
            mu, sd = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                # overlap 0.1 (default 0.25): 21% faster, vocal stem within 27.7 dB SDR of the default
                out = apply_model(self.demucs, ((wav - mu) / sd)[None], device="cpu", split=True, overlap=0.1, progress=False)[0]
            out = out * sd + mu
            for i, s in enumerate(self.demucs.sources):
                parts[s].append(out[i].numpy())
        return {s: np.concatenate(v, axis=1) for s, v in parts.items()}  # (2, N) each

    def process(self, a: Audio) -> dict:
        v = Vocals(a, self.separate(a), self.asr)
        groups = {"separation": f_separation(v), "bass": {"bass_line": v.bass_line}, "vocal": f_vocal(v)}
        names, vec = run_features(groups, EXTRACTOR)
        return {EXTRACTOR: (names, vec), "_transcript": v.transcript()}


class Vocals:
    """Stems of the excerpts plus the vocal analyses several features share."""

    def __init__(self, a: Audio, stems: dict, asr):
        self.a, self.stems, self.asr = a, stems, asr
        self.mix = np.concatenate([x.mean(0) for x in a.stereo_excerpts()]).astype(np.float32)
        self.dur = len(self.mix) / 44100
        self.voc = stems["vocals"].mean(0)
        self._cache = {}

    def once(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def active(self):
        """Vocal-active frames (hop 2048 @ 44.1 kHz): vocal stem within 15 dB of the mix."""
        def f():
            r = librosa.feature.rms(y=self.voc, frame_length=4096, hop_length=2048)[0]
            mix_r = librosa.feature.rms(y=self.mix, frame_length=4096, hop_length=2048)[0]
            return db_(r ** 2) > db_(mix_r ** 2) - 15
        return self.once("active", f)

    @property
    def v16(self):
        return self.once("v16", lambda: librosa.resample(self.voc, orig_sr=44100, target_sr=16000).astype(np.float32))

    def crepe(self):
        def f():
            import torch
            import torchcrepe

            with torch.no_grad():
                p, per = torchcrepe.predict(torch.tensor(self.v16)[None], 16000, hop_length=160, fmin=60, fmax=1000,
                                            model="tiny", return_periodicity=True, batch_size=1024, device="cpu")
            p, per = p[0].numpy(), per[0].numpy()
            r = librosa.feature.rms(y=self.v16, frame_length=1024, hop_length=160)[0][: len(p)]
            act = np.zeros(len(p), bool)
            act[: len(r)] = db_(r ** 2) > db_(np.max(r) ** 2) - 35
            return p, per, act, act & (per > 0.5)
        return self.once("crepe", f)

    def vocal_segments_16k(self, pad_s=0.5, min_s=1.0, gap_s=0.3):
        fps = 44100 / 2048
        k = int(pad_s * fps)
        act = np.convolve(self.active, np.ones(2 * k + 1), "same") > 0
        edges = np.flatnonzero(np.diff(np.r_[0, act.astype(int), 0]))
        gap = np.zeros(int(gap_s * 16000), np.float32)
        parts = [np.r_[self.v16[int(s / fps * 16000):int(e / fps * 16000)], gap]
                 for s, e in zip(edges[::2], edges[1::2]) if (e - s) / fps >= min_s]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def transcript(self) -> str:
        def f():
            y = self.vocal_segments_16k()
            if len(y) < MIN_VOCAL_S * 16000:
                return ""
            pieces = [{"raw": y[i:i + 16000 * 30], "sampling_rate": 16000} for i in range(0, len(y), 16000 * 30)]
            # 128 tokens per 30 s caps repetition loops; highly compressible pieces are hallucinations
            texts = [r["text"].strip() for r in self.asr(pieces, batch_size=1,
                                                         generate_kwargs={"task": "transcribe", "max_new_tokens": 128})]
            return " ".join(t for t in texts if t and len(t) / max(len(gzip.compress(t.encode())), 1) < 2.4)
        return self.once("transcript", f)

    def bass_line(self):
        yb = librosa.resample(self.stems["bass"].mean(0), orig_sr=44100, target_sr=8000)
        f0, voiced, _ = librosa.pyin(yb, fmin=30, fmax=300, sr=8000, frame_length=1024, hop_length=256)
        f = f0[voiced & ~np.isnan(f0)]
        midi = np.round(librosa.hz_to_midi(f)) if len(f) else np.zeros(1)
        motion = np.abs(np.diff(midi))
        motion = motion[(motion > 0) & (motion < 13)]
        return {"pc_hist": np.bincount((midi % 12).astype(int), minlength=12) / len(midi),
                "root_motion_hist": np.bincount(motion.astype(int), minlength=13)[1:13] / max(len(motion), 1)}


def f_separation(v: Vocals):
    def energies():
        e = {k: float((x ** 2).mean()) for k, x in v.stems.items()}
        tot = sum(e.values()) + 1e-12
        return {"rms": {k: np.sqrt(x) for k, x in e.items()}, "share": {k: x / tot for k, x in e.items()}}

    def vocal_activity():
        thirds = np.array_split(v.active, 3)
        return {"vocal_ratio": float(v.active.mean()), "vocal_ratio_thirds": np.array([t.mean() for t in thirds])}

    def speechiness():
        p, per, act, voiced = v.crepe()
        if not act.any():
            return {"speechiness": 0.0}
        cents = 1200 * np.log2(np.maximum(p, 1) / 440)
        jumpy = np.abs(np.diff(cents, prepend=cents[0])) > 50
        return {"speechiness": float(((per < 0.5) | jumpy)[act].mean())}

    def per_stem():
        res = {}
        for k, x in v.stems.items():
            m = librosa.resample(x.mean(0), orig_sr=44100, target_sr=config.SR)
            S = np.abs(librosa.stft(m, n_fft=config.N_FFT, hop_length=config.HOP))
            mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=config.SR), ref=np.max)
            res[k] = np.concatenate([[librosa.feature.spectral_centroid(S=S, sr=config.SR).mean()],
                                     librosa.feature.mfcc(S=mel, n_mfcc=13).mean(1),
                                     [len(librosa.onset.onset_detect(y=m, sr=config.SR)) / v.dur]])
        return res

    return {"stem_energy": energies, "vocal_activity": vocal_activity, "speechiness": speechiness, "per_stem": per_stem}


def f_vocal(v: Vocals):
    def voiced_pitch():
        p, per, act, voiced = v.crepe()
        return p[voiced]

    def pitch_range():
        x = voiced_pitch()
        midi = librosa.hz_to_midi(x) if len(x) else np.zeros(1)
        return {"median_midi": float(np.median(midi)), "range_semitones": float(np.percentile(midi, 95) - np.percentile(midi, 5)),
                "voiced_fraction": len(x) / max(len(v.crepe()[0]), 1)}

    def vibrato():
        p, per, act, voiced = v.crepe()
        cents = 1200 * np.log2(np.maximum(p, 1) / 440)
        runs = np.split(np.arange(len(cents)), np.flatnonzero(np.diff(voiced.astype(int)) != 0) + 1)
        rates, ext = [], []
        for r in runs:
            if len(r) >= 50 and voiced[r[0]]:
                c = scipy.signal.detrend(cents[r])
                F = np.abs(np.fft.rfft(c * np.hanning(len(c))))
                f = np.fft.rfftfreq(len(c), 0.01)
                band = (f >= 4) & (f <= 8)
                rates.append(f[band][F[band].argmax()])
                ext.append(c.std() * 2)
        return {"rate_hz": stats(rates or [0])["mean"], "extent_cents": stats(ext or [0])["mean"]}

    def voice_quality():
        p, per, act, voiced = v.crepe()
        x = p[voiced]
        if len(x) < 3:
            return {"jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0}
        T = 1 / x
        amp = librosa.feature.rms(y=v.v16, frame_length=640, hop_length=160)[0][: len(voiced)]
        amp = amp[voiced[: len(amp)]]
        pr = per[voiced].clip(1e-3, 1 - 1e-3)
        return {"jitter": float(np.mean(np.abs(np.diff(T))) / np.mean(T)),
                "shimmer": float(np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + 1e-12)),
                "hnr_db": float(np.mean(10 * np.log10(pr / (1 - pr))))}

    def rap_sung():
        p, per, act, voiced = v.crepe()
        return {"sung_ratio": float(voiced.sum() / max(act.sum(), 1))}

    def syllables():
        p, per, act, voiced = v.crepe()
        active_s = max(act.sum() / 100, 1e-3)
        on = librosa.onset.onset_detect(y=librosa.resample(v.voc, orig_sr=44100, target_sr=config.SR), sr=config.SR, units="time")
        return {"vocal_onsets_per_s": len(on) / active_s, "words_per_s": len(v.transcript().split()) / active_s}

    def layering():
        C = librosa.feature.chroma_cqt(y=librosa.resample(v.voc, orig_sr=44100, target_sr=config.SR), sr=config.SR)
        act = C.max(0) > 0.5
        n_peaks = (C > 0.6 * C.max(0, keepdims=True)).sum(0)
        return {"mean_simultaneous_pcs": float(n_peaks[act].mean()) if act.any() else 0.0}

    return {"pitch_range": pitch_range, "vibrato": vibrato, "voice_quality": voice_quality,
            "rap_vs_sung": rap_sung, "syllable_word_rate": syllables, "harmony_vocals": layering}
