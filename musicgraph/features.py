"""Handcrafted, interpretable track-level features.

Frame-level descriptors are summarised with a fixed set of statistics; every
dimension carries a name so the vector stays inspectable.
"""

import librosa
import numpy as np

from . import config

EXTRACTOR = "handcrafted"
VERSION = 1

STATS = ("mean", "std", "p10", "p90")


def _summarise(name: str, M: np.ndarray, names: list, values: list) -> None:
    """M: (n_rows, n_frames). Appends len(STATS) * n_rows dims."""
    M = np.atleast_2d(M)
    stats = (M.mean(1), M.std(1), np.percentile(M, 10, axis=1), np.percentile(M, 90, axis=1))
    for stat, v in zip(STATS, stats):
        for i, x in enumerate(v):
            names.append(f"{name}{i if M.shape[0] > 1 else ''}_{stat}")
            values.append(x)


def extract(y: np.ndarray, mel_db: np.ndarray) -> tuple[list[str], np.ndarray]:
    sr, hop, n_fft = config.SR, config.HOP, config.N_FFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    names, values = [], []

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    _summarise("mfcc", mfcc, names, values)
    _summarise("mfcc_delta", librosa.feature.delta(mfcc), names, values)

    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)
    _summarise("chroma", chroma, names, values)
    _summarise("tonnetz", librosa.feature.tonnetz(chroma=chroma), names, values)

    _summarise("centroid", librosa.feature.spectral_centroid(S=S, sr=sr), names, values)
    _summarise("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr), names, values)
    _summarise("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr), names, values)
    _summarise("flatness", librosa.feature.spectral_flatness(S=S), names, values)
    _summarise("contrast", librosa.feature.spectral_contrast(S=S, sr=sr), names, values)
    _summarise("zcr", librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop), names, values)
    _summarise("rms", librosa.feature.rms(S=S, frame_length=n_fft), names, values)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr)
    _summarise("onset", onset_env, names, values)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    beat_strength = onset_env[beats] if len(beats) else np.zeros(1)
    duration = len(y) / sr
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    globals_ = {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "beat_strength_mean": float(beat_strength.mean()),
        "beat_strength_std": float(beat_strength.std()),
        "beat_regularity": float(np.std(np.diff(beats)) / (np.mean(np.diff(beats)) + 1e-9)) if len(beats) > 2 else 0.0,
        "onset_rate": len(onsets) / duration,
        "duration_s": duration,
    }
    names += list(globals_)
    values += list(globals_.values())

    vec = np.nan_to_num(np.asarray(values, dtype=np.float32))
    return names, vec
