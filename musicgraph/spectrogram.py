from pathlib import Path

import librosa
import numpy as np

from . import config


def load_audio(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(config.DATA_DIR / audio_path, sr=config.SR, mono=True)
    return y


def log_mel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP, n_mels=config.N_MELS
    )
    return librosa.power_to_db(S, ref=np.max)


def spec_path(track_id: int) -> Path:
    return config.SPEC_DIR / f"{track_id}.npy"


def compute_and_save(track_id: int, y: np.ndarray) -> tuple[Path, int]:
    M = log_mel(y)
    path = spec_path(track_id)
    np.save(path, M.astype(np.float16))
    return path, M.shape[1]


def load(track_id: int) -> np.ndarray:
    return np.load(spec_path(track_id)).astype(np.float32)
