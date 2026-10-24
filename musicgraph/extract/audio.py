"""One track's audio plus lazily computed intermediates shared by several extractors."""

from functools import cached_property

import librosa
import numpy as np

from .. import config

# Model-based stages analyse a fixed budget of audio so cost doesn't grow with track length:
# EXCERPT_S-second windows centred at these fractions of the track (whole track if it's shorter).
EXCERPT_CENTRES = (0.2, 0.5, 0.8)
EXCERPT_S = 30.0


def excerpt_bounds(n: int, sr: int) -> list[tuple[int, int]]:
    w = int(EXCERPT_S * sr)
    if n <= w * len(EXCERPT_CENTRES):
        return [(0, n)]
    starts = [min(max(0, int(c * n) - w // 2), n - w) for c in EXCERPT_CENTRES]
    return [(s, s + w) for s in starts]


def db_(x, floor=1e-10):
    return 10 * np.log10(np.maximum(x, floor))


class Audio:
    """Decoded once at 44.1 kHz stereo; every other representation is derived on first use."""

    def __init__(self, track):
        self.track = track
        self.path = str(config.DATA_DIR / track["audio_path"])
        self.sr = config.SR
        st = librosa.load(self.path, sr=44100, mono=False)[0]
        self.stereo44 = np.stack([st, st]) if st.ndim == 1 else st
        self.mono44 = self.stereo44.mean(0).astype(np.float32)
        self.dur = len(self.mono44) / 44100

    def resampled(self, sr: int) -> np.ndarray:
        if sr == 44100:
            return self.mono44
        cache = self.__dict__.setdefault("_resampled", {})
        if sr not in cache:
            cache[sr] = librosa.resample(self.mono44, orig_sr=44100, target_sr=sr).astype(np.float32)
        return cache[sr]

    def excerpts(self, sr: int) -> list[np.ndarray]:
        y = self.resampled(sr)
        return [y[a:b] for a, b in excerpt_bounds(len(y), sr)]

    def stereo_excerpts(self) -> list[np.ndarray]:
        return [self.stereo44[:, a:b] for a, b in excerpt_bounds(self.stereo44.shape[1], 44100)]

    # ---- full-track intermediates at config.SR (22.05 kHz)
    @cached_property
    def y(self):
        return self.resampled(self.sr)

    @property
    def fps(self):
        return self.sr / config.HOP

    @cached_property
    def S(self):
        return np.abs(librosa.stft(self.y, n_fft=config.N_FFT, hop_length=config.HOP))

    @cached_property
    def P(self):
        return self.S ** 2

    @cached_property
    def freqs(self):
        return librosa.fft_frequencies(sr=self.sr, n_fft=config.N_FFT)

    @cached_property
    def mel_db(self):
        return librosa.power_to_db(librosa.feature.melspectrogram(S=self.P, sr=self.sr, n_mels=config.N_MELS), ref=np.max)

    @cached_property
    def onset_env(self):
        return librosa.onset.onset_strength(S=self.mel_db, sr=self.sr)

    @cached_property
    def _beat(self):
        tempo, beats = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=self.sr, hop_length=config.HOP)
        return float(np.atleast_1d(tempo)[0]), beats

    @property
    def tempo(self):
        return self._beat[0]

    @property
    def beats(self):
        return self._beat[1]

    @cached_property
    def beat_times(self):
        return librosa.frames_to_time(self.beats, sr=self.sr, hop_length=config.HOP)

    @cached_property
    def _hpss(self):
        return librosa.effects.hpss(self.y)

    @property
    def yh(self):
        return self._hpss[0]

    @property
    def yp(self):
        return self._hpss[1]

    @cached_property
    def chroma(self):
        return librosa.feature.chroma_stft(S=self.P, sr=self.sr)

    @cached_property
    def mfcc(self):
        return librosa.feature.mfcc(S=self.mel_db, n_mfcc=20)

    @cached_property
    def rms(self):
        return librosa.feature.rms(S=self.S, frame_length=config.N_FFT)[0]

    # ---- essentia 44.1 kHz frame analysis
    @cached_property
    def ess_spectra(self):
        import essentia.standard as es

        w, spec = es.Windowing(type="blackmanharris62"), es.Spectrum()
        return [spec(w(f)) for f in es.FrameGenerator(self.mono44, frameSize=2048, hopSize=1024, startFromZero=True)]

    @cached_property
    def _ess_peaks_pitch(self):
        import essentia.standard as es

        pk = es.SpectralPeaks(orderBy="frequency", minFrequency=20, maxFrequency=8000, maxPeaks=60,
                              magnitudeThreshold=1e-5, sampleRate=44100)
        yin = es.PitchYinFFT(frameSize=2048, sampleRate=44100)
        return [pk(s) for s in self.ess_spectra], [yin(s) for s in self.ess_spectra]

    @property
    def ess_peaks(self):
        return self._ess_peaks_pitch[0]

    @property
    def ess_pitch(self):
        return self._ess_peaks_pitch[1]
