"""Shared waveform segments and fixed-reference log-mel preprocessing."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration_sec: float = 4.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float | None = None
    power_floor: float = 1e-10

    def __post_init__(self):
        if (self.sample_rate <= 0 or self.duration_sec <= 0 or self.n_fft < 16
                or self.hop_length <= 0 or self.n_mels < 16
                or self.fmin < 0 or self.power_floor <= 0):
            raise ValueError('Invalid audio configuration')
        if not self.fmin < (self.fmax or self.sample_rate / 2) <= self.sample_rate / 2:
            raise ValueError('Invalid mel frequency limits')
        if self.samples < self.n_fft or 1 + self.samples // self.hop_length < 16:
            raise ValueError('Audio too short for four CNN pooling blocks')

    @property
    def samples(self):
        return round(self.sample_rate * self.duration_sec)

    def to_dict(self):
        return asdict(self)


def read_audio(path, sample_rate):
    import soundfile as sf

    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    if not np.isfinite(audio).all():
        raise ValueError(f'Invalid resampled audio: {path}')
    return np.ascontiguousarray(audio, dtype=np.float32)


def segment(audio, config, start_sample=0, reject_silence=True):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Expected finite nonempty mono audio')
    if start_sample < 0 or start_sample >= len(audio):
        raise ValueError('Crop starts outside audio')
    result = audio[start_sample:start_sample + config.samples]
    if reject_silence and np.max(np.abs(result)) < 1e-7:
        raise ValueError('Silent audio segment')
    return np.pad(result, (0, config.samples - len(result))).astype(np.float32)


def log_mel(audio, config):
    import librosa

    audio = np.asarray(audio)
    if audio.shape != (config.samples,) or not np.isfinite(audio).all():
        raise ValueError('Expected a finite waveform matching the configured segment')
    power = librosa.feature.melspectrogram(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, power=2, pad_mode='constant')
    # Fixed scale for both inputs. Never normalize each example by its own peak.
    if not np.isfinite(power).all():
        raise ValueError('Non-finite mel spectrum')
    return (np.log(np.maximum(power, config.power_floor)) / 10).astype(np.float32)[None]
