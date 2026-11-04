"""Symmetric log-STFT distance over jointly active, energy-weighted frames."""

import librosa
import numpy as np


N_FFT = 2048
HOP_LENGTH = 512
# A unit-amplitude bin-centred sine has Hann-windowed magnitude N_FFT / 4.
# This is a fixed reference, never a peak measured independently per input.
STFT_SINE_REFERENCE = N_FFT / 4
NOISE_FLOOR_DB = -120.0
MAGNITUDE_FLOOR = STFT_SINE_REFERENCE * 10 ** (NOISE_FLOOR_DB / 20)
METRIC_VERSION = "log_stft_joint_active_energy_v2"


def _waveform(value):
    audio = np.asarray(value)
    if audio.ndim != 1:
        raise ValueError("Waveforms must be one-dimensional.")
    if audio.size == 0:
        raise ValueError("Waveforms must not be empty.")
    if audio.dtype.kind not in "fiu":
        raise ValueError("Waveforms must contain real numeric samples.")
    with np.errstate(over="ignore", invalid="ignore"):
        audio = audio.astype(np.float64, copy=False)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Waveforms must contain only finite samples.")
    return audio


def _magnitude(audio):
    # Complete the analysis windows that overlap either end of the signal.
    # Without this fixed zero extension, adding silence exposes additional
    # partly occupied windows that the unpadded input never evaluated.
    audio = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.abs(librosa.stft(
            audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
            window="hann", center=True, pad_mode="constant",
        ))
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Waveform magnitude is too large for a finite STFT.")
    return magnitude


def spectral_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compare equal-length 1D waveforms without amplitude normalization.

    Use fixed STFT settings and complete the overlapping boundary windows.
    Clamp magnitudes to MAGNITUDE_FLOOR before taking natural logarithms;
    average their absolute difference over frequency in each frame.
    A frame is active if either input has a bin above the common floor.
    Average active frame errors with weight max(energy_x, energy_y), so
    common silence adds no weight and one-sided sounds remain included.
    Return a Python float in natural-log units (not dB), or 0.0 when neither
    input has an active frame. v1 thresholds must be recalibrated for v2.
    Assume a shared audio sample scale, conventionally full scale = 1.
    Inputs must be finite, non-empty, real 1D arrays of equal length.
    Invalid input or STFT overflow raises ValueError; no amplitude rescaling
    or independent peak normalization is performed.
    """
    x, y = _waveform(x), _waveform(y)
    if x.shape != y.shape:
        raise ValueError("Waveforms must have the same length.")

    magnitude_x, magnitude_y = _magnitude(x), _magnitude(y)
    active = np.any(np.maximum(magnitude_x, magnitude_y) > MAGNITUDE_FLOOR, axis=0)
    if not np.any(active):
        return 0.0

    magnitude_x, magnitude_y = magnitude_x[:, active], magnitude_y[:, active]
    frame_errors = np.mean(np.abs(
        np.log(np.maximum(magnitude_x, MAGNITUDE_FLOOR))
        - np.log(np.maximum(magnitude_y, MAGNITUDE_FLOOR))
    ), axis=0)

    # A SINGLE shared reference keeps squared energies finite. It cancels in
    # the weighted average and never rescales either log spectrum or removes
    # their gain difference. No input has its own normalization reference.
    reference = max(float(np.max(magnitude_x)), float(np.max(magnitude_y)))
    energy_x = np.sum((magnitude_x / reference) ** 2, axis=0)
    energy_y = np.sum((magnitude_y / reference) ** 2, axis=0)
    weights = np.maximum(energy_x, energy_y)
    return float(np.average(frame_errors, weights=weights))
