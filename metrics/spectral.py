"""Baseline distance between log-magnitude STFT spectrograms."""

import librosa
import numpy as np


def spectral_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compare equal-length 1D waveforms without amplitude normalization.

    Use librosa's default STFT settings and the natural logarithm, with
    epsilon to keep zero magnitudes finite. Return the mean absolute
    difference between the two log-magnitude spectrograms.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Waveforms must be one-dimensional.")
    if x.shape != y.shape:
        raise ValueError("Waveforms must have the same length.")

    epsilon = 1e-8
    log_magnitude_x = np.log(np.abs(librosa.stft(x)) + epsilon)
    log_magnitude_y = np.log(np.abs(librosa.stft(y)) + epsilon)
    return float(np.mean(np.abs(log_magnitude_x - log_magnitude_y)))
