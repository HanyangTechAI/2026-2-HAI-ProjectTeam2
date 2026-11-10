"""Symmetric distance between short-time RMS envelopes, in decibels."""

from numbers import Real

import numpy as np


FRAME_MS = 30.0
HOP_MS = 10.0
# Fixed RMS reference: a constant waveform of amplitude 1 has level 0 dBFS.
# This is independent of spectral.py's STFT-bin reference and noise floor.
RMS_FLOOR_DBFS = -100.0
RMS_FLOOR = 10 ** (RMS_FLOOR_DBFS / 20)
MAX_SAMPLE_RATE = 768_000
METRIC_VERSION = "rms_db_joint_active_uniform_v1"


def _waveform(value):
    try:
        audio = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Waveforms must be real numeric one-dimensional arrays.") from exc
    if audio.ndim != 1:
        raise ValueError("Waveforms must be one-dimensional mono arrays.")
    if audio.size == 0:
        raise ValueError("Waveforms must not be empty.")
    if audio.dtype.kind not in "fiu":
        raise ValueError("Waveforms must contain real numeric samples, not bool or complex.")
    with np.errstate(over="ignore", invalid="ignore"):
        audio = audio.astype(np.float64, copy=False)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Waveforms must contain only finite samples.")
    return audio


def _sample_rate(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not 1 <= value <= MAX_SAMPLE_RATE or not float(value).is_integer()):
        raise ValueError(
            f"sample_rate must be one common integer rate from 1 to {MAX_SAMPLE_RATE} Hz."
        )
    return int(value)


def _rms(audio, frame_length, hop_length):
    # Evaluate EVERY hop-grid window overlapping the input, including partial
    # windows at both ends. Complete these with zeros using a fixed divisor.
    # Appended common zeros therefore cannot expose a new nonzero window.
    left = ((frame_length - 1) // hop_length) * hop_length
    padded = np.pad(audio, (left, frame_length - 1))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
    if not np.all(np.isfinite(rms)):
        raise ValueError("Waveform amplitudes are too large for finite RMS power calculations.")
    return rms


def dynamics_distance(x: np.ndarray, y: np.ndarray, sample_rate: int) -> float:
    """Return mean absolute RMS-level difference over jointly active frames.

    Use rectangular 30 ms windows every 10 ms, rounded to the nearest sample
    (at least one sample each). Levels are 20*log10(max(RMS, 1e-5)) relative
    to a FIXED RMS of 1, never to either input's peak. Average uniformly over
    frames where either RMS exceeds -100 dBFS; return 0.0 if none do.
    A score of zero means equal measured envelopes, not identical waveforms.

    Inputs must be equal-length, nonempty, finite, real mono waveforms with
    a shared amplitude scale, sample rate, and time alignment. No resampling,
    alignment, clipping, or per-input normalization is performed. Integer
    samples retain their raw scale; convert integer PCM using its full scale
    BEFORE calling. Values above 1 are allowed, but RMS overflow is rejected.
    Invalid inputs raise ValueError. sample_rate accepts integer-valued real
    scalars from 1 through 768000 Hz (an allocation guard, not an audio rule).
    Arrays carry no rate metadata: callers must verify equal source rates.

    The floor and timing are initial engineering choices, not calibrated
    hearing thresholds. Subfloor useful sounds are lost; short transitions
    are smoothed, and long matching active passages dilute local errors.
    """
    rate = _sample_rate(sample_rate)
    x, y = _waveform(x), _waveform(y)
    if x.shape != y.shape:
        raise ValueError("Waveforms must have the same length.")
    frame_length = max(1, round(rate * FRAME_MS / 1000))
    hop_length = max(1, round(rate * HOP_MS / 1000))
    rms_x = _rms(x, frame_length, hop_length)
    rms_y = _rms(y, frame_length, hop_length)
    active = np.maximum(rms_x, rms_y) > RMS_FLOOR
    if not np.any(active):
        return 0.0
    level_x = 20 * np.log10(np.maximum(rms_x[active], RMS_FLOOR))
    level_y = 20 * np.log10(np.maximum(rms_y[active], RMS_FLOOR))
    return float(np.mean(np.abs(level_x - level_y)))
