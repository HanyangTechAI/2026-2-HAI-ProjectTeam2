"""Explicitly scaled weighted average of the existing audio distances."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from metrics.dynamics import dynamics_distance
from metrics.spectral import spectral_distance


@dataclass(frozen=True)
class CombinedDistanceResult:
    """Raw scores, dimensionless scaled scores, and the settings actually used."""

    combined: float
    spectral: float  # Natural-log magnitude difference, not dB.
    dynamics: float  # Mean absolute RMS-level difference in dB.
    scaled_spectral: float
    scaled_dynamics: float
    spectral_weight: float  # Normalized weight.
    dynamics_weight: float  # Normalized weight.
    spectral_scale: float
    dynamics_scale: float


def _finite_nonnegative(value, name, *, positive=False):
    message = f"{name} must be a finite {'positive' if positive else 'nonnegative'} real scalar."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(message)
    return value


def combined_distance_details(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: int,
    *,
    spectral_weight: float,
    dynamics_weight: float,
    spectral_scale: float,
    dynamics_scale: float,
) -> CombinedDistanceResult:
    """Compute both raw distances and their explicitly scaled weighted average.

    All four settings are required: there is no calibrated default. Scales are
    finite positive DIVISORS, in each raw metric's units, fixed across pairs.
    Supply explicit engineering settings or statistics fitted on a separate
    calibration/training set; never fit them separately to each input pair.
    Scaled scores are dimensionless, nonnegative, and not limited to [0, 1].

    Weights must be finite nonnegative real scalars, with at least one positive.
    Normalize by their sum (using a common maximum first to avoid overflow).
    The result is w_s * (spectral / spectral_scale)
                + w_d * (dynamics / dynamics_scale).

    ALWAYS call both original metrics, even with a zero weight, preserving
    their validation and providing both raw scores. In particular, a valid
    common sample_rate is required even for spectral-only weighting. Inputs
    must be equal-length, nonempty, finite real 1D mono waveforms with a shared
    amplitude scale, source rate, and time alignment. No input normalization,
    resampling, alignment, or mutation is performed. Integer PCM must first
    be converted by the caller using its common full scale.

    The spectral -80 dB STFT floor and dynamics -100 dBFS RMS floor remain
    separate and unchanged. Invalid settings/inputs, existing metric overflow,
    or nonfinite scaled scores raise ValueError, including for a zero-weight
    component. Equality at zero does not guarantee identical waveforms or
    perceptual similarity. Scores and settings in the result are Python floats.
    """
    sw = _finite_nonnegative(spectral_weight, "spectral_weight")
    dw = _finite_nonnegative(dynamics_weight, "dynamics_weight")
    ss = _finite_nonnegative(spectral_scale, "spectral_scale", positive=True)
    ds = _finite_nonnegative(dynamics_scale, "dynamics_scale", positive=True)
    maximum = max(sw, dw)
    if maximum == 0:
        raise ValueError("At least one weight must be positive.")
    sw, dw = sw / maximum, dw / maximum
    total = sw + dw
    sw, dw = sw / total, dw / total

    # Dynamics validates sample_rate and malformed waveforms before analysis.
    # Do not short-circuit either metric, including for identical inputs.
    dynamics = _finite_nonnegative(dynamics_distance(x, y, sample_rate), "dynamics score")
    spectral = _finite_nonnegative(spectral_distance(x, y), "spectral score")
    scaled_spectral = _finite_nonnegative(spectral / ss, "scaled_spectral")
    scaled_dynamics = _finite_nonnegative(dynamics / ds, "scaled_dynamics")
    combined = _finite_nonnegative(sw * scaled_spectral + dw * scaled_dynamics, "combined score")
    return CombinedDistanceResult(
        combined=combined, spectral=spectral, dynamics=dynamics,
        scaled_spectral=scaled_spectral, scaled_dynamics=scaled_dynamics,
        spectral_weight=sw, dynamics_weight=dw,
        spectral_scale=ss, dynamics_scale=ds,
    )


def combined_distance(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: int,
    *,
    spectral_weight: float,
    dynamics_weight: float,
    spectral_scale: float,
    dynamics_scale: float,
) -> float:
    """Return the combined Python float; see combined_distance_details for rules.

    Use combined_distance_details with the same arguments to inspect raw
    scores, scaled scores, normalized weights, and scales in a single call.
    """
    return combined_distance_details(
        x, y, sample_rate,
        spectral_weight=spectral_weight, dynamics_weight=dynamics_weight,
        spectral_scale=spectral_scale, dynamics_scale=dynamics_scale,
    ).combined
