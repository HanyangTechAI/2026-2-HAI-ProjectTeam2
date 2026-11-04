"""Audit quiet one-sided events without inventing perceptual pass labels.

Run: python -m metrics.analyze_quiet_events
The production v2 metric is unchanged. Alternative aggregations and a known
event interval are diagnostic measurements, not new acceptance criteria.
"""

import json
import platform

import librosa
import numpy as np

from metrics.spectral import (
    HOP_LENGTH, MAGNITUDE_FLOOR, METRIC_VERSION, N_FFT, spectral_distance,
)


def compare_aggregations(x, y, event_samples=None):
    """Return global alternatives and optional known-event diagnostics.

    event_samples=(start, end) is an externally supplied half-open sample
    interval, not an inferred audible region. All frames whose windows
    overlap it are included, so edge windows are not discarded. In synthetic
    cases below this interval contains every waveform difference.
    No independent amplitude normalization or acceptance decision is made.
    """
    production_score = spectral_distance(x, y)  # also validates both inputs
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if event_samples is not None:
        if (len(event_samples) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in event_samples)
                or not 0 <= event_samples[0] < event_samples[1] <= len(x)):
            raise ValueError("event_samples must be a nonempty interval inside the waveforms")

    magnitudes = [np.abs(librosa.stft(
        np.pad(audio, (N_FFT // 2, N_FFT // 2)), n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=N_FFT, window="hann",
        center=True, pad_mode="constant",
    )) for audio in (x, y)]
    a, b = magnitudes
    active = np.any(np.maximum(a, b) > MAGNITUDE_FLOOR, axis=0)
    result = {
        "energy_weighted_v2": production_score,
        "uniform_active_mean": 0.0,
        "sqrt_energy_mean": 0.0,
        "active_frame_p95": 0.0,
        "active_frame_max": 0.0,
        "active_frames": int(active.sum()),
        "reconstructed_energy_score": 0.0,
        "event_energy_share": None,
        "event_energy_weighted_error": None,
        "event_uniform_error": None,
        "acceptability": None,
    }
    if not active.any():
        if event_samples is not None:
            result["event_energy_share"] = 0.0
        return result

    errors = np.mean(np.abs(np.log(np.maximum(a, MAGNITUDE_FLOOR))
                            - np.log(np.maximum(b, MAGNITUDE_FLOOR))), axis=0)
    # One shared scale only stabilizes squared energies; log spectra keep
    # their absolute levels. This is algebraically the v2 weighting rule.
    reference = max(float(a.max()), float(b.max()))
    weights = np.maximum(np.sum((a / reference) ** 2, axis=0),
                         np.sum((b / reference) ** 2, axis=0))
    weights[~active] = 0
    result.update(
        uniform_active_mean=float(errors[active].mean()),
        sqrt_energy_mean=float(np.average(errors[active], weights=np.sqrt(weights[active]))),
        active_frame_p95=float(np.percentile(errors[active], 95)),
        active_frame_max=float(errors[active].max()),
        reconstructed_energy_score=float(np.average(errors, weights=weights)),
    )
    if event_samples is not None:
        start, end = event_samples
        centers = np.arange(len(active)) * HOP_LENGTH - N_FFT // 2
        event = active & (centers + N_FFT // 2 > start) & (centers - N_FFT // 2 < end)
        result["event_energy_share"] = float(weights[event].sum() / weights.sum())
        if event.any():
            result["event_energy_weighted_error"] = float(np.average(errors[event], weights=weights[event]))
            result["event_uniform_error"] = float(errors[event].mean())
    return result


def make_case(amplitude=0.05, duration=1.0, gap=1.0, head_duration=1.0,
              envelope="steady", tau=0.5, truncate_at=None):
    """Make a matched loud head and a one-sided 440 Hz event at 22050 Hz.

    With truncate_at, both signals share the beginning of the quiet event
    and only its remaining sustain/decay is missing. Otherwise the event
    is added entirely to the second signal, matching the reported example.
    """
    if (not all(np.isfinite(v) for v in (amplitude, duration, gap, head_duration, tau))
            or amplitude < 0 or duration <= 0 or gap < 0 or head_duration <= 0 or tau <= 0):
        raise ValueError("Expected finite amplitudes/durations and positive tau")
    if envelope not in ("steady", "decay"):
        raise ValueError("Unknown envelope")
    rate = 22050
    event_count, head_count, gap_count = round(duration * rate), round(head_duration * rate), round(gap * rate)
    if event_count < 1 or head_count < 1:
        raise ValueError("Head and event must contain at least one sample")
    time = np.arange(event_count) / rate
    event = amplitude * np.sin(2 * np.pi * 440 * time)
    if envelope == "decay":
        event *= np.exp(-time / tau)
    head = 0.5 * np.sin(2 * np.pi * 440 * np.arange(head_count) / rate)
    start = head_count + gap_count
    x = np.concatenate([head, np.zeros(gap_count + event_count)])
    y = np.concatenate([head, np.zeros(gap_count), event])
    if truncate_at is not None:
        if not np.isfinite(truncate_at) or not 0 <= truncate_at < duration:
            raise ValueError("truncate_at must lie inside the event")
        kept = round(truncate_at * rate)
        if kept >= event_count:
            raise ValueError("Truncation must leave at least one missing sample")
        x[start:start + kept] = event[:kept]
        start += kept
    return x, y, (start, len(x))


def audit_cases():
    for amplitude in (0.5, 0.2, 0.1, 0.05, 0.02, 0.005, 0.001, 1e-5, 0.0):
        yield "amplitude", {"amplitude": amplitude}
    for duration in (0.1, 0.5, 1.0, 2.0, 5.0):
        yield "sustain_duration", {"duration": duration}
    for tau in (0.1, 0.5, 1.0, 2.0):
        yield "decay", {"duration": 2.0, "envelope": "decay", "tau": tau}
    for truncate_at in (0.1, 0.5, 1.5):
        for envelope in ("steady", "decay"):
            yield "truncated_tail", {"duration": 2.0, "truncate_at": truncate_at, "envelope": envelope}
    for gap in (0.25, 1.0, 2.0, 5.0):
        yield "internal_rest", {"gap": gap}
    for head_duration in (0.25, 1.0, 4.0, 8.0):
        yield "matched_head_duration", {"head_duration": head_duration}


def main():
    records = []
    for family, settings in audit_cases():
        x, y, event = make_case(**settings)
        records.append({"family": family, "settings": settings,
                        "event_samples": list(event), "sample_count": len(x),
                        **compare_aggregations(x, y, event)})
    print(json.dumps({
        "metric_version": METRIC_VERSION,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__},
        "sample_rate": 22050, "frequency_hz": 440,
        "matched_head_amplitude": 0.5,
        "default_settings": {"amplitude": 0.05, "duration": 1.0, "gap": 1.0, "head_duration": 1.0,
                             "envelope": "steady", "tau": 0.5, "truncate_at": None},
        "acceptability": None,
        "status": "No listening labels; no threshold fitted or perceptual acceptance inferred",
        "records": records,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
