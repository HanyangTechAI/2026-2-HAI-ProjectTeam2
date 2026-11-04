"""Reproducible synthetic calibration/holdout experiment; never runs search.

Run from the repository root: python -m metrics.evaluate_step0
Output is JSON; no dataset files are written. Synthetic results are diagnostic,
not a validated listening-quality or real-guitar acceptance standard.
"""

import importlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import librosa
import numpy as np

from metrics.parameters import evaluate_parameters, load_effect_definitions
from metrics.spectral import MAGNITUDE_FLOOR, METRIC_VERSION, N_FFT, NOISE_FLOOR_DB, spectral_distance


def _groups(records):
    groups = {True: [], False: []}
    for record in records:
        distance = record["distance"]
        if type(record["expected_match"]) is not bool or not math.isfinite(distance) or distance < 0:
            raise ValueError("Expected boolean match labels and finite nonnegative distances")
        groups[record["expected_match"]].append(distance)
    if not all(groups.values()):
        raise ValueError("Both intended matches and intended mismatches are required")
    return groups


def calibrate_audio_threshold(records):
    """Midpoint of an observed gap, using calibration records only.

    None means this simple rule cannot separate the intended classes.
    A returned number is still provisional outside the supplied population.
    """
    groups = _groups(records)
    upper, lower = max(groups[True]), min(groups[False])
    return (upper + lower) / 2 if upper < lower else None


def assess_audio_threshold(records, threshold):
    groups = _groups(records)
    if threshold is None:
        return {"threshold": None, "status": "no separating calibration threshold"}
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    return {
        "threshold": threshold,
        "false_rejects": sum(d > threshold for d in groups[True]),
        "false_accepts": sum(d <= threshold for d in groups[False]),
        "matches": len(groups[True]), "mismatches": len(groups[False]),
        "distributions": {
            str(label): dict(zip(("min", "median", "p95", "max"),
                                map(float, np.percentile(values, [0, 50, 95, 100]))))
            for label, values in groups.items()
        },
    }


def _renderer():
    # The existing module creates output directories on import. Confine that
    # side effect to a temporary directory without changing the team's code.
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            module = importlib.import_module("scripts.Create_Synthetic_pairs")
        finally:
            os.chdir(previous)
    return module.apply_effects


def _cases(seed, render):
    rng = np.random.default_rng(seed)
    rate = 22050
    time = np.arange(rate, dtype=np.float64) / rate
    frequency, phase = rng.uniform(110, 880), rng.uniform(0, 2 * np.pi)
    tone = 0.5 * np.sin(2 * np.pi * frequency * time + phase)
    noise = rng.normal(0, 1e-6, rate)
    dry = 0.2 * np.exp(-2 * time) * sum(
        np.sin(2 * np.pi * frequency * k * time + phase) / k for k in (1, 2, 3)
    )
    comp = {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
    drive = {"drive": 10.0, "level": 0.0}
    target = render(dry, rate, comp, "overdrive", drive)
    return [
        ("identical", tone, tone.copy(), True),
        ("tiny_noise", tone, tone + noise, True),
        ("gain_half", tone, tone * 0.5, False),
        ("frequency_x1.5", tone, 0.5 * np.sin(2 * np.pi * frequency * 1.5 * time + phase), False),
        ("same_parameters", target, render(dry, rate, comp, "overdrive", drive), True),
        ("drive_10_to_2", target, render(dry, rate, comp, "overdrive", {**drive, "drive": 2.0}), False),
        ("level_0_to_minus6", target, render(dry, rate, comp, "overdrive", {**drive, "level": -6.0}), False),
        ("one_sided_sound", np.zeros_like(tone), tone, False),
        ("missing_second_burst", np.concatenate([tone, np.zeros(rate), tone]),
         np.concatenate([tone, np.zeros(rate * 2)]), False),
        ("noise_only", np.zeros_like(tone), noise, True),
    ]


def _duration_variants(seed, render):
    """Variants stay in their source's split; they are not independent sources."""
    rng = np.random.default_rng(seed + 5000)
    for name, target, output, expected in _cases(seed, render):
        for seconds in (0, 1, 2, 5):
            padding = (0, seconds * 22050)
            yield name, f"silence_{seconds}s", np.pad(target, padding), np.pad(output, padding), expected
        yield (name, "noise_tail_5s",
               np.concatenate([target, rng.normal(0, 1e-6, 5 * 22050)]),
               np.concatenate([output, rng.normal(0, 1e-6, 5 * 22050)]), expected)


def silence_extension_sweep():
    """Independently reproduce v1 and compare aggregation alternatives."""
    rate = 22050
    time = np.arange(rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * time)
    rows = []
    for seconds in (0, 1, 2, 5):
        x = np.pad(tone, (0, seconds * rate))
        a, b = np.abs(librosa.stft(x)), np.abs(librosa.stft(x * 0.5))
        errors = np.mean(np.abs(np.log(np.maximum(a, MAGNITUDE_FLOOR))
                                - np.log(np.maximum(b, MAGNITUDE_FLOOR))), axis=0)
        active = np.any(np.maximum(a, b) > MAGNITUDE_FLOOR, axis=0)
        reference = max(float(np.max(a)), float(np.max(b)))
        weights = np.maximum(np.sum((a / reference) ** 2, axis=0),
                             np.sum((b / reference) ** 2, axis=0)) * active
        rows.append({
            "silence_seconds": seconds, "frames_v1": len(errors), "active_frames_v1": int(active.sum()),
            "v1_all_frame_mean": float(np.mean(errors)),
            "union_active_mean": float(np.mean(errors[active])),
            "union_energy_without_completed_windows": float(np.average(errors, weights=weights)),
            "v2_distance": spectral_distance(x, x * 0.5),
        })
    return rows


def floor_sweep():
    rate = 22050
    time = np.arange(rate) / rate
    silence = np.zeros(rate)
    tiny = np.random.default_rng(42).normal(0, 1e-6, rate)
    quiet = 1e-5 * np.sin(2 * np.pi * (rate * 41 / N_FFT) * time)
    tone = 0.5 * np.sin(2 * np.pi * 440 * time)
    cases = [("silence/tiny_noise", silence, tiny), ("silence/quiet_tone", silence, quiet),
             ("gain_half", tone, tone * 0.5),
             ("440_to_880", tone, 0.5 * np.sin(2 * np.pi * 880 * time)),
             ("identical", tone, tone), ("tone/tiny_noise", tone, tone + tiny)]
    rows = []
    for name, x, y in cases:
        a, b = np.abs(librosa.stft(x)), np.abs(librosa.stft(y))
        row = {"case": name, "before": float(np.mean(np.abs(np.log(a + 1e-8) - np.log(b + 1e-8))))}
        for db in (-140, -120, -100):
            floor = (N_FFT / 4) * 10 ** (db / 20)
            row[str(db)] = float(np.mean(np.abs(np.log(np.maximum(a, floor)) - np.log(np.maximum(b, floor)))))
        row["after"] = spectral_distance(x, y)
        rows.append(row)
    return rows


def main():
    render = _renderer()
    partitions = {}
    # Fresh, disjoint source seeds fixed before inspecting v2 holdout scores.
    # Previous v1 calibration/holdout used 100..109 and 200..209 respectively.
    for partition, seeds in (("calibration", range(300, 310)), ("holdout", range(400, 410))):
        records = []
        for seed in seeds:
            for name, variant, target, output, expected in _duration_variants(seed, render):
                records.append({"seed": seed, "case": name, "variant": variant, "expected_match": expected,
                                "distance": spectral_distance(target, output)})
        partitions[partition] = records
    threshold = calibrate_audio_threshold(partitions["calibration"])
    duration_sweep = silence_extension_sweep()
    for row in duration_sweep:
        row["v1_pass_with_invalidated_threshold"] = row["v1_all_frame_mean"] <= 0.049579072722078545
        row["v2_pass_with_recalibrated_threshold"] = row["v2_distance"] <= threshold if threshold is not None else None
    definitions = load_effect_definitions()
    knob_examples = {}
    for effect, name, value in (("overdrive", "drive", 2.0), ("compressor", "ratio", 8.0),
                                ("fuzz", "mode", "Modern")):
        truth = {spec["name"]: spec["default"] for spec in definitions[effect]["params"]}
        if effect == "overdrive":
            truth["drive"] = 10.0
        knob_examples[f"{effect}.{name}"] = evaluate_parameters(effect, truth, {**truth, name: value}, definitions)

    time = np.arange(22050) / 22050
    dry = 0.2 * np.sin(2 * np.pi * 220 * time)
    comp = {spec["name"]: spec["default"] for spec in definitions["compressor"]["params"]}
    fuzz = {spec["name"]: spec["default"] for spec in definitions["fuzz"]["params"]}
    mode_distance = spectral_distance(render(dry, 22050, comp, "fuzz", fuzz),
                                      render(dry, 22050, comp, "fuzz", {**fuzz, "mode": "Modern"}))

    # Verify the exact import/call used by search, without running its optimizer.
    from search.pipeline import spectral_distance as pipeline_metric
    assert pipeline_metric is spectral_distance
    assert isinstance(pipeline_metric(np.zeros(4096), np.zeros(4096)), float)
    import pedalboard
    import scipy
    print(json.dumps({
        "metric_version": METRIC_VERSION,
        "threshold_scope": {
            "purpose": "Original synthetic benchmark only; not a general audio acceptance rule",
            "quiet_local_events": "Not validated; see python -m metrics.analyze_quiet_events",
            "perceptual_acceptability": None,
        },
        "invalidated_v1_threshold": 0.049579072722078545,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "librosa": librosa.__version__, "pedalboard": pedalboard.__version__,
                     "scipy": scipy.__version__},
        "sample_rate": 22050, "base_case_durations_seconds": [1, 3],
        "appended_silence_seconds": [0, 1, 2, 5], "appended_noise_seconds": 5,
        "noise_floor_db": NOISE_FLOOR_DB, "raw_magnitude_floor": MAGNITUDE_FLOOR,
        "aggregation": "union-active frames; max of input frame energies; complete both boundary windows",
        "silence_extension_sweep": duration_sweep,
        "floor_sweep_note": "before=epsilon baseline; -140/-120/-100=v1 all-frame means; after=current v2",
        "floor_sweep": floor_sweep(), "knob_examples": knob_examples,
        "renderer_limit_example": {"case": "fuzz Vintage to Modern", "audio_distance": mode_distance,
                                   "knobs_match": False,
                                   "note": "Existing renderer ignores mode; excluded from audio mismatch calibration"},
        "pipeline_metric_import_and_call": "passed",
        "audio_results": {
            name: {"summary": assess_audio_threshold(records, threshold), "records": records}
            for name, records in partitions.items()
        },
        "status": "provisional synthetic diagnostic; real-data acceptance threshold not established",
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
