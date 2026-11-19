"""The team's spectral + dynamics metric with TRAIN-only calibration."""
from dataclasses import asdict

import numpy as np

from metrics.combined import combined_distance_details
from metrics.dynamics import dynamics_distance
from metrics.spectral import spectral_distance


def raw_distances(dry, wet, rate):
    return [spectral_distance(dry, wet), dynamics_distance(dry, wet, rate)]


def calibrate(raw, train_indices):
    scales = np.maximum(np.median(np.asarray(raw)[train_indices], axis=0), 1e-8)
    return {'spectral_weight': 0.5, 'dynamics_weight': 0.5,
            'spectral_scale': float(scales[0]), 'dynamics_scale': float(scales[1]),
            'calibration': 'median of generated-dry/target distances on train originals ONLY',
            'version': 'train_median_spectral_dynamics_equal_weight_v1'}


def compare(x, y, rate, settings):
    keys = ('spectral_weight', 'dynamics_weight', 'spectral_scale', 'dynamics_scale')
    return asdict(combined_distance_details(x, y, rate, **{k: settings[k] for k in keys}))
