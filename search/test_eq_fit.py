"""search/eq_fit.py 테스트 (VST 불필요): python -m unittest search.test_eq_fit"""
import unittest

import numpy as np

from search.eq_fit import apply_eq, eq_metric, fit_eq
from metrics.spectral import spectral_distance

SR = 44100


def guitar_like(seconds=1.0, seed=0):
    """여러 배음 + 약한 잡음: 전 대역에 에너지가 있는 테스트 소리."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    tone = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 60))
    return (0.1 * tone * np.exp(-2 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


class FitEqTest(unittest.TestCase):
    def test_recovers_known_eq(self):
        x = guitar_like()
        truth = {"low_gain_db": 6.0, "mid_freq_hz": 1500.0, "mid_gain_db": -7.0, "high_gain_db": 4.0}
        got, info = fit_eq(x, apply_eq(x, SR, truth), SR)
        self.assertTrue(info["active"])
        for key in ("low_gain_db", "mid_gain_db", "high_gain_db"):
            self.assertAlmostEqual(got[key], truth[key], delta=0.5, msg=key)
        self.assertLess(abs(np.log2(got["mid_freq_hz"] / truth["mid_freq_hz"])), 0.2)

    def test_no_eq_reports_inactive_defaults(self):
        x = guitar_like()
        got, info = fit_eq(x, x.copy(), SR)
        self.assertFalse(info["active"])
        self.assertEqual(got, {"low_gain_db": 0.0, "mid_freq_hz": 1000.0, "mid_gain_db": 0.0, "high_gain_db": 0.0})

    def test_values_stay_in_schema_range(self):
        x = guitar_like()
        extreme = apply_eq(apply_eq(x, SR, {"low_gain_db": 12.0, "mid_freq_hz": 1000.0, "mid_gain_db": 0.0,
                                             "high_gain_db": -12.0}),
                           SR, {"low_gain_db": 12.0, "mid_freq_hz": 1000.0, "mid_gain_db": 0.0, "high_gain_db": -12.0})
        got, _ = fit_eq(x, extreme, SR)
        for key in ("low_gain_db", "mid_gain_db", "high_gain_db"):
            self.assertTrue(-12.0 <= got[key] <= 12.0)
        self.assertTrue(200.0 <= got["mid_freq_hz"] <= 5000.0)

    def test_metric_removes_eq_difference(self):
        x = guitar_like(seed=1)
        target = apply_eq(x, SR, {"low_gain_db": -5.0, "mid_freq_hz": 800.0, "mid_gain_db": 6.0, "high_gain_db": 3.0})
        record = []
        metric = eq_metric(SR, record=record)
        self.assertLess(metric(x, target), spectral_distance(x, target) * 0.2)
        self.assertEqual(len(record), 1)


if __name__ == "__main__":
    unittest.main()
