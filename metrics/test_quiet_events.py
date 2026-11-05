"""Characterize quiet-event risks, without assigning listening pass labels."""

import unittest

import numpy as np

from metrics.analyze_quiet_events import compare_aggregations, make_case
from metrics.spectral import spectral_distance


class QuietEventAuditTests(unittest.TestCase):
    def test_reproduces_reported_amplitude_examples(self):
        # Fixed -80 dB results; the -120 dB measurements remain in the report.
        for amplitude, expected in ((0.5, 0.04368052239822288),
                                    (0.05, 0.00032929048475491285),
                                    (0.005, 1.1223283074645755e-6)):
            x, y, _ = make_case(amplitude=amplitude)
            with self.subTest(amplitude=amplitude):
                self.assertAlmostEqual(spectral_distance(x, y), expected, delta=1e-10)

    def test_energy_share_matches_squared_amplitude_ratio(self):
        # Use aligned, flat-spectrum impulses to isolate the quadratic law.
        # The weakest active bin is 0.1 > 0.0512, so no frame is removed by
        # the floor. Sine edge frames can be removed at -80 dB and do NOT
        # obey the unmasked full-waveform ratio exactly. Precision is kept.
        for amplitude in (20.0, 2.0, 0.2):
            x = np.zeros(8192)
            x[2048] = 20.0
            y = x.copy()
            y[6144] = amplitude
            result = compare_aggregations(x, y, (4096, 8192))
            relative_energy = (amplitude / 20.0) ** 2
            self.assertAlmostEqual(result["event_energy_share"], relative_energy / (1 + relative_energy), places=9)

    def test_local_error_explains_small_global_score(self):
        x, y, event = make_case(amplitude=0.005)
        result = compare_aggregations(x, y, event)
        explained = result["event_energy_share"] * result["event_energy_weighted_error"]
        self.assertAlmostEqual(explained, result["energy_weighted_v2"], places=12)
        self.assertGreater(result["event_energy_weighted_error"], result["energy_weighted_v2"] * 1000)
        self.assertIsNone(result["acceptability"])

    def test_amplitude_sweep_is_finite_and_retains_nonzero_events(self):
        scores = []
        for amplitude in (0.0, 1e-5, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5):
            x, y, _ = make_case(amplitude=amplitude)
            score = spectral_distance(x, y)
            self.assertIs(type(score), float)
            self.assertTrue(np.isfinite(score))
            scores.append(score)
        # Zero and the old 1e-5 event now coincide because of the floor.
        self.assertEqual(scores[:2], [0.0, 0.0])
        self.assertTrue(all(a < b for a, b in zip(scores[1:], scores[2:])))

    def test_longer_sustain_increases_event_contribution(self):
        scores, shares = [], []
        for duration in (0.1, 0.5, 1.0, 2.0, 5.0):
            x, y, event = make_case(duration=duration)
            result = compare_aggregations(x, y, event)
            scores.append(result["energy_weighted_v2"])
            shares.append(result["event_energy_share"])
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))
        self.assertTrue(all(a < b for a, b in zip(shares, shares[1:])))

    def test_decay_time_changes_energy_contribution(self):
        scores = []
        for tau in (0.1, 0.5, 1.0, 2.0):
            x, y, _ = make_case(duration=2.0, envelope="decay", tau=tau)
            scores.append(spectral_distance(x, y))
        steady_x, steady_y, _ = make_case(duration=2.0)
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))
        self.assertLess(scores[-1], spectral_distance(steady_x, steady_y))

    def test_sustain_and_decay_truncation_remain_measurable(self):
        for envelope in ("steady", "decay"):
            scores = []
            for truncate_at in (0.1, 0.5, 1.5):
                x, y, _ = make_case(duration=2.0, truncate_at=truncate_at, envelope=envelope)
                scores.append(spectral_distance(x, y))
            with self.subTest(envelope=envelope):
                self.assertTrue(all(a > b > 0 for a, b in zip(scores, scores[1:])))

    def test_internal_rests_do_not_restore_zero_frame_dilution(self):
        scores = []
        for gap in (0.25, 1.0, 2.0, 5.0):
            x, y, _ = make_case(gap=gap)
            scores.append(spectral_distance(x, y))
        # Non-hop-aligned rests move the later event's STFT grid. A bounded
        # small change is expected, rather than exact translation invariance.
        self.assertLess(max(scores) / min(scores), 1.05)

    def test_longer_matched_loud_head_suppresses_same_quiet_event(self):
        scores = []
        for head_duration in (0.25, 1.0, 4.0, 8.0):
            x, y, _ = make_case(head_duration=head_duration)
            scores.append(spectral_distance(x, y))
        self.assertTrue(all(a > b > 0 for a, b in zip(scores, scores[1:])))

    def test_long_silence_stability_for_quiet_and_decaying_events(self):
        for settings in ({"amplitude": 0.05}, {"amplitude": 0.005},
                         {"envelope": "decay", "duration": 2.0},
                         {"envelope": "decay", "duration": 2.0, "truncate_at": 1.5}):
            x, y, _ = make_case(**settings)
            baseline = spectral_distance(x, y)
            for seconds in (2, 5):
                with self.subTest(settings=settings, silence=seconds):
                    score = spectral_distance(np.pad(x, (0, 22050 * seconds)), np.pad(y, (0, 22050 * seconds)))
                    self.assertAlmostEqual(score, baseline, places=12)

    def test_extra_and_missing_event_diagnostics_are_symmetric(self):
        for settings in ({}, {"amplitude": 0.005}, {"envelope": "decay"},
                         {"duration": 2.0, "truncate_at": 0.5}):
            x, y, event = make_case(**settings)
            forward = compare_aggregations(x, y, event)
            reverse = compare_aggregations(y, x, event)
            self.assertEqual(forward, reverse)
            self.assertAlmostEqual(forward["reconstructed_energy_score"], forward["energy_weighted_v2"], places=12)
            self.assertIsNone(forward["acceptability"])

    def test_alternative_averages_measure_different_sensitivity(self):
        x, y, event = make_case(amplitude=0.05)
        result = compare_aggregations(x, y, event)
        self.assertLess(result["energy_weighted_v2"], result["sqrt_energy_mean"])
        self.assertLess(result["sqrt_energy_mean"], result["uniform_active_mean"])
        self.assertLess(result["uniform_active_mean"], result["active_frame_max"])
        # No existing v2 threshold is applied to these alternative values.

    def test_silence_and_subfloor_noise_have_safe_diagnostics(self):
        silence = np.zeros(22050 * 3)
        noise = np.random.default_rng(47).normal(0, 1e-6, silence.size)
        for x, y in ((silence, silence), (silence, noise)):
            result = compare_aggregations(x, y, (22050, len(x)))
            for name in ("energy_weighted_v2", "uniform_active_mean", "sqrt_energy_mean", "active_frame_p95", "active_frame_max"):
                self.assertEqual(result[name], 0.0)
            self.assertEqual(result["active_frames"], 0)
            self.assertEqual(result["event_energy_share"], 0.0)
            self.assertIsNone(result["event_energy_weighted_error"])
            self.assertIsNone(result["acceptability"])


if __name__ == "__main__":
    unittest.main()
