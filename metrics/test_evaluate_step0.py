"""Check threshold failure modes and frozen calibration/holdout use."""

import unittest

from metrics.evaluate_step0 import assess_audio_threshold, calibrate_audio_threshold


def records(matches, mismatches):
    return ([{"expected_match": True, "distance": value} for value in matches]
            + [{"expected_match": False, "distance": value} for value in mismatches])


class AudioThresholdTests(unittest.TestCase):
    def test_threshold_from_calibration_and_equality_boundary(self):
        threshold = calibrate_audio_threshold(records([0, 0.02], [0.1, 0.2]))
        self.assertAlmostEqual(threshold, 0.06)
        heldout = assess_audio_threshold(records([threshold], [0.11]), threshold)
        self.assertEqual(heldout["false_rejects"], 0)
        self.assertEqual(heldout["false_accepts"], 0)

    def test_heldout_failures_do_not_adjust_threshold(self):
        threshold = calibrate_audio_threshold(records([0, 0.02], [0.1]))
        result = assess_audio_threshold(records([0.08], [0.03]), threshold)
        self.assertEqual(result["false_rejects"], 1)
        self.assertEqual(result["false_accepts"], 1)
        self.assertEqual(result["threshold"], threshold)

    def test_overlapping_and_touching_classes_cannot_define_gap(self):
        for data in (records([0.1], [0.09]), records([0.1], [0.1])):
            self.assertIsNone(calibrate_audio_threshold(data))
            self.assertIsNone(assess_audio_threshold(data, None)["threshold"])

    def test_invalid_or_single_class_records(self):
        for data in ([], records([0], []), records([], [1]),
                     records([float("nan")], [1]), records([0], [float("inf")]), records([-1], [1]),
                     [{"expected_match": "yes", "distance": 0}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                calibrate_audio_threshold(data)


if __name__ == "__main__":
    unittest.main()
