"""Analytic checks for range/scale-aware parameter errors and chain alignment."""

import copy
import json
import math
from pathlib import Path
import unittest

from parameters import evaluate_chain, evaluate_parameters, load_effect_definitions


class ParameterErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_effect_definitions()

    def defaults(self, effect):
        return {spec["name"]: spec["default"] for spec in self.definitions[effect]["params"]}

    def test_identical_parameters_for_every_effect(self):
        for effect in self.definitions:
            values = self.defaults(effect)
            result = evaluate_parameters(effect, values, values)
            self.assertEqual(result["mean_normalized_error"], 0)
            self.assertEqual(result["max_normalized_error"], 0)
            self.assertEqual(result["categorical_all_match"], True if effect == "fuzz" else None)

    def test_linear_units_and_range(self):
        truth = self.defaults("overdrive")
        predicted = {**truth, "drive": truth["drive"] + 3, "level": 6}
        result = evaluate_parameters("overdrive", truth, predicted)
        self.assertEqual(result["per_parameter"]["drive"]["absolute_error"], 3)
        self.assertEqual(result["per_parameter"]["drive"]["unit"], "dB")
        self.assertAlmostEqual(result["mean_normalized_error"], (3 / 15 + 6 / 24) / 2)
        self.assertEqual(result["max_normalized_error"], 0.25)

    def test_log_scale_equal_ratios_have_equal_errors(self):
        values = self.defaults("compressor")
        first = evaluate_parameters("compressor", {**values, "ratio": 2}, {**values, "ratio": 4})
        second = evaluate_parameters("compressor", {**values, "ratio": 4}, {**values, "ratio": 8})
        first_error = first["per_parameter"]["ratio"]["normalized_error"]
        self.assertAlmostEqual(first_error, math.log(2) / math.log(20))
        self.assertAlmostEqual(first_error, second["per_parameter"]["ratio"]["normalized_error"])

    def test_all_continuous_range_endpoints(self):
        for effect, definition in self.definitions.items():
            for spec in definition["params"]:
                if spec["type"] != "continuous":
                    continue
                actual, estimate = self.defaults(effect), self.defaults(effect)
                actual[spec["name"]], estimate[spec["name"]] = spec["range"]
                with self.subTest(effect=effect, parameter=spec["name"]):
                    result = evaluate_parameters(effect, actual, estimate)
                    self.assertAlmostEqual(result["per_parameter"][spec["name"]]["normalized_error"], 1)

    def test_categorical_mismatch_not_hidden_in_continuous_mean(self):
        values = self.defaults("fuzz")
        result = evaluate_parameters("fuzz", values, {**values, "mode": "Modern"})
        self.assertFalse(result["categorical_all_match"])
        self.assertFalse(result["per_parameter"]["mode"]["match"])
        self.assertEqual(result["mean_normalized_error"], 0)

    def test_missing_and_unknown_values_rejected(self):
        values = self.defaults("overdrive")
        for bad in ({"drive": 6}, {**values, "typo": 0}):
            for truth, predicted in ((bad, values), (values, bad)):
                with self.assertRaises(ValueError):
                    evaluate_parameters("overdrive", truth, predicted)

    def test_invalid_continuous_values_on_either_side(self):
        values = self.defaults("compressor")
        for value in (float("nan"), float("inf"), -1, 0, 21, True, "4"):
            bad = {**values, "ratio": value}
            for truth, predicted in ((values, bad), (bad, values)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    evaluate_parameters("compressor", truth, predicted)

    def test_invalid_category_and_unknown_effect(self):
        values = self.defaults("fuzz")
        with self.assertRaises(ValueError):
            evaluate_parameters("fuzz", values, {**values, "mode": "unknown"})
        with self.assertRaises(ValueError):
            evaluate_parameters("unknown", {}, {})

    def test_boolean_and_absent_continuous_aggregate(self):
        definitions = {"switch": {"params": [{"name": "enabled", "type": "boolean"}]}}
        result = evaluate_parameters("switch", {"enabled": True}, {"enabled": False}, definitions)
        self.assertFalse(result["categorical_all_match"])
        self.assertIsNone(result["mean_normalized_error"])
        self.assertIsNone(result["max_normalized_error"])

    def example_chain(self):
        path = Path(__file__).resolve().parents[1] / "schemas" / "target_sample.data.json"
        return json.loads(path.read_text(encoding="utf-8"))["ground_truth_chain"]

    def test_schema_example_chain_and_parameter_weighting(self):
        truth = self.example_chain()
        predicted = copy.deepcopy(truth)
        predicted["chain"][1]["values"]["drive"] += 3
        result = evaluate_chain(truth, predicted)
        self.assertTrue(result["structure_match"])
        self.assertAlmostEqual(result["mean_normalized_error"], (3 / 15) / 10)
        self.assertAlmostEqual(result["max_normalized_error"], 3 / 15)

    def test_structure_and_amp_mismatch_have_no_knob_score(self):
        truth = self.example_chain()
        variants = [copy.deepcopy(truth) for _ in range(3)]
        variants[0]["amp_id"] = "other_amp"
        variants[1]["chain"].reverse()
        variants[2]["chain"].pop()
        for predicted in variants:
            with self.subTest(predicted=predicted):
                result = evaluate_chain(truth, predicted)
                self.assertFalse(result["structure_match"])
                self.assertIsNone(result["mean_normalized_error"])
                self.assertIsNone(result["per_effect"])

    def test_invalid_values_not_hidden_by_topology_mismatch(self):
        truth = self.example_chain()
        predicted = copy.deepcopy(truth)
        predicted["amp_id"] = "other_amp"
        predicted["chain"][0]["values"]["ratio"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate_chain(truth, predicted)


if __name__ == "__main__":
    unittest.main()
