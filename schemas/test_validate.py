"""Check conditional fields, schema references, and invalid settings."""

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from validate import load_schemas, validate_references


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parent
        cls.schemas, cls.registry = load_schemas(directory)
        cls.data = {
            name: json.loads((directory / f"{name}.data.json").read_text(encoding="utf-8"))
            for name in cls.schemas
        }

    def check(self, name, value):
        Draft202012Validator(
            self.schemas[name], registry=self.registry, format_checker=FormatChecker()
        ).validate(value)

    def test_examples(self):
        for name, value in self.data.items():
            self.check(name, value)
        validate_references(self.data)

    def test_synthetic_requires_ground_truth(self):
        for field in ("dry_sample_id", "ground_truth_chain"):
            for replacement in ("missing", None):
                with self.subTest(field=field, replacement=replacement):
                    target = copy.deepcopy(self.data["target_sample"])
                    if replacement == "missing":
                        del target[field]
                    else:
                        target[field] = replacement
                    with self.assertRaises(ValidationError):
                        self.check("target_sample", target)

    def test_real_target_requires_source(self):
        target = {"target_id": "real", "is_synthetic": False, "file_path": "example.wav"}
        with self.assertRaises(ValidationError):
            self.check("target_sample", target)
        target["source_dataset"] = "example"
        self.check("target_sample", target)

    def test_nested_chain_reference(self):
        target = copy.deepcopy(self.data["target_sample"])
        del target["ground_truth_chain"]["amp_id"]
        with self.assertRaises(ValidationError):
            self.check("target_sample", target)

    def test_amp_constraints(self):
        for update in ({"fixed": False}, {"render_method": "impulse_response"}, {"render_method": "nam_capture"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                self.check("amp", self.data["amp"] | update)

    def test_inventory_constraints(self):
        for update in ({"owned_effect_types": ["fuzz", "fuzz"]}, {"updated_at": "yesterday"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                self.check("user_inventory", self.data["user_inventory"] | update)

    def test_cross_references_and_parameter_values(self):
        mutations = [
            lambda d: d["chain"].update(amp_id="unknown"),
            lambda d: d["chain"]["chain"][0].update(effect_type="unknown"),
            lambda d: d["chain"]["chain"][0]["values"].update(ratio=100),
            lambda d: d["chain"]["chain"][0]["values"].update(ratio=True),
            lambda d: d["chain"]["chain"][0]["values"].pop("ratio"),
            lambda d: d["target_sample"].update(dry_sample_id="unknown"),
            lambda d: d["user_inventory"]["owned_effect_types"].append("unknown"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaises((ValueError, ValidationError)):
                    validate_references(data)


if __name__ == "__main__":
    unittest.main()
