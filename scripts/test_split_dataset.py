"""Group-level split checks using virtual labels; no WAV/library dependency."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from scripts.split_dataset import (
    SPLITS, group_counts, label_ids, load_labels, split_labels, validate_splits, write_splits,
)


ROOT = Path(__file__).resolve().parents[1]


def virtual_labels(groups=20, variations=3):
    labels = []
    for group in range(groups):
        for variant in range(variations):
            sample_id, dry_id = f"wet_{group:03d}_{variant:03d}", f"dry_{group:03d}"
            if variant % 2:
                label = {"synthetic_id": sample_id, "dry_sample": {"sample_id": dry_id},
                         "applied_effects": {"future_effect": {"parameters": {"gain": variant}}}}
            else:
                label = {"target_id": sample_id, "dry_sample_id": dry_id,
                         "ground_truth_chain": {"chain": [{"effect_type": "unknown_future_effect"}]}}
            label["file_path"] = f"nonexistent_audio/{sample_id}.wav"
            labels.append(label)
    return labels


class SplitDatasetTests(unittest.TestCase):
    def test_default_group_ratios_and_every_label_exactly_once(self):
        labels = virtual_labels(100, 3)
        parts = split_labels(labels)
        summary = validate_splits(labels, parts)
        self.assertEqual(summary["group_counts"], {"train": 80, "val": 10, "test": 10})
        self.assertEqual(summary["label_counts"], {"train": 240, "val": 30, "test": 30})
        output_ids = [label_ids(label)[0] for name in SPLITS for label in parts[name]]
        self.assertEqual(len(output_ids), len(set(output_ids)))
        self.assertEqual(set(output_ids), {label_ids(label)[0] for label in labels})

    def test_all_variants_of_each_dry_are_in_only_one_split(self):
        labels = virtual_labels(78, 9)
        parts = split_labels(labels)
        ownership = {}
        for name in SPLITS:
            for label in parts[name]:
                _, dry_id = label_ids(label)
                ownership.setdefault(dry_id, set()).add(name)
        self.assertEqual(len(ownership), 78)
        self.assertTrue(all(len(names) == 1 for names in ownership.values()))
        summary = validate_splits(labels, parts)
        self.assertEqual(summary["group_counts"], {"train": 62, "val": 8, "test": 8})
        self.assertEqual(summary["label_counts"], {"train": 558, "val": 72, "test": 72})

    def test_same_seed_and_reordered_input_give_identical_lists(self):
        labels = virtual_labels(40)
        reordered = deepcopy(labels)
        random.Random(901).shuffle(reordered)
        self.assertEqual(split_labels(labels, seed=42), split_labels(reordered, seed=42))
        self.assertEqual(split_labels(labels, seed=42), split_labels(list(reversed(labels)), seed=42))

    def test_other_seed_changes_membership_but_not_counts(self):
        labels = virtual_labels(100)
        first, second = split_labels(labels, seed=42), split_labels(labels, seed=43)
        self.assertNotEqual(first, second)
        self.assertEqual(validate_splits(labels, first)["group_counts"],
                         validate_splits(labels, second)["group_counts"])

    def test_global_random_state_is_preserved(self):
        before = random.getstate()
        split_labels(virtual_labels())
        self.assertEqual(random.getstate(), before)

    def test_fraction_and_weight_ratios_are_equivalent(self):
        labels = virtual_labels(31)
        self.assertEqual(split_labels(labels, ratios=(80, 10, 10)),
                         split_labels(labels, ratios=(0.8, 0.1, 0.1)))

    def test_custom_and_zero_ratios(self):
        labels = virtual_labels(100)
        parts = split_labels(labels, ratios=(60, 20, 20))
        self.assertEqual(validate_splits(labels, parts, (60, 20, 20))["group_counts"],
                         {"train": 60, "val": 20, "test": 20})
        parts = split_labels(labels, ratios=(0, 1, 0))
        summary = validate_splits(labels, parts, (0, 1, 0))
        self.assertEqual(summary["group_counts"], {"train": 0, "val": 100, "test": 0})
        self.assertEqual(summary["warnings"], [])

    def test_small_dataset_warns_about_empty_positive_splits(self):
        labels = virtual_labels(1)
        summary = validate_splits(labels, split_labels(labels))
        self.assertEqual(summary["group_counts"], {"train": 1, "val": 0, "test": 0})
        self.assertEqual(len(summary["warnings"]), 2)

    def test_rounding_and_tie_rule_are_explicit(self):
        self.assertEqual(group_counts(4, (1, 1, 1)), {"train": 2, "val": 1, "test": 1})
        self.assertEqual(group_counts(5, (1, 1, 1)), {"train": 2, "val": 2, "test": 1})
        self.assertEqual(group_counts(7, (0, 1, 1)), {"train": 0, "val": 4, "test": 3})

    def test_many_group_counts_have_no_rounding_loss(self):
        for count in range(150):
            for weights in ((80, 10, 10), (3, 2, 1), (0, 1, 1)):
                allocation = group_counts(count, weights)
                self.assertEqual(sum(allocation.values()), count)
                for name, weight in zip(SPLITS, weights):
                    ideal = Fraction(count * weight, sum(weights))
                    self.assertLess(abs(allocation[name] - ideal), 1)

    def test_unequal_variations_report_label_ratio_separately(self):
        labels = [label for group in range(10)
                  for label in virtual_labels(10, group + 1) if label_ids(label)[1] == f"dry_{group:03d}"]
        summary = validate_splits(labels, split_labels(labels))
        self.assertEqual(summary["total_labels"], 55)
        self.assertEqual(summary["group_counts"], {"train": 8, "val": 1, "test": 1})
        self.assertEqual(summary["actual_group_ratios"]["val"], 0.1)
        self.assertNotEqual(summary["actual_label_ratios"]["val"], 0.1)

    def test_two_formats_and_matching_aliases_share_one_group(self):
        labels = [{"target_id": "one", "dry_sample_id": "same"},
                  {"synthetic_id": "two", "dry_sample": {"sample_id": "same"}},
                  {"target_id": "three", "synthetic_id": "three", "dry_sample_id": "same",
                   "dry_sample": {"sample_id": "same"}}]
        parts = split_labels(labels)
        self.assertEqual(len(parts["train"]), 3)
        self.assertEqual(validate_splits(labels, parts)["total_dry_sources"], 1)

    def test_missing_invalid_and_conflicting_identifiers_are_rejected(self):
        good = {"target_id": "wet", "dry_sample_id": "dry"}
        invalid = [{"target_id": "wet"}, {"dry_sample_id": "dry"},
                   {**good, "synthetic_id": "different"},
                   {**good, "dry_sample": {"sample_id": "different"}},
                   {**good, "dry_sample": None}, {**good, "dry_sample": {}},
                   {**good, "dry_sample_id": None}, {**good, "dry_sample_id": " dry"},
                   {**good, "dry_sample_id": ""}, {**good, "target_id": 1}, 42]
        for label in invalid:
            with self.subTest(label=label), self.assertRaises(ValueError):
                split_labels([label])

    def test_duplicate_input_samples_are_not_silently_deduplicated(self):
        for second in ({"target_id": "wet", "dry_sample_id": "dry"},
                       {"synthetic_id": "wet", "dry_sample": {"sample_id": "other"}}):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                split_labels([{"target_id": "wet", "dry_sample_id": "dry"}, second])

    def test_empty_and_nonfinite_label_data_are_rejected(self):
        with self.assertRaises(ValueError):
            split_labels([])
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                split_labels([{"target_id": "wet", "dry_sample_id": "dry", "parameter": value}])

    def test_invalid_ratios_and_seeds_are_rejected(self):
        for ratios in ((0, 0, 0), (-1, 1, 1), (1,), (1, 1, 1, 1), (True, 1, 1),
                       (float("nan"), 1, 1), (float("inf"), 1, 1), None):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_labels(virtual_labels(), ratios=ratios)
        for seed in (True, "42", 42.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                split_labels(virtual_labels(), seed=seed)
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                group_counts(count)

    def test_input_and_arbitrary_ten_effect_payloads_are_preserved(self):
        labels = virtual_labels(10)
        for label in labels:
            label["future_chain"] = [{"effect": f"effect_{i}", "params": {"unknown": [i, True, None]}}
                                     for i in range(10)]
        before = deepcopy(labels)
        parts = split_labels(labels)
        self.assertEqual(labels, before)
        self.assertTrue(validate_splits(labels, parts)["validation"]["labels_preserved"])
        parts["train"][0]["future_chain"][0]["params"]["unknown"][0] = -1
        self.assertEqual(labels, before)

    def test_more_effect_variants_keep_existing_dry_assignments(self):
        small, expanded = virtual_labels(30, 3), virtual_labels(30, 10)
        def ownership(parts):
            return {label_ids(label)[1]: name for name in SPLITS for label in parts[name]}
        self.assertEqual(ownership(split_labels(small)), ownership(split_labels(expanded)))

    def test_validator_detects_group_leakage(self):
        labels = virtual_labels(20)
        parts = split_labels(labels)
        parts["val"].append(parts["train"].pop(0))
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_splits(labels, parts)

    def test_validator_detects_duplicates_missing_unknown_and_modified_labels(self):
        labels = virtual_labels(20)
        for kind in ("duplicate", "missing", "unknown", "modified"):
            parts = split_labels(labels)
            if kind == "duplicate":
                parts["val"].append(deepcopy(parts["train"][0]))
            elif kind == "missing":
                parts["train"].pop()
            elif kind == "unknown":
                parts["train"].append({"target_id": "unknown", "dry_sample_id": "unknown"})
            else:
                parts["train"][0]["unrequested_change"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_splits(labels, parts)

    def test_validator_detects_wrong_group_counts_without_leakage(self):
        labels = virtual_labels(10)
        parts = split_labels(labels)
        parts["train"].extend(parts["val"])
        parts["val"] = []
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_splits(labels, parts)

    def test_validator_rejects_wrong_partition_names_and_types(self):
        labels = virtual_labels()
        parts = split_labels(labels)
        for bad in ({"train": parts["train"]}, {**parts, "extra": []}, {**parts, "val": None}):
            with self.assertRaises(ValueError):
                validate_splits(labels, bad)

    def test_loads_recursive_json_objects_arrays_bom_and_uppercase_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "nested").mkdir()
            labels = virtual_labels(1)
            (folder / "a.JSON").write_text(json.dumps(labels[0]), encoding="utf-8-sig")
            (folder / "nested" / "b.json").write_text(json.dumps(labels[1:]), encoding="utf-8")
            (folder / "notes.txt").write_text("not a label", encoding="utf-8")
            self.assertEqual(load_labels(folder), labels)

    def test_loader_rejects_empty_invalid_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaises(ValueError):
                load_labels(folder)
            path = folder / "bad.json"
            for text in ('{"target_id":"a","target_id":"b","dry_sample_id":"d"}',
                         '{"target_id":"a","dry_sample_id":"d","value":NaN}',
                         '{"target_id":"a","dry_sample_id":"d","value":Infinity}',
                         "broken", "42", "[42]"):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_labels(path)

    def test_loader_rejects_duplicate_sample_ids_across_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            label = virtual_labels(1, 1)[0]
            for name in ("a.json", "b.json"):
                (folder / name).write_text(json.dumps(label), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_labels(folder)

    def test_schema_example_can_be_split_without_opening_its_wav(self):
        labels = load_labels(ROOT / "schemas" / "target_sample.data.json")
        summary = validate_splits(labels, split_labels(labels))
        self.assertEqual(summary["total_labels"], 1)
        self.assertEqual(summary["dry_sample_ids"]["train"], ["dry_example_001"])

    def test_writer_is_reproducible_and_keeps_sources_and_audio_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "한글 labels with spaces"
            folder.mkdir()
            source, audio = folder / "labels.json", folder / "original.wav"
            source.write_text(json.dumps(virtual_labels(10)), encoding="utf-8")
            audio.write_bytes(b"sentinel: not decoded by splitter")
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (source, audio)}
            first, second = folder / "one", folder / "two"
            summary = write_splits(load_labels(source), first, input_source=source)
            write_splits(load_labels(source), second, input_source=source)
            self.assertEqual({p.name for p in first.iterdir()}, {"train.json", "val.json", "test.json", "summary.json"})
            for p in first.iterdir():
                self.assertEqual(p.read_bytes(), (second / p.name).read_bytes())
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (source, audio)})
            parts = {name: json.loads((first / f"{name}.json").read_text(encoding="utf-8")) for name in SPLITS}
            self.assertEqual(validate_splits(load_labels(source), parts)["group_counts"], summary["group_counts"])

    def test_existing_output_and_input_directories_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            existing = folder / "existing"
            existing.mkdir()
            sentinel = existing / "train.json"
            sentinel.write_text("leave unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_splits(virtual_labels(), existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "leave unchanged")
            labels_dir = folder / "labels"
            labels_dir.mkdir()
            with self.assertRaises(ValueError):
                write_splits(virtual_labels(), labels_dir / "new-output", input_source=labels_dir)
            self.assertFalse((labels_dir / "new-output").exists())

    def test_validation_errors_create_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new" / "output"
            with self.assertRaises(ValueError):
                write_splits([{"target_id": "no-dry-id"}], output)
            self.assertFalse(output.parent.exists())

    def test_cli_accepts_virtual_labels_and_custom_ratios_without_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, output = folder / "input.json", folder / "new-output"
            source.write_text(json.dumps(virtual_labels(20, 4)), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "scripts.split_dataset", "--input", str(source),
                                     "--output", str(output), "--ratios", "60", "20", "20", "--seed", "7"],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["group_counts"], {"train": 12, "val": 4, "test": 4})
            self.assertEqual(summary["total_labels"], 80)

    def test_cli_rejects_bad_input_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, output = folder / "input.json", folder / "new-output"
            label = virtual_labels(1, 1)[0]
            source.write_text(json.dumps([label, label]), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "scripts.split_dataset", "--input", str(source),
                                     "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Duplicate", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
