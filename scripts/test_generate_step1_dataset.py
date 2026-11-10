"""Small, local-only fixtures; no project/Drive audio is opened or modified."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from effects import render_chain
from schemas.validate import validate_chain
from scripts import generate_step1_dataset as step1


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.generator = step1.Generator()

    def make(self, comp, eq, drives=("overdrive",)):
        policy = (step1.Stage(("compressor",), comp), step1.Stage(drives),
                  step1.Stage(("eq_3band",), eq))
        return step1.Generator(step1.Config(effects=("compressor", *drives, "eq_3band"), policy=policy))

    def test_drive_only_and_optional_effects_omitted(self):
        chain = self.make(0, 0).chain("source", 0)
        self.assertEqual([x["effect_type"] for x in chain["chain"]], ["overdrive"])

    def test_compressor_and_drive(self):
        self.assertEqual([x["effect_type"] for x in self.make(1, 0).chain("a", 0)["chain"]],
                         ["compressor", "overdrive"])

    def test_drive_and_eq(self):
        self.assertEqual([x["effect_type"] for x in self.make(0, 1).chain("a", 0)["chain"]],
                         ["overdrive", "eq_3band"])

    def test_compressor_drive_eq(self):
        self.assertEqual([x["effect_type"] for x in self.make(1, 1).chain("a", 0)["chain"]],
                         ["compressor", "overdrive", "eq_3band"])

    def test_each_drive_can_be_selected(self):
        for name in ("overdrive", "distortion", "fuzz"):
            with self.subTest(name=name):
                self.assertEqual(self.make(0, 0, (name,)).chain("a", 0)["chain"][0]["effect_type"], name)

    def test_default_policy_randomizes_all_drives_and_four_shapes(self):
        chains = [self.generator.chain("source", i)["chain"] for i in range(160)]
        shapes = {tuple("drive" if x["effect_type"] in ("overdrive", "distortion", "fuzz")
                        else x["effect_type"] for x in chain) for chain in chains}
        self.assertEqual(shapes, {("drive",), ("compressor", "drive"),
                                  ("drive", "eq_3band"), ("compressor", "drive", "eq_3band")})
        self.assertEqual({x["effect_type"] for chain in chains for x in chain}, set(step1.DEFAULT_EFFECTS))

    def test_all_parameters_in_schema_ranges(self):
        for i in range(120):
            chain = self.generator.chain("dry", i)
            validate_chain(chain, self.generator.effects, self.generator.amp)
            for item in chain["chain"]:
                params = self.generator.effects[item["effect_type"]]["params"]
                self.assertEqual(set(item["values"]), {p["name"] for p in params})
                for p in params:
                    value = item["values"][p["name"]]
                    if p["type"] == "continuous":
                        self.assertTrue(math.isfinite(value))
                        self.assertLessEqual(p["range"][0], value)
                        self.assertLessEqual(value, p["range"][1])
                    elif p["type"] == "categorical":
                        self.assertIn(value, p["options"])

    def test_unrendered_fuzz_mode_is_fixed_and_warned(self):
        g = self.make(0, 0, ("fuzz",))
        self.assertTrue(any("fuzz.mode" in message for message in g.warnings))
        self.assertEqual({g.chain("dry", i)["chain"][0]["values"]["mode"] for i in range(25)}, {"Vintage"})

    def test_continuous_linear_sampling(self):
        param = {"type": "continuous", "range": [-12, 12]}
        self.assertEqual(step1.sample_parameter(param, random.Random(9)), random.Random(9).uniform(-12, 12))

    def test_logarithmic_sampling(self):
        param = {"type": "continuous", "range": [0.1, 100], "scale": "log"}
        expected = math.exp(random.Random(9).uniform(math.log(.1), math.log(100)))
        self.assertEqual(step1.sample_parameter(param, random.Random(9)), expected)

    def test_categorical_sampling_selects_all_options(self):
        param = {"type": "categorical", "options": ["A", "B", "C"]}
        rng = random.Random(17)
        self.assertEqual({step1.sample_parameter(param, rng) for _ in range(100)}, {"A", "B", "C"})

    def test_boolean_sampling(self):
        rng = random.Random(17)
        self.assertEqual({step1.sample_parameter({"type": "boolean"}, rng) for _ in range(100)}, {True, False})

    def test_same_seed_reproduces_chain(self):
        other = step1.Generator()
        self.assertEqual([self.generator.chain("a", i) for i in range(15)],
                         [other.chain("a", i) for i in range(15)])

    def test_different_seeds_produce_diversity(self):
        other = step1.Generator(step1.Config(seed=999))
        self.assertNotEqual([self.generator.chain("a", i) for i in range(15)],
                            [other.chain("a", i) for i in range(15)])

    def test_global_rng_state_is_preserved(self):
        state = random.getstate()
        self.generator.chain("a", 0)
        self.assertEqual(state, random.getstate())

    def test_custom_stage_order_and_single_compressor(self):
        config = step1.Config(effects=("compressor",), policy=(step1.Stage(("compressor",)),))
        self.assertEqual([x["effect_type"] for x in step1.Generator(config).chain("a", 0)["chain"]],
                         ["compressor"])
        config = step1.Config(policy=tuple(reversed(step1.DEFAULT_POLICY)))
        g = step1.Generator(config)
        for i in range(10):
            names = [x["effect_type"] for x in g.chain("a", i)["chain"]]
            if "eq_3band" in names:
                self.assertEqual(names[0], "eq_3band")

    def test_invalid_configuration_rejected(self):
        invalid = [
            replace(step1.Config(), variations=value) for value in (0, -1, True, 1.5)
        ] + [replace(step1.Config(), seed=value) for value in (True, 1.5, "42")]
        invalid += [replace(step1.Config(), effects=names) for names in
                    ((), ("missing",), ("fuzz", "fuzz"), ("compressor",))]
        invalid += [replace(step1.Config(), source_dataset=""),
                    replace(step1.Config(), policy=()),
                    replace(step1.Config(), policy=(step1.Stage(("missing",)),)),
                    replace(step1.Config(), policy=(step1.Stage(step1.DEFAULT_EFFECTS, .5),)),
                    replace(step1.Config(), policy=(step1.Stage(step1.DEFAULT_EFFECTS),
                                                   step1.Stage(("fuzz",))))]
        for value in (-.1, 1.1, float("nan"), float("inf"), True, "half"):
            invalid.append(replace(step1.Config(), policy=(step1.Stage(step1.DEFAULT_EFFECTS, value),)))
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                step1.Generator(config)

    def test_unsupported_renderer_source_rejected(self):
        original = step1.build_effect

        def unsupported(name, values, effects):
            catalog = deepcopy(effects)
            catalog[name]["source"] = "vst"
            return original(name, values, catalog)

        with patch.object(step1, "build_effect", side_effect=unsupported), self.assertRaises(NotImplementedError):
            step1.Generator()


class LocalPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="step1-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.dry = self.root / "input"
        self.dry.mkdir()
        self.sr = 16000
        t = np.arange(1600, dtype=np.float32) / self.sr
        self.wave = (.2 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)).astype(np.float32)
        sf.write(self.dry / "same.wav", self.wave, self.sr, subtype="FLOAT")
        self.inputs = [("guitar_a", self.dry)]

    def plan(self, **kwargs):
        return step1.Plan(self.inputs, **kwargs)

    def test_labels_match_schema_and_fixed_amp(self):
        plan = self.plan(config=step1.Config(variations=20))
        for _, label in plan.items():
            plan.generator.validator.validate(label)
            self.assertEqual(set(label), {"target_id", "is_synthetic", "dry_sample_id",
                                         "ground_truth_chain", "source_dataset", "file_path"})
            self.assertEqual(label["ground_truth_chain"]["amp_id"], "fixed_amp_v1")

    def test_same_settings_reproduce_labels(self):
        self.assertEqual(list(self.plan().items()), list(self.plan().items()))

    def test_variation_count_and_dry_grouping(self):
        short = [label for _, label in self.plan(config=step1.Config(variations=2)).items()]
        long = [label for _, label in self.plan(config=step1.Config(variations=7)).items()]
        self.assertEqual(len(long), 7)
        self.assertEqual(short, long[:2])
        self.assertEqual(len({label["dry_sample_id"] for label in long}), 1)
        self.assertEqual(len({label["target_id"] for label in long}), 7)

    def test_same_basename_in_multiple_guitars_has_unique_ids(self):
        other = self.root / "other"
        other.mkdir()
        sf.write(other / "same.wav", self.wave, self.sr)
        labels = [label for _, label in step1.Plan(self.inputs + [("guitar_b", other)]).items()]
        self.assertEqual(len({label["dry_sample_id"] for label in labels}), 2)
        self.assertEqual(len({label["target_id"] for label in labels}), 2)

    def test_nested_same_basenames_have_unique_ids(self):
        nested = self.dry / "nested"
        nested.mkdir()
        sf.write(nested / "same.WAV", self.wave, self.sr)
        self.assertEqual(len({s.dry_id for s in self.plan().sources}), 2)

    def test_input_order_does_not_change_labels(self):
        other = self.root / "other"
        other.mkdir()
        sf.write(other / "same.wav", self.wave, self.sr)
        inputs = self.inputs + [("guitar_b", other)]
        self.assertEqual(list(step1.Plan(inputs).items()), list(step1.Plan(list(reversed(inputs))).items()))

    def test_settings_change_target_ids(self):
        old = next(self.plan().items())[1]
        new = next(self.plan(config=step1.Config(seed=43)).items())[1]
        self.assertEqual(old["dry_sample_id"], new["dry_sample_id"])
        self.assertNotEqual(old["target_id"], new["target_id"])

    def test_duplicate_names_and_overlapping_inputs_rejected(self):
        for inputs in (self.inputs + [("guitar_a", self.dry)],
                       self.inputs + [("GUITAR_A", self.dry)],
                       self.inputs + [("alias", self.dry)],
                       self.inputs + [("parent", self.root)],
                       [("../bad", self.dry)]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                step1.Plan(inputs)

    def test_hardlinked_input_aliases_rejected(self):
        other = self.root / "other"
        other.mkdir()
        os.link(self.dry / "same.wav", other / "alias.wav")
        with self.assertRaisesRegex(ValueError, "same physical"):
            step1.Plan(self.inputs + [("guitar_b", other)])

    def test_missing_empty_and_file_inputs_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        for inputs in ([], [("a", empty)], [("a", self.root / "missing")],
                       [("a", self.dry / "same.wav")]):
            with self.subTest(inputs=inputs), self.assertRaises((ValueError, OSError)):
                step1.Plan(inputs)

    def test_dry_run_creates_nothing_and_does_not_read_audio(self):
        before = sorted(self.root.rglob("*"))
        out = self.root / "preview-output"
        with patch.object(sf, "read", side_effect=AssertionError("dry-run read audio")), \
                patch.object(step1, "render_chain", side_effect=AssertionError("dry-run rendered")), \
                redirect_stdout(io.StringIO()) as stdout:
            step1.main(["--dry-input", "guitar_a", str(self.dry), "--output", str(out)])
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["mode"], "dry-run")
        self.assertEqual(summary["total_output_files"], 3)
        self.assertEqual(before, sorted(self.root.rglob("*")))

    def test_explicit_dry_run_without_output(self):
        with redirect_stdout(io.StringIO()) as stream:
            step1.main(["--dry-input", "guitar_a", str(self.dry), "--dry-run", "--variations", "4"])
        result = json.loads(stream.getvalue())
        self.assertEqual(result["pairs"], 4)
        self.assertTrue(result["output"].startswith("<OUTPUT>"))

    def test_execute_requires_output(self):
        with self.assertRaises(ValueError):
            self.plan().execute()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step1.main(["--dry-input", "a", str(self.dry), "--execute"])

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "keep.json"
        sentinel.write_text("unchanged", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.plan(output=output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_output_created_after_plan_is_rejected(self):
        plan = self.plan(output=self.root / "output")
        plan.output.mkdir()
        with self.assertRaises(FileExistsError):
            plan.execute()
        self.assertEqual(list(plan.output.iterdir()), [])

    def test_existing_wav_appearing_during_render_is_not_overwritten(self):
        plan = self.plan(output=self.root / "out")
        label = next(plan.items())[1]
        path = Path(label["file_path"])

        def occupied(audio, sr, chain, effects):
            path.write_bytes(b"existing WAV sentinel")
            return render_chain(audio, sr, chain, effects)

        with patch.object(step1, "render_chain", side_effect=occupied), self.assertRaises(FileExistsError):
            plan.execute()
        self.assertEqual(path.read_bytes(), b"existing WAV sentinel")
        self.assertEqual(list((plan.output / "labels").iterdir()), [])

    def test_existing_json_appearing_during_render_is_not_overwritten(self):
        plan = self.plan(output=self.root / "out")
        label = next(plan.items())[1]
        path = plan.output / "labels" / (label["target_id"] + ".json")

        def occupied(audio, sr, chain, effects):
            path.write_text("existing JSON sentinel", encoding="utf-8")
            return render_chain(audio, sr, chain, effects)

        with patch.object(step1, "render_chain", side_effect=occupied), self.assertRaises(FileExistsError):
            plan.execute()
        self.assertEqual(path.read_text("utf-8"), "existing JSON sentinel")
        self.assertFalse((plan.output / "manifest.json").exists())

    def test_project_data_and_shared_paths_rejected(self):
        paths = [step1.ROOT / "data/raw/result", step1.ROOT / "data/dry/Fender_Strat_Clean/audio/result",
                 step1.ROOT / "data/synthetic/Fender_Strat_Clean/result",
                 self.root / "Google Drive/result", self.root / "HAI_2026-2_project2_drive/result",
                 self.root / "data/synthetic/result", self.root / ".shortcut-targets-by-id/id/result",
                 self.root / "OneDrive/result"]
        for path in paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.plan(output=path)

    def make_link(self, link, target):
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_real_junction_or_symlink_ancestor_is_rejected(self):
        target = self.root / "physical"
        target.mkdir()
        link = self.root / "innocent-name"
        self.make_link(link, target)
        with self.assertRaisesRegex(ValueError, "link/junction/reparse"):
            self.plan(output=link / "new" / "nested")
        self.assertEqual(list(target.iterdir()), [])

    def test_junction_inserted_after_plan_is_rejected(self):
        plan = self.plan(output=self.root / "parent" / "out")
        target = self.root / "physical"
        target.mkdir()
        self.make_link(self.root / "parent", target)
        with self.assertRaisesRegex(ValueError, "link/junction/reparse"):
            plan.execute()
        self.assertEqual(list(target.iterdir()), [])

    def test_output_inside_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside every input"):
            self.plan(output=self.dry / "out")

    def test_plan_reports_counts_output_and_sources(self):
        summary = self.plan(config=step1.Config(variations=7), output=self.root / "out").summary()
        self.assertEqual((summary["wav_files"], summary["label_files"], summary["total_output_files"]),
                         (7, 7, 15))
        self.assertEqual(len(summary["preview_labels"]), 3)
        self.assertEqual(summary["dry_sources"], 1)
        self.assertIn(str(self.root / "out"), summary["output"])

    def test_rendered_chain_equals_label_and_audio_roundtrip(self):
        plan = self.plan(config=step1.Config(variations=3), output=self.root / "out")
        with patch.object(step1, "render_chain", wraps=render_chain) as renderer:
            manifest = plan.execute()
        labels = [label for _, label in plan.items()]
        self.assertEqual(renderer.call_count, 3)
        dry, sr = sf.read(self.dry / "same.wav", dtype="float32", always_2d=True)
        for call, label in zip(renderer.call_args_list, labels):
            self.assertEqual(call.args[2], label["ground_truth_chain"])
            stored = json.loads((plan.output / "labels" / (label["target_id"] + ".json")).read_text("utf-8"))
            self.assertEqual(stored, label)
            rendered, out_sr = sf.read(label["file_path"], dtype="float32", always_2d=True)
            expected = render_chain(dry.T.copy(), sr, label["ground_truth_chain"])
            np.testing.assert_array_equal(rendered, expected.T)
            self.assertEqual(sr, out_sr)
        self.assertEqual(manifest["completed_pairs"], 3)
        self.assertEqual(len(manifest["dry_id_map"]), 1)
        self.assertEqual(len(list(plan.output.rglob("*.*"))), 7)

    def test_real_renderer_all_combinations_and_drives(self):
        for drive in ("overdrive", "distortion", "fuzz"):
            for comp, eq in ((0, 0), (1, 0), (0, 1), (1, 1)):
                with self.subTest(drive=drive, compressor=comp, eq=eq):
                    policy = (step1.Stage(("compressor",), comp), step1.Stage((drive,)),
                              step1.Stage(("eq_3band",), eq))
                    config = step1.Config(effects=("compressor", drive, "eq_3band"), policy=policy)
                    out = self.root / f"{drive}-{comp}-{eq}"
                    self.assertEqual(self.plan(config=config, output=out).execute()["completed_pairs"], 1)

    def test_stereo_channel_layout_preserved(self):
        stereo = np.column_stack([self.wave, .5 * self.wave])
        sf.write(self.dry / "same.wav", stereo, self.sr, subtype="FLOAT")
        plan = self.plan(output=self.root / "out")
        plan.execute()
        label = next(plan.items())[1]
        output, sr = sf.read(label["file_path"], dtype="float32", always_2d=True)
        np.testing.assert_array_equal(output, render_chain(stereo.T, sr, label["ground_truth_chain"]).T)
        self.assertEqual(output.shape, stereo.shape)

    def test_nonfinite_renderer_output_rejected(self):
        plan = self.plan(output=self.root / "out")
        with patch.object(step1, "render_chain", return_value=np.full((1, 1600), np.nan)), \
                self.assertRaisesRegex(ValueError, "renderer returned"):
            plan.execute()
        self.assertEqual(list((plan.output / "audio").iterdir()), [])

    def test_input_audio_is_unchanged(self):
        before = (self.dry / "same.wav").read_bytes()
        self.plan(output=self.root / "out").execute()
        self.assertEqual(before, (self.dry / "same.wav").read_bytes())

    def test_cli_custom_policy_and_drive_subset(self):
        with redirect_stdout(io.StringIO()) as stream:
            step1.main(["--dry-input", "guitar_a", str(self.dry), "--effects", "distortion", "eq_3band",
                        "--eq-probability", "1", "--chain-order", "eq", "drive", "compressor"])
        chain = json.loads(stream.getvalue())["preview_labels"][0]["ground_truth_chain"]["chain"]
        self.assertEqual([x["effect_type"] for x in chain], ["eq_3band", "distortion"])

    def test_cli_explicit_local_execution(self):
        output = self.root / "cli-out"
        with redirect_stdout(io.StringIO()) as stream:
            step1.main(["--dry-input", "test", str(self.dry), "--output", str(output),
                        "--variations", "2", "--execute"])
        preview, result = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(preview["mode"], "execute")
        self.assertEqual(result["completed_pairs"], 2)
        self.assertTrue((output / "manifest.json").is_file())

    def test_rendering_is_reproducible_across_local_runs(self):
        first = self.plan(config=step1.Config(variations=3), output=self.root / "first")
        second = self.plan(config=step1.Config(variations=3), output=self.root / "second")
        first.execute()
        second.execute()
        for (_, a), (_, b) in zip(first.items(), second.items()):
            self.assertEqual(a["ground_truth_chain"], b["ground_truth_chain"])
            self.assertEqual(Path(a["file_path"]).read_bytes(), Path(b["file_path"]).read_bytes())

    def test_cli_invalid_settings_create_nothing(self):
        before = sorted(self.root.rglob("*"))
        cases = [["--variations", "0"], ["--effects", "unknown"],
                 ["--compressor-probability", "nan"], ["--eq-probability", "2"],
                 ["--chain-order", "drive", "drive", "eq"], ["--execute", "--dry-run"]]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                step1.main(["--dry-input", "a", str(self.dry), *args])
        self.assertEqual(before, sorted(self.root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
