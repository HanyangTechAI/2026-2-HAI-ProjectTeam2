"""Preview random effect chains; explicit execution writes only to local temp.

Does not import Create_Synthetic_pairs (that module creates directories on import).
See STEP1_DATA_GENERATION.md. Run: python -m scripts.generate_step1_dataset --help
"""

import argparse
from copy import deepcopy
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jsonschema import Draft202012Validator

from effects import build_effect, render_chain
from effects.audio_io import STANDARD_SAMPLE_RATE, read_mono
from schemas.validate import load_schemas, validate_chain, validate_effects


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "step1-schema-random-v2"
# Step 1 드라이브: MultiDrive PRO 7종 (pedalboard overdrive/distortion/fuzz는 대체용으로 --effects로 선택 가능)
DRIVES = ("808_scream", "b_drive", "o_drive", "one_distortion", "bat_distortion", "f_fuzz", "m_fuzz")
DEFAULT_EFFECTS = ("compressor", *DRIVES, "eq_3band")
SOURCE_DATASET = "IDMT-SMT-Guitar V2"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stage:
    """Choose at most one of effect_types, with the given inclusion probability."""

    effect_types: tuple
    probability: float = 1.0


DEFAULT_POLICY = (
    Stage(("compressor",), 0.5),
    Stage(DRIVES),
    Stage(("eq_3band",), 0.5),
)


@dataclass(frozen=True)
class Config:
    variations: int = 1  # variants per dry source; with per_drive, per dry source AND per drive
    seed: int = 42
    effects: tuple = DEFAULT_EFFECTS
    policy: tuple = DEFAULT_POLICY
    source_dataset: str = SOURCE_DATASET
    per_drive: bool = False  # True: every enabled drive gets `variations` chains per dry source
    dry_peak_db: float = None  # set: scale each dry to this peak (dBFS) before rendering; wet is untouched


def sample_parameter(param, rng):
    """Schema-validated continuous, categorical and boolean parameter sampling."""
    if param["type"] == "continuous":
        low, high = param["range"]
        if param.get("scale", "linear") == "log":
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
            return min(high, max(low, value))  # floating point endpoint rounding
        return rng.uniform(low, high)
    if param["type"] == "categorical":
        return rng.choice(param["options"])
    if param["type"] == "boolean":
        return bool(rng.getrandbits(1))
    raise ValueError(f"Unsupported parameter type: {param['type']}")


def _mapped(definition, param):
    classes = (definition["composite_of"] if definition["class_name"] == "composite"
               else [definition["class_name"]])
    cls, _, attr = param["maps_to"].partition(".")
    return cls in classes and bool(attr)


class Generator:
    def __init__(self, config=Config()):
        self.config = config
        self.schemas, registry = load_schemas(ROOT / "schemas")
        self.effects = json.loads((ROOT / "schemas/effects.data.json").read_text(encoding="utf-8"))
        self.amp = json.loads((ROOT / "schemas/amp.data.json").read_text(encoding="utf-8"))
        Draft202012Validator(self.schemas["effects"]).validate(self.effects)
        validate_effects(self.effects)
        self.validator = Draft202012Validator(self.schemas["target_sample"], registry=registry)
        self._validate_config()
        self.warnings = [
            "Temporary experiment policy; team has not finalized chain order/probabilities.",
            f"amp '{self.amp['amp_id']}' ({self.amp['render_method']}) is applied after the chain by the team renderer.",
            "No loudness rejection or clipping correction is performed on wet audio.",
        ]
        self.warnings.append(f"Dry is converted to mono {STANDARD_SAMPLE_RATE} Hz before rendering (saved in dry/).")
        if config.dry_peak_db is None:
            self.warnings.append("Dry audio is rendered at its original level (no --dry-peak-db).")
        for name in config.effects:
            definition = self.effects[name]
            defaults = {p["name"]: p["default"] for p in definition["params"]}
            # Use the team's own builder to reject unsupported sources/classes/knobs.
            # VST plugins are only loaded when rendering, so previews work without the plugin.
            if definition["source"] != "vst":
                build_effect(name, defaults, self.effects)
            for param in definition["params"]:
                if not _mapped(definition, param):
                    self.warnings.append(
                        f"{name}.{param['name']} is not rendered; fixed to {param['default']!r}.")
        self.settings = {
            "algorithm": ALGORITHM, "seed": config.seed,
            "effects": list(config.effects), "source_dataset": config.source_dataset,
            "policy": [{"effect_types": list(s.effect_types), "probability": s.probability}
                       for s in config.policy],
            "per_drive": config.per_drive, "dry_peak_db": config.dry_peak_db,
            "amp_id": self.amp["amp_id"],
            "effects_digest": _digest({name: self.effects[name] for name in config.effects}),
        }
        self.run_id = _digest(self.settings)

    def drives(self):
        """Enabled gain-stage (drive) effects, in policy order."""
        return [name for stage in self.config.policy for name in stage.effect_types
                if name in self.config.effects and self.effects[name]["category"] == "gain_stage"]

    def _validate_config(self):
        c = self.config
        if isinstance(c.variations, bool) or not isinstance(c.variations, int) or c.variations < 1:
            raise ValueError("variations must be a positive integer")
        if isinstance(c.seed, bool) or not isinstance(c.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(c.source_dataset, str) or not c.source_dataset.strip():
            raise ValueError("source_dataset must be a nonempty string")
        if not c.effects or len(set(c.effects)) != len(c.effects) or set(c.effects) - self.effects.keys():
            raise ValueError("effects must be unique names present in schemas/effects.data.json")
        if not c.policy or not all(isinstance(stage, Stage) for stage in c.policy):
            raise ValueError("policy must contain Stage objects")
        used, guaranteed = set(), False
        for stage in c.policy:
            if (isinstance(stage.probability, bool)
                    or not isinstance(stage.probability, (int, float))
                    or not math.isfinite(stage.probability) or not 0 <= stage.probability <= 1):
                raise ValueError("stage probability must be finite and within [0, 1]")
            choices = stage.effect_types
            if not choices or len(set(choices)) != len(choices) or set(choices) - self.effects.keys():
                raise ValueError("stage choices must be unique schema effect names")
            if used.intersection(choices):
                raise ValueError("an effect may occur in only one stage")
            used.update(choices)
            active = set(choices).intersection(c.effects)
            guaranteed |= bool(active) and stage.probability == 1
        if set(c.effects) - used:
            raise ValueError("every enabled effect must belong to a policy stage")
        if not guaranteed:
            raise ValueError("policy must guarantee at least one enabled effect")
        if not isinstance(c.per_drive, bool):
            raise ValueError("per_drive must be a boolean")
        if c.per_drive:
            drive_stages = [s for s in c.policy if any(
                name in c.effects and self.effects[name]["category"] == "gain_stage" for name in s.effect_types)]
            if len(drive_stages) != 1 or drive_stages[0].probability != 1:
                raise ValueError("per_drive needs exactly one always-on stage with the enabled drives")
        if c.dry_peak_db is not None and (isinstance(c.dry_peak_db, bool)
                                          or not isinstance(c.dry_peak_db, (int, float))
                                          or not math.isfinite(c.dry_peak_db) or c.dry_peak_db > 0):
            raise ValueError("dry_peak_db must be a finite number <= 0 (dBFS)")

    def chain(self, dry_id, variation, drive=None):
        """Random chain for one variant. drive (per_drive mode) fixes the drive stage's choice."""
        # Per-source/per-variant RNG avoids changes caused by input order or count.
        key = [self.config.seed, dry_id, variation] + ([drive] if drive is not None else [])
        rng = random.Random(_digest(key))
        chain = []
        for stage in self.config.policy:
            choices = [name for name in stage.effect_types if name in self.config.effects]
            if drive is not None and drive in choices:
                name = drive
            elif not choices or rng.random() >= stage.probability:
                continue
            else:
                name = rng.choice(choices)
            definition = self.effects[name]
            values = {p["name"]: (sample_parameter(p, rng) if _mapped(definition, p)
                                  else p["default"]) for p in definition["params"]}
            chain.append({"effect_type": name, "values": values})
        result = {"amp_id": self.amp["amp_id"], "chain": chain}
        validate_chain(result, self.effects, self.amp)
        return result


@dataclass(frozen=True)
class DrySource:
    path: Path
    dry_id: str
    namespace: str
    relative_path: str

    @property
    def name(self):
        """Readable dry name for file names: {namespace}_{relative path without extension}."""
        stem = self.relative_path.rsplit(".", 1)[0]
        return re.sub(r"[^A-Za-z0-9_-]", "-", f"{self.namespace}_{stem}")


def discover_sources(inputs):
    """inputs is a sequence of (stable namespace, WAV directory) pairs."""
    if not inputs:
        raise ValueError("at least one --dry-input NAME DIRECTORY is required")
    sources, names, roots, files = [], set(), [], set()
    for namespace, directory in inputs:
        if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", namespace):
            raise ValueError("input NAME must be 1-64 ASCII letters/digits/_/- (start with letter/digit)")
        if namespace.casefold() in names:
            raise ValueError("input names must be unique (case-insensitive)")
        names.add(namespace.casefold())
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"input must be a directory: {root}")
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots):
            raise ValueError("input directories must not overlap or alias each other")
        roots.append(root)
        count = 0
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() != ".wav" or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            info = resolved.stat()
            identity = (info.st_dev, info.st_ino) if info.st_ino else str(resolved)
            if identity in files:
                raise ValueError(f"same physical dry WAV selected more than once: {path}")
            files.add(identity)
            relative = path.relative_to(root).as_posix()
            dry_id = "dry_" + _digest([namespace, relative])
            sources.append(DrySource(resolved, dry_id, namespace, relative))
            count += 1
        if not count:
            raise ValueError(f"no WAV files in input: {root}")
    names = [source.name for source in sources]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("two dry files map to the same output name; rename one of them")
    return sorted(sources, key=lambda source: source.dry_id)


def _check_plain_path(path):
    """Reject links AND Windows junction/mount/cloud reparse points, including parents."""
    for ancestor in (path, *path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0)
                                        & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"output cannot traverse a link/junction/reparse point: {ancestor}")


def safe_output(output):
    """Development restriction: new directory strictly below a physical local temp root."""
    path = Path(os.path.abspath(Path(output).expanduser()))
    temp = Path(os.path.abspath(tempfile.gettempdir()))
    _check_plain_path(path)
    _check_plain_path(temp)
    resolved, temp = path.resolve(), temp.resolve()
    if resolved == temp or not resolved.is_relative_to(temp):
        raise ValueError(f"development output must be a NEW directory under local temp: {temp}")
    markers = {"google drive", "googledrive", "drivefs", ".shortcut-targets-by-id",
               "my drive", "shared drives", "내 드라이브", "공유 드라이브", "onedrive",
               "hai_2026-2_project2_drive"}
    if any(part.casefold() in markers or part.casefold().startswith("onedrive -")
           for part in resolved.parts):
        raise ValueError("shared/synchronized drive output is forbidden")
    # Also forbid project data layouts even inside local temp.
    parts = [part.casefold() for part in resolved.parts]
    if any(parts[i] == "data" and parts[i + 1] in {"raw", "dry", "synthetic"}
           for i in range(len(parts) - 1)):
        raise ValueError("project data paths are forbidden as development output")
    if os.name == "nt":
        if ctypes.windll.kernel32.GetDriveTypeW(str(resolved.anchor)) != 3:  # DRIVE_FIXED
            raise ValueError("output must reside on a fixed local disk")
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists; nothing is overwritten: {path}")
    return resolved


class Plan:
    def __init__(self, inputs, config=Config(), output=None):
        self.generator = Generator(config)
        self.sources = discover_sources(inputs)
        self.output = safe_output(output) if output is not None else None
        self.inputs = [(name, str(Path(directory).expanduser().resolve())) for name, directory in inputs]
        if self.output and any(self.output.is_relative_to(Path(directory)) for _, directory in self.inputs):
            raise ValueError("output must be outside every input directory")

    def _jobs(self):
        g = self.generator
        drives = g.drives() if g.config.per_drive else [None]
        for drive in drives:
            for index in range(g.config.variations):
                yield drive, index

    def items(self):
        g = self.generator
        for source in self.sources:
            for drive, index in self._jobs():
                chain = g.chain(source.dry_id, index, drive)
                drive_name = drive or next((x["effect_type"] for x in chain["chain"]
                                            if self.generator.effects[x["effect_type"]]["category"] == "gain_stage"),
                                           "nodrive")
                # wet_{dry}_{schema_key}_{번호}.wav  (번호: 같은 dry 안에서 1부터)
                target_id = f"wet_{source.name}_{drive_name}_{index + 1:03d}"
                audio_relative = f"audio/{target_id}.wav"
                label = {
                    "target_id": target_id, "is_synthetic": True,
                    "dry_sample_id": source.dry_id,
                    "ground_truth_chain": chain,
                    "source_dataset": g.config.source_dataset,
                    "file_path": ((self.output / audio_relative).as_posix() if self.output
                                  else f"<OUTPUT>/{audio_relative}"),
                }
                g.validator.validate(label)
                yield source, label

    def summary(self, preview_count=3):
        examples = []
        for _, label in self.items():
            examples.append(label)
            if len(examples) >= preview_count:
                break
        pairs = len(self.sources) * sum(1 for _ in self._jobs())
        return {
            **self.generator.settings, "run_id": self.generator.run_id,
            "variations_per_source": self.generator.config.variations,
            "drives": self.generator.drives(),
            "dry_sources": len(self.sources), "pairs": pairs,
            "wav_files": pairs, "label_files": pairs, "manifest_files": 1,
            "dry_files": len(self.sources),
            "total_output_files": 2 * pairs + 1 + len(self.sources),
            "output": str(self.output) if self.output else "<OUTPUT> (not specified)",
            "inputs": self.inputs, "warnings": self.generator.warnings,
            "preview_labels": examples,
        }

    def execute(self):
        if self.output is None:
            raise ValueError("--execute requires an explicit --output")
        # Re-check immediately before writing; existing directories are never reused.
        safe_output(self.output)
        import numpy as np
        import soundfile as sf

        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "audio").mkdir()
        (self.output / "labels").mkdir()
        (self.output / "dry").mkdir()
        completed, current_path = 0, None
        for source, label in self.items():
            if source.path != current_path:
                # Team input format: mono, 44.1 kHz (effects.audio_io); stereo is averaged.
                audio, sr = read_mono(source.path)
                dry = audio[np.newaxis, :]  # renderer expects channels x samples
                peak = float(np.max(np.abs(dry)))
                if self.generator.config.dry_peak_db is not None and peak > 0:
                    # Same input level for every dry, so drive amount depends on knobs, not recording level.
                    dry = (dry * (10 ** (self.generator.config.dry_peak_db / 20) / peak)).astype(np.float32)
                # Save the rendered dry: search/training must start from the same audio as the renderer.
                _check_plain_path(self.output / "dry")
                with (self.output / "dry" / f"{source.name}.wav").open("xb") as stream:
                    sf.write(stream, dry.T, sr, format="WAV", subtype="FLOAT")
                current_path = source.path
            chain = label["ground_truth_chain"]
            # Pass a copy; the saved ground truth remains independent of renderer mutation.
            wet = np.asarray(render_chain(dry.copy(), sr, deepcopy(chain), self.generator.effects))
            if wet.shape != dry.shape or not np.isfinite(wet).all():
                raise ValueError("renderer returned nonfinite audio or changed channel/sample shape")
            audio_path = Path(label["file_path"])
            label_path = self.output / "labels" / (label["target_id"] + ".json")
            _check_plain_path(self.output / "audio")
            _check_plain_path(self.output / "labels")
            # Exclusive creation also prevents overwriting after the directory preflight.
            with audio_path.open("xb") as stream:
                sf.write(stream, wet.T, sr, format="WAV", subtype="FLOAT")
            with label_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(_json(label) + "\n")
            completed += 1
        manifest = {**self.summary(), "completed_pairs": completed,
                    "dry_id_map": [{"dry_sample_id": s.dry_id, "namespace": s.namespace,
                                    "relative_path": s.relative_path, "input_path": str(s.path),
                                    # dry actually fed to the renderer (mono 44.1 kHz copy, scaled if dry_peak_db)
                                    "render_input": f"dry/{s.name}.wav"}
                                   for s in self.sources]}
        # CNN(estimation)이 IR 데이터로 인정하도록 앰프 정의 + IR 파일 hash를 기록
        manifest.update(amp_context_fields(self.generator.amp["amp_id"]))
        _check_plain_path(self.output)
        with (self.output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(manifest) + "\n")
        return manifest


def amp_context_fields(amp_id):
    """manifest에 넣을 앰프 정보. estimation.amp와 같은 형식(amp_context + hash)입니다."""
    from estimation.amp import NO_AMP, current_context, no_amp
    from estimation.chain import fingerprint

    context = no_amp() if amp_id == NO_AMP else current_context()
    if context["amp_id"] != amp_id:
        raise ValueError(f"amp_id {amp_id}가 현재 schemas/amp.data.json({context['amp_id']})과 다릅니다.")
    return {"amp_context": context, "amp_context_hash": fingerprint(context)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-input", nargs=2, action="append", required=True,
                        metavar=("NAME", "DIRECTORY"), help="repeat for multiple guitars; keep NAME stable")
    parser.add_argument("--variations", type=int, default=1,
                        help="variants per source (with --per-drive: per source and per drive; default: 1)")
    parser.add_argument("--per-drive", action="store_true",
                        help="make --variations chains for EVERY enabled drive (Step 1: --per-drive --variations 5)")
    parser.add_argument("--dry-peak-db", type=float, default=None,
                        help="scale each dry to this peak dBFS before rendering (Step 1: -12); scaled dry is saved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effects", nargs="+", default=list(DEFAULT_EFFECTS))
    parser.add_argument("--compressor-probability", type=float, default=0.5)
    parser.add_argument("--eq-probability", type=float, default=0.5)
    parser.add_argument("--chain-order", nargs=3, default=["compressor", "drive", "eq"])
    parser.add_argument("--source-dataset", default=SOURCE_DATASET)
    parser.add_argument("--output", help="NEW physical directory under local system temp; no links/junctions")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="explicitly render to local temp only")
    modes.add_argument("--dry-run", action="store_true", help="preview only (the default); writes no files")
    args = parser.parse_args(argv)
    try:
        if set(args.chain_order) != {"compressor", "drive", "eq"}:
            raise ValueError("--chain-order must contain compressor, drive, eq exactly once")
        if args.execute and args.output is None:
            raise ValueError("--execute requires --output")
        catalog = json.loads((ROOT / "schemas/effects.data.json").read_text(encoding="utf-8"))
        all_drives = tuple(name for name, d in catalog.items() if d["category"] == "gain_stage")
        stages = {"compressor": Stage(("compressor",), args.compressor_probability),
                  "drive": Stage(all_drives),
                  "eq": Stage(("eq_3band",), args.eq_probability)}
        config = Config(args.variations, args.seed, tuple(args.effects),
                        tuple(stages[name] for name in args.chain_order), args.source_dataset,
                        args.per_drive, args.dry_peak_db)
        plan = Plan(args.dry_input, config, args.output)
        print(_json({"mode": "execute" if args.execute else "dry-run", **plan.summary()}))
        if args.execute:
            print(_json({"completed_pairs": plan.execute()["completed_pairs"], "output": str(plan.output)}))
    except (OSError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
