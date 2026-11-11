"""Split labels by dry source, without importing generators or opening audio.

Run from the repository root:
    python -m scripts.split_dataset --input LABELS --output NEW_DIRECTORY
Only Python's standard library is required. See DATASET_SPLIT.md for details.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (80, 10, 10)  # Provisional group ratios, not a validated policy.
ALGORITHM = "sha256-dry-group-largest-remainder-v1"


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
    return value


def label_ids(label):
    """Return (variant_id, dry_id), accepting both existing label formats.

    Schema labels use target_id/dry_sample_id; generator labels use
    synthetic_id/dry_sample.sample_id. Present aliases must agree.
    """
    if not isinstance(label, Mapping):
        raise ValueError("Each label must be a JSON object")
    variant = [_text(label[key], key) for key in ("target_id", "synthetic_id") if key in label]
    dry = [_text(label["dry_sample_id"], "dry_sample_id")] if "dry_sample_id" in label else []
    if "dry_sample" in label:
        nested = label["dry_sample"]
        if not isinstance(nested, Mapping) or "sample_id" not in nested:
            raise ValueError("dry_sample must contain sample_id")
        dry.append(_text(nested["sample_id"], "dry_sample.sample_id"))
    if not variant or not dry:
        raise ValueError("Every label needs target_id or synthetic_id, and a dry source ID")
    if len(set(variant)) != 1 or len(set(dry)) != 1:
        raise ValueError("Conflicting sample ID aliases in label")
    return variant[0], dry[0]


def _canonical(label):
    try:
        return json.dumps(label, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must contain finite, JSON-serializable values") from exc


def _index(labels):
    indexed = {}
    for label in labels:
        sample_id, dry_id = label_ids(label)
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        indexed[sample_id] = (dry_id, label, _canonical(label))
    if not indexed:
        raise ValueError("No labels to split")
    return indexed


def _weights(ratios):
    try:
        values = tuple(Fraction(str(value)) for value in ratios)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Ratios must be three finite, nonnegative numbers") from exc
    if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Ratios must be three nonnegative numbers with a positive sum")
    return values


def group_counts(group_count, ratios=DEFAULT_RATIOS):
    """Largest-remainder allocation; ties use train, then val, then test."""
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 0:
        raise ValueError("Group count must be a nonnegative integer")
    weights = _weights(ratios)
    quotas = [group_count * weight / sum(weights) for weight in weights]
    counts = [quota.numerator // quota.denominator for quota in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:group_count - sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def split_labels(labels, ratios=DEFAULT_RATIOS, seed=42):
    """Return three lists of copied, otherwise unchanged original labels.

    Assign whole dry IDs, not individual variations. A seeded SHA-256 order
    makes input/file order irrelevant and does not touch global RNG state.
    Assignment is stable for an identical dataset; adding groups may change it.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    indexed = _index(labels)
    groups = defaultdict(list)
    for sample_id, (dry_id, _, _) in indexed.items():
        groups[dry_id].append(sample_id)
    counts = group_counts(len(groups), ratios)

    def rank(dry_id):
        value = json.dumps([seed, dry_id], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).digest(), dry_id

    ordered = sorted(groups, key=rank)
    result, offset = {}, 0
    for name in SPLITS:
        selected = ordered[offset:offset + counts[name]]
        sample_ids = sorted(sample_id for dry_id in selected for sample_id in groups[dry_id])
        result[name] = [deepcopy(indexed[sample_id][1]) for sample_id in sample_ids]
        offset += counts[name]
    return result


def validate_splits(labels, partitions, ratios=DEFAULT_RATIOS):
    """Raise ValueError on leakage, duplicate/missing/unknown/changed labels,
    or an allocation inconsistent with the requested dry-group ratios.
    """
    source = _index(labels)
    weights = _weights(ratios)
    if not isinstance(partitions, Mapping) or set(partitions) != set(SPLITS):
        raise ValueError("Expected exactly train, val, and test partitions")
    owners, seen, dry_ids, label_counts = {}, set(), {}, {}
    for name in SPLITS:
        if not isinstance(partitions[name], list):
            raise ValueError(f"{name} must be a list of labels")
        group_ids = set()
        for label in partitions[name]:
            sample_id, dry_id = label_ids(label)
            if sample_id in seen:
                raise ValueError(f"Duplicate output sample ID: {sample_id}")
            if sample_id not in source:
                raise ValueError(f"Unknown output sample ID: {sample_id}")
            if _canonical(label) != source[sample_id][2]:
                raise ValueError(f"Output label changed: {sample_id}")
            if dry_id in owners and owners[dry_id] != name:
                raise ValueError(f"Dry source leaked across splits: {dry_id}")
            seen.add(sample_id)
            owners[dry_id] = name
            group_ids.add(dry_id)
        dry_ids[name] = sorted(group_ids)
        label_counts[name] = len(partitions[name])
    missing = set(source) - seen
    if missing:
        raise ValueError(f"Missing output sample IDs: {sorted(missing)}")
    counts = {name: len(dry_ids[name]) for name in SPLITS}
    if counts != group_counts(len(owners), weights):
        raise ValueError("Dry-group counts do not match the requested ratio allocation")
    requested = {name: float(weight / sum(weights)) for name, weight in zip(SPLITS, weights)}
    return {
        "ratio_basis": "dry_source_count",
        "requested_group_ratios": requested,
        "total_dry_sources": len(owners), "total_labels": len(source),
        "group_counts": counts, "label_counts": label_counts,
        "actual_group_ratios": {name: counts[name] / len(owners) for name in SPLITS},
        "actual_label_ratios": {name: label_counts[name] / len(source) for name in SPLITS},
        "dry_sample_ids": dry_ids,
        "validation": {"no_group_leakage": True, "no_duplicates": True,
                       "no_missing_or_unknown_samples": True, "labels_preserved": True,
                       "group_counts_match_allocation": True},
        "warnings": [f"{name} has no dry sources despite a positive requested ratio"
                     for name in SPLITS if requested[name] > 0 and not counts[name]],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def load_labels(input_path):
    """Load a JSON object/array, or recursively read JSON files in a folder.
    Paths inside labels are preserved and never opened or reinterpreted.
    """
    path = Path(input_path)
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".json")
    elif path.is_file() and path.suffix.lower() == ".json":
        paths = [path]
    else:
        raise ValueError(f"Input must be an existing JSON file or label directory: {path}")
    labels = []
    for source in paths:
        try:
            with source.open(encoding="utf-8-sig") as stream:
                data = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid label file {source}: {exc}") from exc
        if not isinstance(data, (dict, list)):
            raise ValueError(f"Label file must contain an object or array: {source}")
        labels.extend(data if isinstance(data, list) else [data])
    _index(labels)  # reject missing IDs and duplicates across files before writing
    return labels


def write_splits(labels, output_dir, ratios=DEFAULT_RATIOS, seed=42, input_source=None):
    """Write train.json, val.json, test.json, summary.json to a NEW directory.
    Existing directories/files are never overwritten. Labels/audio are not moved.
    """
    labels = list(labels)
    ratios = _weights(ratios)
    output = Path(output_dir).expanduser().resolve()
    if input_source is not None:
        source = Path(input_source).expanduser().resolve()
        if source.is_dir() and output.is_relative_to(source):
            raise ValueError("Output must be outside the input label directory")
    partitions = split_labels(labels, ratios, seed)
    summary = {"format_version": 1, "algorithm": ALGORITHM, "seed": seed,
               **validate_splits(labels, partitions, ratios)}
    if input_source is not None:
        summary["input_source"] = str(source)
    artifacts = {f"{name}.json": partitions[name] for name in SPLITS}
    artifacts["summary.json"] = summary
    # Complete validation/serialization before creating any output path.
    contents = {name: json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                                allow_nan=False) + "\n" for name, data in artifacts.items()}
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in contents.items():
        with (output / filename).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split labels by dry source; no audio or generator dependency")
    parser.add_argument("--input", required=True, help="A JSON label file or directory")
    parser.add_argument("--output", required=True, help="A new output directory outside the label directory")
    parser.add_argument("--ratios", nargs=3, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"),
                        help="Nonnegative weights; default 80 10 10 is provisional")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        summary = write_splits(load_labels(args.input), args.output, args.ratios, args.seed, args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(Path(args.output).resolve()), "seed": args.seed,
                      "group_counts": summary["group_counts"], "label_counts": summary["label_counts"],
                      "warnings": summary["warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
