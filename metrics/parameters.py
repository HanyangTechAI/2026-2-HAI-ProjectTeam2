"""Independent Step 0 knob errors; no optimizer or renderer dependencies."""

import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


def load_effect_definitions():
    path = Path(__file__).resolve().parents[1] / "schemas" / "effects.data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(errors, matches):
    return {
        "mean_normalized_error": sum(errors) / len(errors) if errors else None,
        "max_normalized_error": max(errors) if errors else None,
        "categorical_all_match": all(matches) if matches else None,
    }


def evaluate_parameters(effect_type, truth, predicted, definitions=None):
    """Compare complete value dictionaries using effects.data.json ranges.

    Continuous raw errors retain the original unit. Normalized errors use
    linear range width, or log range width for scale='log'. Mean/max include
    only continuous parameters; categorical/boolean matches stay separate.
    Missing, unknown, non-finite and out-of-range values raise ValueError.
    No defaults are filled, values clipped, or acceptance thresholds assumed.
    """
    if definitions is None:
        definitions = load_effect_definitions()
    if not isinstance(effect_type, str) or effect_type not in definitions:
        raise ValueError(f"Unknown effect type: {effect_type}")
    specs = definitions[effect_type]["params"]
    names = {spec["name"] for spec in specs}
    for values in (truth, predicted):
        if not isinstance(values, Mapping) or set(values) != names:
            raise ValueError(f"{effect_type}: expected exactly {sorted(names)}")

    details, errors, matches = {}, [], []
    for spec in specs:
        name, kind = spec["name"], spec["type"]
        actual, estimate = truth[name], predicted[name]
        entry = {"truth": actual, "predicted": estimate, "type": kind}
        if kind == "continuous":
            low, high = spec["range"]
            scale = spec.get("scale", "linear")
            if not low < high or scale not in ("linear", "log"):
                raise ValueError(f"{name}: invalid range or scale")
            if scale == "log" and low <= 0:
                raise ValueError(f"{name}: log range must be positive")
            for value in (actual, estimate):
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or not low <= value <= high):
                    raise ValueError(f"{name}: expected finite number in [{low}, {high}]")
            if scale == "log":
                error = abs(math.log(estimate) - math.log(actual)) / (math.log(high) - math.log(low))
            else:
                error = abs(estimate - actual) / (high - low)
            entry.update(absolute_error=abs(estimate - actual),
                         normalized_error=error, unit=spec.get("unit"), scale=scale)
            errors.append(error)
        elif kind in ("categorical", "boolean"):
            for value in (actual, estimate):
                valid = (isinstance(value, str) and value in spec["options"]
                         if kind == "categorical" else isinstance(value, bool))
                if not valid:
                    raise ValueError(f"{name}: invalid {kind} value")
            entry["match"] = actual == estimate
            matches.append(entry["match"])
        else:
            raise ValueError(f"{name}: unsupported parameter type {kind}")
        details[name] = entry
    return {"per_parameter": details, **_summary(errors, matches)}


def evaluate_chain(truth, predicted, definitions=None):
    """Compare chain.schema.json instances with fixed order and amp identity.

    Validate all values even if topology differs. A topology/amp mismatch
    has no meaningful aligned knob score: return structure_match=False and
    None aggregates instead of silently comparing unrelated knobs.
    """
    if definitions is None:
        definitions = load_effect_definitions()
    validated = []
    for candidate in (truth, predicted):
        if (not isinstance(candidate, Mapping)
                or not isinstance(candidate.get("amp_id"), str) or not candidate["amp_id"]
                or not isinstance(candidate.get("chain"), list)):
            raise ValueError("Expected amp_id and ordered chain list")
        for item in candidate["chain"]:
            if not isinstance(item, Mapping) or not {"effect_type", "values"} <= item.keys():
                raise ValueError("Each chain item needs effect_type and values")
            evaluate_parameters(item["effect_type"], item["values"], item["values"], definitions)
        validated.append([item["effect_type"] for item in candidate["chain"]])
    structure_match = truth["amp_id"] == predicted["amp_id"] and validated[0] == validated[1]
    if not structure_match:
        return {"structure_match": False, "per_effect": None, **_summary([], [])}
    per_effect, errors, matches = [], [], []
    for actual, estimate in zip(truth["chain"], predicted["chain"]):
        result = evaluate_parameters(actual["effect_type"], actual["values"], estimate["values"], definitions)
        per_effect.append({"effect_type": actual["effect_type"], **result})
        for entry in result["per_parameter"].values():
            if entry["type"] == "continuous":
                errors.append(entry["normalized_error"])
            else:
                matches.append(entry["match"])
    return {"structure_match": True, "per_effect": per_effect, **_summary(errors, matches)}
