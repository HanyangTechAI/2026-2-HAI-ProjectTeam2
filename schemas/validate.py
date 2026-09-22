"""Validate schemas, example data, and references without network access."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


def load_schemas(directory):
    schemas = {}
    registry = Registry()
    for path in sorted(directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        schemas[path.name.removesuffix(".schema.json")] = schema
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(path.resolve().as_uri(), resource)
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], resource)
    return schemas, registry


def validate_chain(chain, effects, amp):
    if chain["amp_id"] != amp["amp_id"]:
        raise ValueError(f"Unknown amp_id: {chain['amp_id']}")
    for item in chain["chain"]:
        effect_type = item["effect_type"]
        if effect_type not in effects:
            raise ValueError(f"Unknown effect_type: {effect_type}")
        params = {param["name"]: param for param in effects[effect_type]["params"]}
        if set(item["values"]) != set(params):
            raise ValueError(f"{effect_type}: values must contain exactly {sorted(params)}")
        for name, value in item["values"].items():
            param = params[name]
            rule = {"type": {"continuous": "number", "categorical": "string", "boolean": "boolean"}[param["type"]]}
            if param["type"] == "continuous" and param.get("range") is not None:
                rule.update(minimum=param["range"][0], maximum=param["range"][1])
            if param["type"] == "categorical" and param.get("options") is not None:
                rule["enum"] = param["options"]
            Draft202012Validator(rule).validate(value)


def validate_references(data):
    effects = data["effects"]
    validate_chain(data["chain"], effects, data["amp"])
    target = data["target_sample"]
    if target["is_synthetic"]:
        if target["dry_sample_id"] != data["dry_sample"]["sample_id"]:
            raise ValueError(f"Unknown dry_sample_id: {target['dry_sample_id']}")
        validate_chain(target["ground_truth_chain"], effects, data["amp"])
    unknown = set(data["user_inventory"]["owned_effect_types"]) - set(effects)
    if unknown:
        raise ValueError(f"Unknown owned_effect_types: {sorted(unknown)}")


def main():
    directory = Path(__file__).resolve().parent
    schemas, registry = load_schemas(directory)
    data = {}
    for name, schema in schemas.items():
        data[name] = json.loads((directory / f"{name}.data.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()).validate(data[name])
        print(f"Validation passed: {name}.data.json")
    validate_references(data)
    print(f"All {len(schemas)} schemas and example references passed.")


if __name__ == "__main__":
    main()
