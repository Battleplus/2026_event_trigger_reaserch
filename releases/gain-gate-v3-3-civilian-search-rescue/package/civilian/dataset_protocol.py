"""Build and validate the frozen N2 train/development scenario registry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from package_layout import require_registered_data_files_root, require_registered_data_root
from protocol_contract import (
    N2_STAGE_ORDER,
    PACKAGE_RELEASE_ID,
    PROTOCOL_ID,
    STAGE_PROTOCOL_ID,
    validate_protocol_contract,
)


HERE = Path(__file__).resolve().parent
SPLIT_SIZES = {"train": 2, "development": 1}
CONDITIONS = {
    "train": ("base", "carrier_competition"),
    "development": ("carrier_competition",),
}
TASKBOOK_SHA256 = "4b544e39b5fffa89ef364095341412c807d4d609a11117191b6b981aa51bcf5c"
REGISTRY_SCHEMA = "n2-input-registry/2"
REGISTRY_STATUS = "prepared_static_not_preflighted"
BASE_SEED = 8201
MAP_CONTRACT = "relief-grid-100x100-v1"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _n2_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    from scenario_matrix import build_parent_matrix, manual_causal_case, validate_feasibility

    matrix = build_parent_matrix()
    parents: list[dict[str, Any]] = []
    generator_path = HERE / "scenario_matrix.py"
    for split, count in SPLIT_SIZES.items():
        scenarios = matrix.get(split)
        conditions = CONDITIONS[split]
        if not isinstance(scenarios, list) or len(scenarios) != count or len(conditions) != count:
            raise ValueError(f"N2 scenario matrix has an invalid {split} allocation")
        for index, (scenario, condition) in enumerate(zip(scenarios, conditions, strict=True)):
            if scenario.get("synthetic") is not True:
                raise ValueError(f"N2 parent is not explicitly marked synthetic: {scenario.get('scenario_id')}")
            static_feasibility = validate_feasibility(scenario, expected_condition=condition)
            scenario_id = scenario.get("scenario_id")
            if not isinstance(scenario_id, str) or not scenario_id.rsplit("-", 1)[-1]:
                raise ValueError("N2 scenario ID does not encode its deterministic seed")
            try:
                seed = int(scenario_id.rsplit("-", 1)[-1], 16)
            except ValueError as exc:
                raise ValueError(f"N2 scenario ID has an invalid seed: {scenario_id}") from exc
            relative = Path("data") / split / f"{scenario_id}.json"
            parents.append({
                "split": split,
                "parent_id": scenario_id,
                "condition": condition,
                "index": index,
                "seed": seed,
                "path": relative.as_posix(),
                "scenario": scenario,
                "static_feasibility": static_feasibility,
            })
    causal_case = manual_causal_case()
    if causal_case.get("scenario", {}).get("synthetic") is not True:
        raise ValueError("N2 causal fixture is not explicitly marked synthetic")
    return parents, causal_case, _sha256_file(generator_path)


def build_registry(output_dir: str | Path) -> dict[str, Any]:
    output = require_registered_data_root(output_dir)
    for split in SPLIT_SIZES:
        parent_dir = output / "data" / split
        if parent_dir.exists():
            for old_file in parent_dir.glob("*.json"):
                if old_file.is_symlink() or not old_file.is_file():
                    raise ValueError(f"unexpected file in registered data split: {old_file}")
                old_file.unlink()
    parents, causal_case, generator_sha256 = _n2_inputs()
    entries: list[dict[str, Any]] = []
    for parent in parents:
        scenario = parent["scenario"]
        encoded = _canonical_bytes(scenario)
        target = output / parent["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        entries.append({
            key: parent[key]
            for key in ("split", "parent_id", "condition", "index", "seed", "path")
        } | {
            "synthetic": True,
            "scenario_sha256": _sha256_bytes(encoded),
            "static_feasibility": parent["static_feasibility"],
        })

    causal_path = Path("fixtures") / "causal_allocation_case.json"
    causal_bytes = _canonical_bytes(causal_case)
    causal_target = output / causal_path
    causal_target.parent.mkdir(parents=True, exist_ok=True)
    causal_target.write_bytes(causal_bytes)
    registry = {
        "schema": REGISTRY_SCHEMA,
        "status": REGISTRY_STATUS,
        "package_release": PACKAGE_RELEASE_ID,
        "protocol_id": PROTOCOL_ID,
        "stage_protocol": STAGE_PROTOCOL_ID,
        "stage_order": list(N2_STAGE_ORDER),
        "base_seed": BASE_SEED,
        "map_contract": MAP_CONTRACT,
        "taskbook_sha256": TASKBOOK_SHA256,
        "generator": {
            "path": "package/civilian/scenario_matrix.py",
            "sha256": generator_sha256,
            "function": "build_parent_matrix",
        },
        "scenario_count": len(entries),
        "split_sizes": dict(SPLIT_SIZES),
        "synthetic_inputs": True,
        "input_scope": "deterministic civilian simulation scenarios; not collected rollout or outcome data",
        "parents": entries,
        "causal_case": {
            "path": causal_path.as_posix(),
            "scenario_id": causal_case["scenario_id"],
            "schema": causal_case["schema"],
            "synthetic": True,
            "sha256": _sha256_bytes(causal_bytes),
        },
        "validation_state": {
            "static_feasibility": "checked_by_scenario_matrix_generator",
            "registry_content": "hash_bound_to_static_files_and_generator",
            "target_runtime_preflight": "pending",
            "environment_replay": "not_run",
            "training_or_rollout": "not_run",
            "server_submission_or_authorization": "not_performed",
        },
    }
    registry_path = output / "input_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_bytes(_canonical_bytes(registry))
    return registry


def _registered_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {".", ".."} for part in candidate.parts):
        raise ValueError("registered input path is not a normalized relative path")
    path = root / candidate
    if path.is_symlink():
        raise ValueError(f"registered input is a symlink: {relative}")
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents or not resolved.is_file():
        raise ValueError(f"registered input escapes its data root or is not a file: {relative}")
    return resolved


def validate_registry(
    output_dir: str | Path,
    *,
    verify_scenarios: bool = True,
    verify_splits: set[str] | None = None,
) -> dict[str, Any]:
    root = require_registered_data_files_root(output_dir)
    registry_path = root / "input_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_protocol_contract(
        package_release=registry.get("package_release"),
        protocol_id=registry.get("protocol_id"),
        stage_protocol=registry.get("stage_protocol"),
        stage_order=registry.get("stage_order", ()),
    )
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status") != REGISTRY_STATUS
        or registry.get("protocol_id") != PROTOCOL_ID
        or registry.get("base_seed") != BASE_SEED
        or registry.get("map_contract") != MAP_CONTRACT
        or registry.get("taskbook_sha256") != TASKBOOK_SHA256
        or registry.get("split_sizes") != SPLIT_SIZES
        or registry.get("scenario_count") != sum(SPLIT_SIZES.values())
        or registry.get("synthetic_inputs") is not True
    ):
        raise ValueError("N2 input registry header does not match the frozen release contract")
    if verify_splits is not None and not verify_splits.issubset(SPLIT_SIZES):
        raise ValueError(f"unknown registered split(s): {sorted(verify_splits - set(SPLIT_SIZES))}")
    verified_splits = set(SPLIT_SIZES) if verify_scenarios and verify_splits is None else (verify_splits or set())
    parents, causal_case, generator_sha256 = _n2_inputs()
    generator = registry.get("generator")
    if generator != {
        "path": "package/civilian/scenario_matrix.py",
        "sha256": generator_sha256,
        "function": "build_parent_matrix",
    }:
        raise ValueError("N2 input registry is not bound to the current scenario generator")
    entries = registry.get("parents")
    if not isinstance(entries, list) or len(entries) != sum(SPLIT_SIZES.values()):
        raise ValueError("N2 parent manifest count mismatch")
    expected_paths = {"input_registry.json"}
    observed_counts = {split: 0 for split in SPLIT_SIZES}
    for entry, parent in zip(entries, parents, strict=True):
        expected = {
            key: parent[key]
            for key in ("split", "parent_id", "condition", "index", "seed", "path")
        }
        if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"N2 parent manifest mismatch: {parent['parent_id']}")
        if set(entry) != {*expected, "synthetic", "scenario_sha256", "static_feasibility"}:
            raise ValueError(f"N2 parent manifest fields mismatch: {parent['parent_id']}")
        if entry.get("synthetic") is not True or entry.get("static_feasibility") != parent["static_feasibility"]:
            raise ValueError(f"N2 parent static validation mismatch: {parent['parent_id']}")
        digest = entry.get("scenario_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid registered scenario digest: {parent['parent_id']}")
        expected_paths.add(parent["path"])
        observed_counts[parent["split"]] += 1
        path = _registered_file(root, parent["path"])
        if verify_scenarios and parent["split"] in verified_splits:
            data = path.read_bytes()
            if _sha256_bytes(data) != digest or data != _canonical_bytes(parent["scenario"]):
                raise ValueError(f"registered scenario changed or differs from generator: {parent['parent_id']}")

    causal_manifest = registry.get("causal_case")
    causal_path = "fixtures/causal_allocation_case.json"
    causal_bytes = _canonical_bytes(causal_case)
    expected_causal = {
        "path": causal_path,
        "scenario_id": causal_case["scenario_id"],
        "schema": causal_case["schema"],
        "synthetic": True,
        "sha256": _sha256_bytes(causal_bytes),
    }
    if causal_manifest != expected_causal:
        raise ValueError("N2 causal fixture manifest mismatch")
    expected_paths.add(causal_path)
    causal_file = _registered_file(root, causal_path)
    if verify_scenarios and causal_file.read_bytes() != causal_bytes:
        raise ValueError("N2 causal allocation fixture changed or differs from generator")
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.json")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise ValueError("registered data contains missing or unregistered JSON inputs")
    return {
        "protocol_id": registry["protocol_id"],
        "generator_sha256": generator_sha256,
        "scenario_count": len(entries),
        "allocation": observed_counts,
        "causal_fixture_verified": bool(verify_scenarios),
        "scenarios_verified": verify_scenarios and verified_splits == set(SPLIT_SIZES),
        "verified_splits": sorted(verified_splits) if verify_scenarios else [],
        "target_runtime_preflight": "pending",
        "execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "registered_data")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(validate_registry(args.output_dir), sort_keys=True))
    else:
        registry = build_registry(args.output_dir)
        print(json.dumps({"scenario_count": registry["scenario_count"], "registry": str(args.output_dir / "input_registry.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
