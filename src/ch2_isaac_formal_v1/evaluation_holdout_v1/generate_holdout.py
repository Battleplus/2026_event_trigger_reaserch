#!/usr/bin/env python3
"""Generate sealed ID-test and communication-dropout OOD scene configs only."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FORMAL_ROOT = PROJECT_ROOT / "formal_collection_v1"
BASE_CONFIG = FORMAL_ROOT / "configs" / "capture_base.json"
FORMAL_HASHES = FORMAL_ROOT / "formal_registry_sha256.json"
CONFIG_ROOT = ROOT / "configs"
ID_ROOT = CONFIG_ROOT / "id_test"
OOD_ROOT = CONFIG_ROOT / "ood_poor_communication"
MANIFEST_PATH = ROOT / "holdout_manifest.json"


def _load_formal_generator():
    sys.path.insert(0, str(PROJECT_ROOT))
    from formal_collection_v1.prepare_formal_registry import (  # type: ignore
        CONDITIONS,
        make_config,
    )
    return CONDITIONS, make_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return value.replace("_", "-")


def _scene_record(config_path: Path, config: dict, split: str, condition: str,
                  source: str, seed: int, family: str, cohort: str,
                  forbidden_reason: str) -> dict:
    return {
        "cohort": cohort,
        "run_id": config["run_id"],
        "scenario_id": config["scenario_id"],
        "split": split,
        "condition": condition,
        "source": source,
        "seed": seed,
        "parent_family": family,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha(config_path),
        "forbidden_for_training": True,
        "forbidden_reason": forbidden_reason,
        "execution": "NOT_RUN",
    }


def main() -> None:
    conditions, make_config = _load_formal_generator()
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    formal_hashes = json.loads(FORMAL_HASHES.read_text(encoding="utf-8"))
    formal_registry_sha = formal_hashes["registry_sha256"]
    id_rows = []
    ood_rows = []
    source = "Existing-ET"

    for condition_index, condition in enumerate(conditions):
        for replicate in range(6):
            seed = 202610230001 + condition_index * 100 + replicate
            scenario_id = f"{condition}-seed{seed}"
            run_id = f"holdout-v1-id-test-s{seed}-{_slug(condition)}-existing_et"
            config = make_config(base, "test", seed, condition, source, run_id, scenario_id)
            config["holdout"] = {
                "cohort": "id_test",
                "forbidden_for_training": True,
                "protocol": "evaluation_holdout_v1",
                "source_formal_registry_sha256": formal_registry_sha,
            }
            path = ID_ROOT / f"{run_id}.json"
            _write_json(path, config)
            id_rows.append(_scene_record(
                path, config, "test", condition, source, seed,
                f"id-test-condition-{condition_index}", "id_test",
                "sealed ID test; generated after training design freeze",
            ))

    for replicate in range(20):
        seed = 202610240001 + replicate
        condition = "composite_all_enabled"
        scenario_id = f"poor_communication-seed{seed}"
        run_id = f"holdout-v1-ood-poor-communication-s{seed}-composite_all_enabled-existing_et"
        config = make_config(base, "ood", seed, condition, source, run_id, scenario_id)
        config["communication"] = copy.deepcopy(config["communication"])
        config["communication"]["dropout_probability"] = 0.15
        config["holdout"] = {
            "cohort": "ood_poor_communication",
            "forbidden_for_training": True,
            "protocol": "evaluation_holdout_v1",
            "source_formal_registry_sha256": formal_registry_sha,
            "ood_parameter": "communication.dropout_probability",
            "ood_value": 0.15,
            "formal_baseline_value": 0.03,
        }
        path = OOD_ROOT / f"{run_id}.json"
        _write_json(path, config)
        ood_rows.append(_scene_record(
            path, config, "ood", condition, source, seed,
            "ood-poor-communication", "ood_poor_communication",
            "sealed OOD confirmation; communication dropout is not eligible for training",
        ))

    manifest = {
        "schema": "isaac_roi_evaluation_holdout_v1",
        "holdout_id": "evaluation-holdout-v1",
        "status": "SEALED_CONFIGS_ONLY",
        "execution": "NOT_RUN",
        "forbidden_for_training": True,
        "formal_batch_runner_compatibility": "NOT_APPLICABLE; frozen runner accepts only train/validation registry splits",
        "unsupported_formal_runner_splits": ["test", "ood"],
        "source": source,
        "generator": "evaluation_holdout_v1/generate_holdout.py",
        "generator_sha256": _sha(ROOT / "generate_holdout.py"),
        "formal_registry_sha256": formal_registry_sha,
        "base_config_sha256": _sha(BASE_CONFIG),
        "protocol_file": "PROTOCOL.md",
        "protocol_sha256": _sha(ROOT / "PROTOCOL.md"),
        "id_test": {
            "split": "test", "scene_count": len(id_rows),
            "conditions": list(conditions), "scenes_per_condition": 6,
            "seed_start": 202610230001, "condition_seed_offset": 100,
            "scenes": id_rows,
        },
        "ood_poor_communication": {
            "split": "ood", "scene_count": len(ood_rows),
            "condition": "composite_all_enabled", "seed_start": 202610240001,
            "parameter": "communication.dropout_probability",
            "formal_baseline": 0.03, "ood_value": 0.15,
            "mechanisms_inherited_from": "formal_collection_v1.make_config composite_all_enabled",
            "scenes": ood_rows,
        },
        "total_scene_count": len(id_rows) + len(ood_rows),
    }
    _write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "status": manifest["status"], "id_test_count": len(id_rows),
        "ood_count": len(ood_rows), "manifest": str(MANIFEST_PATH),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
