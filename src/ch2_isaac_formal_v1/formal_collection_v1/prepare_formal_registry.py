#!/usr/bin/env python3
"""Materialize the fixed, 168-scene formal collection registry."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "capture_base.json"
CONFIG_DIR = ROOT / "configs" / "scenes"
REGISTRY_PATH = ROOT / "formal_batch_registry.json"
HASH_PATH = ROOT / "formal_registry_sha256.json"
CONDITIONS = (
    "light_delay",
    "task_activation",
    "platform_fault",
    "energy_anomaly",
    "communication_outage",
    "environment_speed_block",
    "composite_all_enabled",
)
SOURCES = ("Existing-ET", "Periodic-4min", "Random-Allow-Suppress")
SOURCE_SLUGS = {
    "Existing-ET": "existing_et",
    "Periodic-4min": "periodic_4min",
    "Random-Allow-Suppress": "random_allow_suppress",
}
CONDITION_MECHANISMS = {
    "light_delay": {"light_delay"},
    "task_activation": {"task_activation"},
    "platform_fault": {"platform_fault"},
    "energy_anomaly": {"energy_anomaly"},
    "communication_outage": {"communication_outage"},
    "environment_speed_block": {"environment_speed_block"},
    "composite_all_enabled": {
        "light_delay", "task_activation", "platform_fault", "energy_anomaly",
        "communication_outage", "environment_speed_block",
    },
}
ALL_MECHANISMS = tuple(sorted(set().union(*CONDITION_MECHANISMS.values())))
SOURCE_FILES = (
    "mission.py", "isaac_backend.py", "run_capture.py", "state_sampling.py",
    "run_suffix_pairs.py", "run_formal_batch.py", "prepare_formal_registry.py",
    "README.md", "test_formal_registry.py", "test_suffix_workers.py",
    "test_receipt.json",
)
ACTIVATION_TASK_IDS = [
    f"z{zone}-cell{cell}" for zone in range(3) for cell in range(3)
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_config(base: dict, split: str, seed: int, condition: str,
                source: str, run_id: str, scenario_id: str) -> dict:
    config = copy.deepcopy(base)
    config["schema_version"] = 4
    config["run_id"] = run_id
    config["scenario_id"] = scenario_id
    config["seed"] = seed
    config["mode"] = "search"
    config["targets_per_zone"] = 3
    config.pop("representative_events", None)
    enabled = CONDITION_MECHANISMS[condition]
    config["mechanisms"] = {name: {"enabled": name in enabled} for name in ALL_MECHANISMS}
    if "light_delay" in enabled:
        config["mechanisms"]["light_delay"].update({
            "platform_id": "uav-0-1", "additional_delay_s": 1.0,
        })
    config["energy_anomaly"] = ({"aircraft_id": "uav-8-1", "start_s": 1200,
                                  "rate_multiplier": 10.0}
                                 if "energy_anomaly" in enabled else None)
    block = {"block_id": "slow_block_0", "rect_offset_km": [-1.0, -1.0, 1.0, 1.0],
             "interval_s": [0, 120], "max_speed_mps": 20.0}
    config["scene"]["spatial_speed_blocks"] = (
        [block] if "environment_speed_block" in enabled else [])
    activation_on = "task_activation" in enabled
    config["evaluator_schedule"] = {
        "activation_task_ids": ACTIVATION_TASK_IDS if activation_on else [],
        "activation_times_s": [600.0] * len(ACTIVATION_TASK_IDS) if activation_on else [],
        "disturbances": {
            "platform_fault_time_s": 900,
            "communication_outage_time_s": 1800,
            "communication_outage_duration_s": 120,
        },
        "visibility": "evaluator only; never supplied in public state projection",
    }
    config["pilot"] = {
        "ordinary_event_threshold": 1,
        "ordinary_event_cooldown_s": 240,
        "state_sampling_seed": seed ^ 0x5A17C0DE,
        "planning_delay_s": 1,
    }
    config["formal"] = {
        "batch_id": "isaac-roi-formal-collection-v1",
        "scenario_id": scenario_id,
        "split": split,
        "condition": condition,
        "source": source,
        "periodic_interval_s": 240,
        "random_allow_probability": 0.5,
        "baseline_observation_delay_s": 1.0,
        "state_sample_limit": 8,
        "suffix_seeds": [730201, 730202],
        "gpu_worker_count": 1,
        "cpu_threads_per_worker": 1,
        "pilot_completed": False,
        "targets_per_zone_status": "fixed design value; no load search or optimum claim",
    }
    config["runtime"]["multi_gpu"] = False
    config["study_schedule"] = {
        "representative_phase": "complete 90-minute acquisition episode; not a prerequisite gate",
        "formal": "168 fixed scenes; 7 conditions x 12 train + 12 validation",
        "pilot": "NOT_RUN; no result used to select this design",
        "training": "NOT_RUN; only after trajectory audit",
    }
    return config


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    split_seed_bases = {"train": 202610210001, "validation": 202610220001}
    for split, seed_base in split_seed_bases.items():
        for condition_index, condition in enumerate(CONDITIONS):
            for replicate in range(12):
                seed = seed_base + condition_index * 100 + replicate
                source = SOURCES[replicate % len(SOURCES)]
                run_id = (f"formal-v1-search3-{split}-s{seed}-{condition}-"
                          f"{SOURCE_SLUGS[source]}")
                scenario_id = f"{condition}-seed{seed}"
                config_path = CONFIG_DIR / f"{run_id}.json"
                config = make_config(base, split, seed, condition, source, run_id, scenario_id)
                write_json(config_path, config)
                runs.append({
                    "run_id": run_id,
                    "split": split,
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "parent_family": f"{split}-seed{seed}",
                    "condition": condition,
                    "source": source,
                    "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                    "config_sha256": sha(config_path),
                    "output_path": f"scenes/{run_id}",
                    "execution": "NOT_RUN",
                })
    registry = {
        "schema_version": 1,
        "registry_id": "isaac-roi-formal-collection-v1",
        "design_status": "fixed formal collection plan; no pilot-based selection",
        "execution": "NOT_RUN",
        "run_count": len(runs),
        "conditions": list(CONDITIONS),
        "sources": list(SOURCES),
        "split_counts": {"train": 84, "validation": 84},
        "source_counts_per_condition_split": {source: 4 for source in SOURCES},
        "representative_duration_s": int(base["representative_duration_s"]),
        "state_sample_limit": 8,
        "suffix_seeds": [730201, 730202],
        "default_mode": "search",
        "targets_per_zone": 3,
        "targets_per_zone_selection": "fixed design value; not empirically selected as optimal",
        "gpu_worker_count": 1,
        "cpu_threads_per_worker": 1,
        "retry_policy": "none; failed scenes remain failed and are not replaced",
        "runs": runs,
    }
    write_json(REGISTRY_PATH, registry)
    file_hashes = {"formal_batch_registry.json": sha(REGISTRY_PATH)}
    for path in sorted(CONFIG_DIR.glob("*.json")):
        file_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in SOURCE_FILES:
        path = ROOT / name
        if path.is_file():
            file_hashes[name] = sha(path)
    write_json(HASH_PATH, {"schema": "isaac_roi_formal_sha256_v1",
                           "registry_file": "formal_batch_registry.json",
                           "registry_sha256": file_hashes["formal_batch_registry.json"],
                           "files": file_hashes})
    print(f"wrote {len(runs)} scene configs and registry {file_hashes['formal_batch_registry.json']}")


if __name__ == "__main__":
    main()
