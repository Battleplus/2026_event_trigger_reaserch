"""Staged, evidence-producing runner for the N02 event-trigger experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from .event_oracle import MODEL_HEADS, make_model_sample, recompute_consequence_labels
except ImportError:
    from event_oracle import MODEL_HEADS, make_model_sample, recompute_consequence_labels


RUN_SCHEMA = "n02-event-trigger-run/1"
LEDGER_SCHEMA = "n02-event-resource-settlement/1"
STAGE_ORDER = ("m0", "m1", "m2", "m3", "m4")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "EVENT_TRIGGER_CONFIG.json"
ARTIFACTS = (
    "EVENT_TRIGGER_CONFIG.json",
    "THRESHOLD_MANIFEST.json",
    "AFFECTED_SCOPE_REPORT.jsonl",
    "CANDIDATE_RECONSTRUCTION_REPORT_M1.json",
    "CANDIDATE_RECONSTRUCTION_REPORT.json",
    "EVENT_TRIGGER_LOG.jsonl",
    "CANDIDATE_TABLE.jsonl",
    "BRANCH_TRAJECTORIES.jsonl",
    "CONSEQUENCE_LABELS.jsonl",
    "M2_DEVELOPMENT_MODEL_SAMPLES.jsonl",
    "WORLD_MODEL_ENSEMBLE_METRICS.json",
    "G_GWM_COMPARISON.json",
    "DEVELOPMENT_EVALUATION.json",
    "RESOURCE_SETTLEMENT.json",
    "SOL_HIGH_REVIEW.json",
    "FINAL_EVENT_TRIGGER_REPORT.md",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _private_simulator_state_sha256(env: Any) -> str:
    # Pickle is not a stable identity across Python versions: set/dict table
    # layout can change after deepcopy even when the simulator state is equal.
    # Bind only the simulator's private logical state in canonical JSON so the
    # same snapshot has the same identity on local Python 3.13 and server 3.11.
    base = getattr(env, "_base", None)
    payload = {
        "time": int(getattr(env, "time", 0)),
        "event_tape_id": str(getattr(env, "event_tape_id", "")),
        "base_tasks": copy.deepcopy(getattr(base, "_tasks", {})),
        "base_platforms": copy.deepcopy(getattr(base, "_platforms", {})),
        "base_queues": copy.deepcopy(getattr(base, "_queues", {})),
        "task_specs": copy.deepcopy(getattr(env, "_task_specs", {})),
        "queues": copy.deepcopy(getattr(env, "_queues", {})),
        "environment": copy.deepcopy(getattr(env, "_environment", [])),
        "communication": copy.deepcopy(getattr(env, "_communication", [])),
        "pending_communications": copy.deepcopy(getattr(env, "_pending_communications", [])),
        "event_ids_applied": sorted(str(value) for value in getattr(env, "_event_ids_applied", set())),
        "reported_platforms": copy.deepcopy(getattr(env, "_reported_platforms", {})),
        "reported_tasks": copy.deepcopy(getattr(env, "_reported_tasks", {})),
        "reported_depots": copy.deepcopy(getattr(env, "_reported_depots", {})),
        "reported_progress_counts": copy.deepcopy(getattr(env, "_reported_progress_counts", {})),
        "reported_energy_used": copy.deepcopy(getattr(env, "_reported_energy_used", {})),
        "extension_events": copy.deepcopy(getattr(env, "_extension_events", [])),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("EVENT_RUNNER_NONFINITE_JSON")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"EVENT_RUNNER_UNSERIALIZABLE:{type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


class _JsonlStream:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("x", encoding="utf-8", newline="\n")
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        self.rows += 1

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EVENT_RUNNER_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"EVENT_RUNNER_JSONL_OBJECT_REQUIRED:{path.name}")
            rows.append(row)
    return rows


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None) -> tuple[dict[str, Any], Path, str]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve(strict=True)
    raw = config_path.read_bytes()
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("EVENT_RUNNER_CONFIG_OBJECT_REQUIRED")
    config = copy.deepcopy(loaded)
    trigger = config.get("trigger")
    if not isinstance(trigger, dict):
        raise ValueError("EVENT_RUNNER_TRIGGER_CONFIG_REQUIRED")
    weights = trigger.get("weights", [0.35, 0.30, 0.25, 0.10])
    if isinstance(weights, list):
        if len(weights) != 4:
            raise ValueError("EVENT_RUNNER_TRIGGER_WEIGHT_SHAPE")
        weights = dict(zip(("d_state", "d_degradation", "d_margin", "d_uncertainty"), map(float, weights)))
    if not isinstance(weights, dict):
        raise ValueError("EVENT_RUNNER_TRIGGER_WEIGHTS")
    trigger["weights"] = weights
    if "quantile" not in trigger and "threshold_quantile" in trigger:
        trigger["quantile"] = trigger["threshold_quantile"]
    trigger.setdefault("quantile", 0.75)
    trigger.setdefault("delta_min", 4)
    trigger.setdefault("history_length", 4)
    expected_weights = {"d_state": 0.35, "d_degradation": 0.30, "d_margin": 0.25, "d_uncertainty": 0.10}
    if any(not math.isclose(float(weights.get(key, math.nan)), value, abs_tol=1e-12) for key, value in expected_weights.items()):
        raise ValueError("EVENT_RUNNER_TRIGGER_WEIGHTS_NOT_PREREGISTERED")
    if not math.isclose(float(trigger["quantile"]), 0.75, abs_tol=1e-12) or int(trigger["delta_min"]) != 4:
        raise ValueError("EVENT_RUNNER_TRIGGER_CALIBRATION_NOT_PREREGISTERED")
    config.setdefault("collection", {})
    config.setdefault("world_model", {})
    config.setdefault("ppo", {})
    config.setdefault("scenario", {})
    config.setdefault("runtime", {})
    config["collection"].setdefault("windows", [8, 16, 24, 32])
    config["collection"].setdefault("branch_horizon", 30)
    config["collection"].setdefault("max_candidates", 16)
    config["collection"].setdefault("minimum_distinct_window_fraction", 0.8)
    config["world_model"].setdefault("seeds", [8201, 8202, 8203])
    config["world_model"].setdefault("epochs", 40)
    config["world_model"].setdefault("batch_size", 8)
    config["ppo"].setdefault("seeds", [8201, 8202, 8203])
    config["ppo"].setdefault("rounds", 4)
    config["ppo"].setdefault("episodes_per_round", 4)
    config["ppo"].setdefault("epochs", 4)
    config["ppo"].setdefault("minibatch_size", 32)
    config["ppo"].setdefault("max_updates", 128)
    config["scenario"].setdefault("horizon", 60)
    config["runtime"].setdefault("threads", 1)
    config["runtime"].setdefault("device", "cpu")
    if int(config["collection"]["max_candidates"]) > 16 or int(config["collection"]["max_candidates"]) < 1:
        raise ValueError("EVENT_RUNNER_CANDIDATE_LIMIT")
    if list(config["collection"]["windows"]) != [8, 16, 24, 32]:
        raise ValueError("EVENT_RUNNER_WINDOWS_NOT_PREREGISTERED")
    if int(config["collection"]["branch_horizon"]) != 30:
        raise ValueError("EVENT_RUNNER_BRANCH_HORIZON_NOT_PREREGISTERED")
    if list(config["world_model"]["seeds"]) != [8201, 8202, 8203] or int(config["world_model"]["epochs"]) != 40 or int(config["world_model"]["batch_size"]) != 8:
        raise ValueError("EVENT_RUNNER_WM_TRAINING_CONFIG_NOT_PREREGISTERED")
    if list(config["ppo"]["seeds"]) != [8201, 8202, 8203] or int(config["ppo"]["max_updates"]) != 128:
        raise ValueError("EVENT_RUNNER_PPO_BUDGET_NOT_PREREGISTERED")
    if str(config["runtime"]["device"]).lower() != "cpu" or int(config["runtime"]["threads"]) != 1:
        raise ValueError("EVENT_RUNNER_RUNTIME_NOT_CPU1")
    return config, config_path, hashlib.sha256(raw).hexdigest()


def _trigger_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config["trigger"])


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "event_runner.py", "event_oracle.py", "event_environment.py", "event_scenarios.py",
        "event_trigger.py", "event_models.py", "event_policy.py",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names if (root / name).is_file()
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLedger:
    def __init__(self, output: Path, run: dict[str, Any]):
        self.output = output
        self.run = run
        self.path = output / "RESOURCE_SETTLEMENT.json"
        self.current_stage: str | None = None
        self.stage_start_wall = 0.0
        self.stage_start_cpu = 0.0
        self._ensure()

    def _ensure(self) -> None:
        if not self.path.is_file():
            self.run["ledger"] = {
                "schema": LEDGER_SCHEMA,
                "run_id": self.run["run_id"],
                "config_sha256": self.run["config_sha256"],
                "source_identity": self.run["source_identity"],
                "started_utc": self.run["created_utc"],
                "completed_utc": None,
                "stages": {},
                "totals": {"cpu_seconds": 0.0, "wall_seconds": 0.0, "environment_steps": 0, "branches": 0, "world_model_forwards": 0, "policy_forwards": 0, "optimizer_updates": 0},
                "first_error": None,
                "run_reuse": False,
                "authorization_reused": False,
            }
            self._save()
        else:
            self.run["ledger"] = _read_json(self.path)
            ledger = self.run["ledger"]
            if ledger.get("schema") != LEDGER_SCHEMA or ledger.get("run_id") != self.run["run_id"]:
                raise ValueError("EVENT_RUNNER_LEDGER_IDENTITY")

    def _save(self) -> None:
        _write_json(self.path, self.run["ledger"])

    def begin(self, stage: str) -> None:
        if stage in self.run["ledger"]["stages"]:
            raise ValueError(f"EVENT_RUNNER_STAGE_ALREADY_RECORDED:{stage}")
        self.current_stage = stage
        self.stage_start_wall = time.perf_counter()
        self.stage_start_cpu = time.process_time()
        self.run["ledger"]["stages"][stage] = {
            "status": "running", "started_utc": _utc_now(),
            "counts": {"environment_steps": 0, "branches": 0, "world_model_forwards": 0, "policy_forwards": 0, "optimizer_updates": 0},
        }
        self._save()

    def count(self, name: str, amount: int) -> None:
        if self.current_stage is None or amount < 0 or name not in self.run["ledger"]["totals"]:
            raise ValueError("EVENT_RUNNER_LEDGER_COUNT")
        stage_counts = self.run["ledger"]["stages"][self.current_stage]["counts"]
        stage_counts[name] += int(amount)
        self.run["ledger"]["totals"][name] += int(amount)

    def add_world_model_forward_calls(self, calls: int) -> None:
        self.count("world_model_forwards", int(calls) * 3)

    def finish(self, status: str = "complete", error: BaseException | None = None) -> None:
        if self.current_stage is None:
            return
        wall = time.perf_counter() - self.stage_start_wall
        cpu = time.process_time() - self.stage_start_cpu
        record = self.run["ledger"]["stages"][self.current_stage]
        record.update({
            "status": status, "finished_utc": _utc_now(),
            "cpu_seconds": cpu, "wall_seconds": wall,
        })
        self.run["ledger"]["totals"]["cpu_seconds"] += cpu
        self.run["ledger"]["totals"]["wall_seconds"] += wall
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            if self.run["ledger"].get("first_error") is None:
                first = {
                    "stage": self.current_stage, "type": type(error).__name__,
                    "message": str(error), "timestamp_utc": _utc_now(),
                    "traceback": traceback.format_exc()[-12000:],
                }
                self.run["ledger"]["first_error"] = first
                _write_json(self.output / "FIRST_FAILURE.json", first)
        self.current_stage = None
        self._save()

    def close(self) -> None:
        self.run["ledger"]["completed_utc"] = _utc_now()
        self._save()


def _begin_run(output: Path, config_path: Path, config_sha256: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "EVENT_TRIGGER_RUN.json"
    prior_outputs = [name for name in ARTIFACTS if (output / name).exists()]
    prior_outputs.extend(name for name in (
        "RESOURCE_SETTLEMENT.json", "M0_CONTRACT_CHECKS.json", "M2_MODEL_SAMPLES.jsonl",
        "FIRST_FAILURE.json", "WORLD_MODEL_RESTORE_REQUEST.json", "M4_POLICY_RESTORE_REQUEST.json",
    ) if (output / name).exists())
    if run_path.exists() or prior_outputs:
        raise ValueError("EVENT_RUNNER_REFUSES_RUN_OR_AUTHORIZATION_REUSE")
    source_identity = _source_identity()
    run = {
        "schema": RUN_SCHEMA,
        "run_id": uuid.uuid4().hex,
        "created_utc": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source_identity": source_identity,
        "run_reuse": False,
        "authorization_reused": False,
    }
    _write_json(run_path, run)
    return run


def _resume_run(output: Path, config_sha256: str) -> dict[str, Any]:
    run_path = output / "EVENT_TRIGGER_RUN.json"
    if not run_path.is_file():
        raise ValueError("EVENT_RUNNER_NO_ACTIVE_RUN; start with m0 or all")
    run = _read_json(run_path)
    if run.get("schema") != RUN_SCHEMA or run.get("config_sha256") != config_sha256:
        raise ValueError("EVENT_RUNNER_CONFIG_OR_RUN_IDENTITY_MISMATCH")
    if run.get("source_identity") != _source_identity():
        raise ValueError("EVENT_RUNNER_SOURCE_IDENTITY_CHANGED_START_NEW_OUTPUT")
    return run


def _require_stage(run: dict[str, Any], required_stage: str) -> None:
    stage = run.get("ledger", {}).get("stages", {}).get(required_stage)
    if not isinstance(stage, dict) or stage.get("status") != "complete":
        raise ValueError(f"EVENT_RUNNER_PREREQUISITE_STAGE_MISSING:{required_stage}")


def _modules():
    try:
        from .event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
        from .event_scenarios import build_event_scenarios, validate_event_scenario
        from .event_trigger import TriggerEngine, calibrate_threshold, identify_affected_scope
    except ImportError:
        from event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
        from event_scenarios import build_event_scenarios, validate_event_scenario
        from event_trigger import TriggerEngine, calibrate_threshold, identify_affected_scope
    return EventReliefEnv, generate_reconstruction_candidates, repair_action, identify_affected_scope, build_event_scenarios, validate_event_scenario, TriggerEngine, calibrate_threshold


def _build_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    _, _, _, _, build_event_scenarios, validate_event_scenario, _, _ = _modules()
    scenarios = build_event_scenarios()
    if len(scenarios) != 6:
        raise ValueError("EVENT_RUNNER_EXPECTED_SIX_PARENTS")
    report = [validate_event_scenario(scenario) for scenario in scenarios]
    parents = [scenario["parent_id"] for scenario in scenarios]
    if len(parents) != len(set(parents)):
        raise ValueError("EVENT_RUNNER_DUPLICATE_PARENT")
    if sum(row["split"] == "train" for row in scenarios) != 4 or sum(row["split"] == "development" for row in scenarios) != 2:
        raise ValueError("EVENT_RUNNER_PARENT_SPLIT_COUNTS")
    for scenario in scenarios:
        base = scenario["base_scenario"]
        if base["map"]["rows"] != 100 or base["map"]["cols"] != 100 or len(base["platforms"]) != 6 or len(base["tasks"]) != 12:
            raise ValueError("EVENT_RUNNER_SCENARIO_SHAPE")
    return scenarios


def _assert_public_only(value: Any, path: str = "observation") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"truth", "raw_truth", "event_tape", "future_events", "future_event_tape", "hidden_tasks", "_tasks", "_platforms"}:
                raise ValueError(f"EVENT_RUNNER_PUBLIC_TRUTH_LEAK:{path}.{key}")
            _assert_public_only(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_only(item, f"{path}[{index}]")


def _candidate_catalog(env: Any, generate: Callable, max_candidates: int, scope: Any = None) -> list[dict[str, Any]]:
    catalog = generate(env, scope=scope, max_candidates=max_candidates)
    if not isinstance(catalog, list) or not catalog or len(catalog) > max_candidates:
        raise ValueError("EVENT_RUNNER_CANDIDATE_CATALOG_SHAPE")
    keys = [str(row.get("candidate_key", "")) for row in catalog]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("EVENT_RUNNER_CANDIDATE_KEYS")
    for record in catalog:
        action = record.get("full_environment_action")
        if not isinstance(action, dict) or not isinstance(record.get("legal"), bool):
            raise ValueError("EVENT_RUNNER_CANDIDATE_ACTION")
        if record.get("snapshot_hash") is None or record.get("event_tape_id") is None:
            raise ValueError("EVENT_RUNNER_CANDIDATE_PROVENANCE")
    return catalog


def _noop_record(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        row for row in candidates
        if isinstance(row.get("full_environment_action"), dict)
        and row["full_environment_action"].get("noop") is True
        and not row["full_environment_action"].get("queue_operations")
        and not row["full_environment_action"].get("dispatches")
    ]
    if len(matches) != 1:
        raise ValueError("EVENT_RUNNER_EXPLICIT_NOOP_CANDIDATE_REQUIRED")
    return matches[0]


def _collect_public_sequence(scenario: dict[str, Any], max_candidates: int, ledger: RunLedger | None = None) -> dict[str, Any]:
    EventReliefEnv, generate, _, _, _, _, _, _ = _modules()
    env = EventReliefEnv(scenario)
    observation = env.reset()
    _assert_public_only(observation)
    observations = [copy.deepcopy(observation)]
    candidate_catalogs: list[list[dict[str, Any]]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        candidates = _candidate_catalog(env, generate, max_candidates)
        candidate_catalogs.append(copy.deepcopy(candidates))
        noop = _noop_record(candidates)
        next_observation, _, terminated, truncated, _ = env.step(copy.deepcopy(noop["full_environment_action"]))
        _assert_public_only(next_observation)
        observations.append(copy.deepcopy(next_observation))
        if ledger is not None:
            ledger.count("environment_steps", 1)
    candidate_catalogs.append(_candidate_catalog(env, generate, max_candidates))
    return {
        "parent_id": scenario["parent_id"], "split": scenario["split"],
        "observations": observations, "candidate_catalogs": candidate_catalogs,
    }


def _calibration_rows(sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sequence in sequences:
        observations = sequence["observations"]
        for index in range(1, len(observations)):
            rows.append({
                "parent_id": sequence["parent_id"], "split": sequence["split"],
                "previous_public": observations[index - 1], "current_public": observations[index],
                "wm_summary": {"d_degradation": 0.0, "d_uncertainty": 0.0, "model_hash": None},
            })
    return rows


def _calibrate_initial(scenarios: list[dict[str, Any]], config: dict[str, Any], ledger: RunLedger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    *_, calibrate_threshold = _modules()
    train_scenarios = [scenario for scenario in scenarios if scenario["split"] == "train"]
    sequences = [_collect_public_sequence(scenario, int(config["collection"]["max_candidates"]), ledger) for scenario in train_scenarios]
    _write_jsonl(ledger.output / "TRAINING_OBSERVATION_SEQUENCES.jsonl", sequences)
    threshold = calibrate_threshold(sequences, _trigger_config(config))
    if not isinstance(threshold, dict) or "threshold" not in threshold:
        raise ValueError("EVENT_RUNNER_INITIAL_THRESHOLD_MANIFEST")
    threshold = copy.deepcopy(threshold)
    threshold.update({
        "calibration_stage": "M1_initial_diagnostic_no_world_model",
        "source_split": "train",
        "source_parent_ids": sorted(row["parent_id"] for row in train_scenarios),
        "training_sequence_sha256": _sha256(sequences),
        "world_model_used": False,
        "world_model_uncertainty": 0.0,
        "world_model_degradation": 0.0,
        "frozen_before_development_evaluation": True,
    })
    _write_json(ledger.output / "THRESHOLD_MANIFEST_INITIAL.json", threshold)
    return threshold, sequences


def _run_m0(config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    scenarios = _build_scenarios(config)
    EventReliefEnv, generate, repair, identify, _, _, TriggerEngine, _ = _modules()
    checks = {"scenario_validation": True, "public_only_observations": True, "candidate_identity_and_limit": True, "noop_execution_and_repair": True}
    for scenario in scenarios:
        env = EventReliefEnv(scenario)
        observation = env.reset()
        _assert_public_only(observation)
        candidates = _candidate_catalog(env, generate, int(config["collection"]["max_candidates"]))
        repeated = _candidate_catalog(env, generate, int(config["collection"]["max_candidates"]))
        if candidates != repeated:
            raise ValueError("EVENT_RUNNER_CANDIDATE_GENERATOR_NOT_DETERMINISTIC")
        noop = _noop_record(candidates)
        repaired, receipt = repair(env, copy.deepcopy(noop["full_environment_action"]))
        if repaired != noop["full_environment_action"] or receipt.get("postrepair_legal") is not True:
            raise ValueError("EVENT_RUNNER_NOOP_REPAIR_FAILED")
        _, _, _, _, info = env.step(repaired)
        if info.get("full_environment_action") != repaired:
            raise ValueError("EVENT_RUNNER_ACTION_EXECUTION_BINDING")
        ledger.count("environment_steps", 1)
        detection = {
            "event_severity": 0.0, "triggered": False,
            "d_state": 0.0, "d_degradation": 0.0, "d_margin": 0.0, "d_uncertainty": 0.0,
        }
        affected = identify(env.public_observation(), detection)
        if not isinstance(affected, dict):
            raise ValueError("EVENT_RUNNER_AFFECTED_SCOPE_API")
    initial_threshold, sequences = _calibrate_initial(scenarios, config, ledger)
    # A development row must be rejected by the threshold calibrator.
    *_, calibrate_threshold = _modules()
    rejected_dev = False
    probe = {"parent_id": next(row["parent_id"] for row in scenarios if row["split"] == "development"),
             "split": "development", "observations": [scenarios[-1]["base_scenario"]]}
    try:
        calibrate_threshold(sequences + [probe], _trigger_config(config))
    except (ValueError, PermissionError):
        rejected_dev = True
    if not rejected_dev:
        raise ValueError("EVENT_RUNNER_THRESHOLD_ACCEPTED_DEVELOPMENT_INPUT")
    _write_json(ledger.output / "EVENT_TRIGGER_CONFIG.json", {
        **config,
        "resolved_trigger": _trigger_config(config),
        "source_config_path": str(ledger.run["config_path"]),
        "source_config_sha256": ledger.run["config_sha256"],
    })
    _write_json(ledger.output / "THRESHOLD_MANIFEST.json", initial_threshold)
    report = {
        "schema": "event-trigger-m0-report/1", "status": "complete",
        "parent_count": len(scenarios), "train_parent_count": 4, "development_parent_count": 2,
        "scenario_checks": [
            {"parent_id": row["parent_id"], "split": row["split"], "condition": row["condition"],
             "platform_count": len(row["base_scenario"]["platforms"]), "task_count": len(row["base_scenario"]["tasks"]),
             "map": copy.deepcopy(row["map"]), "event_tape_id": row["event_tape_id"]}
            for row in scenarios
        ],
        "checks": {**checks, "development_threshold_input_rejected": rejected_dev,
                   "threshold_train_parent_ids": initial_threshold["source_parent_ids"]},
        "initial_threshold_manifest": initial_threshold,
        "training_sequence_sha256": _sha256(sequences),
    }
    _write_json(ledger.output / "M0_CONTRACT_CHECKS.json", report)
    return report


def _scope_value(decision: dict[str, Any]) -> str:
    value = decision.get("scope", decision.get("reconstruction_scope", "NOOP"))
    if value in {"LOCAL_REPAIR", "GLOBAL_REPLAN", "NOOP"}:
        return str(value)
    return "NOOP"


def _event_fields(detection: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    severity = detection.get("event_severity", detection.get("severity", {}))
    if isinstance(severity, dict):
        severity_value = max((_number(value) for value in severity.values()), default=0.0)
        severity_label = "strong" if severity_value >= 1.0 else "moderate" if severity_value > 0.0 else "light"
    else:
        # TriggerEngine emits a semantic severity label. Preserve it in the
        # evidence row instead of coercing strings to 0.0; the numeric gamma
        # remains available in ``gamma_value`` for threshold analysis.
        severity_label = str(severity).lower() if severity is not None else "light"
        severity_value = _number(severity) if not isinstance(severity, str) else 0.0
        if severity_label not in {"light", "moderate", "strong"}:
            severity_label = "strong" if severity_value >= 1.0 else "moderate" if severity_value > 0.0 else "light"
    affected = decision.get("affected_scope")
    if isinstance(affected, dict):
        affected_scope = affected.get("scope", "NOOP")
    else:
        affected_scope = detection.get("affected_scope", decision.get("scope", "NOOP"))
    return {
        "event_type": detection.get("event_type", detection.get("event_types", [])),
        "event_severity": severity_label,
        "event_severity_value": severity_value,
        "gamma_value": detection.get("gamma_value", detection.get("gamma", 0.0)),
        "threshold": detection.get("threshold"),
        "triggered": bool(decision.get("triggered", False)),
        "cooldown_suppressed": bool(decision.get("cooldown_suppressed", detection.get("cooldown_suppressed", False))),
        "cooldown_active": bool(decision.get("cooldown_active", False)),
        "minutes_since_reconstruction": decision.get("minutes_since_reconstruction"),
        "minutes_since_trigger": decision.get("minutes_since_trigger"),
        "strong_event": bool(detection.get("strong_event", False)),
        "affected_scope": affected_scope,
        "d_state": detection.get("d_state", 0.0),
        "d_degradation": detection.get("d_degradation", 0.0),
        "d_margin": detection.get("d_margin", 0.0),
        "d_uncertainty": detection.get("d_uncertainty", 0.0),
    }


def _m1_catalog_scope_fallback(
    env: Any,
    generate: Callable,
    max_candidates: int,
    affected: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    local_count = sum(
        row.get("legal") is True
        and row.get("full_environment_action", {}).get("noop") is not True
        and str(row.get("full_environment_action", {}).get("scope", "")).upper() == "LOCAL_REPAIR"
        for row in candidates
    )
    global_count = sum(
        row.get("legal") is True
        and row.get("full_environment_action", {}).get("noop") is not True
        and str(row.get("full_environment_action", {}).get("scope", "")).upper() == "GLOBAL_REPLAN"
        for row in candidates
    )
    if not bool(affected.get("local_eligible")) or local_count:
        return candidates, affected, {
            "status": "available",
            "local_non_noop_candidate_count": local_count,
            "global_non_noop_candidate_count": global_count,
        }

    global_candidates = _candidate_catalog(env, generate, max_candidates, scope="GLOBAL_REPLAN")
    global_count = sum(
        row.get("legal") is True
        and row.get("full_environment_action", {}).get("noop") is not True
        and str(row.get("full_environment_action", {}).get("scope", "")).upper() == "GLOBAL_REPLAN"
        for row in global_candidates
    )
    adjusted = copy.deepcopy(affected)
    adjusted["local_eligible"] = False
    adjusted["global_eligible"] = True
    adjusted["scope"] = "GLOBAL_REPLAN"
    reasons = list(adjusted.get("reasons", []))
    if "local_repair_no_feasible_solution" not in reasons:
        reasons.append("local_repair_no_feasible_solution")
    adjusted["reasons"] = reasons
    return global_candidates, adjusted, {
        "status": "fallback",
        "code": "local_repair_no_feasible_solution",
        "local_non_noop_candidate_count": 0,
        "global_non_noop_candidate_count": global_count,
    }


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _run_m1_mode(scenario: dict[str, Any], mode: str, threshold: dict[str, Any], config: dict[str, Any], ledger: RunLedger) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    EventReliefEnv, generate, repair, identify, _, _, TriggerEngine, _ = _modules()
    env = EventReliefEnv(scenario)
    observation = env.reset()
    previous = copy.deepcopy(observation)
    history = [copy.deepcopy(observation)]
    engine = TriggerEngine(_trigger_config(config), threshold)
    logs, scopes, candidates_used = [], [], []
    terminated = truncated = False
    wall_total = 0.0
    while not (terminated or truncated):
        wm_summary = {"d_degradation": 0.0, "d_uncertainty": 0.0, "model_hash": None}
        started = time.perf_counter()
        detection = engine.detect(previous, observation, history=history[-4:], wm_summary=wm_summary)
        decision = engine.decide(detection, int(observation.get("time", env.time)), mode=mode)
        affected = identify(observation, detection)
        if decision.get("triggered"):
            decision["high_level_action"] = affected["scope"]
            decision["affected_scope"] = copy.deepcopy(affected)
        scope_arg = affected if decision.get("triggered") else "NOOP"
        candidates = _candidate_catalog(env, generate, int(config["collection"]["max_candidates"]), scope=scope_arg)
        scope_catalog_receipt = {"status": "not_applicable"}
        if decision.get("triggered"):
            candidates, affected, scope_catalog_receipt = _m1_catalog_scope_fallback(
                env, generate, int(config["collection"]["max_candidates"]), affected, candidates,
            )
            decision["high_level_action"] = affected["scope"]
            decision["affected_scope"] = copy.deepcopy(affected)
            decision["scope_catalog_receipt"] = copy.deepcopy(scope_catalog_receipt)
        noop = _noop_record(candidates)
        selected = noop
        if decision.get("triggered"):
            requested_scope = decision.get("high_level_action")
            selected = next((row for row in candidates if row["legal"] and row is not noop
                             and row["full_environment_action"].get("scope") == requested_scope), noop)
        action, receipt = repair(env, copy.deepcopy(selected["full_environment_action"]))
        if receipt.get("postrepair_legal") is not True:
            raise ValueError("EVENT_RUNNER_M1_REPAIR_FAILED")
        if action != selected["full_environment_action"]:
            selected = noop
            action, receipt = repair(env, copy.deepcopy(noop["full_environment_action"]))
        next_observation, reward, terminated, truncated, info = env.step(action)
        effective_mutation_count = int(info.get("effective_reconstruction_count", 0))
        if effective_mutation_count not in (0, 1):
            raise ValueError("EVENT_RUNNER_M1_EFFECTIVE_RECONSTRUCTION_COUNT")
        engine.record_reconstruction(
            int(observation.get("time", env.time - 1)),
            effective=effective_mutation_count > 0,
        )
        decision_wall = time.perf_counter() - started
        wall_total += decision_wall
        step_events = info.get("raw_events", [])
        effective_mutation = effective_mutation_count > 0
        row = {
            "schema": "event-trigger-decision/1", "parent_id": scenario["parent_id"],
            "split": scenario["split"], "condition": scenario["condition"], "mode": mode,
            "time": int(observation.get("time", 0)), **_event_fields(detection, decision),
            "affected_scope_detail": copy.deepcopy(affected),
            "scope_catalog_receipt": copy.deepcopy(scope_catalog_receipt),
            "decision": copy.deepcopy(decision), "detection": copy.deepcopy(detection),
            "selected_candidate_key": selected["candidate_key"],
            "selected_full_environment_action": copy.deepcopy(action),
            "repair_receipt": copy.deepcopy(receipt), "reward": float(reward),
            "planning_call": bool(decision.get("triggered")),
            "nonnoop_requested": int(info.get("requested_reconstruction_count", 0)) > 0,
            "effective_assignment_change": effective_mutation,
            "requested_reconstruction_count": int(info.get("requested_reconstruction_count", 0)),
            "effective_reconstruction_count": effective_mutation_count,
            "task_owner_change_count": int(info.get("task_owner_change_count", 0)),
            "task_switch_count": int(info.get("task_switch_count", 0)),
            "queue_reorder_count": int(info.get("queue_reorder_count", 0)),
            "step_wall_seconds": decision_wall,
            "communication_message_count": sum(int(event.get("message_count", 0)) for event in info.get("raw_events", []) if event.get("type") == "communication_tx"),
        }
        logs.append(row)
        scopes.append({"parent_id": scenario["parent_id"], "split": scenario["split"], "mode": mode,
                       "time": row["time"], "triggered": row["triggered"], "scope": copy.deepcopy(affected)})
        candidates_used.append({"parent_id": scenario["parent_id"], "split": scenario["split"], "mode": mode,
                                "time": row["time"], "candidate_count": len(candidates),
                                "candidate_keys": [item["candidate_key"] for item in candidates],
                                "selected_candidate_key": selected["candidate_key"],
                                "scope_catalog_receipt": copy.deepcopy(scope_catalog_receipt),
                                "snapshot_hash": observation.get("provenance", {}).get("snapshot_hash"),
                                "candidate_catalog_sha256": _sha256(candidates)})
        previous = copy.deepcopy(observation)
        observation = next_observation
        history.append(copy.deepcopy(observation))
        history = history[-4:]
        ledger.count("environment_steps", 1)
    return logs, scopes, {
        "parent_id": scenario["parent_id"], "split": scenario["split"], "mode": mode,
        "metrics": env.episode_metrics(), "ticks": len(logs),
        "trigger_count": sum(row["triggered"] for row in logs),
        "planning_call_count": sum(row["planning_call"] for row in logs),
        "nonnoop_requested_count": sum(row["nonnoop_requested"] for row in logs),
        "effective_reconstruction_count": sum(row["effective_assignment_change"] for row in logs),
        "local_catalog_fallback_count": sum(row.get("scope_catalog_receipt", {}).get("status") == "fallback" for row in logs),
        "communication_messages": sum(row["communication_message_count"] for row in logs),
        "decision_wall_seconds": wall_total,
    }, candidates_used


def _run_m1(scenarios: list[dict[str, Any]], config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    threshold_path = ledger.output / "THRESHOLD_MANIFEST_INITIAL.json"
    if not threshold_path.is_file():
        threshold_path = ledger.output / "THRESHOLD_MANIFEST.json"
    threshold = _read_json(threshold_path)
    if threshold.get("world_model_used") is not False or threshold.get("source_split") != "train":
        raise ValueError("EVENT_RUNNER_M1_THRESHOLD_NOT_INITIAL_TRAIN_ONLY")
    all_logs: list[dict[str, Any]] = []
    all_scopes: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        for mode in ("event", "periodic"):
            logs, scopes, summary, used = _run_m1_mode(scenario, mode, threshold, config, ledger)
            all_logs.extend(logs)
            all_scopes.extend(scopes)
            candidate_rows.extend(used)
            summaries.append(summary)
    _write_jsonl(ledger.output / "EVENT_TRIGGER_LOG.jsonl", all_logs)
    _write_jsonl(ledger.output / "AFFECTED_SCOPE_REPORT.jsonl", all_scopes)
    report = {
        "schema": "event-candidate-reconstruction-report/1", "stage": "M1",
        "status": "complete", "threshold_manifest_sha256": _sha256(threshold),
        "row_count": len(all_logs), "scope_row_count": len(all_scopes),
        "comparison_status": "event_and_periodic_rule_diagnostics",
        "candidate_generation_rows": candidate_rows, "episode_summaries": summaries,
        "paired_parent_results": [
            {"parent_id": scenario["parent_id"],
             "event": next(row for row in summaries if row["parent_id"] == scenario["parent_id"] and row["mode"] == "event"),
             "periodic": next(row for row in summaries if row["parent_id"] == scenario["parent_id"] and row["mode"] == "periodic")}
            for scenario in scenarios
        ],
        "policy_training": False,
        "world_model_uncertainty": 0.0,
    }
    # M2 has its own CANDIDATE_RECONSTRUCTION_REPORT.json. Keep M1's
    # rule/periodic diagnostic report under a distinct identity so M2 cannot
    # erase the evidence required to audit trigger counts, cooldown, scope,
    # communication and effective reconstruction changes.
    _write_json(ledger.output / "CANDIDATE_RECONSTRUCTION_REPORT_M1.json", report)
    return {"status": "complete", "rows": len(all_logs), "scope_rows": len(all_scopes), "episodes": summaries,
            "report_path": "CANDIDATE_RECONSTRUCTION_REPORT_M1.json"}


def _advance_to(env: Any, target_time: int, generate: Callable, max_candidates: int, ledger: RunLedger) -> list[dict[str, Any]]:
    history = [copy.deepcopy(env.public_observation())]
    while env.time < target_time:
        catalog = _candidate_catalog(env, generate, max_candidates)
        noop = _noop_record(catalog)
        observation, _, terminated, truncated, _ = env.step(copy.deepcopy(noop["full_environment_action"]))
        if terminated or truncated:
            raise ValueError("EVENT_RUNNER_WINDOW_ADVANCE_TERMINATED")
        history.append(copy.deepcopy(observation))
        history = history[-4:]
        ledger.count("environment_steps", 1)
    return history


def _window_trajectory(env: Any, start: int, end: int, candidate: dict[str, Any], repair_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    raw = env.export_trajectory()
    selected_steps = [copy.deepcopy(row) for row in raw.get("steps", []) if start <= int(row.get("time", -1)) < end]
    if len(selected_steps) != end - start or len(repair_receipts) != len(selected_steps):
        raise ValueError("EVENT_RUNNER_BRANCH_TRACE_LENGTH")
    for step, repair_receipt in zip(selected_steps, repair_receipts):
        receipts = step.get("operation_receipts", [])
        receipts = list(receipts) if isinstance(receipts, list) else []
        receipts.append({
            "postrepair_legal": repair_receipt.get("postrepair_legal"),
            "legal_before": repair_receipt.get("legal_before"),
            "repaired": repair_receipt.get("repaired"),
            "repair_issues": repair_receipt.get("issues_before", []),
        })
        step["operation_receipts"] = receipts
    raw_events = [event for step in selected_steps for event in step.get("raw_events", [])]
    return {
        "schema": "event-relief-window-trace/1", "parent_id": raw["parent_id"],
        "split": raw["split"], "event_tape_id": raw["event_tape_id"],
        "start_time": start, "end_time": end, "steps": selected_steps,
        "raw_events": copy.deepcopy(raw_events), "raw_truth": copy.deepcopy(raw["raw_truth"]),
    }


def _run_m2(scenarios: list[dict[str, Any]], config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    EventReliefEnv, generate, repair, _, _, _, _, _ = _modules()
    max_candidates = int(config["collection"]["max_candidates"])
    branch_horizon = int(config["collection"]["branch_horizon"])
    windows = [int(value) for value in config["collection"]["windows"]]
    window_reports: list[dict[str, Any]] = []
    branch_stream = _JsonlStream(ledger.output / "BRANCH_TRAJECTORIES.jsonl")
    candidate_stream = _JsonlStream(ledger.output / "CANDIDATE_TABLE.jsonl")
    label_stream = _JsonlStream(ledger.output / "CONSEQUENCE_LABELS.jsonl")
    sample_stream = _JsonlStream(ledger.output / "M2_MODEL_SAMPLES.jsonl")
    development_sample_stream = _JsonlStream(ledger.output / "M2_DEVELOPMENT_MODEL_SAMPLES.jsonl")
    sample_count = 0
    development_sample_count = 0
    try:
        for scenario in scenarios:
            env = EventReliefEnv(scenario)
            initial_observation = env.reset()
            _assert_public_only(initial_observation)
            for start in windows:
                history = _advance_to(env, start, generate, max_candidates, ledger)
                observation = copy.deepcopy(env.public_observation())
                _assert_public_only(observation)
                catalog = _candidate_catalog(env, generate, max_candidates)
                repeated_catalog = _candidate_catalog(env, generate, max_candidates)
                if catalog != repeated_catalog:
                    raise ValueError("EVENT_RUNNER_SNAPSHOT_CATALOG_NOT_DETERMINISTIC")
                noop = _noop_record(catalog)
                event_tape_id = str(scenario["event_tape_id"])
                if any(row.get("event_tape_id") != event_tape_id for row in catalog):
                    raise ValueError("EVENT_RUNNER_CANDIDATE_TAPE_MISMATCH")
                snapshot_hash = str(observation.get("provenance", {}).get("snapshot_hash", ""))
                if not snapshot_hash or any(row.get("snapshot_hash") != snapshot_hash for row in catalog):
                    raise ValueError("EVENT_RUNNER_CANDIDATE_SNAPSHOT_MISMATCH")
                window_id = f"{scenario['parent_id']}-t{start:02d}"
                rejected = copy.deepcopy(getattr(env, "last_candidate_rejections", []))
                for candidate in catalog:
                    candidate_stream.write({
                        "schema": "event-candidate-table-row/1", "window_id": window_id,
                        "parent_id": scenario["parent_id"], "split": scenario["split"],
                        "window_start": start, "snapshot_hash": snapshot_hash,
                        "event_tape_id": event_tape_id, "candidate": copy.deepcopy(candidate),
                        "rejected": False,
                    })
                for candidate in rejected:
                    candidate_stream.write({
                        "schema": "event-candidate-table-row/1", "window_id": window_id,
                        "parent_id": scenario["parent_id"], "split": scenario["split"],
                        "window_start": start, "snapshot_hash": snapshot_hash,
                        "event_tape_id": event_tape_id, "candidate": copy.deepcopy(candidate),
                        "rejected": True,
                    })
                snapshot_env = copy.deepcopy(env)
                private_snapshot_sha256 = _private_simulator_state_sha256(snapshot_env)
                private_tape_sha256 = _sha256(scenario["event_tape"])
                planned_end = start + min(branch_horizon, int(config["scenario"]["horizon"]) - start)
                branches: list[dict[str, Any]] = []
                for candidate in catalog:
                    if candidate.get("legal") is not True:
                        continue
                    branch_env = copy.deepcopy(snapshot_env)
                    if branch_env.public_observation() != observation or branch_env.event_tape_id != event_tape_id:
                        raise ValueError("EVENT_RUNNER_DEEPCOPY_SNAPSHOT_MISMATCH")
                    branch_initial_sha256 = _private_simulator_state_sha256(branch_env)
                    if branch_initial_sha256 != private_snapshot_sha256:
                        raise ValueError("EVENT_RUNNER_PRIVATE_BRANCH_STATE_MISMATCH")
                    receipts: list[dict[str, Any]] = []
                    for index in range(planned_end - start):
                        requested = candidate["full_environment_action"] if index == 0 else noop["full_environment_action"]
                        action, receipt = repair(branch_env, copy.deepcopy(requested))
                        if receipt.get("postrepair_legal") is not True or action != requested:
                            raise ValueError("EVENT_RUNNER_LEGAL_CANDIDATE_CHANGED_DURING_REPAIR")
                        _, _, terminated, truncated, _ = branch_env.step(action)
                        receipts.append(copy.deepcopy(receipt))
                        ledger.count("environment_steps", 1)
                        if terminated or truncated:
                            break
                    if not receipts:
                        raise ValueError("EVENT_RUNNER_EMPTY_BRANCH")
                    actual_end = start + len(receipts)
                    trajectory = _window_trajectory(branch_env, start, actual_end, candidate, receipts)
                    private_binding = {
                        "simulator_snapshot_sha256": branch_initial_sha256,
                        "event_tape_sha256": private_tape_sha256,
                    }
                    branch = {
                        "candidate_key": candidate["candidate_key"], "candidate": copy.deepcopy(candidate),
                        "trajectory": trajectory, "private_binding": private_binding,
                    }
                    branches.append(branch)
                    branch_stream.write({
                        "schema": "event-branch-trajectory/1", "window_id": window_id,
                        "parent_id": scenario["parent_id"], "split": scenario["split"],
                        "candidate_key": candidate["candidate_key"], "candidate": copy.deepcopy(candidate),
                        "trajectory": trajectory, "private_binding": private_binding,
                    })
                    ledger.count("branches", 1)
                window = {
                    "window_id": window_id, "parent_id": scenario["parent_id"],
                    "split": scenario["split"], "window_start": start, "window_end": planned_end,
                    "event_tape_id": event_tape_id, "public_observation": observation,
                    "private_binding": {
                        "simulator_snapshot_sha256": private_snapshot_sha256,
                        "event_tape_sha256": private_tape_sha256,
                    },
                    "history": history[-int(config["world_model"].get("history_length", 4)):],
                    "candidates": catalog, "branches": branches,
                    "reward_weights": copy.deepcopy(config["reward"]),
                }
                labels = recompute_consequence_labels(window)
                label_stream.write(labels)
                if scenario["split"] == "train":
                    sample_stream.write(make_model_sample(window, labels))
                    sample_count += 1
                else:
                    development_sample_stream.write(make_model_sample(window, labels))
                    development_sample_count += 1
                report = {
                    "window_id": window_id, "parent_id": scenario["parent_id"], "split": scenario["split"],
                    "window_start": start, "window_end": planned_end, "event_tape_id": event_tape_id,
                    "snapshot_hash": snapshot_hash, "candidate_count": len(catalog),
                    "legal_candidate_count": len(branches),
                    "candidate_keys": [row["candidate_key"] for row in catalog],
                    "rejected_candidates": rejected,
                    "valid_window": labels["valid_window"],
                    "distinct_legal_outcome_count": labels["distinct_legal_outcome_count"],
                    "has_two_distinct_legal_outcomes": labels["valid_window_has_two_distinct_legal_outcomes"],
                    "outcome_pair_differences": copy.deepcopy(labels["outcome_pair_differences"]),
                    "target_task_ids": copy.deepcopy(labels["public_task_ids_at_start"]),
                    "non_target_truth_task_ids_audited": copy.deepcopy(labels["all_truth_task_ids_including_non_targets"]),
                }
                window_reports.append(report)
    finally:
        branch_stream.close()
        candidate_stream.close()
        label_stream.close()
        sample_stream.close()
        development_sample_stream.close()
    train_reports = [row for row in window_reports if row["split"] == "train"]
    diverse_train = sum(bool(row["has_two_distinct_legal_outcomes"]) for row in train_reports)
    valid_train = sum(bool(row["valid_window"]) for row in train_reports)
    train_fraction = diverse_train / 16
    all_diverse = sum(bool(row["has_two_distinct_legal_outcomes"]) for row in window_reports)
    overall_fraction = all_diverse / 24
    minimum_fraction = float(config["collection"]["minimum_distinct_window_fraction"])
    window_counts_valid = len(window_reports) == 24 and len(train_reports) == 16
    gate_passed = window_counts_valid and train_fraction >= minimum_fraction
    first_failure = next((row for row in train_reports if not row["has_two_distinct_legal_outcomes"]), None)
    summary = {
        "schema": "event-candidate-reconstruction-report/1", "stage": "M2",
        "status": "complete" if gate_passed else "stopped_on_distinctness_gate",
        "planned_window_count": 24, "observed_window_count": len(window_reports),
        "planned_train_window_count": 16, "observed_train_window_count": len(train_reports),
        "planned_window_denominator": 24, "planned_train_window_denominator": 16,
        "planned_window_counts_valid": window_counts_valid,
        "valid_train_window_count": valid_train,
        "train_windows_with_two_distinct_legal_outcomes": diverse_train,
        "train_diversity_fraction_over_all_planned_train_windows": train_fraction,
        "all_parent_windows_with_two_distinct_legal_outcomes": all_diverse,
        "all_parent_diversity_fraction_over_all_planned_windows": overall_fraction,
        "minimum_train_diversity_fraction": minimum_fraction,
        "gate_passed": gate_passed,
        "development_diversity_is_report_only": True,
        "first_training_window_without_two_distinct_outcomes": first_failure,
        "windows": window_reports,
        "model_sample_count": sample_count,
        "development_model_sample_count": development_sample_count,
        "oracle_schema": "event-consequence-oracle/1",
    }
    _write_json(ledger.output / "CANDIDATE_RECONSTRUCTION_REPORT.json", summary)
    if not gate_passed:
        first = {
            "stage": "m2", "type": "DistinctOutcomeGateError",
            "message": f"training diversity {diverse_train}/16 is below {minimum_fraction:.3f} or planned windows are missing",
            "timestamp_utc": _utc_now(), "window_id": first_failure["window_id"] if first_failure else None,
            "evidence": summary,
        }
        ledger.run["ledger"]["first_error"] = ledger.run["ledger"].get("first_error") or first
        _write_json(ledger.output / "FIRST_FAILURE.json", ledger.run["ledger"]["first_error"])
        ledger._save()
        raise DistinctOutcomeGateError(first["message"])
    return summary


class DistinctOutcomeGateError(RuntimeError):
    pass


def _fresh_process_json(script: str, request_path: Path, timeout_seconds: int) -> dict[str, Any]:
    module_root = str(Path(__file__).resolve().parent)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = module_root + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(request_path.resolve())],
        cwd=module_root, env=environment, capture_output=True, text=True,
        timeout=max(1, int(timeout_seconds)), check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "EVENT_RUNNER_FRESH_PROCESS_FAILED:"
            + completed.stderr[-4000:]
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("EVENT_RUNNER_FRESH_PROCESS_NO_JSON")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("EVENT_RUNNER_FRESH_PROCESS_JSON_OBJECT_REQUIRED")
    return result


def _prediction_digest(predictions: dict[str, Any]) -> str:
    return _sha256(predictions)


def _value_row(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _mask_row(value: Any) -> list[bool]:
    if isinstance(value, (list, tuple)):
        return [bool(item) for item in value]
    return [bool(value)]


def _evaluate_world_model(samples: list[dict[str, Any]], ensemble: Any, ledger: RunLedger) -> dict[str, Any]:
    try:
        from .event_models import predict_ensemble
    except ImportError:
        from event_models import predict_ensemble
    def fresh_head_accumulators() -> dict[str, dict[str, Any]]:
        return {
            head: {
                "absolute_error": 0.0, "squared_error": 0.0, "variance": 0.0,
                "bce_sum": 0.0,
                "valid_values": 0, "target_slots": 0, "one_sigma_covered": 0,
                "two_sigma_covered": 0, "calibration_rows": [],
            }
            for head in MODEL_HEADS
        }

    accumulators = fresh_head_accumulators()
    parent_accumulators: dict[str, dict[str, Any]] = {}
    sample_reports = []
    pairwise_concordance: list[float] = []
    utility_regrets: list[float] = []
    utility_best_values: list[float] = []
    utility_selected_values: list[float] = []
    utility_oracle_best_hits: list[bool] = []
    evaluated_candidates = 0
    for sample in samples:
        parent_id = str(sample.get("parent_id", ""))
        parent = parent_accumulators.setdefault(parent_id, {
            "sample_count": 0, "candidate_count": 0,
            "heads": fresh_head_accumulators(), "pairwise_concordance": [],
            "utility_regrets": [], "utility_best_values": [], "utility_selected_values": [],
            "oracle_best_hits": [],
        })
        parent["sample_count"] += 1
        predictions = predict_ensemble(ensemble, sample["history"], sample["candidates"])
        ledger.count("world_model_forwards", 3)
        candidate_keys = [str(row.get("candidate_key", "")) for row in sample["candidates"]]
        if predictions.get("candidate_keys") != candidate_keys:
            raise ValueError("EVENT_RUNNER_WORLD_MODEL_CANDIDATE_ORDER")
        evaluated_candidates += len(candidate_keys)
        parent["candidate_count"] += len(candidate_keys)
        sample_metrics: dict[str, Any] = {}
        utility_rows: list[tuple[int, float, float]] = []
        for head in MODEL_HEADS:
            target_rows = sample["targets"][head]["values"]
            mask_rows = sample["targets"][head]["valid"]
            prediction_rows = predictions["means"][head]
            variance_rows = predictions["variances"][head]
            for target, mask, prediction, variance in zip(target_rows, mask_rows, prediction_rows, variance_rows):
                truth_values = _value_row(target)
                valid_values = _mask_row(mask)
                predicted_values = _value_row(prediction)
                variance_values = _value_row(variance)
                if not (len(truth_values) == len(valid_values) == len(predicted_values) == len(variance_values)):
                    raise ValueError(f"EVENT_RUNNER_WORLD_MODEL_METRIC_SHAPE:{head}")
                for values in (accumulators[head], parent["heads"][head]):
                    values["target_slots"] += len(valid_values)
                for truth, valid, predicted, variance_value in zip(truth_values, valid_values, predicted_values, variance_values):
                    if not valid:
                        continue
                    variance_value = max(0.0, variance_value)
                    error = predicted - truth
                    standard_deviation = math.sqrt(variance_value)
                    for values in (accumulators[head], parent["heads"][head]):
                        values["absolute_error"] += abs(error)
                        values["squared_error"] += error * error
                        if head in {"failure", "communication_risk", "constraint_risk"}:
                            probability = min(1.0 - 1e-12, max(1e-12, predicted))
                            target_probability = min(1.0, max(0.0, truth))
                            values["bce_sum"] += -(
                                target_probability * math.log(probability)
                                + (1.0 - target_probability) * math.log(1.0 - probability)
                            )
                        values["variance"] += variance_value
                        values["valid_values"] += 1
                        values["one_sigma_covered"] += abs(error) <= standard_deviation
                        values["two_sigma_covered"] += abs(error) <= 2.0 * standard_deviation
                        values["calibration_rows"].append((variance_value, error * error))
                if head == "utility" and truth_values and valid_values and valid_values[0]:
                    utility_rows.append((len(utility_rows), truth_values[0], predicted_values[0]))
            target_slots = sum(len(_mask_row(row)) for row in mask_rows)
            valid_values = sum(sum(_mask_row(row)) for row in mask_rows)
            sample_metrics[head] = {
                "target_slots": target_slots,
                "valid_values": valid_values,
                "unknown_values": target_slots - valid_values,
                "unknown_fraction": (target_slots - valid_values) / target_slots if target_slots else None,
            }
        if utility_rows:
            oracle_best_value = max(row[1] for row in utility_rows)
            oracle_best_indices = {
                index for index, row in enumerate(utility_rows)
                if math.isclose(row[1], oracle_best_value, rel_tol=1e-9, abs_tol=1e-9)
            }
            predicted_best_value = max(row[2] for row in utility_rows)
            predicted_best_indices = {
                index for index, row in enumerate(utility_rows)
                if math.isclose(row[2], predicted_best_value, rel_tol=1e-9, abs_tol=1e-9)
            }
            oracle_best_hit = bool(oracle_best_indices & predicted_best_indices)
            parent["oracle_best_hits"].append(oracle_best_hit)
            utility_oracle_best_hits.append(oracle_best_hit)
        if len(utility_rows) >= 2:
            pair_scores = []
            for left_index in range(len(utility_rows)):
                for right_index in range(left_index + 1, len(utility_rows)):
                    _, left_truth, left_prediction = utility_rows[left_index]
                    _, right_truth, right_prediction = utility_rows[right_index]
                    truth_delta = left_truth - right_truth
                    if math.isclose(truth_delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
                        continue
                    prediction_delta = left_prediction - right_prediction
                    pair_scores.append(
                        0.5 if math.isclose(prediction_delta, 0.0, rel_tol=1e-9, abs_tol=1e-9)
                        else float((truth_delta > 0.0) == (prediction_delta > 0.0))
                    )
            if pair_scores:
                score = sum(pair_scores) / len(pair_scores)
                pairwise_concordance.append(score)
                parent["pairwise_concordance"].append(score)
            selected_index = max(range(len(utility_rows)), key=lambda index: utility_rows[index][2])
            best_value = max(row[1] for row in utility_rows)
            selected_value = utility_rows[selected_index][1]
            regret = max(0.0, best_value - selected_value)
            predicted_best_value = utility_rows[selected_index][2]
            utility_regrets.append(regret)
            utility_best_values.append(best_value)
            utility_selected_values.append(selected_value)
            parent["utility_regrets"].append(regret)
            parent["utility_best_values"].append(best_value)
            parent["utility_selected_values"].append(selected_value)
        sample_reports.append({"sample_id": sample.get("sample_id"), "parent_id": parent_id, "heads": sample_metrics})

    def summarize_heads(source: dict[str, dict[str, Any]]) -> dict[str, Any]:
        head_metrics: dict[str, Any] = {}
        for head, row in source.items():
            count = int(row["valid_values"])
            calibration_rows = sorted(row["calibration_rows"], key=lambda item: item[0])
            calibration_bins = []
            bin_count = min(5, len(calibration_rows))
            for index in range(bin_count):
                start = index * len(calibration_rows) // bin_count
                end = (index + 1) * len(calibration_rows) // bin_count
                selected = calibration_rows[start:end]
                mean_variance = sum(item[0] for item in selected) / len(selected)
                empirical_mse = sum(item[1] for item in selected) / len(selected)
                calibration_bins.append({
                    "count": len(selected), "mean_ensemble_variance": mean_variance,
                    "empirical_mse": empirical_mse,
                    "empirical_to_predicted_variance_ratio": empirical_mse / mean_variance if mean_variance > 1e-12 else None,
                })
            mean_variance = row["variance"] / count if count else None
            empirical_mse = row["squared_error"] / count if count else None
            head_metrics[head] = {
                "target_slots": int(row["target_slots"]),
                "valid_values": count,
                "unknown_values": int(row["target_slots"] - count),
                "valid_fraction": count / int(row["target_slots"]) if row["target_slots"] else None,
                "unknown_fraction": (
                    (int(row["target_slots"]) - count) / int(row["target_slots"])
                    if row["target_slots"] else None
                ),
                "mae": row["absolute_error"] / count if count else None,
                "mse": empirical_mse,
                "rmse": math.sqrt(empirical_mse) if empirical_mse is not None else None,
                "masked_bce": (
                    row["bce_sum"] / count
                    if head in {"failure", "communication_risk", "constraint_risk"} and count else None
                ),
                "bce": (
                    row["bce_sum"] / count
                    if head in {"failure", "communication_risk", "constraint_risk"} and count else None
                ),
                "mean_ensemble_variance": mean_variance,
                "uncertainty_calibration": {
                    "empirical_mse_to_mean_variance_ratio": (
                        empirical_mse / mean_variance
                        if empirical_mse is not None and mean_variance is not None and mean_variance > 1e-12
                        else None
                    ),
                    "one_sigma_empirical_coverage": row["one_sigma_covered"] / count if count else None,
                    "two_sigma_empirical_coverage": row["two_sigma_covered"] / count if count else None,
                    "variance_quantile_bins": calibration_bins,
                },
            }
        return head_metrics

    def summarize_utility(source: dict[str, list[float]]) -> dict[str, Any]:
        regret_count = len(source["utility_regrets"])
        concordance = source["pairwise_concordance"]
        return {
            "utility_ranking": {
                "candidate_sets_with_non_tied_pairs": len(concordance),
                "mean_pairwise_concordance": sum(concordance) / len(concordance) if concordance else None,
                "paired_set_count": regret_count,
            },
            "utility_regret": {
                "paired_set_count": regret_count,
                "mean_regret_from_predicted_best": sum(source["utility_regrets"]) / regret_count if regret_count else None,
                "mean_oracle_best_utility": sum(source["utility_best_values"]) / regret_count if regret_count else None,
                "mean_predicted_selection_utility": sum(source["utility_selected_values"]) / regret_count if regret_count else None,
            },
            "oracle_best_candidate_hit_rate": {
                "hit_count": sum(bool(value) for value in source["oracle_best_hits"]),
                "denominator": len(source["oracle_best_hits"]),
                "hit_rate": (
                    sum(bool(value) for value in source["oracle_best_hits"]) / len(source["oracle_best_hits"])
                    if source["oracle_best_hits"] else None
                ),
                "tie_policy": "any_shared_candidate_among_predicted_and_oracle_ties",
            },
        }

    head_metrics = summarize_heads(accumulators)
    overall_utility = summarize_utility({
        "pairwise_concordance": pairwise_concordance,
        "utility_regrets": utility_regrets,
        "utility_best_values": utility_best_values,
        "utility_selected_values": utility_selected_values,
        "oracle_best_hits": utility_oracle_best_hits,
    })
    parent_metrics = {
        parent_id: {
            "sample_count": values["sample_count"],
            "candidate_count": values["candidate_count"],
            "heads": summarize_heads(values["heads"]),
            **summarize_utility(values),
        }
        for parent_id, values in sorted(parent_accumulators.items())
    }
    return {
        "heads": head_metrics,
        "sample_count": len(samples),
        "parent_count": len({str(row.get("parent_id", "")) for row in samples}),
        "parent_ids": sorted({str(row.get("parent_id", "")) for row in samples}),
        "per_parent": parent_metrics,
        "candidate_count": evaluated_candidates,
        "unknown_fraction": (
            sum(int(row["unknown_values"]) for row in head_metrics.values())
            / sum(int(row["target_slots"]) for row in head_metrics.values())
            if sum(int(row["target_slots"]) for row in head_metrics.values()) else None
        ),
        **overall_utility,
        "sample_reports": sample_reports,
    }


_WM_RESTORE_SCRIPT = r"""
import hashlib, json, sys
import torch
from event_models import load_ensemble, predict_ensemble
with open(sys.argv[1], encoding='utf-8') as handle:
    request = json.load(handle)
torch.set_num_threads(1)
ensemble = load_ensemble(request['checkpoint_paths'])
predictions = predict_ensemble(ensemble, request['history'], request['candidates'])
payload = json.dumps(predictions, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
print(json.dumps({'status':'complete','ensemble_hash':ensemble.ensemble_hash,
                  'prediction_sha256':hashlib.sha256(payload).hexdigest(),
                  'predictions':predictions}, sort_keys=True, separators=(',', ':'), allow_nan=False))
"""


_POLICY_RESTORE_SCRIPT = r"""
import hashlib, json, sys
import torch
from event_models import load_ensemble
from event_policy import verify_policy_checkpoint
with open(sys.argv[1], encoding='utf-8') as handle:
    request = json.load(handle)
torch.set_num_threads(1)
ensemble = load_ensemble(request['ensemble_checkpoint_paths']) if request.get('ensemble_checkpoint_paths') else None
checks = []
for checkpoint in request['checkpoints']:
    path = checkpoint['path']
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if digest != checkpoint['sha256']:
        raise ValueError('EVENT_RUNNER_POLICY_CHECKPOINT_FILE_DIGEST')
    checks.append(verify_policy_checkpoint(path, ensemble=ensemble))
print(json.dumps({'status':'complete','checks':checks}, sort_keys=True, separators=(',', ':'), allow_nan=False))
"""


def _run_m3(scenarios: list[dict[str, Any]], config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    _require_stage(ledger.run, "m2")
    gate = _read_json(ledger.output / "CANDIDATE_RECONSTRUCTION_REPORT.json")
    if gate.get("gate_passed") is not True:
        raise ValueError("EVENT_RUNNER_M3_DISTINCTNESS_GATE_REQUIRED")
    samples = _read_jsonl(ledger.output / "M2_MODEL_SAMPLES.jsonl")
    development_samples = _read_jsonl(ledger.output / "M2_DEVELOPMENT_MODEL_SAMPLES.jsonl")
    train_parent_ids = sorted(row["parent_id"] for row in scenarios if row["split"] == "train")
    if len(samples) != 16 or {row.get("split") for row in samples} != {"train"} or {row.get("parent_id") for row in samples} != set(train_parent_ids):
        raise ValueError("EVENT_RUNNER_M3_TRAINING_SAMPLE_SET")
    development_parent_ids = sorted(row["parent_id"] for row in scenarios if row["split"] == "development")
    if (len(development_samples) != 8
            or {row.get("split") for row in development_samples} != {"development"}
            or {row.get("parent_id") for row in development_samples} != set(development_parent_ids)):
        raise ValueError("EVENT_RUNNER_M3_DEVELOPMENT_SAMPLE_SET")
    try:
        from .event_models import load_ensemble, predict_ensemble, train_ensemble
        from .event_policy import world_model_trigger_summary
        from .event_trigger import calibrate_threshold
    except ImportError:
        from event_models import load_ensemble, predict_ensemble, train_ensemble
        from event_policy import world_model_trigger_summary
        from event_trigger import calibrate_threshold
    training = train_ensemble(samples, ledger.output / "WORLD_MODEL", config["world_model"])
    checkpoint_records = training.get("checkpoints", [])
    if len(checkpoint_records) != 3 or training.get("status") != "complete":
        raise ValueError("EVENT_RUNNER_M3_CHECKPOINT_SET")
    training_forwards = int(training.get("training_forward_calls", 0))
    ledger.count("world_model_forwards", training_forwards)
    ledger.count("optimizer_updates", sum(int(row.get("optimizer_updates", 0)) for row in checkpoint_records))
    if any(float(row.get("parameter_delta_l2", 0.0)) <= 0.0 for row in checkpoint_records):
        raise ValueError("EVENT_RUNNER_M3_MODEL_PARAMETERS_UNCHANGED")
    ensemble = load_ensemble(training["checkpoint_paths"])
    training_evaluation = _evaluate_world_model(samples, ensemble, ledger)
    probe = samples[0]
    expected = predict_ensemble(ensemble, probe["history"], probe["candidates"])
    ledger.count("world_model_forwards", 3)
    restore_request = {
        "checkpoint_paths": training["checkpoint_paths"],
        "history": probe["history"], "candidates": probe["candidates"],
        "expected_prediction_sha256": _prediction_digest(expected),
        "expected_ensemble_hash": ensemble.ensemble_hash,
    }
    restore_path = ledger.output / "WORLD_MODEL_RESTORE_REQUEST.json"
    _write_json(restore_path, restore_request)
    restored = _fresh_process_json(_WM_RESTORE_SCRIPT, restore_path, int(config["runtime"]["wall_timeout_seconds"]))
    if restored.get("status") != "complete" or restored.get("ensemble_hash") != ensemble.ensemble_hash or restored.get("prediction_sha256") != restore_request["expected_prediction_sha256"]:
        raise ValueError("EVENT_RUNNER_M3_FRESH_PROCESS_RESTORE_MISMATCH")
    ledger.count("world_model_forwards", 3)
    restore_report = {
        "status": "complete", "fresh_process": True,
        "ensemble_hash": restored["ensemble_hash"],
        "prediction_sha256": restored["prediction_sha256"],
        "checkpoint_paths": training["checkpoint_paths"],
    }
    _write_json(ledger.output / "WORLD_MODEL_RESTORE_CHECK.json", restore_report)

    sequences = _read_jsonl(ledger.output / "TRAINING_OBSERVATION_SEQUENCES.jsonl")
    if len(sequences) != 4:
        raise ValueError("EVENT_RUNNER_M3_THRESHOLD_SEQUENCE_SET")
    calibration_sequences = copy.deepcopy(sequences)
    calibration_forwards = 0
    for sequence in calibration_sequences:
        observations = sequence.get("observations", [])
        catalogs = sequence.get("candidate_catalogs", [])
        if len(observations) < 2 or len(catalogs) != len(observations):
            raise ValueError("EVENT_RUNNER_M3_THRESHOLD_CATALOG_ALIGNMENT")
        summaries: list[dict[str, Any] | None] = [None] * len(observations)
        for index in range(1, len(observations)):
            history = observations[max(0, index - 3):index + 1]
            prediction = predict_ensemble(ensemble, history, catalogs[index])
            calibration_forwards += 3
            summaries[index] = world_model_trigger_summary(prediction, catalogs[index])
        sequence["wm_summaries"] = summaries
    ledger.count("world_model_forwards", calibration_forwards)
    threshold = calibrate_threshold(calibration_sequences, _trigger_config(config))
    if threshold.get("wm_summaries_used") is not True or threshold.get("source_parents") != train_parent_ids:
        raise ValueError("EVENT_RUNNER_M3_FINAL_THRESHOLD_PROVENANCE")
    threshold.update({
        "calibration_stage": "M3_final_world_model",
        "source_split": "train",
        "source_parent_ids": train_parent_ids,
        "world_model_used": True,
        "world_model_hash": ensemble.ensemble_hash,
        "world_model_forward_calls_for_calibration": calibration_forwards,
        "initial_threshold_sha256": _sha256(_read_json(ledger.output / "THRESHOLD_MANIFEST_INITIAL.json")),
        "pre_model_threshold": float(_read_json(ledger.output / "THRESHOLD_MANIFEST_INITIAL.json")["threshold"]),
        "frozen_before_development_evaluation": True,
    })
    _write_json(ledger.output / "THRESHOLD_MANIFEST_FINAL.json", threshold)
    _write_json(ledger.output / "THRESHOLD_MANIFEST.json", threshold)
    development_evaluation = _evaluate_world_model(development_samples, ensemble, ledger)
    metrics = {
        "schema": "event-world-model-ensemble-metrics/1",
        "status": "complete", "training": training,
        "evaluation_sample_counts": {
            "training_samples_used_for_fit_and_reconstruction_check": len(samples),
            "development_samples_held_out_from_fit_and_threshold_calibration": len(development_samples),
            "development_parent_ids": development_parent_ids,
        },
        "evaluation_on_training_samples": training_evaluation,
        "evaluation_on_development_samples": development_evaluation,
        "checkpoint_restore": restore_report,
        "ensemble_hash": ensemble.ensemble_hash,
        "final_threshold_manifest_sha256": _sha256(threshold),
        "threshold_calibration": {
            "stage": threshold["calibration_stage"], "source_split": threshold["source_split"],
            "source_parent_ids": threshold["source_parent_ids"],
            "threshold": threshold["threshold"], "wm_summaries_used": threshold["wm_summaries_used"],
            "training_input_sha256": threshold["training_input_sha256"],
        },
    }
    _write_json(ledger.output / "WORLD_MODEL_ENSEMBLE_METRICS.json", metrics)
    return metrics


def _development_rows(policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for variant, seeds in policy.get("variants", {}).items():
        rows = []
        for seed, record in seeds.items():
            for episode in record.get("development", []):
                rows.append({"variant": variant, "seed": int(seed), **copy.deepcopy(episode)})
        result[variant] = rows
    return result


def _development_metric_means(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, episodes in rows.items():
        values: dict[str, list[float]] = {}
        for episode in episodes:
            for name, value in episode.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    values.setdefault(name, []).append(float(value))
            environment_metrics = episode.get("environment_metrics", {})
            if isinstance(environment_metrics, dict):
                for name, value in environment_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                        values.setdefault(name, []).append(float(value))
        result[variant] = {
            name: {"mean": sum(numbers) / len(numbers), "count": len(numbers)}
            for name, numbers in values.items() if numbers
        }
    return result


def _run_m4(scenarios: list[dict[str, Any]], config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    _require_stage(ledger.run, "m3")
    try:
        from .event_models import load_ensemble
        from .event_policy import train_four_variants
    except ImportError:
        from event_models import load_ensemble
        from event_policy import train_four_variants
    model_metrics = _read_json(ledger.output / "WORLD_MODEL_ENSEMBLE_METRICS.json")
    checkpoint_paths = model_metrics["training"]["checkpoint_paths"]
    ensemble = load_ensemble(checkpoint_paths)
    threshold = _read_json(ledger.output / "THRESHOLD_MANIFEST_FINAL.json")
    if threshold.get("world_model_hash") != ensemble.ensemble_hash:
        raise ValueError("EVENT_RUNNER_M4_THRESHOLD_MODEL_BINDING")
    policy = train_four_variants(scenarios, ensemble, threshold, ledger.output / "POLICY", config)
    if policy.get("status") != "complete":
        raise ValueError("EVENT_RUNNER_M4_POLICY_TRAINING_INCOMPLETE")
    ensemble_forwards = int(policy.get("ensemble_forward_counts", {}).get("total_forwards", 0))
    ledger.count("world_model_forwards", ensemble_forwards)
    checkpoints = []
    policy_forward_count = 0
    same_process_restore_forwards = 0
    update_count = 0
    for variant, seeds in policy.get("variants", {}).items():
        for seed, record in seeds.items():
            training = record.get("training", {})
            counters = training.get("forward_counts", {})
            policy_forward_count += int(counters.get("total_actor", 0)) + int(counters.get("total_critic", 0))
            update_count += int(training.get("optimizer_updates", 0))
            checkpoint = record.get("checkpoint", {})
            checkpoint_forward_counts = checkpoint.get("forward_counts", {})
            same_process_restore_forwards += (
                int(checkpoint_forward_counts.get("actor", 0))
                + int(checkpoint_forward_counts.get("critic", 0))
            )
            checkpoints.append({"variant": variant, "seed": int(seed), "path": checkpoint.get("path"), "sha256": checkpoint.get("sha256")})
    if len(checkpoints) != 12 or any(not row.get("path") or not row.get("sha256") for row in checkpoints):
        raise ValueError("EVENT_RUNNER_M4_CHECKPOINT_SET")
    restore_request = {
        "ensemble_checkpoint_paths": checkpoint_paths,
        "checkpoints": checkpoints,
    }
    restore_path = ledger.output / "M4_POLICY_RESTORE_REQUEST.json"
    _write_json(restore_path, restore_request)
    restored = _fresh_process_json(_POLICY_RESTORE_SCRIPT, restore_path, int(config["runtime"]["wall_timeout_seconds"]))
    restore_checks = restored.get("checks", [])
    if restored.get("status") != "complete" or len(restore_checks) != len(checkpoints) or any(row.get("status") != "complete" for row in restore_checks):
        raise ValueError("EVENT_RUNNER_M4_FRESH_PROCESS_RESTORE_MISMATCH")
    fresh_policy_forwards = sum(int(row.get("actor_forward_calls", 0)) + int(row.get("critic_forward_calls", 0)) for row in restore_checks)
    policy_forward_count += fresh_policy_forwards
    policy_forward_count += same_process_restore_forwards
    ledger.count("policy_forwards", policy_forward_count)
    ledger.count("optimizer_updates", update_count)
    training_environment_steps = sum(
        int(episode.get("ticks", episode.get("steps", 0)))
        for seeds in policy.get("variants", {}).values()
        for record in seeds.values()
        for episode in record.get("training", {}).get("episodes", [])
    )
    development_environment_steps = sum(
        int(episode.get("ticks", episode.get("steps", 0)))
        for seeds in policy.get("variants", {}).values()
        for record in seeds.values()
        for episode in record.get("development", [])
    )
    ledger.count("environment_steps", training_environment_steps + development_environment_steps)
    restore_report = {
        "status": "complete", "fresh_process": True,
        "checkpoint_count": len(restore_checks), "checks": restore_checks,
    }
    _write_json(ledger.output / "M4_POLICY_RESTORE_CHECK.json", restore_report)
    policy["fresh_process_restore_check"] = restore_report
    policy["runner_resource_counts"] = {
        "world_model_forwards": ensemble_forwards,
        "policy_actor_and_critic_forwards_including_fresh_restore": policy_forward_count,
        "policy_same_process_checkpoint_restore_forwards": same_process_restore_forwards,
        "policy_fresh_process_restore_forwards": fresh_policy_forwards,
        "environment_steps_training": training_environment_steps,
        "environment_steps_development": development_environment_steps,
        "optimizer_updates": update_count,
    }
    _write_json(ledger.output / "EVENT_POLICY_TRAINING.json", policy)

    development = _development_rows(policy)
    means = _development_metric_means(development)
    _write_json(ledger.output / "DEVELOPMENT_EVALUATION.json", {
        "schema": "event-development-evaluation/1", "status": "complete",
        "claim_scope": config.get("analysis", {}).get("claim_scope"),
        "development_parent_count": 2, "seed_count": len(config["ppo"]["seeds"]),
        "episodes": development, "descriptive_means": means,
        "fixed_snapshot_probes": {
            variant: {seed: record.get("fixed_snapshot_probes", []) for seed, record in seeds.items()}
            for variant, seeds in policy.get("variants", {}).items()
        },
    })
    comparisons = []
    for left, right in (("Event-G", "Periodic-G"), ("Event-G-WM", "Event-G"), ("Event-G-WM", "Periodic-G")):
        metrics = sorted(set(means.get(left, {})) & set(means.get(right, {})))
        comparisons.append({
            "left": left, "right": right,
            "metric_mean_differences_left_minus_right": {
                name: means[left][name]["mean"] - means[right][name]["mean"]
                for name in metrics
            },
        })
    _write_json(ledger.output / "G_GWM_COMPARISON.json", {
        "schema": "event-g-gwm-comparison/1", "status": "descriptive_only",
        "semantics": {
            "Periodic-G": "periodic trigger with graph policy and no world model",
            "Event-G": "event trigger with graph policy and no world model",
            "Periodic-G-WM": "periodic trigger with model-assisted critic; periodic timing fixed",
            "Event-G-WM": "model-assisted trigger and model-assisted critic",
            "Event-G-WM_vs_Event-G": "combined model-assisted trigger and critic contribution; critic effect is not isolated",
        },
        "development_descriptive_means": means,
        "comparisons": comparisons,
        "world_model_frozen": all(
            bool(record.get("training", {}).get("world_model_frozen"))
            for seeds in policy.get("variants", {}).values() for record in seeds.values()
        ),
        "optimizer_updates_per_seed_method": policy.get("paired_optimizer_updates_per_seed_method"),
        "updates_per_epoch_by_round": policy.get("updates_per_epoch_by_round"),
    })
    _write_json(ledger.output / "SOL_HIGH_REVIEW.json", {
        "schema": "event-sol-high-review/1", "review_status": "pending_independent_read_only_review",
        "requested_reviewer": "GPT-6.1 Sol High",
        "audit_targets": [
            "trigger execution and information boundary", "candidate count and physical outcome diversity",
            "world-model training and fresh-process restore", "model-assisted trigger and critic wiring",
            "constraint repair", "paired policy budgets", "development-only claim limits", "resource settlement",
        ],
        "artifacts_available": list(ARTIFACTS),
    })
    return {"status": "complete", "policy": policy, "restore_check": restore_report, "development_means": means}


def _run_restore_check(config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    _require_stage(ledger.run, "m3")
    _require_stage(ledger.run, "m4")
    wm_request_path = ledger.output / "WORLD_MODEL_RESTORE_REQUEST.json"
    policy_request_path = ledger.output / "M4_POLICY_RESTORE_REQUEST.json"
    wm_request = _read_json(wm_request_path)
    policy_request = _read_json(policy_request_path)
    wm_result = _fresh_process_json(_WM_RESTORE_SCRIPT, wm_request_path, int(config["runtime"]["wall_timeout_seconds"]))
    if wm_result.get("prediction_sha256") != wm_request.get("expected_prediction_sha256"):
        raise ValueError("EVENT_RUNNER_RESTORE_CHECK_WORLD_MODEL_MISMATCH")
    policy_result = _fresh_process_json(_POLICY_RESTORE_SCRIPT, policy_request_path, int(config["runtime"]["wall_timeout_seconds"]))
    if policy_result.get("status") != "complete" or len(policy_result.get("checks", [])) != len(policy_request.get("checkpoints", [])):
        raise ValueError("EVENT_RUNNER_RESTORE_CHECK_POLICY_MISMATCH")
    report = {
        "status": "complete", "fresh_process": True,
        "world_model": {"ensemble_hash": wm_result["ensemble_hash"], "prediction_sha256": wm_result["prediction_sha256"]},
        "policies": policy_result["checks"],
    }
    _write_json(ledger.output / "RESTORE_CHECK.json", report)
    ledger.count("world_model_forwards", 3)
    ledger.count("policy_forwards", sum(int(row.get("actor_forward_calls", 0)) + int(row.get("critic_forward_calls", 0)) for row in policy_result["checks"]))
    return report


def _write_final_report(output: Path, config: dict[str, Any]) -> None:
    m2_path = output / "CANDIDATE_RECONSTRUCTION_REPORT.json"
    m2 = _read_json(m2_path) if m2_path.is_file() else {}
    m1_path = output / "CANDIDATE_RECONSTRUCTION_REPORT_M1.json"
    m1 = _read_json(m1_path) if m1_path.is_file() else {}
    wm_path = output / "WORLD_MODEL_ENSEMBLE_METRICS.json"
    wm = _read_json(wm_path) if wm_path.is_file() else {}
    policy_path = output / "EVENT_POLICY_TRAINING.json"
    policy = _read_json(policy_path) if policy_path.is_file() else {}
    settlement_path = output / "RESOURCE_SETTLEMENT.json"
    settlement = _read_json(settlement_path) if settlement_path.is_file() else {}
    stage_records = settlement.get("stages", {})
    stage_status = {
        name: stage_records.get(name, {}).get("status", "not_started")
        for name in STAGE_ORDER
    }
    # A failed M2 may leave the prior M1 summary at this path. Only treat a
    # report explicitly marked M2 as an M2 summary, so failure reporting cannot
    # raise a secondary KeyError and hide the first error.
    if m2.get("stage") != "M2":
        m2 = {}
    if m2:
        m2_status = (
            f"{m2.get('status', 'unknown')}; train diversity "
            f"{m2.get('train_windows_with_two_distinct_legal_outcomes', 'unknown')}/"
            f"{m2.get('planned_train_window_denominator', 'unknown')}; gate passed "
            f"{m2.get('gate_passed', 'unknown')}"
        )
        m2_development = (
            f"M2 development samples: {m2.get('development_model_sample_count', 'unknown')}/8."
        )
    else:
        m2_status = f"{stage_status['m2']} (summary unavailable; counts unavailable)"
        m2_development = "M2 development samples: not available."
    first_error = settlement.get("first_error")
    lines = [
        "# N02 Event-Trigger Experiment Report", "",
        "Stage status: " + ", ".join(f"{name}={stage_status[name]}" for name in STAGE_ORDER) + ".",
        f"M2: {m2_status}.",
        m2_development,
        *(
            [f"M2 all-window diversity: {m2.get('all_parent_windows_with_two_distinct_legal_outcomes', 'unknown')}/{m2.get('planned_window_denominator', m2.get('planned_train_window_denominator', 'unknown'))} (development diversity is report-only)."]
            if m2 else []
        ),
        f"World-model ensemble: {wm.get('status', 'not run')}; hash: {wm.get('ensemble_hash', 'unavailable')}.",
        f"Policy training: {policy.get('status', 'not run')}; paired updates per seed/method: {policy.get('paired_optimizer_updates_per_seed_method', 'unavailable')}.",
        f"Independent Sol High review: {('complete' if (output / 'SOL_HIGH_REVIEW.json').is_file() and _read_json(output / 'SOL_HIGH_REVIEW.json').get('review_status') == 'complete' else 'pending')}.",
        "",
        "Development summaries are descriptive for two held-out parents and do not support a broad superiority claim.",
        "Event-G-WM versus Event-G combines model-assisted trigger and auxiliary critic effects; the critic effect is not isolated.",
        "Configured reward targets are recomputed from raw trace primitives and checked against recorded step rewards.",
        "World-model development metrics are held out from fitting and threshold calibration.",
        "",
        f"M1 diagnostic episodes: {len(m1.get('episode_summaries', [])) if m1 else 'not available'}.",
        f"M1 trigger rows: {m1.get('row_count', 'unknown')}; event/periodic comparison: {m1.get('comparison_status', 'unavailable')}.",
        f"Resource totals: {json.dumps(settlement.get('totals', {}), sort_keys=True)}.",
    ]
    if isinstance(first_error, Mapping):
        lines.extend([
            "",
            f"First recorded failure: stage={first_error.get('stage', 'unknown')}; type={first_error.get('type', 'unknown')}; message={first_error.get('message', 'unavailable')}.",
        ])
    (output / "FINAL_EVENT_TRIGGER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _dispatch_stage(stage: str, config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    if stage == "m0":
        return _run_m0(config, ledger)
    scenarios = _build_scenarios(config)
    if stage == "m1":
        _require_stage(ledger.run, "m0")
        return _run_m1(scenarios, config, ledger)
    if stage == "m2":
        _require_stage(ledger.run, "m1")
        return _run_m2(scenarios, config, ledger)
    if stage == "m3":
        return _run_m3(scenarios, config, ledger)
    if stage == "m4":
        return _run_m4(scenarios, config, ledger)
    if stage == "restore-check":
        return _run_restore_check(config, ledger)
    raise ValueError(f"EVENT_RUNNER_UNKNOWN_STAGE:{stage}")


def run(stage: str, output: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    config, resolved_config, config_sha256 = load_config(config_path)
    output_path = Path(output).resolve()
    if stage == "all" or stage == "m0":
        run_record = _begin_run(output_path, resolved_config, config_sha256)
        stages = list(STAGE_ORDER) if stage == "all" else ["m0"]
    else:
        run_record = _resume_run(output_path, config_sha256)
        stages = [stage]
    ledger = RunLedger(output_path, run_record)
    result: dict[str, Any] = {"output": str(output_path), "stages": {}}
    try:
        for current in stages:
            ledger.begin(current)
            try:
                result["stages"][current] = _dispatch_stage(current, config, ledger)
                ledger.finish("complete")
            except BaseException as error:
                ledger.finish("failed", error)
                raise
        ledger.close()
        if stage in {"m4", "all", "restore-check"}:
            _write_final_report(output_path, config)
    except BaseException:
        ledger.close()
        try:
            _write_final_report(output_path, config)
        except Exception as report_error:
            ledger.run.setdefault("ledger", {})["final_report_error"] = f"{type(report_error).__name__}: {report_error}"
            ledger._save()
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run staged N02 event-trigger experiments.")
    parser.add_argument("stage", choices=(*STAGE_ORDER, "all", "restore-check"))
    parser.add_argument("--output", required=True, help="new output directory for m0/all; existing run directory for later stages")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="preregistered JSON config")
    args = parser.parse_args(argv)
    try:
        result = run(args.stage, args.output, args.config)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
