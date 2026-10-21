"""CLI entry point for V3 source-bound validation, smoke, and shard collection."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    from . import v3_shards as shards
except ImportError:
    import v3_shards as shards


PROJECT_ROOT = shards.PROJECT_ROOT
PACKAGE_ROOT = shards.PACKAGE_ROOT
CONFIG_PATH = shards.CONFIG_PATH
SOURCE_NAMES = ("Existing-ET", "Periodic-4min", "Random-Allow-Suppress")
TASK_MODES = ("search_only", "search_rescue")
TARGET_DENSITIES = (1, 3, 6)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise ValueError("V3_NONFINITE_JSON")
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"V3_UNSERIALIZABLE:{type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "gain-gate-v3-3-civilian-search-rescue-36platform":
        raise ValueError("V3_CONFIG_IDENTITY_INVALID")
    return value


def _current_rss() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _runtime_binding_report() -> dict[str, Any]:
    binding = shards.current_binding()
    config = _load_config()
    scenario = config["scenario"]
    return {
        "schema": "gain-gate-v33-binding-report/1",
        **binding,
        "source_manifest": shards.source_manifest(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sealed_splits": ["test", "ood"],
        "registry_count": len(shards.expected_registry(
            task_mode=scenario["default_task_mode"],
            targets_per_region=int(scenario["default_targets_per_region"]),
        )),
    }


def _online_plan(source: str, forced: bool, rng: random.Random, planner):
    """Decide suppression before the candidate catalog or planner is invoked."""
    allowed = forced or source != "Random-Allow-Suppress" or rng.random() < 0.5
    if not allowed:
        return False, None, 0.0
    started_cpu = time.process_time()
    receipt = planner()
    return True, receipt, max(0.0, time.process_time() - started_cpu)


def _scenario_factory():
    try:
        from .v33_scenarios import make_v33_scenario
    except ImportError:
        from v33_scenarios import make_v33_scenario
    return make_v33_scenario


def _platform_usage(trajectory_trace: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = {}
    for tick in trajectory_trace:
        public = tick.get("public_before", {})
        queues = public.get("queues", {})
        tasks = {str(row.get("id")): row for row in public.get("tasks", [])}
        for platform in public.get("platforms", []):
            platform_id = str(platform.get("id", ""))
            if not platform_id:
                continue
            row = usage.setdefault(platform_id, {
                "observed_ticks": 0, "active_task_ticks": 0,
                "assigned_queue_ticks": 0, "cross_region_queue_ticks": 0,
            })
            row["observed_ticks"] += 1
            if platform.get("active_task_id") is not None:
                row["active_task_ticks"] += 1
            queue = queues.get(platform_id, [])
            if queue:
                row["assigned_queue_ticks"] += 1
            if any(
                task_id in tasks and tasks[task_id].get("region_id") != platform.get("home_region")
                for task_id in queue
            ):
                row["cross_region_queue_ticks"] += 1
    return usage


def _scenario_and_row(spec: Mapping[str, Any], source: str, *, scenario: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Collect one trajectory and paired labels from the actual V3 simulator."""
    try:
        from . import counterfactual_pairs as pairs
        from .event_runner import _modules
        from .event_trigger import TriggerEngine
        from . import fixed_replanner, gain_safety, gain_oracle
    except ImportError:
        import counterfactual_pairs as pairs
        from event_runner import _modules
        from event_trigger import TriggerEngine
        import fixed_replanner, gain_safety, gain_oracle
    config = _load_config()
    if fixed_replanner.REPLANNER_VERSION != config["replanner"]["version"]:
        raise ValueError("V33_REPLANNER_IDENTITY_MISMATCH")
    if scenario is None:
        scenario = _scenario_factory()(
            split=str(spec["split"]), condition=str(spec["condition"]),
            seed_variant=int(spec["seed_variant"]), instance_index=int(spec["instance_index"]),
            task_mode=str(spec.get("task_mode", config["scenario"]["default_task_mode"])),
            targets_per_region=int(spec.get("targets_per_region", config["scenario"]["default_targets_per_region"])),
        )
    EventReliefEnv, _, _, identify_scope, _, _, _, _ = _modules()
    env = EventReliefEnv(scenario)
    observation = env.reset()
    trigger_config = config.get("trigger", {})
    engine = TriggerEngine(trigger_config, {"weights": trigger_config.get("weights", {
        "d_state": 0.35, "d_degradation": 0.30, "d_margin": 0.25, "d_uncertainty": 0.10,
    }), "threshold": trigger_config.get("threshold", 0.5), "delta_min": trigger_config.get("delta_min", 4)})
    rng_seed = int.from_bytes(hashlib.sha256(_canonical({"scenario": scenario["parent_id"], "source": source})).digest()[:8], "big")
    policy_rng = random.Random(rng_seed ^ 0x6A09E667F3BCC909)
    reservoir_rng = random.Random(rng_seed ^ 0xBB67AE8584CAA73B)
    selected_ordinary: list[dict[str, Any]] = []
    selected_forced: list[dict[str, Any]] = []
    ordinary_seen = 0
    source_action_counts = {"allowed": 0, "suppressed": 0}
    previous: dict[str, Any] | None = None
    terminated = truncated = False
    elapsed_ticks = 0
    trajectory_trace: list[dict[str, Any]] = []
    max_candidates = int(config["replanner"]["max_candidates"])
    safety_continuation_engines: dict[int, Any] = {}

    def continuation(branch_env: Any) -> Mapping[str, Any]:
        branch_observation = pairs.model_public_observation(branch_env)
        history = pairs._public_history(branch_env)
        prior = history[-2] if len(history) >= 2 else None
        branch_engine = safety_continuation_engines.get(id(branch_env))
        if branch_engine is None:
            branch_engine = TriggerEngine(trigger_config, engine.threshold_manifest)
            safety_continuation_engines[id(branch_env)] = branch_engine
        branch_detection = branch_engine.detect(prior, branch_observation, history=history[-4:])
        safety = gain_safety.assess_queue_safety(branch_observation)
        forced = bool(branch_detection.get("strong_event") or safety.get("forced") or not safety.get("safe_to_suppress", False))
        live_receipt = gain_safety.apply_safety_disposition(branch_env, branch_observation, safety)
        refreshed = pairs.model_public_observation(branch_env)
        safety_disposition = gain_safety.verify_live_safety_disposition(
            branch_env, branch_observation, safety, live_receipt, refreshed,
        )
        unknown_reason = safety_disposition.get("unknown_reason") if safety_disposition.get("resolved") is not True else None
        decision = branch_engine.decide(branch_detection, int(branch_env.time), mode="event")
        if not forced or safety_disposition.get("resolved") is not True:
            return {
                "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
                "action": copy.deepcopy(pairs.NOOP_ACTION), "planner_invoked": False,
                "planner_cpu_seconds": 0.0, "policy_version": "shared-safety-trigger-v1",
                "safety_report": safety, "safety_apply_receipt": live_receipt,
                "safety_disposition": safety_disposition, "unknown_reason": unknown_reason,
                "detection": branch_detection, "trigger_decision": decision,
            }
        scope = identify_scope(refreshed, branch_detection)
        started_cpu = time.process_time()
        receipt = fixed_replanner.replan(branch_env, affected_scope=scope, max_candidates=int(config["replanner"]["max_candidates"]))
        planner_cpu = max(0.0, time.process_time() - started_cpu)
        return {
            "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
            "action": receipt["selected_action"], "planner_invoked": True,
            "planner_cpu_seconds": planner_cpu, "policy_version": receipt.get("replanner_version"),
            "planner_receipt": receipt, "safety_report": safety,
            "safety_apply_receipt": live_receipt, "safety_disposition": safety_disposition,
            "unknown_reason": unknown_reason, "detection": branch_detection,
            "trigger_decision": decision,
        }

    while not (terminated or truncated):
        current = pairs.model_public_observation(env)
        detection = engine.detect(previous, current)
        mode = "periodic" if source == "Periodic-4min" else "event"
        decision = engine.decide(detection, int(env.time), mode=mode)
        safety = gain_safety.assess_queue_safety(current)
        forced = bool(detection.get("strong_event") or safety.get("forced") or not safety.get("safe_to_suppress", False))
        due = bool(decision.get("triggered")) and (int(env.time) > 0 or forced)
        action = copy.deepcopy(pairs.NOOP_ACTION)
        planner_receipt = None
        live_receipt = gain_safety.apply_safety_disposition(env, current, safety)
        post_disposition_observation = pairs.model_public_observation(env)
        safety_disposition = gain_safety.verify_live_safety_disposition(
            env, current, safety, live_receipt, post_disposition_observation,
        )
        safety_unknown_reason = (
            safety_disposition.get("unknown_reason")
            if safety_disposition.get("resolved") is not True else None
        )
        planner_cpu_seconds = 0.0
        allowed = False
        if due or forced:
            if safety_disposition.get("resolved") is True:
                scope = identify_scope(post_disposition_observation, detection)
                allowed, planner_receipt, planner_cpu_seconds = _online_plan(
                    source, forced, policy_rng,
                    lambda: fixed_replanner.replan(env, affected_scope=scope, max_candidates=max_candidates),
                )
                source_action_counts["allowed" if allowed else "suppressed"] += 1
            action = copy.deepcopy(planner_receipt["selected_action"]) if planner_receipt is not None else copy.deepcopy(pairs.NOOP_ACTION)
            if not allowed:
                action = copy.deepcopy(pairs.NOOP_ACTION)
            state_id = f"{scenario['parent_id']}-{source.lower().replace('-', '_')}-t{int(env.time):03d}-s{ordinary_seen + len(selected_ordinary) + len(selected_forced) + 1:03d}"
            record = {
                "state_id": state_id, "scenario_id": scenario["parent_id"],
                "parent_family": scenario.get("parent_family", scenario["parent_id"]),
                "split": scenario["split"], "event_type": detection.get("event_type") or "none",
                "forced": forced, "detection": copy.deepcopy(detection),
                "trigger_decision": copy.deepcopy(decision), "safety": copy.deepcopy(safety),
                "safety_apply_receipt": copy.deepcopy(live_receipt),
                "safety_disposition": copy.deepcopy(safety_disposition),
                "unknown_reason": safety_unknown_reason,
                "replan_action": None if planner_receipt is None else copy.deepcopy(planner_receipt["selected_action"]),
                "candidate_receipt": None if planner_receipt is None else copy.deepcopy(planner_receipt),
                "planner_cpu_seconds": planner_cpu_seconds,
                "source_action_allowed": allowed if safety_disposition.get("resolved") is True else None,
                "trajectory_source": source, "time": int(env.time),
                "snapshot_json": pairs.capture_snapshot(
                    env, state_id=state_id, scenario_id=str(scenario["parent_id"]),
                    parent_family=str(scenario.get("parent_family", scenario["parent_id"])),
                    split=str(scenario["split"]), event_type=str(detection.get("event_type") or "none"),
                    forced=forced, replanner_version=str(getattr(fixed_replanner, "REPLANNER_VERSION", "Public-C1-Catalog-Replanner-v33")),
                ),
            }
            if forced:
                selected_forced.append(record)
            else:
                ordinary_seen += 1
                limit = int(config["collection"]["max_states_per_trajectory"])
                if len(selected_ordinary) < limit:
                    selected_ordinary.append(record)
                else:
                    replacement = reservoir_rng.randrange(ordinary_seen)
                    if replacement < limit:
                        selected_ordinary[replacement] = record
        if planner_receipt is not None and action != pairs.NOOP_ACTION:
            engine.record_reconstruction(int(env.time), True)
        if safety_disposition.get("resolved") is not True:
            action = copy.deepcopy(pairs.NOOP_ACTION)
        observation, reward, terminated, truncated, step_info = env.step(copy.deepcopy(action))
        elapsed_ticks += 1
        trajectory_trace.append({
            "tick": elapsed_ticks, "time_before": int(current.get("time", 0)),
            "public_before": current, "detection": detection, "trigger_decision": decision,
            "safety_report": safety, "safety_apply_receipt": live_receipt,
            "safety_disposition": safety_disposition, "unknown_reason": safety_unknown_reason,
            "planner_receipt": planner_receipt, "action": action,
            "public_after": pairs.model_public_observation(env),
            "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated),
            "step_info": step_info,
        })
        previous = copy.deepcopy(current)
    scenario_rows: list[dict[str, Any]] = []
    try:
        from .v33_scenarios import sample_v33_future_suffix
    except ImportError:
        from v33_scenarios import sample_v33_future_suffix
    selected = [*selected_ordinary, *selected_forced]
    for record in selected:
        restored = pairs.restore_snapshot(record["snapshot_json"], env_factory=EventReliefEnv)
        offline_planner_cpu_seconds = 0.0
        if record["replan_action"] is None and record["safety_disposition"].get("resolved") is True:
            scope = identify_scope(pairs.model_public_observation(restored), record["detection"])
            offline_started = time.process_time()
            offline_receipt = fixed_replanner.replan(restored, affected_scope=scope, max_candidates=max_candidates)
            offline_planner_cpu_seconds = max(0.0, time.process_time() - offline_started)
            record["replan_action"] = copy.deepcopy(offline_receipt["selected_action"])
            record["candidate_receipt"] = copy.deepcopy(offline_receipt)
        elif record["replan_action"] is None:
            record["replan_action"] = copy.deepcopy(pairs.NOOP_ACTION)
        suffix_rows = []
        if not record["forced"]:
            for suffix_index in range(int(config["labels"]["matched_future_suffixes"])):
                suffix_seed = int.from_bytes(hashlib.sha256(_canonical({
                    "namespace": "v3-counterfactual-suffix-v1", "scenario_id": scenario["parent_id"],
                    "state_id": record["state_id"], "index": suffix_index,
                })).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF
                future_events = sample_v33_future_suffix(
                    scenario, after_time=int(record["time"]),
                    horizon_minutes=int(scenario["base_scenario"]["horizon"]) - int(record["time"]),
                    suffix_seed=suffix_seed,
                )
                suffix_rows.append({"suffix_seed": suffix_seed, "future_events": future_events})
        label_cpu_cost = offline_planner_cpu_seconds or float(record["planner_cpu_seconds"])
        label = pairs.collect_paired_label(
            restored, state_id=record["state_id"], scenario_id=str(scenario["parent_id"]),
            parent_family=str(scenario.get("parent_family", scenario["parent_id"])),
            split=str(scenario["split"]), event_type=str(record["event_type"]),
            forced=bool(record["forced"]), replanner_version=str(
                record["candidate_receipt"].get("replanner_version")
                if isinstance(record["candidate_receipt"], Mapping)
                else fixed_replanner.REPLANNER_VERSION
            ),
            replan_action=record["replan_action"], suffix_rows=suffix_rows,
            env_factory=EventReliefEnv, continuation_policy=continuation,
            replanner_cpu_seconds=label_cpu_cost,
            horizon_minutes=int(config["labels"]["horizon_minutes"]),
        )
        oracle = gain_oracle.validate_gain_label(label)
        scenario_rows.append({
            "schema": "gain-gate-v33-training-row/1",
            **{key: value for key, value in record.items() if key != "snapshot_json"},
            "offline_label_planner_cpu_seconds": offline_planner_cpu_seconds,
            "label": label, "oracle_receipt": oracle,
            "scenario_content_sha256": scenario.get("scenario_content_sha256"),
        })
    final_tasks = trajectory_trace[-1].get("public_after", {}).get("tasks", []) if trajectory_trace else []
    final_status_counts: dict[str, int] = {}
    for task in final_tasks:
        status = str(task.get("status", "unknown"))
        final_status_counts[status] = final_status_counts.get(status, 0) + 1
    usage = _platform_usage(trajectory_trace)
    unresolved_safety = [
        step for step in trajectory_trace
        if isinstance(step.get("safety_disposition"), Mapping)
        and step["safety_disposition"].get("resolved") is False
    ]
    summary = {
        "trajectory_source": source, "elapsed_ticks": elapsed_ticks,
        "task_mode": scenario.get("task_mode"),
        "targets_per_region": scenario.get("targets_per_region"),
        "target_site_count": 9 * int(scenario.get("targets_per_region", 0)),
        "platform_usage": usage,
        "platforms_observed": len(usage),
        "platforms_with_active_task": sum(row["active_task_ticks"] > 0 for row in usage.values()),
        "cross_region_queue_platform_ticks": sum(row["cross_region_queue_ticks"] for row in usage.values()),
        "final_visible_task_status_counts": final_status_counts,
        "live_unresolved_safety_disposition_count": len(unresolved_safety),
        "live_unresolved_safety_unknown_reasons": [str(step.get("unknown_reason")) for step in unresolved_safety],
        "selected_state_count": len(scenario_rows), "ordinary_selected_count": len(selected_ordinary),
        "forced_state_count": len(selected_forced), "ordinary_candidate_count": ordinary_seen,
        **{f"source_{key}_count": value for key, value in source_action_counts.items()},
        "terminated": bool(terminated), "truncated": bool(truncated),
        "trajectory_trace": trajectory_trace,
    }
    return scenario, scenario_rows, summary


def _build_and_commit(spec: Mapping[str, Any], ordinal: int, *, output_dir: Path, run_id: str, binding: Mapping[str, str], stage: str = "collect", scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    source = SOURCE_NAMES[ordinal % len(SOURCE_NAMES)]
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        scenario, rows, summary = _scenario_and_row(spec, source, scenario=scenario)
        shard_resource = shards.resource_receipt(
            stage=stage, started_wall=start_wall,
            started_cpu=start_cpu, current_rss_bytes=_current_rss(),
        )
        shard_resource["exit_state"] = "complete"
    except BaseException as exc:
        failed_resource = shards.resource_receipt(
            stage=stage, started_wall=start_wall,
            started_cpu=start_cpu, current_rss_bytes=_current_rss(),
        )
        failed_resource["exit_state"] = "failed"
        failed_resource["error_type"] = type(exc).__name__
        failed_resource["error"] = str(exc)
        scenario_id = (scenario or {}).get("parent_id", f"registry-{ordinal:03d}")
        failure_path = output_dir / run_id / "failures" / f"{scenario_id}.failure.json"
        _write_json_atomic(failure_path, {
            "schema": "gain-gate-v33-shard-failure/1", "run_id": run_id,
            "scenario_id": scenario_id, "registry_spec": dict(spec),
            "first_error": {"type": type(exc).__name__, "message": str(exc)},
            "resource_receipt": failed_resource, "completion_marker_written": False,
        })
        raise
    rows_with_metadata = [{"schema": "gain-gate-v33-shard-header/1", "scenario": scenario, "trajectory": summary}]
    rows_with_metadata.extend({"schema": "gain-gate-v33-shard-row/1", "training_row": row} for row in rows)
    result = shards.commit_scenario_shard(
        output_dir=output_dir, run_id=run_id, scenario=scenario, rows=rows_with_metadata,
        binding=binding, runtime_receipt=shard_resource,
    )
    result["state_summary"] = shards.summarize_gain_states(rows)
    result["trajectory_summary"] = {key: value for key, value in summary.items() if key != "trajectory_trace"}
    result["task_mode"] = scenario.get("task_mode")
    result["targets_per_region"] = scenario.get("targets_per_region")
    result["seed_variant"] = scenario.get("seed_variant")
    return result


def _validate_scenario_registry(*, task_mode: str, targets_per_region: int) -> dict[str, Any]:
    registry = shards.validate_registry(
        shards.expected_registry(task_mode=task_mode, targets_per_region=targets_per_region),
        task_mode=task_mode, targets_per_region=targets_per_region,
    )
    factory = _scenario_factory()
    counts: dict[str, int] = {}
    sources = [0, 0, 0]
    scenario_ids: set[str] = set()
    physical_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for ordinal, spec in enumerate(registry):
        scenario = factory(
            split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"],
            instance_index=spec["instance_index"], task_mode=task_mode,
            targets_per_region=targets_per_region,
        )
        if scenario.get("split") != spec["split"] or scenario.get("condition") != spec["condition"]:
            raise ValueError("V3_SCENARIO_REGISTRY_IDENTITY_MISMATCH")
        if scenario.get("parent_id") in scenario_ids:
            raise ValueError("V3_DUPLICATE_SCENARIO_ID")
        scenario_ids.add(str(scenario["parent_id"]))
        base = scenario.get("base_scenario", {})
        if base.get("map", {}).get("rows") != 100 or base.get("map", {}).get("cols") != 100:
            raise ValueError("V3_MAP_SIZE_MISMATCH")
        task_multiplier = 1 if task_mode == "search_only" else 2
        if (
            len(base.get("platforms", [])) != 36
            or len(base.get("tasks", [])) != 9 * targets_per_region * task_multiplier
            or int(base.get("horizon", -1)) != 90
        ):
            raise ValueError("V3_SCENARIO_SCALE_MISMATCH")
        if base.get("region_contract") != "relief-regions-3x3-v1":
            raise ValueError("V3_REGION_METADATA_MISSING")
        physical_hash = scenario.get("scenario_content_sha256")
        if not isinstance(physical_hash, str) or len(physical_hash) != 64:
            raise ValueError("V3_SCENARIO_CONTENT_HASH_REQUIRED")
        physical_by_split[spec["split"]].add(physical_hash)
        counts[f"{spec['split']}:{spec['condition']}"] = counts.get(f"{spec['split']}:{spec['condition']}", 0) + 1
        sources[ordinal % 3] += 1
    if physical_by_split["train"] & physical_by_split["validation"]:
        raise ValueError("V3_TRAIN_VALIDATION_PHYSICAL_SCENARIO_LEAKAGE")
    if len(physical_by_split["train"]) != 84 or len(physical_by_split["validation"]) != 84:
        raise ValueError("V3_DUPLICATE_PHYSICAL_SCENARIO_WITHIN_SPLIT")
    return {
        "registry_count": len(registry), "condition_split_counts": counts,
        "source_counts": dict(zip(SOURCE_NAMES, sources)), "sealed_splits": ["test", "ood"],
        "task_mode": task_mode, "targets_per_region": targets_per_region,
    }


def _write_run_receipt(path: Path, *, run_id: str, stage: str, binding: Mapping[str, str], resource: Mapping[str, Any], results: list[Mapping[str, Any]], study_config: Mapping[str, Any]) -> None:
    receipt = {
        "schema": shards.RUN_RECEIPT_SCHEMA, "run_id": run_id, "stage": stage,
        "binding": dict(binding), "study_config": dict(study_config),
        "result_count": len(results), "results": list(results), "resource_receipt": dict(resource),
    }
    _write_json_atomic(path / "RUN_RECEIPT.json", receipt)


def _task_spec(*, split: str, condition: str, instance_index: int, seed_variant: int, task_mode: str, targets_per_region: int) -> dict[str, Any]:
    return {
        "split": split, "condition": condition, "instance_index": instance_index,
        "seed_variant": seed_variant, "task_mode": task_mode,
        "targets_per_region": targets_per_region,
    }


def _smoke_specs(task_mode: str, targets_per_region: int) -> list[dict[str, Any]]:
    conditions = shards.CONDITIONS[1:]
    return [
        _task_spec(
            split="pilot", condition=condition, instance_index=instance_index,
            seed_variant=900_001 + condition_index * 100 + instance_index,
            task_mode=task_mode, targets_per_region=targets_per_region,
        )
        for condition_index, condition in enumerate(conditions)
        for instance_index in range(3)
    ]


def _pilot_specs() -> list[dict[str, Any]]:
    seeds = (930_101, 930_102, 930_103)
    return [
        _task_spec(
            split="pilot", condition="task_update_or_arrival", instance_index=seed_index,
            seed_variant=seed, task_mode=task_mode, targets_per_region=targets_per_region,
        )
        for task_mode in TASK_MODES
        for targets_per_region in TARGET_DENSITIES
        for seed_index, seed in enumerate(seeds)
    ]


def _validate_smoke(results: list[Mapping[str, Any]], *, binding: Mapping[str, str], task_mode: str, targets_per_region: int) -> dict[str, Any]:
    checks = {
        "instance_count": len(results) == 18,
        "scenario_scale_and_regions": True,
        "complete_public_trajectory": True,
        "platform_usage_recorded": True,
        "oracle_structure_recorded": True,
        "oracle_label_mask_recorded": True,
        "valid_label_mask_matches_oracle": True,
        "committed_shard_integrity": True,
    }
    label_counts = {
        "rows": 0, "valid_mask": 0, "masked": 0,
        "oracle_valid": 0, "oracle_invalid": 0,
        "oracle_label_valid": 0, "oracle_label_masked": 0,
    }
    for instance in results:
        marker = shards.verify_scenario_shard(instance["marker_path"], expected_binding=binding)
        if marker.get("marker_sha256") != instance.get("marker_sha256"):
            checks["committed_shard_integrity"] = False
        shard_path = Path(instance["marker_path"]).with_name(marker["shard_file"])
        with gzip.open(shard_path, "rt", encoding="ascii") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        headers = [row for row in rows if row.get("schema") == "gain-gate-v33-shard-header/1"]
        if len(headers) != 1:
            checks["scenario_scale_and_regions"] = False
            continue
        header = headers[0]
        scenario = header["scenario"]
        trajectory = header["trajectory"]
        base = scenario.get("base_scenario", {})
        expected_tasks = 9 * targets_per_region * (1 if task_mode == "search_only" else 2)
        if (
            scenario.get("task_mode") != task_mode
            or scenario.get("targets_per_region") != targets_per_region
            or base.get("map", {}).get("rows") != 100
            or base.get("map", {}).get("cols") != 100
            or len(base.get("platforms", [])) != 36
            or len(base.get("tasks", [])) != expected_tasks
            or base.get("region_contract") != "relief-regions-3x3-v1"
        ):
            checks["scenario_scale_and_regions"] = False
        trace = trajectory.get("trajectory_trace", [])
        if len(trace) != 90 or not trace[-1].get("terminated") and not trace[-1].get("truncated"):
            checks["complete_public_trajectory"] = False
        usage = trajectory.get("platform_usage", {})
        if len(usage) != 36:
            checks["platform_usage_recorded"] = False
        for row in rows:
            if row.get("schema") != "gain-gate-v33-shard-row/1":
                continue
            training_row = row.get("training_row", {})
            label = training_row.get("label", {})
            oracle = training_row.get("oracle_receipt")
            label_counts["rows"] += 1
            if not isinstance(label.get("valid_mask"), bool):
                checks["valid_label_mask_matches_oracle"] = False
            elif label["valid_mask"]:
                label_counts["valid_mask"] += 1
            else:
                label_counts["masked"] += 1
            if not isinstance(oracle, Mapping) or not isinstance(oracle.get("valid"), bool):
                checks["oracle_structure_recorded"] = False
                continue
            if not isinstance(oracle.get("label_valid_mask"), bool):
                checks["oracle_label_mask_recorded"] = False
            elif oracle["label_valid_mask"]:
                label_counts["oracle_label_valid"] += 1
            else:
                label_counts["oracle_label_masked"] += 1
            if oracle["valid"]:
                label_counts["oracle_valid"] += 1
            else:
                label_counts["oracle_invalid"] += 1
            if label.get("valid_mask") != oracle.get("label_valid_mask"):
                checks["valid_label_mask_matches_oracle"] = False
    return {"checks": checks, "label_counts": label_counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "validate", "smoke", "pilot", "collect", "verify"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs-v33")
    parser.add_argument("--run-id")
    parser.add_argument("--task-mode", choices=TASK_MODES, default="search_rescue")
    parser.add_argument("--targets-per-region", type=int, choices=TARGET_DENSITIES, default=3)
    args = parser.parse_args(argv)
    if args.stage == "verify":
        if not args.run_id:
            parser.error("verify requires --run-id")
        result = shards.verify_run(args.output_dir, args.run_id, expected_binding=shards.current_binding())
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    binding = shards.current_binding()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    study_config = {
        "task_mode": args.task_mode,
        "targets_per_region": args.targets_per_region,
        "map": [100, 100], "regions": [3, 3], "platforms": {"uav": 18, "helicopter": 18},
        "horizon_minutes": 90, "label_horizon_minutes": 30, "matched_suffixes": 2,
    }
    try:
        if args.stage == "audit":
            results = [_runtime_binding_report()]
        elif args.stage == "validate":
            results = [
                _runtime_binding_report(),
                _validate_scenario_registry(task_mode=args.task_mode, targets_per_region=args.targets_per_region),
            ]
        elif args.stage == "smoke":
            specs = _smoke_specs(args.task_mode, args.targets_per_region)
            conditions = shards.CONDITIONS[1:]
            study_config["conditions"] = list(conditions)
            results = []
            for ordinal, spec in enumerate(specs):
                result = _build_and_commit(
                    spec, ordinal, output_dir=args.output_dir, run_id=run_id,
                    binding=binding, stage="smoke",
                )
                results.append(result)
                print(json.dumps({"completed": len(results), "total": 18, "scenario_id": result["scenario_id"]}), flush=True)
            validation = _validate_smoke(
                results, binding=binding, task_mode=args.task_mode,
                targets_per_region=args.targets_per_region,
            )
            if not all(validation["checks"].values()):
                raise ValueError(f"V33_SMOKE_CHECKS_FAILED:{validation['checks']}")
            manifest_path = run_root / "SMOKE_MANIFEST.json"
            manifest = {
                "schema": "gain-gate-v33-smoke-manifest/1", "status": "complete",
                "completed_instances": 18, "binding": dict(binding), "study_config": study_config,
                "families": list(conditions), "instances": [{
                    "scenario_id": row["scenario_id"], "split": row["split"],
                    "condition": row["condition"], "marker_path": row["marker_path"],
                    "marker_sha256": row["marker_sha256"], "compressed_sha256": row["compressed_sha256"],
                    "row_count": row["row_count"],
                } for row in results],
                "checks": validation["checks"], "label_counts": validation["label_counts"],
            }
            _write_json_atomic(manifest_path, manifest)
            summary = {
                "schema": "gain-gate-v33-smoke-summary/1", "status": "complete",
                "completed_instances": 18, "binding": dict(binding), "study_config": study_config,
                "checks": validation["checks"], "label_counts": validation["label_counts"],
                "manifest_sha256": shards.sha256_file(manifest_path),
            }
            _write_json_atomic(run_root / "SMOKE_SUMMARY.json", summary)
        elif args.stage == "pilot":
            specs = _pilot_specs()
            study_config.update({
                "pilot_condition": "task_update_or_arrival",
                "task_modes": list(TASK_MODES), "target_densities": list(TARGET_DENSITIES),
                "shared_seed_variants": [930_101, 930_102, 930_103],
            })
            results = []
            for ordinal, spec in enumerate(specs):
                result = _build_and_commit(
                    spec, ordinal, output_dir=args.output_dir, run_id=run_id,
                    binding=binding, stage="pilot",
                )
                results.append(result)
                print(json.dumps({
                    "completed": len(results), "total": 18, "scenario_id": result["scenario_id"],
                    "task_mode": spec["task_mode"], "targets_per_region": spec["targets_per_region"],
                    "seed_variant": spec["seed_variant"],
                }), flush=True)
            _write_json_atomic(run_root / "PILOT_SUMMARY.json", {
                "schema": "gain-gate-v33-pilot-summary/1", "status": "complete",
                "completed_instances": 18, "study_config": study_config,
                "instances": [{"scenario_id": row["scenario_id"], "task_mode": row["task_mode"],
                               "targets_per_region": row["targets_per_region"],
                               "seed_variant": row["seed_variant"], "marker_sha256": row["marker_sha256"],
                               "trajectory_summary": row["trajectory_summary"]}
                              for row in results],
            })
        else:
            registry = shards.expected_registry(task_mode=args.task_mode, targets_per_region=args.targets_per_region)
            shards.validate_registry(registry, task_mode=args.task_mode, targets_per_region=args.targets_per_region)
            study_config["formal_instances"] = len(registry)
            results = []
            for ordinal, spec in enumerate(registry):
                result = _build_and_commit(
                    spec, ordinal, output_dir=args.output_dir, run_id=run_id,
                    binding=binding, stage="collect",
                )
                results.append(result)
                print(json.dumps({"completed": len(results), "total": len(registry), "scenario_id": result["scenario_id"]}), flush=True)
            totals = {"positive": 0, "non_positive": 0, "forced_excluded": 0, "empty_excluded": 0, "unknown": 0}
            by_split: dict[str, dict[str, int]] = {}
            scenario_groups: dict[str, dict[str, list[str]]] = {}
            for result in results:
                for split, summary in result["state_summary"]["splits"].items():
                    split_totals = by_split.setdefault(split, {key: 0 for key in totals})
                    for key in totals:
                        split_totals[key] += int(summary[key])
                    groups = scenario_groups.setdefault(split, {"positive": [], "non_positive": []})
                    groups["positive"].extend(summary["positive_scenario_groups"])
                    groups["non_positive"].extend(summary["non_positive_scenario_groups"])
            for split_summary in by_split.values():
                for key in totals:
                    totals[key] += split_summary[key]
            gate = totals["positive"] >= 30 and totals["non_positive"] >= 30
            coverage = all(
                by_split.get(split, {}).get("positive", 0) > 0
                and by_split.get(split, {}).get("non_positive", 0) > 0
                and len(set(scenario_groups.get(split, {}).get("positive", []))) > 1
                and len(set(scenario_groups.get(split, {}).get("non_positive", []))) > 1
                for split in ("train", "validation")
            )
            _write_json_atomic(run_root / "GAIN_DATA_SUMMARY.json", {
                "schema": "gain-gate-v33-gain-data-summary/1",
                "unique_ordinary_states": totals,
                "by_split": by_split,
                "scenario_group_coverage": {
                    split: {key: sorted(set(values)) for key, values in groups.items()}
                    for split, groups in scenario_groups.items()
                },
                "thirty_thirty_met": gate and coverage,
                "training_gate": "eligible_for_lightweight_models" if gate and coverage else "stop_and_report_diagnostics",
            })

        if shards.current_binding() != binding:
            raise ValueError("V33_BINDING_CHANGED_DURING_RUN")
        resource = shards.resource_receipt(
            stage=args.stage, started_wall=start_wall, started_cpu=start_cpu,
            current_rss_bytes=_current_rss(),
        )
        _write_run_receipt(
            run_root, run_id=run_id, stage=args.stage, binding=binding,
            resource=resource, results=results, study_config=study_config,
        )
        print(json.dumps({
            "run_id": run_id, "stage": args.stage, "result_count": len(results),
            "resource_receipt": resource,
        }, sort_keys=True, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failed_resource = shards.resource_receipt(
            stage=args.stage, started_wall=start_wall, started_cpu=start_cpu,
            current_rss_bytes=_current_rss(),
        )
        failure = {
            "schema": "gain-gate-v33-run-failure/1", "run_id": run_id, "stage": args.stage,
            "binding": binding, "error_type": type(exc).__name__, "error": str(exc),
            "resource_receipt": failed_resource,
        }
        _write_json_atomic(run_root / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
