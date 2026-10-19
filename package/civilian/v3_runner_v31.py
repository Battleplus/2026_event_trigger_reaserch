"""CLI entry point for V3 source-bound validation, smoke, and shard collection."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
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
REVIEW_SCHEMA = "gain-gate-v3-readonly-review/1"
_PEAK_RSS_OBSERVED = 0


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
    if not isinstance(value, dict) or value.get("version") != "gain-gate-v3-9region-30platform":
        raise ValueError("V3_CONFIG_IDENTITY_INVALID")
    return value


def _enforce_runtime() -> list[int] | None:
    try:
        import psutil
        process = psutil.Process()
        allowed = sorted(process.cpu_affinity())
        if not allowed:
            raise RuntimeError("V3_CPU_AFFINITY_EMPTY")
        process.cpu_affinity([allowed[0]])
        actual = sorted(process.cpu_affinity())
        if len(actual) != 1:
            raise RuntimeError("V3_CPU_AFFINITY_NOT_SINGLE")
    except ImportError:
        actual = None
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError as exc:
        raise RuntimeError("V3_THREADPOOLCTL_REQUIRED") from exc
    try:
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if torch.cuda.is_available() or torch.cuda.is_initialized():
            raise RuntimeError("V3_CUDA_MUST_BE_DISABLED")
    except ImportError:
        pass
    return actual


def _peak_rss() -> int:
    global _PEAK_RSS_OBSERVED
    try:
        import resource_supervisor
        current, high_water = resource_supervisor._read_rss_metrics()
        del current
        _PEAK_RSS_OBSERVED = max(_PEAK_RSS_OBSERVED, int(high_water))
        return _PEAK_RSS_OBSERVED
    except (ImportError, AttributeError):
        try:
            import psutil
            current = int(psutil.Process().memory_info().rss)
        except ImportError as exc:
            raise RuntimeError("V3_RSS_MEASUREMENT_UNAVAILABLE") from exc
    _PEAK_RSS_OBSERVED = max(_PEAK_RSS_OBSERVED, current)
    return _PEAK_RSS_OBSERVED


def _check_limits(start_wall: float, start_cpu: float | None = None, *, max_wall: int = shards.MAX_WALL_SECONDS, max_cpu: int = shards.MAX_CPU_SECONDS) -> int:
    wall = time.perf_counter() - start_wall
    rss = _peak_rss()
    if wall > max_wall:
        raise TimeoutError("V3_JOB_WALL_LIMIT_EXCEEDED")
    if start_cpu is not None and time.process_time() - start_cpu > max_cpu:
        raise TimeoutError("V3_JOB_CPU_LIMIT_EXCEEDED")
    if rss > shards.MAX_RSS_BYTES:
        raise MemoryError("V3_JOB_RSS_LIMIT_EXCEEDED")
    return rss


def _runtime_binding_report() -> dict[str, Any]:
    binding = shards.current_binding()
    return {
        "schema": "gain-gate-v3-binding-report/1",
        **binding,
        "source_manifest": shards.source_manifest(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sealed_splits": ["test", "ood"],
        "registry_count": len(shards.expected_registry()),
    }


def _load_review(path: str | Path, binding: Mapping[str, str]) -> dict[str, Any]:
    review_path = Path(path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("V3_READONLY_REVIEW_SCHEMA_REQUIRED")
    if review.get("binding") != dict(binding):
        raise ValueError("V3_READONLY_REVIEW_BINDING_MISMATCH")
    if review.get("decision") not in {"approved", "pass"}:
        raise ValueError("V3_READONLY_REVIEW_NOT_APPROVED")
    smoke = review.get("smoke_evidence")
    if not isinstance(smoke, Mapping):
        raise ValueError("V3_READONLY_REVIEW_SMOKE_EVIDENCE_REQUIRED")
    evidence_paths = {}
    evidence_values = {}
    for name in ("manifest", "summary"):
        artifact_path = Path(str(smoke.get(f"{name}_path", "")))
        if not artifact_path.is_absolute():
            artifact_path = review_path.parent / artifact_path
        if not artifact_path.is_file() or shards.sha256_file(artifact_path) != smoke.get(f"{name}_sha256"):
            raise ValueError(f"V3_READONLY_REVIEW_SMOKE_{name.upper()}_HASH_INVALID")
        evidence_paths[name] = str(artifact_path.resolve())
        evidence_values[name] = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest, summary = evidence_values["manifest"], evidence_values["summary"]
    if (
        manifest.get("schema") != "gain-gate-v3-smoke-manifest/1"
        or manifest.get("status") != "complete"
        or manifest.get("completed_instances") != 18
        or manifest.get("binding") != dict(binding)
        or len(manifest.get("instances", [])) != 18
        or len({row.get("scenario_id") for row in manifest["instances"]}) != 18
        or (manifest.get("families") is not None and manifest.get("families") != list(shards.CONDITIONS[1:]))
        or (manifest.get("families") is not None and {row.get("condition") for row in manifest["instances"]} != set(shards.CONDITIONS[1:]))
        or manifest.get("checks") != {
            "scenario_scale_and_regions": True,
            "complete_public_trajectory": True,
            "paired_label_independent_recomputation": True,
            "committed_shard_integrity": True,
        }
    ):
        raise ValueError("V3_READONLY_REVIEW_SMOKE_MANIFEST_INVALID")
    if (
        summary.get("schema") != "gain-gate-v3-smoke-summary/1"
        or summary.get("status") != "complete"
        or summary.get("completed_instances") != 18
        or summary.get("binding") != dict(binding)
        or summary.get("resource_receipt", {}).get("within_limits") is not True
        or summary.get("manifest_sha256") != shards.sha256_file(Path(evidence_paths["manifest"]))
    ):
        raise ValueError("V3_READONLY_REVIEW_SMOKE_SUMMARY_INVALID")
    for row in manifest["instances"]:
        marker_path = Path(str(row.get("marker_path", "")))
        if not marker_path.is_absolute():
            marker_path = Path(evidence_paths["manifest"]).parent / marker_path
        marker = shards.verify_scenario_shard(marker_path, expected_binding=binding)
        if marker.get("marker_sha256") != row.get("marker_sha256") or marker.get("scenario_id") != row.get("scenario_id"):
            raise ValueError("V3_READONLY_REVIEW_SMOKE_INSTANCE_MARKER_MISMATCH")
    return {
        "path": str(review_path.resolve()), "sha256": shards.sha256_file(review_path),
        "decision": review["decision"], "smoke_evidence": evidence_paths,
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
        from .v3_scenarios import build_v3_gain_scenario
    except ImportError:
        from v3_scenarios import build_v3_gain_scenario
    return build_v3_gain_scenario


def _scenario_and_row(spec: Mapping[str, Any], source: str, *, max_minutes: int | None = None, max_cpu_seconds: int = shards.MAX_CPU_SECONDS, scenario: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
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
    if scenario is None:
        scenario = _scenario_factory()(
            split=str(spec["split"]), condition=str(spec["condition"]),
            seed_variant=int(spec["seed_variant"]), instance_index=int(spec["instance_index"]),
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
    start_wall, start_cpu = time.perf_counter(), time.process_time()
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
        decision = branch_engine.decide(branch_detection, int(branch_env.time), mode="event")
        if not forced:
            return {
                "action": copy.deepcopy(pairs.NOOP_ACTION), "planner_invoked": False,
                "planner_cpu_seconds": 0.0, "policy_version": "shared-safety-trigger-v1",
                "safety_report": safety, "safety_apply_receipt": live_receipt,
                "detection": branch_detection, "trigger_decision": decision,
            }
        refreshed = pairs.model_public_observation(branch_env)
        scope = identify_scope(refreshed, branch_detection)
        started_cpu = time.process_time()
        receipt = fixed_replanner.replan(branch_env, affected_scope=scope, max_candidates=int(config["replanner"]["max_candidates"]))
        planner_cpu = max(0.0, time.process_time() - started_cpu)
        return {
            "action": receipt["selected_action"], "planner_invoked": True,
            "planner_cpu_seconds": planner_cpu, "policy_version": receipt.get("replanner_version"),
            "planner_receipt": receipt, "safety_report": safety,
            "safety_apply_receipt": live_receipt, "detection": branch_detection,
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
        if due or forced:
            scope = identify_scope(pairs.model_public_observation(env), detection)
            allowed, planner_receipt, planner_cpu_seconds = _online_plan(
                source, forced, policy_rng,
                lambda: fixed_replanner.replan(env, affected_scope=scope, max_candidates=max_candidates),
            )
            action = copy.deepcopy(planner_receipt["selected_action"]) if planner_receipt is not None else copy.deepcopy(pairs.NOOP_ACTION)
            source_action_counts["allowed" if allowed else "suppressed"] += 1
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
                "replan_action": None if planner_receipt is None else copy.deepcopy(planner_receipt["selected_action"]),
                "candidate_receipt": None if planner_receipt is None else copy.deepcopy(planner_receipt),
                "planner_cpu_seconds": planner_cpu_seconds, "source_action_allowed": allowed,
                "trajectory_source": source, "time": int(env.time),
                "snapshot_json": pairs.capture_snapshot(
                    env, state_id=state_id, scenario_id=str(scenario["parent_id"]),
                    parent_family=str(scenario.get("parent_family", scenario["parent_id"])),
                    split=str(scenario["split"]), event_type=str(detection.get("event_type") or "none"),
                    forced=forced, replanner_version=str(getattr(fixed_replanner, "REPLANNER_VERSION", "Public-C1-Catalog-Replanner-v3")),
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
        if forced and live_receipt.get("resolved") is False:
            action = copy.deepcopy(pairs.NOOP_ACTION)
        observation, reward, terminated, truncated, step_info = env.step(copy.deepcopy(action))
        elapsed_ticks += 1
        trajectory_trace.append({
            "tick": elapsed_ticks, "time_before": int(current.get("time", 0)),
            "public_before": current, "detection": detection, "trigger_decision": decision,
            "safety_report": safety, "safety_apply_receipt": live_receipt,
            "planner_receipt": planner_receipt, "action": action,
            "public_after": pairs.model_public_observation(env),
            "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated),
            "step_info": step_info,
        })
        if elapsed_ticks % 5 == 0:
            _check_limits(start_wall, start_cpu, max_cpu=max_cpu_seconds)
        if max_minutes is not None and elapsed_ticks >= max_minutes:
            break
        previous = copy.deepcopy(current)
    scenario_rows: list[dict[str, Any]] = []
    try:
        from .v3_scenarios import sample_v3_future_suffix
    except ImportError:
        from v3_scenarios import sample_v3_future_suffix
    selected = [*selected_ordinary, *selected_forced]
    for record in selected:
        _check_limits(start_wall, start_cpu, max_cpu=max_cpu_seconds)
        restored = pairs.restore_snapshot(record["snapshot_json"], env_factory=EventReliefEnv)
        offline_planner_cpu_seconds = 0.0
        if record["replan_action"] is None:
            scope = identify_scope(pairs.model_public_observation(restored), record["detection"])
            offline_started = time.process_time()
            offline_receipt = fixed_replanner.replan(restored, affected_scope=scope, max_candidates=max_candidates)
            offline_planner_cpu_seconds = max(0.0, time.process_time() - offline_started)
            record["replan_action"] = copy.deepcopy(offline_receipt["selected_action"])
            record["candidate_receipt"] = copy.deepcopy(offline_receipt)
        suffix_rows = []
        if not record["forced"]:
            for suffix_index in range(int(config["labels"]["matched_future_suffixes"])):
                suffix_seed = int.from_bytes(hashlib.sha256(_canonical({
                    "namespace": "v3-counterfactual-suffix-v1", "scenario_id": scenario["parent_id"],
                    "state_id": record["state_id"], "index": suffix_index,
                })).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF
                future_events = sample_v3_future_suffix(
                    scenario, after_time=int(record["time"]),
                    horizon_minutes=int(config["labels"]["horizon_minutes"]), suffix_seed=suffix_seed,
                )
                suffix_rows.append({"suffix_seed": suffix_seed, "future_events": future_events})
        label_cpu_cost = offline_planner_cpu_seconds or float(record["planner_cpu_seconds"])
        label = pairs.collect_paired_label(
            restored, state_id=record["state_id"], scenario_id=str(scenario["parent_id"]),
            parent_family=str(scenario.get("parent_family", scenario["parent_id"])),
            split=str(scenario["split"]), event_type=str(record["event_type"]),
            forced=bool(record["forced"]), replanner_version=str(record["candidate_receipt"].get("replanner_version")),
            replan_action=record["replan_action"], suffix_rows=suffix_rows,
            env_factory=EventReliefEnv, continuation_policy=continuation,
            replanner_cpu_seconds=label_cpu_cost,
            horizon_minutes=int(config["labels"]["horizon_minutes"]),
        )
        oracle = gain_oracle.validate_gain_label(label)
        scenario_rows.append({
            "schema": "gain-gate-v3-training-row/1",
            **{key: value for key, value in record.items() if key != "snapshot_json"},
            "offline_label_planner_cpu_seconds": offline_planner_cpu_seconds,
            "label": label, "oracle_receipt": oracle,
            "scenario_content_sha256": scenario.get("scenario_content_sha256"),
        })
    summary = {
        "trajectory_source": source, "elapsed_ticks": elapsed_ticks,
        "selected_state_count": len(scenario_rows), "ordinary_selected_count": len(selected_ordinary),
        "forced_state_count": len(selected_forced), "ordinary_candidate_count": ordinary_seen,
        **{f"source_{key}_count": value for key, value in source_action_counts.items()},
        "terminated": bool(terminated), "truncated": bool(truncated),
        "trajectory_trace": trajectory_trace,
    }
    return scenario, scenario_rows, summary


def _build_and_commit(spec: Mapping[str, Any], ordinal: int, *, output_dir: Path, run_id: str, binding: Mapping[str, str], smoke: bool = False, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    source = SOURCE_NAMES[ordinal % len(SOURCE_NAMES)]
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        smoke_cpu_cap = int(_load_config()["collection"]["v3_smoke_cpu_budget_seconds"]) if smoke else shards.MAX_CPU_SECONDS
        scenario, rows, summary = _scenario_and_row(spec, source, max_minutes=None, max_cpu_seconds=smoke_cpu_cap, scenario=scenario)
        _check_limits(start_wall, start_cpu, max_cpu=smoke_cpu_cap)
        shard_resource = shards.resource_receipt(
            stage="smoke" if smoke else "collect", started_wall=start_wall,
            started_cpu=start_cpu, peak_rss_bytes=_peak_rss(),
        )
        shard_resource["exit_state"] = "complete"
        try:
            import psutil
            shard_resource["cpu_affinity"] = sorted(psutil.Process().cpu_affinity())
        except ImportError:
            shard_resource["cpu_affinity"] = None
        if not shard_resource["within_limits"]:
            raise RuntimeError("V3_SHARD_RESOURCE_LIMIT_RECEIPT_FAILED")
    except BaseException as exc:
        failed_resource = shards.resource_receipt(
            stage="smoke" if smoke else "collect", started_wall=start_wall,
            started_cpu=start_cpu, peak_rss_bytes=_peak_rss(),
        )
        failed_resource["exit_state"] = "failed"
        failed_resource["error_type"] = type(exc).__name__
        failed_resource["error"] = str(exc)
        try:
            scenario_id = (scenario or {}).get("parent_id", f"registry-{ordinal:03d}")
            failure_path = output_dir / run_id / "failures" / f"{scenario_id}.failure.json"
            _write_json_atomic(failure_path, {
                "schema": "gain-gate-v3-shard-failure/1", "run_id": run_id,
                "scenario_id": scenario_id, "registry_spec": dict(spec),
                "first_error": {"type": type(exc).__name__, "message": str(exc)},
                "resource_receipt": failed_resource, "completion_marker_written": False,
            })
        except Exception:
            pass
        raise
    rows_with_metadata = [{"schema": "gain-gate-v3-shard-header/1", "scenario": scenario, "trajectory": summary}]
    rows_with_metadata.extend({"schema": "gain-gate-v3-shard-row/1", "training_row": row} for row in rows)
    return shards.commit_scenario_shard(
        output_dir=output_dir, run_id=run_id, scenario=scenario, rows=rows_with_metadata,
        binding=binding, runtime_receipt=shard_resource,
    )


def _validate_scenario_registry() -> dict[str, Any]:
    registry = shards.validate_registry(shards.expected_registry())
    factory = _scenario_factory()
    counts: dict[str, int] = {}
    sources = [0, 0, 0]
    scenario_ids: set[str] = set()
    physical_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for ordinal, spec in enumerate(registry):
        scenario = factory(split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"], instance_index=spec["instance_index"])
        if scenario.get("split") != spec["split"] or scenario.get("condition") != spec["condition"]:
            raise ValueError("V3_SCENARIO_REGISTRY_IDENTITY_MISMATCH")
        if scenario.get("parent_id") in scenario_ids:
            raise ValueError("V3_DUPLICATE_SCENARIO_ID")
        scenario_ids.add(str(scenario["parent_id"]))
        base = scenario.get("base_scenario", {})
        if base.get("map", {}).get("rows") != 100 or base.get("map", {}).get("cols") != 100:
            raise ValueError("V3_MAP_SIZE_MISMATCH")
        if len(base.get("platforms", [])) != 30 or len(base.get("tasks", [])) != 54 or int(base.get("horizon", -1)) != 90:
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
    return {"registry_count": len(registry), "condition_split_counts": counts, "source_counts": dict(zip(SOURCE_NAMES, sources)), "sealed_splits": ["test", "ood"]}


def _write_run_receipt(path: Path, *, run_id: str, stage: str, binding: Mapping[str, str], resource: Mapping[str, Any], results: list[Mapping[str, Any]], review: Mapping[str, Any] | None = None, resume_reference: Mapping[str, Any] | None = None) -> None:
    receipt = {
        "schema": shards.RUN_RECEIPT_SCHEMA, "run_id": run_id, "stage": stage,
        "binding": dict(binding), "review": dict(review) if review else None,
        "resume_reference": dict(resume_reference) if resume_reference else None,
        "result_count": len(results), "results": list(results), "resource_receipt": dict(resource),
    }
    _write_json_atomic(path / "RUN_RECEIPT.json", receipt)


def _child_peak_rss(process: Any, pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    info = process.memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _process_tree_cpu_seconds(root: Any) -> float:
    """Return CPU time for the guarded worker and any child processes."""
    import psutil

    total = 0.0
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    for process in processes:
        try:
            times = process.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
            continue
        total += float(times.user + times.system)
    return total


def _terminate_process_tree(root: Any) -> None:
    """Terminate the guarded worker and descendants, then reap them."""
    import psutil

    try:
        processes = root.children(recursive=True)
    except psutil.Error:
        processes = []
    for process in reversed(processes):
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        root.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    if processes:
        psutil.wait_procs(processes, timeout=5)


def _monitor_guarded_process(
    process_handle: Any,
    *,
    cpu_limit: float,
    wall_limit: float = shards.MAX_WALL_SECONDS,
    rss_limit: int = shards.MAX_RSS_BYTES,
    poll_interval: float = 0.05,
) -> tuple[str | None, float, int, float]:
    """Poll one worker tree and terminate it when a configured limit is exceeded."""
    import psutil

    root = psutil.Process(process_handle.pid)
    started = time.perf_counter()
    child_cpu = 0.0
    max_rss = 0
    reason = None
    while process_handle.poll() is None:
        wall = time.perf_counter() - started
        child_cpu = max(child_cpu, _process_tree_cpu_seconds(root))
        try:
            max_rss = max(max_rss, _child_peak_rss(root, process_handle.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
            break
        if wall > wall_limit:
            reason = "wall_limit_exceeded"
        elif child_cpu > cpu_limit:
            reason = "cpu_limit_exceeded"
        elif max_rss > rss_limit:
            reason = "rss_limit_exceeded"
        if reason is not None:
            _terminate_process_tree(root)
            try:
                process_handle.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process_handle.kill()
                process_handle.wait()
            break
        time.sleep(poll_interval)
    return reason, child_cpu, max_rss, max(0.0, time.perf_counter() - started)


def _run_guarded(args: argparse.Namespace) -> int:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("V3_PROCESS_GUARD_REQUIRES_PSUTIL") from exc
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), args.stage,
        "--output-dir", str(args.output_dir.resolve()), "--run-id", run_id, "--_worker",
    ]
    if args.review is not None:
        command.extend(["--review", str(args.review.resolve())])
    if args.resume_from:
        command.extend(["--resume-from", args.resume_from])
    if args.scenario_id:
        command.extend(["--scenario-id", args.scenario_id])
    started_wall = time.perf_counter()
    process_handle = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu_limit = int(_load_config()["collection"]["v3_smoke_cpu_budget_seconds"]) if args.stage == "smoke" else shards.MAX_CPU_SECONDS
    reason, child_cpu, max_rss, elapsed_wall = _monitor_guarded_process(
        process_handle, cpu_limit=cpu_limit,
    )
    stdout, stderr = process_handle.communicate()
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    if reason is not None:
        run_root = args.output_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_root / "FAILURE.json", {
            "schema": "gain-gate-v3-run-failure/1", "run_id": run_id,
            "stage": args.stage, "error_type": "OuterResourceGuardian",
            "error": reason, "completion_receipt_written": False,
            "resource_receipt": {
                "wall_seconds": elapsed_wall,
                "child_process_cpu_seconds": child_cpu, "peak_rss_bytes": max_rss,
                "limits": {"wall_seconds": shards.MAX_WALL_SECONDS, "cpu_seconds": cpu_limit, "rss_bytes": shards.MAX_RSS_BYTES},
                "exit_state": "killed",
                "cpu_measurement": "guarded_process_tree_user_plus_system",
            },
        })
        return 124
    return int(process_handle.returncode or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "validate", "smoke", "collect", "verify"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-id")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--resume-from")
    parser.add_argument("--scenario-id")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stage == "verify":
        if not args.run_id:
            parser.error("verify requires --run-id")
        result = shards.verify_run(args.output_dir, args.run_id, expected_binding=shards.current_binding())
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if not args._worker:
        return _run_guarded(args)
    _enforce_runtime()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    binding = shards.current_binding()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("invalid run id")
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    review = None
    resume_reference = None
    try:
        if args.stage == "audit":
            results = [_runtime_binding_report()]
        elif args.stage == "validate":
            results = [_runtime_binding_report(), _validate_scenario_registry()]
        elif args.stage == "smoke":
            if args.scenario_id:
                raise ValueError("V3_SMOKE_IS_FROZEN_18_INSTANCE_BATCH")
            smoke_conditions = shards.CONDITIONS[1:]
            smoke_specs = [
                {"split": "train", "condition": condition, "instance_index": instance_index,
                 "seed_variant": 900_001 + condition_index * 100 + instance_index}
                for condition_index, condition in enumerate(smoke_conditions)
                for instance_index in range(3)
            ]
            results = [
                _build_and_commit(spec, ordinal, output_dir=args.output_dir, run_id=run_id, binding=binding, smoke=True)
                for ordinal, spec in enumerate(smoke_specs)
            ]
            instances = [{
                "scenario_id": row["scenario_id"], "split": row["split"], "condition": row["condition"],
                "marker_path": row["marker_path"], "marker_sha256": row["marker_sha256"],
                "compressed_sha256": row["compressed_sha256"], "row_count": row["row_count"],
            } for row in results]
            manifest = {
                "schema": "gain-gate-v3-smoke-manifest/1", "status": "pending",
                "completed_instances": 0, "binding": dict(binding),
                "families": list(smoke_conditions), "instances": instances,
            }
            summary = {
                "schema": "gain-gate-v3-smoke-summary/1", "status": "pending",
                "completed_instances": 0, "binding": dict(binding),
            "resource_receipt": None,
            }
            _write_json_atomic(run_root / "SMOKE_MANIFEST.json", manifest)
            _write_json_atomic(run_root / "SMOKE_SUMMARY.json", summary)
            results.extend([manifest, summary])
        else:
            if args.review is None:
                raise ValueError("V3_COLLECT_REQUIRES_READONLY_REVIEW_PATH")
            review = _load_review(args.review, binding)
            prior_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
            if args.resume_from:
                old = shards.verify_run(args.output_dir, args.resume_from, expected_binding=binding)
                previous_rows = list(old["verified"])
                previous_rows.extend(old.get("referenced", []))
                prior_by_identity = {
                    (row["split"], row["condition"], row["scenario_id"]): row
                    for row in previous_rows
                }
            registry = shards.expected_registry()
            results = []
            referenced_shards = []
            factory = _scenario_factory()
            for ordinal, spec in enumerate(registry):
                _check_limits(start_wall, start_cpu, max_cpu=int(_load_config()["collection"]["v3_smoke_cpu_budget_seconds"]) if args.stage == "smoke" else shards.MAX_CPU_SECONDS)
                scenario = factory(split=spec["split"], condition=spec["condition"], seed_variant=spec["seed_variant"], instance_index=spec["instance_index"])
                identity = (str(scenario["split"]), str(scenario["condition"]), str(scenario["parent_id"]))
                prior = prior_by_identity.get(identity)
                if prior is not None:
                    if prior.get("scenario_content_sha256") != scenario.get("scenario_content_sha256"):
                        raise ValueError("V3_RESUME_SCENARIO_CONTENT_MISMATCH")
                    referenced_shards.append({"marker_path": prior["marker_path"], "marker_sha256": prior["marker_sha256"]})
                    continue
                results.append(_build_and_commit(spec, ordinal, output_dir=args.output_dir, run_id=run_id, binding=binding, scenario=scenario))
            if args.resume_from:
                resume_reference = {
                    "schema": "gain-gate-v3-resume-reference/1", "run_id": args.resume_from,
                    "binding": dict(binding), "verified_shards": referenced_shards,
                }
                _write_json_atomic(run_root / "resume_reference.json", resume_reference)
        if shards.current_binding() != binding:
            raise ValueError("V3_BINDING_CHANGED_DURING_RUN")
        resource = shards.resource_receipt(stage=args.stage, started_wall=start_wall, started_cpu=start_cpu, peak_rss_bytes=_peak_rss())
        if not resource["within_limits"]:
            raise RuntimeError("V3_RESOURCE_LIMIT_RECEIPT_FAILED")
        if args.stage == "smoke":
            summary_path = run_root / "SMOKE_SUMMARY.json"
            smoke_summary = json.loads(summary_path.read_text(encoding="ascii"))
            smoke_summary["resource_receipt"] = resource
            manifest_path = run_root / "SMOKE_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            checks = {
                "scenario_scale_and_regions": True,
                "complete_public_trajectory": True,
                "paired_label_independent_recomputation": True,
                "committed_shard_integrity": True,
            }
            for instance in manifest["instances"]:
                marker = shards.verify_scenario_shard(instance["marker_path"], expected_binding=binding)
                if marker.get("marker_sha256") != instance.get("marker_sha256"):
                    raise ValueError("V3_SMOKE_MARKER_HASH_MISMATCH")
                with gzip.open(Path(instance["marker_path"]).with_name(marker["shard_file"]), "rt", encoding="ascii") as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                headers = [row for row in rows if row.get("schema") == "gain-gate-v3-shard-header/1"]
                if len(headers) != 1:
                    raise ValueError("V3_SMOKE_TRAJECTORY_HEADER_REQUIRED")
                header = headers[0]
                scenario = header["scenario"]
                trace = header["trajectory"].get("trajectory_trace", [])
                base = scenario.get("base_scenario", {})
                if (
                    base.get("map", {}).get("rows") != 100 or base.get("map", {}).get("cols") != 100
                    or len(base.get("platforms", [])) != 30 or len(base.get("tasks", [])) != 54
                    or base.get("region_contract") != "relief-regions-3x3-v1"
                    or len(trace) != 90 or not trace[-1].get("terminated") and not trace[-1].get("truncated")
                ):
                    checks["scenario_scale_and_regions"] = False
                    checks["complete_public_trajectory"] = False
                for row in rows:
                    if row.get("schema") == "gain-gate-v3-shard-row/1":
                        oracle = row.get("training_row", {}).get("oracle_receipt")
                        if not isinstance(oracle, Mapping) or not isinstance(oracle.get("valid"), bool):
                            checks["paired_label_independent_recomputation"] = False
            if not all(checks.values()):
                raise ValueError(f"V3_SMOKE_CHECKS_FAILED:{checks}")
            if shards.current_binding() != binding:
                raise ValueError("V3_BINDING_CHANGED_DURING_RUN")
            manifest["status"] = "complete"
            manifest["completed_instances"] = 18
            manifest["checks"] = checks
            _write_json_atomic(manifest_path, manifest)
            smoke_summary["status"] = "complete"
            smoke_summary["completed_instances"] = 18
            smoke_summary["checks"] = checks
            smoke_summary["manifest_sha256"] = shards.sha256_file(manifest_path)
            _write_json_atomic(summary_path, smoke_summary)
        _write_run_receipt(run_root, run_id=run_id, stage=args.stage, binding=binding, resource=resource, results=results, review=review, resume_reference=resume_reference)
        print(json.dumps({"run_id": run_id, "stage": args.stage, "result_count": len(results), "resource_receipt": resource}, sort_keys=True, indent=2))
        return 0
    except BaseException as exc:
        failed_resource = shards.resource_receipt(
            stage=args.stage, started_wall=start_wall, started_cpu=start_cpu, peak_rss_bytes=_peak_rss(),
        )
        failed_resource["exit_state"] = "failed"
        failure = {
            "schema": "gain-gate-v3-run-failure/1", "run_id": run_id, "stage": args.stage,
            "binding": binding, "error_type": type(exc).__name__, "error": str(exc),
            "resource_receipt": failed_resource, "completion_receipt_written": False,
        }
        try:
            _write_json_atomic(run_root / "FAILURE.json", failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
