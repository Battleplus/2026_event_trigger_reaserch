from __future__ import annotations

import cProfile
import gzip
import hashlib
import json
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "performance"
OUT.mkdir(exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
PROFILE_ENABLED = MODE.endswith("_profile") or MODE == "baseline"
PID_PATH = OUT / f"{MODE}.pid"
PID_PATH.write_text(str(os.getpid()), encoding="ascii")
sys.path.insert(0, str(ROOT / "package" / "civilian"))
import v3_runner

spec = {
    "split": "train",
    "condition": "task_update_or_arrival",
    "instance_index": 1,
    "seed_variant": 900002,
}
build_wall0, build_cpu0 = time.perf_counter(), time.process_time()
scenario = v3_runner._scenario_factory()(**spec)
scenario_build_wall, scenario_build_cpu = time.perf_counter() - build_wall0, time.process_time() - build_cpu0
source = "Periodic-4min"
v3_runner._enforce_runtime()

timings: dict[str, dict[str, float | int]] = {}


def wrap(owner, name: str, stage: str):
    original = getattr(owner, name)

    def measured(*args, **kwargs):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        try:
            return original(*args, **kwargs)
        finally:
            item = timings.setdefault(stage, {"calls": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0})
            item["calls"] = int(item["calls"]) + 1
            item["wall_seconds"] = float(item["wall_seconds"]) + time.perf_counter() - wall0
            item["cpu_seconds"] = float(item["cpu_seconds"]) + time.process_time() - cpu0

    setattr(owner, name, measured)
    return original


import counterfactual_pairs as pairs
import fixed_replanner
import gain_oracle
import v3_scenarios
from event_runner import _modules

env_cls = _modules()[0]
wrap(env_cls, "step", "environment_step")
wrap(fixed_replanner, "replan", "replanner")
wrap(pairs, "collect_paired_label", "paired_label")
wrap(gain_oracle, "validate_gain_label", "oracle_validation")
wrap(v3_scenarios, "sample_v3_future_suffix", "suffix_sampling")

profile = cProfile.Profile()
wall0, cpu0 = time.perf_counter(), time.process_time()
rows = []
summary = {"trajectory_trace": [], "elapsed_ticks": 0, "ordinary_candidate_count": 0}
if PROFILE_ENABLED:
    profile.enable()
failure = None
try:
    scenario_out, rows, summary = v3_runner._scenario_and_row(
        spec, source, max_minutes=None, max_cpu_seconds=1200, scenario=scenario,
    )
except BaseException as exc:
    failure = {"type": type(exc).__name__, "message": str(exc)}
finally:
    if PROFILE_ENABLED:
        profile.disable()
        profile.dump_stats(OUT / f"{MODE}.pstats")
wall_seconds, cpu_seconds = time.perf_counter() - wall0, time.process_time() - cpu0

def sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()

trace = summary["trajectory_trace"]
actions = [{"tick": row["tick"], "action": row["action"]} for row in trace]
physical = [{
    "tick": row["tick"], "time_before": row["time_before"], "public_before": row["public_before"],
    "public_after": row["public_after"], "reward": row["reward"], "terminated": row["terminated"],
    "truncated": row["truncated"], "step_info": row["step_info"],
} for row in trace]
labels = []
for row in rows:
    label = row["label"]
    labels.append({
        "state_id": row["state_id"], "forced": row["forced"],
        "replan_action": row["replan_action"], "label": label,
        "oracle_receipt": row["oracle_receipt"],
    })

profile_path = OUT / f"{MODE}.pstats" if PROFILE_ENABLED else None
top = []
if PROFILE_ENABLED:
    stats = pstats.Stats(profile).strip_dirs()
    for (filename, line, func), (cc, nc, tt, ct, _) in list(stats.stats.items()):
        top.append({"function": f"{filename}:{line}:{func}", "calls": nc, "primitive_calls": cc, "total_seconds": tt, "cumulative_seconds": ct})
    top.sort(key=lambda row: row["cumulative_seconds"], reverse=True)

def pair_summaries(rows):
    result = []
    for row in rows:
        label = row["label"]
        for pair in label.get("pairs", []):
            arms = {}
            for arm_name in ("keep", "replan"):
                arm = pair.get(arm_name)
                trace = arm.get("trace") if isinstance(arm, dict) else None
                truth = trace.get("raw_truth", {}) if isinstance(trace, dict) else {}
                completed = sorted(
                    str(task.get("id")) for task in truth.get("tasks", [])
                    if task.get("status") == "completed"
                )
                arms[arm_name] = None if trace is None else {
                    "horizon_actual_minutes": trace.get("horizon_actual_minutes"),
                    "valid_mask": trace.get("valid_mask"),
                    "action_applied_step": trace.get("action_applied_step"),
                    "action_expired_reason": trace.get("action_expired_reason"),
                    "planned_action_sha256": trace.get("planned_action_sha256"),
                    "completed_task_ids": completed,
                    "collection_error": trace.get("collection_error"),
                    "J_total": arm.get("cost", {}).get("J_total") if isinstance(arm.get("cost"), dict) else None,
                }
            result.append({
                "state_id": pair.get("state_id"), "suffix_seed": pair.get("suffix_seed"),
                "valid_mask": pair.get("valid_mask"), "unknown_reason": pair.get("unknown_reason"),
                "horizon_actual_minutes": pair.get("horizon_actual_minutes"),
                "collection_error": pair.get("collection_error"), "arms": arms,
            })
    return result

report = {
    "mode": MODE, "pid": os.getpid(), "scenario_id": scenario["parent_id"],
    "scenario_content_sha256": scenario["scenario_content_sha256"], "source": source,
    "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
    "scenario_build_wall_seconds": scenario_build_wall, "scenario_build_cpu_seconds": scenario_build_cpu,
    "elapsed_ticks": summary["elapsed_ticks"], "selected_state_count": len(rows),
    "ordinary_candidate_count": summary["ordinary_candidate_count"],
    "actions_sha256": sha(actions), "physical_trajectory_sha256": sha(physical),
    "labels_sha256": sha(labels), "timings": timings, "profile_path": str(profile_path),
    "profile_top_cumulative": top[:40],
    "pair_summaries": pair_summaries(rows),
    "planner_cpu_seconds": sum(float(row["planner_cpu_seconds"]) + float(row["offline_label_planner_cpu_seconds"]) for row in rows),
    "label_cpu_cost_sum": sum(float(row["label"].get("replanner_cpu_seconds", 0.0)) for row in rows),
    "oracle_valid_count": sum(row.get("oracle_receipt", {}).get("valid") is True for row in rows),
    "oracle_invalid_count": sum(row.get("oracle_receipt", {}).get("valid") is False for row in rows),
    "valid_mask_count": sum(row.get("label", {}).get("valid_mask") is True for row in rows),
    "unknown_reasons": {str(reason): sum(row.get("label", {}).get("unknown_reason") == reason for row in rows) for reason in sorted({row.get("label", {}).get("unknown_reason") for row in rows if row.get("label", {}).get("unknown_reason")})},
    "failure": failure,
}
(OUT / f"{MODE}.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="ascii")
print(json.dumps({key: value for key, value in report.items() if key not in {"profile_top_cumulative"}}, sort_keys=True), flush=True)
if failure is not None:
    raise SystemExit(1)
