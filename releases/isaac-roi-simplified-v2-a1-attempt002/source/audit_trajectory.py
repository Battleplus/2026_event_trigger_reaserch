#!/usr/bin/env python3
"""Independent integrity and cost audit for persisted civil ROI trajectories."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


TOL_M = 0.02
TOL_WH = 1e-7
EXPECTED_TASK_IDS = {f"z{zone}-cell{cell}" for zone in range(9) for cell in range(9)}
BRANCHES = {"representative", "pair_prefix", "keep", "replan"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def _distance(a, b) -> float:
    if len(a) != 3 or len(b) != 3:
        raise ValueError("positions must be 3D")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _task_zone(task_id: str) -> int:
    return int(task_id.split("-", 1)[0][1:])


def _task_allowed_kind(task_id: str) -> str:
    cell = int(task_id.rsplit("cell", 1)[1])
    return "uav" if cell % 3 < 2 else "helicopter"


def audit(trajectory_path: Path, events_path: Path, manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(trajectory_path)
    events = _read_jsonl(events_path)
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("manifest.config is required for independent cost recomputation")

    failures: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, detail: str = ""):
        checks[name] = bool(condition)
        if not condition:
            failures.append(f"{name}: {detail or 'failed'}")

    required_receipt = ("public_task_manifest", "actor_manifest", "initial_positions_m")
    for key in required_receipt:
        check(f"receipt_has_{key}", isinstance(receipt.get(key), (dict, list)), f"missing or invalid {key}")
    task_manifest = receipt.get("public_task_manifest", {})
    actor_manifest = receipt.get("actor_manifest", {})
    initial_positions = receipt.get("initial_positions_m", {})
    if isinstance(task_manifest, list):
        task_manifest = {str(item["task_id"]): item for item in task_manifest if isinstance(item, dict) and "task_id" in item}
    if isinstance(actor_manifest, list):
        actor_manifest = {str(item["aircraft_id"]): item for item in actor_manifest if isinstance(item, dict) and "aircraft_id" in item}
    expected_ids = set(receipt.get("expected_task_ids", task_manifest))
    check("public_task_keyset_is_81_cells", set(task_manifest) == EXPECTED_TASK_IDS and expected_ids == EXPECTED_TASK_IDS,
          f"manifest={len(task_manifest)} expected={len(EXPECTED_TASK_IDS)}")
    check("task_manifest_is_public_only", all("target" not in str(k).lower() and "truth" not in str(k).lower()
                                                   for item in task_manifest.values() for k in item),
          "task manifest contains a hidden-target field")
    check("actor_manifest_has_36_platforms", len(actor_manifest) == 36, f"found {len(actor_manifest)}")

    for task_id in EXPECTED_TASK_IDS & set(task_manifest):
        task = task_manifest[task_id]
        expected_zone = _task_zone(task_id)
        expected_kind = _task_allowed_kind(task_id)
        if int(task.get("zone", -1)) != expected_zone:
            failures.append(f"task {task_id}: public zone mismatch")
        if task.get("allowed_kind") != expected_kind:
            failures.append(f"task {task_id}: expected allowed_kind={expected_kind}")
    checks["task_manifest_zone_and_type_rules"] = not any("public zone mismatch" in f or "allowed_kind" in f for f in failures)

    actors = {}
    for aircraft_id, item in actor_manifest.items():
        actors[aircraft_id] = {
            "kind": str(item.get("aircraft_type", item.get("type", item.get("kind", "")))),
            "zone": int(item.get("home_zone", item.get("zone", -1))),
            "max_speed_mps": (float(item["max_speed_mps"]) if item.get("max_speed_mps") is not None else None),
        }
    check("actor_manifest_has_valid_types_and_zones", all(
        a["kind"] in {"uav", "helicopter"} and 0 <= a["zone"] < 9
        for a in actors.values()), "invalid actor kind or zone")
    speed_data_complete = bool(actors) and all(a["max_speed_mps"] is not None and a["max_speed_mps"] > 0
                                                for a in actors.values())

    required_row_fields = {
        "run_id", "phase", "branch", "sim_time_s", "dt_s", "kit_frame", "aircraft_id",
        "aircraft_type", "zone", "task_id", "search_cell_visited", "task_completed",
        "target_discovered_ids", "mission_xy_km", "usd_position_m", "velocity_mps",
        "usd_pose_error_m", "previous_usd_position_m", "faulted", "energy", "communication",
    }
    missing = sorted(required_row_fields - set.intersection(*(set(r) for r in rows))) if rows else sorted(required_row_fields)
    check("trajectory_has_audit_fields", not missing, f"missing fields: {missing}")
    check("trajectory_nonempty", bool(rows), "no trajectory rows")
    check("receipt_status_complete", receipt.get("status") == "complete", str(receipt.get("status")))
    check("manifest_run_id_matches_receipt", manifest.get("run_id") == receipt.get("run_id"), "run IDs differ")

    row_by_key = {}
    branch_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_actor_time: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        branch = str(row.get("branch", ""))
        aid = str(row.get("aircraft_id", ""))
        try:
            t = int(row.get("sim_time_s"))
        except (TypeError, ValueError):
            failures.append(f"trajectory row {index}: invalid sim_time_s")
            continue
        key = (branch, aid, t)
        if key in row_by_key:
            failures.append(f"duplicate actor-second row {key}")
        row_by_key[key] = row
        branch_rows[branch].append(row)
        per_actor_time[(branch, aid)][t] = row
        if branch not in BRANCHES:
            failures.append(f"unknown branch {branch}")
        if aid not in actors:
            failures.append(f"unknown actor {aid}")
            continue
        actor = actors[aid]
        if row.get("aircraft_type") != actor["kind"]:
            failures.append(f"{branch}/{aid}/{t}: aircraft_type mismatch")
        if int(row.get("zone", -1)) != actor["zone"]:
            failures.append(f"{branch}/{aid}/{t}: home zone mismatch")
        task_id = row.get("task_id")
        if task_id is not None:
            task_id = str(task_id)
            task = task_manifest.get(task_id)
            if task is None:
                failures.append(f"{branch}/{aid}/{t}: unknown public task {task_id}")
            else:
                allowed = str(task.get("allowed_kind", _task_allowed_kind(task_id)))
                zone = int(task.get("zone", _task_zone(task_id)))
                if actor["kind"] != allowed:
                    failures.append(f"{branch}/{aid}/{t}: {actor['kind']} assigned incompatible {allowed} task {task_id}")
                if actor["kind"] == "helicopter" and (actor["zone"] != zone or int(row.get("zone", -1)) != zone):
                    failures.append(f"{branch}/{aid}/{t}: helicopter crossed zone boundary for {task_id}")
        if row.get("hidden_truth_logged") is not False:
            failures.append(f"{branch}/{aid}/{t}: hidden truth is not explicitly excluded")
        position = row.get("usd_position_m")
        velocity = row.get("velocity_mps")
        if not isinstance(position, list) or not isinstance(velocity, list) or len(position) != 3 or len(velocity) != 3:
            failures.append(f"{branch}/{aid}/{t}: invalid 3D USD position/velocity")
            continue
        if not all(math.isfinite(float(v)) for v in position + velocity):
            failures.append(f"{branch}/{aid}/{t}: nonfinite position/velocity")
        if float(row.get("usd_pose_error_m", math.inf)) > TOL_M:
            failures.append(f"{branch}/{aid}/{t}: USD/motion pose error exceeds tolerance")
        mission_xy = row.get("mission_xy_km")
        if not isinstance(mission_xy, list) or len(mission_xy) != 2 or any(
            abs(float(mission_xy[i]) - float(position[i]) / 1000.0) > 1e-6 for i in range(2)
        ):
            failures.append(f"{branch}/{aid}/{t}: mission XY differs from USD world position")
        dt = float(row.get("dt_s", 0.0))
        if dt <= 0 or not math.isclose(dt, 1.0, abs_tol=1e-9):
            failures.append(f"{branch}/{aid}/{t}: dt_s must be exactly 1 second")
        speed = math.sqrt(sum(float(v) ** 2 for v in velocity))
        if actor["max_speed_mps"] is not None and speed > actor["max_speed_mps"] + 0.05:
            failures.append(f"{branch}/{aid}/{t}: speed exceeds max_speed_mps")

    # Every branch is a complete synchronized sample set over its declared horizon.
    branch_durations = receipt.get("simulated_seconds_by_branch", {})
    for branch, group in branch_rows.items():
        times = sorted({int(r["sim_time_s"]) for r in group})
        expected_count = int(branch_durations.get(branch, max(times, default=0)))
        expected_start = times[0] if times else 1
        if branch in {"representative", "pair_prefix"}:
            expected_start = 1
        if branch in {"keep", "replan"} and "pair_prefix" in branch_durations:
            expected_start = int(branch_durations["pair_prefix"]) + 1
        expected_times = set(range(expected_start, expected_start + expected_count))
        check(f"{branch}_time_coverage", set(times) == expected_times,
              f"observed {len(times)} seconds, expected {len(expected_times)}")
        expected_actors = set(actors)
        check(f"{branch}_actor_time_keyset", all(set(per_actor_time[(branch, aid)]) == expected_times for aid in expected_actors),
              "missing or extra actor-second rows")

    check("all_public_task_keys_audited", set(task_manifest) == EXPECTED_TASK_IDS,
          "the full 81-task public manifest is required; unfinished tasks remain in the score denominator")

    # Independently recompute each actor-second energy increment from USD displacement.
    rates = config.get("energy_wh_per_km", {})
    recomputed_energy: dict[str, float] = defaultdict(float)
    branch_energy: dict[str, float] = defaultdict(float)
    position_errors = []
    for (branch, aid), samples in per_actor_time.items():
        if aid not in actors:
            continue
        previous = None
        if branch in {"representative", "pair_prefix"}:
            raw_initial = initial_positions.get(aid)
            if raw_initial is not None:
                previous = [float(x) for x in raw_initial]
        elif branch in {"keep", "replan"}:
            prefix = per_actor_time.get(("pair_prefix", aid), {})
            prefix_end = max(prefix, default=None)
            if prefix_end is not None:
                previous = list(map(float, prefix[prefix_end]["usd_position_m"]))
        for t in sorted(samples):
            row = samples[t]
            current = list(map(float, row["usd_position_m"]))
            saved_previous = row.get("previous_usd_position_m")
            if saved_previous is not None:
                saved_previous = list(map(float, saved_previous))
                if previous is not None and _distance(previous, saved_previous) > TOL_M:
                    failures.append(f"{branch}/{aid}/{t}: persisted previous position does not chain")
                previous = saved_previous
            if previous is None:
                failures.append(f"{branch}/{aid}/{t}: no independent prior USD position available")
                previous = current
            distance_m = _distance(previous, current)
            expected_wh = distance_m * float(rates.get(actors[aid]["kind"], math.nan)) / 1000.0
            actual_wh = float(row.get("energy", {}).get("incremental_wh", math.nan))
            if not math.isfinite(expected_wh) or not math.isclose(actual_wh, expected_wh, rel_tol=1e-7, abs_tol=TOL_WH):
                failures.append(f"{branch}/{aid}/{t}: energy mismatch saved={actual_wh} recomputed={expected_wh}")
            recomputed_energy[aid] += expected_wh
            branch_energy[branch] += expected_wh
            position_errors.append(_distance(previous, saved_previous) if saved_previous is not None else 0.0)
            previous = current

    # Compare event-derived task and discovery labels with the per-second records.
    event_ids = set()
    completion_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    discovery_events: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    fault_events: dict[tuple[str, str], int] = {}
    donor_events = []
    for index, event in enumerate(events):
        event_id = event.get("event_id")
        if not event_id or event_id in event_ids:
            failures.append(f"event {index}: missing or duplicate event_id")
        event_ids.add(event_id)
        branch = str(event.get("branch", ""))
        kind = str(event.get("event", ""))
        aid = event.get("aircraft_id")
        t = int(event.get("sim_time_s", -1))
        task_id = event.get("task_id")
        target_id = event.get("target_id")
        if kind == "search_cell_complete":
            completion_events[(branch, str(task_id))].append(event)
            if aid not in actors or task_id not in task_manifest:
                failures.append(f"{branch}/{t}: completion references unknown actor/task")
            elif actors[aid]["kind"] != task_manifest[task_id].get("allowed_kind"):
                failures.append(f"{branch}/{t}: completion type incompatible with task {task_id}")
            elif actors[aid]["kind"] == "helicopter" and actors[aid]["zone"] != int(task_manifest[task_id]["zone"]):
                failures.append(f"{branch}/{t}: helicopter completed a task outside its home zone")
        elif kind == "target_discovered" and target_id is not None:
            discovery_events[(branch, str(aid), t)].add(str(target_id))
        elif kind == "platform_fault" and aid is not None:
            fault_events[(branch, str(aid))] = min(t, fault_events.get((branch, str(aid)), t))
        elif kind == "donor_support_transfer":
            donor_events.append(event)
        for forbidden in ("truth_xy", "private_xy", "target_position", "hidden_coordinates"):
            if forbidden in event:
                failures.append(f"event {index}: contains forbidden truth field {forbidden}")

    duplicate_completions = [key for key, values in completion_events.items() if len(values) != 1]
    check("search_completion_events_deduplicated", not duplicate_completions,
          f"duplicate completion keys: {duplicate_completions[:8]}")
    discovered_ids = defaultdict(set)
    duplicate_discoveries = []
    for key, values in discovery_events.items():
        branch, aid, t = key
        for target_id in values:
            identity = (branch, target_id)
            if target_id in discovered_ids[branch]:
                duplicate_discoveries.append(identity)
            discovered_ids[branch].add(target_id)
    check("target_discoveries_deduplicated", not duplicate_discoveries,
          f"duplicate discovery IDs: {duplicate_discoveries[:8]}")

    event_completion_keys = set(completion_events)
    row_completion_keys = set()
    row_discovery_keys = set()
    for row in rows:
        branch = str(row.get("branch")); aid = str(row.get("aircraft_id")); t = int(row.get("sim_time_s", -1))
        if row.get("task_completed"):
            row_completion_keys.add((branch, str(row.get("task_id"))))
            matches = [e for e in completion_events.get((branch, str(row.get("task_id"))), [])
                       if str(e.get("aircraft_id")) == aid and int(e.get("sim_time_s", -2)) == t]
            if len(matches) != 1:
                failures.append(f"{branch}/{aid}/{t}: task_completed label lacks exactly one event")
        found = row.get("target_discovered_ids", [])
        if not isinstance(found, list) or len(found) != len(set(found)):
            failures.append(f"{branch}/{aid}/{t}: malformed or duplicate target_discovered_ids")
        row_ids = set(map(str, found if isinstance(found, list) else []))
        event_ids_here = discovery_events.get((branch, aid, t), set())
        if row_ids != event_ids_here:
            failures.append(f"{branch}/{aid}/{t}: discovery label/event mismatch")
        row_discovery_keys |= {(branch, aid, t, target_id) for target_id in row_ids}
        if bool(row.get("faulted")) != ((branch, aid) in fault_events and t >= fault_events[(branch, aid)]):
            failures.append(f"{branch}/{aid}/{t}: faulted label does not match fault event")
    check("completion_labels_match_events", row_completion_keys == event_completion_keys,
          "row labels and separate completion events differ")
    check("discovery_labels_match_events", not any("discovery label/event mismatch" in f for f in failures),
          "row discovery IDs and separate events differ")
    check("fault_labels_match_events", not any("faulted label" in f for f in failures),
          "row fault flags and platform_fault events differ")

    # Validate actual hold after declared fault/safety events using USD motion.
    for (branch, aid), samples in per_actor_time.items():
        actor = actors.get(aid)
        if actor is None:
            continue
        ordered = sorted(samples)
        fault_time = fault_events.get((branch, aid))
        for i, t in enumerate(ordered):
            row = samples[t]
            safety = bool(row.get("energy", {}).get("safety_pause", False))
            failed = fault_time is not None and t >= fault_time
            if not safety and not failed:
                continue
            current = row.get("usd_position_m", [])
            next_time = t + 1
            next_row = samples.get(next_time)
            if next_row is None:
                continue
            moved = _distance(current, next_row["usd_position_m"])
            if moved > TOL_M:
                failures.append(f"{branch}/{aid}/{t}: declared hold moved {moved:.3f} m by next USD sample")
    check("declared_holds_are_stationary", not any("declared hold moved" in f for f in failures),
          "one or more safety/fault holds moved")

    for event in donor_events:
        donor = event.get("from_aircraft_id")
        recipient = event.get("to_aircraft_id")
        task_id = event.get("task_id")
        if donor not in actors or recipient not in actors or task_id not in task_manifest:
            failures.append("donor transfer references unknown actor/task")
            continue
        if actors[donor]["kind"] != "uav" or actors[recipient]["kind"] != "uav":
            failures.append("donor transfer must be UAV-to-UAV")
        if actors[recipient]["kind"] == "helicopter":
            failures.append("helicopter received cross-zone support")
        if actors[recipient]["kind"] != task_manifest[task_id].get("allowed_kind"):
            failures.append(f"donor transfer assigned incompatible platform to {task_id}")
    check("donor_support_respects_type_and_zone_lock", not any("donor transfer" in f or "helicopter received" in f for f in failures),
          "invalid donor support")

    branch_metrics = {}
    for branch in sorted(branch_rows):
        completed = {task for b, task in event_completion_keys if b == branch}
        discovered = set(discovered_ids.get(branch, set()))
        branch_rows_here = branch_rows[branch]
        actor_ids = {str(row["aircraft_id"]) for row in branch_rows_here}
        faulted = {aid for b, aid in fault_events if b == branch}
        serviced = {str(event.get("target_id")) for event in events
                    if event.get("branch") == branch and event.get("event") == "abstract_strike_complete"
                    and event.get("target_id") is not None}
        branch_events = [event for event in events if event.get("branch") == branch]
        command_attempts = sum(event.get("event") in {"command_delivered", "command_dropped"}
                                for event in branch_events)
        planner_calls = sum(event.get("event") in {"planner_call", "replan_catalog_choice"}
                            for event in branch_events)
        switches = sum(event.get("event") in {"controller_switch", "planner_switch", "policy_switch"}
                       for event in branch_events)
        violations = sum(event.get("event") in {"safety_violation", "boundary_violation", "constraint_violation"}
                         for event in branch_events)
        branch_metrics[branch] = {
            "public_task_denominator": len(EXPECTED_TASK_IDS),
            "completed_public_tasks": len(completed),
            "remaining_public_tasks": len(EXPECTED_TASK_IDS - completed),
            "unfinished_public_task_ids": sorted(EXPECTED_TASK_IDS - completed),
            "discovered_target_count": len(discovered),
            "serviced_target_count": len(serviced),
            "analytical_energy_kwh_recomputed": float(branch_energy.get(branch, 0.0)) / 1000.0,
            "command_attempts": int(command_attempts),
            "switches": int(switches),
            "planner_calls": int(planner_calls),
            "violations": int(violations),
            "faulted_aircraft_count": len(faulted),
            "row_count": len(branch_rows_here),
            "actor_count": len(actor_ids),
        }

    pair_delta = None
    if "keep" in branch_metrics and "replan" in branch_metrics:
        keep = branch_metrics["keep"]
        replan = branch_metrics["replan"]
        pair_delta = {
            "completed_public_tasks_keep_minus_replan": keep["completed_public_tasks"] - replan["completed_public_tasks"],
            "discovered_target_count_keep_minus_replan": keep["discovered_target_count"] - replan["discovered_target_count"],
            "analytical_energy_kwh_keep_minus_replan": keep["analytical_energy_kwh_recomputed"] - replan["analytical_energy_kwh_recomputed"],
            "serviced_target_count_keep_minus_replan": keep["serviced_target_count"] - replan["serviced_target_count"],
            "command_attempts_keep_minus_replan": keep["command_attempts"] - replan["command_attempts"],
            "switches_keep_minus_replan": keep["switches"] - replan["switches"],
            "planner_calls_keep_minus_replan": keep["planner_calls"] - replan["planner_calls"],
            "violations_keep_minus_replan": keep["violations"] - replan["violations"],
            "faulted_aircraft_count_keep_minus_replan": keep["faulted_aircraft_count"] - replan["faulted_aircraft_count"],
        }
        production_pair = receipt.get("pair_summary")
        if production_pair is not None:
            expected = production_pair.get("gain_keep_minus_replan")
            if isinstance(expected, dict):
                for key, actual in pair_delta.items():
                    if key in expected and not math.isclose(float(expected[key]), float(actual), rel_tol=1e-9, abs_tol=1e-9):
                        failures.append(f"production pair gain mismatch for {key}: saved={expected[key]} recomputed={actual}")
                check("production_pair_summary_matches_raw", not any("production pair gain mismatch" in f for f in failures),
                      "production summary differs from raw recomputation")

    checks["all_validation_failures_absent"] = not failures
    audit_status = "fail" if failures else ("pass" if speed_data_complete else "not_verified")
    return {
        "schema_version": 1,
        "audit": "independent_raw_trajectory_audit",
        "status": audit_status,
        "run_id": receipt.get("run_id"),
        "trajectory_rows": len(rows),
        "event_rows": len(events),
        "checks": checks,
        "failures": failures,
        "speed_validation": {
            "status": "verified" if speed_data_complete else "NOT_VERIFIED",
            "reason": None if speed_data_complete else "actor manifest does not define max_speed_mps for every actor",
        },
        "branch_metrics": branch_metrics,
        "pair_gain_keep_minus_replan": pair_delta,
        "production_scalar": {"status": "NOT_VERIFIED", "reason": "pilot scalar formula is not frozen"},
        "energy_recalculation": {
            "model": "USD position displacement in meters * configured Wh/km / 1000",
            "per_actor_wh": dict(recomputed_energy),
            "per_branch_wh": dict(branch_energy),
        },
        "limitations": ["energy is an analytical estimate, not an Isaac physical measurement",
                        "the run uses 20 km trajectory altitude with 2D analytical target discovery, not physical sensing",
                        "platform motion uses source kinematic models; no turn-rate realism claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest or args.trajectory.parent / "manifest.json"
    receipt = args.receipt or args.trajectory.parent / "run_receipt.json"
    report = audit(args.trajectory, args.events, manifest, receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": report["status"], "failures": len(report["failures"]),
                      "production_scalar": report["production_scalar"]["status"],
                      "output": str(args.output)}, separators=(",", ":")))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
