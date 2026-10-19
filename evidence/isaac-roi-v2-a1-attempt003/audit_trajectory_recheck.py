#!/usr/bin/env python3
"""Independent integrity and cost audit for persisted civil ROI trajectories."""
from __future__ import annotations

import argparse
import gzip
import hashlib
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


def _inside_bounds(xy, bounds, tolerance=1e-9) -> bool:
    xmin, ymin, xmax, ymax = map(float, bounds)
    return xmin - tolerance <= float(xy[0]) <= xmax + tolerance and ymin - tolerance <= float(xy[1]) <= ymax + tolerance


def audit(trajectory_path: Path, events_path: Path, manifest_path: Path, receipt_path: Path,
          truth_path: Path | None = None) -> dict[str, Any]:
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

    geometry_data_complete = all(
        isinstance(task.get("public_bounds_km"), list) and len(task["public_bounds_km"]) == 4
        and isinstance(task.get("sweep_waypoints_km"), list) and bool(task["sweep_waypoints_km"])
        for task in task_manifest.values()
    )
    zone_bounds = receipt.get("zone_bounds_km", {})
    zone_geometry_complete = all(str(zone) in zone_bounds and isinstance(zone_bounds[str(zone)], list)
                                 and len(zone_bounds[str(zone)]) == 4 for zone in range(9))
    geometry_tolerance_km = float(receipt.get("waypoint_tolerance_km", config.get("waypoint_tolerance_km", math.nan)))
    sensor_radius_km = float(receipt.get("sensor_radius_km", config.get("sensor_radius_km", math.nan)))
    geometry_data_complete = geometry_data_complete and math.isfinite(geometry_tolerance_km) and geometry_tolerance_km > 0
    truth_file = receipt.get("evaluator_truth_file")
    if truth_path is None and truth_file:
        truth_path = receipt_path.parent / str(truth_file)
    elif truth_path is None:
        candidate = receipt_path.parent / "evaluator_truth.json"
        truth_path = candidate if candidate.exists() else None
    truth = None
    truth_digest_valid = False
    if truth_path is not None and truth_path.exists():
        truth_bytes = truth_path.read_bytes()
        truth_digest_valid = hashlib.sha256(truth_bytes).hexdigest() == receipt.get("evaluator_truth_sha256")
        if truth_digest_valid:
            truth = _read_json(truth_path)
        else:
            failures.append("evaluator truth artifact digest does not match receipt")
    raw_targets = truth.get("targets") if isinstance(truth, dict) else None
    truth_targets = {}
    if isinstance(raw_targets, list):
        for target in raw_targets:
            if isinstance(target, dict) and target.get("target_id") is not None:
                truth_targets[str(target["target_id"])] = target
    truth_data_complete = (truth_digest_valid and isinstance(raw_targets, list)
                           and len(truth_targets) == len(raw_targets)
                           and math.isfinite(sensor_radius_km) and sensor_radius_km > 0)

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
        elif zone_geometry_complete and actor["kind"] == "helicopter":
            home_bounds = zone_bounds.get(str(actor["zone"]))
            if home_bounds is not None and not _inside_bounds(mission_xy, home_bounds):
                failures.append(f"{branch}/{aid}/{t}: helicopter USD XY outside home-zone bounds")
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

    # Replay the public sweep from raw USD XY samples, independent of mission code.
    completion_geometry_complete = geometry_data_complete
    if completion_geometry_complete:
        for task_id, task in task_manifest.items():
            bounds = task["public_bounds_km"]
            if (not all(math.isfinite(float(v)) for v in bounds) or float(bounds[0]) > float(bounds[2])
                    or float(bounds[1]) > float(bounds[3]) or not task["sweep_waypoints_km"]):
                completion_geometry_complete = False
                break
            if any(not _inside_bounds(point, bounds) for point in task["sweep_waypoints_km"]):
                completion_geometry_complete = False
                break
    completion_geometry_failures = []
    if completion_geometry_complete:
        prefix_progress = {}
        for branch in ("representative", "pair_prefix", "keep", "replan"):
            branch_group = [row for row in rows if str(row.get("branch")) == branch]
            if branch in {"keep", "replan"}:
                progress = dict(prefix_progress)
            else:
                progress = {}
            for row in branch_group:
                aid = str(row.get("aircraft_id", ""))
                task_id = row.get("task_id")
                if task_id is None or str(task_id) not in task_manifest:
                    if row.get("search_cell_visited") or row.get("task_completed"):
                        completion_geometry_failures.append(f"{branch}/{aid}/{row.get('sim_time_s')}: completion label without active public task")
                    continue
                task_id = str(task_id)
                waypoints = task_manifest[task_id]["sweep_waypoints_km"]
                state_key = (aid, task_id)
                waypoint_index = int(progress.get(state_key, 0))
                xy = row.get("mission_xy_km")
                visited = False
                completed = False
                if waypoint_index < len(waypoints) and isinstance(xy, list) and len(xy) == 2:
                    waypoint = waypoints[waypoint_index]
                    visited = math.dist((float(xy[0]), float(xy[1])), (float(waypoint[0]), float(waypoint[1]))) <= geometry_tolerance_km
                    if visited:
                        waypoint_index += 1
                        completed = waypoint_index == len(waypoints)
                progress[state_key] = waypoint_index
                if bool(row.get("search_cell_visited")) != visited:
                    completion_geometry_failures.append(f"{branch}/{aid}/{row.get('sim_time_s')}: search_cell_visited differs from USD sweep replay")
                if bool(row.get("task_completed")) != completed:
                    completion_geometry_failures.append(f"{branch}/{aid}/{row.get('sim_time_s')}: task_completed differs from USD sweep replay")
            if branch == "pair_prefix":
                prefix_progress = dict(progress)
    if completion_geometry_complete:
        check("completion_labels_match_usd_sweep_replay", not completion_geometry_failures,
              f"mismatches={completion_geometry_failures[:8]}")
    else:
        checks["completion_labels_match_usd_sweep_replay"] = "NOT_VERIFIED"

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
            dt = float(row.get("dt_s", 0.0))
            if dt > 0 and actors[aid]["max_speed_mps"] is not None:
                displacement_speed = distance_m / dt
                if displacement_speed > actors[aid]["max_speed_mps"] + 0.05:
                    failures.append(f"{branch}/{aid}/{t}: USD displacement speed exceeds max_speed_mps")
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

    discovery_fields_complete = truth_data_complete and all(
        isinstance(row.get("public_target_observations"), list) for row in rows
    ) and all(
        isinstance(target.get("target_xy_km"), list) and len(target["target_xy_km"]) == 2
        and target.get("kind") in {"uav", "helicopter"} and 0 <= int(target.get("zone", -1)) < 9
        for target in truth_targets.values()
    )
    discovery_geometry_failures = []
    if discovery_fields_complete:
        discovered_by_branch: dict[str, set[str]] = defaultdict(set)
        initialized_branches = set()
        for row in rows:
            branch = str(row.get("branch", ""))
            if branch in {"keep", "replan"} and branch not in initialized_branches:
                discovered_by_branch[branch].update(discovered_by_branch["pair_prefix"])
            initialized_branches.add(branch)
            aid = str(row.get("aircraft_id", ""))
            time_s = int(row.get("sim_time_s", -1))
            task_id = row.get("task_id")
            expected_now = set()
            if task_id is not None and str(task_id) in task_manifest and not bool(row.get("faulted")):
                task_zone = int(task_manifest[str(task_id)]["zone"])
                xy = row.get("mission_xy_km")
                if isinstance(xy, list) and len(xy) == 2:
                    for target_id, target in truth_targets.items():
                        if (target_id not in discovered_by_branch[branch]
                                and int(target["zone"]) == task_zone
                                and math.dist(tuple(map(float, xy)), tuple(map(float, target["target_xy_km"]))) <= sensor_radius_km):
                            expected_now.add(target_id)
            row_ids = set(map(str, row.get("target_discovered_ids", [])))
            observations = row.get("public_target_observations", [])
            observation_ids = {str(item.get("target_id")) for item in observations if isinstance(item, dict)}
            if row_ids != expected_now:
                discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: discovered IDs differ from evaluator-truth radius replay")
            if observation_ids != expected_now or len(observations) != len(observation_ids):
                discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: public observation payload differs from first geometric detections")
            for item in observations:
                if not isinstance(item, dict):
                    continue
                target_id = str(item.get("target_id"))
                target = truth_targets.get(target_id)
                if target is None:
                    discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: public observation has unknown target {target_id}")
                    continue
                if (int(item.get("zone", -1)) != int(target["zone"])
                        or item.get("kind") != target["kind"]
                        or item.get("target_xy_km") != target["target_xy_km"]
                        or int(item.get("observed_at_sim_time_s", -2)) != time_s):
                    discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: public observation disagrees with evaluator truth/time")
            for target_id in expected_now:
                matching_events = [event for event in events
                                   if event.get("event") == "target_discovered"
                                   and str(event.get("branch")) == branch
                                   and str(event.get("aircraft_id")) == aid
                                   and int(event.get("sim_time_s", -2)) == time_s
                                   and str(event.get("target_id")) == target_id]
                if len(matching_events) != 1:
                    discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: geometric detection lacks exactly one event for {target_id}")
                else:
                    event = matching_events[0]
                    target = truth_targets[target_id]
                    if (event.get("task_id") != task_id or int(event.get("zone", -1)) != int(target["zone"])
                            or event.get("target_xy_km") != target["target_xy_km"] or event.get("kind") != target["kind"]):
                        discovery_geometry_failures.append(f"{branch}/{aid}/{time_s}: discovery event disagrees with public observation/truth")
            discovered_by_branch[branch].update(expected_now)
        check("discovery_labels_match_evaluator_truth_radius", not discovery_geometry_failures,
              f"mismatches={discovery_geometry_failures[:8]}")
    else:
        checks["discovery_labels_match_evaluator_truth_radius"] = "NOT_VERIFIED"

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

    communication_holds: dict[tuple[str, str], int] = {}
    communication_hold_failures = []
    for event in events:
        branch = str(event.get("branch", ""))
        aid = str(event.get("aircraft_id", ""))
        event_time = int(event.get("sim_time_s", -1))
        if event.get("event") == "communication_safety_hold":
            key = (branch, aid)
            if key in communication_holds:
                communication_hold_failures.append(f"{branch}/{aid}: duplicate communication hold without resume")
            communication_holds[key] = event_time
            hold_position = event.get("details", {}).get("hold_position_m")
            row = per_actor_time.get(key, {}).get(event_time)
            if row is None or hold_position is None or _distance(row.get("usd_position_m", []), hold_position) > TOL_M:
                communication_hold_failures.append(f"{branch}/{aid}/{event_time}: hold event position differs from USD sample")
        elif event.get("event") == "communication_safety_resume":
            key = (branch, aid)
            start = communication_holds.pop(key, None)
            if start is None:
                communication_hold_failures.append(f"{branch}/{aid}/{event_time}: resume without hold")
                continue
            samples = per_actor_time.get(key, {})
            for t in range(start, event_time):
                row = samples.get(t)
                next_row = samples.get(t + 1)
                if row is None or next_row is None:
                    communication_hold_failures.append(f"{branch}/{aid}/{t}: communication hold interval has missing samples")
                    continue
                if _distance(row["usd_position_m"], next_row["usd_position_m"]) > TOL_M:
                    communication_hold_failures.append(f"{branch}/{aid}/{t}: moved during communication hold")
            if samples:
                baseline = samples.get(start, {}).get("usd_position_m")
                for t in range(start, event_time):
                    row = samples.get(t)
                    if row is not None and baseline is not None and _distance(row["usd_position_m"], baseline) > TOL_M:
                        communication_hold_failures.append(f"{branch}/{aid}/{t}: position drifted during communication hold")
    for (branch, aid), start in communication_holds.items():
        samples = per_actor_time.get((branch, aid), {})
        baseline = samples.get(start, {}).get("usd_position_m")
        for t in sorted(t for t in samples if t >= start):
            row = samples[t]
            if baseline is None or _distance(row["usd_position_m"], baseline) > TOL_M:
                communication_hold_failures.append(f"{branch}/{aid}/{t}: open communication hold moved or has no baseline")
    if any(event.get("event") in {"communication_safety_hold", "communication_safety_resume"} for event in events):
        check("communication_outage_holds_stationary_and_resume", not communication_hold_failures,
              f"mismatches={communication_hold_failures[:8]}")
    else:
        checks["communication_outage_holds_stationary_and_resume"] = "NOT_VERIFIED"

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
    prefix_completed = {task for b, task in event_completion_keys if b == "pair_prefix"}
    prefix_discovered = set(discovered_ids.get("pair_prefix", set()))
    prefix_energy_kwh = float(branch_energy.get("pair_prefix", 0.0)) / 1000.0
    for branch in sorted(branch_rows):
        window_completed = {task for b, task in event_completion_keys if b == branch}
        window_discovered = set(discovered_ids.get(branch, set()))
        completed = (prefix_completed | window_completed) if branch in {"keep", "replan"} else window_completed
        discovered = (prefix_discovered | window_discovered) if branch in {"keep", "replan"} else window_discovered
        terminal_energy_kwh = float(branch_energy.get(branch, 0.0)) / 1000.0
        if branch in {"keep", "replan"}:
            terminal_energy_kwh += prefix_energy_kwh
        branch_rows_here = branch_rows[branch]
        actor_ids = {str(row["aircraft_id"]) for row in branch_rows_here}
        faulted = {aid for b, aid in fault_events if b == branch}
        serviced = {str(event.get("target_id")) for event in events
                    if event.get("branch") == branch and event.get("event") == "abstract_strike_complete"
                    and event.get("target_id") is not None}
        branch_events = [event for event in events if event.get("branch") == branch]
        command_attempt_data_complete = all(
            isinstance(row.get("communication"), dict)
            and "command_attempted" in row["communication"]
            for row in branch_rows_here
        )
        command_attempts = (
            sum(bool(row["communication"]["command_attempted"]) for row in branch_rows_here)
            if command_attempt_data_complete else None
        )
        command_deliveries = sum(event.get("event") == "command_delivered" for event in branch_events)
        command_drops = sum(event.get("event") == "command_dropped" for event in branch_events)
        planner_calls = sum(event.get("event") in {"planner_call", "replan_catalog_choice"}
                            for event in branch_events)
        switches = sum(event.get("event") in {"controller_switch", "planner_switch", "policy_switch"}
                       for event in branch_events)
        violations = sum(event.get("event") in {"safety_violation", "boundary_violation", "constraint_violation"}
                         for event in branch_events)
        branch_metrics[branch] = {
            "public_task_denominator": len(EXPECTED_TASK_IDS),
            "completed_public_tasks": len(completed),
            "completed_public_tasks_window": len(window_completed),
            "completed_public_tasks_terminal_cumulative": len(completed),
            "remaining_public_tasks": len(EXPECTED_TASK_IDS - completed),
            "unfinished_public_task_ids": sorted(EXPECTED_TASK_IDS - completed),
            "discovered_target_count": len(discovered),
            "discovered_target_count_window": len(window_discovered),
            "discovered_target_count_terminal_cumulative": len(discovered),
            "serviced_target_count": len(serviced),
            "analytical_energy_kwh_recomputed": float(branch_energy.get(branch, 0.0)) / 1000.0,
            "analytical_energy_kwh_window": float(branch_energy.get(branch, 0.0)) / 1000.0,
            "analytical_energy_kwh_terminal_cumulative": terminal_energy_kwh,
            "command_attempts": int(command_attempts) if command_attempts is not None else None,
            "command_attempts_status": "verified_from_rows" if command_attempt_data_complete else "NOT_VERIFIED",
            "command_deliveries": int(command_deliveries),
            "command_drops": int(command_drops),
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
            "completed_public_tasks_window_keep_minus_replan": keep["completed_public_tasks_window"] - replan["completed_public_tasks_window"],
            "discovered_target_count_keep_minus_replan": keep["discovered_target_count"] - replan["discovered_target_count"],
            "discovered_target_count_window_keep_minus_replan": keep["discovered_target_count_window"] - replan["discovered_target_count_window"],
            "analytical_energy_kwh_keep_minus_replan": keep["analytical_energy_kwh_recomputed"] - replan["analytical_energy_kwh_recomputed"],
            "analytical_energy_kwh_terminal_cumulative_keep_minus_replan": keep["analytical_energy_kwh_terminal_cumulative"] - replan["analytical_energy_kwh_terminal_cumulative"],
            "serviced_target_count_keep_minus_replan": keep["serviced_target_count"] - replan["serviced_target_count"],
            "command_attempts_keep_minus_replan": (
                keep["command_attempts"] - replan["command_attempts"]
                if keep["command_attempts"] is not None and replan["command_attempts"] is not None
                else None
            ),
            "command_deliveries_keep_minus_replan": keep["command_deliveries"] - replan["command_deliveries"],
            "command_drops_keep_minus_replan": keep["command_drops"] - replan["command_drops"],
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

    checks["helicopter_positions_within_home_zone"] = (
        not any("helicopter USD XY outside home-zone bounds" in f for f in failures)
        if zone_geometry_complete else "NOT_VERIFIED"
    )
    checks["all_validation_failures_absent"] = not failures
    semantic_evidence_complete = (speed_data_complete and geometry_data_complete
                                  and completion_geometry_complete and truth_data_complete
                                  and discovery_fields_complete and zone_geometry_complete
                                  and checks["communication_outage_holds_stationary_and_resume"] is True)
    audit_status = "fail" if failures else ("pass" if semantic_evidence_complete else "not_verified")
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
            "basis": "USD position displacement per dt checked against manifest max_speed_mps",
            "reason": None if speed_data_complete else "actor manifest does not define max_speed_mps for every actor",
        },
        "completion_geometry_validation": {
            "status": "verified" if completion_geometry_complete else "NOT_VERIFIED",
            "basis": "USD XY samples replayed against public sweep waypoints in order",
        },
        "discovery_geometry_validation": {
            "status": "verified" if discovery_fields_complete else "NOT_VERIFIED",
            "basis": "first same-zone USD XY sample within sensor radius of offline evaluator truth",
        },
        "zone_geometry_validation": {
            "status": "verified" if zone_geometry_complete else "NOT_VERIFIED",
            "basis": "helicopter USD XY checked against declared home-zone rectangle",
        },
        "communication_hold_validation": {
            "status": ("verified" if checks["communication_outage_holds_stationary_and_resume"] is True
                       else "NOT_VERIFIED" if checks["communication_outage_holds_stationary_and_resume"] == "NOT_VERIFIED"
                       else "fail"),
            "basis": "USD positions held at event-declared location until resume; resume-step movement is allowed",
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
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest or args.trajectory.parent / "manifest.json"
    receipt = args.receipt or args.trajectory.parent / "run_receipt.json"
    report = audit(args.trajectory, args.events, manifest, receipt, args.truth)
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
