#!/usr/bin/env python3
"""Independent audit of continuous abstract service evidence.

This module deliberately does not import production mission or completion code.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _distance(a, b):
    return math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def audit_service(trajectory_path, events_path, manifest_path, truth_path):
    """Return an auditable report with PASS, FAIL, NOT_VERIFIED, or NOT_RUN checks."""
    if not Path(trajectory_path).exists():
        return {"status": "NOT_RUN", "checks": {"trajectory_available": "NOT_RUN"},
                "failures": ["trajectory.jsonl is absent"], "completion_count": 0}
    manifest = _read_json(manifest_path)
    events = _read_jsonl(events_path)
    truth = _read_json(truth_path)
    trajectory = _read_jsonl(trajectory_path)
    config = manifest.get("config", {})
    duration = float(config.get("abstract_strike_service_duration_s", 20.0))
    radius = float(config.get("abstract_strike_service_radius_km", 0.3))
    targets = {str(item["target_id"]): item for item in truth.get("targets", [])}
    checks = {}
    failures = []

    if not trajectory:
        return {"status": "NOT_RUN", "checks": {"trajectory_available": "NOT_RUN"},
                "failures": ["trajectory.jsonl is absent or empty"], "completion_count": 0}

    rows = defaultdict(dict)
    actor_rows = defaultdict(list)
    for row in trajectory:
        key = (str(row.get("branch")), str(row.get("aircraft_id")))
        rows[key][int(row.get("sim_time_s", -1))] = row
        actor_rows[str(row.get("aircraft_id"))].append(row)
    starts = defaultdict(list)
    completes = defaultdict(list)
    resets = defaultdict(list)
    observations = defaultdict(list)
    for event in events:
        kind = event.get("event")
        target = event.get("target_id")
        key = (str(event.get("branch")), str(event.get("aircraft_id", "")), str(target))
        if kind == "abstract_strike_service_start":
            starts[key].append(event)
        elif kind == "abstract_strike_service_reset":
            resets[key].append(event)
        elif kind == "abstract_strike_complete":
            completes[key].append(event)
        elif kind == "public_observation" and target is not None:
            observations[(str(event.get("branch")), str(target))].append(event)

    all_completions = [event for values in completes.values() for event in values]
    per_branch_target = defaultdict(list)
    for event in all_completions:
        per_branch_target[(str(event.get("branch")), str(event.get("target_id")))].append(event)
    duplicate_targets = [key for key, values in per_branch_target.items() if len(values) > 1]
    checks["completion_unique_per_logical_target"] = "FAIL" if duplicate_targets else "PASS"
    if duplicate_targets:
        failures.append(f"branch/target completed more than once: {duplicate_targets}")

    if not all_completions:
        checks["service_completion_evidence"] = "NOT_VERIFIED"
        checks["active_service_evidence"] = "NOT_VERIFIED"
        return {"status": "NOT_VERIFIED", "checks": checks, "failures": failures,
                "completion_count": 0}

    active_missing = []
    unverifiable = []
    proven_failures = []
    for key, completions in completes.items():
        if not starts.get(key):
            active_missing.append(key)
    checks["active_service_evidence"] = "NOT_VERIFIED" if active_missing else "PASS"
    if active_missing:
        failures.append(f"missing service-start evidence for {len(active_missing)} completion(s)")
        unverifiable.append("active service start/reset evidence is absent")

    for key, completion_events in completes.items():
        branch, aid, target_id = key
        target = targets.get(target_id)
        samples_for_actor = actor_rows.get(aid, [])
        actor_types = {row.get("aircraft_type") for row in samples_for_actor if row.get("aircraft_type") is not None}
        actor_zones = {row.get("zone") for row in samples_for_actor if row.get("zone") is not None}
        actor_type = next(iter(actor_types)) if len(actor_types) == 1 else None
        actor_zone = next(iter(actor_zones)) if len(actor_zones) == 1 else None
        for complete in completion_events:
            complete_t = float(complete.get("sim_time_s", math.nan))
            starts_for_completion = [event for event in starts.get(key, [])
                                     if float(event.get("sim_time_s", math.inf)) <= complete_t]
            if not starts_for_completion or target is None:
                if target is None:
                    unverifiable.append(f"{branch}/{aid}/{target_id}: evaluator target record missing")
                continue
            start = max(starts_for_completion, key=lambda event: float(event["sim_time_s"]))
            start_t = float(start["sim_time_s"])
            reset_times = [float(event.get("sim_time_s", math.inf)) for event in resets.get(key, [])]
            if any(start_t <= reset_t < complete_t for reset_t in reset_times):
                proven_failures.append(f"{branch}/{aid}/{target_id}: service reset occurred before completion")
                continue
            sample_rows = rows.get((branch, aid), {})
            target_xy = target.get("target_xy_km")
            kind_ok = actor_type == target.get("kind") if actor_type is not None else None
            zone_ok = (actor_type != "helicopter" or actor_zone is not None and int(actor_zone) == int(target.get("zone", -2))) if actor_type is not None else None
            observation_events = observations.get((branch, target_id), [])
            available_values = [event.get("available_at_sim_time_s") for event in observation_events
                                if event.get("available_at_sim_time_s") is not None]
            available_t = min((float(value) for value in available_values), default=None)
            observation_ok = available_t is not None and available_t <= start_t
            if not actor_types or len(actor_types) > 1 or not actor_zones or len(actor_zones) > 1:
                unverifiable.append(f"{branch}/{aid}/{target_id}: actor type or zone evidence missing/inconsistent")
            if target_xy is None or not isinstance(target_xy, list) or len(target_xy) < 2:
                unverifiable.append(f"{branch}/{aid}/{target_id}: evaluator target position missing")
            if available_t is None:
                unverifiable.append(f"{branch}/{aid}/{target_id}: public observation availability evidence missing")
            elif available_t > start_t:
                proven_failures.append(f"{branch}/{aid}/{target_id}: service starts before public observation")
            eligible = True
            missing = []
            missing_fields = []
            t = int(start_t)
            while t <= int(complete_t):
                row = sample_rows.get(t)
                if row is None:
                    missing.append(t); t += 1; continue
                if row.get("dt_s") is None or float(row["dt_s"]) != 1.0:
                    missing_fields.append(f"{t}:dt_s"); t += 1; continue
                position = row.get("mission_xy_km")
                energy = row.get("energy")
                comm = row.get("communication")
                needed = [name for name, present in (("aircraft_type", row.get("aircraft_type")),
                          ("zone", row.get("zone")), ("faulted", row.get("faulted")),
                          ("mission_xy_km", position), ("energy", energy), ("communication", comm)) if present is None]
                if energy is not None and energy.get("safety_pause") is None:
                    needed.append("energy.safety_pause")
                if comm is not None and (comm.get("safety_hold") is None or comm.get("forced_outage") is None):
                    needed.append("communication hold flags")
                if needed:
                    missing_fields.extend(f"{t}:{name}" for name in needed)
                    t += 1
                    continue
                if not isinstance(target_xy, list) or len(target_xy) < 2:
                    missing_fields.append(f"{t}:evaluator target position")
                    t += 1
                    continue
                in_radius = isinstance(position, list) and len(position) >= 2 and target_xy is not None and _distance(position, target_xy) <= radius + 1e-9
                paused = bool(row["faulted"]) or bool(energy["safety_pause"]) or bool(comm["safety_hold"]) or bool(comm["forced_outage"])
                if not in_radius or paused:
                    eligible = False
                t += 1
            if complete_t - start_t < duration:
                proven_failures.append(f"{branch}/{aid}/{target_id}: completion dwell is shorter than {duration}s")
            if kind_ok is False:
                proven_failures.append(f"{branch}/{aid}/{target_id}: aircraft type mismatch")
            if zone_ok is False:
                proven_failures.append(f"{branch}/{aid}/{target_id}: helicopter home-zone violation")
            if missing:
                unverifiable.append(f"{branch}/{aid}/{target_id}: missing trajectory samples {missing[:8]}")
            if missing_fields:
                unverifiable.append(f"{branch}/{aid}/{target_id}: required row evidence missing {missing_fields[:8]}")
            if not eligible:
                proven_failures.append(f"{branch}/{aid}/{target_id}: ineligible service sample during dwell")

    failures.extend(proven_failures)
    failures.extend(unverifiable)
    checks["continuous_dwell_and_eligibility"] = "FAIL" if any(
        "dwell" in failure or "reset occurred" in failure or "ineligible service" in failure for failure in proven_failures) else (
        "NOT_VERIFIED" if any("trajectory" in failure or "required row evidence" in failure or "actor type or zone" in failure for failure in unverifiable) else "PASS")
    checks["observation_available_before_service"] = "FAIL" if any("before public observation" in failure for failure in proven_failures) else (
        "NOT_VERIFIED" if any("observation availability" in failure for failure in unverifiable) else "PASS")
    checks["type_and_home_zone"] = "FAIL" if any("type mismatch" in failure or "home-zone" in failure for failure in proven_failures) else (
        "NOT_VERIFIED" if any("actor type or zone" in failure for failure in unverifiable) else "PASS")
    checks["evidence_completeness"] = "NOT_VERIFIED" if unverifiable else "PASS"
    if active_missing:
        checks["active_service_evidence"] = "NOT_VERIFIED"
    if duplicate_targets:
        checks["completion_unique_per_logical_target"] = "FAIL"
    status = "FAIL" if any(value == "FAIL" for value in checks.values()) else (
        "NOT_VERIFIED" if any(value == "NOT_VERIFIED" for value in checks.values()) else "PASS")
    return {"status": status, "checks": checks, "failures": failures,
            "completion_count": len(all_completions), "duplicate_targets": duplicate_targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluator-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_service(args.trajectory, args.events, args.manifest, args.evaluator_truth)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"PASS", "NOT_VERIFIED", "NOT_RUN"} else 2)


if __name__ == "__main__":
    main()
