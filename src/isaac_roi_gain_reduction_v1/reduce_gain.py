#!/usr/bin/env python3
"""Offline scalar reduction for one paired suffix; no production imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find(root, name):
    direct = root / name
    if direct.exists():
        return direct
    for branch in ("keep", "replan"):
        candidate = root / branch / name
        if candidate.exists():
            return candidate
    return None


def _branch_file(root, branch, name):
    path = root / branch / name
    return path if path.exists() else root / f"{branch}_{name}"


def _energy_wh(rows):
    total = 0.0
    for row in rows:
        current = row.get("usd_position_m")
        previous = row.get("previous_usd_position_m")
        energy = row.get("energy") or {}
        rate = energy.get("effective_wh_per_km")
        if not (isinstance(current, list) and isinstance(previous, list) and len(current) >= 2
                and len(previous) >= 2 and rate is not None):
            raise ValueError("missing USD positions or effective_wh_per_km")
        total += math.dist(current[:2], previous[:2]) * float(rate) / 1000.0
    return total


def _audit_bound(report_path, files):
    if not report_path or not report_path.exists():
        return False
    report = _json(report_path)
    binding = report.get("input_sha256")
    return str(report.get("status", "")).lower() == "pass" and isinstance(binding, dict) and all(
        binding.get(name) == _sha(path) for name, path in files.items())


def _events(events, branch, kind):
    return [event for event in events if event.get("branch") == branch and event.get("event") == kind]


def _initial_ids(receipt, key):
    for container in (receipt.get("suffix", {}), receipt.get("pair_snapshot", {}), receipt):
        value = container.get(key)
        if isinstance(value, list):
            return set(map(str, value))
    return None


def _branch_score(branch, rows, events, truth_count, initial_discovered, initial_serviced,
                  search_strike, audit_ok, service_ok, expected_rows):
    reasons = []
    if not audit_ok:
        reasons.append("discovery audit is not PASS with input hash binding")
    if search_strike and not service_ok:
        reasons.append("service audit is not PASS with input hash binding")
    if len(rows) != expected_rows:
        reasons.append(f"row coverage {len(rows)} != expected {expected_rows}")
    if not rows:
        reasons.append("empty trajectory")
    try:
        energy_wh = _energy_wh(rows)
    except ValueError as exc:
        energy_wh = None
        reasons.append(str(exc))
    # Execution quality uses the physical detection event.  Public observation
    # availability is an input-timing field and may lag detection by a second;
    # counting it here would silently turn sensor latency into task failure.
    discovered = initial_discovered | {str(event["target_id"]) for event in _events(events, branch, "target_discovered") if event.get("target_id") is not None}
    serviced = initial_serviced | {str(event["target_id"]) for event in _events(events, branch, "abstract_strike_complete") if event.get("target_id") is not None}
    if truth_count <= 0:
        reasons.append("evaluator truth target count is missing or zero")
    q_terminal = None if truth_count <= 0 else ((0.5 * (1 - len(discovered) / truth_count) + 0.5 * (1 - len(serviced) / truth_count))
                                               if search_strike else (1 - len(discovered) / truth_count))
    times = sorted({int(row.get("sim_time_s")) for row in rows})
    unresolved = []
    for time_s in times:
        d = len(initial_discovered | {str(event["target_id"]) for event in _events(events, branch, "target_discovered")
                                      if event.get("target_id") is not None and float(event.get("sim_time_s", math.inf)) <= time_s})
        s = len(initial_serviced | {str(event["target_id"]) for event in _events(events, branch, "abstract_strike_complete")
                                    if event.get("target_id") is not None and float(event.get("sim_time_s", math.inf)) <= time_s})
        unresolved.append((0.5 * (1 - d / truth_count) + 0.5 * (1 - s / truth_count)) if search_strike else (1 - d / truth_count))
    L = (sum(unresolved) / len(unresolved)) if unresolved and truth_count > 0 else None
    violations = sum(event.get("event") in {"safety_violation", "boundary_violation", "constraint_violation"}
                     for event in events if event.get("branch") == branch)
    if any(reason for reason in reasons):
        return {"valid": False, "reasons": reasons, "energy_wh": energy_wh, "q_terminal": q_terminal, "L": L,
                "discovered_count": len(discovered), "serviced_count": len(serviced), "violations": violations}
    j_exec = q_terminal + 0.2 * L + 0.01 * ((energy_wh / 1000.0) / 1000.0) + 0.25 * violations
    return {"valid": True, "reasons": [], "energy_wh": energy_wh, "q_terminal": q_terminal, "L": L,
            "j_exec": j_exec, "discovered_count": len(discovered), "serviced_count": len(serviced),
            "violations": violations}


def reduce_suffix(suffix_dir, output):
    root = Path(suffix_dir)
    receipt_path = _find(root, "run_receipt.json") or _find(root, "receipt.json")
    manifest_path = _find(root, "manifest.json")
    truth_path = _find(root, "evaluator_truth.json")
    if not all((receipt_path, manifest_path, truth_path)):
        report = {"status": "UNKNOWN", "reason": "missing receipt, manifest, or evaluator truth"}
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    receipt = _json(receipt_path); manifest = _json(manifest_path); truth = _json(truth_path)
    reasons = []
    if receipt.get("status") != "complete" or receipt.get("actual_isaac_run") is not True:
        reasons.append("receipt is not complete actual Isaac")
    pair = receipt.get("pair_snapshot", {})
    if not (pair.get("restore_full_state_match") is True and pair.get("state_sha256") == pair.get("restored_state_sha256")):
        reasons.append("snapshot restore hash/match is missing or mismatched")
    initial_discovered = _initial_ids(receipt, "initial_discovered_target_ids")
    initial_serviced = _initial_ids(receipt, "initial_serviced_target_ids")
    if initial_discovered is None or initial_serviced is None:
        reasons.append("initial snapshot discovered/serviced target IDs are missing")
    config = manifest.get("config", {})
    search_strike = config.get("mode") == "search_strike"
    expected_h = min(1800, 5400 - (int(pair.get("sim_time_s", 1200)) - 1))
    actor_count = len(manifest.get("actor_manifest", []))
    expected_rows = expected_h * actor_count
    files = {}
    branch_data = {}
    for branch in ("keep", "replan"):
        trajectory = _branch_file(root, branch, "trajectory.jsonl")
        events_path = _branch_file(root, branch, "events.jsonl")
        audit_path = _branch_file(root, branch, "audit_report.json")
        service_path = _branch_file(root, branch, "service_audit.json")
        if not trajectory.exists() or not events_path.exists():
            reasons.append(f"missing {branch} trajectory/events")
            continue
        rows = _jsonl(trajectory); events = _jsonl(events_path)
        files = {"trajectory.jsonl": trajectory, "events.jsonl": events_path, "evaluator_truth.json": truth_path}
        discovery_ok = _audit_bound(audit_path, files)
        service_ok = _audit_bound(service_path, files) if search_strike else True
        branch_data[branch] = {"rows": rows, "events": events,
                               "audit_ok": discovery_ok, "service_ok": service_ok,
                               "metrics": _branch_score(branch, rows, events, len(truth.get("targets", [])),
                                                         initial_discovered or set(), initial_serviced or set(),
                                                         search_strike, discovery_ok, service_ok, expected_rows)}
    if set(branch_data) != {"keep", "replan"}:
        reasons.append("both keep and replan suffixes are required")
    if reasons or not all(item["metrics"]["valid"] for item in branch_data.values()):
        reasons.extend(reason for item in branch_data.values() for reason in item["metrics"]["reasons"])
        report = {"status": "UNKNOWN", "reasons": sorted(set(reasons)), "branches": {k: v["metrics"] for k, v in branch_data.items()}}
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    keep = branch_data["keep"]; replan = branch_data["replan"]
    def count(branch, kinds):
        return sum(event.get("event") in kinds for event in branch["events"])
    messages_increment = count(replan, {"transmission_sent", "command_dropped"}) - count(keep, {"transmission_sent", "command_dropped"})
    switches_increment = count(replan, {"controller_switch"}) - count(keep, {"controller_switch"})
    calls_increment = count(replan, {"candidate_generation"}) - count(keep, {"candidate_generation"})
    cpu_increment = (sum(float(event.get("timing", {}).get("cpu_duration_ns", 0)) for event in replan["events"] if event.get("event") == "candidate_generation")
                     - sum(float(event.get("timing", {}).get("cpu_duration_ns", 0)) for event in keep["events"] if event.get("event") == "candidate_generation")) / 1e9
    changed = set()
    replan_decision_id = f"replan-decision-{pair.get('sim_time_s')}"
    for event in replan["events"]:
        if (event.get("event") == "donor_support_transfer"
                and event.get("decision_id") == replan_decision_id):
            changed.update(filter(None, (event.get("from_aircraft_id"), event.get("to_aircraft_id"))))
    c_increment = 0.01 * (messages_increment / 1000) + 0.05 * (switches_increment / 100) + 0.05 * (calls_increment / 100) + 0.01 * (len(changed) / 36) + 0.01 * cpu_increment
    gain = keep["metrics"]["j_exec"] - replan["metrics"]["j_exec"] - c_increment
    suffix_meta = receipt.get("suffix", {})
    report = {"status": "PASS", "fixture_only": bool(manifest.get("fixture_only", False)),
              "state_id": suffix_meta.get("state_id"),
              "suffix_seed": suffix_meta.get("seed"),
              "forced": bool(suffix_meta.get("audit_interface", {}).get("forced", False)),
              "J_exec_keep": keep["metrics"]["j_exec"], "J_exec_replan": replan["metrics"]["j_exec"],
              "C_increment": c_increment, "G": gain,
              "incremental": {"messages": messages_increment, "switches": switches_increment,
                               "planner_calls": calls_increment, "changed_actors": len(changed),
                               "planning_cpu_seconds": cpu_increment},
              "branches": {k: v["metrics"] for k, v in branch_data.items()}}
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--suffix-dir", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); report = reduce_suffix(args.suffix_dir, args.output); print(json.dumps(report, sort_keys=True)); raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
