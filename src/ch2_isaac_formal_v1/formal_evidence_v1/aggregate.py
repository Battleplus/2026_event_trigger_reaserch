#!/usr/bin/env python3
"""Reduce two independently seeded suffix checks into state labels and coverage."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(root: Path, output: Path):
    root = Path(root).resolve()
    output = Path(output).resolve()
    reports = []
    unreadable = []
    for path in sorted(root.rglob("reduction.json")):
        if path.resolve() == output:
            continue
        try:
            report = _read(path)
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append({"path": str(path), "reason": f"unreadable_reduction:{exc}"})
            continue
        report["path"] = str(path.relative_to(root))
        reports.append(report)

    groups = defaultdict(list)
    orphan_rows = []
    for report in reports:
        state_id = report.get("state_id")
        if not state_id:
            orphan_rows.append({"path": report["path"], "valid_mask": False,
                                "unknown_reason": "missing_state_id"})
            continue
        groups[str(state_id)].append(report)

    states = []
    for state_id, items in sorted(groups.items()):
        reasons = []
        split_values = {item.get("split") for item in items}
        if len(split_values) > 1:
            reasons.append("cross_split_duplicate_state_id")
        if len(items) != 2:
            reasons.append("requires_exactly_two_suffix_rows")
        seeds = [item.get("suffix_seed") for item in items]
        if any(seed is None for seed in seeds) or len(set(map(str, seeds))) != 2:
            reasons.append("suffix_seeds_must_be_distinct")
        if any(item.get("status") != "PASS" or item.get("valid_mask") is not True for item in items):
            reasons.append("failed_or_unknown_suffix")
        if any(item.get("fixture_only", False) for item in items):
            reasons.append("fixture_only")
        if any(item.get("forced", False) for item in items):
            reasons.append("forced_state")
        if any(int(item.get("horizon_steps", 0) or 0) <= 0 for item in items):
            reasons.append("empty_or_missing_window")
        snapshots = {item.get("snapshot_sha256") for item in items}
        configs = {item.get("config_sha256") for item in items}
        if None in snapshots or len(snapshots) != 1:
            reasons.append("suffixes_do_not_share_snapshot")
        if None in configs or len(configs) != 1:
            reasons.append("suffixes_do_not_share_config")
        scenario_ids = {item.get("scenario_id") for item in items}
        local_ids = {item.get("local_state_id") for item in items}
        if len(scenario_ids) != 1 or None in scenario_ids or len(local_ids) != 1 or None in local_ids:
            reasons.append("scenario_or_local_state_identity_mismatch")
        split = next(iter(split_values)) if len(split_values) == 1 else None
        gains = []
        if not reasons:
            try:
                gains = [float(item["G"]) for item in items]
            except (KeyError, TypeError, ValueError):
                reasons.append("missing_or_invalid_gain")
        if reasons:
            states.append({"state_id": state_id, "scenario_id": next(iter(scenario_ids), None),
                           "local_state_id": next(iter(local_ids), None), "split": split,
                           "valid_mask": False, "unknown_reason": sorted(set(reasons)),
                           "suffix_count": len(items), "suffix_seeds": sorted(seeds, key=str),
                           "source_reports": [item["path"] for item in items]})
            continue
        mean = sum(gains) / 2.0
        variance = sum((value - mean) ** 2 for value in gains)  # sample variance for n=2
        states.append({"state_id": state_id, "scenario_id": next(iter(scenario_ids)),
                       "local_state_id": next(iter(local_ids)), "split": split,
                       "valid_mask": True, "gain_mean": mean, "gain_variance": variance,
                       "suffix_count": 2, "suffix_seeds": sorted(seeds, key=str),
                       "snapshot_sha256": next(iter(snapshots)), "config_sha256": next(iter(configs)),
                       "source_reports": [item["path"] for item in items]})

    valid = [state for state in states if state.get("valid_mask") is True]
    positive = [state for state in valid if state["gain_mean"] > 0]
    nonpositive = [state for state in valid if state["gain_mean"] <= 0]
    split_counts = {label: {split: 0 for split in ("train", "validation")}
                    for label in ("positive", "nonpositive")}
    for state in positive:
        if state["split"] in split_counts["positive"]:
            split_counts["positive"][state["split"]] += 1
    for state in nonpositive:
        if state["split"] in split_counts["nonpositive"]:
            split_counts["nonpositive"][state["split"]] += 1
    count_gate = len(positive) >= 30 and len(nonpositive) >= 30
    split_gate = all(split_counts[label][split] > 0
                     for label in ("positive", "nonpositive")
                     for split in ("train", "validation"))

    per_scenario = {}
    for scenario_id in sorted({str(state.get("scenario_id")) for state in states
                               if state.get("scenario_id") is not None}):
        scenario_states = [state for state in states if str(state.get("scenario_id")) == scenario_id]
        scenario_valid = [state for state in scenario_states if state.get("valid_mask") is True]
        scenario_positive = [state for state in scenario_valid if state["gain_mean"] > 0]
        scenario_nonpositive = [state for state in scenario_valid if state["gain_mean"] <= 0]
        per_scenario[scenario_id] = {
            "state_count": len(scenario_states),
            "valid_state_count": len(scenario_valid),
            "positive_count": len(scenario_positive),
            "nonpositive_count": len(scenario_nonpositive),
            "unknown_count": len(scenario_states) - len(scenario_valid),
            "gain_mean": (sum(state["gain_mean"] for state in scenario_valid) / len(scenario_valid)
                          if scenario_valid else None),
            "split_counts": {
                "train": sum(state.get("split") == "train" for state in scenario_valid),
                "validation": sum(state.get("split") == "validation" for state in scenario_valid),
            },
        }

    result = {
        "schema_version": 1, "aggregator": "independent_suffix_state_reduction_v1",
        "status": "PASS" if count_gate and split_gate else "INSUFFICIENT_COVERAGE",
        "state_count": len(states), "valid_state_count": len(valid),
        "positive_count": len(positive), "nonpositive_count": len(nonpositive),
        "count_30_30_met": count_gate, "split_coverage_met": split_gate,
        "met_30_30": count_gate and split_gate,
        "split_counts": split_counts,
        "per_scenario": per_scenario,
        "excluded_state_count": len(states) - len(valid),
        "orphan_reduction_count": len(orphan_rows), "unreadable_reduction_count": len(unreadable),
        "orphan_rows": orphan_rows, "unreadable_rows": unreadable,
        "states": states,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="directory containing per-suffix reduction.json files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.root, args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "state_count", "valid_state_count", "positive_count",
                       "nonpositive_count", "met_30_30")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
