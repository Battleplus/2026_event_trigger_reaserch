#!/usr/bin/env python3
"""Aggregate independent suffix reductions into state labels and 30/30 counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(root: Path, output: Path):
    reports = []
    for path in sorted(root.rglob("reduction.json")):
        try:
            report = _read(path)
        except (OSError, json.JSONDecodeError) as exc:
            reports.append({"status": "UNKNOWN", "unknown_reason": f"unreadable:{exc}", "path": str(path)})
            continue
        report["path"] = str(path)
        reports.append(report)

    groups = {}
    for report in reports:
        state_id = report.get("state_id")
        if not state_id:
            report["valid_mask"] = False
            report["unknown_reason"] = "missing_state_id"
            continue
        groups.setdefault(str(state_id), []).append(report)

    states = []
    for state_id, items in sorted(groups.items()):
        valid = [item for item in items if item.get("status") == "PASS"
                 and item.get("valid_mask", True) is not False
                 and not item.get("forced", False)]
        seeds = {item.get("suffix_seed") for item in valid}
        reason = None
        if len(items) != 2 or len(valid) != 2 or len(seeds) != 2:
            reason = "requires_two_valid_distinct_suffixes"
        gains = [float(item["G"]) for item in valid] if reason is None else []
        if reason is None:
            mean = sum(gains) / len(gains)
            variance = sum((value - mean) ** 2 for value in gains) / (len(gains) - 1)
            states.append({"state_id": state_id, "valid_mask": True, "gain_mean": mean,
                           "gain_variance": variance, "suffix_count": len(gains),
                           "split": sorted({item.get("split") for item in valid if item.get("split") is not None}),
                           "suffix_seeds": sorted(seeds)})
        else:
            states.append({"state_id": state_id, "valid_mask": False, "unknown_reason": reason,
                           "suffix_count": len(items), "suffix_seeds": sorted(seeds, key=str)})

    ordinary = [state for state in states if state.get("valid_mask") is True]
    positive = [state for state in ordinary if state["gain_mean"] > 0]
    nonpositive = [state for state in ordinary if state["gain_mean"] <= 0]
    split_counts = {"positive": {"train": 0, "validation": 0},
                    "nonpositive": {"train": 0, "validation": 0}}
    for state in positive + nonpositive:
        label = "positive" if state["gain_mean"] > 0 else "nonpositive"
        for split in state.get("split", []):
            if split in split_counts[label]:
                split_counts[label][split] += 1
    count_gate = len(positive) >= 30 and len(nonpositive) >= 30
    split_gate = all(split_counts[label][split] > 0
                     for label in ("positive", "nonpositive")
                     for split in ("train", "validation"))
    result = {"status": "PASS", "state_count": len(states), "valid_state_count": len(ordinary),
              "positive_count": len(positive), "nonpositive_count": len(nonpositive),
              "count_30_30_met": count_gate, "split_coverage_met": split_gate,
              "met_30_30": count_gate and split_gate, "split_counts": split_counts,
              "states": states,
              "excluded_count": len(states) - len(ordinary),
              "source_report_count": len(reports)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root, args.output)
    print(json.dumps({key: result[key] for key in ("state_count", "valid_state_count", "positive_count", "nonpositive_count", "met_30_30")}, sort_keys=True))


if __name__ == "__main__":
    main()
