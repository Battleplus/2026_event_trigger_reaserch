#!/usr/bin/env python3
"""Render the first committed V33 selected cross-region action with public trace positions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGION = re.compile(r"(?:^|-)r(\d{2})(?:-|$)")
OBJECT_TO_CONTRACT_REGION = {f"{i:02d}": f"r{i // 3}{i % 3}" for i in range(9)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def region(value: Any) -> str | None:
    match = REGION.search(str(value))
    return match.group(1) if match else None


def is_cross(action: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for op in action.get("queue_operations", []):
        tr, pr = region(op.get("task_id")), region(op.get("to_platform"))
        donor = region(op.get("from_platform"))
        cross_home = tr is not None and pr is not None and tr != pr
        cross_transfer = donor is not None and pr is not None and donor != pr
        if cross_home or cross_transfer:
            found.append({"op": op.get("op"), "task_id": op.get("task_id"), "task_region": tr,
                          "task_contract_region": OBJECT_TO_CONTRACT_REGION.get(tr),
                          "platform_id": op.get("to_platform"), "platform_home_region": pr,
                          "platform_home_contract_region": OBJECT_TO_CONTRACT_REGION.get(pr),
                          "from_platform": op.get("from_platform"), "donor_region": donor,
                          "donor_contract_region": OBJECT_TO_CONTRACT_REGION.get(donor),
                          "cross_home_region": cross_home, "cross_platform_transfer": cross_transfer})
    for dispatch in action.get("dispatches", []):
        tr, pr = region(dispatch.get("task_id")), region(dispatch.get("platform_id"))
        if tr is not None and pr is not None and tr != pr:
            found.append({"op": "direct_dispatch", "task_id": dispatch.get("task_id"), "task_region": tr,
                          "task_contract_region": OBJECT_TO_CONTRACT_REGION.get(tr),
                          "platform_id": dispatch.get("platform_id"), "platform_home_region": pr,
                          "platform_home_contract_region": OBJECT_TO_CONTRACT_REGION.get(pr),
                          "from_platform": None, "donor_region": None, "donor_contract_region": None,
                          "cross_home_region": True, "cross_platform_transfer": False})
    return found


def committed_rows(run_dir: Path):
    sys.path.insert(0, str(ROOT / "package" / "civilian"))
    import v3_shards

    markers = []
    for marker in run_dir.rglob("*.commit.json"):
        commit = json.loads(marker.read_text(encoding="utf-8"))
        markers.append((str(commit.get("scenario_id", "")), marker))
    for scenario_key, marker in sorted(markers, key=lambda item: item[0]):
        commit = v3_shards.verify_scenario_shard(marker)
        shard = marker.with_name(str(commit["shard_file"]))
        if sha256(shard) != commit.get("compressed_sha256"):
            raise ValueError(f"compressed hash mismatch: {shard}")
        header = None
        rows = []
        with gzip.open(shard, "rt", encoding="ascii") as stream:
            for line in stream:
                if not line.strip():
                    continue
                wrapped = json.loads(line)
                if wrapped.get("schema") == "gain-gate-v33-shard-header/1":
                    if header is not None:
                        raise ValueError(f"multiple headers: {shard}")
                    header = wrapped
                elif wrapped.get("schema") == "gain-gate-v33-shard-row/1":
                    rows.append(wrapped["training_row"])
        if header is None or str(header.get("scenario", {}).get("parent_id")) != str(commit.get("scenario_id")):
            raise ValueError(f"header/commit scenario mismatch: {shard}")
        for row in sorted(rows, key=lambda item: (int(item.get("time", 0)), str(item.get("state_id", "")))):
            yield str(commit["scenario_id"]), int(row.get("time", 0)), str(row.get("state_id", "")), shard, commit, header, row


def trace_nodes(pair: dict[str, Any], platform_id: str) -> list[dict[str, Any]]:
    trace = pair.get("replan", {}).get("trace", {})
    nodes = []

    def add(time_value: Any, platforms: Any, source: str):
        for platform in platforms or []:
            pos = platform.get("position")
            if platform.get("id") == platform_id and isinstance(pos, list) and len(pos) >= 2:
                point = {"time": time_value, "x": float(pos[0]), "y": float(pos[1]), "source": source}
                if not nodes or (nodes[-1]["x"], nodes[-1]["y"], nodes[-1]["time"]) != (point["x"], point["y"], point["time"]):
                    nodes.append(point)

    start = trace.get("start_truth", {})
    add(start.get("time", trace.get("start_time")), start.get("platforms"), "truth_start_endpoint")
    for step in trace.get("steps", []):
        obs = step.get("next_public_observation", {})
        add(obs.get("time"), obs.get("platforms"), "public_observation")
    end = trace.get("raw_truth", {})
    add(end.get("time", trace.get("end_time")), end.get("platforms"), "truth_end_endpoint")
    return nodes


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {"action_type": action.get("action_type"), "scope": action.get("scope"), "noop": action.get("noop"),
            "queue_operations": [{key: op.get(key) for key in ("op", "task_id", "from_platform", "to_platform")}
                                 for op in action.get("queue_operations", [])],
            "dispatches": [{key: dispatch.get(key) for key in ("task_id", "platform_id")}
                           for dispatch in action.get("dispatches", [])]}


def action_hash(action: dict[str, Any]) -> str:
    payload = json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compact_data(data: dict[str, Any]) -> dict[str, Any]:
    data["selection_basis"] = "ascending selected-state public_cost_estimate; ties by SHA-256 of canonical full action JSON"
    route = data.get("route", {})
    route["task_contract_region"] = OBJECT_TO_CONTRACT_REGION.get(route.get("task_region"))
    route["platform_home_contract_region"] = OBJECT_TO_CONTRACT_REGION.get(route.get("platform_home_region"))
    original = data["selected_action"]
    if "full_action_sha256" not in original:
        data["selected_action"] = {"action": compact_action(original), "full_action_sha256": action_hash(original)}
    for candidate in data.get("candidate_order", []):
        if "full_action_sha256" in candidate:
            continue
        original = candidate.pop("action", {})
        candidate["full_action_sha256"] = action_hash(original)
        candidate["action"] = compact_action(original)
    source = data.get("source", {})
    header = source.get("trajectory_header", {})
    source["trajectory_header"] = {key: header.get(key) for key in
        ("trajectory_source", "ordinary_candidate_count", "selected_state_count", "elapsed_ticks", "terminated", "truncated")}
    trace = data.get("trace", {})
    nodes = trace.get("public_position_nodes", [])
    existing_start = trace.get("truth_start_endpoint")
    if nodes and not any(node.get("source") for node in nodes):
        trace["truth_start_endpoint"] = {**nodes[0], "source": "truth_start_endpoint"}
        trace["public_position_nodes"] = [{**node, "source": "public_observation"} for node in nodes[1:]]
    else:
        trace["truth_start_endpoint"] = next((node for node in nodes if node.get("source") == "truth_start_endpoint"), existing_start)
        trace["public_position_nodes"] = [node for node in nodes if node.get("source") == "public_observation"]
    trace["node_count"] = len(trace["public_position_nodes"])
    trace["sampling"] = "all public-observation position nodes; consecutive duplicate records removed"
    return data


def choose(run_dir: Path) -> dict[str, Any]:
    for scenario_id, time_value, state_id, shard, commit, header, row in committed_rows(run_dir):
        action = row.get("replan_action", {})
        routes = is_cross(action)
        if not routes:
            continue
        label = row.get("label", {})
        pairs = label.get("pairs", [])
        for route in routes:
            for pair in sorted(pairs, key=lambda x: (int(x.get("time", 0)), str(x.get("pair_sha256", "")))):
                nodes = trace_nodes(pair, route["platform_id"])
                if sum(node.get("source") == "public_observation" for node in nodes) < 2:
                    continue
                receipt = row.get("candidate_receipt", {})
                candidates = []
                for candidate in receipt.get("candidates", []):
                    ca = candidate.get("full_environment_action", {})
                    candidates.append({"public_cost_estimate": candidate.get("public_cost_estimate"),
                                       "candidate_key": candidate.get("candidate_key"),
                                       "cross_region": bool(is_cross(ca)), "action": ca})
                candidates.sort(key=lambda c: (float("inf") if c["public_cost_estimate"] is None else float(c["public_cost_estimate"]),
                                               action_hash(c["action"])))
                valid = label.get("valid_mask") is True and pair.get("valid_mask") is True
                forced = row.get("forced") is True or label.get("forced") is True or pair.get("forced") is True
                return {
                    "schema": "v33-support-trajectory/1", "scenario_id": scenario_id, "state_id": state_id,
                    "split": row.get("split"), "time": time_value, "event_type": row.get("event_type"),
                    "selected_action": action, "route": route, "selected_public_cost_estimate": receipt.get("selected_public_cost_estimate"),
                    "selection_basis": "ascending selected-state candidate public_cost_estimate; ties by canonical sorted-key action JSON",
                    "candidate_order": candidates, "forced": forced,
                    "gain_mask": {"label_valid_mask": label.get("valid_mask"), "pair_valid_mask": pair.get("valid_mask"),
                                  "effective_gain_usable": bool(valid and not forced), "gain_mean": label.get("gain_mean"),
                                  "pair_gain": pair.get("gain"), "unknown_reason": label.get("unknown_reason")},
                    "trace": {"arm": "replan", "pair_sha256": pair.get("pair_sha256"), "platform_id": route["platform_id"],
                              "public_position_nodes": nodes, "node_count": len(nodes), "total_trace_steps": len(pair["replan"]["trace"].get("steps", [])),
                              "sampling": "all available public position observations for the selected platform; duplicate consecutive (time,x,y) values removed"},
                    "source": {"run_id": commit.get("run_id"), "shard_file": shard.name,
                               "relative_shard_path": str(shard.relative_to(ROOT)),
                               "compressed_sha256": commit.get("compressed_sha256"), "commit_marker": str(shard.with_suffix("").with_suffix(".commit.json").relative_to(ROOT)),
                               "commit_marker_sha256": sha256(shard.with_suffix("").with_suffix(".commit.json")),
                    "scenario_content_sha256": row.get("scenario_content_sha256"),
                    "trajectory_source": row.get("trajectory_source"),
                               "trajectory_header": {key: header.get("trajectory", {}).get(key) for key in
                                  ("trajectory_source", "ordinary_candidate_count", "selected_state_count", "elapsed_ticks",
                                   "terminated", "truncated")}}
                }
    raise ValueError("No committed selected cross-region action with at least two public trajectory position nodes found")


def render(data: dict[str, Any], png: Path, svg: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    nodes = data["trace"]["public_position_nodes"]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = ["#e9f1f2", "#f3eee2", "#e8eee5", "#f0e8e4", "#edf0e5", "#f4f0e4", "#e6eef0", "#f1e9e6", "#e8ede8"]
    for index in range(9):
        row, col = divmod(index, 3)
        boundaries = (0, 33, 66, 100)
        ax.add_patch(Rectangle((boundaries[col], boundaries[row]), boundaries[col + 1] - boundaries[col], boundaries[row + 1] - boundaries[row],
                               facecolor=colors[index], edgecolor="#6b7778", linewidth=1.2))
        ordinal = f"r{index:02d}"
        contract = OBJECT_TO_CONTRACT_REGION[f"{index:02d}"]
        ax.text((boundaries[col] + boundaries[col + 1]) / 2, (boundaries[row] + boundaries[row + 1]) / 2,
                f"{contract}\nobject {ordinal}", color="#687476", fontsize=10,
                ha="center", va="center", alpha=.9)
    xs, ys = [n["x"] for n in nodes], [n["y"] for n in nodes]
    if xs:
        ax.plot(xs, ys, color="#bb4b38", linewidth=2.4, marker="o", markersize=4, zorder=5,
                label=f"{data['trace']['platform_id']} public observations")
        ax.scatter(xs[-1], ys[-1], marker="*", s=120, color="#bb4b38", edgecolor="white", linewidth=.5, zorder=7, label="last public observation")
    start = data["trace"].get("truth_start_endpoint")
    if start:
        ax.scatter(start["x"], start["y"], marker="s", s=65, facecolor="white", edgecolor="#23818a", linewidth=1.5,
                   zorder=6, label="truth start endpoint")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="x (map units)", ylabel="y (map units)",
           title="Observed platform positions during a selected cross-region action")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 33, 66, 100], labels=["0", "33", "66", "100"])
    ax.set_yticks([0, 33, 66, 100], labels=["0", "33", "66", "100"])
    route = data["route"]
    mask = data["gain_mask"]
    forced_text = "FORCED: gain unavailable" if data["forced"] else f"forced=false; effective gain mask={mask['effective_gain_usable']}"
    start_note = "; open square is truth start endpoint" if data["trace"].get("truth_start_endpoint") else ""
    fig.text(.5, -.01,
             f"{data['scenario_id']} | {data['state_id']} | {route['platform_id']} home object {route['platform_home_region']} ({route['platform_home_contract_region']}) -> task {route['task_id']} object {route['task_region']} ({route['task_contract_region']})\n"
             f"Ascending public cost (selected={data['selected_public_cost_estimate']:.3f}; canonical action key breaks ties); {forced_text}. No rescue completion or benefit claim.\n"
             f"{sum(n.get('source') == 'public_observation' for n in nodes)} public observation nodes{start_note}. Source shard SHA-256: {data['source']['compressed_sha256']}",
             ha="center", va="top", fontsize=8, color="#263436", wrap=True)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs-v33" / "v33-collect-resume-20261020T075751Z-cf8baae09a4a")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "v33-support-trajectory")
    parser.add_argument("--from-data", action="store_true", help="rebuild existing report from data.json without reading shards")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png, svg = args.output_dir / "trajectory.png", args.output_dir / "trajectory.svg"
    if args.from_data:
        data = compact_data(json.loads((args.output_dir / "data.json").read_text(encoding="utf-8")))
    else:
        data = compact_data(choose(args.run_dir.resolve()))
    render(data, png, svg)
    data["artifacts"] = {name: {"file": name, "sha256": sha256(args.output_dir / name)} for name in ("trajectory.png", "trajectory.svg")}
    (args.output_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_id": data["scenario_id"], "state_id": data["state_id"], "forced": data["forced"],
                      "gain_mask": data["gain_mask"], "source": data["source"], "artifacts": data["artifacts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
