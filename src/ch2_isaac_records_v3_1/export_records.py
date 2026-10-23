#!/usr/bin/env python3
"""Export one completed public capture as config.json plus records.jsonl."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from public_trace import PublicTraceReader, PublicTraceWriter


def _scene_config(static: dict, receipt: dict | None) -> dict:
    receipt = receipt or {}
    aircraft = static.get("aircraft", static.get("actor_manifest", []))
    platforms = []
    for row in aircraft:
        aid = row.get("id", row.get("aircraft_id"))
        kind = row.get("type", row.get("kind", row.get("aircraft_type")))
        platforms.append({"id": aid, "type": str(kind).upper() if kind is not None else None,
                          "home_zone": row.get("home_zone"),
                          "capabilities": row.get("capabilities"),
                          "max_speed_mps": row.get("max_speed_mps"),
                          "energy_capacity_wh": row.get("energy_capacity_wh")})
    geometry = static.get("public_task_geometry", {})
    tasks = []
    for task_id, row in sorted(geometry.items()):
        xy = row.get("public_xy_km", [None, None])
        tasks.append({"id": task_id, "zone": row.get("zone"),
                      "allowed_platform_types": [str(row.get("allowed_kind")).upper() if row.get("allowed_kind") else None],
                      "position": ([xy[0] * 1000.0, xy[1] * 1000.0]
                                   if xy[0] is not None and xy[1] is not None else None),
                      "priority": row.get("priority"), "deadline_s": row.get("deadline_s"),
                      "initial_assignment": row.get("initial_assignment", receipt.get("initial_assignment", {}).get(task_id)),
                      "public_bounds_km": row.get("public_bounds_km"),
                      "cell": row.get("cell"), "kind": row.get("kind"),
                      "sweep_waypoints_km": row.get("sweep_waypoints_km")})
    return {
        "schema": "scene_config_v1",
        "run_id": receipt.get("run_id", static.get("run_id")),
        "scene_id": static.get("scene_id", receipt.get("run_id", static.get("run_id"))),
        "split": (receipt.get("formal_identity") or {}).get("split", "unknown"),
        "seed": receipt.get("seed", static.get("seed")),
        "version": static.get("version", "legacy-converted"),
        "simulator": static.get("simulator"),
        "coordinate_units": {"position": "m", "time": "s", "energy": "Wh"},
        "map": {"size_km": [static.get("map_size_km")] * 2,
                "roi_center_km": static.get("roi_center_km"),
                "roi_km": [static.get("roi_width_km")] * 2,
                "zones": static.get("zones"),
                "zone_bounds_km": static.get("zone_bounds_km")},
        "platforms": platforms,
        "tasks": tasks,
        "disturbance": {"type": static.get("condition", "legacy"), "parameters": {}},
        "deadline_supported": False,
        "energy_model": {"source": "analytical_model"},
    }


def export_capture(capture_dir: Path, output_dir: Path, receipt_path: Path | None = None):
    capture_dir = capture_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if (capture_dir / "config.json").is_file() and (capture_dir / "records.jsonl").is_file():
        shutil.copyfile(capture_dir / "config.json", output_dir / "config.json")
        shutil.copyfile(capture_dir / "records.jsonl", output_dir / "records.jsonl")
        if (capture_dir / "manifest.json").is_file():
            shutil.copyfile(capture_dir / "manifest.json", output_dir / "manifest.json")
        if (capture_dir / "labels").is_dir():
            shutil.copytree(capture_dir / "labels", output_dir / "labels")
        rows = [json.loads(line) for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        return {"config": str(output_dir / "config.json"),
                "records": str(output_dir / "records.jsonl"),
                "sample_count": sum(row.get("kind") == "sample" for row in rows),
                "event_count": sum(row.get("kind") == "event" for row in rows),
                "decision_count": sum(row.get("kind") == "sample" and row.get("action") is not None for row in rows),
                "t_end": next((row.get("t_end") for row in reversed(rows) if row.get("kind") == "summary"), None)}
    static = json.loads((capture_dir / "static_config.json").read_text(encoding="utf-8"))
    receipt = (json.loads(receipt_path.read_text(encoding="utf-8"))
               if receipt_path is not None and receipt_path.is_file() else {})
    config = _scene_config(static, receipt)
    writer = PublicTraceWriter(output_dir, config)
    last_t = None
    for item in PublicTraceReader(capture_dir).iter_records():
        tick = {**item["projection"], "events": item["events"], "decisions": item["decisions"]}
        writer.append(tick, hidden_audit={})
        last_t = tick["sim_time_s"]
    metrics = receipt.get("branch_metrics", {}).get("representative", {})
    total = metrics.get("public_task_denominator")
    completed = metrics.get("cumulative_completed_public_tasks")
    energy_kwh = metrics.get("analytical_energy_kwh")
    writer.set_summary({
        "t_end": last_t, "status": receipt.get("status", "unknown"),
        "completed_tasks": completed, "total_tasks": total,
        "completion_rate": completed / total if total and completed is not None else None,
        "uncompleted_tasks": total - completed if total is not None and completed is not None else None,
        "total_energy_wh": energy_kwh * 1000.0 if energy_kwh is not None else None,
        "energy_source": "analytical_model" if energy_kwh is not None else "unknown",
        "total_delay_s": None, "total_delay_reason": "deadline_supported_false",
        "safety_events": metrics.get("violations"), "planner_calls": metrics.get("planner_calls"),
        "planner_cpu_s": None, "planner_cpu_reason": "not_available_in_legacy_receipt",
        "messages": metrics.get("command_attempts"), "task_switches": metrics.get("switches"),
        "waiting_s": None, "waiting_reason": "not_available_in_legacy_receipt",
    })
    storage = writer.file_info()
    labels_dir = output_dir / "labels"
    (labels_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (labels_dir / "pairs").mkdir(parents=True, exist_ok=True)
    index = {"schema": "isaac_roi_state_sampling_v1", "status": "unavailable",
             "selected_count": 0, "states": [],
             "reason": "legacy_public_capture_has_no_private_snapshots"}
    index_path = labels_dir / "state_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": "scene_manifest_v1", "source_format": "legacy_ticks",
                "source_capture": str(capture_dir),
                "scene_files": {"config": "config.json", "records": "records.jsonl",
                                "manifest": "manifest.json", "labels": "labels"},
                "public_capture_storage": storage,
                "public_config_sha256": storage["config.json"]["sha256"],
                "public_records_sha256": storage["records.jsonl"]["sha256"],
                "labels_state_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "legacy_decision_state_phase": "as_logged_predecision_phase_unverified",
                "labels": index}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    return {"config": str(output_dir / "config.json"),
            "records": str(output_dir / "records.jsonl"),
            "sample_count": writer.sample_count, "event_count": writer.event_count,
            "decision_count": writer.decision_count, "t_end": last_t}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = export_capture(args.capture_dir, args.output_dir, args.receipt)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
