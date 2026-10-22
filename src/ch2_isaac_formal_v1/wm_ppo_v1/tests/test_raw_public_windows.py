from __future__ import annotations

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from wm_ppo_v1.features import collate_graph_sequences, public_history_to_graph
from wm_ppo_v1.raw_public_windows import iter_raw_public_windows


def _row(branch: str, time_s: int, energy: float, *, hidden_marker: str = "secret",
         available_marker=..., observation_available: int = 1):
    row = {
        "schema_version": 3, "run_id": "raw-fixture", "phase": "representative",
        "branch": branch, "sim_time_s": time_s, "dt_s": 1.0, "aircraft_id": "a0",
        "aircraft_type": "uav", "zone": 0, "task_id": "task-a",
        "faulted": False, "usd_position_m": [1000 + time_s, 2000, 300],
        "velocity_mps": [1.0, 2.0, 0.0], "energy": {"remaining_wh": energy, "safety_pause": time_s == 3},
        "communication": {"safety_hold": False, "command_delivered": False},
        "target_discovered_ids": [hidden_marker], "detected_target_ids": [hidden_marker],
        "hidden_truth": {"coordinate": [999999, -999999]}, "scenario_sha256": hidden_marker,
        "public_target_observations": [{"observation_id": "delayed", "target_id": "hidden-id",
            "zone": 0, "target_xy_km": [4.0, 5.0], "observed_at_sim_time_s": 0,
            "available_at_sim_time_s": observation_available}],
    }
    if available_marker is not ...:
        row["available"] = available_marker
    return row


def _write_fixture(root: Path, *, hidden_marker: str = "secret"):
    root.mkdir(parents=True, exist_ok=True)
    trajectory = root / "trajectory.jsonl.gz"
    events = root / "events.jsonl"
    receipt_path = root / "run_receipt.json"
    manifest_path = root / "manifest.json"
    rows = [
        _row("representative", 0, 10, hidden_marker=hidden_marker),
        _row("representative", 1, 8, hidden_marker=hidden_marker),
        _row("representative", 3, 7, hidden_marker=hidden_marker, available_marker=False),
        _row("representative", 4, 6, hidden_marker=hidden_marker),
        _row("other", 4, 5, hidden_marker=hidden_marker),
        _row("other", 5, 4, hidden_marker=hidden_marker),
        _row("keep", 10, 3, hidden_marker=hidden_marker),
        _row("keep", 11, 2, hidden_marker=hidden_marker),
        _row("replan", 10, 3, hidden_marker=hidden_marker),
        _row("replan", 11, 2, hidden_marker=hidden_marker),
    ]
    with gzip.open(trajectory, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    events_rows = [
        {"branch": "representative", "sim_time_s": 0, "event": "task_activated", "task_id": "task-a"},
        {"branch": "representative", "sim_time_s": 1, "event": "search_cell_complete", "task_id": "task-a"},
        {"branch": "representative", "sim_time_s": 2, "event": "task_activated", "task_id": "future-task"},
        {"branch": "representative", "sim_time_s": 0, "available_at_sim_time_s": 2,
         "event": "energy_safety_hold", "aircraft_id": "a0"},
        {"branch": "representative", "sim_time_s": 1, "available_at_sim_time_s": 1,
         "event": "target_discovered", "target_id": "hidden-id"},
        {"branch": "other", "sim_time_s": 4, "event": "task_activated", "task_id": "task-a"},
        {"branch": "representative", "sim_time_s": 4, "event": "boundary_violation", "aircraft_id": "a0"},
        {"branch": "representative", "sim_time_s": 0, "event": "transmission_sent", "aircraft_id": "a0", "decision_id": "d1"},
        {"branch": "representative", "sim_time_s": 1, "event": "command_applied", "aircraft_id": "a0", "decision_id": "d1"},
        {"branch": "representative", "sim_time_s": 3, "event": "command_applied", "aircraft_id": "a0", "decision_id": "d1"},
        {"branch": "pair_prefix", "sim_time_s": 9, "event": "task_activated", "task_id": "prefix-active"},
        {"branch": "pair_prefix", "sim_time_s": 10, "event": "task_activated", "task_id": "prefix-done"},
        {"branch": "pair_prefix", "sim_time_s": 10, "event": "search_cell_complete", "task_id": "prefix-done"},
    ]
    events.write_text("".join(json.dumps(row) + "\n" for row in events_rows), encoding="utf-8")
    task_manifest = [
        {"task_id": "task-a", "zone": 0, "cell": 0, "allowed_kind": "uav",
         "public_bounds_km": [0.0, 0.0, 2.0, 2.0]},
        {"task_id": "future-task", "zone": 1, "cell": 0, "allowed_kind": "uav",
         "public_bounds_km": [2.0, 0.0, 4.0, 2.0]},
        {"task_id": "prefix-active", "zone": 2, "public_bounds_km": [4.0, 0.0, 6.0, 2.0]},
        {"task_id": "prefix-done", "zone": 3, "public_bounds_km": [6.0, 0.0, 8.0, 2.0]},
    ]
    receipt = {"run_id": "raw-fixture", "phase": "representative", "status": "complete",
        "actual_isaac_run": True, "actor_manifest": [{"aircraft_id": "a0"}],
        "public_task_manifest": task_manifest,
        "pair_summary": {"gain_keep_minus_replan": 987654.0},
        "pair_snapshot": {"sim_time_s": 10}}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest_path.write_text(json.dumps({"run_id": "raw-fixture", "logs": {
        "trajectory": trajectory.name, "events": events.name}}), encoding="utf-8")
    return trajectory, events, receipt_path


class RawPublicWindowTests(unittest.TestCase):
    def test_streaming_public_windows_branch_gaps_hidden_fields_and_targets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            root = Path(temp)
            trajectory, events, receipt = _write_fixture(root)
            windows = list(iter_raw_public_windows(trajectory, events, receipt, history_steps=2))
            self.assertEqual([(w.branch, w.input_time_s, w.target_time_s) for w in windows], [
                ("representative", 0, 1), ("representative", 3, 4), ("other", 4, 5),
                ("keep", 10, 11), ("replan", 10, 11)])
            first = windows[0]
            self.assertEqual(first.history[0].tasks[0]["_public_id"], "task-a")
            self.assertEqual({task["_public_id"] for task in first.history[0].tasks}, {"task-a"})
            self.assertEqual(first.target.dynamic["energy_delta"], -2.0)
            self.assertTrue(first.target.dynamic_mask["energy_delta"])
            self.assertEqual(first.target.dynamic["task_progress"], 1.0)
            self.assertTrue(first.target.dynamic_mask["task_progress"])
            self.assertTrue(first.target.dynamic_mask["delay"])
            self.assertEqual(first.target.dynamic["delay"], 1.0)
            self.assertIsNone(first.target.gain)
            risk_window = next(w for w in windows if w.input_time_s == 3)
            self.assertTrue(risk_window.target.dynamic_mask["risk"])
            self.assertEqual(risk_window.target.dynamic["risk"], 1.0)
            self.assertEqual({event["event"] for event in first.history[0].public_events},
                             {"task_activated", "transmission_sent"})
            self.assertFalse(first.history[0].observations)  # availability at t=1 is not visible at t=0
            self.assertEqual(collate_graph_sequences([first.history]).action_features[0, 0, 0], 0.0)
            # Unknown availability and explicitly unavailable are distinct input features.
            self.assertEqual(public_history_to_graph(first.history[0])[0, 8], 0.0)
            self.assertEqual(public_history_to_graph(first.history[0])[0, 27], 0.0)
            explicit_unavailable = public_history_to_graph(windows[1].history[0])[0]
            self.assertEqual(explicit_unavailable[8], 0.0)
            self.assertEqual(explicit_unavailable[27], 1.0)
            self.assertFalse(windows[1].history[0].platforms[0]["active"])  # energy safety pause is public
            self.assertIn("energy_safety_hold", {event["event"] for event in windows[1].history[0].public_events})

            paired = [w for w in windows if w.branch in {"keep", "replan"}]
            self.assertEqual(len(paired), 2)
            for window in paired:
                self.assertEqual(window.history[0].time_s, 10.0)
                tasks = {task["_public_id"]: task for task in window.history[0].tasks}
                self.assertTrue(tasks["prefix-active"]["active"])
                self.assertFalse(tasks["prefix-active"]["completed"])
                self.assertFalse(tasks["prefix-done"]["active"])
                self.assertTrue(tasks["prefix-done"]["completed"])

            changed_path, changed_events, changed_receipt = _write_fixture(root / "changed", hidden_marker="changed-secret")
            changed = list(iter_raw_public_windows(changed_path, changed_events, changed_receipt, history_steps=2))
            for left, right in zip(windows, changed):
                self.assertEqual(left.target, right.target)
                left_batch = collate_graph_sequences([left.history])
                right_batch = collate_graph_sequences([right.history])
                self.assertEqual(left_batch.event_features.tolist(), right_batch.event_features.tolist())
                self.assertEqual(left_batch.node_features.tolist(), right_batch.node_features.tolist())


if __name__ == "__main__":
    unittest.main()
