from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wm_ppo_v1.decision_windows import iter_decision_windows
from wm_ppo_v1.features import EVENT_DIM, EVENT_NAMES, EVENT_SCHEMA_VERSION, collate_graph_sequences


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _index_sidecar(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "  state_index.json\n", encoding="ascii")


def _fixture(root: Path, *, bad_projection_hash: bool = False, wrong_phase: bool = False,
             bad_backend_time: bool = False, include_decision_tick: bool = False):
    scene = root / "scenes" / "scene-0"
    sampling = scene / "representative" / "state_sampling"
    projection = {
        "sim_time_s": 3,
        "aircraft": {"a0": {
            "kind": "uav", "home_zone": 0, "available": True, "faulted": False,
            "safety_paused": False, "communication_held": False,
            "position_m": [9000.0, 2000.0, 300.0], "velocity_mps": [1.0, 2.0, 0.0],
            "available_energy_wh": 90.0, "active_task_ids": [], "queue_task_ids": [],
        }},
        "public_observations": [],
    }
    public_bytes = (json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    public_path = sampling / "state_00_000003.public.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public_bytes)
    entry = {
        "sim_time_s": 3, "decision_time_s": 3, "backend_time_s": 1.0 if bad_backend_time else 2.0,
        "continuation_phase": "wrong_phase" if wrong_phase else
            "post_public_updates_pre_candidate_generation_pre_step",
        "decision_sequence": 9, "private_file": "private_states/state_00_000003.pickle",
        "public_file": public_path.name,
        "public_sha256": "0" * 64 if bad_projection_hash else hashlib.sha256(public_bytes).hexdigest(),
        "ordinary_events": ["task_activated", "public_observation", "search_cell_complete", "energy_anomaly"],
    }
    index_path = sampling / "state_index.json"
    _write_json(index_path, {"schema": "isaac_roi_state_sampling_v1", "states": [entry]})
    _index_sidecar(index_path)

    trajectory = scene / "representative" / "trajectory.jsonl.gz"
    rows = []
    for tick, x, energy in ((1, 1000.0, 100.0), (2, 2000.0, 95.0)):
        rows.append({"branch": "representative", "sim_time_s": tick, "aircraft_id": "a0",
            "aircraft_type": "uav", "zone": 0, "task_id": None, "faulted": False,
            "usd_position_m": [x, 2000.0, 300.0], "velocity_mps": [1.0, 2.0, 0.0],
            "energy": {"remaining_wh": energy, "safety_pause": False},
            "communication": {"safety_hold": False}, "public_target_observations": []})
    if include_decision_tick:
        rows.append({**rows[-1], "sim_time_s": 3, "usd_position_m": [250000.0, 0.0, 0.0],
                     "energy": {"remaining_wh": 1.0, "safety_pause": True}})
    with gzip.open(trajectory, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    events = scene / "representative" / "events.jsonl"
    events.write_text("", encoding="utf-8")
    receipt = {
        "run_id": "run-0", "status": "complete", "actual_isaac_run": True,
        "actor_manifest": [{"aircraft_id": "a0"}], "events": events.name,
        "trajectory": trajectory.name, "logs": {"events": events.name, "trajectory": trajectory.name},
    }
    _write_json(scene / "representative" / "run_receipt.json", receipt)
    _write_json(scene / "representative" / "manifest.json", {
        "logs": {"events": events.name, "trajectory": trajectory.name}})

    scenes = [{"run_id": f"run-{i}", "scenario_id": "scenario-0" if i == 0 else f"scenario-{i}",
               "split": "train", "output_path": f"scenes/scene-{i}", "execution": "COMPLETE",
               "representative": {"status": "complete"}} for i in range(168)]
    _write_json(root / "batch_receipt.json", {
        "execution": "COMPLETE", "completed_at_utc": "synthetic", "scene_count": 168,
        "scenes": scenes, "end_reduction": {"aggregate_output": "aggregate.json"},
    })
    _write_json(root / "aggregate.json", {
        "status": "PASS", "state_count": 60, "valid_state_count": 60,
        "positive_count": 30, "nonpositive_count": 30, "count_30_30_met": True,
        "split_coverage_met": True, "met_30_30": True, "excluded_state_count": 0,
        "states": [{"state_id": "scenario-0::state_00_000003", "scenario_id": "scenario-0",
            "local_state_id": "state_00_000003", "split": "train", "valid_mask": True,
            "gain_mean": 7.25, "source_reports": [
                "scene-0/state_00_000003/suffix_730201/reduction.json",
                "scene-0/state_00_000003/suffix_730202/reduction.json"]}],
    })
    return trajectory


class DecisionWindowTests(unittest.TestCase):
    def test_gain_binds_only_hashed_predecision_projection_and_prior_raw_history(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            root = Path(temp)
            _fixture(root, include_decision_tick=False)
            windows = list(iter_decision_windows(root, history_steps=4))
            self.assertEqual(len(windows), 1)
            window = windows[0]
            self.assertEqual(window.gain_target, 7.25)
            self.assertEqual((window.state_id, window.split), ("scenario-0::state_00_000003", "train"))
            self.assertEqual(window.decision_time_s, 3)
            self.assertEqual(window.backend_time_s, 2.0)
            self.assertEqual(window.raw_history_phase, "post_step_through_decision_time_minus_one")
            self.assertEqual(window.decision_projection_phase,
                             "post_public_updates_pre_candidate_generation_pre_step")
            self.assertEqual([frame.time_s for frame in window.history], [1.0, 2.0, 3.0])
            self.assertEqual(window.history[-1].platforms[0]["x_norm"], 9000.0 / 300000.0)
            self.assertEqual({event["event"] for event in window.history[-1].public_events},
                             {"task_activated", "public_observation", "search_cell_complete", "energy_anomaly"})
            self.assertIsNone(window.history[-1].action)
            batch = collate_graph_sequences([window.history])
            self.assertEqual(batch.event_features.shape[-1], EVENT_DIM)
            self.assertEqual(EVENT_DIM, 12)
            self.assertEqual(EVENT_SCHEMA_VERSION, "public_events_v2")
            for event_name in ("task_activated", "public_observation", "energy_anomaly"):
                event_index = EVENT_NAMES.index(event_name)
                self.assertGreater(batch.event_features[0, -1, event_index], 0.0)

            # Adding a post-decision row cannot change the gain input.
            _fixture(root / "future", include_decision_tick=True)
            future_window = list(iter_decision_windows(root / "future", history_steps=4))[0]
            self.assertEqual(window.history, future_window.history)

            # A projection hash mismatch or a changed phase cannot bind a decision sample.
            _fixture(root / "bad-hash", bad_projection_hash=True)
            self.assertEqual(list(iter_decision_windows(root / "bad-hash")), [])
            _fixture(root / "bad-phase", wrong_phase=True)
            self.assertEqual(list(iter_decision_windows(root / "bad-phase")), [])
            _fixture(root / "bad-backend-time", bad_backend_time=True)
            self.assertEqual(list(iter_decision_windows(root / "bad-backend-time")), [])


if __name__ == "__main__":
    unittest.main()
