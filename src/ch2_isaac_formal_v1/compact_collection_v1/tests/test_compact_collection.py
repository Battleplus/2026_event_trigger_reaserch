from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from compact_collection_v1 import (
    CompactCollectionWriter,
    CompactPublicReader,
    iter_public_one_step_examples,
)


def _state(time_s: int, *, future_observation: bool = False, secret: int = 1) -> dict:
    observations = [{"observation_id": "obs-1", "target_xy_km": [4.0, 5.0],
                     "observed_at_sim_time_s": 1, "available_at_sim_time_s": 1}]
    if future_observation:
        observations.append({"observation_id": "obs-late", "target_xy_km": [8.0, 9.0],
                             "observed_at_sim_time_s": 1, "available_at_sim_time_s": 2})
    return {
        "sim_time_s": time_s,
        "scenario_id": f"hidden-{secret}",
        "hidden_target_coordinates": [[float(secret), 99.0]],
        "aircraft": {"a1": {
            "kind": "uav", "home_zone": 2, "available": True, "faulted": False,
            "safety_paused": False, "communication_held": False,
            "position_m": [10.0 + time_s, 20.0, 30.0], "velocity_mps": [1.0, 0.0, 0.0],
            "available_energy_wh": 90.0 - time_s,
            "active_task_ids": ["task-1"], "queue_task_ids": ["task-1", "task-2"],
        }},
        "tasks": {"task-1": {"assignment": "a1", "queue_position": 0}},
        "public_observations": observations,
    }


class CompactCollectionTests(unittest.TestCase):
    def test_stream_reconstructs_public_ticks_and_adapts_one_step(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            path = Path(temp) / "run"
            writer = CompactCollectionWriter(path, {"terrain": "flat", "tree_count": 140})
            writer.append_tick(_state(1, future_observation=True),
                events=[{"event": "transmission_sent", "aircraft_id": "a1",
                         "decision_id": "d1", "sim_time_s": 1}],
                decisions=[{"reason": "new_task", "replan": True}], costs=[{"planner_ms": 3.5}],
                event_log_complete=True, hidden_truth={"target": [900, 800]})
            writer.append_tick(_state(2, future_observation=True),
                events=[{"event": "command_applied", "aircraft_id": "a1",
                         "decision_id": "d1", "sim_time_s": 2}],
                event_log_complete=True, hidden_truth={"target": [901, 800]})
            changed = _state(3, future_observation=True)
            changed["aircraft"]["a1"]["position_m"] = [12.0, 20.0, 30.0]
            changed["aircraft"]["a1"]["queue_task_ids"] = ["task-2", "task-1"]
            changed["aircraft"]["a1"]["active_task_ids"] = ["task-2"]
            changed["aircraft"]["a1"]["safety_paused"] = True
            changed["tasks"] = {"task-2": {"assignment": "a1", "queue_position": 0},
                                 "task-3": {"assignment": None, "queue_position": 1}}
            writer.append_tick(changed, event_log_complete=True)
            writer.append_pair_snapshot("pair-1", "keep", _state(1),
                                        pair_engine_snapshot_ref="pair-engine://pair-1/keep",
                                        pair_engine_snapshot_sha256="a" * 64)

            reader = CompactPublicReader(path)
            rows = list(reader.iter_records())
            first, second = (row["projection"] for row in rows[:2])
            self.assertEqual(first["aircraft"], _state(1)["aircraft"])
            self.assertEqual(first["tasks"], _state(1)["tasks"])
            self.assertEqual([x["observation_id"] for x in first["public_observations"]], ["obs-1"])
            self.assertEqual([x["observation_id"] for x in second["public_observations"]],
                             ["obs-1", "obs-late"])
            third = rows[2]["projection"]
            self.assertEqual(third["aircraft"], changed["aircraft"])
            self.assertEqual(third["tasks"], changed["tasks"])
            self.assertEqual(third["aircraft"]["a1"]["position_m"], second["aircraft"]["a1"]["position_m"])
            self.assertEqual(rows[0]["decisions"][0]["reason"], "new_task")
            self.assertEqual(rows[0]["costs"][0]["planner_ms"], 3.5)
            self.assertTrue((path / "evaluation_truth.jsonl").exists())
            self.assertTrue((path / "paired_snapshots.jsonl").exists())
            pair_record = json.loads((path / "paired_snapshots.jsonl").read_text("utf-8"))
            self.assertIn("pair_engine_snapshot_ref", pair_record)
            self.assertEqual(pair_record["pair_engine_snapshot_sha256"], "a" * 64)
            examples = list(iter_public_one_step_examples(reader, history_steps=1))
            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0].input_time_s, 1)
            self.assertEqual(examples[0].target_time_s, 2)
            self.assertEqual(examples[0].target.dynamic["energy_delta"], -1.0)
            self.assertEqual(examples[0].target.dynamic["delay"], 1.0)

    def test_hidden_truth_changes_do_not_change_public_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            outputs = []
            for index, truth in enumerate((111, 999)):
                path = Path(temp) / f"run-{index}"
                writer = CompactCollectionWriter(path, {"terrain": "flat"})
                state = _state(1, future_observation=True, secret=truth)
                writer.append_tick(state, hidden_truth={"target": [truth, -truth]})
                outputs.append(list(CompactPublicReader(path).iter_public_states()))
            self.assertEqual(outputs[0], outputs[1])

    def test_requires_new_directory_and_contiguous_ticks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            path = Path(temp) / "run"
            writer = CompactCollectionWriter(path, {})
            with self.assertRaisesRegex(ValueError, "contiguous"):
                writer.append_tick(_state(1))
                writer.append_tick(_state(3))
            with self.assertRaises(FileExistsError):
                CompactCollectionWriter(path, {})


if __name__ == "__main__":
    unittest.main()
