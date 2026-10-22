from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compact_collection_v1 import CompactCollectionWriter, CompactPublicReader
from closedloop_v1.compact_capture import compact_tick


def _row(time_s: int, energy: float):
    return {"aircraft_id": "a1", "aircraft_type": "uav", "zone": 0,
            "faulted": False, "usd_position_m": [time_s, 2.0, 3.0],
            "velocity_mps": [1.0, 0.0, 0.0],
            "energy": {"remaining_wh": energy, "incremental_wh": 2.0,
                       "safety_pause": False},
            "communication": {"safety_hold": False, "command_attempted": True,
                              "command_delivered": time_s > 1},
            "public_target_observations": []}


class CompactCaptureTests(unittest.TestCase):
    def test_runtime_adapter_filters_future_and_hidden_events_and_keeps_tick_changes(self):
        tasks = {
            "now": SimpleNamespace(task_id="now", zone=0, cell=0, kind="search",
                allowed_kind="uav", public_xy_km=(1.0, 2.0),
                public_bounds_km=(0.0, 1.0, 2.0, 3.0)),
            "later": SimpleNamespace(task_id="later", zone=0, cell=1, kind="search",
                allowed_kind="uav", public_xy_km=(3.0, 4.0),
                public_bounds_km=(2.0, 3.0, 4.0, 5.0)),
        }
        queues = {"a1": ["now", "later"]}
        cursor = {"a1": [0, 0]}
        activation = {"now": 0, "later": 5}
        routes = {"a1": {"now": [(1, 2)], "later": [(3, 4)]}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            writer = CompactCollectionWriter(Path(temp) / "capture", {"terrain_grid": 128})
            for now in (1, 2):
                events = [
                    {"event": "task_activated", "sim_time_s": now, "task_id": "now"},
                    {"event": "target_discovered", "sim_time_s": now,
                     "target_id": "hidden-id", "target_xy_km": [99, 99]},
                    {"event": "task_activated", "sim_time_s": 5, "task_id": "later"},
                ]
                state, public_events, decisions, costs, events_complete = compact_tick(
                    [_row(now, 100.0 - now)], tasks, queues, cursor, activation,
                    routes, set(), now, events,
                    {"a1": {"available": False, "safety_paused": True}},
                    {"now": "a1", "later": "a1"})
                writer.append_tick(state, events=public_events, decisions=decisions,
                                   costs=costs, event_log_complete=events_complete,
                                   hidden_truth={"private": now})
            rows = list(CompactPublicReader(Path(temp) / "capture").iter_records())
            self.assertEqual(set(rows[0]["projection"]["public_tasks"]), {"now"})
            self.assertNotIn("hidden-id", str(rows[0]))
            self.assertEqual(rows[0]["projection"]["aircraft"]["a1"]["available_energy_wh"], 99.0)
            self.assertEqual(rows[1]["projection"]["aircraft"]["a1"]["available_energy_wh"], 98.0)
            self.assertFalse(rows[0]["projection"]["aircraft"]["a1"]["available"])
            self.assertEqual(rows[0]["projection"]["aircraft"]["a1"]["queue_task_ids"], ["now"])
            self.assertFalse(rows[0]["event_log_complete"])
            self.assertEqual(rows[1]["costs"][0]["energy_incremental_wh"], 2.0)
            self.assertEqual(len((Path(temp) / "capture" / "evaluation_truth.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
