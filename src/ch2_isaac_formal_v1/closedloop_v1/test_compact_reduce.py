from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compact_collection_v1 import CompactCollectionWriter, CompactPublicReader
from closedloop_v1.compact_capture import (
    append_evaluation_audit, compact_tick, write_evaluation_context,
)
from closedloop_v1.evaluate_episode import evaluate_compact_collection, reduce_episode


PHASES = ("event_state_safety", "projection_policy_planning", "command_ready",
          "post_step_online_state", "simulator_step", "post_step_evaluator_telemetry",
          "raw_artifact_logging")


def _timing(time_s: int):
    return {"event": "online_timing_tick", "event_id": f"timing:{time_s}",
            "sim_time_s": time_s, "timing": {"phases": {
                name: {"wall_duration_ns": 20, "cpu_duration_ns": 10} for name in PHASES}}}


def _raw_row(time_s: int, position_m: float, previous_m: float,
             *, visited: bool = False, completed: bool = False,
             discoveries: list[str] | None = None):
    return {"sim_time_s": time_s, "branch": "representative", "aircraft_id": "a1",
            "aircraft_type": "uav", "zone": 0, "usd_position_m": [position_m, 0.0, 0.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "previous_usd_position_m": [previous_m, 0.0, 0.0],
            "mission_xy_km": [position_m / 1000.0, 0.0], "faulted": False,
            "task_id": "task-1", "search_cell_visited": visited, "task_completed": completed,
            "target_discovered_ids": discoveries or [],
            "energy": {"incremental_wh": abs(position_m - previous_m) / 1000.0 * 100.0,
                       "remaining_wh": 999.9, "safety_pause": False},
            "communication": {"safety_hold": False, "command_attempted": time_s == 1,
                "command_attempt_accepted": True, "command_delivered": time_s == 1,
                "command_dropped": False, "forced_outage": False},
            "public_target_observations": []}


class CompactReducerTests(unittest.TestCase):
    def test_compact_reconstruction_matches_determinate_raw_reducer_evidence(self):
        cfg = {"mode": "search", "representative_duration_s": 2,
               "sensor_radius_km": 0.01, "waypoint_tolerance_km": 0.0001,
               "energy_wh_per_km": {"uav": 100.0}, "energy_capacity_wh": {"uav": 1000.0},
               "energy_reserve_fraction": 0.1, "energy_anomaly": None,
               "mechanisms": {"platform_fault": {"enabled": False},
                              "communication_outage": {"enabled": False}},
               "evaluator_schedule": {"disturbances": {}}}
        actors = [{"aircraft_id": "a1", "aircraft_type": "uav", "home_zone": 0}]
        task = SimpleNamespace(task_id="task-1", zone=0, cell=0, kind="search",
            allowed_kind="uav", public_xy_km=(0.001, 0.0),
            public_bounds_km=(0.0, -1.0, 1.0, 1.0))
        tasks = {"task-1": task}
        truth = {"targets": [{"target_id": "target-1", "zone": 0, "target_xy_km": [0.0, 0.0]}]}
        task_spec = {"task_id": "task-1", "zone": 0, "cell": 0,
            "allowed_kind": "uav", "public_bounds_km": [0.0, -1.0, 1.0, 1.0],
            "sweep_waypoints_km": [[0.001, 0.0]]}
        context = {"schema": "closedloop_compact_evaluation_context_v1", "run_id": "synthetic",
            "evaluation_config": cfg, "actor_manifest": actors,
            "initial_positions_m": {"a1": [0.0, 0.0, 0.0]},
            "zone_bounds_km": {"0": [-1.0, -1.0, 1.0, 1.0]},
            "runtime_semantics_id": None}
        manifest = {"run_id": "synthetic", "evaluation_config": cfg,
            "actor_manifest": actors, "initial_positions_m": context["initial_positions_m"],
            "zone_bounds_km": context["zone_bounds_km"],
            "public_task_manifest": [task_spec]}
        receipt = {"closedloop": {"predictor_device": "cpu"}}
        raw_rows = []
        raw_events = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            capture_dir = Path(temp) / "compact"
            writer = CompactCollectionWriter(capture_dir, {"schema": "public-static-test"})
            write_evaluation_context(capture_dir, context)
            for time_s, position, previous, visited, completed in (
                    (1, 0.0, 0.0, False, False), (2, 1.0, 0.0, True, True)):
                row = _raw_row(time_s, position, previous, visited=visited, completed=completed,
                               discoveries=["target-1"] if time_s == 1 else [])
                events = ([{"event": "target_discovered", "event_id": "discovered:1",
                            "sim_time_s": 1, "aircraft_id": "a1", "task_id": "task-1",
                            "target_id": "target-1"}] if time_s == 1 else []) + (
                         [{"event": "search_cell_complete", "event_id": "complete:2",
                            "sim_time_s": 2, "aircraft_id": "a1", "task_id": "task-1"}]
                          if completed else []) + [_timing(time_s)]
                raw_rows.append(row)
                raw_events.extend(events)
                state, public_events, decisions, costs, complete = compact_tick(
                    [row], tasks, {"a1": ["task-1"]}, {"a1": [0 if time_s == 1 else 1, 0]},
                    {"task-1": 0}, {"a1": {"task-1": [[0.001, 0.0]]}},
                    {"task-1"} if completed else set(), time_s, events,
                    {"a1": {"available": True, "safety_paused": False}}, {"task-1": "a1"})
                writer.append_tick(state, events=public_events, decisions=decisions,
                    costs=costs, event_log_complete=complete,
                    hidden_truth=truth if time_s == 1 else None)
                append_evaluation_audit(capture_dir, time_s, [row], events)
            raw_result = reduce_episode(manifest, truth, raw_rows, raw_events, receipt)
            compact_result = evaluate_compact_collection(capture_dir)
            compact_events = [event for tick in CompactPublicReader(capture_dir).iter_records()
                              for event in tick["events"]]
        self.assertEqual(raw_result["status"], "UNKNOWN")
        self.assertEqual(compact_result["status"], "UNKNOWN")
        self.assertIn("energy", compact_result, compact_result)
        self.assertTrue(any("full forced-safety evidence" in reason
                            for reason in compact_result["reasons"]))
        self.assertFalse(any(event.get("event") == "target_discovered" for event in compact_events))
        self.assertEqual(compact_result["energy"]["energy_wh"], raw_result["energy"]["energy_wh"])
        self.assertEqual(compact_result["tasks"]["waypoint_visits"], raw_result["tasks"]["waypoint_visits"])
        self.assertEqual(compact_result["tasks"]["completed_task_ids"], raw_result["tasks"]["completed_task_ids"])
        self.assertEqual(compact_result["timing"]["online_phases"], raw_result["timing"]["online_phases"])
        self.assertEqual(compact_result["communication"], raw_result["communication"])
        self.assertEqual(compact_result["discovery"]["discovered_target_ids"],
                         raw_result["discovery"]["discovered_target_ids"])


if __name__ == "__main__":
    unittest.main()
