from __future__ import annotations

import unittest
from types import SimpleNamespace

from closedloop_v1.decision import DecisionAdapter, WMHistoryUnavailable
from wm_ppo_v1.decision_windows import _decision_frame
from wm_ppo_v1.closedloop_history import LivePublicHistory
from wm_ppo_v1.raw_public_windows import _tick_frame


class WMHistoryTests(unittest.TestCase):
    def test_unavailable_history_suppresses_planner_and_reports_reason(self):
        adapter = DecisionAdapter(mode="existing_et_gain_gate", gain_predictor=lambda _p: (_ for _ in ()).throw(WMHistoryUnavailable()))
        actor = "a"
        projection = {"aircraft": {actor: {}}}
        result = adapter.decide(None, 1, projection, ["task_activated"], lambda: self.fail("planner called"),
                                readback={actor: {}})
        self.assertTrue(result["gate_suppressed"])
        self.assertEqual(result["gate_suppression_reason"], "WM_HISTORY_UNAVAILABLE")
        self.assertIsNone(result["predicted_gain"])
        self.assertFalse(result["planner_called"])

    def test_forced_safety_bypasses_unavailable_predictor(self):
        adapter = DecisionAdapter(mode="existing_et_gain_gate", gain_predictor=lambda _p: self.fail("predictor called"))
        actor = "a"
        result = adapter.decide(None, 1, {"aircraft": {actor: {}}}, ["platform_fault"],
                                lambda: ([], {"kind": "noop"}), readback={actor: {}})
        self.assertTrue(result["planner_called"])

    def test_prediction_metadata_survives_decision_adapter(self):
        prediction = SimpleNamespace(score=0.7, gain_mean=0.8, gain_lower_bound=0.7,
            threshold=0.2, mode="lower_bound", version="wm-test", device="cuda:0",
            integration_status="NOT_INTEGRATED", timings_ms={"feature_construction_ms": 1.0})
        adapter = DecisionAdapter(mode="existing_et_gain_gate", gain_predictor=lambda _p: prediction)
        result = adapter.decide(None, 1, {"aircraft": {"a": {}}}, ["task_activated"],
                                lambda: ([], {"kind": "noop"}), readback={"a": {}})
        self.assertEqual(result["predicted_gain"], 0.7)
        self.assertEqual(result["prediction_metadata"]["gain_mean"], 0.8)
        self.assertEqual(result["prediction_metadata"]["timings_ms"]["feature_construction_ms"], 1.0)

    def test_live_history_is_bounded_and_drops_untrusted_raw_fields(self):
        history = LivePublicHistory(["a"])
        for tick in range(1, 10):
            history.append_post_step(tick, [{"aircraft_id": "a", "usd_position_m": [0, 0, 0],
                "velocity_mps": [0, 0, 0], "energy": {"remaining_wh": 50.0},
                "communication": {"safety_hold": False}, "target_discovered_ids": ["secret"]}], [])
        self.assertEqual(len(history.frames), 7)
        self.assertNotIn("secret", repr(history.frames))
        with self.assertRaises(WMHistoryUnavailable):
            history.decision_history({}, 11, [])

    def test_effective_window_matches_raw_training_history_and_decision_projection(self):
        history = LivePublicHistory(["a"])
        raw_frames = []
        for tick in range(1, 8):
            rows = [{"aircraft_id": "a", "aircraft_type": "uav", "zone": 2,
                "usd_position_m": [tick * 1000.0, 2000.0, 3000.0],
                "velocity_mps": [1.0, 2.0, 3.0], "energy": {"remaining_wh": 50.0 + tick},
                "communication": {"safety_hold": False}, "faulted": False,
                "public_target_observations": [], "target_discovered_ids": ["hidden"]}]
            tick_events = []
            raw_frames.append(_tick_frame(tick, rows, {}, set(), set(), tick_events))
            history.append_post_step(tick, rows, tick_events)

        projection = {"sim_time_s": 8, "aircraft": {"a": {
            "kind": "uav", "home_zone": 2, "available": True, "faulted": False,
            "safety_paused": False, "communication_held": False,
            "position_m": [8000.0, 2000.0, 3000.0], "velocity_mps": [1.0, 2.0, 3.0],
            "available_energy_wh": 58.0}}, "public_observations": []}
        ordinary_events = ["public_observation"]
        raw_history = tuple(history.decision_history(projection, 8, ordinary_events))
        training_history = tuple(raw_frames) + (_decision_frame(projection, ordinary_events),)
        self.assertEqual(len(raw_history), 8)
        self.assertEqual(raw_history, training_history)
        for left, right in zip(raw_history, training_history):
            self.assertEqual(left.time_s, right.time_s)
            self.assertEqual(left.platforms, right.platforms)
            self.assertEqual(left.tasks, right.tasks)
            self.assertEqual(left.observations, right.observations)
            self.assertEqual(left.public_events, right.public_events)
            self.assertNotIn("hidden", repr(left))


if __name__ == "__main__":
    unittest.main()
