from __future__ import annotations

import unittest

from closedloop_v1.evaluate_episode import _forced_safety_evidence


class ForcedSafetyEvidenceTests(unittest.TestCase):
    def test_communication_hold_and_resume_require_matching_state_transitions(self):
        cfg = {
            "mechanisms": {
                "platform_fault": {"enabled": False},
                "communication_outage": {"enabled": True},
            },
            "evaluator_schedule": {"disturbances": {
                "communication_outage_time_s": 2,
                "communication_outage_duration_s": 2,
            }},
            "energy_capacity_wh": {"uav": 100.0},
            "energy_reserve_fraction": 0.1,
        }
        actor = "uav-1-0"
        events = [
            {"event": "communication_safety_hold", "aircraft_id": actor,
             "sim_time_s": 2, "event_id": f"representative:2:communication_safety_hold:{actor}:"},
            {"event": "communication_safety_resume", "aircraft_id": actor,
             "sim_time_s": 4, "event_id": f"representative:4:communication_safety_resume:{actor}:"},
        ]
        states = {
            (1, actor): {"communication_held": False},
            (2, actor): {"communication_held": True, "displacement_m": 0.0},
            (3, actor): {"communication_held": True, "displacement_m": 0.0},
            (4, actor): {"communication_held": False, "displacement_m": 1.0},
        }
        result = _forced_safety_evidence(cfg, events, states, 5,
                                         {actor: {"aircraft_type": "uav"}}, [])
        self.assertIsNone(result["all_forced_safety_actions_applied"])
        self.assertTrue(result["pause_resume_actions_applied"])
        self.assertEqual(result["forced_safety_action_counts"],
                         {"expected": 2, "applied": 2, "unknown": 0, "mismatch": 0})

    def test_missing_hold_state_is_unknown_and_never_certifies_completion(self):
        cfg = {
            "mechanisms": {
                "platform_fault": {"enabled": False},
                "communication_outage": {"enabled": True},
            },
            "evaluator_schedule": {"disturbances": {
                "communication_outage_time_s": 2,
                "communication_outage_duration_s": 2,
            }},
            "energy_capacity_wh": {"uav": 100.0},
            "energy_reserve_fraction": 0.1,
        }
        actor = "uav-1-0"
        events = [{"event": "communication_safety_hold", "aircraft_id": actor,
                   "sim_time_s": 2, "event_id": "synthetic-hold"}]
        result = _forced_safety_evidence(cfg, events, {}, 5,
                                         {actor: {"aircraft_type": "uav"}}, [])
        self.assertFalse(result["all_forced_safety_actions_applied"])
        self.assertFalse(result["pause_resume_actions_applied"])
        self.assertEqual(result["forced_safety_action_counts"]["unknown"], 1)
        self.assertGreaterEqual(result["forced_safety_action_counts"]["mismatch"], 1)

    def test_early_release_is_mismatch_and_full_scope_remains_unknown(self):
        cfg = {
            "mechanisms": {"platform_fault": {"enabled": False},
                           "communication_outage": {"enabled": True}},
            "evaluator_schedule": {"disturbances": {
                "communication_outage_time_s": 2,
                "communication_outage_duration_s": 2,
            }},
            "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1,
        }
        actor = "uav-1-0"
        events = [{"event": "communication_safety_hold", "aircraft_id": actor,
                   "sim_time_s": 2, "event_id": "synthetic-hold"}]
        states = {(2, actor): {"communication_held": True, "displacement_m": 0.0},
                  (3, actor): {"communication_held": False, "displacement_m": 0.0}}
        result = _forced_safety_evidence(cfg, events, states, 4,
                                         {actor: {"aircraft_type": "uav"}}, [])
        self.assertIsNone(result["all_forced_safety_actions_applied"])
        self.assertFalse(result["pause_resume_actions_applied"])
        self.assertGreaterEqual(result["forced_safety_action_counts"]["mismatch"], 1)
        self.assertIn("hold_state_released_early@3", result["forced_safety_action_events"][0]["reasons"])

    def test_missing_actor_capacity_and_command_identity_are_unknown(self):
        cfg = {
            "mechanisms": {"platform_fault": {"enabled": False},
                           "communication_outage": {"enabled": True}},
            "evaluator_schedule": {"disturbances": {
                "communication_outage_time_s": 2,
                "communication_outage_duration_s": 2,
            }},
            "energy_capacity_wh": {}, "energy_reserve_fraction": 0.1,
        }
        actor = "uav-1-0"
        events = [
            {"event": "communication_safety_hold", "aircraft_id": actor,
             "sim_time_s": 2, "event_id": "synthetic-hold"},
            {"event": "transmission_sent", "aircraft_id": actor, "sim_time_s": 1,
             "command_effect_sim_time_s": 2},
        ]
        states = {(2, actor): {"communication_held": True, "displacement_m": 0.0},
                  (3, actor): {"communication_held": True, "displacement_m": 0.0},
                  (4, actor): {"communication_held": False}}
        result = _forced_safety_evidence(cfg, events, states, 4,
                                         {actor: {"aircraft_type": "uav"}}, [])
        self.assertFalse(result["pause_resume_actions_applied"])
        self.assertGreaterEqual(result["forced_safety_action_counts"]["unknown"], 2)
        self.assertIn("energy_capacity:uav-1-0", result["unknown_expected_event_families"])
        self.assertIn("pending_command_identity_unknown", result["forced_safety_action_events"][0]["reasons"])

    def test_energy_crossing_tick_may_move_but_next_tick_must_hold(self):
        cfg = {
            "mechanisms": {"platform_fault": {"enabled": False},
                           "communication_outage": {"enabled": False}},
            "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1,
        }
        actor = "uav-0-0"
        events = [{"event": "energy_safety_hold", "aircraft_id": actor,
                   "sim_time_s": 3, "event_id": "energy-hold-t3"}]
        states = {
            (2, actor): {"energy_paused": False, "displacement_m": 5.0},
            (3, actor): {"energy_paused": True, "displacement_m": 0.0},
        }
        result = _forced_safety_evidence(cfg, events, states, 3,
                                         {actor: {"aircraft_type": "uav"}}, [(3, actor)])
        self.assertTrue(result["pause_resume_actions_applied"])
        self.assertEqual(result["forced_safety_action_counts"]["applied"], 1)
        self.assertEqual(result["forced_safety_action_counts"]["mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
