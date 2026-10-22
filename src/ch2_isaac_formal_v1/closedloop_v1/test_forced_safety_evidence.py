from __future__ import annotations

import copy
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

    def test_fault_release_uses_executor_queue_audit_and_preserves_unrecovered_failure_cost(self):
        actor = "uav-0-0"
        audit = {
            "schema": "fault_queue_release_state_v1",
            "source": "executor_queues_cursor_and_mission_assignments_after_release",
                "pre_fault": {"queue_task_ids": ["task-active", "task-unrecovered"],
                          "cursor": [0, 1], "unfinished_task_ids": ["task-active", "task-unrecovered"]},
            "post_fault": {
                "queues_by_aircraft": {actor: [], "uav-0-1": []},
                "cursors_by_aircraft": {actor: [0, 0], "uav-0-1": [0, 0]},
                "assignments_by_task": {},
                "completed_task_ids": [],
                "actor_state_by_id": {
                    actor: {"kind": "uav", "home_zone": 0, "available": False,
                            "faulted": True, "safety_paused": False, "communication_held": False},
                    "uav-0-1": {"kind": "uav", "home_zone": 0, "available": True,
                                "faulted": False, "safety_paused": False, "communication_held": False},
                },
            },
        }
        events = [
            {"event": "platform_fault", "aircraft_id": actor, "sim_time_s": 2,
             "event_id": "fault-2", "details": {"unfinished_tasks_released": ["task-active", "task-unrecovered"],
                 "queue_release_state": audit}},
            {"event": "fault_task_unrecovered", "aircraft_id": actor,
             "task_id": "task-active", "sim_time_s": 2,
             "details": {"failure_cost_unit": "one_unrecovered_public_task"}},
            {"event": "fault_task_unrecovered", "aircraft_id": actor,
             "task_id": "task-unrecovered", "sim_time_s": 2,
             "details": {"failure_cost_unit": "one_unrecovered_public_task"}},
        ]
        result = _forced_safety_evidence(
            {"mechanisms": {"platform_fault": {"enabled": True},
                            "communication_outage": {"enabled": False}},
             "evaluator_schedule": {"disturbances": {"platform_fault_time_s": 2}},
             "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1},
            events,
            {(2, actor): {"faulted": True, "displacement_m": 0.0}},
            2,
            {actor: {"aircraft_type": "uav", "home_zone": 0}},
            [],
            task_by_id={"task-unrecovered": {"allowed_kind": "uav", "zone": 0}},
            runtime_semantics_id="closedloop_v1_fault_release_accounting_v2",
        )
        self.assertTrue(result["pause_resume_actions_applied"])
        self.assertTrue(result["all_forced_safety_actions_applied"], result)
        self.assertEqual(result["fault_task_release"]["status"], "applied")
        self.assertEqual(result["fault_task_release"]["released_task_ids"], ["task-active", "task-unrecovered"])
        self.assertEqual(result["fault_task_release"]["unrecovered_task_ids"], ["task-active", "task-unrecovered"])
        self.assertEqual(result["fault_task_release"]["unrecovered_task_failure_cost"], 2)

        corrupt_audit = copy.deepcopy(audit)
        corrupt_audit["post_fault"]["queues_by_aircraft"][actor] = ["task-unrecovered"]
        corrupt_audit["post_fault"]["assignments_by_task"]["task-unrecovered"] = actor
        corrupt_events = copy.deepcopy(events)
        corrupt_events[0]["details"]["queue_release_state"] = corrupt_audit
        rejected = _forced_safety_evidence(
            {"mechanisms": {"platform_fault": {"enabled": True},
                            "communication_outage": {"enabled": False}},
             "evaluator_schedule": {"disturbances": {"platform_fault_time_s": 2}},
             "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1},
            corrupt_events, {(2, actor): {"faulted": True, "displacement_m": 0.0}},
            2, {actor: {"aircraft_type": "uav", "home_zone": 0}}, [],
            task_by_id={"task-unrecovered": {"allowed_kind": "uav", "zone": 0}},
            runtime_semantics_id="closedloop_v1_fault_release_accounting_v2")
        self.assertFalse(rejected["all_forced_safety_actions_applied"])
        self.assertEqual(rejected["fault_task_release"]["status"], "mismatch")
        self.assertTrue(any("released_task_retained_by_fault_actor" in row["reasons"]
                            for row in rejected["fault_task_release"]["events"]))

    def test_reassigned_task_without_public_legality_spec_stays_unknown(self):
        actor, recipient = "uav-0-0", "uav-0-1"
        event = {"event": "platform_fault", "aircraft_id": actor, "sim_time_s": 2,
                 "event_id": "fault-2", "details": {"unfinished_tasks_released": ["task-moved"],
                 "queue_release_state": {
                     "schema": "fault_queue_release_state_v1",
                     "source": "executor_queues_cursor_and_mission_assignments_after_release",
                     "pre_fault": {"queue_task_ids": ["task-moved"], "cursor": [0, 0],
                                   "unfinished_task_ids": ["task-moved"]},
                     "post_fault": {
                         "queues_by_aircraft": {actor: [], recipient: ["task-moved"]},
                         "cursors_by_aircraft": {actor: [0, 0], recipient: [0, 0]},
                         "assignments_by_task": {"task-moved": recipient},
                         "completed_task_ids": [],
                         "actor_state_by_id": {
                             actor: {"kind": "uav", "home_zone": 0, "available": False,
                                     "faulted": True, "safety_paused": False, "communication_held": False},
                             recipient: {"kind": "uav", "home_zone": 1, "available": True,
                                         "faulted": False, "safety_paused": False, "communication_held": False},
                         },
                     },
                 }}}
        result = _forced_safety_evidence(
            {"mechanisms": {"platform_fault": {"enabled": True},
                            "communication_outage": {"enabled": False}},
             "evaluator_schedule": {"disturbances": {"platform_fault_time_s": 2}},
             "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1},
            [event], {(2, actor): {"faulted": True, "displacement_m": 0.0}},
            2, {actor: {"aircraft_type": "uav", "home_zone": 0}}, [],
            task_by_id={}, runtime_semantics_id="closedloop_v1_fault_release_accounting_v2")
        self.assertIsNone(result["all_forced_safety_actions_applied"])
        self.assertEqual(result["fault_task_release"]["status"], "unknown")
        self.assertIsNone(result["fault_task_release"]["unrecovered_task_failure_cost"])

    def test_pause_resume_pass_alone_cannot_certify_full_forced_safety(self):
        cfg = {"mechanisms": {"platform_fault": {"enabled": True},
                              "communication_outage": {"enabled": False}},
               "evaluator_schedule": {"disturbances": {"platform_fault_time_s": 2}},
               "energy_capacity_wh": {"uav": 100.0}, "energy_reserve_fraction": 0.1}
        event = {"event": "platform_fault", "aircraft_id": "uav-0-0", "sim_time_s": 2,
                 "event_id": "fault-2"}
        states = {(2, "uav-0-0"): {"faulted": True, "displacement_m": 0.0}}
        result = _forced_safety_evidence(cfg, [event], states, 2,
            {"uav-0-0": {"aircraft_type": "uav", "home_zone": 0}}, [],
            runtime_semantics_id="closedloop_v1_fault_release_accounting_v2")
        self.assertTrue(result["pause_resume_actions_applied"])
        self.assertIsNone(result["all_forced_safety_actions_applied"])
        self.assertEqual(result["fault_task_release"]["status"], "unknown")
        self.assertIsNone(result["fault_task_release"]["unrecovered_task_failure_cost"])


if __name__ == "__main__":
    unittest.main()
