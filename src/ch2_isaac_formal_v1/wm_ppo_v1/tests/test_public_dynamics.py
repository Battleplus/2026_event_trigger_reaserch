from __future__ import annotations

import copy
import unittest

import numpy as np

from wm_ppo_v1.features import collate_graph_sequences, public_history_to_graph
from wm_ppo_v1.public_dynamics import build_public_one_step_example, public_projection_to_frame


def _projection(time_s: int, energy: float = 90.0) -> dict:
    return {
        "sim_time_s": time_s,
        "scenario_id": "must-not-enter-features",
        "hidden_target_coordinates": [[999.0, 999.0]],
        "aircraft": {"actor-private-id": {
            "kind": "uav", "home_zone": 1, "available": True, "faulted": False,
            "safety_paused": False, "communication_held": False,
            "position_m": [1000.0 + time_s, 2000.0, 300.0],
            "velocity_mps": [1.0, 2.0, 0.0], "available_energy_wh": energy,
            "active_task_ids": ["task-a"], "queue_task_ids": ["task-a", "task-b"],
            "private_future_event": "do-not-use",
        }},
        "public_observations": [{"observation_id": "o1", "target_id": "t1", "kind": "public",
            "zone": 1, "target_xy_km": [12.0, 15.0], "observed_at_sim_time_s": 1,
            "available_at_sim_time_s": 1},
            {"observation_id": "future", "target_xy_km": [30.0, 40.0],
             "observed_at_sim_time_s": time_s, "available_at_sim_time_s": time_s + 1}],
    }


class PublicDynamicsTests(unittest.TestCase):
    def test_home_zone_normalization_preserves_zone_four_and_eight(self):
        zone_four = _projection(1)
        zone_eight = _projection(1)
        zone_four["aircraft"]["actor-private-id"]["home_zone"] = 4
        zone_eight["aircraft"]["actor-private-id"]["home_zone"] = 8
        row_four = public_history_to_graph(public_projection_to_frame(zone_four))[0]
        row_eight = public_history_to_graph(public_projection_to_frame(zone_eight))[0]
        self.assertEqual(row_four[20], 0.5)
        self.assertEqual(row_eight[20], 1.0)
        self.assertNotEqual(row_four[20], row_eight[20])

    def test_adjacent_public_transition_masks_and_ignores_hidden_fields(self):
        before, after = _projection(1), _projection(2, energy=85.0)
        changed = copy.deepcopy(before)
        changed["hidden_target_coordinates"] = [[-12345.0, 777.0]]
        changed["scenario_id"] = "different-private-scenario"
        changed["future_event_tape"] = [{"sim_time_s": 2, "event": "future-secret"}]
        changed["aircraft"]["actor-private-id"]["private_future_event"] = "changed-secret"
        graph_a = public_history_to_graph(public_projection_to_frame(before))
        graph_b = public_history_to_graph(public_projection_to_frame(changed))
        np.testing.assert_array_equal(graph_a, graph_b)
        example = build_public_one_step_example([before, after], [
            {"event": "transmission_sent", "aircraft_id": "actor-private-id", "decision_id": "d1",
             "sim_time_s": 1},
            {"event": "command_applied", "aircraft_id": "actor-private-id", "decision_id": "d1",
             "sim_time_s": 2},
            {"event": "search_cell_complete", "sim_time_s": 2, "task_id": "task-a"},
            {"event": "future_private_event", "sim_time_s": 3},
        ], events_complete=True)
        self.assertEqual((example.input_time_s, example.target_time_s), (1, 2))
        self.assertEqual(len(example.history), 1)
        self.assertEqual(example.target.dynamic["task_progress"], 1.0)
        self.assertEqual(example.target.dynamic["energy_delta"], -5.0)
        self.assertEqual(example.target.dynamic["delay"], 1.0)
        self.assertEqual(example.target.dynamic["risk"], 0.0)
        self.assertTrue(all(example.target.dynamic_mask.values()))
        batch = collate_graph_sequences([example.history], [example.target])
        self.assertEqual(batch.dynamic_target_mask.shape, (1, 4))
        self.assertTrue(batch.dynamic_target_mask.all())

    def test_future_observation_is_filtered_and_unobserved_targets_are_masked(self):
        frame = public_projection_to_frame(_projection(2))
        self.assertEqual(len(frame.observations), 1)
        example = build_public_one_step_example([_projection(1), _projection(2)], events_complete=False)
        self.assertFalse(example.target.dynamic_mask["task_progress"])
        self.assertTrue(example.target.dynamic_mask["energy_delta"])
        self.assertFalse(example.target.dynamic_mask["delay"])
        self.assertFalse(example.target.dynamic_mask["risk"])
        self.assertIsNone(example.target.gain)
        batch = collate_graph_sequences([example.history], [example.target])
        self.assertFalse(batch.dynamic_target_mask[0, 0])
        self.assertTrue(batch.dynamic_target_mask[0, 1])
        self.assertTrue(np.isnan(batch.gain_target[0]))

    def test_gap_is_not_relabelled_as_one_step_and_gain_requires_pair_validation(self):
        with self.assertRaisesRegex(ValueError, "adjacent 1-second"):
            build_public_one_step_example([_projection(1), _projection(1801)])
        with self.assertRaisesRegex(ValueError, "independently validated"):
            build_public_one_step_example([_projection(1), _projection(2)], gain_label=0.4)

    def test_ambiguous_repeated_transmissions_do_not_create_delay_label(self):
        events = [
            {"event": "transmission_sent", "aircraft_id": "actor-private-id",
             "decision_id": "d1", "sim_time_s": 1},
            {"event": "transmission_sent", "aircraft_id": "actor-private-id",
             "decision_id": "d1", "sim_time_s": 1},
            {"event": "command_applied", "aircraft_id": "actor-private-id",
             "decision_id": "d1", "sim_time_s": 2},
        ]
        example = build_public_one_step_example([_projection(1), _projection(2)], events)
        self.assertFalse(example.target.dynamic_mask["delay"])
        self.assertEqual(example.target.dynamic["delay"], 0.0)


if __name__ == "__main__":
    unittest.main()
