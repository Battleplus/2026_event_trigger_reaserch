from __future__ import annotations

import copy
import unittest

import numpy as np

from wm_ppo_v1.candidate_features import FEATURE_NAMES, build_candidate_features


def _projection() -> dict:
    return {
        "sim_time_s": 100,
        "scenario_id": "hidden-scenario",
        "hidden_target_coordinates": [[999.0, -999.0]],
        "aircraft": {
            "donor-a": {
                "kind": "uav", "home_zone": 2, "available": True, "faulted": False,
                "position_m": [1000.0, 2000.0, 300.0], "available_energy_wh": 80.0,
                "queue_task_ids": ["task-a", "task-b"],
                "private_future_event": "secret",
            },
            "recipient-b": {
                "kind": "uav", "home_zone": 3, "available": True, "faulted": False,
                "position_m": [5000.0, 6000.0, 400.0], "available_energy_wh": 120.0,
                "queue_task_ids": ["task-c"],
            },
        },
        "public_observations": [],
    }


def _catalog() -> list[dict]:
    return [
        {"kind": "noop", "estimated_gain_s": 0.0, "reason": "retain current allocation"},
        {"kind": "uav_transfer", "donor": "donor-a", "recipient": "recipient-b",
         "task_id": "task-b", "estimated_task_eta_s": 120.0,
         "estimated_route_distance_km": 8.5, "estimated_energy_wh": 12.0,
         "estimated_gain_s": 45.0, "switch_cost_s": 30.0,
         "private_outcome": "must-not-enter-features"},
    ]


class CandidateFeatureTests(unittest.TestCase):
    def test_noop_transfer_masks_and_stable_mapping(self):
        catalog = _catalog()
        batch = build_candidate_features(_projection(), catalog)
        self.assertEqual(batch.features.shape, (2, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(batch.candidate_mask, [True, True])
        np.testing.assert_array_equal(batch.noop_mask, [True, False])
        self.assertEqual(batch.action_keys, (("noop",), ("uav_transfer", "donor-a", "recipient-b", "task-b")))
        self.assertEqual(batch.actions[1], catalog[1])
        self.assertEqual(batch.features[1, FEATURE_NAMES.index("task_queue_position")], 1.0)
        self.assertEqual(batch.features[1, FEATURE_NAMES.index("donor_home_zone_norm")], 0.25)
        self.assertEqual(batch.features[1, FEATURE_NAMES.index("donor_available")], 1.0)
        self.assertEqual(batch.features[1, FEATURE_NAMES.index("donor_available_known")], 1.0)
        self.assertEqual(batch.features[1, FEATURE_NAMES.index("estimated_gain_s")], 45.0)
        self.assertFalse(batch.wm_action_known)
        self.assertIsNone(batch.wm_action_summary)

    def test_hidden_projection_changes_do_not_change_outputs(self):
        original = _projection()
        changed = copy.deepcopy(original)
        changed["scenario_id"] = "another-hidden-scenario"
        changed["hidden_target_coordinates"] = [[-12345.0, 7.0]]
        changed["future_event_tape"] = [{"event": "secret", "sim_time_s": 101}]
        changed["aircraft"]["donor-a"]["private_future_event"] = "changed"
        first = build_candidate_features(original, _catalog())
        second = build_candidate_features(changed, _catalog())
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.candidate_mask, second.candidate_mask)
        np.testing.assert_array_equal(first.noop_mask, second.noop_mask)
        self.assertEqual(first.action_keys, second.action_keys)

    def test_missing_public_fields_have_unknown_bits_and_limit_is_enforced(self):
        projection = _projection()
        del projection["aircraft"]["donor-a"]["available_energy_wh"]
        batch = build_candidate_features(projection, _catalog())
        donor_energy_known = FEATURE_NAMES.index("donor_energy_wh_known")
        route_known = FEATURE_NAMES.index("estimated_route_distance_km_known")
        self.assertEqual(batch.features[1, donor_energy_known], 0.0)
        self.assertEqual(batch.features[0, route_known], 0.0)
        too_many = [{"kind": "noop"}]
        too_many.extend({"kind": "uav_transfer", "donor": "donor-a", "recipient": "recipient-b",
                         "task_id": f"task-{index}"} for index in range(16))
        with self.assertRaisesRegex(ValueError, "exceeds the 16-candidate limit"):
            build_candidate_features(projection, too_many)


if __name__ == "__main__":
    unittest.main()
