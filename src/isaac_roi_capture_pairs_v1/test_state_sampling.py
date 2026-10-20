import unittest

from state_sampling import StateSampler, public_projection


class StateSamplingTests(unittest.TestCase):
    def test_projection_and_selection_ignore_gain_and_hidden_truth(self):
        def run(gain, hidden):
            sampler = StateSampler(9127, 8)
            selected = []
            for sequence, time_s in enumerate(range(1, 20), 1):
                readback = {"a": {"position_m": [time_s, 2, 3], "velocity_mps": [1, 0, 0],
                                  "hidden_truth": hidden}}
                projection = public_projection(time_s, readback, {"a": 10}, {"a": ["task-1"]},
                                               {"a": ["task-1"]}, [], {"a": {"kind": "uav",
                                               "home_zone": 1, "available": True, "faulted": False,
                                               "safety_paused": False, "communication_held": False}})
                sampler.offer(sequence, time_s, ["public_observation"], False,
                              lambda: {"gain": gain, "hidden": hidden}, projection)
            return [(x["sim_time_s"], x["priority"], x["public_projection"]) for x in sampler.selected]

        self.assertEqual(run(1, "truth-A"), run(999999, "truth-B"))
        projection = public_projection(5, {"a": {"position_m": [1, 2, 3],
             "velocity_mps": [0, 0, 0], "truth": "secret", "scene_id_hash": "secret"}},
             {"a": 5}, {"a": ["active"]}, {"a": ["active"]},
             [{"target_id": "t1", "target_xy_km": [2, 3], "kind": "vehicle",
               "activation_times": [9], "private_mission": "secret"}],
             {"a": {"kind": "uav", "home_zone": 1, "available": True,
                    "faulted": False, "safety_paused": False, "communication_held": False}})
        encoded = repr(projection)
        self.assertIn("target_xy_km", encoded)
        for forbidden in ("truth", "scene_id_hash", "activation_times", "private_mission"):
            self.assertNotIn(forbidden, encoded)

    def test_duplicate_time_and_limit(self):
        sampler = StateSampler(11, 8)
        for sequence, time_s in enumerate(range(1, 20), 1):
            sampler.offer(sequence, time_s, ["task_activated"], False, lambda: {"v": time_s}, {})
        before = tuple(row["priority"] for row in sampler.selected)
        sampler.offer(100, 19, ["public_observation"], False, lambda: {"v": 100}, {})
        self.assertEqual(len(sampler.selected), 8)
        self.assertEqual(before, tuple(row["priority"] for row in sampler.selected))
        forced = StateSampler(11, 8)
        self.assertFalse(forced.offer(1, 3, ["task_activated"], True, lambda: {}, {}))
        self.assertEqual(len(forced.selected), 0)

    def test_selected_state_is_deep_copied(self):
        sampler = StateSampler(5)
        state = {"nested": {"value": [1]}}
        projection = {"aircraft": {"a": {"position_m": [1, 2, 3]}}}
        sampler.offer(1, 7, ["search_cell_complete"], False, lambda: state, projection)
        state["nested"]["value"].append(2)
        projection["aircraft"]["a"]["position_m"][0] = 99
        self.assertEqual(sampler.selected[0]["private_state"]["nested"]["value"], [1])
        self.assertEqual(sampler.selected[0]["public_projection"]["aircraft"]["a"]["position_m"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
