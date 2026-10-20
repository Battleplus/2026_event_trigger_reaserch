import unittest

from mission import abstract_strike, build_mission


class CollectionServiceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.mission = build_mission(3, (0.0, 0.0), 1, "search_strike")
        self.target_id = next(iter(self.mission.truth))
        zone, self.xy = self.mission.truth[self.target_id]
        self.zone = zone
        self.kind = self.mission.truth_kind[self.target_id]
        self.mission.discovered.add(self.target_id)

    def service(self, time_s, **kwargs):
        return abstract_strike(self.mission, self.target_id, self.kind, self.zone,
                               self.xy, time_s, 0.3, 20.0, **kwargs)

    def test_unavailable_step_resets_dwell(self):
        self.assertFalse(self.service(0.0))
        self.assertFalse(self.service(5.0, available=False))
        self.assertFalse(self.service(6.0))
        self.assertFalse(self.service(25.0))
        self.assertTrue(self.service(26.0))

    def test_paused_step_resets_dwell(self):
        self.assertFalse(self.service(0.0))
        self.assertFalse(self.service(5.0, paused=True))
        self.assertFalse(self.service(6.0))
        self.assertFalse(self.service(25.0))
        self.assertTrue(self.service(26.0))


if __name__ == "__main__":
    unittest.main()
