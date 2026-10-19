import unittest

from mission import (abstract_strike, assign_cross_zone_support, build_mission,
                     initial_c1_greedy, mission_snapshot, restore_mission,
                     task_waypoint, task_sweep)


class MissionTests(unittest.TestCase):
    def test_hidden_load_does_not_change_public_search_problem(self):
        a = build_mission(17, (4.0, -3.0), 1)
        b = build_mission(17, (4.0, -3.0), 5)
        self.assertEqual([t.public_xy_km for t in a.tasks], [t.public_xy_km for t in b.tasks])
        self.assertEqual(len(a.tasks), 81)
        self.assertEqual(len(a.truth), 9)
        self.assertEqual(len(b.truth), 45)
        self.assertEqual(task_waypoint(a, a.tasks[0], 100.0, (0.0, 0.0)),
                         task_waypoint(b, b.tasks[0], 100.0, (0.0, 0.0)))
        self.assertEqual([t.allowed_kind for t in a.tasks], [t.allowed_kind for t in b.tasks])

    def test_platform_specialization_and_helicopter_zone_lock(self):
        m = build_mission(1, (0.0, 0.0), 3)
        allocation = initial_c1_greedy(m)
        by_id = {a.aircraft_id:a for a in m.aircraft}
        tasks = {t.task_id:t for t in m.tasks}
        for task_id, actor_id in allocation.items():
            self.assertEqual(by_id[actor_id].kind, tasks[task_id].allowed_kind)
        self.assertFalse(assign_cross_zone_support(m, 1, "heli-0-0"))
        self.assertTrue(assign_cross_zone_support(m, 1, "uav-0-0"))

    def test_abstract_strike_is_idempotent_and_mode_gated(self):
        m = build_mission(3, (0.0, 0.0), 1, "search_strike")
        target_id = "target-z0-0"
        m.discovered.add(target_id)
        zone, xy = m.truth[target_id]
        kind = m.truth_kind[target_id]
        self.assertFalse(abstract_strike(m,target_id,kind,zone,xy,30.0))
        self.assertFalse(abstract_strike(m,target_id,kind,zone,xy,49.0))
        self.assertTrue(abstract_strike(m,target_id,kind,zone,xy,50.0))
        self.assertFalse(abstract_strike(m,target_id,kind,zone,xy,51.0))
        self.assertEqual(sum(e["event"] == "abstract_strike_complete" for e in m.event_log), 1)

    def test_public_sweep_stays_inside_cell_and_has_sensor_spacing(self):
        m = build_mission(9, (0.0, 0.0), 3)
        task = m.tasks[0]
        path = task_sweep(task, lane_spacing_km=0.8, point_spacing_km=0.4)
        xmin,xmax,ymin,ymax=task.public_bounds_km
        self.assertTrue(all(xmin <= x <= xmax and ymin <= y <= ymax for x,y in path))
        self.assertLessEqual(max(abs(path[i][1]-path[i-1][1]) for i in range(1,len(path)) if path[i][0] < path[i-1][0] or path[i][1] != path[i-1][1]), 0.8)

    def test_mission_snapshot_restore(self):
        m = build_mission(5, (1.0, 2.0), 3)
        initial_c1_greedy(m)
        snap = mission_snapshot(m)
        m.completed.add(m.tasks[0].task_id)
        m.event_log.append({"event":"mutated"})
        restore_mission(m, snap)
        self.assertNotIn(m.tasks[0].task_id, m.completed)
        self.assertNotIn({"event":"mutated"}, m.event_log)
        self.assertEqual(snap["aircraft_available"], {a.aircraft_id:a.available for a in m.aircraft})


if __name__ == "__main__":
    unittest.main()
