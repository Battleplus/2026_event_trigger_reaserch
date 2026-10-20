import json
import shutil
import unittest
from pathlib import Path

from audit_service import audit_service


class ServiceAuditFixtureTests(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(Path(__file__).parent / "_fixture_tmp", ignore_errors=True)

    def write_fixture(self, paused=False, branches=("fixture",), missing_field=None):
        root = Path(__file__).parent / "_fixture_tmp"
        root.mkdir(exist_ok=True)
        manifest = {"config": {"abstract_strike_service_duration_s": 20,
                                "abstract_strike_service_radius_km": 0.3},
                    "actor_manifest": [{"aircraft_id": "uav-1", "aircraft_type": "uav", "home_zone": 0}]}
        truth = {"targets": [{"target_id": "target-1", "zone": 0,
                               "target_xy_km": [1.0, 1.0], "kind": "uav"}]}
        events = []
        for branch in branches:
            events.extend([{"branch": branch, "event": "public_observation", "target_id": "target-1",
                            "sim_time_s": 0, "available_at_sim_time_s": 0},
                           {"branch": branch, "aircraft_id": "uav-1", "event": "abstract_strike_service_start",
                            "target_id": "target-1", "sim_time_s": 0},
                           {"branch": branch, "aircraft_id": "uav-1", "event": "abstract_strike_complete",
                            "target_id": "target-1", "sim_time_s": 20}])
        rows = []
        for branch in branches:
            for t in range(21):
                row = {"branch": branch, "aircraft_id": "uav-1", "aircraft_type": "uav", "zone": 0,
                       "sim_time_s": t, "dt_s": 1.0, "mission_xy_km": [1.0, 1.0],
                       "faulted": paused and t == 10, "energy": {"safety_pause": False},
                       "communication": {"safety_hold": False, "forced_outage": False}}
                if missing_field == "communication" and t == 10:
                    row.pop("communication")
                rows.append(row)
        for name, value in (("manifest.json", manifest), ("evaluator_truth.json", truth)):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        (root / "events.jsonl").write_text("\n".join(json.dumps(x) for x in events) + "\n", encoding="utf-8")
        (root / "trajectory.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        return root

    def test_legal_twenty_second_service_passes(self):
        root = self.write_fixture(branches=("keep", "replan"))
        report = audit_service(root / "trajectory.jsonl", root / "events.jsonl",
                               root / "manifest.json", root / "evaluator_truth.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["completion_unique_per_logical_target"], "PASS")
        self.assertEqual(report["completion_count"], 2)

    def test_pause_mid_dwell_fails(self):
        root = self.write_fixture(paused=True)
        report = audit_service(root / "trajectory.jsonl", root / "events.jsonl",
                               root / "manifest.json", root / "evaluator_truth.json")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("ineligible service" in item for item in report["failures"]))

        root = self.write_fixture()
        with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"branch": "fixture", "aircraft_id": "uav-1",
                                     "event": "abstract_strike_service_reset",
                                     "target_id": "target-1", "sim_time_s": 10}) + "\n")
        report = audit_service(root / "trajectory.jsonl", root / "events.jsonl",
                               root / "manifest.json", root / "evaluator_truth.json")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("service reset occurred" in item for item in report["failures"]))

        root = self.write_fixture(missing_field="communication")
        report = audit_service(root / "trajectory.jsonl", root / "events.jsonl",
                               root / "manifest.json", root / "evaluator_truth.json")
        self.assertEqual(report["status"], "NOT_VERIFIED")
        self.assertEqual(report["checks"]["evidence_completeness"], "NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()
