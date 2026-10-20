import hashlib
import json
import shutil
import unittest
from pathlib import Path

from reduce_gain import reduce_suffix


class GainReductionFixtureTests(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(Path(__file__).parent / "_fixture_tmp", ignore_errors=True)

    def make_fixture(self, *, replan_distance=2.0, omit_baseline=False, restore_ok=True, audit_ok=True):
        root = Path(__file__).parent / "_fixture_tmp"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(exist_ok=True)
        truth = {"targets": [{"target_id": "target-1", "zone": 0, "target_xy_km": [0.0, 0.0], "kind": "uav"}]}
        manifest = {"fixture_only": True, "config": {"mode": "search", "abstract_strike_service_duration_s": 20},
                    "actor_manifest": [{"aircraft_id": "uav-1", "aircraft_type": "uav", "home_zone": 0}]}
        receipt = {"status": "complete", "actual_isaac_run": True,
                   "pair_snapshot": {"sim_time_s": 5399, "restore_full_state_match": restore_ok,
                                     "state_sha256": "state", "restored_state_sha256": "state" if restore_ok else "other"}}
        if not omit_baseline:
            receipt["suffix"] = {"initial_discovered_target_ids": [], "initial_serviced_target_ids": []}
        for name, value in (("manifest.json", manifest), ("evaluator_truth.json", truth), ("run_receipt.json", receipt)):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        for branch, distance in (("keep", 1.0), ("replan", replan_distance)):
            branch_dir = root / branch; branch_dir.mkdir()
            rows = [{"branch": branch, "aircraft_id": "uav-1", "aircraft_type": "uav", "zone": 0,
                     "sim_time_s": 5400 + i, "dt_s": 1.0, "usd_position_m": [distance * (i + 1), 0, 0],
                     "previous_usd_position_m": [distance * i, 0, 0],
                     "energy": {"effective_wh_per_km": 50000000.0}} for i in range(2)]
            events = [{"branch": branch, "event": "target_discovered", "target_id": "target-1", "sim_time_s": 5400},
                      {"branch": branch, "event": "public_observation", "target_id": "target-1", "sim_time_s": 5401}]
            if branch == "replan":
                events.extend([{"branch": branch, "event": "transmission_sent", "sim_time_s": 5400},
                               {"branch": branch, "event": "donor_support_transfer", "sim_time_s": 5400,
                                "from_aircraft_id": "uav-1", "to_aircraft_id": "uav-1"}])
            (branch_dir / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (branch_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            if audit_ok:
                binding = {"trajectory.jsonl": hashlib.sha256((branch_dir / "trajectory.jsonl").read_bytes()).hexdigest(),
                           "events.jsonl": hashlib.sha256((branch_dir / "events.jsonl").read_bytes()).hexdigest(),
                           "evaluator_truth.json": hashlib.sha256((root / "evaluator_truth.json").read_bytes()).hexdigest()}
                (branch_dir / "audit_report.json").write_text(json.dumps({"status": "pass", "input_sha256": binding}), encoding="utf-8")
        return root

    def test_energy_and_incremental_cost_boundaries(self):
        with self.subTest("positive gain"):
            root = self.make_fixture(replan_distance=2.0)
            report = reduce_suffix(root, root / "reduction.json")
            self.assertEqual(report["status"], "PASS")
            self.assertLess(report["G"], 0.0)
            self.assertGreater(report["C_increment"], 0.0)
        with self.subTest("negative energy delta"):
            root = self.make_fixture(replan_distance=0.5)
            report = reduce_suffix(root, root / "reduction.json")
            self.assertEqual(report["status"], "PASS")
            self.assertGreater(report["J_exec_keep"], report["J_exec_replan"])
            self.assertGreater(report["branches"]["keep"]["energy_wh"] - report["branches"]["replan"]["energy_wh"], 0.0)

    def test_missing_baseline_restore_or_audit_is_unknown(self):
        for kwargs in ({"omit_baseline": True}, {"restore_ok": False}, {"audit_ok": False}):
            with self.subTest(kwargs=kwargs):
                root = self.make_fixture(**kwargs)
                report = reduce_suffix(root, root / "reduction.json")
                self.assertEqual(report["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
