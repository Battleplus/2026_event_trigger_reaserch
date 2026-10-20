from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aggregate import aggregate
from evaluate_suffix import _sha, _snapshot_digest, _sweep, evaluate_suffix


def _write_json(path: Path, value, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


class FormalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="suffix-evidence-")
        self.root = Path(self.tmp.name)
        self.case = self._fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def _fixture(self):
        source_dir = self.root / "source_run"
        sampling_dir = source_dir / "state_sampling"
        private_dir = sampling_dir / "private_states"
        suffix_dir = self.root / "suffix_730201"
        actor_id = "uav-0-0"
        task_id = "z0-cell0"
        decision_time = 5399
        config = {
            "run_id": "capture-scenario-a", "scenario_id": "scenario-a", "split": "train",
            "seed": 41, "mode": "search", "roi_width_km": 50.0,
            "formal": {"scenario_id": "scenario-a", "split": "train"},
            "evaluator_schedule": None,
            "sensor_radius_km": 0.5, "sweep_lane_spacing_km": 0.8,
            "waypoint_tolerance_km": 0.12,
            "energy_wh_per_km": {"uav": 240.0, "helicopter": 900.0},
            "energy_anomaly": None,
            "abstract_strike_service_radius_km": 0.3,
            "abstract_strike_service_duration_s": 20,
            "communication": {"delay_s": 2, "dropout_probability": 0.0},
            "representative_events": {"uav_fault_time_s": 8000,
                                      "comm_outage_time_s": 9000,
                                      "comm_outage_duration_s": 120},
            "pilot": {"activation_task_ids": [task_id], "activation_times_s": [0]},
        }
        config_path = source_dir / "config.json"
        _write_json(config_path, config)
        config_hash = _sha(config_path)
        zone_bounds = {}
        for zone in range(9):
            col, row = zone % 3, zone // 3
            edge = 50.0 / 3.0
            zone_bounds[str(zone)] = [-25 + col * edge, -25 + row * edge,
                                      -25 + (col + 1) * edge, -25 + (row + 1) * edge]
        task_bounds = [-25.0, -25.0, -25.0 + 50.0 / 9.0, -25.0 + 50.0 / 9.0]
        task_record = {"task_id": task_id, "zone": 0, "cell": 0,
                       "allowed_kind": "uav", "public_bounds_km": task_bounds}
        route = _sweep(task_record, 0.8, 0.5)
        public_task = {**task_record, "sweep_waypoints_km": [list(point) for point in route]}
        actor = {"aircraft_id": actor_id, "aircraft_type": "uav", "home_zone": 0,
                 "max_speed_mps": 65.0}
        truth = {"targets": [{"target_id": "target-z0-0", "zone": 0,
                              "target_xy_km": [-18.0, -18.0], "kind": "uav"}]}
        truth_path = source_dir / "evaluator_truth.json"
        _write_json(truth_path, truth)

        position_scene = [-220.0, -220.0, 200.0]
        backend_state = {"scene_time_s": decision_time - 1, "kit_frame": 10,
                         "platforms": {actor_id: {"motion_model_state": {"position": position_scene},
                                                   "reference_position_scene_units": position_scene}}}
        mission = {"assignments": {task_id: actor_id}, "discovered": [], "completed": [],
                   "mode": "search", "event_log": [], "strike_tasks": {},
                   "aircraft_available": {actor_id: True}}
        queues = {actor_id: [task_id]}
        public_projection = {
            "sim_time_s": decision_time,
            "aircraft": {actor_id: {
                "kind": "uav", "home_zone": 0, "available": True, "faulted": False,
                "safety_paused": False, "communication_held": False,
                "position_m": [-22000.0, -22000.0, 20000.0], "velocity_mps": [0.0, 0.0, 0.0],
                "available_energy_wh": 100.0, "active_task_ids": [task_id],
                "queue_task_ids": [task_id],
            }},
            "public_observations": [],
        }
        common = (
            queues, {actor_id: {task_id: route}}, {actor_id: [0, 0]}, {actor_id: []},
            {actor_id: 100.0}, {actor_id: [-22.0, -22.0, 20.0]},
            {actor_id: [-22.0, -22.0, 20.0]}, {actor_id: []}, set(), set(), set(),
            {}, [], [], {}, None, mission,
        )
        private_path = private_dir / "state_00_005399.pickle"
        private_dir.mkdir(parents=True)
        with private_path.open("wb") as handle:
            pickle.dump((backend_state, common), handle, protocol=pickle.HIGHEST_PROTOCOL)
        public_path = sampling_dir / "state_00_005399.public.json"
        _write_json(public_path, public_projection, compact=True)
        entry = {"sim_time_s": decision_time, "decision_time_s": decision_time,
                 "backend_time_s": decision_time - 1,
                 "continuation_phase": "post_public_updates_pre_candidate_generation_pre_step",
                 "decision_sequence": 3, "priority_sha256": "priority",
                 "private_file": "private_states/state_00_005399.pickle",
                 "private_sha256": _sha(private_path), "public_file": public_path.name,
                 "public_sha256": _sha(public_path)}
        index = {"schema": "isaac_roi_state_sampling_v1", "sampling_seed": 42,
                 "limit": 8, "selected_count": 1, "states": [entry],
                 "sampler_offer_wall_ns": 0, "sampler_offer_cpu_ns": 0}
        index_path = sampling_dir / "state_index.json"
        _write_json(index_path, index)
        index_hash = _sha(index_path)
        (sampling_dir / "state_index.sha256").write_text(index_hash + "  state_index.json\n", encoding="ascii")
        source_manifest = {"run_id": config["run_id"], "scenario_id": config["scenario_id"],
                           "split": config["split"], "config": config,
                           "config_sha256": config_hash}
        source_receipt = {
            "run_id": config["run_id"], "status": "complete", "actual_isaac_run": True,
            "config_sha256": config_hash, "evaluator_truth_sha256": _sha(truth_path),
            "formal_identity": {"scenario_id": "scenario-a", "split": "train"},
            "state_sampling": index, "meters_per_unit": 100.0,
            "expected_task_ids": [task_id], "public_task_manifest": [public_task],
            "actor_manifest": [actor], "zone_bounds_km": zone_bounds,
        }
        _write_json(source_dir / "manifest.json", source_manifest)
        _write_json(source_dir / "run_receipt.json", source_receipt)

        hashes = {"snapshot": _snapshot_digest((backend_state, common))}
        suffix_manifest = {"run_id": config["run_id"], "config": config,
                           "config_sha256": config_hash, "suffix_seed": 730201,
                           "logs": {"trajectory": "trajectory.jsonl.gz", "events": "events.jsonl"}}
        suffix_receipt = {
            "run_id": config["run_id"], "status": "complete", "actual_isaac_run": True,
            "config_sha256": config_hash,
            "formal_identity": {"scenario_id": "scenario-a", "split": "train"},
            "suffix": {
                "state_id": private_path.stem, "state_index": str(index_path.resolve()),
                "private_snapshot": str(private_path.resolve()),
                "private_snapshot_sha256": _sha(private_path), "config_sha256": config_hash,
                "source_manifest": str((source_dir / "manifest.json").resolve()),
                "source_manifest_config_sha256": config_hash, "sim_time_s": decision_time,
                "backend_time_s": decision_time - 1,
                "continuation_phase": entry["continuation_phase"], "seed": 730201,
                "horizon_steps": 2, "decision_time_s": decision_time,
                "snapshot_backend_time_s": decision_time - 1,
                "snapshot_sha256": hashes["snapshot"],
                "keep_restore_sha256": hashes["snapshot"],
                "replan_restore_sha256": hashes["snapshot"],
                "audit_interface": {"state_id": private_path.stem,
                                    "state_kind": "ordinary_sampled_decision_state", "forced": False,
                                    "initial_discovered_target_ids": [],
                                    "initial_serviced_target_ids": [],
                                    "initial_completed_public_task_ids": [],
                                    "initial_remaining_public_task_ids": [task_id]},
            },
        }
        _write_json(suffix_dir / "manifest.json", suffix_manifest)
        _write_json(suffix_dir / "run_receipt.json", suffix_receipt)
        rows = []
        for branch in ("keep", "replan"):
            for time_s in (decision_time, decision_time + 1):
                rows.append({
                    "schema_version": 3, "run_id": config["run_id"], "phase": "suffix_pair",
                    "branch": branch, "seed": config["seed"], "sim_time_s": time_s, "dt_s": 1.0,
                    "aircraft_id": actor_id, "aircraft_type": "uav", "zone": 0,
                    "task_id": task_id, "search_cell_visited": False, "task_completed": False,
                    "faulted": False, "target_discovered_ids": [], "detected_target_ids": [],
                    "mission_xy_km": [-22.0, -22.0],
                    "usd_position_m": [-22000.0, -22000.0, 20000.0],
                    "public_target_observations": [], "public_observation_ids": [],
                    "previous_usd_position_m": [-22000.0, -22000.0, 20000.0],
                    "energy": {"status": "analytical_linear", "incremental_wh": 0.0,
                               "remaining_wh": 100.0, "effective_wh_per_km": 240.0,
                               "anomaly_active": False, "safety_pause": False},
                    "communication": {"status": "simulated_command_queue",
                                      "command_delivered": False, "command_dropped": False,
                                      "forced_outage": False, "safety_hold": False,
                                      "command_attempt_accepted": True,
                                      "command_attempted": False, "delay_s": 2},
                    "hidden_truth_logged": False,
                })
        _write_trace(suffix_dir / "trajectory.jsonl.gz", rows)
        events = [{"schema_version": 1, "branch": "replan", "sim_time_s": decision_time,
                   "event": "candidate_generation", "decision_id": f"replan-decision-{decision_time}",
                   "timing": {"cpu_duration_ns": 1000, "wall_duration_ns": 2000}}]
        _write_jsonl(suffix_dir / "events.jsonl", events)
        return {"suffix_dir": suffix_dir, "source_index": index_path,
                "source_root": Path(__file__).resolve().parents[4],
                "receipt": suffix_receipt, "rows": rows, "events": events,
                "truth": truth, "actor_id": actor_id}

    def _report(self):
        case = self.case
        return evaluate_suffix(case["suffix_dir"], case["source_index"], case["source_root"],
                               self._collector_root())

    @staticmethod
    def _collector_root():
        return Path(__file__).resolve().parents[1] / "formal_collection_v1"

    def test_independent_snapshot_and_position_replay_produces_bound_gain(self):
        report = self._report()
        self.assertEqual(report["status"], "PASS", report.get("reasons"))
        self.assertTrue(report["valid_mask"])
        self.assertEqual(report["snapshot_sha256"], self.case["receipt"]["suffix"]["snapshot_sha256"])
        self.assertEqual(report["horizon_steps"], 2)
        self.assertAlmostEqual(report["G"], -0.00050001, places=8)

    def test_restore_hash_mismatch_is_rejected(self):
        self.case["receipt"]["suffix"]["replan_restore_sha256"] = "wrong"
        _write_json(self.case["suffix_dir"] / "run_receipt.json", self.case["receipt"])
        report = self._report()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("restore hashes", report["reasons"][0])

    def test_trace_cannot_shift_its_first_physical_time(self):
        rows = self.case["rows"]
        rows[0]["sim_time_s"] += 2
        _write_trace(self.case["suffix_dir"] / "trajectory.jsonl.gz", rows)
        report = self._report()
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("outside frozen suffix window" in reason for reason in report["reasons"]))

    def test_schedule_none_falls_back_to_legacy_pilot_schedule(self):
        from evaluate_suffix import _activation_schedule

        schedule = _activation_schedule({
            "evaluator_schedule": None,
            "pilot": {"activation_task_ids": ["z0-cell0"], "activation_times_s": [600]},
        }, {"z0-cell0", "z0-cell1"})
        self.assertEqual(schedule["z0-cell0"], 600.0)
        self.assertEqual(schedule["z0-cell1"], 0.0)

    def test_suffix_formal_identity_must_match_source(self):
        receipt = self.case["receipt"]
        receipt["formal_identity"]["split"] = "validation"
        _write_json(self.case["suffix_dir"] / "run_receipt.json", receipt)
        report = self._report()
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("suffix split metadata is inconsistent" in reason
                            for reason in report["reasons"]))

    def test_source_config_hash_aliases_must_agree(self):
        receipt = self.case["receipt"]
        receipt["suffix"]["source_config_sha256"] = "wrong"
        _write_json(self.case["suffix_dir"] / "run_receipt.json", receipt)
        report = self._report()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("source config SHA256 aliases disagree", report["reasons"][0])

    def test_cli_loads_event_trigger_gate_snapshot_in_fresh_process(self):
        case = self.case
        collector_root = self._collector_root()
        collector_path = str(collector_root)
        sys.path.insert(0, collector_path)
        try:
            from mission import EventTriggerGate

            private_path = case["source_index"].parent / "private_states" / "state_00_005399.pickle"
            with private_path.open("rb") as handle:
                backend, common = pickle.load(handle)
            backend["fixture_event_trigger_gate"] = EventTriggerGate(threshold=2)
            with private_path.open("wb") as handle:
                pickle.dump((backend, common), handle, protocol=pickle.HIGHEST_PROTOCOL)
            snapshot_hash = _snapshot_digest((backend, common))
        finally:
            if collector_path in sys.path:
                sys.path.remove(collector_path)
            sys.modules.pop("mission", None)

        index_path = case["source_index"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["states"][0]["private_sha256"] = _sha(private_path)
        _write_json(index_path, index)
        index_hash = _sha(index_path)
        index_path.with_suffix(".sha256").write_text(index_hash + "  state_index.json\n", encoding="ascii")

        source_receipt_path = index_path.parent.parent / "run_receipt.json"
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        source_receipt["state_sampling"] = index
        _write_json(source_receipt_path, source_receipt)
        suffix_receipt = case["receipt"]
        suffix = suffix_receipt["suffix"]
        suffix["private_snapshot_sha256"] = _sha(private_path)
        for key in ("snapshot_sha256", "keep_restore_sha256", "replan_restore_sha256"):
            suffix[key] = snapshot_hash
        _write_json(case["suffix_dir"] / "run_receipt.json", suffix_receipt)

        output = self.root / "cli_reduction.json"
        command = [sys.executable, str(Path(__file__).with_name("evaluate_suffix.py")),
                   "--suffix-dir", str(case["suffix_dir"]),
                   "--source-index", str(index_path),
                   "--source-root", str(case["source_root"]),
                   "--collector-root", str(collector_root), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS", report.get("reasons"))

    def test_public_observation_cannot_appear_before_available_event(self):
        rows = self.case["rows"]
        rows[0]["public_target_observations"] = [{
            "target_id": "target-z0-0", "kind": "uav", "zone": 0,
            "target_xy_km": [-18.0, -18.0], "detected_at_sim_time_s": 5398,
            "available_at_sim_time_s": 5400, "observation_id": "future-obs",
        }]
        _write_trace(self.case["suffix_dir"] / "trajectory.jsonl.gz", rows)
        report = self._report()
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("public observation view" in reason for reason in report["reasons"]))

    def test_duplicate_suffix_seed_does_not_form_state_label(self):
        reduction_root = self.root / "reductions"
        self._write_reduction(reduction_root, "scenario-a::state-1", "state-1", 730201, "train", 1.0)
        self._write_reduction(reduction_root, "scenario-a::state-1", "state-1", 730201, "train", 3.0)
        result = aggregate(reduction_root, reduction_root / "aggregate.json")
        self.assertEqual(result["valid_state_count"], 0)
        self.assertIn("suffix_seeds_must_be_distinct", result["states"][0]["unknown_reason"])

    def test_same_state_id_crossing_splits_is_rejected(self):
        reduction_root = self.root / "reductions"
        self._write_reduction(reduction_root, "scenario-a::state-1", "state-1", 730201, "train", 1.0)
        self._write_reduction(reduction_root, "scenario-a::state-1", "state-1", 730202, "validation", 3.0)
        result = aggregate(reduction_root, reduction_root / "aggregate.json")
        self.assertEqual(result["valid_state_count"], 0)
        self.assertIn("cross_split_duplicate_state_id", result["states"][0]["unknown_reason"])

    def _write_reduction(self, root, state_id, local_id, seed, split, gain):
        path = root / "scenario-a" / local_id / split / f"suffix_{seed}" / "reduction.json"
        _write_json(path, {
            "status": "PASS", "valid_mask": True, "fixture_only": False, "forced": False,
            "scenario_id": "scenario-a", "local_state_id": local_id,
            "state_id": state_id, "split": split, "suffix_seed": seed,
            "horizon_steps": 2, "snapshot_sha256": "snapshot",
            "config_sha256": "config", "G": gain,
        })


def _write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
