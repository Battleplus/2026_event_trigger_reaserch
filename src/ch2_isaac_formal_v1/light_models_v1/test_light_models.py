from __future__ import annotations

import hashlib
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    from .data_gate import check_feature_coverage, inspect_batch_output
    from .dataset import load_feature_dataset
    from .features import FEATURE_SCHEMA, extract_features
except ImportError:
    from data_gate import check_feature_coverage, inspect_batch_output
    from dataset import load_feature_dataset
    from features import FEATURE_SCHEMA, extract_features


@contextmanager
def _workspace_tmp():
    path = Path(__file__).resolve().parent / f"_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _receipt(root: Path, *, terminal: bool = True) -> dict:
    scenes = [{"run_id": f"scene-{index:03d}", "output_path": f"scenes/scene-{index:03d}",
               "execution": "COMPLETE", "representative": None, "states": []}
              for index in range(168)]
    return {
        "schema": "isaac_roi_formal_batch_receipt_v1",
        "execution": "COMPLETE" if terminal else "NOT_RUN",
        "completed_at_utc": "2026-10-21T00:00:00+00:00" if terminal else None,
        "scene_count": 168,
        "scenes": scenes,
        "end_reduction": {"status": "failed", "aggregate_output": str(root / "aggregate.json")},
    }


def _aggregate(states=None) -> dict:
    return {
        "schema_version": 1, "status": "PASS", "state_count": 60, "valid_state_count": 60,
        "positive_count": 30, "nonpositive_count": 30, "count_30_30_met": True,
        "split_coverage_met": True, "met_30_30": True, "excluded_state_count": 0,
        "states": [] if states is None else states,
    }


class LightModelInterfaceTests(unittest.TestCase):
    def test_nonterminal_batch_stops_before_aggregate(self):
        with _workspace_tmp() as root:
            (root / "batch_receipt.json").write_text(json.dumps(_receipt(root, terminal=False)), encoding="utf-8")
            (root / "aggregate.json").write_text("not-json", encoding="utf-8")
            result = inspect_batch_output(root)
            self.assertEqual(result["status"], "NOT_RUN")
            self.assertEqual(result["reason"], "BATCH_NOT_TERMINAL")

    def test_terminal_receipt_and_summary_gate(self):
        with _workspace_tmp() as root:
            (root / "batch_receipt.json").write_text(json.dumps(_receipt(root)), encoding="utf-8")
            (root / "aggregate.json").write_text(json.dumps(_aggregate()), encoding="utf-8")
            result = inspect_batch_output(root)
            self.assertTrue(result["ready"])
            self.assertEqual(result["aggregate_summary"]["positive_count"], 30)

    def test_fixed_public_feature_schema_contains_no_identifiers(self):
        projection = {
            "sim_time_s": 120,
            "aircraft": {
                "uav-0-0": {"kind": "uav", "home_zone": 0, "available": True,
                             "faulted": False, "safety_paused": False, "communication_held": False,
                             "position_m": [10, 20, 30], "velocity_mps": [3, 4, 0],
                             "available_energy_wh": 90000, "active_task_ids": ["z0-cell0"],
                             "queue_task_ids": ["z0-cell1"]},
                "heli-0-0": {"kind": "helicopter", "home_zone": 0, "available": False,
                              "faulted": True, "position_m": [30, 20, 30], "velocity_mps": [0, 0, 0],
                              "available_energy_wh": 150000, "active_task_ids": [], "queue_task_ids": []},
            },
            "public_observations": [{"target_id": "hidden-id", "kind": "uav", "zone": 0,
                                      "target_xy_km": [1, 2], "observed_at_sim_time_s": 119,
                                      "available_at_sim_time_s": 120}],
        }
        features = extract_features(projection)
        self.assertEqual(list(features), FEATURE_SCHEMA)
        self.assertTrue(all(isinstance(value, float) for value in features.values()))
        self.assertTrue(all("id" not in name.lower() for name in features))

    def test_feature_coverage_is_recomputed_after_exclusion(self):
        rows = []
        for label, target in (("positive", 1.0), ("nonpositive", -1.0)):
            for split in ("train", "validation"):
                rows.extend({"target": target, "split": split, "excluded": False,
                             "scenario_id": f"{split}-{label}-{index}"}
                            for index in range(15))
        rows.append({"target": 1.0, "split": "train", "excluded": True})
        coverage = check_feature_coverage(rows)
        self.assertTrue(coverage["met_30_30"])
        self.assertEqual(coverage["excluded_count"], 1)

    def test_dataset_binds_public_projection_by_hash(self):
        with _workspace_tmp() as root:
            receipt = _receipt(root)
            receipt["scenes"][0]["scenario_id"] = "scenario-0"
            receipt["scenes"][0]["split"] = "train"
            state_id = "state_00_000100"
            scene_root = root / "scenes" / "scene-000" / "representative" / "state_sampling"
            scene_root.mkdir(parents=True)
            projection = {"sim_time_s": 100, "aircraft": {}, "public_observations": []}
            public_bytes = (json.dumps(projection, sort_keys=True) + "\n").encode("utf-8")
            (scene_root / f"{state_id}.public.json").write_bytes(public_bytes)
            (scene_root / "state_index.json").write_text(json.dumps({"states": [{
                "private_file": f"private_states/{state_id}.pickle",
                "public_file": f"{state_id}.public.json",
                "public_sha256": hashlib.sha256(public_bytes).hexdigest(),
            }]}), encoding="utf-8")
            state = {"state_id": "scenario-0::state_00_000100", "scenario_id": "scenario-0",
                     "local_state_id": state_id, "split": "train", "valid_mask": True,
                     "gain_mean": 1.0, "source_reports": [
                         f"scene-000/{state_id}/suffix_730201/reduction.json",
                         f"scene-000/{state_id}/suffix_730202/reduction.json"]}
            (root / "batch_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            (root / "aggregate.json").write_text(json.dumps(_aggregate([state])), encoding="utf-8")
            result = load_feature_dataset(root)
            self.assertTrue(result["gate"]["ready"])
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(result["rows"][0]["target"], 1.0)


if __name__ == "__main__":
    unittest.main()
