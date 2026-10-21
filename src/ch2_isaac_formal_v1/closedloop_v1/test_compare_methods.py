from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from closedloop_v1.compare_methods import _holm, compare


def _reduction(*, run_id: str, cost: float, completed: int = 8, violations: int = 0) -> dict:
    return {
        "schema": "closedloop_episode_reduction_v1", "status": "PASS", "valid_mask": True,
        "scope": {"run_id": run_id, "duration_s": 5400, "branch": "representative"},
        "tasks": {"catalog_count": 10, "completed_count": completed},
        "discovery": {"target_count": 10, "discovered_count": 7},
        "decision_effect_delays": {"status": "incomplete", "effect_delays": "unknown"},
        "safety": {"controllable_violation_events": violations, "hold_motion_violation_check": True,
                   "all_forced_safety_actions_applied": True},
        "checks": {"stationary_hold_positions": True},
        "planning": {"candidate_generation_calls": 4},
        "registered_objective": {"absolute_online_episode_cost": {
            "status": "measured", "measured_online_method_cost": {"status": "measured", "value": cost}}},
    }


class CompareMethodsTests(unittest.TestCase):
    def test_paired_scenes_aggregate_training_seeds_before_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            rows = []
            for index in range(4):
                scene = f"scene-{index}"
                common = {"scenario_id": scene, "split": "test", "condition": "common",
                          "parent_family": f"family-{index}"}
                control_path = f"{scene}-control.json"
                control_id = f"{scene}-primary"
                (root / control_path).write_text(json.dumps(_reduction(run_id=control_id, cost=1.0)), encoding="utf-8")
                rows.append({**common, "run_id": control_id, "method": "primary", "reduction_path": control_path})
                for training_seed in (1, 2):
                    method_path = f"{scene}-method-{training_seed}.json"
                    run_id = f"{scene}-candidate-{training_seed}"
                    (root / method_path).write_text(json.dumps(_reduction(run_id=run_id, cost=0.5)), encoding="utf-8")
                    rows.append({**common, "run_id": run_id, "method": "candidate", "training_seed": training_seed,
                                 "reduction_path": method_path})
            registry = root / "registry.json"
            registry.write_text(json.dumps({"runs": rows}), encoding="utf-8")
            result = compare(registry, root, "primary", bootstrap_replicates=10000)
            candidate = result["splits"]["test"]["method_comparisons"]["candidate"]
            self.assertEqual(candidate["status"],
                "COMPLETE_NUMERICAL_ASSESSMENT_REQUIRES_PRE_REGISTERED_DECISION_REVIEW")
            self.assertTrue(candidate["decision_rule"]["thresholds_met_numerically"])
            self.assertTrue(candidate["decision_rule"]["numerical_thresholds_are_not_support_decisions"])
            self.assertEqual(candidate["paired_favorable_differences"]["online_method_cost"]["n_scenes"], 4)
            self.assertAlmostEqual(candidate["paired_favorable_differences"]["online_method_cost"]["mean_difference"], 0.5)
            self.assertEqual(candidate["paired_favorable_differences"]["mean_effect_delay_s"]["n_scenes"], 0)
            self.assertEqual(candidate["per_method_episode_metrics"]["online_method_cost"]["sample_sd"], 0.0)

    def test_duplicate_seed_and_missing_reduction_make_comparison_inconclusive(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            rows = []
            for method, run_id, seed in (("primary", "p", None), ("candidate", "c1", 7),
                                         ("candidate", "c2", 7)):
                file = f"{run_id}.json"
                if run_id != "c2":
                    (root / file).write_text(json.dumps(_reduction(run_id=run_id, cost=1.0)), encoding="utf-8")
                row = {"scenario_id": "same-scene", "split": "test", "method": method,
                       "run_id": run_id, "reduction_path": file}
                if seed is not None:
                    row["training_seed"] = seed
                rows.append(row)
            registry = root / "registry.json"
            registry.write_text(json.dumps({"runs": rows}), encoding="utf-8")
            result = compare(registry, root, "primary", bootstrap_replicates=10)
            candidate = result["splits"]["test"]["method_comparisons"]["candidate"]
            self.assertEqual(candidate["status"], "INCONCLUSIVE")
            self.assertTrue(any(group["reason"] == "duplicate_training_seed"
                                for group in candidate["incomplete_scene_method_groups"]))
            self.assertTrue(any(group["reason"] == "reduction_missing_or_invalid"
                                for group in candidate["incomplete_scene_method_groups"]))

    def test_holm_step_down_adjustment(self):
        adjusted = _holm({"a": 0.01, "b": 0.04, "c": 0.03, "missing": None})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.06, "b": 0.06, "missing": None})


if __name__ == "__main__":
    unittest.main()
