from __future__ import annotations

import unittest

import numpy as np
import torch

from closedloop_v1.runtime import _planning_timing, _select_catalogue_action
from wm_ppo_v1.candidate_features import FEATURE_NAMES as CANDIDATE_FEATURE_NAMES
from wm_ppo_v1.features import FEATURE_NAMES as NODE_FEATURE_NAMES
from wm_ppo_v1.policy_choice import PolicyUnavailable, select_public_candidate
from wm_ppo_v1.ppo_core import MaskedCandidateActorCritic
from wm_ppo_v1.public_dynamics import public_projection_to_frame


def _projection(time_s: int) -> dict:
    return {"sim_time_s": time_s, "aircraft": {
        "donor": {"kind": "uav", "home_zone": 1, "available": True, "faulted": False,
                  "safety_paused": False, "communication_held": False,
                  "position_m": [1000.0 + time_s, 2000.0, 300.0], "velocity_mps": [1, 0, 0],
                  "available_energy_wh": 80.0, "active_task_ids": ["task-a"],
                  "queue_task_ids": ["task-a", "task-b"]},
        "recipient": {"kind": "uav", "home_zone": 2, "available": True, "faulted": False,
                      "safety_paused": False, "communication_held": False,
                      "position_m": [5000.0, 6000.0, 400.0], "velocity_mps": [0, 1, 0],
                      "available_energy_wh": 100.0, "active_task_ids": [],
                      "queue_task_ids": ["task-c"]}}, "public_observations": []}


def _catalog() -> list[dict]:
    return [{"kind": "uav_transfer", "donor": "donor", "recipient": "recipient",
             "task_id": "task-b", "estimated_gain_s": 5.0},
            {"kind": "noop", "estimated_gain_s": 0.0}]


def _normalization() -> dict:
    return {"scope": "train_split_only", "node_feature_names": list(NODE_FEATURE_NAMES),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "node_center": np.zeros(len(NODE_FEATURE_NAMES), dtype=np.float32),
            "node_scale": np.ones(len(NODE_FEATURE_NAMES), dtype=np.float32),
            "candidate_center": np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32),
            "candidate_scale": np.ones(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)}


class PolicyChoiceSyntheticTests(unittest.TestCase):
    def test_evaluation_tie_uses_stable_key_and_maps_to_legal_original_action(self):
        model = MaskedCandidateActorCritic(candidate_dim=len(CANDIDATE_FEATURE_NAMES), hidden_dim=8)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        projection = _projection(2)
        history = (public_projection_to_frame(_projection(1)),
                   public_projection_to_frame(projection))
        choice = select_public_candidate(
            model, projection, history, _catalog(), normalization=_normalization(),
            model_version="synthetic:zeros-v1", mode="evaluate", device="cpu",
            synthetic_cpu_fixture=True)
        self.assertEqual(choice["status"], "selected")
        self.assertEqual(choice["key"], ("noop",))
        self.assertEqual(choice["index"], 1)
        self.assertEqual(choice["action"], _catalog()[1])
        self.assertEqual(choice["legal_mask"], [True, True])
        self.assertEqual(choice["noop_mask"], [False, True])
        self.assertAlmostEqual(choice["old_logprob"], -np.log(2.0), places=6)
        self.assertTrue(np.isfinite(choice["value"]))
        self.assertEqual(choice["device"], "cpu")

    def test_sampling_is_masked_and_cpu_requires_explicit_synthetic_mode(self):
        model = MaskedCandidateActorCritic(candidate_dim=len(CANDIDATE_FEATURE_NAMES), hidden_dim=8)
        projection = _projection(2)
        history = (public_projection_to_frame(_projection(1)), public_projection_to_frame(projection))
        with self.assertRaisesRegex(RuntimeError, "SYNTHETIC_FIXTURE_ONLY"):
            select_public_candidate(model, projection, history, _catalog(),
                normalization=_normalization(), model_version="untrained-v1", mode="sample", device="cpu")
        choice = select_public_candidate(model, projection, history, _catalog(),
            normalization=_normalization(), model_version="synthetic:random-v1", mode="sample",
            device="cpu", synthetic_cpu_fixture=True,
            generator=torch.Generator().manual_seed(4))
        self.assertTrue(choice["legal_mask"][choice["index"]])
        self.assertEqual(choice["action"], _catalog()[choice["index"]])
        self.assertTrue(np.isfinite(choice["old_logprob"]))

    def test_history_unavailable_is_explicit_and_not_a_model_error(self):
        with self.assertRaises(PolicyUnavailable):
            select_public_candidate(object(), _projection(2), (), _catalog(),
                normalization=_normalization(), model_version="synthetic:test", mode="evaluate",
                device="cpu", synthetic_cpu_fixture=True)

    def test_forced_or_suppressed_gate_never_calls_ordinary_policy(self):
        calls = []
        def policy(**kwargs):
            calls.append(kwargs)
            raise AssertionError("must not run")
        forced = _select_catalogue_action(policy, {}, (), _catalog(), triggered=True, forced_safety=True)
        suppressed = _select_catalogue_action(policy, {}, (), _catalog(), triggered=False,
                                              gate_suppressed=True)
        self.assertEqual(forced["source"], "forced_safety_local")
        self.assertEqual(suppressed["source"], "gate_suppressed")
        self.assertEqual(calls, [])

    def test_policy_mapping_errors_fail_and_only_known_unavailability_is_soft(self):
        action = _catalog()[1]
        valid_result = {"status": "selected", "index": 1, "key": ("noop",), "action": action,
                        "legal_mask": [True, True], "model_version": "synthetic:test"}
        selected = _select_catalogue_action(lambda **kwargs: valid_result, {}, (), _catalog(), triggered=True)
        self.assertEqual(selected["action"], action)
        with self.assertRaisesRegex(ValueError, "MAPPING_MISMATCH"):
            _select_catalogue_action(lambda **kwargs: {**valid_result, "action": _catalog()[0]},
                                     {}, (), _catalog(), triggered=True)
        unavailable = _select_catalogue_action(
            lambda **kwargs: (_ for _ in ()).throw(PolicyUnavailable("POLICY_HISTORY_UNAVAILABLE")),
            {}, (), _catalog(), triggered=True)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["reason"], "POLICY_HISTORY_UNAVAILABLE")
        self.assertIn("wall_duration_ns", unavailable["call_timing"])

    def test_catalog_and_policy_costs_are_combined_once(self):
        timing = _planning_timing({"planner_wall_duration_ns": 11, "planner_cpu_duration_ns": 5},
                                  {"call_timing": {"wall_duration_ns": 7, "cpu_duration_ns": 3}})
        self.assertEqual(timing["wall_duration_ns"], 18)
        self.assertEqual(timing["cpu_duration_ns"], 8)
        self.assertEqual(timing["policy_selection_wall_duration_ns"], 7)


if __name__ == "__main__":
    unittest.main()
