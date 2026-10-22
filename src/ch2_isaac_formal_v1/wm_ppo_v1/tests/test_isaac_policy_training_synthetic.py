"""Synthetic-only gate, reward attribution, PPO update and restore checks."""
from __future__ import annotations

import gzip
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
import uuid

import numpy as np
import torch

from wm_ppo_v1.candidate_features import FEATURE_NAMES as CANDIDATE_NAMES
from wm_ppo_v1.contracts import PublicHistoryFrame
from wm_ppo_v1.isaac_policy_training import (
    _update_from_episode,
    attribute_decision_intervals,
    check_training_gates,
    reconstruct_tick_rewards,
    restore_g_checkpoint,
    save_g_checkpoint,
)
from wm_ppo_v1.ppo_core import (MaskedCandidateActorCritic, PPOCore, _masked_categorical,
                                compute_time_aware_gae)
from closedloop_v1.evaluate_episode import evaluate_run


class IsaacPolicyTrainingSyntheticTests(unittest.TestCase):
    def _fixture_dir(self):
        root = Path(__file__).resolve().parent / f"fixture-isaac-ppo-{uuid.uuid4().hex}"
        root.mkdir()
        return root

    @staticmethod
    def _remove_fixture(root: Path):
        shutil.rmtree(root)

    def test_gate_refuses_before_any_runtime_setup(self):
        fake = {"gate": {"ready": False, "reason": "BATCH_NOT_TERMINAL"}, "coverage": None, "rows": []}
        with patch("light_models_v1.dataset.load_feature_dataset", return_value=fake):
            _, receipt = check_training_gates("unused")
        self.assertEqual(receipt.status, "REFUSED")
        self.assertIn("BATCH_NOT_TERMINAL", receipt.reasons)

    def test_two_decision_gae_uses_next_value_bootstrap(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 1.0])
        next_values = torch.tensor([1.0, 0.0])
        advantages, returns = compute_time_aware_gae(rewards, values, next_values,
            torch.tensor([1.0, 1.0]), torch.tensor([False, True]), torch.tensor([False, False]),
            gamma=1.0, gae_lambda=1.0)
        self.assertEqual(advantages.tolist(), [2.5, 1.0])
        self.assertEqual(returns.tolist(), [3.0, 2.0])

    def test_tick_rewards_and_decision_intervals_match_registered_cost(self):
        root = self._fixture_dir()
        try:
            config = {"mode": "search", "representative_duration_s": 1, "sensor_radius_km": 0.5,
                "waypoint_tolerance_km": 0.1, "energy_wh_per_km": {"uav": 240.0},
                "energy_capacity_wh": {"uav": 100000.0}, "energy_reserve_fraction": 0.1,
                "mechanisms": {"platform_fault": {"enabled": False},
                               "communication_outage": {"enabled": False}},
                "evaluator_schedule": {"disturbances": {}}}
            actors = [{"aircraft_id": "uav-0-0", "aircraft_type": "uav", "home_zone": 0}]
            tasks = [{"task_id": "task-0", "allowed_kind": "uav", "zone": 0,
                      "sweep_waypoints_km": [[10.0, 10.0]]}]
            manifest = {"run_id": "synthetic-reducer-episode", "evaluation_config": config,
                "actor_manifest": actors, "public_task_manifest": tasks,
                "initial_positions_m": {"uav-0-0": [0.0, 0.0, 0.0]},
                "zone_bounds_km": {}, "logs": {"trajectory": "trajectory.jsonl.gz", "events": "events.jsonl"},
                "evaluation_truth_file": "evaluator_truth.json"}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            row = {"schema_version": 3, "run_id": manifest["run_id"], "phase": "representative",
                "branch": "representative", "sim_time_s": 1, "aircraft_id": "uav-0-0",
                "aircraft_type": "uav", "zone": 0, "task_id": "task-0", "search_cell_visited": False,
                "task_completed": False, "faulted": False, "target_discovered_ids": [],
                "mission_xy_km": [0.0, 0.0], "usd_position_m": [0.0, 0.0, 0.0],
                "previous_usd_position_m": [0.0, 0.0, 0.0], "velocity_mps": [0.0, 0.0, 0.0],
                "energy": {"incremental_wh": 0.0, "safety_pause": False},
                "communication": {"safety_hold": False}}
            with gzip.open(root / "trajectory.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            phases = {name: {"wall_duration_ns": 0, "cpu_duration_ns": 0} for name in (
                "event_state_safety", "projection_policy_planning", "command_ready",
                "post_step_online_state", "simulator_step", "post_step_evaluator_telemetry",
                "raw_artifact_logging")}
            events = [
                {"sim_time_s": 1, "event": "transmission_sent"},
                {"sim_time_s": 1, "event": "controller_switch"},
                {"sim_time_s": 1, "event": "candidate_generation", "timing": {"cpu_duration_ns": 1_000_000_000}},
                {"sim_time_s": 1, "event": "donor_support_transfer", "from_aircraft_id": "a", "to_aircraft_id": "b"},
                {"sim_time_s": 1, "event": "decision_created", "decision_id": "d1", "decision_created_sim_time_s": 1},
                {"sim_time_s": 1, "event": "decision_timing_finalized", "decision_id": "d1", "decision_complete_sim_time_s": 1},
                {"sim_time_s": 1, "event": "online_timing_tick", "timing": {"phases": phases}},
            ]
            (root / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
            (root / "evaluator_truth.json").write_text(json.dumps({"targets": [
                {"target_id": "target-0", "zone": 0, "target_xy_km": [1.0, 1.0]}]}), encoding="utf-8")
            (root / "run_receipt.json").write_text(json.dumps({"closedloop": {
                "runtime_semantics_id": "closedloop_v1_fault_release_accounting_v2"}}), encoding="utf-8")
            reduction = evaluate_run(root)
            self.assertEqual(reduction["status"], "PASS", reduction.get("reasons"))
            tick_rewards, prefix = reconstruct_tick_rewards(root, reduction)
            subtotal = reduction["registered_objective"]["absolute_online_episode_cost"][
                "registered_planner_only_subtotal"]["value"]
            self.assertAlmostEqual(sum(tick_rewards), -subtotal, places=10)
            intervals = attribute_decision_intervals(tick_rewards, [1], duration_s=1)
            self.assertAlmostEqual(prefix, intervals["prefix_reward"], places=10)
            self.assertAlmostEqual(intervals["prefix_reward"]
                                   + sum(item["reward"] for item in intervals["intervals"]),
                                   -subtotal, places=10)
            self.assertEqual(intervals["intervals"][0]["start_s"], 1)
            self.assertEqual(intervals["intervals"][0]["stop_exclusive_s"], 2)
        finally:
            self._remove_fixture(root)

    def test_cpu_episode_update_and_g_checkpoint_restore(self):
        torch.manual_seed(19)
        model = MaskedCandidateActorCritic(candidate_dim=len(CANDIDATE_NAMES), hidden_dim=8)
        core = PPOCore(model, device="cpu", learning_rate=1e-2)
        histories = [
            (PublicHistoryFrame(time_s=1.0, platforms=({"x_norm": 0.2, "available": True},)),),
            (PublicHistoryFrame(time_s=2.0, platforms=({"x_norm": 0.7, "available": True},)),),
        ]
        normalization = {"node_center": [0.0] * 28, "node_scale": [1.0] * 28,
            "candidate_center": [0.0] * len(CANDIDATE_NAMES),
            "candidate_scale": [1.0] * len(CANDIDATE_NAMES)}
        from wm_ppo_v1.isaac_policy_training import _torch_graph_batch
        graph = _torch_graph_batch(histories, normalization, torch.device("cpu"))
        candidate = torch.zeros((2, 2, len(CANDIDATE_NAMES)))
        candidate[:, 1, 0] = 1.0
        legal_mask = torch.ones((2, 2), dtype=torch.bool)
        actions = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            output = model(graph, candidate)
            old_logprobs = _masked_categorical(output["logits"], legal_mask).log_prob(actions)
            old_values = output["value"].clone()
        records = []
        for i, history in enumerate(histories):
            features = candidate[i].numpy()
            records.append({"history": history, "catalog": ({"kind": "noop"}, {"kind": "search"}),
                "candidate_features": features.tolist(), "action_index": int(actions[i]),
                "legal_mask": [True, True], "noop_mask": [True, False],
                "old_logprob": old_logprobs[i].item(), "value": old_values[i].item()})
        actor_before = [p.detach().clone() for p in model.actor.parameters()]
        critic_before = [p.detach().clone() for p in model.value.parameters()]
        result = _update_from_episode(model, core, records,
            {"intervals": [{"reward": -1.0, "elapsed_s": 1},
                           {"reward": 2.0, "elapsed_s": 1}]},
            normalization, torch.device("cpu"), epochs=1)
        self.assertEqual(result["updates"], 1)
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(actor_before, model.actor.parameters())))
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(critic_before, model.value.parameters())))
        metadata = {"method": "G", "normalization": {"scope": "train_split_only"}}
        root = self._fixture_dir()
        try:
            checkpoint = root / "g.pt"
            save_g_checkpoint(checkpoint, model, core.optimizer, metadata)
            restored = MaskedCandidateActorCritic(candidate_dim=len(CANDIDATE_NAMES), hidden_dim=8)
            restored_core = PPOCore(restored, device="cpu")
            receipt = restore_g_checkpoint(checkpoint, restored, restored_core.optimizer)
        finally:
            self._remove_fixture(root)
        self.assertEqual(receipt["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
