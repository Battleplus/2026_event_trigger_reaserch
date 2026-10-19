"""Engineering-only synthetic probe of the G-WM critic tensor interface."""

from __future__ import annotations

import unittest
import copy

import torch

from graph_policy import build_policy
from n2_stage_pipeline import _policy_proposals
from relief_env import ReliefEnv
from scenario_matrix import build_parent_matrix
from stable_identity import key_digest
from world_model import (
    CriticAugmentedGraphPolicy,
    CriticOnlyGppoAdapter,
    DynamicConsequenceModel,
    GLOBAL_HEAD_NAMES,
    assert_paired_actor_initialization,
    public_policy_tensors,
)


EXPECTED_GLOBAL_HEAD_NAMES = (
    "other_order_lateness",
    "unmet_demand",
    "energy",
    "reassignment",
    "global_utility_delta",
)


class SyntheticCriticInterfaceTests(unittest.TestCase):
    def test_100x100_public_action_path_reaches_detached_auxiliary_critic(self):
        torch.set_num_threads(1)
        torch.manual_seed(8201)
        self.assertEqual(tuple(GLOBAL_HEAD_NAMES), EXPECTED_GLOBAL_HEAD_NAMES)
        scenario = build_parent_matrix()["train"][0]
        self.assertIs(scenario["synthetic"], True)
        self.assertEqual(scenario["map"]["rows"], 100)
        self.assertEqual(scenario["map"]["cols"], 100)

        env = ReliefEnv(scenario)
        env.reset(policy="LEARNED")
        policy_input = env.policy_input()
        self.assertEqual(policy_input["observation"]["map"]["rows"], 100)
        self.assertEqual(policy_input["observation"]["map"]["cols"], 100)

        base_g = build_policy("L1", 8201, device="cpu")
        base_gwm = build_policy("L1", 8201, device="cpu")
        world_model = DynamicConsequenceModel()
        policy_g = CriticAugmentedGraphPolicy(
            base_g, world_model, CriticOnlyGppoAdapter(), wm_enabled=False
        )
        policy_gwm = CriticAugmentedGraphPolicy(
            base_gwm, world_model, CriticOnlyGppoAdapter(), wm_enabled=True
        )
        policy_input["proposal_batches"] = _policy_proposals(base_gwm, env, policy_input)
        proposals = policy_input["proposal_batches"]
        expected_candidate_ids = [key_digest(row["actions"]) for row in proposals]
        self.assertGreaterEqual(len(proposals), 2)
        self.assertEqual(len(expected_candidate_ids), len(set(expected_candidate_ids)))
        self.assertEqual(
            [row["candidate_id"] for row in proposals], expected_candidate_ids
        )
        self.assertTrue(all(
            [mask["selected_id"] for mask in row["masks"]] == row["actions"]
            for row in proposals
        ))

        tensors, binding = public_policy_tensors(policy_input)
        self.assertEqual(binding.candidate_ids[0], tuple(expected_candidate_ids))
        self.assertEqual(int(tensors["candidate_mask"].sum().item()), len(proposals))
        assert_paired_actor_initialization(policy_g, policy_gwm, policy_input)

        attention_calls = []
        task_update_calls = []
        attention_hook = base_gwm.literal_attention.register_forward_hook(
            lambda _module, _inputs, _output: attention_calls.append(True)
        )
        task_update_hook = base_gwm.task_update.register_forward_hook(
            lambda _module, _inputs, _output: task_update_calls.append(True)
        )
        try:
            _, original_logits, _ = base_gwm(policy_input)
            changed_input = copy.deepcopy(policy_input)
            changed_task = changed_input["observation"]["tasks"][0]
            changed_task["position"][0] = min(99.0, changed_task["position"][0] + 7.0)
            _, changed_logits, _ = base_gwm(changed_input)
        finally:
            attention_hook.remove()
            task_update_hook.remove()
        self.assertGreaterEqual(len(attention_calls), 2)
        self.assertGreaterEqual(len(task_update_calls), 2)
        self.assertFalse(torch.allclose(original_logits, changed_logits))

        dispatch = next(row for row in proposals if row["actions"] != ["NOOP"])
        actions = [action for action in dispatch["actions"] if action != "NOOP"]
        masks = dispatch["masks"]
        g_result = policy_g.evaluate_batch(policy_input, actions, masks)
        gwm_initial = policy_gwm.evaluate_batch(policy_input, actions, masks)
        self.assertTrue(torch.equal(g_result["value"], gwm_initial["value"]))
        self.assertEqual(gwm_initial["proposal_binding"]["candidate_ids"], expected_candidate_ids)
        self.assertEqual(
            gwm_initial["proposal_binding"]["wm_prediction_cache"]["forward_batches"], 1
        )
        self.assertEqual(
            gwm_initial["proposal_binding"]["wm_prediction_cache"]["candidate_evaluations"],
            len(proposals),
        )

        sample = {"actions": actions, "masks": masks}
        proposal_log_probs, wm_predictions, metadata = policy_gwm._proposal_scores(
            policy_input, sample
        )
        self.assertIsNotNone(wm_predictions)
        assert wm_predictions is not None
        self.assertFalse(wm_predictions.requires_grad)
        self.assertEqual(
            tuple(wm_predictions.shape),
            (1, len(proposals), len(EXPECTED_GLOBAL_HEAD_NAMES)),
        )
        self.assertEqual(metadata["candidate_ids"], expected_candidate_ids)
        proposal_probabilities = proposal_log_probs.detach().softmax(dim=1)
        self.assertFalse(proposal_probabilities.requires_grad)
        wm_summary = (proposal_probabilities.unsqueeze(-1) * wm_predictions).sum(dim=1)
        self.assertFalse(wm_summary.requires_grad)
        channel = int(wm_summary.abs().argmax(dim=1).item())
        self.assertGreater(float(wm_summary[0, channel].abs().item()), 0.0)

        adapter_layers = policy_gwm.adapter.value_adapter
        with torch.no_grad():
            for parameter in adapter_layers.parameters():
                parameter.zero_()
            adapter_layers[0].weight[0, channel] = 1.0
            adapter_layers[2].weight[0, 0] = 1.0

        policy_gwm.zero_grad(set_to_none=True)
        result = policy_gwm.evaluate_batch(policy_input, actions, masks)
        self.assertFalse(torch.equal(g_result["value"], result["value"]))
        value_loss = (result["value"] - (g_result["value"].detach() + 3.0)).square()
        self.assertTrue(bool(torch.isfinite(value_loss)))
        value_loss.backward()

        adapter_gradients = [
            parameter.grad
            for parameter in policy_gwm.adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(adapter_gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in adapter_gradients))
        self.assertGreater(
            sum(float(grad.abs().sum().item()) for grad in adapter_gradients), 0.0
        )
        actor_head_parameters = [
            parameter
            for name, parameter in base_gwm.named_parameters()
            if name.startswith(("pair_actor.", "noop_actor."))
        ]
        self.assertTrue(actor_head_parameters)
        self.assertTrue(all(parameter.grad is None for parameter in actor_head_parameters))
        self.assertTrue(all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in world_model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
