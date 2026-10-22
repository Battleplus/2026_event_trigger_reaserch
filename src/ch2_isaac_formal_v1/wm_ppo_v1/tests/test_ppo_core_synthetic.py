"""CPU synthetic tests only; these do not constitute Isaac PPO training."""
from __future__ import annotations

import unittest

import torch

from wm_ppo_v1.contracts import PublicHistoryFrame
from wm_ppo_v1.features import ACTION_DIM, FEATURE_DIM, collate_graph_sequences
from wm_ppo_v1.ppo_core import (AUX_LAMBDA, MaskedCandidateActorCritic, PPOCore,
    _masked_categorical, compute_time_aware_gae)
from wm_ppo_v1.world_model import EventAwareWorldModel


def _graph_batch(batch_size=2):
    histories = [[PublicHistoryFrame(time_s=1.0 + index,
        platforms=({"x_norm": 0.1 * (index + 1), "energy_log_norm": 0.7,
                    "platform_kind": "uav", "available": True},))]
        for index in range(batch_size)]
    collated = collate_graph_sequences(histories)
    nodes = torch.as_tensor(collated.node_features, dtype=torch.float32)
    return type(collated)(node_features=nodes,
        node_mask=torch.as_tensor(collated.node_mask, dtype=torch.bool),
        sequence_mask=torch.as_tensor(collated.sequence_mask, dtype=torch.bool),
        edge_index=torch.as_tensor(collated.edge_index, dtype=torch.long),
        edge_mask=torch.as_tensor(collated.edge_mask, dtype=torch.bool),
        edge_type=torch.as_tensor(collated.edge_type, dtype=torch.long),
        event_features=torch.as_tensor(collated.event_features, dtype=torch.float32),
        action_features=torch.as_tensor(collated.action_features, dtype=torch.float32))


class PPOCoreSyntheticTests(unittest.TestCase):
    def test_time_aware_gae_terminal_and_truncation_bootstrap(self):
        rewards = torch.tensor([1.0, 2.0, 2.0])
        values = torch.tensor([0.5, 1.0, 1.0])
        next_values = torch.tensor([1.0, 7.0, 4.0])
        elapsed = torch.tensor([2.0, 1.0, 2.0])
        terminal = torch.tensor([False, True, False])
        truncated = torch.tensor([False, False, True])
        advantages, returns = compute_time_aware_gae(rewards, values, next_values, elapsed,
            terminal, truncated, gamma=0.5, gae_lambda=0.8)
        self.assertAlmostEqual(float(advantages[1]), 1.0, places=6)
        self.assertAlmostEqual(float(advantages[0]), 0.91, places=6)
        self.assertAlmostEqual(float(advantages[2]), 2.0, places=6)
        self.assertAlmostEqual(float(returns[2]), 3.0, places=6)

    def test_illegal_candidate_has_zero_probability(self):
        distribution = _masked_categorical(torch.tensor([[0.0, 1.0, 100.0]]),
                                            torch.tensor([[True, True, False]]))
        self.assertEqual(float(distribution.probs[0, 2]), 0.0)
        core = PPOCore(MaskedCandidateActorCritic(candidate_dim=7, hidden_dim=8), device="cpu")
        batch = _graph_batch(batch_size=1)
        with self.assertRaisesRegex(ValueError, "no-truncation"):
            core.update(batch, candidate_features=torch.zeros((1, 17, 7)),
                candidate_mask=torch.ones((1, 17), dtype=torch.bool),
                noop_mask=torch.tensor([[True] + [False] * 16]), actions=torch.tensor([0]),
                old_log_probs=torch.zeros(1), old_values=torch.zeros(1), returns=torch.zeros(1),
                advantages=torch.zeros(1))

    def test_ppo_and_frozen_wm_auxiliary_optimizer_path(self):
        torch.manual_seed(7)
        batch = _graph_batch()
        wm = EventAwareWorldModel(FEATURE_DIM, hidden_dim=16)
        wm.eval()
        model = MaskedCandidateActorCritic(candidate_dim=7, hidden_dim=16)
        core = PPOCore(model, device="cpu", learning_rate=1e-2, auxiliary_lambda=AUX_LAMBDA,
                       world_model=wm, dynamic_scale=(1.0, 10.0, 2.0, 1.0),
                       dynamic_scale_evidence={"scope": "train_split_only",
                           "target_names": ["task_progress", "energy_delta", "delay", "risk"],
                           "estimator": "standard_deviation", "scale": [1.0, 10.0, 2.0, 1.0],
                           "dataset_sha256": "a" * 64, "count_per_target": [100, 100, 100, 100]})
        candidates = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                    [0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8]]]).repeat(2, 1, 1)
        legal = torch.tensor([[True, True, False], [True, True, True]])
        noop = torch.tensor([[True, False, False], [True, False, False]])
        actions = torch.tensor([1, 2])
        wm_candidates = torch.zeros((2, 3, ACTION_DIM))
        wm_action = torch.tensor([[1.0, 1.0, 0.25, 0.5, 0.0]] * 2)
        wm_candidates[0, 1] = wm_action[0]
        wm_candidates[1, 2] = wm_action[1]
        before = model(batch, candidates)
        dist = _masked_categorical(before["logits"], legal)
        old_log_probs = dist.log_prob(actions).detach()
        old_values = before["value"].detach()
        targets = core._auxiliary_targets(batch, wm_action) / core.dynamic_scale
        rows = torch.arange(2)
        pure_auxiliary_loss = (before["auxiliary_dynamic"][rows, actions] / core.dynamic_scale - targets).square().mean()
        critic_aux_grads = torch.autograd.grad(pure_auxiliary_loss,
            tuple(model.critic_encoder.parameters()), retain_graph=True, allow_unused=True)
        actor_aux_grads = torch.autograd.grad(pure_auxiliary_loss,
            tuple(model.actor.parameters()), retain_graph=True, allow_unused=True)
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0
                            for gradient in critic_aux_grads))
        self.assertTrue(all(gradient is None for gradient in actor_aux_grads))
        self.assertTrue(all(parameter.grad is None for parameter in wm.parameters()))
        actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in model.critic_encoder.parameters()]
        wm_before = {key: value.detach().clone() for key, value in wm.state_dict().items()}
        result = core.update(batch, candidate_features=candidates,
            candidate_wm_action_features=wm_candidates, candidate_mask=legal,
            noop_mask=noop, actions=actions, old_log_probs=old_log_probs, old_values=old_values,
            returns=torch.tensor([1.0, -1.0]), advantages=torch.tensor([1.0, -1.0]),
            executed_action_features=wm_action,
            action_known=torch.tensor([True, True]), wm_dynamic_mask=torch.ones((2, 4), dtype=torch.bool),
            wm_joint_support_mask=torch.ones((2, 4), dtype=torch.bool), epochs=1)
        self.assertEqual(result.auxiliary_valid_elements, 8)
        self.assertGreater(result.auxiliary_critic_grad_norm, 0.0)
        self.assertGreater(result.auxiliary_loss, 0.0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(actor_before, model.actor.parameters())))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(critic_before, model.critic_encoder.parameters())))
        self.assertTrue(all(torch.equal(wm_before[key], value) for key, value in wm.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in wm.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in wm.parameters()))


if __name__ == "__main__":
    unittest.main()
