"""Small masked-candidate PPO core for public dynamic graph observations.

This module defines an optimizer path only. It does not create an environment,
collect rollouts, load checkpoints, or launch Isaac.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .features import ACTION_DIM, EVENT_DIM, FEATURE_DIM

try:
    import torch
    from torch import Tensor, nn
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover - explicit at construction
    torch = None
    Tensor = Any
    nn = object  # type: ignore[assignment]
    Categorical = Any  # type: ignore[assignment]


MAX_CANDIDATES = 16
AUX_DYNAMIC_NAMES = ("task_progress", "energy_delta", "delay", "risk")
AUX_LAMBDA = 0.1


def compute_time_aware_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    elapsed_s: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute GAE using actual decision intervals; truncations bootstrap once."""
    if torch is None:
        raise RuntimeError("PYTORCH_UNAVAILABLE")
    tensors = (rewards, values, next_values, elapsed_s, terminated, truncated)
    if any(item.ndim != 1 for item in tensors) or len({item.shape[0] for item in tensors}) != 1:
        raise ValueError("GAE inputs must be aligned one-dimensional transition arrays")
    if (elapsed_s <= 0).any() or ((terminated.bool()) & (truncated.bool())).any():
        raise ValueError("elapsed_s must be positive and transitions cannot be both terminal and truncated")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    advantages = torch.zeros_like(rewards)
    next_advantage = rewards.new_zeros(())
    for index in range(rewards.shape[0] - 1, -1, -1):
        term = terminated[index].bool()
        boundary = term | truncated[index].bool()
        discount = torch.pow(rewards.new_tensor(gamma), elapsed_s[index])
        trace_discount = torch.pow(rewards.new_tensor(gamma * gae_lambda), elapsed_s[index])
        bootstrap = torch.where(term, torch.zeros_like(next_values[index]), next_values[index])
        delta = rewards[index] + discount * bootstrap - values[index]
        next_advantage = delta + torch.where(boundary, torch.zeros_like(next_advantage),
                                             trace_discount * next_advantage)
        advantages[index] = next_advantage
    return advantages, advantages + values


if torch is not None:

    class _PublicGraphEncoder(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.node = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.Tanh())
            self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.edge = nn.Embedding(3, hidden_dim)
            self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
            self.event = nn.Sequential(nn.Linear(EVENT_DIM, hidden_dim), nn.Tanh())
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        def forward(self, batch: Any) -> Tensor:
            nodes = batch.node_features
            node_mask = batch.node_mask.bool()
            sequence_mask = batch.sequence_mask.bool()
            if nodes.ndim != 4 or nodes.shape[-1] != FEATURE_DIM:
                raise ValueError("node_features must have shape [batch,time,nodes,FEATURE_DIM]")
            encoded = self.node(nodes)
            edge_index = batch.edge_index
            edge_mask = batch.edge_mask
            edge_type = batch.edge_type
            if edge_index is not None and edge_mask is not None and edge_type is not None and edge_mask.any():
                bsz, steps, count, width = encoded.shape
                src, dst = edge_index[:, :, 0].long(), edge_index[:, :, 1].long()
                valid = edge_mask.bool()
                flat_mask = node_mask.reshape(bsz * steps, count)
                flat_src, flat_dst = src.reshape(bsz * steps, -1), dst.reshape(bsz * steps, -1)
                flat_valid = valid.reshape(bsz * steps, -1)
                if count == 0 and flat_valid.any():
                    raise ValueError("active edge exists for an empty dynamic graph")
                if count > 0 and (((flat_src[flat_valid] < 0) | (flat_src[flat_valid] >= count)
                                   | (flat_dst[flat_valid] < 0) | (flat_dst[flat_valid] >= count)).any()):
                    raise ValueError("active edge endpoint exceeds the dynamic graph node dimension")
                safe_src, safe_dst = flat_src.clamp(0, max(0, count - 1)), flat_dst.clamp(0, max(0, count - 1))
                ids = torch.arange(bsz * steps, device=nodes.device).unsqueeze(1).expand_as(safe_src)
                flat_valid = flat_valid & flat_mask[ids, safe_src] & flat_mask[ids, safe_dst]
                flat = encoded.reshape(bsz * steps, count, width)
                types = edge_type.reshape(bsz * steps, -1).long().clamp(0, 2)
                msg = self.message(flat[ids, safe_src]) + self.edge(types)
                msg = msg * flat_valid.unsqueeze(-1).to(msg.dtype)
                aggregate = torch.zeros_like(flat)
                aggregate.index_put_((ids[flat_valid], safe_dst[flat_valid]), msg[flat_valid], accumulate=True)
                degree = torch.zeros((bsz * steps, count, 1), dtype=encoded.dtype, device=encoded.device)
                degree.index_put_((ids[flat_valid], safe_dst[flat_valid]),
                                  torch.ones((int(flat_valid.sum()), 1), dtype=encoded.dtype, device=encoded.device),
                                  accumulate=True)
                aggregate = aggregate / degree.clamp_min(1.0)
                encoded = self.update(torch.cat((flat, aggregate), dim=-1)).reshape(bsz, steps, count, width)
                encoded = encoded * node_mask.unsqueeze(-1).to(encoded.dtype)
            mask = node_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            events = batch.event_features
            if events is not None:
                pooled = pooled + self.event(events)
            sequence, _ = self.gru(pooled)
            last = sequence_mask.long().sum(dim=1).clamp_min(1) - 1
            return sequence[torch.arange(sequence.shape[0], device=sequence.device), last]


    class MaskedCandidateActorCritic(nn.Module):
        """Independent public-graph actor/critic with a critic-only WM head."""
        def __init__(self, candidate_dim: int, hidden_dim: int = 64, dynamic_dim: int = 4) -> None:
            super().__init__()
            if candidate_dim < 1 or dynamic_dim != len(AUX_DYNAMIC_NAMES):
                raise ValueError("candidate_dim must be positive and dynamic_dim must be four")
            self.actor_encoder = _PublicGraphEncoder(hidden_dim)
            self.critic_encoder = _PublicGraphEncoder(hidden_dim)
            self.actor = nn.Sequential(nn.Linear(hidden_dim + candidate_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, 1))
            self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.auxiliary_critic = nn.Sequential(
                nn.Linear(hidden_dim + candidate_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dynamic_dim))

        def forward(self, graph_batch: Any, candidate_features: Tensor) -> dict[str, Tensor]:
            if candidate_features.ndim != 3 or candidate_features.shape[0] != graph_batch.node_features.shape[0]:
                raise ValueError("candidate_features must have shape [batch,candidates,candidate_dim]")
            actor_state = self.actor_encoder(graph_batch)
            critic_state = self.critic_encoder(graph_batch)
            actor_state = actor_state.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
            critic_state = critic_state.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
            actor_input = torch.cat((actor_state, candidate_features), dim=-1)
            critic_input = torch.cat((critic_state, candidate_features), dim=-1)
            return {"logits": self.actor(actor_input).squeeze(-1),
                    "value": self.value(critic_state[:, 0]).squeeze(-1),
                    "auxiliary_dynamic": self.auxiliary_critic(critic_input)}


    @dataclass(frozen=True)
    class PPOUpdate:
        policy_loss: float
        value_loss: float
        entropy: float
        auxiliary_loss: float
        auxiliary_valid_elements: int
        auxiliary_critic_grad_norm: float
        auxiliary_status: str
        updates: int


    class PPOCore:
        """Clipped PPO updates over caller-collected dynamic graph transitions."""
        def __init__(self, model: MaskedCandidateActorCritic, *, device: str = "cpu",
                     learning_rate: float = 3e-4, clip_ratio: float = 0.2,
                     value_clip: float = 0.2, value_coefficient: float = 0.5,
                     entropy_coefficient: float = 0.01, auxiliary_lambda: float = 0.0,
                     world_model: nn.Module | None = None,
                     dynamic_scale: tuple[float, ...] | None = None,
                     dynamic_scale_evidence: Mapping[str, Any] | None = None) -> None:
            if auxiliary_lambda not in (0.0, AUX_LAMBDA):
                raise ValueError(f"auxiliary_lambda must be 0 or frozen protocol value {AUX_LAMBDA}")
            if auxiliary_lambda and (world_model is None or dynamic_scale is None or dynamic_scale_evidence is None):
                raise ValueError("G-WM auxiliary updates require frozen WM and evidenced train-split dynamic scales")
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.clip_ratio = float(clip_ratio)
            self.value_clip = float(value_clip)
            self.value_coefficient = float(value_coefficient)
            self.entropy_coefficient = float(entropy_coefficient)
            self.auxiliary_lambda = float(auxiliary_lambda)
            self.world_model = world_model
            self.dynamic_scale = (torch.as_tensor(dynamic_scale, dtype=torch.float32, device=self.device)
                                  if dynamic_scale is not None else None)
            if self.dynamic_scale is not None and (self.dynamic_scale.shape != (4,)
                    or not torch.isfinite(self.dynamic_scale).all() or (self.dynamic_scale <= 0).any()):
                raise ValueError("dynamic_scale must contain four finite positive train-split scales")
            if auxiliary_lambda:
                evidence = dict(dynamic_scale_evidence or {})
                counts = evidence.get("count_per_target")
                digest = evidence.get("dataset_sha256")
                expected_scale = [float(value) for value in dynamic_scale]
                if (evidence.get("scope") != "train_split_only"
                        or evidence.get("target_names") != list(AUX_DYNAMIC_NAMES)
                        or evidence.get("estimator") != "standard_deviation"
                        or evidence.get("scale") != expected_scale
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest.lower())
                        or not isinstance(counts, list) or len(counts) != 4
                        or any(not isinstance(value, int) or value < 1 for value in counts)):
                    raise ValueError("dynamic scale evidence must bind train-only target scales and coverage")
                self.dynamic_scale_evidence = evidence
            else:
                self.dynamic_scale_evidence = None
            if world_model is not None:
                world_model.to(self.device).eval()
                for parameter in world_model.parameters():
                    parameter.requires_grad_(False)
            actor_params = {id(parameter) for parameter in model.actor_encoder.parameters()} | {
                id(parameter) for parameter in model.actor.parameters()}
            critic_params = {id(parameter) for parameter in model.critic_encoder.parameters()} | {
                id(parameter) for parameter in model.value.parameters()} | {
                id(parameter) for parameter in model.auxiliary_critic.parameters()}
            if actor_params & critic_params:
                raise ValueError("actor and critic parameters must be disjoint")
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def _auxiliary_targets(self, graph_batch: Any, action_features: Tensor) -> Tensor:
            if self.world_model is None:
                raise ValueError("G-WM auxiliary targets require a frozen world model")
            model_actions = torch.zeros_like(graph_batch.action_features, device=self.device)
            last = graph_batch.sequence_mask.long().sum(dim=1).clamp_min(1) - 1
            model_actions[torch.arange(last.shape[0], device=self.device), last] = action_features
            with torch.no_grad():
                return self.world_model(
                    graph_batch.node_features, graph_batch.node_mask, graph_batch.sequence_mask,
                    graph_batch.edge_index, graph_batch.edge_mask, graph_batch.edge_type,
                    graph_batch.event_features, model_actions)["dynamic"]

        def update(self, graph_batch: Any, *, candidate_features: Tensor, candidate_mask: Tensor,
                   candidate_wm_action_features: Tensor | None = None,
                   noop_mask: Tensor, actions: Tensor, old_log_probs: Tensor, old_values: Tensor,
                   returns: Tensor, advantages: Tensor, executed_action_features: Tensor | None = None,
                   action_known: Tensor | None = None, wm_dynamic_mask: Tensor | None = None,
                   wm_joint_support_mask: Tensor | None = None,
                   epochs: int = 1) -> PPOUpdate:
            candidate_features = candidate_features.to(self.device)
            candidate_mask, noop_mask = candidate_mask.to(self.device).bool(), noop_mask.to(self.device).bool()
            actions = actions.to(self.device).long()
            old_log_probs, old_values = old_log_probs.to(self.device), old_values.to(self.device)
            returns, advantages = returns.to(self.device), advantages.to(self.device)
            batch_size, count = candidate_mask.shape
            if count > MAX_CANDIDATES:
                raise ValueError(f"candidate count {count} exceeds the no-truncation limit {MAX_CANDIDATES}")
            if (candidate_features.shape[:2] != (batch_size, count) or noop_mask.shape != candidate_mask.shape
                    or not (noop_mask.sum(dim=1) == 1).all() or not (noop_mask & candidate_mask).any(dim=1).all()
                    or not candidate_mask.any(dim=1).all()):
                raise ValueError("each state needs a legal NOOP and at least one legal candidate")
            rows = torch.arange(batch_size, device=self.device)
            if actions.shape != (batch_size,) or ((actions < 0) | (actions >= count)).any():
                raise ValueError("executed action indices are out of bounds")
            if (old_log_probs.shape != (batch_size,) or old_values.shape != (batch_size,)
                    or returns.shape != (batch_size,) or advantages.shape != (batch_size,)):
                raise ValueError("PPO scalars must have one value per transition")
            if not candidate_mask[rows, actions].all():
                raise ValueError("each executed action must index a legal candidate")
            if epochs < 1:
                raise ValueError("epochs must be positive")
            graph_batch = _batch_to(graph_batch, self.device)
            if self.auxiliary_lambda:
                required = (executed_action_features, action_known, wm_dynamic_mask, wm_joint_support_mask)
                if any(value is None for value in required):
                    raise ValueError("G-WM updates require executed action features and explicit masks")
                executed_action_features = executed_action_features.to(self.device)
                action_known = action_known.to(self.device).bool()
                wm_dynamic_mask = wm_dynamic_mask.to(self.device).bool()
                wm_joint_support_mask = wm_joint_support_mask.to(self.device).bool()
                if candidate_wm_action_features is None:
                    raise ValueError("G-WM updates require per-candidate WM action encodings")
                candidate_wm_action_features = candidate_wm_action_features.to(self.device)
                if executed_action_features.shape != (batch_size, ACTION_DIM):
                    raise ValueError("executed action features must use the WM ACTION_DIM/order")
                if candidate_wm_action_features.shape != (batch_size, count, ACTION_DIM):
                    raise ValueError("candidate WM action features must have shape [batch,candidates,ACTION_DIM]")
                if (wm_dynamic_mask.shape != (batch_size, 4) or wm_joint_support_mask.shape != (batch_size, 4)
                        or action_known.shape != (batch_size,)):
                    raise ValueError("action and checkpoint dynamic masks have invalid shapes")
                if ((executed_action_features[:, 0] < 0.5) | (executed_action_features[:, 1] < 0.5))[action_known].any():
                    raise ValueError("known executed actions must use [known=1, applied=1, ...] WM encoding")
                if action_known.any() and not torch.allclose(
                        executed_action_features[action_known],
                        candidate_wm_action_features[rows, actions][action_known]):
                    raise ValueError("executed action encoding must match its selected candidate's WM action features")
                targets = self._auxiliary_targets(graph_batch, executed_action_features)
                targets = targets / self.dynamic_scale
                aux_mask = wm_dynamic_mask & wm_joint_support_mask & action_known.unsqueeze(-1)
                aux_mask = aux_mask & candidate_mask[rows, actions].unsqueeze(-1)
            else:
                targets = None
                aux_mask = torch.zeros((batch_size, 4), dtype=torch.bool, device=self.device)
            policy_total = value_total = entropy_total = aux_total = grad_total = 0.0
            updates = 0
            for _ in range(epochs):
                result = self.model(graph_batch, candidate_features)
                dist = _masked_categorical(result["logits"], candidate_mask)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_prob - old_log_probs)
                policy_loss = -torch.minimum(ratio * advantages,
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages).mean()
                clipped_value = old_values + (result["value"] - old_values).clamp(-self.value_clip, self.value_clip)
                value_loss = 0.5 * torch.maximum((result["value"] - returns).square(),
                                                  (clipped_value - returns).square()).mean()
                if self.auxiliary_lambda and aux_mask.any():
                    predicted = result["auxiliary_dynamic"][rows, actions] / self.dynamic_scale
                    squared = (predicted - targets).square()
                    aux_loss = squared[aux_mask].mean()
                    aux_grads = torch.autograd.grad(aux_loss, tuple(self.model.critic_encoder.parameters()),
                                                    retain_graph=True, allow_unused=True)
                    grad_total += float(torch.sqrt(sum(g.detach().square().sum() for g in aux_grads if g is not None)).cpu())
                else:
                    aux_loss = result["auxiliary_dynamic"].sum() * 0.0
                loss = (policy_loss + self.value_coefficient * value_loss
                        - self.entropy_coefficient * entropy + self.auxiliary_lambda * aux_loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                policy_total += float(policy_loss.detach().cpu())
                value_total += float(value_loss.detach().cpu())
                entropy_total += float(entropy.detach().cpu())
                aux_total += float(aux_loss.detach().cpu())
                updates += 1
            valid_elements = int(aux_mask.sum().item())
            auxiliary_grad_norm = grad_total / updates
            if self.auxiliary_lambda == 0.0:
                auxiliary_status = "DISABLED_G_BASELINE"
            elif valid_elements > 0 and auxiliary_grad_norm > 0.0:
                auxiliary_status = "ACTIVE"
            else:
                auxiliary_status = "UNAVAILABLE_G_EQUIVALENT"
            return PPOUpdate(policy_total / updates, value_total / updates, entropy_total / updates,
                aux_total / updates, valid_elements, auxiliary_grad_norm, auxiliary_status, updates)

else:  # pragma: no cover
    class MaskedCandidateActorCritic:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PYTORCH_UNAVAILABLE")

    class PPOCore:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PYTORCH_UNAVAILABLE")


def _batch_to(batch: Any, device: Any) -> Any:
    """Copy a DynamicGraphBatch's tensor fields to the selected device."""
    import dataclasses
    import torch

    values = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return type(batch)(**values)


def _masked_categorical(logits: Tensor, mask: Tensor) -> Categorical:
    import torch
    if logits.shape != mask.shape or mask.dtype != torch.bool or not mask.any(dim=-1).all():
        raise ValueError("candidate mask must align with logits and leave at least one legal action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))
