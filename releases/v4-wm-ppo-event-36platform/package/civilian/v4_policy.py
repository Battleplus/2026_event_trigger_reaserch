"""V4 dynamic graph actor-critic policy and semi-Markov PPO utilities."""

from __future__ import annotations

import copy
import os
import math
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

try:
    from .event_models import NODE_FEATURE_DIM, NODE_TYPES, RELATION_COUNT, _node_features
    from .v4_models import ACTION_DIM, ACTION_RELATIONS, _action_relations, encode_v4_action_features
    from .event_policy import SCOPE_NAMES, _candidate_scope, _distribution, _is_noop, _scope_masks
except ImportError:
    from event_models import NODE_FEATURE_DIM, NODE_TYPES, RELATION_COUNT, _node_features
    from v4_models import ACTION_DIM, ACTION_RELATIONS, _action_relations, encode_v4_action_features
    from event_policy import SCOPE_NAMES, _candidate_scope, _distribution, _is_noop, _scope_masks


POLICY_SCHEMA = "n02-event-v4-hierarchical-ppo/1"
VARIANTS = (
    "Periodic-G", "Event-G", "Periodic-G-WM", "Event-G-WM-Decision",
    "Event-G-WM", "Event-G-WM-NoU", "Event-G-WM-Global",
)
POLICY_SEEDS = (8201, 8202, 8203)
POLICY_HIDDEN_DIM = 48
WM_SLOT_DIM = 9


def _wm_actor_features(predictions: dict[str, Any] | None, *, device: torch.device, enabled: bool,
                       candidate_count: int, no_utility: bool = False) -> torch.Tensor:
    """Return candidate-aligned scalar head and pooled node prediction means."""
    result = torch.zeros((candidate_count, WM_SLOT_DIM), dtype=torch.float32, device=device)
    if not enabled:
        if predictions is not None:
            raise ValueError("V4_BASELINE_MUST_NOT_RECEIVE_WM")
        return result
    if not isinstance(predictions, dict):
        raise ValueError("V4_WM_ACTOR_PREDICTIONS_REQUIRED")
    names = ("progress", "delay", "energy", "failure", "communication_risk")
    means = predictions.get("means", predictions)
    if not isinstance(means, dict):
        raise ValueError("V4_WM_MEANS_REQUIRED")
    values = []
    for name in names:
        value = means.get(name)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                raise ValueError(f"V4_WM_EMPTY_HEAD:{name}")
            value = value.detach().float().reshape(-1).cpu().tolist()
        elif isinstance(value, (list, tuple)):
            flat = [float(item) for item in value if isinstance(item, (int, float)) and math.isfinite(float(item))]
            if not flat:
                raise ValueError(f"V4_WM_EMPTY_HEAD:{name}")
            value = flat
        elif isinstance(value, (int, float)):
            value = [float(value)]
        else:
            raise ValueError(f"V4_WM_HEAD_REQUIRED:{name}")
        if any(not math.isfinite(float(item)) for item in value):
            raise ValueError(f"V4_WM_NONFINITE_HEAD:{name}")
        values.append(value)
    next_minute = predictions.get("next_minute")
    if not isinstance(next_minute, torch.Tensor) or next_minute.ndim < 2 or next_minute.shape[0] != candidate_count:
        raise ValueError("V4_WM_NEXT_MINUTE_CANDIDATE_ALIGNMENT")
    next_rows = next_minute.detach().float().reshape(candidate_count, -1, next_minute.shape[-1])
    entity_mask = predictions.get("entity_mask")
    if isinstance(entity_mask, torch.Tensor):
        if entity_mask.shape != next_rows.shape[:2]:
            raise ValueError("V4_WM_ENTITY_MASK_ALIGNMENT")
        valid = entity_mask.to(device=next_rows.device, dtype=torch.float32).unsqueeze(-1)
        next_rows = (next_rows * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    else:
        entity_ids = predictions.get("entity_ids")
        if isinstance(entity_ids, (list, tuple)) and len(entity_ids) == candidate_count:
            pooled = []
            for index, ids in enumerate(entity_ids):
                count = len(ids)
                if count < 1 or count > next_rows.shape[1]:
                    raise ValueError("V4_WM_ENTITY_IDS_ALIGNMENT")
                pooled.append(next_rows[index, :count].mean(0))
            next_rows = torch.stack(pooled)
        else:
            next_rows = next_rows.mean(1)
    next_rows = next_rows.cpu()
    result[:, 5:9] = next_rows[:, :4].to(device)
    for index, value in enumerate(values):
        if len(value) == 1:
            result[:, index] = float(value[0])
        elif len(value) == candidate_count:
            result[:, index] = torch.tensor(value, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"V4_WM_CANDIDATE_ALIGNMENT:{names[index]}")
    return result


class _DynamicPublicEncoder(nn.Module):
    """Typed relational message passing over a variable number of public nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.node_type = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.relations = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(RELATION_COUNT))
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.pool = nn.Sequential(nn.Linear(5 * hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, observation: dict) -> torch.Tensor:
        rows, types, edges = _node_features(observation)
        if not rows:
            rows, types, edges = [[0.0] * NODE_FEATURE_DIM], [0], [(0, 0, 0)]
        device = next(self.parameters()).device
        features = torch.tensor(rows, dtype=torch.float32, device=device)
        type_ids = torch.tensor(types, dtype=torch.long, device=device).clamp(0, 4)
        adjacency = torch.zeros((RELATION_COUNT, len(rows), len(rows)), dtype=torch.float32, device=device)
        for source, target, relation in edges:
            adjacency[relation, target, source] = 1.0
        adjacency /= adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        hidden = self.node(features) + self.node_type(type_ids)
        aggregate = sum(layer(adjacency[index] @ hidden) for index, layer in enumerate(self.relations))
        hidden = self.update(torch.cat((hidden, aggregate), dim=-1))
        pooled = []
        for node_type in range(len(NODE_TYPES)):
            mask = (type_ids == node_type).to(hidden.dtype).unsqueeze(-1)
            pooled.append((hidden * mask).sum(0) / mask.sum(0).clamp_min(1.0))
        return self.pool(torch.cat(pooled, dim=-1))


class _DynamicActionEncoder(nn.Module):
    """Encode complete actions over every affected public entity."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(NODE_FEATURE_DIM + ACTION_DIM, hidden_dim), nn.Tanh())
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.relations = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False)
                                       for _ in range(ACTION_RELATIONS))
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.pool = nn.Sequential(nn.Linear(2 * len(NODE_TYPES) * hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, observation: dict, action: dict, device: torch.device) -> torch.Tensor:
        node_ids, action_rows = encode_v4_action_features(observation, action)
        node_rows, types, _ = _node_features(observation)
        if len(types) != action_rows.shape[0] or len(node_ids) != len(node_rows):
            raise ValueError("V4_ACTION_NODE_TYPE_ALIGNMENT")
        values = torch.cat((torch.tensor(node_rows, dtype=torch.float32, device=device),
                            action_rows.to(device)), dim=-1)
        type_ids = torch.tensor(types, dtype=torch.long, device=device).clamp(0, len(NODE_TYPES) - 1)
        encoded = self.entity(values) + self.type_embedding(type_ids)
        action_edges = _action_relations({"action": action}, node_ids)
        adjacency = torch.tensor(action_edges, dtype=torch.float32, device=device)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        aggregate = sum(layer(adjacency[index] @ encoded) for index, layer in enumerate(self.relations))
        encoded = self.update(torch.cat((encoded, aggregate), dim=-1))
        summaries = []
        for node_type in range(len(NODE_TYPES)):
            selected = encoded[type_ids == node_type]
            if selected.shape[0]:
                summaries.extend((selected.mean(0), selected.max(0).values))
            else:
                zero = encoded.new_zeros(encoded.shape[-1])
                summaries.extend((zero, zero))
        return self.pool(torch.cat(summaries))


class V4HierarchicalPolicy(nn.Module):
    """Hierarchical actor with WM mean inputs and public-state-only value critic."""

    def __init__(self, *, use_wm_actor: bool, no_utility: bool = False, hidden_dim: int = POLICY_HIDDEN_DIM):
        super().__init__()
        self.use_wm_actor = bool(use_wm_actor)
        self.no_utility = bool(no_utility)
        self.hidden_dim = int(hidden_dim)
        self.actor_calls = 0
        self.critic_calls = 0
        self.actor_state = _DynamicPublicEncoder(hidden_dim)
        # WM slots exist in every variant so paired variants have identical capacity.
        self.actor_fusion = nn.Sequential(nn.Linear(hidden_dim + WM_SLOT_DIM, hidden_dim), nn.Tanh())
        self.candidate_encoder = _DynamicActionEncoder(hidden_dim)
        self.high_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, len(SCOPE_NAMES)))
        self.low_actor = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.critic_state = _DynamicPublicEncoder(hidden_dim)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def actor_logits(self, observation: dict, candidates: Sequence[dict], wm_predictions: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.actor_calls += 1
        state = self.actor_state(observation)
        wm = _wm_actor_features(wm_predictions, device=state.device, enabled=self.use_wm_actor,
                                candidate_count=len(candidates), no_utility=self.no_utility)
        candidate_context = torch.stack([
            self.candidate_encoder(observation, row.get("full_environment_action", {}), state.device)
            for row in candidates
        ])
        shared_context = self.actor_fusion(torch.cat((state, wm.mean(0))))
        candidate_context = candidate_context + self.actor_fusion(torch.cat((state.unsqueeze(0).expand(len(candidates), -1), wm), -1))
        high = self.high_actor(shared_context)
        low = self.low_actor(torch.cat((shared_context.unsqueeze(0).expand(len(candidates), -1), candidate_context), -1)).squeeze(-1)
        return high, low

    def value(self, observation: dict) -> torch.Tensor:
        self.critic_calls += 1
        return self.value_head(self.critic_state(observation)).reshape(())

    def act(self, observation: dict, candidates: Sequence[dict], high_level_mask=None, candidate_mask=None,
            wm_predictions: dict | None = None, *, deterministic: bool = False) -> dict[str, Any]:
        high_mask, lows = _scope_masks(candidates, high_level_mask, candidate_mask)
        high_logits, low_logits = self.actor_logits(observation, candidates, wm_predictions)
        high_dist = _distribution(high_logits, high_mask)
        high_index = int(torch.argmax(high_dist.logits).item()) if deterministic else int(high_dist.sample().item())
        scope = SCOPE_NAMES[high_index]
        low_dist = _distribution(low_logits, lows[scope])
        candidate_index = int(torch.argmax(low_dist.logits).item()) if deterministic else int(low_dist.sample().item())
        candidate = candidates[candidate_index]
        action = copy.deepcopy(candidate.get("full_environment_action"))
        if not isinstance(action, dict) or str(action.get("scope", "")).upper() != scope:
            raise ValueError("V4_POLICY_ACTION_SCOPE_BINDING")
        high_logp = high_dist.log_prob(torch.tensor(high_index, device=high_logits.device))
        low_logp = low_dist.log_prob(torch.tensor(candidate_index, device=low_logits.device))
        value = self.value(observation)
        return {
            "scope": scope, "scope_index": high_index, "candidate_index": candidate_index,
            "candidate_key": str(candidate.get("candidate_key", candidate.get("id", candidate_index))),
            "full_environment_action": action, "high_logits": high_logits.detach().cpu().tolist(),
            "low_logits": low_logits.detach().cpu().tolist(), "high_level_mask": list(high_mask),
            "candidate_mask": [bool(row.get("legal", True)) and
                               (candidate_mask is None or bool(candidate_mask[index]))
                               for index, row in enumerate(candidates)],
            "high_log_probability": float(high_logp.detach().cpu()),
            "low_log_probability": float(low_logp.detach().cpu()),
            "log_probability": float((high_logp + low_logp).detach().cpu()),
            "entropy": float((high_dist.entropy() + low_dist.entropy()).detach().cpu()),
            "value": float(value.detach().cpu()),
        }

    def evaluate_action(self, observation: dict, candidates: Sequence[dict], *, scope_index: int,
                        candidate_index: int, high_level_mask=None, candidate_mask=None,
                        wm_predictions: dict | None = None) -> dict[str, torch.Tensor]:
        high_mask, lows = _scope_masks(candidates, high_level_mask, candidate_mask)
        if not 0 <= scope_index < len(SCOPE_NAMES) or not 0 <= candidate_index < len(candidates):
            raise ValueError("V4_POLICY_ACTION_INDEX")
        scope = SCOPE_NAMES[scope_index]
        if not high_mask[scope_index] or not lows[scope][candidate_index]:
            raise ValueError("V4_POLICY_ACTION_MASK")
        high_logits, low_logits = self.actor_logits(observation, candidates, wm_predictions)
        high_dist, low_dist = _distribution(high_logits, high_mask), _distribution(low_logits, lows[scope])
        high_logp = high_dist.log_prob(torch.tensor(scope_index, device=high_logits.device))
        low_logp = low_dist.log_prob(torch.tensor(candidate_index, device=low_logits.device))
        return {"log_probability": high_logp + low_logp, "entropy": high_dist.entropy() + low_dist.entropy(), "value": self.value(observation)}


def build_v4_policy(seed: int, *, use_wm_actor: bool = False, no_utility: bool = False,
                    hidden_dim: int = POLICY_HIDDEN_DIM) -> V4HierarchicalPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        policy = V4HierarchicalPolicy(use_wm_actor=use_wm_actor, no_utility=no_utility, hidden_dim=hidden_dim)
    policy.train()
    return policy


def variant_policy_options(variant: str) -> dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError("V4_POLICY_VARIANT")
    # NoU ablates trigger uncertainty only; every WM policy receives the same
    # full set of actor predictions.
    return {"use_wm_actor": "-WM" in variant, "no_utility": False}


def save_v4_checkpoint(path: str | Path, policy: V4HierarchicalPolicy, optimizer: torch.optim.Optimizer,
                       *, variant: str, seed: int, config: dict, source_identity: str,
                       restore_witness: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist all state needed to resume an optimizer in a fresh process."""
    if variant not in VARIANTS or not source_identity:
        raise ValueError("V4_CHECKPOINT_IDENTITY")
    destination = Path(path)
    payload = {
        "schema": POLICY_SCHEMA, "variant": variant, "seed": int(seed),
        "policy_options": variant_policy_options(variant), "hidden_dim": policy.hidden_dim,
        "config": copy.deepcopy(config), "source_identity": str(source_identity),
        "model_state": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
        "optimizer_state": optimizer.state_dict(), "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if restore_witness is not None:
        payload["restore_witness"] = copy.deepcopy(restore_witness)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"path": str(destination.resolve()), "source_identity": source_identity,
            "optimizer_state_saved": True}


def load_v4_checkpoint(path: str | Path, *, device: str | torch.device = "cpu",
                       expected_source_identity: str | None = None) -> tuple[V4HierarchicalPolicy, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA or payload.get("variant") not in VARIANTS:
        raise ValueError("V4_CHECKPOINT_SCHEMA")
    if expected_source_identity is not None and payload.get("source_identity") != expected_source_identity:
        raise ValueError("V4_CHECKPOINT_SOURCE_IDENTITY")
    options = payload["policy_options"]
    policy = build_v4_policy(int(payload["seed"]), hidden_dim=int(payload["hidden_dim"]), **options).to(device)
    policy.load_state_dict(payload["model_state"], strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(payload["config"].get("learning_rate", 0.0003)))
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return policy, optimizer, payload


def attach_smdp_gae(records: list[dict], *, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
    """Attach duration-aware GAE; minute rewards are discounted within each action."""
    if not 0.0 < gamma <= 1.0 or not 0.0 < gae_lambda <= 1.0:
        raise ValueError("V4_SMDP_DISCOUNT_RANGE")
    following_advantage = 0.0
    for index in range(len(records) - 1, -1, -1):
        row = records[index]
        minute_rewards = row.get("minute_rewards")
        if minute_rewards is not None:
            if not isinstance(minute_rewards, (list, tuple)) or not minute_rewards:
                raise ValueError("V4_SMDP_MINUTE_REWARDS")
            dt = len(minute_rewards)
            reward = sum((gamma ** offset) * float(value) for offset, value in enumerate(minute_rewards))
        else:
            dt = int(row.get("dt", row.get("duration", 1)))
            reward = float(row.get("reward", 0.0))
        if dt < 1:
            raise ValueError("V4_SMDP_DURATION")
        terminated = bool(row.get("terminated", False))
        boundary = terminated or bool(row.get("truncated", False)) or bool(row.get("rollout_cut", False))
        if "bootstrap_value" in row:
            next_value = float(row["bootstrap_value"])
        elif index + 1 < len(records) and not terminated:
            next_value = float(records[index + 1]["old_value"])
        else:
            next_value = 0.0
        discount = gamma ** dt
        trace = (gamma * gae_lambda) ** dt
        delta = reward + (0.0 if terminated else discount * next_value) - float(row["old_value"])
        advantage = delta + (0.0 if boundary else trace * following_advantage)
        row.update({"dt": dt, "discounted_reward": reward, "next_value": next_value,
                    "advantage": float(advantage), "return": float(advantage + float(row["old_value"]))})
        following_advantage = advantage


def smdp_ppo_defaults() -> dict[str, Any]:
    return {"seeds": list(POLICY_SEEDS), "rounds": 128, "episodes_per_round": 4,
            "learning_rate": 0.0003, "epochs": 4, "minibatch_size": 64,
            "clip_ratio": 0.2, "entropy_coefficient": 0.01, "gamma": 0.99, "gae_lambda": 0.95}


def update_v4_policy(policy: V4HierarchicalPolicy, optimizer: torch.optim.Optimizer,
                     records: list[dict], *, seed: int, epochs: int = 4,
                     minibatch_size: int = 64, clip_ratio: float = 0.2,
                     entropy_coefficient: float = 0.01) -> list[dict[str, Any]]:
    """Run PPO epochs over all transitions, including each final partial batch."""
    if not records or epochs < 1 or minibatch_size < 1:
        raise ValueError("V4_PPO_BATCH_CONFIG")
    device = next(policy.parameters()).device
    advantages = torch.tensor([float(row["advantage"]) for row in records], dtype=torch.float32, device=device)
    returns = torch.tensor([float(row["return"]) for row in records], dtype=torch.float32, device=device)
    old_logp = torch.tensor([float(row["old_log_probability"]) for row in records], dtype=torch.float32, device=device)
    actor_valid = torch.tensor([
        sum(bool(value) for value in row.get("high_level_mask", [])) > 1
        or sum(bool(allowed) and _candidate_scope(candidate) == SCOPE_NAMES[int(row["scope_index"])]
               for allowed, candidate in zip(row.get("candidate_mask", []), row.get("candidates", []))) > 1
        for row in records
    ], dtype=torch.bool, device=device)
    if bool(actor_valid.any()):
        actor_advantages = advantages[actor_valid]
        normalized = (actor_advantages - actor_advantages.mean()) / actor_advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = advantages.clone()
        advantages[actor_valid] = normalized
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 491)
    logs: list[dict[str, Any]] = []
    for epoch in range(epochs):
        order = torch.randperm(len(records), generator=generator).tolist()
        for batch_start in range(0, len(order), minibatch_size):
            indices = order[batch_start:batch_start + minibatch_size]
            rows = [records[index] for index in indices]
            evaluated = []
            for row, index in zip(rows, indices):
                if bool(actor_valid[index].item()):
                    evaluated.append(policy.evaluate_action(
                        row["observation"], row["candidates"], scope_index=int(row["scope_index"]),
                        candidate_index=int(row["candidate_index"]), high_level_mask=row["high_level_mask"],
                        candidate_mask=row.get("candidate_mask"), wm_predictions=row.get("wm_predictions"),
                    ))
                else:
                    value = policy.value(row["observation"])
                    evaluated.append({"value": value, "log_probability": value.new_zeros(()),
                                      "entropy": value.new_zeros(())})
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            values = torch.stack([item["value"] for item in evaluated])
            value_loss = 0.5 * torch.square(values - returns[index_tensor]).mean()
            valid_local = torch.tensor([bool(actor_valid[index].item()) for index in indices], dtype=torch.bool, device=device)
            if bool(valid_local.any()):
                logp = torch.stack([item["log_probability"] for item in evaluated])[valid_local]
                entropy = torch.stack([item["entropy"] for item in evaluated])[valid_local].mean()
                old = old_logp[index_tensor][valid_local]
                advantage = advantages[index_tensor][valid_local]
                ratio = torch.exp(logp - old)
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                actor_loss = -torch.minimum(unclipped, clipped).mean()
            else:
                entropy = values.new_zeros(())
                actor_loss = values.new_zeros(())
            loss = actor_loss + value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V4_PPO_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("V4_PPO_NONFINITE_GRADIENT")
            optimizer.step()
            logs.append({"update": len(logs) + 1, "epoch": epoch + 1,
                         "sample_indices": indices, "sample_count": len(indices),
                         "actor_sample_count": int(valid_local.sum().item()),
                         "actor_loss": float(actor_loss.detach().cpu()),
                         "value_loss": float(value_loss.detach().cpu()),
                         "entropy": float(entropy.detach().cpu()),
                         "total_loss": float(loss.detach().cpu()),
                         "gradient_norm_preclip": float(norm.detach().cpu())})
    return logs
