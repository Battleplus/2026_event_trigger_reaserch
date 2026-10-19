"""Hierarchical graph PPO for public event-trigger reconstruction catalogs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

try:
    from .event_models import (
        CANDIDATE_FEATURE_DIM,
        ENSEMBLE_SEEDS,
        GRID_CELLS,
        MAP_COLS,
        MAP_CONTRACT,
        MAP_ROWS,
        NODE_FEATURE_DIM,
        RELATION_COUNT,
        WORLD_MODEL_HEADS,
        WorldModelEnsemble,
        _candidate_features,
        _mapping,
        _node_features,
        _public_observation,
        predict_ensemble,
    )
except ImportError:
    from event_models import (
        CANDIDATE_FEATURE_DIM,
        ENSEMBLE_SEEDS,
        GRID_CELLS,
        MAP_COLS,
        MAP_CONTRACT,
        MAP_ROWS,
        NODE_FEATURE_DIM,
        RELATION_COUNT,
        WORLD_MODEL_HEADS,
        WorldModelEnsemble,
        _candidate_features,
        _mapping,
        _node_features,
        _public_observation,
        predict_ensemble,
    )


POLICY_SCHEMA = "n02-event-hierarchical-ppo/1"
VARIANTS = ("Periodic-G", "Event-G", "Periodic-G-WM", "Event-G-WM")
SCOPE_NAMES = ("NOOP", "LOCAL_REPAIR", "GLOBAL_REPLAN")
POLICY_SEEDS = ENSEMBLE_SEEDS
POLICY_HIDDEN_DIM = 48


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii")); digest.update(str(tuple(tensor.shape)).encode("ascii")); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _world_model_member_state_digests(ensemble: WorldModelEnsemble | Sequence[Any]) -> list[str]:
    """Digest each live world-model member independently from its full state."""
    models = getattr(ensemble, "models", ensemble)
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("EVENT_POLICY_WORLD_MODEL_MEMBERS_REQUIRED")
    digests: list[str] = []
    for model in models:
        state = model.state_dict()
        if not isinstance(state, dict) or not state:
            raise ValueError("EVENT_POLICY_WORLD_MODEL_STATE_REQUIRED")
        digests.append(_tensor_digest(state))
    return digests


def _actor_digest(policy: "HierarchicalGraphPolicy") -> str:
    state = {name: value for name, value in policy.state_dict().items() if name.startswith(("actor_encoder.", "candidate_encoder.", "high_actor.", "low_actor."))}
    return _tensor_digest(state)


def _candidate_scope(candidate: dict) -> str:
    action = _mapping(candidate.get("full_environment_action"))
    raw = str(candidate.get("candidate_scope", action.get("scope", ""))).upper()
    aliases = {"LOCAL": "LOCAL_REPAIR", "GLOBAL": "GLOBAL_REPLAN", "NONE": "NOOP"}
    return aliases.get(raw, raw)


def _is_noop(candidate: dict) -> bool:
    action = _mapping(candidate.get("full_environment_action"))
    return bool(action.get("noop")) or str(action.get("action_type", "")).lower() == "noop" or _candidate_scope(candidate) == "NOOP"


def _scope_masks(
    candidates: Sequence[dict],
    high_level_mask: Sequence[bool] | dict[str, bool] | None,
    candidate_mask: Sequence[bool] | None,
) -> tuple[list[bool], dict[str, list[bool]]]:
    if not candidates:
        raise ValueError("EVENT_POLICY_CATALOG_REQUIRED")
    legal = [bool(item.get("legal", True)) for item in candidates]
    if candidate_mask is not None:
        if len(candidate_mask) != len(candidates):
            raise ValueError("EVENT_POLICY_CANDIDATE_MASK_LENGTH")
        legal = [flag and bool(mask) for flag, mask in zip(legal, candidate_mask)]
    noop_indices = [index for index, candidate in enumerate(candidates) if _is_noop(candidate) and legal[index]]
    if not noop_indices:
        raise ValueError("EVENT_POLICY_NO_LEGAL_NOOP")
    scope_to_mask = {scope: [legal[index] and _candidate_scope(candidate) == scope for index, candidate in enumerate(candidates)] for scope in SCOPE_NAMES}
    scope_to_mask["NOOP"] = [legal[index] and (index == noop_indices[0]) for index in range(len(candidates))]
    available = [any(scope_to_mask[scope]) for scope in SCOPE_NAMES]
    if high_level_mask is None:
        requested = [True, True, True]
    elif isinstance(high_level_mask, dict):
        requested = [bool(high_level_mask.get(name, False)) for name in SCOPE_NAMES]
    else:
        requested = [bool(value) for value in high_level_mask]
        if len(requested) != len(SCOPE_NAMES):
            raise ValueError("EVENT_POLICY_HIGH_LEVEL_MASK_LENGTH")
    allowed = [want and have for want, have in zip(requested, available)]
    if not any(allowed):
        raise ValueError("EVENT_POLICY_NO_VALID_HIGH_LEVEL_ACTION")
    return allowed, scope_to_mask


def _distribution(logits: torch.Tensor, mask: Sequence[bool]) -> Categorical:
    mask_tensor = torch.tensor(list(mask), dtype=torch.bool, device=logits.device)
    if mask_tensor.shape != logits.shape or not bool(mask_tensor.any()):
        raise ValueError("EVENT_POLICY_INVALID_ACTION_MASK")
    safe_logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
    return Categorical(logits=safe_logits)


def _world_model_critic_features(predictions: dict[str, Any] | None, *, device: torch.device) -> torch.Tensor:
    if not isinstance(predictions, dict):
        raise ValueError("EVENT_POLICY_WM_PREDICTIONS_REQUIRED")
    means = _mapping(predictions.get("means")); variances = _mapping(predictions.get("variances"))
    values: list[float] = []
    for collection in (means, variances):
        for head in WORLD_MODEL_HEADS:
            array = collection.get(head)
            if not isinstance(array, list) or not array:
                raise ValueError(f"EVENT_POLICY_WM_HEAD:{head}")
            flattened: list[float] = []
            for row in array:
                row_values = list(row) if isinstance(row, (list, tuple)) else [row]
                flattened.extend(float(value) for value in row_values if math.isfinite(float(value)))
            if not flattened:
                raise ValueError(f"EVENT_POLICY_WM_EMPTY_HEAD:{head}")
            values.append(sum(flattened) / len(flattened))
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("EVENT_POLICY_WM_NONFINITE")
    return tensor.detach()


class _GraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.type_embedding = nn.Embedding(5, hidden_dim)
        self.relations = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(RELATION_COUNT))
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.pool = nn.Sequential(nn.Linear(hidden_dim * 5, hidden_dim), nn.Tanh())

    def forward(self, observation: dict) -> torch.Tensor:
        rows, types, edges = _node_features(observation)
        if not rows:
            rows, types, edges = [[0.0] * NODE_FEATURE_DIM], [0], [(0, 0, 0)]
        device = next(self.parameters()).device
        count = len(rows)
        x = torch.tensor(rows, dtype=torch.float32, device=device)
        type_ids = torch.tensor(types, dtype=torch.long, device=device).clamp(0, 4)
        adjacency = torch.zeros((RELATION_COUNT, count, count), dtype=torch.float32, device=device)
        for source, target, relation in edges:
            adjacency[relation, target, source] = 1.0
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        hidden = self.node_encoder(x) + self.type_embedding(type_ids)
        aggregate = torch.zeros_like(hidden)
        for relation, transform in enumerate(self.relations):
            aggregate = aggregate + transform(torch.matmul(adjacency[relation], hidden))
        hidden = self.update(torch.cat((hidden, aggregate), dim=-1))
        pooled = []
        for node_type in range(5):
            mask = (type_ids == node_type).to(hidden.dtype).unsqueeze(-1)
            pooled.append((hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0))
        return self.pool(torch.cat(pooled, dim=-1))


class HierarchicalGraphPolicy(nn.Module):
    """Two-stage public graph actor and a disjoint ordinary/auxiliary critic."""

    def __init__(self, *, use_wm_aux: bool, hidden_dim: int = POLICY_HIDDEN_DIM):
        super().__init__()
        self.variant = "unknown"
        self.use_wm_aux = bool(use_wm_aux)
        self.hidden_dim = int(hidden_dim)
        self.actor_forward_calls = 0
        self.critic_forward_calls = 0
        # Actor and critic encoders are disjoint: auxiliary value gradients have
        # no path into the actor representation or action logits.
        self.actor_encoder = _GraphEncoder(hidden_dim)
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_FEATURE_DIM, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.high_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, len(SCOPE_NAMES)))
        self.low_actor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.critic_encoder = _GraphEncoder(hidden_dim)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        if self.use_wm_aux:
            self.auxiliary_critic = nn.Sequential(nn.Linear(hidden_dim + 2 * len(WORLD_MODEL_HEADS), hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        else:
            self.auxiliary_critic = None

    def actor_logits(self, observation: dict, candidates: Sequence[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        self.actor_forward_calls += 1
        actor_context = self.actor_encoder(observation)
        candidate_rows = [_candidate_features(candidate, observation) for candidate in candidates]
        candidate_features = torch.tensor(candidate_rows, dtype=torch.float32, device=actor_context.device)
        candidate_context = self.candidate_encoder(candidate_features)
        repeated = actor_context.unsqueeze(0).expand(len(candidates), -1)
        high_logits = self.high_actor(actor_context)
        low_logits = self.low_actor(torch.cat((repeated, candidate_context), dim=-1)).squeeze(-1)
        return high_logits, low_logits

    def critic_values(self, observation: dict, wm_predictions: dict[str, Any] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.critic_forward_calls += 1
        critic_context = self.critic_encoder(observation)
        base_value = self.value_head(critic_context).reshape(())
        if self.use_wm_aux:
            wm_features = _world_model_critic_features(wm_predictions, device=critic_context.device)
            auxiliary_value = self.auxiliary_critic(torch.cat((critic_context, wm_features), dim=-1)).reshape(())
        else:
            if wm_predictions is not None:
                raise ValueError("EVENT_POLICY_BASELINE_MUST_NOT_RECEIVE_WM")
            auxiliary_value = base_value.new_zeros(())
        return base_value + auxiliary_value, base_value, auxiliary_value

    def act(
        self,
        observation: dict,
        candidates: Sequence[dict],
        high_level_mask: Sequence[bool] | dict[str, bool] | None = None,
        candidate_mask: Sequence[bool] | None = None,
        wm_predictions: dict[str, Any] | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        high_mask, low_masks = _scope_masks(candidates, high_level_mask, candidate_mask)
        high_logits, low_logits = self.actor_logits(observation, candidates)
        high_distribution = _distribution(high_logits, high_mask)
        if deterministic:
            high_index = int(torch.argmax(high_distribution.logits).item())
        else:
            high_index = int(high_distribution.sample().item())
        scope = SCOPE_NAMES[high_index]
        low_distribution = _distribution(low_logits, low_masks[scope])
        if deterministic:
            candidate_index = int(torch.argmax(low_distribution.logits).item())
        else:
            candidate_index = int(low_distribution.sample().item())
        candidate = candidates[candidate_index]
        action = copy.deepcopy(candidate.get("full_environment_action"))
        if not isinstance(action, dict):
            raise ValueError("EVENT_POLICY_FULL_ACTION_REQUIRED")
        if str(action.get("scope", "")).upper() != scope:
            raise ValueError("EVENT_POLICY_SCOPE_ACTION_BINDING")
        high_log_probability = high_distribution.log_prob(torch.tensor(high_index, device=high_logits.device))
        low_log_probability = low_distribution.log_prob(torch.tensor(candidate_index, device=low_logits.device))
        entropy = high_distribution.entropy() + low_distribution.entropy()
        value, base_value, auxiliary_value = self.critic_values(observation, wm_predictions)
        return {
            "scope": scope,
            "scope_index": high_index,
            "candidate_index": candidate_index,
            "candidate_key": str(candidate.get("candidate_key", candidate.get("id", candidate_index))),
            "full_environment_action": action,
            "high_logits": high_logits.detach().cpu().tolist(),
            "low_logits": low_logits.detach().cpu().tolist(),
            "high_level_mask": list(high_mask),
            "candidate_mask": list(low_masks[scope]),
            "high_log_probability": float(high_log_probability.detach().cpu().item()),
            "low_log_probability": float(low_log_probability.detach().cpu().item()),
            "log_probability": float((high_log_probability + low_log_probability).detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "value": float(value.detach().cpu().item()),
            "base_value": float(base_value.detach().cpu().item()),
            "auxiliary_value": float(auxiliary_value.detach().cpu().item()),
        }

    def evaluate_action(
        self,
        observation: dict,
        candidates: Sequence[dict],
        *,
        scope_index: int,
        candidate_index: int,
        high_level_mask: Sequence[bool] | dict[str, bool] | None,
        candidate_mask: Sequence[bool] | None,
        wm_predictions: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        high_mask, low_masks = _scope_masks(candidates, high_level_mask, candidate_mask)
        if scope_index < 0 or scope_index >= len(SCOPE_NAMES) or candidate_index < 0 or candidate_index >= len(candidates):
            raise ValueError("EVENT_POLICY_ACTION_INDEX")
        scope = SCOPE_NAMES[scope_index]
        if not high_mask[scope_index] or not low_masks[scope][candidate_index]:
            raise ValueError("EVENT_POLICY_SAMPLED_ACTION_MASK")
        high_logits, low_logits = self.actor_logits(observation, candidates)
        high_distribution = _distribution(high_logits, high_mask)
        low_distribution = _distribution(low_logits, low_masks[scope])
        high_logp = high_distribution.log_prob(torch.tensor(scope_index, device=high_logits.device))
        low_logp = low_distribution.log_prob(torch.tensor(candidate_index, device=low_logits.device))
        value, base_value, auxiliary_value = self.critic_values(observation, wm_predictions)
        return {"log_probability": high_logp + low_logp, "high_log_probability": high_logp, "low_log_probability": low_logp, "entropy": high_distribution.entropy() + low_distribution.entropy(), "value": value, "base_value": base_value, "auxiliary_value": auxiliary_value}


def build_hierarchical_policy(seed: int, use_wm_aux: bool = False, *, hidden_dim: int = POLICY_HIDDEN_DIM) -> HierarchicalGraphPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        policy = HierarchicalGraphPolicy(use_wm_aux=use_wm_aux, hidden_dim=hidden_dim)
    policy.train()
    return policy


def assert_paired_actor_initialization(policies: Sequence[HierarchicalGraphPolicy]) -> str:
    if len(policies) != len(VARIANTS):
        raise ValueError("EVENT_POLICY_PAIRED_VARIANT_COUNT")
    digests = {_actor_digest(policy) for policy in policies}
    if len(digests) != 1:
        raise ValueError("EVENT_POLICY_INITIAL_ACTOR_MISMATCH")
    return next(iter(digests))


def world_model_trigger_summary(predictions: dict[str, Any], candidates: Sequence[dict] | None = None) -> dict[str, Any]:
    means = _mapping(predictions.get("means")); variances = _mapping(predictions.get("variances"))
    utility_rows = means.get("utility", [])
    variance_values: list[float] = []
    for head in WORLD_MODEL_HEADS:
        rows = variances.get(head, [])
        for row in rows:
            values = list(row) if isinstance(row, (list, tuple)) else [row]
            variance_values.extend(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
    noop_utility = 0.0
    keys = predictions.get("candidate_keys", [])
    noop_index = None
    if candidates is not None:
        noop_index = next((index for index, candidate in enumerate(candidates) if _is_noop(candidate)), None)
    elif isinstance(keys, list):
        noop_index = next((index for index, row in enumerate(predictions.get("per_candidate", [])) if _mapping(row).get("candidate_key") == "NOOP"), None)
    if noop_index is not None and noop_index < len(utility_rows):
        value = utility_rows[noop_index]
        noop_utility = float(sum(value) / len(value)) if isinstance(value, (list, tuple)) and value else float(value)
    valid_indices = [index for index, row in enumerate(predictions.get("per_candidate", [])) if bool(_mapping(row).get("valid", True))]
    if candidates is not None:
        valid_indices = [index for index, candidate in enumerate(candidates) if bool(candidate.get("legal", True))]
    variance_values = []
    for head in WORLD_MODEL_HEADS:
        for index in valid_indices:
            rows = variances.get(head, [])
            if index >= len(rows):
                continue
            row = rows[index]
            values = list(row) if isinstance(row, (list, tuple)) else [row]
            variance_values.extend(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
    uncertainty = math.sqrt(sum(variance_values) / max(1, len(variance_values)))
    return {
        "d_degradation": max(0.0, -noop_utility),
        "d_uncertainty": uncertainty,
        "utility_noop": noop_utility,
        "candidate_count": len(keys),
        "model_hash": str(predictions.get("ensemble_hash", "")),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _import_event_apis():
    try:
        from .event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
        from .event_trigger import TriggerEngine, identify_affected_scope, high_level_action_mask
    except ImportError:
        from event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
        from event_trigger import TriggerEngine, identify_affected_scope, high_level_action_mask
    return EventReliefEnv, generate_reconstruction_candidates, repair_action, TriggerEngine, identify_affected_scope, high_level_action_mask


def _physical_action_key(candidate: dict) -> str:
    action = _mapping(candidate.get("full_environment_action"))
    payload = {
        "queue_operations": action.get("queue_operations", []),
        "dispatches": sorted(
            action.get("dispatches", []),
            key=lambda row: (str(_mapping(row).get("task_id", "")), str(_mapping(row).get("platform_id", ""))),
        ),
    }
    return json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _generate_policy_catalog(
    env: Any,
    generate_candidates: Any,
    max_candidates: int,
    affected_scope: dict | None = None,
) -> list[dict]:
    local: list[dict] = []
    if affected_scope is not None and bool(affected_scope.get("local_eligible")):
        local = generate_candidates(env, scope=affected_scope, max_candidates=max_candidates)
    global_rows = generate_candidates(env, scope="GLOBAL_REPLAN", max_candidates=max_candidates)
    selected_by_physical: dict[str, dict] = {}
    key_to_physical: dict[str, str] = {}
    for candidate in list(local) + list(global_rows):
        key = str(candidate.get("candidate_key", ""))
        if not key:
            raise ValueError("EVENT_POLICY_CANDIDATE_IDENTITY")
        physical_key = _physical_action_key(candidate)
        if key in key_to_physical and key_to_physical[key] != physical_key:
            raise ValueError("EVENT_POLICY_DUPLICATE_CANDIDATE_IDENTITY")
        key_to_physical[key] = physical_key
        prior = selected_by_physical.get(physical_key)
        if prior is None or (not bool(prior.get("legal", True)) and bool(candidate.get("legal", True))):
            selected_by_physical[physical_key] = candidate
    unique = list(selected_by_physical.values())
    noops = [candidate for candidate in unique if _is_noop(candidate)]
    local_rows = [candidate for candidate in unique if not _is_noop(candidate) and _candidate_scope(candidate) == "LOCAL_REPAIR"]
    global_candidates = [candidate for candidate in unique if not _is_noop(candidate) and _candidate_scope(candidate) == "GLOBAL_REPLAN"]
    merged = noops[:1]
    for offset in range(max(len(local_rows), len(global_candidates))):
        if offset < len(local_rows):
            merged.append(local_rows[offset])
        if offset < len(global_candidates):
            merged.append(global_candidates[offset])
    selected = merged[:max_candidates]
    if not any(_is_noop(candidate) for candidate in selected):
        raise ValueError("EVENT_POLICY_CATALOG_NOOP_REQUIRED")
    return selected


def _catalog_scope_fallback(affected_scope: dict, candidates: Sequence[dict]) -> tuple[dict, dict]:
    adjusted = copy.deepcopy(affected_scope)
    local_count = sum(
        not _is_noop(candidate) and _candidate_scope(candidate) == "LOCAL_REPAIR"
        for candidate in candidates
    )
    global_count = sum(
        not _is_noop(candidate) and _candidate_scope(candidate) == "GLOBAL_REPLAN"
        for candidate in candidates
    )
    if bool(adjusted.get("local_eligible")) and local_count == 0:
        receipt = {
            "status": "fallback",
            "code": "local_repair_no_feasible_solution",
            "local_non_noop_candidate_count": 0,
            "global_non_noop_candidate_count": global_count,
        }
        adjusted["local_eligible"] = False
        adjusted["global_eligible"] = True
        adjusted["scope"] = "GLOBAL_REPLAN"
        reasons = list(adjusted.get("reasons", []))
        if receipt["code"] not in reasons:
            reasons.append(receipt["code"])
        adjusted["reasons"] = reasons
    else:
        receipt = {
            "status": "available",
            "local_non_noop_candidate_count": local_count,
            "global_non_noop_candidate_count": global_count,
        }
    return adjusted, receipt


def _mask_from_trigger(decision: dict, detection: dict, affected: dict, high_level_action_mask: Any) -> list[bool]:
    if not bool(decision.get("triggered", False)):
        return [True, False, False]
    returned = high_level_action_mask(detection, affected)
    if isinstance(returned, dict):
        mask = [bool(returned.get(scope, False)) for scope in SCOPE_NAMES]
    elif isinstance(returned, (list, tuple)):
        mask = [bool(value) for value in returned]
    else:
        mask = [True, bool(affected.get("local_eligible", False)), bool(affected.get("global_eligible", False))]
    if len(mask) != len(SCOPE_NAMES):
        raise ValueError("EVENT_POLICY_TRIGGER_MASK_SHAPE")
    mask[0] = True
    return mask


def _variant_threshold_manifest(threshold_manifest: dict, variant: str) -> tuple[dict, dict]:
    use_world_model = variant.endswith("-WM")
    manifest = copy.deepcopy(threshold_manifest)
    final_threshold = float(manifest.get("threshold", manifest.get("tau", 0.5)))
    baseline_threshold = manifest.get("pre_model_threshold")
    if use_world_model:
        threshold, source = final_threshold, "final_world_model_training_calibration"
    elif baseline_threshold is not None:
        threshold, source = float(baseline_threshold), "initial_public_u0_training_calibration"
        manifest["threshold"] = threshold
        manifest["world_model_used"] = False
        manifest["wm_summaries_used"] = False
    else:
        threshold, source = final_threshold, "final_threshold_fallback_missing_pre_model_value"
        manifest["world_model_used"] = False
        manifest["wm_summaries_used"] = False
    return manifest, {
        "threshold": threshold,
        "source": source,
        "source_split": manifest.get("source_split"),
        "source_parent_ids": copy.deepcopy(manifest.get("source_parent_ids", [])),
        "quantile": manifest.get("quantile"),
        "weights": copy.deepcopy(manifest.get("weights", {})),
        "uses_world_model_trigger": use_world_model,
        "uses_world_model_auxiliary_critic": use_world_model,
    }


def _event_episode(
    scenario: dict,
    policy: HierarchicalGraphPolicy,
    *,
    variant: str,
    seed: int,
    threshold_manifest: dict,
    ensemble: WorldModelEnsemble,
    config: dict,
    training: bool,
    deterministic: bool,
) -> tuple[list[dict], dict]:
    EventReliefEnv, generate_candidates, repair_action, TriggerEngine, identify_scope, high_mask_fn = _import_event_apis()
    trigger_config = _mapping(config.get("trigger")) or config
    max_candidates = int(_mapping(config.get("collection")).get("max_candidates", config.get("max_candidates", 16)))
    use_world_model = variant.endswith("-WM")
    if policy.use_wm_aux != use_world_model:
        raise ValueError("EVENT_POLICY_VARIANT_WM_BINDING")
    selected_threshold_manifest, threshold_semantics = _variant_threshold_manifest(threshold_manifest, variant)
    env = EventReliefEnv(scenario)
    observation = env.reset()
    previous_observation = copy.deepcopy(observation)
    public_history = [copy.deepcopy(observation)]
    engine = TriggerEngine(trigger_config, selected_threshold_manifest)
    records: list[dict] = []
    episode_id = str(scenario.get("parent_id", scenario.get("scenario_id", "unknown")))
    started = time.perf_counter()
    terminated = truncated = False
    max_steps = int(scenario.get("horizon", 60))
    for step_index in range(max_steps):
        if terminated or truncated:
            break
        public_detection = engine.detect(previous_observation, observation, history=public_history[-4:])
        public_affected = identify_scope(observation, public_detection)
        candidates = _generate_policy_catalog(env, generate_candidates, max_candidates, public_affected)
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("EVENT_POLICY_EMPTY_ACTION_CATALOG")
        if use_world_model:
            predictions = predict_ensemble(ensemble, public_history[-4:], candidates)
            wm_summary = world_model_trigger_summary(predictions, candidates)
            detection = engine.detect(previous_observation, observation, history=public_history[-4:], wm_summary=wm_summary)
        else:
            predictions = None
            wm_summary = None
            detection = public_detection
        affected = identify_scope(observation, detection)
        scope_fields = ("affected_task_ids", "affected_platform_ids", "local_eligible", "global_eligible")
        if any(affected.get(key) != public_affected.get(key) for key in scope_fields):
            raise ValueError("EVENT_POLICY_WM_SCOPE_CATALOG_MISMATCH")
        affected, scope_catalog_receipt = _catalog_scope_fallback(affected, candidates)
        mode = "periodic" if variant.startswith("Periodic") else "event"
        decision = engine.decide(detection, int(observation.get("time", step_index)), mode=mode)
        detection.update({
            "triggered": bool(decision.get("triggered", False)),
            "cooldown_suppressed": bool(decision.get("cooldown_suppressed", False)),
        })
        high_mask = _mask_from_trigger(decision, detection, affected, high_mask_fn)
        decision["affected_scope"] = copy.deepcopy(affected)
        decision["high_level_mask"] = list(high_mask)
        decision["scope_catalog_receipt"] = copy.deepcopy(scope_catalog_receipt)
        decision["high_level_action"] = (
            "GLOBAL_REPLAN" if decision["triggered"] and high_mask[2]
            else "LOCAL_REPAIR" if decision["triggered"] and high_mask[1]
            else "NOOP"
        )
        wm_for_critic = predictions if use_world_model else None
        policy_output = policy.act(observation, candidates, high_mask, None, wm_for_critic, deterministic=deterministic)
        selected_candidate = candidates[policy_output["candidate_index"]]
        repaired_action, receipt = repair_action(env, policy_output["full_environment_action"])
        if not isinstance(repaired_action, dict) or not isinstance(receipt, dict):
            raise ValueError("EVENT_POLICY_REPAIR_RECEIPT")
        next_observation, reward, terminated, truncated, info = env.step(repaired_action)
        if "effective_reconstruction_count" not in info:
            raise ValueError("EVENT_POLICY_EFFECTIVE_RECONSTRUCTION_METRIC_REQUIRED")
        engine.record_reconstruction(
            int(observation.get("time", step_index)),
            bool(int(info["effective_reconstruction_count"]) > 0),
        )
        record = {
            "schema": "n02-event-policy-transition/1",
            "variant": variant,
            "seed": seed,
            "parent_id": episode_id,
            "split": scenario.get("split"),
            "step": step_index,
            "time": observation.get("time"),
            "observation": copy.deepcopy(observation),
            "candidate_keys": [str(item.get("candidate_key", item.get("id", index))) for index, item in enumerate(candidates)],
            "candidates": copy.deepcopy(candidates),
            "detection": copy.deepcopy(detection),
            "affected_scope": copy.deepcopy(affected),
            "scope_catalog_receipt": copy.deepcopy(scope_catalog_receipt),
            "decision": copy.deepcopy(decision),
            "wm_trigger_summary": wm_summary,
            "wm_predictions": copy.deepcopy(predictions) if use_world_model else None,
            "world_model_use": {
                "trigger": use_world_model,
                "auxiliary_critic": policy.use_wm_aux,
            },
            "trigger_threshold_semantics": copy.deepcopy(threshold_semantics),
            "high_level_mask": list(policy_output["high_level_mask"]),
            "candidate_mask": list(policy_output["candidate_mask"]),
            "scope": policy_output["scope"],
            "scope_index": policy_output["scope_index"],
            "candidate_index": policy_output["candidate_index"],
            "candidate_key": policy_output["candidate_key"],
            "high_logits": policy_output["high_logits"],
            "low_logits": policy_output["low_logits"],
            "high_log_probability": policy_output["high_log_probability"],
            "low_log_probability": policy_output["low_log_probability"],
            "old_log_probability": policy_output["log_probability"],
            "old_value": policy_output["value"],
            "base_value": policy_output["base_value"],
            "auxiliary_value": policy_output["auxiliary_value"],
            "entropy": policy_output["entropy"],
            "selected_action": copy.deepcopy(policy_output["full_environment_action"]),
            "requested_non_noop_decision": not _is_noop(selected_candidate),
            "repair_receipt": copy.deepcopy(receipt),
            "executed_action": copy.deepcopy(repaired_action),
            "effective_reconstruction_count": int(info["effective_reconstruction_count"]),
            "operational_dispatch_count": int(info.get("operational_dispatch_count", 0)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "rollout_cut": False,
            "info": copy.deepcopy(info),
        }
        records.append(record)
        previous_observation = copy.deepcopy(observation)
        observation = next_observation
        public_history.append(copy.deepcopy(observation))
        public_history = public_history[-4:]
    if not records:
        raise ValueError("EVENT_POLICY_EMPTY_EPISODE")
    _attach_episode_gae(records)
    metrics = env.episode_metrics()
    trajectory = env.export_trajectory()
    if not isinstance(metrics, dict) or not isinstance(trajectory, dict) or not isinstance(trajectory.get("summary"), dict):
        raise ValueError("EVENT_POLICY_ENVIRONMENT_METRIC_SCHEMA")
    summary = {
        "parent_id": episode_id,
        "split": scenario.get("split"),
        "ticks": len(records),
        "total_reward": sum(record["reward"] for record in records),
        "trigger_count": sum(bool(_mapping(record.get("decision")).get("triggered")) for record in records),
        "trigger_invocation_count": sum(bool(_mapping(record.get("decision")).get("triggered")) for record in records),
        "requested_non_noop_decision_count": sum(bool(record["requested_non_noop_decision"]) for record in records),
        "executed_requested_reconstruction_count": int(metrics["requested_reconstruction_count"]),
        "effective_reconstruction_count": int(metrics["effective_reconstruction_count"]),
        "reconstruction_size": int(metrics["reconstruction_size"]),
        "operational_dispatch_count": int(metrics["operational_dispatch_count"]),
        "trigger_threshold_semantics": threshold_semantics,
        "world_model_use": {"trigger": use_world_model, "auxiliary_critic": policy.use_wm_aux},
        "environment_metrics": _jsonable(metrics),
        "trajectory_summary": _jsonable(trajectory["summary"]),
        "wall_seconds": time.perf_counter() - started,
    }
    return records, summary


def _attach_episode_gae(records: list[dict], *, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
    accumulator = 0.0
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        terminated = bool(record["terminated"])
        if terminated or index == len(records) - 1:
            next_value = 0.0
        else:
            next_value = float(records[index + 1]["old_value"])
        delta = float(record["reward"]) + gamma * next_value * (not terminated) - float(record["old_value"])
        carry = 0.0 if terminated or bool(record["truncated"]) else 1.0
        accumulator = delta + gamma * gae_lambda * carry * accumulator
        record["next_value"] = next_value
        record["advantage"] = float(accumulator)
        record["return"] = float(accumulator + float(record["old_value"]))


def _update_policy(
    policy: HierarchicalGraphPolicy,
    optimizer: torch.optim.Optimizer,
    records: list[dict],
    *,
    seed: int,
    epochs: int,
    minibatch_size: int,
    updates_per_epoch: int,
    learning_rate: float,
    clip_ratio: float,
    entropy_coefficient: float,
    update_offset: int = 0,
) -> list[dict]:
    if len(records) < minibatch_size * updates_per_epoch:
        raise ValueError("EVENT_POLICY_SHARED_UPDATE_SAMPLE_COUNT")
    advantages = torch.tensor([record["advantage"] for record in records], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([record["return"] for record in records], dtype=torch.float32)
    old_log_probs = torch.tensor([record["old_log_probability"] for record in records], dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 491)
    history: list[dict] = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(records), generator=generator)
        for batch_index in range(updates_per_epoch):
            indices = permutation[batch_index * minibatch_size:(batch_index + 1) * minibatch_size].tolist()
            evaluated = []
            for index in indices:
                row = records[index]
                evaluated.append(policy.evaluate_action(
                    row["observation"], row["candidates"],
                    scope_index=int(row["scope_index"]), candidate_index=int(row["candidate_index"]),
                    high_level_mask=row["high_level_mask"], candidate_mask=None,
                    wm_predictions=row.get("wm_predictions"),
                ))
            batch_logp = torch.stack([row["log_probability"] for row in evaluated])
            batch_values = torch.stack([row["value"] for row in evaluated])
            batch_aux_values = torch.stack([row["auxiliary_value"] for row in evaluated])
            batch_indices = torch.tensor(indices, dtype=torch.long)
            ratio = torch.exp(batch_logp - old_log_probs[batch_indices])
            if epoch == 0 and batch_index == 0 and float(torch.max(torch.abs(ratio.detach() - 1.0)).item()) > 2e-5:
                raise AssertionError("EVENT_POLICY_INITIAL_LOGPROB_RATIO")
            unclipped = ratio * advantages[batch_indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_indices]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(batch_values - returns[batch_indices]).mean()
            entropy = torch.stack([row["entropy"] for row in evaluated]).mean()
            loss = actor_loss + value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("EVENT_POLICY_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError("EVENT_POLICY_EMPTY_OR_NONFINITE_GRADIENT")
            aux_parameters = list(policy.auxiliary_critic.parameters()) if policy.auxiliary_critic is not None else []
            aux_gradient_norm = math.sqrt(sum(float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in aux_parameters if parameter.grad is not None))
            actor_parameters = list(policy.actor_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.high_actor.parameters()) + list(policy.low_actor.parameters())
            actor_gradient_norm = math.sqrt(sum(float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in actor_parameters if parameter.grad is not None))
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("EVENT_POLICY_NONFINITE_GRADIENT_NORM")
            before = _tensor_digest(policy.state_dict())
            optimizer.step()
            after = _tensor_digest(policy.state_dict())
            if before == after:
                raise AssertionError("EVENT_POLICY_OPTIMIZER_DID_NOT_UPDATE")
            history.append({
                "update": update_offset + len(history) + 1,
                "epoch": epoch + 1,
                "minibatch": batch_index,
                "sample_indices": indices,
                "sample_count": len(indices),
                "actor_loss": float(actor_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "auxiliary_value_abs_mean": float(batch_aux_values.detach().abs().mean().item()),
                "entropy": float(entropy.detach().item()),
                "total_loss": float(loss.detach().item()),
                "actor_gradient_norm_preclip": actor_gradient_norm,
                "auxiliary_critic_gradient_norm_preclip": aux_gradient_norm,
                "gradient_norm_preclip": float(gradient_norm.detach().item()),
                "parameter_digest_before": before,
                "parameter_digest_after": after,
                "optimizer_step_applied": True,
            })
    return history


def _catalog_digest(candidates: Sequence[dict]) -> str:
    canonical = json.dumps(_jsonable(list(candidates)), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _public_snapshot_digest(observation: dict) -> str:
    public = copy.deepcopy(observation)
    public.pop("provenance", None)
    canonical = json.dumps(_jsonable(public), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _initial_fixed_catalogs(development_scenarios: Sequence[dict], max_candidates: int) -> dict[str, dict]:
    EventReliefEnv, generate_candidates, _, TriggerEngine, identify_scope, _ = _import_event_apis()
    result: dict[str, dict] = {}
    for scenario in development_scenarios:
        parent_id = str(scenario.get("parent_id", scenario.get("scenario_id", "unknown")))
        env = EventReliefEnv(scenario)
        observation = env.reset()
        detection = TriggerEngine({}, {}).detect(None, observation)
        affected = identify_scope(observation, detection)
        candidates = _generate_policy_catalog(env, generate_candidates, max_candidates, affected)
        affected, fallback_receipt = _catalog_scope_fallback(affected, candidates)
        local_available = any(not _is_noop(item) and _candidate_scope(item) == "LOCAL_REPAIR" for item in candidates)
        global_available = any(not _is_noop(item) and _candidate_scope(item) == "GLOBAL_REPLAN" for item in candidates)
        result[parent_id] = {
            "observation": copy.deepcopy(observation),
            "candidates": copy.deepcopy(candidates),
            "affected_scope": copy.deepcopy(affected),
            "scope_catalog_receipt": copy.deepcopy(fallback_receipt),
            "high_level_mask": [True, bool(affected.get("local_eligible")) and local_available,
                                bool(affected.get("global_eligible")) and global_available],
            "snapshot_hash": _public_snapshot_digest(observation),
            "candidate_catalog_sha256": _catalog_digest(candidates),
            "candidate_keys": [str(item.get("candidate_key", item.get("id", index))) for index, item in enumerate(candidates)],
        }
    return result


def _save_policy_checkpoint(
    path: Path,
    policy: HierarchicalGraphPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    seed: int,
    config: dict,
    data_identity: str,
    wm_hash: str,
    update_log: list[dict],
    restore_probe: dict,
) -> dict:
    expected = policy.act(
        restore_probe["observation"], restore_probe["candidates"],
        restore_probe["high_level_mask"], None, restore_probe.get("wm_predictions"), deterministic=True,
    )
    probe_expected = {"high_logits": expected["high_logits"], "low_logits": expected["low_logits"], "scope": expected["scope"], "candidate_key": expected["candidate_key"], "log_probability": expected["log_probability"], "value": expected["value"]}
    payload = {
        "schema": POLICY_SCHEMA,
        "map_contract": MAP_CONTRACT,
        "map_rows": MAP_ROWS,
        "map_cols": MAP_COLS,
        "grid_cells": GRID_CELLS,
        "variant": variant,
        "seed": seed,
        "use_wm_aux": policy.use_wm_aux,
        "hidden_dim": policy.hidden_dim,
        "training_config": copy.deepcopy(config),
        "data_identity": data_identity,
        # This is the checkpoint/file identity supplied by the loaded ensemble.
        # Live tensor digests are audited separately by train_four_variants.
        "world_model_hash": wm_hash,
        "world_model_ensemble_identity": wm_hash,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "model_state_sha256": _tensor_digest(policy.state_dict()),
        "optimizer_updates": len(update_log),
        "update_log_digest": hashlib.sha256(json.dumps(update_log, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        "restore_probe": {"observation": restore_probe["observation"], "candidates": restore_probe["candidates"], "high_level_mask": restore_probe["high_level_mask"], "wm_predictions": restore_probe.get("wm_predictions"), "expected": probe_expected},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state_sha256": payload["model_state_sha256"], "optimizer_updates": len(update_log),
        "world_model_hash": wm_hash, "world_model_ensemble_identity": wm_hash,
    }


def load_policy_checkpoint(path: str | Path, expected_variant: str | None = None) -> tuple[HierarchicalGraphPolicy, dict[str, Any]]:
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA or payload.get("variant") not in VARIANTS:
        raise ValueError("EVENT_POLICY_CHECKPOINT_SCHEMA")
    if payload.get("map_contract") != MAP_CONTRACT or payload.get("map_rows") != MAP_ROWS or payload.get("map_cols") != MAP_COLS or payload.get("grid_cells") != GRID_CELLS:
        raise ValueError("EVENT_POLICY_CHECKPOINT_MAP_CONTRACT")
    if expected_variant is not None and payload.get("variant") != expected_variant:
        raise ValueError("EVENT_POLICY_CHECKPOINT_VARIANT")
    policy = build_hierarchical_policy(int(payload["seed"]), bool(payload["use_wm_aux"]), hidden_dim=int(payload["hidden_dim"]))
    policy.load_state_dict(payload["model_state"], strict=True)
    if _tensor_digest(policy.state_dict()) != payload.get("model_state_sha256"):
        raise ValueError("EVENT_POLICY_CHECKPOINT_STATE_DIGEST")
    policy.eval()
    return policy, payload


@torch.no_grad()
def verify_policy_checkpoint(path: str | Path, ensemble: WorldModelEnsemble | None = None) -> dict[str, Any]:
    policy, payload = load_policy_checkpoint(path)
    probe = payload.get("restore_probe")
    if not isinstance(probe, dict):
        raise ValueError("EVENT_POLICY_RESTORE_PROBE_REQUIRED")
    if policy.use_wm_aux and probe.get("wm_predictions") is None:
        if ensemble is None:
            raise ValueError("EVENT_POLICY_RESTORE_WM_REQUIRED")
        predictions = predict_ensemble(ensemble, [probe["observation"]], probe["candidates"])
    else:
        predictions = probe.get("wm_predictions")
    restored = policy.act(probe["observation"], probe["candidates"], probe["high_level_mask"], None, predictions, deterministic=True)
    expected = probe["expected"]
    if restored["scope"] != expected["scope"] or restored["candidate_key"] != expected["candidate_key"]:
        raise ValueError("EVENT_POLICY_RESTORE_ACTION_MISMATCH")
    for name in ("high_logits", "low_logits"):
        observed = torch.tensor(restored[name], dtype=torch.float64)
        target = torch.tensor(expected[name], dtype=torch.float64)
        if observed.shape != target.shape or not torch.equal(observed, target):
            raise ValueError(f"EVENT_POLICY_RESTORE_PREDICTION_MISMATCH:{name}")
    if abs(restored["log_probability"] - float(expected["log_probability"])) > 1e-8 or abs(restored["value"] - float(expected["value"])) > 1e-8:
        raise ValueError("EVENT_POLICY_RESTORE_VALUE_MISMATCH")
    return {"status": "complete", "path": str(Path(path).resolve()), "variant": payload["variant"], "seed": payload["seed"], "model_state_sha256": payload["model_state_sha256"], "probe_prediction_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(), "action": restored["candidate_key"], "scope": restored["scope"], "actor_forward_calls": policy.actor_forward_calls, "critic_forward_calls": policy.critic_forward_calls}


def _paired_snapshot_probe(
    policy: HierarchicalGraphPolicy,
    snapshot: dict,
    *,
    wm_predictions: dict | None,
) -> dict:
    observation, candidates = snapshot["observation"], snapshot["candidates"]
    mask = list(snapshot["high_level_mask"])
    output = policy.act(observation, candidates, mask, None, wm_predictions, deterministic=True)
    raw_high, raw_low = policy.actor_logits(observation, candidates)
    ranks = torch.argsort(torch.argsort(raw_low, descending=True)).detach().cpu().tolist()
    return {
        "snapshot_hash": snapshot["snapshot_hash"],
        "candidate_catalog_sha256": snapshot["candidate_catalog_sha256"],
        "candidate_keys": list(snapshot["candidate_keys"]),
        "high_level_mask": mask,
        "high_logits": raw_high.detach().cpu().tolist(),
        "candidate_logits": raw_low.detach().cpu().tolist(),
        "candidate_rank_zero_best": ranks,
        "affected_scope": copy.deepcopy(snapshot["affected_scope"]),
        "scope_catalog_receipt": copy.deepcopy(snapshot["scope_catalog_receipt"]),
        "selected_scope": output["scope"],
        "selected_candidate_key": output["candidate_key"],
        "wm_auxiliary_value": output["auxiliary_value"],
    }


def train_four_variants(
    scenarios: Sequence[dict],
    ensemble: WorldModelEnsemble,
    threshold_manifest: dict,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train four paired event policies and evaluate held-out development parents."""
    if len(scenarios) != 6:
        raise ValueError("EVENT_POLICY_SIX_PARENT_SCENARIOS_REQUIRED")
    config = dict(config or {})
    ppo = _mapping(config.get("ppo"))
    seeds = tuple(ppo.get("seeds", POLICY_SEEDS))
    if seeds != POLICY_SEEDS:
        raise ValueError("EVENT_POLICY_SEEDS_FROZEN")
    epochs = int(ppo.get("epochs", 4))
    rounds = int(ppo.get("rounds", 4))
    minibatch_size = int(ppo.get("minibatch_size", 32))
    max_updates = int(ppo.get("max_updates", 128))
    learning_rate = float(ppo.get("learning_rate", 3e-4))
    gamma = float(ppo.get("gamma", 0.99))
    gae_lambda = float(ppo.get("gae_lambda", 0.95))
    clip_ratio = float(ppo.get("clip_ratio", 0.2))
    entropy_coefficient = float(ppo.get("entropy_coefficient", 0.01))
    if epochs != 4 or rounds != 4 or minibatch_size != 32 or max_updates != 128 or learning_rate <= 0 or not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("EVENT_POLICY_PPO_CONFIG_FROZEN")
    training_scenarios = [scenario for scenario in scenarios if scenario.get("split") == "train"]
    development_scenarios = [scenario for scenario in scenarios if scenario.get("split") == "development"]
    if len(training_scenarios) != 4 or len(development_scenarios) != 2:
        raise ValueError("EVENT_POLICY_SPLIT_COUNTS")
    training_parent_ids = sorted(str(item.get("parent_id", item.get("scenario_id", "unknown"))) for item in training_scenarios)
    if threshold_manifest.get("source_split") not in (None, "train"):
        raise ValueError("EVENT_POLICY_THRESHOLD_NOT_TRAINING_ONLY")
    if threshold_manifest.get("source_parent_ids") and sorted(str(value) for value in threshold_manifest["source_parent_ids"]) != training_parent_ids:
        raise ValueError("EVENT_POLICY_THRESHOLD_PARENT_MISMATCH")
    pre_model_threshold = threshold_manifest.get("pre_model_threshold")
    if pre_model_threshold is None:
        pre_model_threshold = threshold_manifest.get("threshold", threshold_manifest.get("tau", 0.5))
    variant_threshold_semantics = {
        variant: _variant_threshold_manifest(threshold_manifest, variant)[1]
        for variant in VARIANTS
    }
    max_candidates = int(_mapping(config.get("collection")).get("max_candidates", 16))
    fixed_snapshots = _initial_fixed_catalogs(development_scenarios, max_candidates)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Keep the checkpoint/file identity separate from the live tensors. The
    # former is immutable metadata; only the latter can prove the freeze audit.
    ensemble_hash_before = ensemble.ensemble_hash
    world_model_state_digests_before = _world_model_member_state_digests(ensemble)
    if any(parameter.requires_grad for model in ensemble.models for parameter in model.parameters()):
        raise ValueError("EVENT_POLICY_WORLD_MODEL_NOT_FROZEN")
    variant_policies: dict[str, dict[int, HierarchicalGraphPolicy]] = {variant: {} for variant in VARIANTS}
    variant_optimizers: dict[str, dict[int, torch.optim.Optimizer]] = {variant: {} for variant in VARIANTS}
    initial_actor_digests: dict[int, str] = {}
    training_summaries: dict[str, dict[int, list[dict]]] = {variant: {seed: [] for seed in seeds} for variant in VARIANTS}
    update_logs: dict[str, dict[int, list[dict]]] = {variant: {seed: [] for seed in seeds} for variant in VARIANTS}
    transition_counts: dict[str, dict[int, list[int]]] = {variant: {seed: [] for seed in seeds} for variant in VARIANTS}
    forward_counts: dict[str, dict[int, dict[str, int]]] = {
        variant: {seed: {
            "training_collection_actor": 0, "training_collection_critic": 0,
            "ppo_update_actor": 0, "ppo_update_critic": 0,
            "fixed_snapshot_actor": 0, "fixed_snapshot_critic": 0,
            "development_actor": 0, "development_critic": 0,
            "checkpoint_probe_actor": 0, "checkpoint_probe_critic": 0,
        } for seed in seeds} for variant in VARIANTS
    }
    rollout_paths: dict[str, dict[int, Path]] = {variant: {} for variant in VARIANTS}
    update_paths: dict[str, dict[int, Path]] = {variant: {} for variant in VARIANTS}
    data_identity = hashlib.sha256(json.dumps([{"parent_id": item.get("parent_id"), "split": item.get("split"), "tape": _public_observation(item.get("public_initial", item)).get("provenance", {}).get("event_tape_id")} for item in training_scenarios], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    # Every method receives a byte-identical actor initialization per seed.
    for seed in seeds:
        reference = build_hierarchical_policy(seed, use_wm_aux=False)
        actor_digests = []
        for variant in VARIANTS:
            use_wm = variant.endswith("-WM")
            policy = build_hierarchical_policy(seed, use_wm_aux=use_wm)
            policy.load_state_dict(reference.state_dict(), strict=False)
            policy.variant = variant
            actor_digests.append(_actor_digest(policy))
            variant_policies[variant][seed] = policy
            variant_optimizers[variant][seed] = torch.optim.Adam(policy.parameters(), lr=learning_rate)
            run_dir = output / variant / f"seed_{seed}"
            rollout_paths[variant][seed] = run_dir / "rollouts.jsonl"
            update_paths[variant][seed] = run_dir / "ppo_updates.jsonl"
            _write_jsonl(rollout_paths[variant][seed], [])
            _write_jsonl(update_paths[variant][seed], [])
        if len(set(actor_digests)) != 1:
            raise ValueError("EVENT_POLICY_PAIRED_ACTOR_INIT")
        initial_actor_digests[seed] = actor_digests[0]

    # Persist one round at a time and optimize only from real transitions in that
    # round, keeping every method on the same parent order and update count.
    ensemble_before_training_forward = ensemble.forward_calls
    ensemble_before_training_predictions = ensemble.prediction_calls
    round_minimums: list[int] = []
    updates_per_epoch_by_round: list[int] = []
    update_cap_per_epoch = max_updates // (epochs * rounds)
    if update_cap_per_epoch <= 0:
        raise ValueError("EVENT_POLICY_PPO_UPDATE_CAP_TOO_SMALL")
    for round_index in range(rounds):
        round_files: dict[str, dict[int, Path]] = {variant: {} for variant in VARIANTS}
        round_counts: list[int] = []
        for seed in seeds:
            for variant in VARIANTS:
                policy = variant_policies[variant][seed]
                run_dir = output / variant / f"seed_{seed}"
                round_path = run_dir / "rollout_rounds" / f"round_{round_index + 1:02d}.jsonl"
                _write_jsonl(round_path, [])
                round_files[variant][seed] = round_path
                round_count = 0
                round_seed = seed + 100_003 * round_index
                random.seed(round_seed)
                torch.manual_seed(round_seed)
                actor_before = policy.actor_forward_calls
                critic_before = policy.critic_forward_calls
                for episode_index, scenario in enumerate(training_scenarios):
                    episode_records, summary = _event_episode(
                        scenario, policy, variant=variant, seed=seed,
                        threshold_manifest=threshold_manifest, ensemble=ensemble,
                        config=config, training=True, deterministic=False,
                    )
                    _attach_episode_gae(episode_records, gamma=gamma, gae_lambda=gae_lambda)
                    for row in episode_records:
                        row["round"] = round_index + 1
                        row["episode_index"] = episode_index
                    _append_jsonl(round_path, episode_records)
                    round_count += len(episode_records)
                    summary.update({"round": round_index + 1, "episode_index": episode_index})
                    training_summaries[variant][seed].append(summary)
                forward_counts[variant][seed]["training_collection_actor"] += policy.actor_forward_calls - actor_before
                forward_counts[variant][seed]["training_collection_critic"] += policy.critic_forward_calls - critic_before
                transition_counts[variant][seed].append(round_count)
                round_counts.append(round_count)
        min_records = min(round_counts)
        round_minimums.append(min_records)
        updates_per_epoch = min(update_cap_per_epoch, min_records // minibatch_size)
        if updates_per_epoch <= 0:
            raise ValueError("EVENT_POLICY_INSUFFICIENT_REAL_TRANSITIONS_FOR_PPO")
        updates_per_epoch_by_round.append(updates_per_epoch)
        round_updates = updates_per_epoch * epochs
        for seed in seeds:
            for variant in VARIANTS:
                policy = variant_policies[variant][seed]
                optimizer = variant_optimizers[variant][seed]
                records = _read_jsonl(round_files[variant][seed])
                actor_before = policy.actor_forward_calls
                critic_before = policy.critic_forward_calls
                update_log = _update_policy(
                    policy, optimizer, records, seed=seed + 100_003 * round_index,
                    epochs=epochs, minibatch_size=minibatch_size,
                    updates_per_epoch=updates_per_epoch, learning_rate=learning_rate,
                    clip_ratio=clip_ratio, entropy_coefficient=entropy_coefficient,
                    update_offset=len(update_logs[variant][seed]),
                )
                forward_counts[variant][seed]["ppo_update_actor"] += policy.actor_forward_calls - actor_before
                forward_counts[variant][seed]["ppo_update_critic"] += policy.critic_forward_calls - critic_before
                update_logs[variant][seed].extend(update_log)
                _append_jsonl(rollout_paths[variant][seed], records)
                _append_jsonl(update_paths[variant][seed], update_log)
                if len(update_log) != round_updates:
                    raise ValueError("EVENT_POLICY_ROUND_UPDATE_COUNT_MISMATCH")

    common_updates = len(update_logs[VARIANTS[0]][seeds[0]])
    if any(len(update_logs[variant][seed]) != common_updates for variant in VARIANTS for seed in seeds):
        raise ValueError("EVENT_POLICY_PAIRED_UPDATE_BUDGET_MISMATCH")
    training_ensemble_forward_calls = ensemble.forward_calls - ensemble_before_training_forward
    training_ensemble_prediction_calls = ensemble.prediction_calls - ensemble_before_training_predictions
    results: dict[str, dict[int, dict]] = {variant: {} for variant in VARIANTS}
    for seed in seeds:
        for variant in VARIANTS:
            policy = variant_policies[variant][seed]
            optimizer = variant_optimizers[variant][seed]
            update_log = update_logs[variant][seed]
            records = _read_jsonl(rollout_paths[variant][seed])
            first = records[0]
            run_dir = output / variant / f"seed_{seed}"
            actor_before = policy.actor_forward_calls
            critic_before = policy.critic_forward_calls
            checkpoint = _save_policy_checkpoint(
                run_dir / "checkpoint.pt", policy, optimizer,
                variant=variant, seed=seed,
                config={"ppo": ppo, "data_identity": data_identity},
                data_identity=data_identity, wm_hash=ensemble_hash_before,
                update_log=update_log,
                restore_probe={"observation": first["observation"], "candidates": first["candidates"], "high_level_mask": first["high_level_mask"], "wm_predictions": first.get("wm_predictions")},
            )
            forward_counts[variant][seed]["checkpoint_probe_actor"] += policy.actor_forward_calls - actor_before
            forward_counts[variant][seed]["checkpoint_probe_critic"] += policy.critic_forward_calls - critic_before
            update_summary = {
                "optimizer_updates": len(update_log),
                "transition_count": sum(transition_counts[variant][seed]),
                "round_transition_counts": transition_counts[variant][seed],
                "actor_loss_mean": sum(row["actor_loss"] for row in update_log) / len(update_log),
                "value_loss_mean": sum(row["value_loss"] for row in update_log) / len(update_log),
                "auxiliary_gradient_nonzero_updates": sum(row["auxiliary_critic_gradient_norm_preclip"] > 0.0 for row in update_log),
                "actor_gradient_nonzero_updates": sum(row["actor_gradient_norm_preclip"] > 0.0 for row in update_log),
                "initial_actor_digest": initial_actor_digests[seed],
                "final_actor_digest": _actor_digest(policy),
                "world_model_hash_before": ensemble_hash_before,
                "world_model_hash_after": ensemble.ensemble_hash,
                "world_model_ensemble_identity": ensemble_hash_before,
                "world_model_state_digests_before": copy.deepcopy(world_model_state_digests_before),
                "world_model_state_digests_after": None,
                "world_model_parameter_changed": None,
                "world_model_frozen": None,
                "world_model_use": {
                    "trigger": variant.endswith("-WM"),
                    "auxiliary_critic": policy.use_wm_aux,
                    "threshold": variant_threshold_semantics[variant],
                },
                "forward_counts": forward_counts[variant][seed],
            }
            if variant.endswith("-WM") and update_summary["auxiliary_gradient_nonzero_updates"] != common_updates:
                raise ValueError("EVENT_POLICY_AUXILIARY_CRITIC_GRADIENT_MISSING")
            results[variant][seed] = {"checkpoint": checkpoint, "training": {"episodes": training_summaries[variant][seed], **update_summary}, "fixed_snapshot_probes": []}

    # Evaluate every trained method on exactly the same initial public snapshots
    # and catalogs before reporting its separate closed-loop development path.
    fixed_ensemble_forward_calls = 0
    fixed_ensemble_prediction_calls = 0
    development_ensemble_forward_calls = 0
    development_ensemble_prediction_calls = 0
    for seed in seeds:
        for variant in VARIANTS:
            policy = variant_policies[variant][seed]
            for parent_id, snapshot in fixed_snapshots.items():
                actor_before = policy.actor_forward_calls
                critic_before = policy.critic_forward_calls
                model_forward_before = ensemble.forward_calls
                model_prediction_before = ensemble.prediction_calls
                wm_predictions = (
                    predict_ensemble(ensemble, [snapshot["observation"]], snapshot["candidates"])
                    if policy.use_wm_aux else None
                )
                fixed_ensemble_forward_calls += ensemble.forward_calls - model_forward_before
                fixed_ensemble_prediction_calls += ensemble.prediction_calls - model_prediction_before
                probe = _paired_snapshot_probe(policy, snapshot, wm_predictions=wm_predictions)
                forward_counts[variant][seed]["fixed_snapshot_actor"] += policy.actor_forward_calls - actor_before
                forward_counts[variant][seed]["fixed_snapshot_critic"] += policy.critic_forward_calls - critic_before
                probe["parent_id"] = parent_id
                probe["variant"] = variant
                probe["seed"] = seed
                results[variant][seed]["fixed_snapshot_probes"].append(probe)
            closed_loop = []
            for scenario in development_scenarios:
                actor_before = policy.actor_forward_calls
                critic_before = policy.critic_forward_calls
                model_forward_before = ensemble.forward_calls
                model_prediction_before = ensemble.prediction_calls
                rows, summary = _event_episode(
                    scenario, policy, variant=variant, seed=seed,
                    threshold_manifest=threshold_manifest, ensemble=ensemble,
                    config=config, training=False, deterministic=True,
                )
                development_ensemble_forward_calls += ensemble.forward_calls - model_forward_before
                development_ensemble_prediction_calls += ensemble.prediction_calls - model_prediction_before
                forward_counts[variant][seed]["development_actor"] += policy.actor_forward_calls - actor_before
                forward_counts[variant][seed]["development_critic"] += policy.critic_forward_calls - critic_before
                summary["trajectory_action_digest"] = hashlib.sha256(json.dumps([row["executed_action"] for row in rows], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                closed_loop.append(summary)
                _write_jsonl(output / variant / f"seed_{seed}" / f"development_{summary['parent_id']}.jsonl", rows)
            results[variant][seed]["development"] = closed_loop
            # Full-policy checkpoint restore performs a real inference check.
            restore = verify_policy_checkpoint(results[variant][seed]["checkpoint"]["path"], ensemble=ensemble)
            results[variant][seed]["checkpoint"]["restore_check"] = restore
            results[variant][seed]["checkpoint"]["forward_counts"] = {
                "actor": restore["actor_forward_calls"], "critic": restore["critic_forward_calls"],
            }
            results[variant][seed]["training"]["forward_counts"] = forward_counts[variant][seed]
            results[variant][seed]["training"]["forward_counts"]["total_actor"] = policy.actor_forward_calls
            results[variant][seed]["training"]["forward_counts"]["total_critic"] = policy.critic_forward_calls
    world_model_state_digests_after = _world_model_member_state_digests(ensemble)
    world_model_parameter_changed = world_model_state_digests_before != world_model_state_digests_after
    if world_model_parameter_changed:
        raise ValueError("EVENT_POLICY_WORLD_MODEL_STATE_CHANGED")
    if any(parameter.requires_grad for model in ensemble.models for parameter in model.parameters()):
        raise ValueError("EVENT_POLICY_WORLD_MODEL_NOT_FROZEN_AFTER_INFERENCE")
    if ensemble.ensemble_hash != ensemble_hash_before:
        raise ValueError("EVENT_POLICY_WORLD_MODEL_HASH_CHANGED")
    for variant in VARIANTS:
        for seed in seeds:
            training = results[variant][seed]["training"]
            training.update({
                "world_model_state_digests_after": copy.deepcopy(world_model_state_digests_after),
                "world_model_parameter_changed": world_model_parameter_changed,
                "world_model_frozen": not world_model_parameter_changed,
            })
    catalog_manifest = {parent: {key: value for key, value in snapshot.items() if key not in {"observation", "candidates"}} for parent, snapshot in fixed_snapshots.items()}
    final = {
        "status": "complete",
        "schema": POLICY_SCHEMA,
        "variants": {variant: {str(seed): results[variant][seed] for seed in seeds} for variant in VARIANTS},
        "fixed_snapshot_catalogs": catalog_manifest,
        "paired_optimizer_updates_per_seed_method": common_updates,
        "updates_per_epoch_by_round": updates_per_epoch_by_round,
        "transition_count_minimum_by_round": round_minimums,
        "ensemble_forward_counts": {
            "scope": "calls executed by train_four_variants",
            "training_rollout_and_trigger_forwards": training_ensemble_forward_calls,
            "training_rollout_and_trigger_predictions": training_ensemble_prediction_calls,
            "fixed_snapshot_forwards": fixed_ensemble_forward_calls,
            "fixed_snapshot_predictions": fixed_ensemble_prediction_calls,
            "development_forwards": development_ensemble_forward_calls,
            "development_predictions": development_ensemble_prediction_calls,
            "total_forwards": training_ensemble_forward_calls + fixed_ensemble_forward_calls + development_ensemble_forward_calls,
            "total_predictions": training_ensemble_prediction_calls + fixed_ensemble_prediction_calls + development_ensemble_prediction_calls,
        },
        # Checkpoint/file identity remains distinct from the live state audit.
        "world_model_hash": ensemble_hash_before,
        "world_model_ensemble_identity": ensemble_hash_before,
        "world_model_state_digests_before": world_model_state_digests_before,
        "world_model_state_digests_after": world_model_state_digests_after,
        "world_model_parameter_changed": world_model_parameter_changed,
        "world_model_frozen": not world_model_parameter_changed,
        "threshold_calibration": {
            "source_split": threshold_manifest.get("source_split", "train"),
            "source_parent_ids": training_parent_ids,
            "quantile": threshold_manifest.get("quantile"),
            "weights": copy.deepcopy(threshold_manifest.get("weights", {})),
            "pre_model_u0_threshold": float(pre_model_threshold),
            "world_model_calibrated_threshold": float(threshold_manifest.get("threshold", threshold_manifest.get("tau", 0.5))),
            "initial_threshold_manifest_sha256": threshold_manifest.get("initial_threshold_sha256"),
            "wm_threshold_manifest_sha256": threshold_manifest.get("manifest_sha256"),
            "variant_thresholds": variant_threshold_semantics,
            "training_only": True,
        },
        "data_identity": data_identity,
        "config": {"seeds": list(seeds), "rounds": rounds, "episodes_per_round": len(training_scenarios), "epochs": epochs, "minibatch_size": minibatch_size, "max_updates": max_updates, "updates_per_epoch_by_round": updates_per_epoch_by_round, "learning_rate": learning_rate, "gamma": gamma, "gae_lambda": gae_lambda, "clip_ratio": clip_ratio, "entropy_coefficient": entropy_coefficient},
    }
    (output / "EVENT_POLICY_TRAINING.json").write_text(json.dumps(_jsonable(final), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return final
