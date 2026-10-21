"""Identity-bound cache for detached full-proposal world-model predictions."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:
    from .stable_identity import key_digest
except ImportError:
    from stable_identity import key_digest


SCHEMA = "n2-frozen-wm-prediction-cache/1"
DEFAULT_MAX_ENTRIES = 2048


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _proposal_rows(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("WM_CACHE_PROPOSAL_RECORD")
        actions = proposal.get("actions")
        candidate_id = proposal.get("candidate_id")
        if (
            not isinstance(actions, list)
            or not actions
            or any(type(action) is not str for action in actions)
            or len(actions) != len(set(actions))
            or ("NOOP" in actions and actions[-1] != "NOOP")
            or candidate_id != key_digest(actions)
            or candidate_id in seen
        ):
            raise ValueError("WM_CACHE_COMPLETE_PROPOSAL_IDENTITY")
        seen.add(candidate_id)
        rows.append({"candidate_id": candidate_id, "actions": list(actions)})
    if not rows or sum(row["actions"] == ["NOOP"] for row in rows) != 1:
        raise ValueError("WM_CACHE_NOOP_PROPOSAL_COVERAGE")
    return sorted(rows, key=lambda row: row["candidate_id"])


def _layout_identity(
    tensors: dict[str, torch.Tensor], binding: Any,
) -> dict[str, Any]:
    candidate_ids = tuple(binding.candidate_ids[0])
    task_ids = tuple(binding.task_ids[0])
    active_candidates = tensors["candidate_mask"][0].detach().cpu().tolist()
    active_tasks = tensors["task_mask"][0].detach().cpu().tolist()
    return {
        "candidate_ids": sorted(candidate_ids),
        "task_ids": list(task_ids),
        "tensor_layout": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(tensors.items())
        },
        "active_candidate_count": sum(bool(value) for value in active_candidates),
        "active_task_count": sum(bool(value) for value in active_tasks),
    }


@dataclass(frozen=True)
class PredictionBatch:
    by_candidate_id: dict[str, torch.Tensor]
    cache_key_sha256: str
    cache_hit: bool
    forward_batches: int
    candidate_evaluations: int


class FrozenWmPredictionCache:
    """Cache one detached prediction tensor per complete public proposal set."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("WM_CACHE_MAX_ENTRIES")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._forward_batches = 0
        self._candidate_evaluations = 0

    def predict(
        self,
        model: torch.nn.Module,
        policy_input: dict[str, Any],
        proposals: Sequence[dict[str, Any]],
        *,
        tensors: dict[str, torch.Tensor],
        binding: Any,
        expected_model_state_sha256: str,
    ) -> PredictionBatch:
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("WM_CACHE_MODEL_MUST_BE_FROZEN_AND_EVAL")
        if not isinstance(policy_input, dict) or not isinstance(policy_input.get("observation"), dict):
            raise ValueError("WM_CACHE_PUBLIC_PREFIX")
        if not isinstance(policy_input.get("candidates"), list):
            raise ValueError("WM_CACHE_PUBLIC_CANDIDATES")

        model_state_sha256 = _state_sha256(model)
        if model_state_sha256 != expected_model_state_sha256:
            raise ValueError("WM_CACHE_MODEL_IDENTITY_CHANGED")
        proposal_rows = _proposal_rows(proposals)
        binding_ids = tuple(binding.candidate_ids[0])
        if set(binding_ids) != {row["candidate_id"] for row in proposal_rows}:
            raise ValueError("WM_CACHE_PROPOSAL_LAYOUT_MISMATCH")

        model_layout = {
            "format_version": getattr(model, "format_version", None),
            "config": getattr(model, "config", None),
            "normalization": model.normalization_parameters()
            if hasattr(model, "normalization_parameters") else None,
            "global_head_names": list(getattr(model, "config", {}).get("global_head_names", ())),
            "global_head_units": list(getattr(model, "config", {}).get("global_head_units", ())),
        }
        public_prefix_sha256 = _sha256(_canonical({
            "observation": policy_input["observation"],
            "candidates": policy_input["candidates"],
        }))
        key_payload = {
            "schema": SCHEMA,
            "model_state_sha256": model_state_sha256,
            "model_layout": model_layout,
            "public_prefix_sha256": public_prefix_sha256,
            "proposal_set": proposal_rows,
            "layout": _layout_identity(tensors, binding),
        }
        cache_key = _sha256(_canonical(key_payload))
        cached = self._entries.get(cache_key)
        if cached is not None:
            self._entries.move_to_end(cache_key)
            self._hits += 1
            return PredictionBatch(
                by_candidate_id={
                    candidate_id: prediction.to(device=tensors["context"].device)
                    for candidate_id, prediction in cached.items()
                },
                cache_key_sha256=cache_key,
                cache_hit=True,
                forward_batches=0,
                candidate_evaluations=0,
            )

        with torch.no_grad():
            output = model(**tensors, binding=binding)
        predictions = output.get("global")
        if (
            not torch.is_tensor(predictions)
            or predictions.ndim != 3
            or predictions.shape[0] != 1
            or predictions.shape[1] != len(binding_ids)
            or not bool(torch.isfinite(predictions).all())
        ):
            raise ValueError("WM_CACHE_FULL_PROPOSAL_OUTPUT")
        candidate_mask = tensors["candidate_mask"][0].detach().cpu().tolist()
        if len(candidate_mask) != len(binding_ids) or not all(candidate_mask):
            raise ValueError("WM_CACHE_FULL_PROPOSAL_MASK")
        detached = {
            candidate_id: predictions[0, index].detach().cpu().clone()
            for index, candidate_id in enumerate(binding_ids)
        }
        if set(detached) != {row["candidate_id"] for row in proposal_rows}:
            raise ValueError("WM_CACHE_PREDICTION_IDENTITY")
        self._entries[cache_key] = detached
        self._entries.move_to_end(cache_key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        evaluations = int(tensors["candidate_mask"].sum().item())
        self._misses += 1
        self._forward_batches += 1
        self._candidate_evaluations += evaluations
        return PredictionBatch(
            by_candidate_id={
                candidate_id: prediction.to(device=tensors["context"].device)
                for candidate_id, prediction in detached.items()
            },
            cache_key_sha256=cache_key,
            cache_hit=False,
            forward_batches=1,
            candidate_evaluations=evaluations,
        )

    def stats(self) -> dict[str, int | str]:
        return {
            "schema": SCHEMA,
            "entry_count": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "forward_batches": self._forward_batches,
            "candidate_evaluations": self._candidate_evaluations,
        }
