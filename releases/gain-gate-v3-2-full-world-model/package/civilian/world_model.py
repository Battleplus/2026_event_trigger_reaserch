"""Public-input consequence model and critic-only G/G-WM integration.

The world model is frozen during PPO.  Complete environment action batches
are scored through the civilian policy's sequential masks; candidate-level
softmax is deliberately not used as a substitute for a joint action
probability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .stable_identity import key_digest
    from .wm_prediction_cache import FrozenWmPredictionCache
    from .scenario_schema import MAP_CONTRACT, MAP_ROWS, MAP_COLS, GRID_CELLS, validate_map_contract
except ImportError:
    from stable_identity import key_digest
    from wm_prediction_cache import FrozenWmPredictionCache
    from scenario_schema import MAP_CONTRACT, MAP_ROWS, MAP_COLS, GRID_CELLS, validate_map_contract


SCHEMA = "n2-civilian-consequence/5"
GLOBAL_HEAD_NAMES = (
    "other_order_lateness",
    "unmet_demand",
    "energy",
    "reassignment",
    "global_utility_delta",
)
GLOBAL_HEAD_UNITS = ("count", "count", "energy_units", "count", "reward")
GLOBAL_HEAD_COUNT = len(GLOBAL_HEAD_NAMES)
TASK_HEAD_NAMES = ("delivery_on_time", "inspection_on_time", "report_on_time")
TASK_HEAD_UNITS = ("indicator", "indicator", "indicator")
TASK_HEAD_COUNT = len(TASK_HEAD_NAMES)
NORMALIZATION_SOURCE = "training_window_zscore_v1"
NORMALIZED_FEATURES = ("context", "candidate_features", "task_features")


@dataclass(frozen=True)
class InputBinding:
    candidate_ids: tuple[tuple[str, ...], ...]
    task_ids: tuple[tuple[str, ...], ...]


def _validate_binding(binding: InputBinding, candidate_mask: torch.Tensor, task_mask: torch.Tensor) -> None:
    if candidate_mask.dtype is not torch.bool or task_mask.dtype is not torch.bool:
        raise ValueError("PUBLIC_MASK_DTYPE")
    if candidate_mask.ndim != 2 or task_mask.ndim != 2 or candidate_mask.shape[0] != task_mask.shape[0]:
        raise ValueError("PUBLIC_MASK_SHAPE")
    if len(binding.candidate_ids) != candidate_mask.shape[0] or len(binding.task_ids) != task_mask.shape[0]:
        raise ValueError("PUBLIC_BINDING_BATCH")
    for row, candidate_ids in enumerate(binding.candidate_ids):
        if len(candidate_ids) != candidate_mask.shape[1]:
            raise ValueError("PUBLIC_CANDIDATE_BINDING")
        active = [candidate_ids[index] for index, flag in enumerate(candidate_mask[row].tolist()) if flag]
        if not active or len(active) != len(set(active)) or any(not item for item in active):
            raise ValueError("PUBLIC_CANDIDATE_IDENTITY")
        task_ids = binding.task_ids[row]
        if len(task_ids) != task_mask.shape[1] or len(task_ids) != len(set(task_ids)):
            raise ValueError("PUBLIC_TASK_BINDING")


def public_policy_tensors(policy_input: dict[str, Any], *, device: torch.device | str = "cpu") -> tuple[dict[str, torch.Tensor], InputBinding]:
    """Encode only the public observation and candidate catalog."""
    observation = policy_input.get("observation") if isinstance(policy_input, dict) else None
    candidates = policy_input.get("candidates") if isinstance(policy_input, dict) else None
    if not isinstance(observation, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("WM_PUBLIC_POLICY_INPUT")
    tasks = sorted(observation.get("tasks", []), key=lambda item: str(item.get("id")))
    task_by_id = {str(item["id"]): item for item in tasks}
    catalog_ids = tuple(str(item["id"]) for item in candidates)
    task_ids = tuple(str(item["id"]) for item in tasks)
    if catalog_ids.count("NOOP") != 1 or len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("WM_PUBLIC_CANDIDATE_IDS")
    if any(item.get("task_id") is not None and str(item["task_id"]) not in task_by_id for item in candidates):
        raise ValueError("WM_PRIVATE_TASK_REFERENCE")
    map_data = observation.get("map", {})
    validate_map_contract(map_data)
    horizon = max(float(observation.get("horizon", 1.0)), 1.0)
    now = float(observation.get("time", 0.0))
    context = torch.tensor([[
        now / horizon,
        horizon / 100.0,
        float(map_data.get("rows", 0.0)) / 100.0,
        float(map_data.get("cols", 0.0)) / 100.0,
        len(tasks) / max(1.0, len(tasks) + len(observation.get("platforms", []))),
    ]], dtype=torch.float32, device=device)
    candidate_by_id = {str(item["id"]): item for item in candidates}
    proposals = policy_input.get("proposal_batches")
    if proposals is None:
        proposals = [{"actions": [candidate_id]} for candidate_id in catalog_ids]
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("WM_PUBLIC_PROPOSAL_BATCHES")
    proposal_ids: list[str] = []
    candidate_rows: list[list[float]] = []
    max_assignments = max(1, len(observation.get("platforms", [])))
    for proposal in proposals:
        if not isinstance(proposal, dict) or not isinstance(proposal.get("actions"), list):
            raise ValueError("WM_PUBLIC_PROPOSAL_RECORD")
        actions = list(proposal["actions"])
        if not actions and proposal.get("masks"):
            masks = proposal["masks"]
            if isinstance(masks, list) and masks[-1].get("selected_id") == "NOOP":
                actions = ["NOOP"]
        if not actions or any(type(item) is not str for item in actions) or len(actions) != len(set(actions)):
            raise ValueError("WM_PUBLIC_PROPOSAL_ACTIONS")
        if "NOOP" in actions and actions[-1] != "NOOP":
            raise ValueError("WM_PUBLIC_PROPOSAL_NOOP_ORDER")
        if any(item not in candidate_by_id for item in actions):
            raise ValueError("WM_PUBLIC_PROPOSAL_CATALOG")
        proposal_id = key_digest(actions)
        if proposal_id in proposal_ids:
            raise ValueError("WM_DUPLICATE_PROPOSAL_ID")
        proposal_ids.append(proposal_id)
        selected = [candidate_by_id[item] for item in actions if item != "NOOP"]
        candidate_rows.append([
            float(not selected),
            float(len(selected)) / max_assignments,
            max((max(0.0, float(item.get("estimated_completion", now)) - now) for item in selected), default=0.0) / horizon,
            sum(float(item.get("required_energy", 0.0)) for item in selected) / 100.0,
            len({str(item.get("task_id")) for item in selected if item.get("task_id") is not None}) / max_assignments,
        ])
    task_rows = []
    for task in tasks:
        task_rows.append([
            float(task.get("kind") == "inspection"),
            float(task.get("kind") == "delivery"),
            max(0.0, float(task.get("deadline", now)) - now) / horizon,
            float(task.get("priority", 0.0)) / 5.0,
            float(task.get("quantity") or 0.0) / 20.0,
        ])
    if not task_rows:
        task_tensor = torch.empty((1, 0, 5), dtype=torch.float32, device=device)
    else:
        task_tensor = torch.tensor([task_rows], dtype=torch.float32, device=device)
    tensors = {
        "context": context,
        "candidate_features": torch.tensor([candidate_rows], dtype=torch.float32, device=device),
        "task_features": task_tensor,
        "candidate_mask": torch.ones((1, len(proposal_ids)), dtype=torch.bool, device=device),
        "task_mask": torch.ones((1, len(task_ids)), dtype=torch.bool, device=device),
    }
    return tensors, InputBinding((tuple(proposal_ids),), (task_ids,))


class DynamicConsequenceModel(nn.Module):
    """Dynamic candidate encoder with the five required global consequence heads."""

    format_version = SCHEMA

    def __init__(self, context_dim: int = 5, candidate_dim: int = 5, task_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        if min(context_dim, candidate_dim, task_dim, hidden_dim) <= 0:
            raise ValueError("MODEL_DIMENSION")
        self.config = {
            "context_dim": int(context_dim),
            "candidate_dim": int(candidate_dim),
            "task_dim": int(task_dim),
            "hidden_dim": int(hidden_dim),
            "map_contract": MAP_CONTRACT,
            "map_rows": MAP_ROWS,
            "map_cols": MAP_COLS,
            "grid_cells": GRID_CELLS,
            "global_head_names": GLOBAL_HEAD_NAMES,
            "global_head_units": GLOBAL_HEAD_UNITS,
            "task_head_names": TASK_HEAD_NAMES,
            "task_head_units": TASK_HEAD_UNITS,
        }
        for name, width in (("context", context_dim), ("candidate_features", candidate_dim), ("task_features", task_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(width, dtype=torch.float32))
            self.register_buffer(f"{name}_scale", torch.ones(width, dtype=torch.float32))
        self.candidate_encoder = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.global_heads = nn.Linear(hidden_dim, GLOBAL_HEAD_COUNT)
        self.task_encoder = nn.Sequential(
            nn.Linear(hidden_dim + task_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.task_heads = nn.Linear(hidden_dim, TASK_HEAD_COUNT)

    def set_normalization(self, statistics: dict[str, Any]) -> None:
        if set(statistics) != set(NORMALIZED_FEATURES):
            raise ValueError("WM_NORMALIZATION_FEATURES")
        for name in NORMALIZED_FEATURES:
            entry = statistics[name]
            width = self.config["context_dim" if name == "context" else "candidate_dim" if name == "candidate_features" else "task_dim"]
            mean, scale = entry.get("mean"), entry.get("scale")
            observations = entry.get("observations")
            if (
                not isinstance(mean, list) or not isinstance(scale, list)
                or len(mean) != width or len(scale) != width
                or type(observations) is not int or observations < 0
                or any(type(value) not in (int, float) or not torch.isfinite(torch.tensor(float(value))) for value in mean + scale)
                or any(float(value) <= 0.0 for value in scale)
            ):
                raise ValueError("WM_NORMALIZATION_STATISTICS")
            getattr(self, f"{name}_mean").copy_(torch.tensor(mean, dtype=torch.float32, device=getattr(self, f"{name}_mean").device))
            getattr(self, f"{name}_scale").copy_(torch.tensor(scale, dtype=torch.float32, device=getattr(self, f"{name}_scale").device))

    def normalization_parameters(self) -> dict[str, dict[str, list[float]]]:
        return {
            name: {
                "mean": getattr(self, f"{name}_mean").detach().cpu().tolist(),
                "scale": getattr(self, f"{name}_scale").detach().cpu().tolist(),
            }
            for name in NORMALIZED_FEATURES
        }

    def forward(
        self,
        context: torch.Tensor,
        candidate_features: torch.Tensor,
        task_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        task_mask: torch.Tensor,
        binding: InputBinding,
    ) -> dict[str, torch.Tensor | tuple[tuple[str, ...], ...]]:
        _validate_binding(binding, candidate_mask, task_mask)
        if context.ndim != 2 or candidate_features.ndim != 3 or task_features.ndim != 3:
            raise ValueError("PUBLIC_FEATURE_RANK")
        batch, candidate_slots, candidate_dim = candidate_features.shape
        if context.shape != (batch, self.config["context_dim"]):
            raise ValueError("PUBLIC_CONTEXT_SHAPE")
        if candidate_dim != self.config["candidate_dim"] or task_features.shape[0] != batch or task_features.shape[2] != self.config["task_dim"]:
            raise ValueError("PUBLIC_FEATURE_SHAPE")
        if not all(bool(torch.isfinite(value).all()) for value in (context, candidate_features, task_features)):
            raise ValueError("PUBLIC_FEATURE_NONFINITE")
        context = (context - self.context_mean) / self.context_scale
        candidate_features = (candidate_features - self.candidate_features_mean) / self.candidate_features_scale
        task_features = (task_features - self.task_features_mean) / self.task_features_scale
        expanded = context[:, None, :].expand(-1, candidate_slots, -1)
        hidden = self.candidate_encoder(torch.cat((expanded, candidate_features), dim=-1))
        active = candidate_mask.unsqueeze(-1)
        global_predictions = self.global_heads(hidden).masked_fill(~active, 0.0)
        task_slots = task_features.shape[1]
        task_hidden = self.task_encoder(torch.cat((
            hidden[:, :, None, :].expand(-1, -1, task_slots, -1),
            task_features[:, None, :, :].expand(-1, candidate_slots, -1, -1),
        ), dim=-1)) if task_slots else hidden.new_empty((batch, candidate_slots, 0, hidden.shape[-1]))
        per_task = self.task_heads(task_hidden).masked_fill(
            ~(candidate_mask[:, :, None] & task_mask[:, None, :]).unsqueeze(-1), 0.0
        )
        return {
            "global": global_predictions,
            "global_names": GLOBAL_HEAD_NAMES,
            "global_units": GLOBAL_HEAD_UNITS,
            "task_names": TASK_HEAD_NAMES,
            "task_units": TASK_HEAD_UNITS,
            "per_task": per_task,
            "candidate_mask": candidate_mask,
            "task_mask": task_mask,
            "candidate_ids": binding.candidate_ids,
            "task_ids": binding.task_ids,
        }

    @torch.no_grad()
    def predict_action_batch(self, policy_input: dict[str, Any], action_batch: Sequence[str]) -> torch.Tensor:
        """Pool frozen WM predictions for one complete, legal actor action batch."""
        tensors, binding = public_policy_tensors(policy_input)
        actions = list(action_batch)
        if not actions or any(type(item) is not str for item in actions) or len(actions) != len(set(actions)):
            raise ValueError("WM_ACTION_BATCH_IDENTITY")
        if "NOOP" in actions and actions[-1] != "NOOP":
            raise ValueError("WM_ACTION_BATCH_NOOP_ORDER")
        action_digest = key_digest(actions)
        try:
            slot = binding.candidate_ids[0].index(action_digest)
        except ValueError as exc:
            raise ValueError("WM_ACTION_BATCH_CATALOG") from exc
        output = self(**tensors, binding=binding)
        return output["global"][:, slot, :]


def _validate_global_target(target: dict[str, Any]) -> tuple[list[float], list[bool], list[float], list[float], list[str | None]]:
    if tuple(target.get("names", ())) != GLOBAL_HEAD_NAMES or tuple(target.get("units", ())) != GLOBAL_HEAD_UNITS:
        raise ValueError("TARGET_GLOBAL_SCHEMA")
    values = target.get("values")
    valid = target.get("valid")
    candidate_values = target.get("candidate_values")
    reference_values = target.get("reference_values")
    unknown_reasons = target.get("unknown_reasons")
    if any(not isinstance(items, list) or len(items) != GLOBAL_HEAD_COUNT for items in (values, valid, candidate_values, reference_values, unknown_reasons)):
        raise ValueError("TARGET_GLOBAL_SHAPE")
    if (
        any(type(item) not in (int, float) for items in (values, candidate_values, reference_values) for item in items)
        or any(type(item) is not bool for item in valid)
        or any(item is not None and (not isinstance(item, str) or not item) for item in unknown_reasons)
        or any(flag != (reason is None) for flag, reason in zip(valid, unknown_reasons))
    ):
        raise ValueError("TARGET_GLOBAL_TYPES")
    return (
        [float(item) for item in values], list(valid),
        [float(item) for item in candidate_values], [float(item) for item in reference_values],
        list(unknown_reasons),
    )


def _validate_task_target(target: dict[str, Any], task_ids: tuple[str, ...]) -> tuple[list[list[float]], list[list[bool]], list[list[float]], list[list[float]], list[list[str | None]]]:
    if tuple(target.get("task_ids", ())) != task_ids:
        raise ValueError("TARGET_TASK_IDENTITY")
    if tuple(target.get("names", ())) != TASK_HEAD_NAMES or tuple(target.get("units", ())) != TASK_HEAD_UNITS:
        raise ValueError("TARGET_TASK_SCHEMA")
    values = target.get("values")
    valid = target.get("valid")
    candidate_values = target.get("candidate_values")
    reference_values = target.get("reference_values")
    unknown_reasons = target.get("unknown_reasons")
    if any(not isinstance(items, list) or len(items) != len(task_ids) for items in (values, valid, candidate_values, reference_values, unknown_reasons)):
        raise ValueError("TARGET_TASK_SHAPE")
    for value_row, valid_row, candidate_row, reference_row, reason_row in zip(values, valid, candidate_values, reference_values, unknown_reasons):
        if (
            any(not isinstance(row, list) for row in (value_row, valid_row, candidate_row, reference_row, reason_row))
            or any(len(row) != TASK_HEAD_COUNT for row in (value_row, valid_row, candidate_row, reference_row, reason_row))
            or any(type(item) not in (int, float) for row in (value_row, candidate_row, reference_row) for item in row)
            or any(type(item) is not bool for item in valid_row)
            or any(item is not None and (not isinstance(item, str) or not item) for item in reason_row)
            or any(flag != (reason is None) for flag, reason in zip(valid_row, reason_row))
        ):
            raise ValueError("TARGET_TASK_TYPES")
    return (
        [[float(item) for item in row] for row in values],
        [list(row) for row in valid],
        [[float(item) for item in row] for row in candidate_values],
        [[float(item) for item in row] for row in reference_values],
        [list(row) for row in unknown_reasons],
    )


def _window_identity(record: dict[str, Any]) -> tuple[str, int, str]:
    parent_id, repeat, window_id = record.get("parent_id"), record.get("repeat"), record.get("window_id")
    if not isinstance(parent_id, str) or not parent_id or type(repeat) is not int or repeat < 0 or not isinstance(window_id, str) or not window_id:
        raise ValueError("TARGET_WINDOW_IDENTITY")
    return parent_id, repeat, window_id


def _validate_action_record(record: dict[str, Any]) -> None:
    candidate_actions = record.get("candidate_action_batch")
    if record.get("candidate_is_noop") is not (candidate_actions == ["NOOP"]):
        raise ValueError("TARGET_NOOP_ACTION_BINDING")
    for key in ("candidate_action_batch", "reference_action_batch"):
        actions = record.get(key)
        if not isinstance(actions, list) or not actions or any(type(item) is not str for item in actions):
            raise ValueError("TARGET_ACTION_BATCH")
        if len(actions) != len(set(actions)):
            raise ValueError("TARGET_ACTION_BATCH_DUPLICATE")
        if "NOOP" in actions and actions[-1] != "NOOP":
            raise ValueError("TARGET_ACTION_BATCH_NOOP_ORDER")
    for key in ("source_state_sha256", "scenario_sha256", "source_prefix_sha256", "event_tape_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"TARGET_{key.upper()}")


def branch_records_to_targets(
    candidate_ids: tuple[str, ...],
    task_ids: tuple[str, ...],
    branch_records: list[dict[str, Any]],
    *,
    window_id: tuple[str, int, str],
    policy_input_digest: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[tuple[str, int, str], ...]]:
    """Join one window only; mixed snapshot/prefix/tape records are rejected."""
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("TARGET_CANDIDATE_COVERAGE")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("TARGET_TASK_COVERAGE")
    if len({str(record.get("candidate_id")) for record in branch_records}) != len(branch_records):
        raise ValueError("TARGET_DUPLICATE_CANDIDATE")
    if {str(record.get("candidate_id")) for record in branch_records} != set(candidate_ids):
        raise ValueError("TARGET_CANDIDATE_COVERAGE")
    if sum(record.get("candidate_is_noop") is True for record in branch_records) != 1:
        raise ValueError("TARGET_NOOP_PROPOSAL_COVERAGE")
    by_id = {str(record["candidate_id"]): record for record in branch_records}
    expected_source: tuple[str, str, str, str] | None = None
    for candidate_id in candidate_ids:
        record = by_id[candidate_id]
        if _window_identity(record) != window_id:
            raise ValueError("TARGET_WINDOW_BINDING")
        if type(record.get("window_start")) is not int or record["window_start"] < 0:
            raise ValueError("TARGET_WINDOW_START")
        _validate_action_record(record)
        if key_digest(record["candidate_action_batch"]) != candidate_id:
            raise ValueError("TARGET_CANDIDATE_ACTION_BINDING")
        identity = (
            str(record["source_state_sha256"]), str(record["scenario_sha256"]),
            str(record["source_prefix_sha256"]), str(record["event_tape_id"]),
        )
        if expected_source is None:
            expected_source = identity
        elif identity != expected_source:
            raise ValueError("TARGET_SNAPSHOT_OR_TAPE_MISMATCH")
        if policy_input_digest is not None and record.get("policy_input_sha256") != policy_input_digest:
            raise ValueError("TARGET_POLICY_INPUT_MISMATCH")
    window_starts = {record["window_start"] for record in by_id.values()}
    if len(window_starts) != 1:
        raise ValueError("TARGET_WINDOW_START_MISMATCH")
    target_global = torch.zeros((1, len(candidate_ids), GLOBAL_HEAD_COUNT), dtype=torch.float32, device=device)
    valid_global = torch.zeros_like(target_global, dtype=torch.bool)
    target_task = torch.zeros((1, len(candidate_ids), len(task_ids), TASK_HEAD_COUNT), dtype=torch.float32, device=device)
    valid_task = torch.zeros_like(target_task, dtype=torch.bool)
    action_digests: list[str] = []
    provenance = {
        "global_candidate_values": [],
        "global_reference_values": [],
        "global_unknown_reasons": [],
        "task_deltas": [],
        "task_candidate_values": [],
        "task_reference_values": [],
        "task_unknown_reasons": [],
    }
    for index, candidate_id in enumerate(candidate_ids):
        record = by_id[candidate_id]
        labels = record.get("paired_targets")
        if not isinstance(labels, dict) or not isinstance(labels.get("global"), dict) or not isinstance(labels.get("task"), dict):
            raise ValueError("TARGET_LABELS_MISSING")
        values, valid, global_candidate, global_reference, global_reasons = _validate_global_target(labels["global"])
        target_global[0, index] = torch.tensor(values, dtype=torch.float32, device=device)
        valid_global[0, index] = torch.tensor(valid, dtype=torch.bool, device=device)
        task_values, task_valid, task_candidate, task_reference, task_reasons = _validate_task_target(labels["task"], task_ids)
        if task_ids:
            if any(item not in (0, 1, 0.0, 1.0) for row in task_candidate for item in row):
                raise ValueError("TARGET_TASK_CANDIDATE_VALUE_RANGE")
            target_task[0, index] = torch.tensor(task_candidate, dtype=torch.float32, device=device)
            valid_task[0, index] = torch.tensor(task_valid, dtype=torch.bool, device=device)
        provenance["global_candidate_values"].append(global_candidate)
        provenance["global_reference_values"].append(global_reference)
        provenance["global_unknown_reasons"].append(global_reasons)
        provenance["task_deltas"].append(task_values)
        provenance["task_candidate_values"].append(task_candidate)
        provenance["task_reference_values"].append(task_reference)
        provenance["task_unknown_reasons"].append(task_reasons)
        action_digests.append(candidate_id)
    target = {
        "global": target_global,
        "global_names": GLOBAL_HEAD_NAMES,
        "global_units": GLOBAL_HEAD_UNITS,
        "task": target_task,
        "task_names": TASK_HEAD_NAMES,
        "task_units": TASK_HEAD_UNITS,
        "candidate_action_digest": tuple(action_digests),
        "label_provenance": provenance,
    }
    valid = {"global": valid_global, "task": valid_task}
    return target, valid, (window_id,)


def public_branch_batch(policy_input: dict[str, Any], branch_records: list[dict[str, Any]], *, window_id: tuple[str, int, str], device: torch.device | str = "cpu") -> dict[str, Any]:
    inputs, binding = public_policy_tensors(policy_input, device=device)
    target, valid, window_ids = branch_records_to_targets(
        binding.candidate_ids[0], binding.task_ids[0], branch_records,
        window_id=window_id, policy_input_digest=key_digest({"observation": policy_input["observation"], "candidates": policy_input["candidates"]}), device=device,
    )
    return {"inputs": {**inputs, "binding": binding}, "target": target, "valid": valid, "window_ids": window_ids}


def _model_state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("WM_LOSS_SHAPE")
    count = int(valid.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    return F.smooth_l1_loss(prediction[valid], target[valid], beta=1.0), count


def _masked_task_bce(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype is not torch.bool:
        raise ValueError("WM_TASK_LOSS_SHAPE")
    count = int(valid.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    labels = target[valid]
    if not bool(torch.isfinite(labels).all()) or bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError("WM_TASK_BCE_TARGET_RANGE")
    return F.binary_cross_entropy_with_logits(prediction[valid], labels), count


def _pair_weighted_ranking_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, window_ids: Sequence[tuple[str, int, str]]) -> tuple[torch.Tensor, int]:
    if prediction.shape != target.shape or valid.shape != target.shape or prediction.ndim != 2 or valid.dtype is not torch.bool:
        raise ValueError("WM_RANKING_SHAPE")
    if len(window_ids) != prediction.shape[0]:
        raise ValueError("WM_RANKING_WINDOW_BINDING")
    pair_losses: list[torch.Tensor] = []
    for row, window_id in enumerate(window_ids):
        if not isinstance(window_id, tuple) or len(window_id) != 3:
            raise ValueError("WM_RANKING_WINDOW_ID")
        for left in range(prediction.shape[1]):
            for right in range(left + 1, prediction.shape[1]):
                if not bool(valid[row, left]) or not bool(valid[row, right]):
                    continue
                if bool(target[row, left] == target[row, right]):
                    continue
                direction = torch.sign(target[row, left] - target[row, right])
                pair_losses.append(F.softplus(-direction * (prediction[row, left] - prediction[row, right])))
    loss = torch.stack(pair_losses).mean() if pair_losses else prediction.sum() * 0.0
    return loss, len(pair_losses)


def _collate_window_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("WM_EMPTY_WINDOW_BATCH")
    candidate_width = max(item["inputs"]["candidate_features"].shape[1] for item in samples)
    task_width = max(item["inputs"]["task_features"].shape[1] for item in samples)
    device = samples[0]["inputs"]["context"].device
    contexts = []
    candidates = []
    tasks = []
    candidate_masks = []
    task_masks = []
    target_global = []
    valid_global = []
    target_task = []
    valid_task = []
    candidate_ids = []
    task_ids = []
    for row, sample in enumerate(samples):
        inputs = sample["inputs"]
        candidate_count = inputs["candidate_features"].shape[1]
        task_count = inputs["task_features"].shape[1]
        contexts.append(inputs["context"])
        candidates.append(F.pad(inputs["candidate_features"], (0, 0, 0, candidate_width - candidate_count)))
        tasks.append(F.pad(inputs["task_features"], (0, 0, 0, task_width - task_count)))
        candidate_masks.append(F.pad(inputs["candidate_mask"], (0, candidate_width - candidate_count), value=False))
        task_masks.append(F.pad(inputs["task_mask"], (0, task_width - task_count), value=False))
        target_global.append(F.pad(sample["target"]["global"], (0, 0, 0, candidate_width - candidate_count)))
        valid_global.append(F.pad(sample["valid"]["global"], (0, 0, 0, candidate_width - candidate_count), value=False))
        target_task.append(F.pad(sample["target"]["task"], (0, 0, 0, task_width - task_count, 0, candidate_width - candidate_count)))
        valid_task.append(F.pad(sample["valid"]["task"], (0, 0, 0, task_width - task_count, 0, candidate_width - candidate_count), value=False))
        binding: InputBinding = inputs["binding"]
        candidate_row = list(binding.candidate_ids[0]) + [f"__PAD_CANDIDATE_{row}_{i}" for i in range(candidate_width - candidate_count)]
        task_row = list(binding.task_ids[0]) + [f"__PAD_TASK_{row}_{i}" for i in range(task_width - task_count)]
        candidate_ids.append(tuple(candidate_row))
        task_ids.append(tuple(task_row))
    batch_inputs = {
        "context": torch.cat(contexts, dim=0),
        "candidate_features": torch.cat(candidates, dim=0),
        "task_features": torch.cat(tasks, dim=0),
        "candidate_mask": torch.cat(candidate_masks, dim=0),
        "task_mask": torch.cat(task_masks, dim=0),
        "binding": InputBinding(tuple(candidate_ids), tuple(task_ids)),
    }
    return {
        "inputs": batch_inputs,
        "target": {
            "global": torch.cat(target_global, dim=0),
            "task": torch.cat(target_task, dim=0),
            "label_provenance": [sample["target"]["label_provenance"] for sample in samples],
        },
        "valid": {"global": torch.cat(valid_global, dim=0), "task": torch.cat(valid_task, dim=0)},
    }


def _fit_training_normalization(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    feature_masks = {"context": None, "candidate_features": "candidate_mask", "task_features": "task_mask"}
    for feature_name, mask_name in feature_masks.items():
        rows = []
        width: int | None = None
        for sample in samples:
            inputs = sample["inputs"]
            features = inputs[feature_name].detach().to(device="cpu", dtype=torch.float32)
            if (feature_name == "context" and features.ndim != 2) or (feature_name != "context" and features.ndim != 3):
                raise ValueError("WM_NORMALIZATION_FEATURE_SHAPE")
            width = int(features.shape[-1])
            values = features.reshape(-1, width)
            if mask_name is not None:
                mask = inputs[mask_name].detach().to(device="cpu")
                if mask.dtype is not torch.bool or tuple(mask.shape) != tuple(features.shape[:-1]):
                    raise ValueError("WM_NORMALIZATION_MASK_SHAPE")
                values = values[mask.reshape(-1)]
            if values.numel():
                rows.append(values)
        if width is None:
            raise ValueError("WM_NORMALIZATION_EMPTY_TRAINING_SET")
        if rows:
            values = torch.cat(rows, dim=0)
            mean = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
            observations = int(values.shape[0])
        else:
            mean = torch.zeros(width, dtype=torch.float32)
            scale = torch.ones(width, dtype=torch.float32)
            observations = 0
        statistics[feature_name] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "observations": observations,
        }
    return statistics


def train_consequence_model(
    samples: Sequence[dict[str, Any]],
    *,
    checkpoint_path: str | Path,
    data_identity: str,
    seed: int = 8201,
    epochs: int = 1,
    batch_size: int = 4,
    max_updates: int = 8,
) -> dict[str, Any]:
    if not data_identity or not samples or len(samples) > 8 or seed != 8201 or epochs != 1 or batch_size != 4 or max_updates != 8:
        raise ValueError("WM_TRAINING_BUDGET_OR_IDENTITY")
    required_inputs = {"context", "candidate_features", "task_features", "candidate_mask", "task_mask", "binding"}
    for sample in samples:
        if not isinstance(sample, dict) or set(sample.get("inputs", {})) != required_inputs:
            raise ValueError("WM_TRAINING_INPUT_SCOPE")
        if not {"global", "task"}.issubset(sample.get("target", {})) or not {"global", "task"}.issubset(sample.get("valid", {})):
            raise ValueError("WM_TRAINING_TARGET_HEADS")
    inputs = samples[0]["inputs"]
    torch.manual_seed(int(seed))
    model = DynamicConsequenceModel(
        context_dim=int(inputs["context"].shape[-1]),
        candidate_dim=int(inputs["candidate_features"].shape[-1]),
        task_dim=int(inputs["task_features"].shape[-1]),
    )
    normalization_statistics = _fit_training_normalization(samples)
    model.set_normalization(normalization_statistics)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    training_rows: list[dict[str, Any]] = []
    updates = 0
    model.train()
    for epoch in range(int(epochs)):
        for batch_start in range(0, len(samples), batch_size):
            if updates >= max_updates:
                break
            batch_samples = samples[batch_start:batch_start + batch_size]
            batch = _collate_window_samples(batch_samples)
            sample_index = batch_start // batch_size
            output = model(**batch["inputs"])
            global_head_losses = []
            global_head_counts = []
            for head in range(GLOBAL_HEAD_COUNT):
                head_loss, head_count = _masked_loss(
                    output["global"][..., head], batch["target"]["global"][..., head],
                    batch["valid"]["global"][..., head],
                )
                global_head_losses.append(head_loss)
                global_head_counts.append(head_count)
            task_head_losses = []
            task_head_counts = []
            for head in range(TASK_HEAD_COUNT):
                head_loss, head_count = _masked_task_bce(
                    output["per_task"][..., head], batch["target"]["task"][..., head],
                    batch["valid"]["task"][..., head],
                )
                task_head_losses.append(head_loss)
                task_head_counts.append(head_count)
            window_ids = tuple(sample["window_ids"][0] for sample in batch_samples)
            ranking_loss, ranking_pairs = _pair_weighted_ranking_loss(
                output["global"][..., GLOBAL_HEAD_NAMES.index("global_utility_delta")],
                batch["target"]["global"][..., GLOBAL_HEAD_NAMES.index("global_utility_delta")],
                batch["valid"]["global"][..., GLOBAL_HEAD_NAMES.index("global_utility_delta")],
                window_ids,
            )
            loss = sum(global_head_losses, output["global"].sum() * 0.0) + sum(task_head_losses, output["per_task"].sum() * 0.0) + ranking_loss
            global_count = sum(global_head_counts)
            task_count = sum(task_head_counts)
            if not bool(torch.isfinite(loss)) or global_count + task_count == 0:
                raise ValueError("WM_TRAINING_INVALID_SUPERVISION")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise ValueError("WM_TRAINING_INVALID_GRADIENT")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            updates += 1
            training_rows.append({
                "epoch": epoch + 1,
                "sample_index": sample_index,
                "update_index": updates,
                "global_head_losses": {name: float(value.detach().item()) for name, value in zip(GLOBAL_HEAD_NAMES, global_head_losses)},
                "task_head_losses": {name: float(value.detach().item()) for name, value in zip(TASK_HEAD_NAMES, task_head_losses)},
                "ranking_loss": float(ranking_loss.detach().item()),
                "ranking_reduction": "pair_weighted",
                "valid_ranking_pairs": ranking_pairs,
                "global_valid_count": global_count,
                "task_valid_count": task_count,
                "gradient_norm": float(grad_norm.detach().item()),
            })
        if updates >= max_updates:
            break
    if not updates:
        raise ValueError("WM_TRAINING_NO_UPDATES")
    model.eval()
    state = model.state_dict()
    payload = {
        "schema": "n2-world-model-checkpoint/1",
        "format_version": SCHEMA,
        "data_identity": data_identity,
        "config": copy.deepcopy(model.config),
        "layout_sha256": hashlib.sha256(json.dumps(model.config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "normalization": {
            "source": NORMALIZATION_SOURCE,
            "fitted_on": "training_samples_only",
            "statistics": normalization_statistics,
        },
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _model_state_digest(state),
        "seed": int(seed),
        "epochs": int(epochs),
        "updates": updates,
        "training_rows": training_rows,
    }
    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=target_path.name + ".", suffix=".tmp", dir=target_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "status": "complete",
        "checkpoint_path": target_path.as_posix(),
        "checkpoint_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "model_state_sha256": payload["model_state_sha256"],
        "updates": updates,
        "training_rows": training_rows,
    }


def load_consequence_checkpoint(checkpoint_path: str | Path, *, data_identity: str) -> tuple[DynamicConsequenceModel, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file() or not data_identity:
        raise ValueError("WM_CHECKPOINT_MISSING")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "n2-world-model-checkpoint/1" or payload.get("format_version") != SCHEMA:
        raise ValueError("WM_CHECKPOINT_SCHEMA")
    if payload.get("data_identity") != data_identity:
        raise ValueError("WM_CHECKPOINT_DATA_IDENTITY")
    config = payload.get("config")
    if (
        not isinstance(config, dict)
        or tuple(config.get("global_head_names", ())) != GLOBAL_HEAD_NAMES
        or tuple(config.get("task_head_names", ())) != TASK_HEAD_NAMES
        or config.get("map_contract") != MAP_CONTRACT
        or config.get("map_rows") != MAP_ROWS
        or config.get("map_cols") != MAP_COLS
        or config.get("grid_cells") != GRID_CELLS
    ):
        raise ValueError("WM_CHECKPOINT_LAYOUT")
    if payload.get("layout_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest():
        raise ValueError("WM_CHECKPOINT_LAYOUT_DIGEST")
    normalization = payload.get("normalization")
    if (
        not isinstance(normalization, dict)
        or set(normalization) != {"source", "fitted_on", "statistics"}
        or normalization.get("source") != NORMALIZATION_SOURCE
        or normalization.get("fitted_on") != "training_samples_only"
        or not isinstance(normalization.get("statistics"), dict)
    ):
        raise ValueError("WM_CHECKPOINT_NORMALIZATION_SCOPE")
    model = DynamicConsequenceModel(
        context_dim=int(config["context_dim"]),
        candidate_dim=int(config["candidate_dim"]),
        task_dim=int(config["task_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.set_normalization(normalization["statistics"])
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("WM_CHECKPOINT_STATE")
    model.load_state_dict(state, strict=True)
    expected_normalization = {
        name: {"mean": normalization["statistics"][name]["mean"], "scale": normalization["statistics"][name]["scale"]}
        for name in NORMALIZED_FEATURES
    }
    if model.normalization_parameters() != expected_normalization:
        raise ValueError("WM_CHECKPOINT_NORMALIZATION_STATE")
    if payload.get("model_state_sha256") != _model_state_digest(model.state_dict()):
        raise ValueError("WM_CHECKPOINT_STATE_DIGEST")
    model.eval()
    return model, payload


@torch.no_grad()
def verify_checkpoint_predictions(model: DynamicConsequenceModel, restored: DynamicConsequenceModel, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if _model_state_digest(model.state_dict()) != _model_state_digest(restored.state_dict()):
        raise ValueError("WM_RESTORE_PARAMETER_MISMATCH")
    checked = 0
    for sample in samples:
        inputs = sample["inputs"]
        binding = inputs["binding"]
        features = {key: value for key, value in inputs.items() if key != "binding"}
        first = model(**features, binding=binding)
        second = restored(**features, binding=binding)
        if not torch.equal(first["global"], second["global"]) or not torch.equal(first["per_task"], second["per_task"]):
            raise ValueError("WM_RESTORE_PREDICTION_MISMATCH")
        checked += 1
    if not checked:
        raise ValueError("WM_RESTORE_NO_PREDICTIONS")
    return {"status": "complete", "prediction_batches": checked, "model_state_sha256": _model_state_digest(restored.state_dict())}


def complete_action_log_probability(base_policy: nn.Module, policy_input: dict[str, Any], action_batch: list[str], action_masks: list[dict[str, Any]]) -> torch.Tensor:
    """Use the actor's sequential residual masks to score a complete batch."""
    if not isinstance(action_batch, list) or not action_batch or any(type(item) is not str for item in action_batch):
        raise ValueError("FULL_ACTION_BATCH_REQUIRED")
    if len(action_batch) != len(set(action_batch)) or ("NOOP" in action_batch and action_batch[-1] != "NOOP"):
        raise ValueError("FULL_ACTION_BATCH_IDENTITY")
    if not isinstance(action_masks, list) or not action_masks or any(not isinstance(item, dict) for item in action_masks):
        raise ValueError("FULL_ACTION_MASKS_REQUIRED")
    selected_ids = [item.get("selected_id") for item in action_masks]
    if selected_ids != action_batch:
        raise ValueError("FULL_ACTION_MASK_BINDING")
    dispatch_actions = [item for item in action_batch if item != "NOOP"]
    result = base_policy.evaluate_batch(policy_input, dispatch_actions, action_masks)
    log_probability = result.get("log_prob")
    if not isinstance(log_probability, torch.Tensor) or log_probability.numel() != 1:
        raise ValueError("FULL_ACTION_LOG_PROBABILITY")
    return log_probability.reshape(())


def _proposal_records(policy_input: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = policy_input.get("proposal_batches")
    if proposals is None:
        sampled_actions = list(sample.get("actions", []))
        masks = sample.get("masks", [])
        mask_actions = [record.get("selected_id") for record in masks] if isinstance(masks, list) else []
        if mask_actions and mask_actions[-1] == "NOOP" and sampled_actions == [item for item in mask_actions if item != "NOOP"]:
            sampled_actions = mask_actions
        proposals = [{"actions": sampled_actions, "masks": masks}]
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("PROPOSAL_BATCH_SET")
    result = []
    for proposal in proposals:
        if not isinstance(proposal, dict) or not isinstance(proposal.get("actions"), list) or not isinstance(proposal.get("masks"), list):
            raise ValueError("PROPOSAL_BATCH_RECORD")
        actions = proposal["actions"]
        if not actions or any(type(item) is not str for item in actions) or len(actions) != len(set(actions)):
            raise ValueError("PROPOSAL_FULL_ACTION_IDENTITY")
        if "NOOP" in actions and actions[-1] != "NOOP":
            raise ValueError("PROPOSAL_NOOP_ORDER")
        candidate_id = key_digest(actions)
        if proposal.get("candidate_id", candidate_id) != candidate_id:
            raise ValueError("PROPOSAL_CANDIDATE_BINDING")
        is_noop = actions == ["NOOP"]
        if proposal.get("is_noop", is_noop) is not is_noop:
            raise ValueError("PROPOSAL_NOOP_BINDING")
        result.append({"actions": list(actions), "masks": proposal["masks"], "candidate_id": candidate_id, "is_noop": is_noop})
    if len({item["candidate_id"] for item in result}) != len(result):
        raise ValueError("PROPOSAL_DUPLICATE_BATCH")
    if sum(item["is_noop"] for item in result) != 1:
        raise ValueError("PROPOSAL_NOOP_BATCH_REQUIRED")
    return result


class CriticOnlyGppoAdapter(nn.Module):
    """Five-channel detached WM summary projected by a zero-initialized critic head."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim != 32:
            raise ValueError("GPPO_AUX_HIDDEN_DIM_IS_FROZEN_AT_32")
        self.value_adapter = nn.Sequential(nn.Linear(GLOBAL_HEAD_COUNT, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.value_adapter[-1].weight)
        nn.init.zeros_(self.value_adapter[-1].bias)
        self.transform = "detach-WM; detach-complete-action-probabilities; five-global-channel-summary"

    def forward(self, base_value: torch.Tensor, wm_predictions: torch.Tensor | None, proposal_probabilities: torch.Tensor, *, wm_enabled: bool) -> torch.Tensor:
        if base_value.ndim != 1 or proposal_probabilities.ndim != 2 or proposal_probabilities.shape[0] != base_value.shape[0]:
            raise ValueError("CRITIC_AUX_BATCH_SHAPE")
        if wm_enabled:
            if wm_predictions is None or wm_predictions.ndim != 3 or wm_predictions.shape[:2] != proposal_probabilities.shape or wm_predictions.shape[2] != GLOBAL_HEAD_COUNT:
                raise ValueError("CRITIC_AUX_GLOBAL_SHAPE")
            q = proposal_probabilities.detach().softmax(dim=1)
            summary = (q.unsqueeze(-1) * wm_predictions.detach()).sum(dim=1)
            return base_value + self.value_adapter(summary).squeeze(-1)
        if wm_predictions is not None:
            raise ValueError("BASELINE_MUST_NOT_RECEIVE_WM")
        return base_value


class CriticAugmentedGraphPolicy(nn.Module):
    """Civilian actor facade with a frozen WM and critic-only five-channel path."""

    def __init__(
        self,
        base_policy: nn.Module,
        world_model: DynamicConsequenceModel,
        adapter: CriticOnlyGppoAdapter,
        *,
        wm_enabled: bool,
        prediction_cache: FrozenWmPredictionCache | None = None,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.world_model = world_model.requires_grad_(False).eval()
        self.adapter = adapter
        self.wm_enabled = bool(wm_enabled)
        self.prediction_cache = (
            prediction_cache or FrozenWmPredictionCache()
            if self.wm_enabled else None
        )
        self.variant = getattr(base_policy, "variant", "unknown")
        self.source_identity = self._state_digest(self.world_model.state_dict())

    @staticmethod
    def _state_digest(state: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii")); digest.update(str(tuple(tensor.shape)).encode("ascii")); digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_model.eval()
        return self

    def _proposal_scores(self, policy_input: dict[str, Any], sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        proposals = _proposal_records(policy_input, sample)
        log_probs = []
        for proposal in proposals:
            log_probs.append(complete_action_log_probability(self.base_policy, policy_input, proposal["actions"], proposal["masks"]))
        logs = torch.stack(log_probs).reshape(1, -1)
        cache_result = None
        wm = None
        if self.wm_enabled:
            if self.prediction_cache is None:
                raise RuntimeError("WM_PREDICTION_CACHE_MISSING")
            if self._state_digest(self.world_model.state_dict()) != self.source_identity:
                raise ValueError("WM_SOURCE_IDENTITY_CHANGED")
            tensors, binding = public_policy_tensors(policy_input)
            cache_result = self.prediction_cache.predict(
                self.world_model,
                policy_input,
                proposals,
                tensors=tensors,
                binding=binding,
                expected_model_state_sha256=self.source_identity,
            )
            wm_values = [cache_result.by_candidate_id[item["candidate_id"]] for item in proposals]
            wm = torch.stack(wm_values).reshape(1, -1, GLOBAL_HEAD_COUNT)
        proposal_mass = torch.logsumexp(logs.detach(), dim=1).exp()
        if bool((proposal_mass > 1.0 + 1e-5).any()):
            raise ValueError("PROPOSAL_SET_PROBABILITY_MASS")
        metadata = {
            "proposal_count": len(proposals),
            "proposal_probability_binding": "complete_action_batch_sequential_masks",
            "proposal_set_probability_mass": float(proposal_mass[0].item()),
            "proposal_set_mass_scope": "finite_proposal_subset",
            "critic_aggregation": "conditional_softmax_over_proposal_subset",
            "candidate_ids": [item["candidate_id"] for item in proposals],
        }
        metadata["wm_prediction_cache"] = {
            "enabled": self.wm_enabled,
            "cache_key_sha256": cache_result.cache_key_sha256 if cache_result is not None else None,
            "hit": cache_result.cache_hit if cache_result is not None else False,
            "miss": not cache_result.cache_hit if cache_result is not None else False,
            "forward_batches": cache_result.forward_batches if cache_result is not None else 0,
            "candidate_evaluations": cache_result.candidate_evaluations if cache_result is not None else 0,
        }
        return logs, wm, metadata

    def sample_batch(self, policy_input: dict[str, Any], deterministic: bool = False) -> dict[str, Any]:
        sample = self.base_policy.sample_batch(policy_input, deterministic=deterministic)
        log_probs, wm_values, metadata = self._proposal_scores(policy_input, sample)
        base_eval = self.base_policy.evaluate_batch(policy_input, sample["actions"], sample["masks"])
        sample["log_prob"] = base_eval["log_prob"]
        sample["value"] = self.adapter(base_eval["value"].reshape(1), wm_values, log_probs, wm_enabled=self.wm_enabled).reshape(())
        sample["proposal_binding"] = metadata
        return sample

    def evaluate_batch(self, policy_input: dict[str, Any], actions: list[str], masks: list[dict[str, Any]]) -> dict[str, Any]:
        base_eval = self.base_policy.evaluate_batch(policy_input, actions, masks)
        sample = {"actions": actions, "masks": masks}
        log_probs, wm_values, metadata = self._proposal_scores(policy_input, sample)
        result = dict(base_eval)
        result["value"] = self.adapter(base_eval["value"].reshape(1), wm_values, log_probs, wm_enabled=self.wm_enabled).reshape(())
        result["proposal_binding"] = metadata
        return result

    @property
    def architecture_evidence(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "global_head_names": GLOBAL_HEAD_NAMES,
            "global_head_units": GLOBAL_HEAD_UNITS,
            "critic_adapter": "5->32->1_zero_initialized_output",
            "wm_source_identity": self.source_identity,
            "wm_enabled": self.wm_enabled,
            "actor_probability_binding": "base_policy.evaluate_batch complete action masks",
            "actor_reads_wm": False,
            "wm_trainable_in_policy_optimizer": False,
        }


class GppoEngineeringTrainer:
    """A bounded PPO update that accepts complete action batches only."""

    def __init__(self, policy: CriticAugmentedGraphPolicy, *, data_identity: str, learning_rate: float = 3e-4):
        if not data_identity:
            raise ValueError("GPPO_DATA_IDENTITY_REQUIRED")
        self.policy = policy
        self.data_identity = data_identity
        self.optimizer = torch.optim.Adam([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=learning_rate)
        self.update_index = 0
        self.next_batch_index = 0

    def update(self, policy_input: dict[str, Any], *, action_batch: list[str], action_masks: list[dict[str, Any]], old_log_probability: torch.Tensor, advantage: torch.Tensor, return_target: torch.Tensor, batch_index: int, clip_ratio: float = 0.2) -> dict[str, Any]:
        if batch_index != self.next_batch_index:
            raise ValueError("GPPO_BATCH_BOUNDARY")
        if not action_batch or len(action_batch) != len(set(action_batch)):
            raise ValueError("GPPO_COMPLETE_ACTION_BATCH_REQUIRED")
        result = self.policy.evaluate_batch(policy_input, action_batch, action_masks)
        log_probability = result["log_prob"].reshape(())
        value = result["value"].reshape(())
        ratio = torch.exp(log_probability - old_log_probability.reshape(()))
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
        actor_loss = -torch.minimum(ratio * advantage.reshape(()), clipped * advantage.reshape(()))
        value_loss = 0.5 * (value - return_target.reshape(())) ** 2
        loss = actor_loss + value_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("GPPO_NONFINITE_LOSS")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in self.policy.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise ValueError("GPPO_NONFINITE_OR_EMPTY_GRADIENT")
        self.optimizer.step()
        self.update_index += 1
        self.next_batch_index = batch_index + 1
        return {
            "update_index": self.update_index,
            "batch_index": batch_index,
            "actor_loss": float(actor_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "complete_action_probability": float(log_probability.detach().exp().item()),
            "proposal_binding": result.get("proposal_binding"),
            "world_model_frozen": all(not parameter.requires_grad for parameter in self.policy.world_model.parameters()),
        }


def assert_paired_actor_initialization(first: nn.Module, second: nn.Module, policy_input: dict[str, Any]) -> None:
    first_ids, first_logits, _ = first.base_policy(policy_input)
    second_ids, second_logits, _ = second.base_policy(policy_input)
    if first_ids != second_ids or not torch.equal(first_logits, second_logits):
        raise ValueError("G_GWM_INITIAL_ACTOR_MISMATCH")
