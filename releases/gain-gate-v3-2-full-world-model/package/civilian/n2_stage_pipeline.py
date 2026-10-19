"""Bounded N2 paired-data, world-model, and graph PPO stage actions.

The driver owns admission, guardian process lifetime, receipts, and settlement.
This module owns the five computational actions between admission and export.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

from package_layout import require_package_root

import numpy as np
import torch
from torch.nn import functional as F

from baselines import choose_step_c1_actions
from budget_ledger import WORK_KIND_LIMITS
from branching import (
    ENGINEERING_WINDOW_STARTS,
    build_branch_record,
    capture_snapshot,
    restore_snapshot,
)
from graph_policy import GraphPolicy, _tensor_digest, build_policy
from independent_consequence_oracle import validate_collection_windows
from ppo_trainer import compute_gae
from relief_env import ReliefEnv
from resource_supervisor import (
    SETTLEMENT_RESERVE_CPU_SECONDS,
    SETTLEMENT_RESERVE_WALL_SECONDS,
    cumulative_budget_limits,
)
from scenario_matrix import HORIZON_MINUTES, MAX_PUBLIC_CANDIDATES, build_parent_matrix
from stable_identity import key_digest
from world_model import (
    CriticAugmentedGraphPolicy,
    CriticOnlyGppoAdapter,
    DynamicConsequenceModel,
    GLOBAL_HEAD_NAMES,
    GppoEngineeringTrainer,
    TASK_HEAD_NAMES,
    assert_paired_actor_initialization,
    load_consequence_checkpoint,
    public_branch_batch,
    public_policy_tensors,
    train_consequence_model,
)


SCHEMA = "n2-civilian-stage-pipeline/1"
COLLECTION_BUDGET = {
    "train_parents": 2,
    "development_parents": 1,
    "windows": len(ENGINEERING_WINDOW_STARTS) * 3,
    "branches": 128,
}
PPO_BUDGET = {
    "variants": ("G", "G-WM"),
    "rounds_per_variant": 1,
    "episodes_per_round": 2,
    "updates_per_round": 8,
    "total_updates": 16,
    "development_episodes": 2,
}
WINDOW_REPEATS = 0
WM_SEED = 8201
WM_TRAIN_EPOCHS = 1
WM_TRAIN_BATCH_SIZE = 4
WM_MAX_UPDATES = 8
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 60
STAGE_DEPENDENCIES = {
    "paired_consequence_collection": (
        "scenario_matrix.build_parent_matrix",
        "relief_env.ReliefEnv.reset/step/candidates/policy_input",
        "baselines.choose_step_c1_actions",
        "branching.capture_snapshot/restore_snapshot/build_branch_record",
        "stable_identity.key_digest",
    ),
    "world_model_training": (
        "world_model.public_branch_batch/train_consequence_model",
    ),
    "checkpoint_restore": (
        "world_model.load_consequence_checkpoint/public_branch_batch/public_policy_tensors",
        "subprocess.run(sys.executable, -B) strict-restore worker",
    ),
    "graph_policy_training": (
        "graph_policy.build_policy/encode_public_graph/GraphPolicy.sample_batch/evaluate_batch/_mask_at_step",
        "world_model.CriticAugmentedGraphPolicy/CriticOnlyGppoAdapter/GppoEngineeringTrainer/assert_paired_actor_initialization",
        "ppo_trainer.compute_gae",
        "relief_env.ReliefEnv.reset/step/export_trajectory",
    ),
    "development_evaluation": (
        "graph_policy.build_policy/encode_public_graph",
        "world_model.load_consequence_checkpoint/CriticAugmentedGraphPolicy/CriticOnlyGppoAdapter",
        "relief_env.ReliefEnv.reset/step/export_trajectory",
    ),
}


def _world_model_work_upper_bound() -> dict[str, Any]:
    training_window_count = len(ENGINEERING_WINDOW_STARTS) * COLLECTION_BUDGET["train_parents"]
    development_window_count = len(ENGINEERING_WINDOW_STARTS) * COLLECTION_BUDGET["development_parents"]
    strict_restore_probes = min(4, training_window_count)
    training_forwards = WM_MAX_UPDATES
    strict_restore_forwards = strict_restore_probes * 2
    prediction_check_forwards = development_window_count * 2
    gwm_rollout_transitions = PPO_BUDGET["rounds_per_variant"] * PPO_BUDGET["episodes_per_round"] * HORIZON_MINUTES
    gwm_development_episodes = PPO_BUDGET["development_episodes"] // len(PPO_BUDGET["variants"])
    gwm_development_transitions = gwm_development_episodes * HORIZON_MINUTES
    forward_components = {
        "training": training_forwards,
        "strict_restore": strict_restore_forwards,
        "prediction_check": prediction_check_forwards,
        "gwm_rollout": gwm_rollout_transitions,
        "gwm_ppo_update": 0,
        "gwm_development_evaluation": gwm_development_transitions,
    }
    candidate_components = {
        "training": training_forwards * min(WM_TRAIN_BATCH_SIZE, training_window_count) * MAX_PUBLIC_CANDIDATES,
        "strict_restore": strict_restore_forwards * MAX_PUBLIC_CANDIDATES,
        "prediction_check": prediction_check_forwards * MAX_PUBLIC_CANDIDATES,
        "gwm_rollout": gwm_rollout_transitions * MAX_PUBLIC_CANDIDATES,
        "gwm_ppo_update": 0,
        "gwm_development_evaluation": gwm_development_transitions * MAX_PUBLIC_CANDIDATES,
    }
    return {
        "forward_batch_count": sum(forward_components.values()),
        "candidate_evaluation_count": sum(candidate_components.values()),
        "ppo_cache_lookup_count": PPO_BUDGET["rounds_per_variant"] * PPO_BUDGET["updates_per_round"] * PPO_MINIBATCH_SIZE,
        "forward_components": forward_components,
        "candidate_components": candidate_components,
    }


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _jsonl_write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) for row in rows))


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(supervisor: Any) -> None:
    if supervisor is not None:
        supervisor.check()


def _require_work_ledger(supervisor: Any) -> None:
    if supervisor is None or any(
        not callable(getattr(supervisor, name, None))
        for name in ("reserve_work_call", "complete_work_call", "fail_work_call")
    ):
        raise PermissionError("N2_WORK_CALL_LEDGER_API_REQUIRED")


def _reserve_work_call(supervisor: Any, kind: str, estimated_count: int, metadata: dict[str, Any]) -> int:
    _require_work_ledger(supervisor)
    _check(supervisor)
    return supervisor.reserve_work_call(kind, estimated_count, metadata)


def _fail_work_call(supervisor: Any, token: int, error: BaseException) -> None:
    try:
        supervisor.fail_work_call(token, error)
    except BaseException as ledger_error:
        if hasattr(error, "add_note"):
            error.add_note(f"work-call failure could not be settled: {type(ledger_error).__name__}: {ledger_error}")


def _complete_work_call(
    supervisor: Any,
    token: int,
    actual_count: int,
    metadata: dict[str, Any],
) -> None:
    try:
        supervisor.complete_work_call(token, actual_count, metadata)
    except BaseException as exc:
        _fail_work_call(supervisor, token, exc)
        raise


@contextmanager
def _reserved_work_call(
    supervisor: Any,
    kind: str,
    estimated_count: int,
    metadata: dict[str, Any],
    *,
    actual_count: int | None = None,
    actual_metadata: dict[str, Any] | None = None,
):
    token = _reserve_work_call(supervisor, kind, estimated_count, metadata)
    try:
        yield token
    except BaseException as exc:
        _fail_work_call(supervisor, token, exc)
        raise
    else:
        _complete_work_call(
            supervisor,
            token,
            estimated_count if actual_count is None else actual_count,
            metadata if actual_metadata is None else actual_metadata,
        )


def _run_reserved_work_call(
    supervisor: Any,
    kind: str,
    estimated_count: int,
    metadata: dict[str, Any],
    action: Callable[[], Any],
    *,
    actual_count: Callable[[Any], int] | None = None,
    actual_metadata: Callable[[Any], dict[str, Any]] | None = None,
) -> Any:
    token = _reserve_work_call(supervisor, kind, estimated_count, metadata)
    try:
        result = action()
    except BaseException as exc:
        _fail_work_call(supervisor, token, exc)
        raise
    _complete_work_call(
        supervisor,
        token,
        estimated_count if actual_count is None else actual_count(result),
        metadata if actual_metadata is None else actual_metadata(result),
    )
    return result


def _run_reserved_work_calls(
    supervisor: Any,
    requests: Sequence[dict[str, Any]],
    action: Callable[[], Any],
) -> Any:
    tokens: list[tuple[int, dict[str, Any]]] = []
    try:
        for request in requests:
            token = _reserve_work_call(
                supervisor,
                request["kind"],
                request["estimated_count"],
                request["metadata"],
            )
            tokens.append((token, request))
    except BaseException as exc:
        for token, _ in tokens:
            _fail_work_call(supervisor, token, exc)
        raise
    try:
        result = action()
    except BaseException as exc:
        for token, _ in tokens:
            _fail_work_call(supervisor, token, exc)
        raise
    for index, (token, request) in enumerate(tokens):
        actual_count = request.get("actual_count", request["estimated_count"])
        if callable(actual_count):
            actual_count = actual_count(result)
        actual_metadata = request.get("actual_metadata", request["metadata"])
        if callable(actual_metadata):
            actual_metadata = actual_metadata(result)
        try:
            _complete_work_call(supervisor, token, actual_count, actual_metadata)
        except BaseException as exc:
            for pending_token, _pending_request in tokens[index + 1:]:
                _fail_work_call(supervisor, pending_token, exc)
            raise
    return result


def _forward_candidate_count(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    candidate_mask = kwargs.get("candidate_mask")
    if candidate_mask is None and len(args) > 3:
        candidate_mask = args[3]
    if not torch.is_tensor(candidate_mask):
        raise ValueError("N2_WM_FORWARD_CANDIDATE_MASK_MISSING")
    count = int(candidate_mask.sum().item())
    if count < 1:
        raise ValueError("N2_WM_FORWARD_EMPTY_CANDIDATE_BATCH")
    return count


@contextmanager
def _track_wm_forward_calls(supervisor: Any, label: str, model: DynamicConsequenceModel | None = None):
    counts = {"forward_batches": 0, "candidate_evaluations": 0}
    original_forward = DynamicConsequenceModel.forward if model is None else model.forward

    def tracked_forward(*args, **kwargs):
        candidates = _forward_candidate_count(args, kwargs)
        call_index = counts["forward_batches"] + 1
        metadata = {"component": label, "call_index": call_index, "candidate_count": candidates}
        output = _run_reserved_work_calls(
            supervisor,
            (
                {"kind": "wm_forward_batch", "estimated_count": 1, "metadata": metadata},
                {"kind": "wm_candidate_evaluation", "estimated_count": candidates, "metadata": metadata},
            ),
            lambda: original_forward(*args, **kwargs) if model is not None else original_forward(args[0], *args[1:], **kwargs),
        )
        counts["forward_batches"] += 1
        counts["candidate_evaluations"] += candidates
        return output

    if model is None:
        DynamicConsequenceModel.forward = tracked_forward
    else:
        model.forward = tracked_forward
    try:
        yield counts
    finally:
        if model is None:
            DynamicConsequenceModel.forward = original_forward
        else:
            model.forward = original_forward


def _wm_prediction_cache_usage(binding: Any, forward_call: dict[str, int], *, enabled: bool) -> dict[str, int]:
    if not isinstance(binding, dict):
        raise ValueError("N2_WM_CACHE_BINDING_MISSING")
    cache = binding.get("wm_prediction_cache")
    if not isinstance(cache, dict) or cache.get("enabled") is not enabled:
        raise ValueError("N2_WM_CACHE_ENABLEMENT_MISMATCH")
    hit = cache.get("hit")
    miss = cache.get("miss")
    forward_batches = cache.get("forward_batches")
    candidate_evaluations = cache.get("candidate_evaluations")
    if (
        type(hit) is not bool
        or type(miss) is not bool
        or type(forward_batches) is not int
        or type(candidate_evaluations) is not int
        or forward_batches < 0
        or candidate_evaluations < 0
        or forward_batches != forward_call["forward_batches"]
        or candidate_evaluations != forward_call["candidate_evaluations"]
    ):
        raise ValueError("N2_WM_CACHE_FORWARD_RECONCILIATION")
    if enabled:
        if hit == miss:
            raise ValueError("N2_WM_CACHE_EVENT_CARDINALITY")
        if hit and (forward_batches != 0 or candidate_evaluations != 0):
            raise ValueError("N2_WM_CACHE_HIT_PERFORMED_FORWARD")
        if miss and (forward_batches != 1 or candidate_evaluations != int(binding["proposal_count"])):
            raise ValueError("N2_WM_CACHE_MISS_FORWARD_SHAPE")
    elif hit or miss or forward_batches or candidate_evaluations:
        raise ValueError("N2_BASELINE_USED_WM_CACHE")
    return {
        "hit_count": int(hit),
        "miss_count": int(miss),
        "forward_batches": forward_batches,
        "candidate_evaluations": candidate_evaluations,
    }


@contextmanager
def _track_optimizer_steps(supervisor: Any, optimizer: Any, kind: str, label: str):
    counts = {"optimizer_updates": 0}
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        update_index = counts["optimizer_updates"] + 1
        result = _run_reserved_work_call(
            supervisor,
            kind,
            1,
            {"component": label, "update_index": update_index},
            lambda: original_step(*args, **kwargs),
        )
        counts["optimizer_updates"] += 1
        return result

    optimizer.step = tracked_step
    try:
        yield counts
    finally:
        optimizer.step = original_step


@contextmanager
def _track_adam_optimizer_steps(supervisor: Any, kind: str, label: str):
    counts = {"optimizer_updates": 0}
    original_step = torch.optim.Adam.step

    def tracked_step(optimizer, *args, **kwargs):
        update_index = counts["optimizer_updates"] + 1
        result = _run_reserved_work_call(
            supervisor,
            kind,
            1,
            {"component": label, "update_index": update_index},
            lambda: original_step(optimizer, *args, **kwargs),
        )
        counts["optimizer_updates"] += 1
        return result

    torch.optim.Adam.step = tracked_step
    try:
        yield counts
    finally:
        torch.optim.Adam.step = original_step


@contextmanager
def _track_torch_saves(supervisor: Any, label: str, final_path: Path | None = None):
    original_save = torch.save

    def tracked_save(value, destination, *args, **kwargs):
        raw_path = getattr(destination, "name", destination)
        path = Path(raw_path) if isinstance(raw_path, (str, os.PathLike)) else None
        metadata = {
            "component": label,
            "path": final_path.as_posix() if final_path is not None else (path.as_posix() if path is not None else "<stream>"),
            **({"temporary_path": path.as_posix()} if final_path is not None and path is not None else {}),
        }
        return _run_reserved_work_call(
            supervisor,
            "checkpoint_write",
            1,
            metadata,
            lambda: original_save(value, destination, *args, **kwargs),
        )

    torch.save = tracked_save
    try:
        yield
    finally:
        torch.save = original_save


@contextmanager
def _track_checkpoint_read_bytes(supervisor: Any, path: Path, component: str):
    target = path.resolve()
    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(source_path):
        if source_path.resolve() != target:
            return original_read_bytes(source_path)
        return _run_reserved_work_call(
            supervisor,
            "checkpoint_read",
            1,
            {"path": target.as_posix(), "component": component, "operation": "sha256_read"},
            lambda: original_read_bytes(source_path),
        )

    Path.read_bytes = tracked_read_bytes
    try:
        yield
    finally:
        Path.read_bytes = original_read_bytes


def _tracked_json_write(supervisor: Any, path: Path, value: Any) -> None:
    payload = _canonical(value)
    _run_reserved_work_call(
        supervisor,
        "output_write",
        1,
        {"path": path.as_posix(), "bytes": len(payload), "format": "json"},
        lambda: _json_write(path, value),
    )


def _tracked_jsonl_write(supervisor: Any, path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload_bytes = sum(len(_canonical(row)) for row in rows)
    _run_reserved_work_call(
        supervisor,
        "output_write",
        1,
        {"path": path.as_posix(), "bytes": payload_bytes, "format": "jsonl", "row_count": len(rows)},
        lambda: _jsonl_write(path, rows),
    )


def _tracked_torch_save(supervisor: Any, path: Path, value: Any, metadata: dict[str, Any]) -> None:
    _run_reserved_work_call(
        supervisor,
        "checkpoint_write",
        1,
        {"path": path.as_posix(), **metadata},
        lambda: torch.save(value, path),
    )


def _load_world_model_checkpoint(supervisor: Any, path: Path, *, data_identity: str, component: str):
    return _run_reserved_work_call(
        supervisor,
        "checkpoint_read",
        1,
        {"path": path.as_posix(), "component": component},
        lambda: load_consequence_checkpoint(path, data_identity=data_identity),
    )


def _load_policy_checkpoint(supervisor: Any, path: Path, component: str) -> dict[str, Any]:
    return _run_reserved_work_call(
        supervisor,
        "checkpoint_read",
        1,
        {"path": path.as_posix(), "component": component},
        lambda: torch.load(path, map_location="cpu", weights_only=False),
    )


def _tracked_checkpoint_sha256(supervisor: Any, path: Path, component: str) -> str:
    return _run_reserved_work_call(
        supervisor,
        "checkpoint_read",
        1,
        {"path": path.as_posix(), "component": component, "operation": "sha256_read"},
        lambda: _sha256_file(path),
    )


def _budget_snapshot(supervisor: Any) -> dict[str, Any]:
    if supervisor is None:
        return {"supervisor_present": False}
    wall_limit, cpu_limit = cumulative_budget_limits(supervisor.stage)
    total_wall = float(supervisor._prior_total_wall) + float(supervisor.wall_seconds)
    total_cpu = float(supervisor._prior_total_cpu) + float(supervisor.cpu_seconds)
    return {
        "supervisor_present": True,
        "stage": supervisor.stage,
        "cumulative_work_wall_limit_seconds": wall_limit,
        "cumulative_work_cpu_limit_seconds": cpu_limit,
        "cumulative_wall_before_seconds": total_wall,
        "cumulative_cpu_before_seconds": total_cpu,
        "remaining_cumulative_work_wall_seconds": max(0.0, wall_limit - total_wall),
        "remaining_cumulative_work_cpu_seconds": max(0.0, cpu_limit - total_cpu),
        "settlement_export_reserve_wall_seconds": SETTLEMENT_RESERVE_WALL_SECONDS,
        "settlement_export_reserve_cpu_seconds": SETTLEMENT_RESERVE_CPU_SECONDS,
        "enforcement": "ResourceSupervisor.check and its monitor enforce cumulative work caps before settlement reserve",
    }


def _action_batch(actions: Sequence[str]) -> list[str]:
    items = list(actions)
    if not items:
        return ["NOOP"]
    if "NOOP" in items and items[-1] != "NOOP":
        raise ValueError("N2_ACTION_BATCH_NOOP_ORDER")
    if len(items) != len(set(items)):
        raise ValueError("N2_ACTION_BATCH_DUPLICATE")
    return items


def _candidate_batches(env: ReliefEnv) -> tuple[list[list[str]], int]:
    catalog = env.candidates()
    catalog_ids = [str(item["id"]) for item in catalog]
    if catalog_ids.count("NOOP") != 1:
        raise ValueError("N2_CANDIDATE_CATALOG_NOOP")
    proposals: list[list[str]] = [["NOOP"]]
    c1 = [str(item["id"]) for item in choose_step_c1_actions(env.public_observation(), catalog)]
    proposals.append(_action_batch(c1))
    proposals.extend([[candidate_id] for candidate_id in catalog_ids if candidate_id != "NOOP"])
    unique: list[list[str]] = []
    seen: set[str] = set()
    for actions in proposals:
        digest = key_digest(actions)
        if digest not in seen:
            seen.add(digest)
            unique.append(actions)
    return unique[:MAX_PUBLIC_CANDIDATES], max(0, len(unique) - MAX_PUBLIC_CANDIDATES)


def _step_environment(env: ReliefEnv, actions: list[str], supervisor: Any, metadata: dict[str, Any]):
    return _run_reserved_work_call(
        supervisor,
        "environment_step",
        1,
        metadata,
        lambda: env.step(actions),
    )


def _continue_branch(env: ReliefEnv, first_actions: list[str], horizon: int, supervisor: Any, metadata: dict[str, Any]) -> tuple[float, bool, int]:
    start = int(env.time)
    limit = min(start + 30, int(horizon))
    total_reward = 0.0
    action_batch = [item for item in first_actions if item != "NOOP"]
    first = True
    while int(env.time) < limit:
        if not first:
            catalog = env.candidates()
            action_batch = [str(item["id"]) for item in choose_step_c1_actions(env.public_observation(), catalog)]
        _, reward, terminated, truncated, _ = _step_environment(
            env,
            action_batch,
            supervisor,
            {**metadata, "step_index": len(env.transitions) - int(getattr(env, "_branch_prefix_length", 0)) + 1},
        )
        total_reward += float(reward)
        first = False
        if terminated or truncated:
            break
    return total_reward, bool(int(env.time) >= limit), len(env.transitions) - int(getattr(env, "_branch_prefix_length", 0))


def _collect_parent(scenario: dict[str, Any], split: str, parent_index: int, supervisor: Any) -> tuple[list[dict], dict]:
    parent_id = str(scenario["scenario_id"])
    event_tape_id = key_digest({"scenario_id": parent_id, "disturbances": scenario["disturbances"]})
    source = ReliefEnv(scenario)
    source.reset(policy="C1")
    windows: list[dict] = []
    branches: list[dict] = []
    omitted_total = 0
    branch_step_count = 0
    branch_restore_count = 0
    source_step_count = 0
    terminated = False
    while not terminated:
        _check(supervisor)
        now = int(source.time)
        if now in ENGINEERING_WINDOW_STARTS:
            window_id = f"{parent_id}:0:{now}"
            snapshot = capture_snapshot(
                source,
                parent_id=parent_id,
                repeat=WINDOW_REPEATS,
                window_id=window_id,
                event_tape_id=event_tape_id,
            )
            snapshot_payload = json.loads(snapshot)
            policy_input = {"observation": snapshot_payload["public_prefix"], "candidates": snapshot_payload["candidate_catalog"]}
            snapshot_env = restore_snapshot(snapshot)
            candidate_batches, omitted = _candidate_batches(snapshot_env)
            omitted_total += omitted
            window_records: list[dict] = []
            trajectory_evidence: dict[str, list[list[str]]] = {}
            trajectory_transitions: dict[str, list[dict]] = {}
            noop_reference_env: ReliefEnv | None = None
            noop_reference_return: float | None = None
            for candidate_actions in candidate_batches:
                _check(supervisor)
                candidate_id = key_digest(candidate_actions)
                candidate_meta = {"parent_id": parent_id, "window_id": window_id, "candidate_id": candidate_id}
                def run_candidate_branch():
                    candidate_env = restore_snapshot(snapshot)
                    candidate_outcome = _continue_branch(candidate_env, candidate_actions, int(scenario["horizon"]), supervisor, candidate_meta)
                    return candidate_env, candidate_outcome

                candidate_env, candidate_outcome = _run_reserved_work_call(
                    supervisor,
                    "candidate_branch",
                    1,
                    candidate_meta,
                    run_candidate_branch,
                    actual_metadata=lambda result: {**candidate_meta, "actual_step_count": result[1][2]},
                )
                candidate_return, _, candidate_steps = candidate_outcome
                prefix_length = int(getattr(candidate_env, "_branch_prefix_length", 0))
                trajectory_evidence[candidate_id] = [
                    [str(item.get("id")) for item in transition.get("selected_actions", [])]
                    for transition in candidate_env.transitions[prefix_length:]
                ]
                trajectory_transitions[candidate_id] = copy.deepcopy(candidate_env.transitions[prefix_length:])
                branch_restore_count += 1
                branch_step_count += candidate_steps
                if candidate_actions == ["NOOP"]:
                    noop_reference_env = candidate_env
                    noop_reference_return = candidate_return
                    trajectory_evidence["NOOP"] = trajectory_evidence[candidate_id]
                    trajectory_transitions["NOOP"] = trajectory_transitions[candidate_id]
                if noop_reference_env is None or noop_reference_return is None:
                    raise ValueError("N2_NOOP_REFERENCE_REQUIRED_BEFORE_CANDIDATES")
                record = build_branch_record(
                    snapshot,
                    candidate_action_batch=candidate_actions,
                    reference_action_batch=["NOOP"],
                    parent_id=parent_id,
                    repeat=WINDOW_REPEATS,
                    window_id=window_id,
                    event_tape_id=event_tape_id,
                    candidate_id=key_digest(candidate_actions),
                    candidate_return=candidate_return,
                    reference_return=noop_reference_return,
                    candidate_env=candidate_env,
                    reference_env=noop_reference_env,
                )
                window_records.append(record)
                branches.append(record)
            windows.append({
                "policy_input": policy_input,
                "branch_records": window_records,
                "window_ids": [parent_id, WINDOW_REPEATS, window_id],
                "split": split,
                "snapshot_json": snapshot,
                "trajectory_evidence": trajectory_evidence,
                "trajectory_transitions": trajectory_transitions,
            })
        catalog = source.candidates()
        actions = [str(item["id"]) for item in choose_step_c1_actions(source.public_observation(), catalog)]
        _, _, terminated, truncated, _ = _step_environment(
            source,
            actions,
            supervisor,
            {"component": "collection_source_episode", "parent_id": parent_id, "step_index": source_step_count + 1},
        )
        source_step_count += 1
        if truncated:
            raise ValueError("N2_SOURCE_C1_TRUNCATED")
    return windows, {
        "parent_id": parent_id,
        "split": split,
        "parent_index": parent_index,
        "window_count": len(windows),
        "branch_count": sum(len(item["branch_records"]) for item in windows),
        "candidate_branch_count": sum(len(item["branch_records"]) for item in windows),
        "omitted_proposals": omitted_total,
        "source_prefix_transition_count": len(source.transitions),
        "source_step_count": source_step_count,
        "branch_rollout_step_count": branch_step_count,
        "branch_restore_count": branch_restore_count,
        "event_tape_id": event_tape_id,
    }


def run_paired_consequence_collection(package_root: Path, work_root: Path, supervisor: Any = None) -> dict:
    del package_root
    _require_work_ledger(supervisor)
    started = time.perf_counter()
    matrix = build_parent_matrix()
    if (
        len(matrix["train"]) != COLLECTION_BUDGET["train_parents"]
        or len(matrix["development"]) != COLLECTION_BUDGET["development_parents"]
    ):
        raise ValueError("N2_PARENT_MATRIX_SIZE")
    all_windows: list[dict] = []
    parent_receipts: list[dict] = []
    for split in ("train", "development"):
        for index, scenario in enumerate(matrix[split]):
            rows, receipt = _collect_parent(scenario, split, index, supervisor)
            all_windows.extend(rows)
            parent_receipts.append(receipt)
    train_windows = [item for item in all_windows if item["split"] == "train"]
    dev_windows = [item for item in all_windows if item["split"] == "development"]
    branch_records = [record for item in train_windows for record in item["branch_records"]]
    development_branch_count = sum(len(item["branch_records"]) for item in dev_windows)
    all_branch_count = sum(item["branch_count"] for item in parent_receipts)
    expected_train_windows = len(ENGINEERING_WINDOW_STARTS) * COLLECTION_BUDGET["train_parents"]
    expected_development_windows = len(ENGINEERING_WINDOW_STARTS) * COLLECTION_BUDGET["development_parents"]
    if (
        len(train_windows) != expected_train_windows
        or len(dev_windows) != expected_development_windows
        or len(all_windows) != COLLECTION_BUDGET["windows"]
        or all_branch_count > COLLECTION_BUDGET["branches"]
        or len(branch_records) > COLLECTION_BUDGET["branches"]
        or development_branch_count > COLLECTION_BUDGET["branches"]
        or any(len(item["branch_records"]) > MAX_PUBLIC_CANDIDATES for item in all_windows)
    ):
        raise ValueError("N2_COLLECTION_BUDGET_OR_WINDOW_COUNT")
    def oracle_step(env: ReliefEnv, actions: list[str], metadata: dict[str, Any]):
        return _step_environment(
            env,
            actions,
            supervisor,
            {"component": "independent_consequence_oracle", **metadata},
        )

    oracle_receipt = validate_collection_windows(train_windows, step_callback=oracle_step)
    if oracle_receipt.get("status") != "passed" or oracle_receipt.get("qualifying_window_count", 0) < 1:
        raise PermissionError("N2_COLLECTION_REQUIRES_COMMON_VALID_GLOBAL_DIFFERENCE")
    oracle_receipt["training_windows_identity_sha256"] = _sha256_bytes(_canonical(train_windows))
    root = work_root / "n2_pipeline"
    _tracked_json_write(supervisor, root / "parents.json", matrix)
    _tracked_json_write(supervisor, root / "training_windows.json", train_windows)
    _tracked_json_write(supervisor, root / "development_windows.json", dev_windows)
    _tracked_jsonl_write(supervisor, root / "training_branches.jsonl", branch_records)
    receipt = {
        "schema": SCHEMA,
        "stage": "paired_consequence_collection",
        "status": "complete",
        "train_parent_count": COLLECTION_BUDGET["train_parents"],
        "development_parent_count": COLLECTION_BUDGET["development_parents"],
        "training_window_count": len(train_windows),
        "development_window_count": len(dev_windows),
        "training_branch_count": len(branch_records),
        "development_branch_count": development_branch_count,
        "training_candidate_branch_count": len(branch_records),
        "development_candidate_branch_count": development_branch_count,
        "source_step_count": sum(item["source_step_count"] for item in parent_receipts),
        "candidate_branch_rollout_step_count": sum(item["branch_rollout_step_count"] for item in parent_receipts),
        "branch_restore_count": sum(item["branch_restore_count"] for item in parent_receipts),
        "noop_baseline_window_count": len(all_windows),
        "separate_reference_branch_count": 0,
        "separate_noop_reference_rollout_count": 0,
        "all_branch_count": all_branch_count,
        "all_candidate_branch_count": all_branch_count,
        "maximum_branch_count": COLLECTION_BUDGET["branches"],
        "maximum_branches_per_window": MAX_PUBLIC_CANDIDATES,
        "paired_snapshot_semantics": "each candidate, including NOOP, is restored from one immutable source snapshot; the NOOP candidate rollout is the per-window relative baseline reused by all candidate records; no separate reference rollout is run",
        "parents": parent_receipts,
        "training_windows_sha256": _sha256_file(root / "training_windows.json"),
        "development_windows_sha256": _sha256_file(root / "development_windows.json"),
        "training_branches_sha256": _sha256_file(root / "training_branches.jsonl"),
        "independent_consequence_oracle": oracle_receipt,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    _tracked_json_write(supervisor, root / "collection_receipt.json", receipt)
    return receipt


def _prediction_hash(model: DynamicConsequenceModel, policy_input: dict[str, Any]) -> dict[str, str]:
    inputs, binding = public_policy_tensors(policy_input)
    with torch.no_grad():
        output = model(**inputs, binding=binding)
    return {
        "global": _sha256_bytes(output["global"].detach().cpu().contiguous().numpy().tobytes()),
        "per_task": _sha256_bytes(output["per_task"].detach().cpu().contiguous().numpy().tobytes()),
    }


def _wm_policy_input(window: dict[str, Any]) -> dict[str, Any]:
    """Bind WM proposal slots to the complete action identities in records."""
    policy_input = copy.deepcopy(window["policy_input"])
    proposals = []
    for record in window["branch_records"]:
        actions = list(record["candidate_action_batch"])
        if not actions or len(actions) != len(set(actions)) or ("NOOP" in actions and actions[-1] != "NOOP"):
            raise ValueError("N2_WM_PROPOSAL_ACTION_RECORD")
        proposals.append({"actions": actions, "candidate_id": str(record["candidate_id"])})
    if not proposals:
        raise ValueError("N2_WM_EMPTY_PROPOSALS")
    if [str(item["candidate_id"]) for item in proposals] != [key_digest(item["actions"]) for item in proposals]:
        raise ValueError("N2_WM_PROPOSAL_DIGEST_BINDING")
    policy_input["proposal_batches"] = proposals
    return policy_input


def _strict_restore_subprocess(package_root: Path, checkpoint_path: Path, data_identity: str, policy_inputs: list[dict], receipt_path: Path, supervisor: Any) -> dict:
    if not policy_inputs:
        raise ValueError("WM_STRICT_RESTORE_EMPTY_PROBES")
    request_path = receipt_path.with_suffix(".request.json")
    _tracked_json_write(supervisor, request_path, {"policy_inputs": policy_inputs})
    expected_model, expected_payload = _load_world_model_checkpoint(
        supervisor, checkpoint_path, data_identity=data_identity, component="strict_restore_parent_reference"
    )
    with _track_wm_forward_calls(supervisor, "strict_restore_parent_reference", expected_model) as parent_calls:
        expected_hashes = [_prediction_hash(expected_model, value) for value in policy_inputs]
    expected_candidate_evaluations = sum(
        int(public_policy_tensors(policy_input)[0]["candidate_mask"].sum().item())
        for policy_input in policy_inputs
    )
    worker = r'''import hashlib, json, pathlib, sys
root, checkpoint, data_identity, request_path, receipt_path = sys.argv[1:]
sys.path.insert(0, root)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from world_model import load_consequence_checkpoint, public_policy_tensors
model, payload = load_consequence_checkpoint(checkpoint, data_identity=data_identity)
request = json.loads(pathlib.Path(request_path).read_text(encoding="utf-8"))
hashes = []
candidate_evaluation_count = 0
with torch.no_grad():
    for policy_input in request["policy_inputs"]:
        inputs, binding = public_policy_tensors(policy_input)
        candidate_evaluation_count += int(inputs["candidate_mask"].sum().item())
        output = model(**inputs, binding=binding)
        hashes.append({name: hashlib.sha256(output[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name in ("global", "per_task")})
result = {"schema": "n2-world-model-strict-restore-subprocess/1", "status": "complete", "pid": __import__("os").getpid(), "checkpoint_sha256": hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest(), "model_state_sha256": payload["model_state_sha256"], "prediction_hashes": hashes, "prediction_forward_count": len(hashes), "candidate_evaluation_count": candidate_evaluation_count, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_is_initialized": bool(torch.cuda.is_initialized())}
pathlib.Path(receipt_path).write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
'''
    def run_worker() -> dict[str, Any]:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", worker, str(package_root), str(checkpoint_path), data_identity, str(request_path), str(receipt_path)],
            cwd=str(package_root),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not receipt_path.is_file():
            raise RuntimeError(f"WM_STRICT_RESTORE_SUBPROCESS_FAILED:{completed.returncode}:{completed.stderr[-2000:]}")
        result = _json_read(receipt_path)
        if (
            result.get("status") != "complete"
            or result.get("pid") == os.getpid()
            or result.get("checkpoint_sha256") != _tracked_checkpoint_sha256(supervisor, checkpoint_path, "strict_restore_parent_verification")
            or result.get("model_state_sha256") != expected_payload["model_state_sha256"]
            or result.get("prediction_hashes") != expected_hashes
            or result.get("prediction_forward_count") != len(policy_inputs)
            or result.get("candidate_evaluation_count") != expected_candidate_evaluations
            or result.get("torch_threads") != 1
            or result.get("torch_interop_threads") != 1
            or result.get("cuda_is_initialized") is not False
        ):
            raise ValueError("WM_STRICT_RESTORE_SUBPROCESS_PREDICTION_MISMATCH")
        request_path.unlink(missing_ok=True)
        return result

    result = _run_reserved_work_calls(
        supervisor,
        (
            {"kind": "checkpoint_read", "estimated_count": 2, "metadata": {"path": checkpoint_path.as_posix(), "component": "strict_restore_subprocess", "operations": ["checkpoint_load", "sha256_read"]}},
            {"kind": "wm_forward_batch", "estimated_count": len(policy_inputs), "metadata": {"component": "strict_restore_subprocess"}},
            {"kind": "wm_candidate_evaluation", "estimated_count": expected_candidate_evaluations, "metadata": {"component": "strict_restore_subprocess"}},
            {"kind": "output_write", "estimated_count": 1, "metadata": {"path": receipt_path.as_posix(), "format": "json", "component": "strict_restore_subprocess"}},
        ),
        run_worker,
    )
    result["parent_reference_prediction_forward_count"] = parent_calls["forward_batches"]
    result["parent_reference_candidate_evaluation_count"] = parent_calls["candidate_evaluations"]
    result["strict_restore_prediction_forward_count"] = parent_calls["forward_batches"] + result["prediction_forward_count"]
    result["strict_restore_candidate_evaluation_count"] = parent_calls["candidate_evaluations"] + result["candidate_evaluation_count"]
    return result


def run_world_model_training(package_root: Path, work_root: Path, supervisor: Any = None) -> dict:
    del package_root
    _require_work_ledger(supervisor)
    root = work_root / "n2_pipeline"
    collection = _json_read(root / "collection_receipt.json")
    expected_training_windows = len(ENGINEERING_WINDOW_STARTS) * COLLECTION_BUDGET["train_parents"]
    if collection.get("status") != "complete" or collection.get("training_window_count") != expected_training_windows:
        raise PermissionError("N2_COLLECTION_RECEIPT_REQUIRED")
    windows = _json_read(root / "training_windows.json")
    branches_path = root / "training_branches.jsonl"
    if collection.get("training_windows_sha256") != _sha256_file(root / "training_windows.json") or collection.get("training_branches_sha256") != _sha256_file(branches_path):
        raise PermissionError("N2_COLLECTION_INPUT_HASH_MISMATCH")
    persisted_branches = [json.loads(line) for line in branches_path.read_text(encoding="utf-8").splitlines() if line]
    consumed_branches = [record for window in windows for record in window["branch_records"]]
    if persisted_branches != consumed_branches:
        raise PermissionError("N2_COLLECTION_WM_CONSUMED_DATA_MISMATCH")
    oracle_receipt = collection.get("independent_consequence_oracle", {})
    if (
        oracle_receipt.get("status") != "passed"
        or oracle_receipt.get("qualifying_window_count", 0) < 1
        or oracle_receipt.get("training_windows_identity_sha256") != _sha256_bytes(_canonical(windows))
    ):
        raise PermissionError("N2_COLLECTION_WM_ORACLE_REPLAY_MISMATCH")
    samples = [public_branch_batch(_wm_policy_input(item), item["branch_records"], window_id=tuple(item["window_ids"])) for item in windows]
    data_identity = _sha256_bytes(branches_path.read_bytes())
    consumed_window_identity = _sha256_bytes(_canonical(windows))
    checkpoint_path = root / "world_model.pt"
    with _track_wm_forward_calls(supervisor, "world_model_training") as training_forward_calls:
        with _track_adam_optimizer_steps(supervisor, "wm_optimizer_update", "world_model_training") as training_update_calls:
            with _track_torch_saves(supervisor, "world_model_training", checkpoint_path):
                with _track_checkpoint_read_bytes(supervisor, checkpoint_path, "world_model_training_integrity_check"):
                    training = train_consequence_model(
                        samples,
                        checkpoint_path=checkpoint_path,
                        data_identity=data_identity,
                        seed=WM_SEED,
                        epochs=WM_TRAIN_EPOCHS,
                        batch_size=WM_TRAIN_BATCH_SIZE,
                        max_updates=WM_MAX_UPDATES,
                    )
    if (
        training_forward_calls["forward_batches"] != training["updates"]
        or training_update_calls["optimizer_updates"] != training["updates"]
        or training_forward_calls["forward_batches"] > WM_MAX_UPDATES
    ):
        raise AssertionError("N2_WM_TRAINING_CALL_LEDGER_RECONCILIATION")
    receipt = {
        "schema": SCHEMA,
        "stage": "world_model_training",
        "status": "complete",
        "seed": WM_SEED,
        "epochs": WM_TRAIN_EPOCHS,
        "batch_size": WM_TRAIN_BATCH_SIZE,
        "maximum_updates": WM_MAX_UPDATES,
        "training_windows": len(samples),
        "updates": training["updates"],
        "checkpoint_path": checkpoint_path.relative_to(work_root).as_posix(),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "data_identity": data_identity,
        "consumed_window_identity": consumed_window_identity,
        "independent_consequence_oracle_sha256": oracle_receipt["evidence_sha256"],
        "training_optimizer_updates": training_update_calls["optimizer_updates"],
        "training_prediction_forward_count": training_forward_calls["forward_batches"],
        "training_candidate_evaluation_count": training_forward_calls["candidate_evaluations"],
    }
    _tracked_json_write(supervisor, root / "world_model_training_receipt.json", receipt)
    return receipt


def run_checkpoint_restore(package_root: Path, work_root: Path, supervisor: Any = None) -> dict:
    _require_work_ledger(supervisor)
    root = work_root / "n2_pipeline"
    training_receipt = _json_read(root / "world_model_training_receipt.json")
    if training_receipt.get("status") != "complete":
        raise PermissionError("N2_WM_TRAINING_RECEIPT_REQUIRED")
    training_windows = _json_read(root / "training_windows.json")
    strict = _strict_restore_subprocess(
        package_root,
        work_root / training_receipt["checkpoint_path"],
        training_receipt["data_identity"],
        [_wm_policy_input(item) for item in training_windows[:4]],
        root / "strict_restore_subprocess.json",
        supervisor,
    )
    windows = _json_read(root / "development_windows.json")
    model, _ = _load_world_model_checkpoint(
        supervisor,
        work_root / training_receipt["checkpoint_path"],
        data_identity=training_receipt["data_identity"],
        component="checkpoint_restore",
    )
    check_rows = []
    abs_errors: list[float] = []
    global_head_errors: dict[str, list[float]] = {name: [] for name in GLOBAL_HEAD_NAMES}
    global_head_counts = {name: 0 for name in GLOBAL_HEAD_NAMES}
    global_candidate_opportunities = 0
    task_head_bce_sums = {name: 0.0 for name in TASK_HEAD_NAMES}
    task_head_brier_sums = {name: 0.0 for name in TASK_HEAD_NAMES}
    task_head_counts = {name: 0 for name in TASK_HEAD_NAMES}
    task_opportunities = 0
    utility_index = GLOBAL_HEAD_NAMES.index("global_utility_delta")
    utility_pair_count = 0
    utility_pair_correct = 0
    utility_overestimates: list[float] = []
    utility_overestimate_count = 0
    utility_valid_count = 0
    development_forward_batch_count = 0
    development_candidate_evaluation_count = 0
    for row in windows:
        sample = public_branch_batch(_wm_policy_input(row), row["branch_records"], window_id=tuple(row["window_ids"]))
        inputs = sample["inputs"]
        window_label = f"development_prediction:{row['window_ids'][2]}"
        with _track_wm_forward_calls(supervisor, window_label, model) as forward_call:
            with torch.no_grad():
                prediction = model(**{key: value for key, value in inputs.items() if key != "binding"}, binding=inputs["binding"])
        development_forward_batch_count += forward_call["forward_batches"]
        development_candidate_evaluation_count += forward_call["candidate_evaluations"]
        target = sample["target"]["global"]
        valid = sample["valid"]["global"]
        difference = torch.abs(prediction["global"] - target)
        abs_errors.extend(difference[valid].tolist())
        global_candidate_opportunities += int(inputs["candidate_mask"].sum().item())
        for head_index, head_name in enumerate(GLOBAL_HEAD_NAMES):
            head_valid = valid[..., head_index]
            head_errors = difference[..., head_index][head_valid]
            global_head_errors[head_name].extend(head_errors.tolist())
            global_head_counts[head_name] += int(head_valid.sum().item())

        target_task = sample["target"]["task"]
        valid_task = sample["valid"]["task"]
        task_probabilities = torch.sigmoid(prediction["per_task"])
        task_opportunities += int((inputs["candidate_mask"].unsqueeze(-1) & inputs["task_mask"].unsqueeze(1)).sum().item())
        for head_index, head_name in enumerate(TASK_HEAD_NAMES):
            head_valid = valid_task[..., head_index]
            if bool(head_valid.any()):
                logits = prediction["per_task"][..., head_index][head_valid]
                labels = target_task[..., head_index][head_valid]
                probabilities = task_probabilities[..., head_index][head_valid]
                task_head_bce_sums[head_name] += float(F.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item())
                task_head_brier_sums[head_name] += float(torch.square(probabilities - labels).sum().item())
                task_head_counts[head_name] += int(head_valid.sum().item())

        utility_valid = valid[0, :, utility_index]
        utility_predictions = prediction["global"][0, :, utility_index]
        utility_targets = target[0, :, utility_index]
        utility_valid_predictions = utility_predictions[utility_valid]
        utility_valid_targets = utility_targets[utility_valid]
        utility_valid_count += int(utility_valid.sum().item())
        utility_errors = utility_valid_predictions - utility_valid_targets
        positive_overestimates = utility_errors.clamp_min(0.0)
        utility_overestimates.extend(positive_overestimates.tolist())
        utility_overestimate_count += int((utility_errors > 0).sum().item())
        valid_indices = torch.nonzero(utility_valid, as_tuple=False).flatten().tolist()
        for left_position, left in enumerate(valid_indices):
            for right in valid_indices[left_position + 1:]:
                target_delta = float((utility_targets[left] - utility_targets[right]).item())
                if target_delta == 0.0:
                    continue
                prediction_delta = float((utility_predictions[left] - utility_predictions[right]).item())
                utility_pair_count += 1
                utility_pair_correct += int(prediction_delta != 0.0 and ((prediction_delta > 0.0) == (target_delta > 0.0)))

        with _track_wm_forward_calls(supervisor, window_label + ":prediction_hash", model) as hash_call:
            prediction_hash = _prediction_hash(model, _wm_policy_input(row))
        development_forward_batch_count += hash_call["forward_batches"]
        development_candidate_evaluation_count += hash_call["candidate_evaluations"]

        check_rows.append({
            "parent_id": row["window_ids"][0],
            "window_id": row["window_ids"][2],
            "candidate_count": int(inputs["candidate_mask"].sum().item()),
            "task_count": int(inputs["task_mask"].sum().item()),
            "valid_global_labels": int(valid.sum().item()),
            "prediction_sha256": prediction_hash,
        })
    global_head_metrics = {
        name: {
            "valid_label_count": global_head_counts[name],
            "candidate_coverage": global_head_counts[name] / global_candidate_opportunities if global_candidate_opportunities else None,
            "mae": float(np.mean(global_head_errors[name])) if global_head_errors[name] else None,
        }
        for name in GLOBAL_HEAD_NAMES
    }
    task_head_metrics = {
        name: {
            "valid_label_count": task_head_counts[name],
            "task_candidate_coverage": task_head_counts[name] / task_opportunities if task_opportunities else None,
            "bce": task_head_bce_sums[name] / task_head_counts[name] if task_head_counts[name] else None,
            "brier": task_head_brier_sums[name] / task_head_counts[name] if task_head_counts[name] else None,
        }
        for name in TASK_HEAD_NAMES
    }
    receipt = {
        "schema": SCHEMA,
        "stage": "checkpoint_restore",
        "status": "complete",
        "strict_restore_subprocess": strict,
        "strict_restore_prediction_forward_count": strict["strict_restore_prediction_forward_count"],
        "strict_restore_candidate_evaluation_count": strict["strict_restore_candidate_evaluation_count"],
        "development_window_count": len(windows),
        "prediction_count": len(check_rows),
        "development_prediction_forward_count": development_forward_batch_count,
        "development_candidate_evaluation_count": development_candidate_evaluation_count,
        "global_valid_label_count": sum(row["valid_global_labels"] for row in check_rows),
        "global_mae": float(np.mean(abs_errors)) if abs_errors else None,
        "global_head_metrics": global_head_metrics,
        "task_head_metrics": task_head_metrics,
        "global_utility_ranking": {
            "pair_count": utility_pair_count,
            "correct_pair_count": utility_pair_correct,
            "accuracy": utility_pair_correct / utility_pair_count if utility_pair_count else None,
            "definition": "within-window valid non-tied target pairs; predicted ties count incorrect",
        },
        "global_utility_overestimate": {
            "valid_label_count": utility_valid_count,
            "overestimate_count": utility_overestimate_count,
            "overestimate_rate": utility_overestimate_count / utility_valid_count if utility_valid_count else None,
            "mean_positive_overestimate": float(np.mean(utility_overestimates)) if utility_overestimates else None,
            "maximum_positive_overestimate": float(np.max(utility_overestimates)) if utility_overestimates else None,
        },
        "development_is_diagnostic_only": True,
        "windows": check_rows,
    }
    _tracked_json_write(supervisor, root / "checkpoint_restore_receipt.json", receipt)
    return receipt


def _proposal_masks(policy: GraphPolicy, policy_input: dict[str, Any], actions: list[str]) -> list[dict[str, Any]]:
    from graph_policy import encode_public_graph

    graph = encode_public_graph(policy_input)
    if not actions or "NOOP" in actions[:-1] or len(actions) != len(set(actions)):
        raise ValueError("N2_PROPOSAL_ACTION_BATCH")
    if actions[-1] != "NOOP":
        actions = [*actions, "NOOP"]
    indices = {candidate_id: index for index, candidate_id in enumerate(graph.candidate_ids)}
    used_platforms: set[int] = set()
    used_tasks: set[int] = set()
    stock_limits = {(depot, commodity): amount for depot, commodity, amount in graph.stock_limits}
    reserved_stock: dict[tuple[str, str], float] = {}
    masks = []
    for selected_id in actions:
        if selected_id not in indices:
            raise ValueError("N2_PROPOSAL_ACTION_OUTSIDE_CATALOG")
        valid = policy._mask_at_step(graph.candidate_ids, graph.candidate_pairs, graph.candidate_stock_requirements, stock_limits, used_platforms, used_tasks, reserved_stock)
        selected_index = indices[selected_id]
        if not valid[selected_index]:
            raise ValueError("N2_PROPOSAL_ACTION_MASKED")
        masks.append({"candidate_ids": list(graph.candidate_ids), "valid_mask": valid, "selected_id": selected_id})
        if selected_id == "NOOP":
            break
        pair = graph.candidate_pairs[selected_index]
        if pair is None:
            raise ValueError("N2_PROPOSAL_MISSING_PAIR")
        used_platforms.add(pair[0])
        used_tasks.add(pair[1])
        requirement = graph.candidate_stock_requirements[selected_index]
        if requirement is not None:
            depot, commodity, quantity = requirement
            key = (depot, commodity)
            reserved_stock[key] = reserved_stock.get(key, 0.0) + quantity
    if not masks or masks[-1]["selected_id"] != "NOOP":
        raise ValueError("N2_PROPOSAL_NOOP_TERMINATOR")
    return masks


def _policy_proposals(policy: GraphPolicy, env: ReliefEnv, policy_input: dict[str, Any]) -> list[dict[str, Any]]:
    batches, _ = _candidate_batches(env)
    proposals = []
    for actions in batches:
        complete_actions = list(actions)
        if complete_actions[-1] != "NOOP":
            complete_actions.append("NOOP")
        masks = _proposal_masks(policy, policy_input, complete_actions)
        proposals.append({
            "actions": complete_actions,
            "masks": masks,
            "candidate_id": key_digest(complete_actions),
            "is_noop": complete_actions == ["NOOP"],
        })
    return proposals


def _add_rollout_gae(records: list[dict], policy: CriticAugmentedGraphPolicy, episode_offsets: Sequence[tuple[int, int]], gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
    for start, end in episode_offsets:
        episode = records[start:end]
        if not episode or len(episode) != HORIZON_MINUTES:
            raise ValueError("N2_PPO_EPISODE_TICK_COUNT")
        values = [record["old_value"] for record in episode]
        rewards = [record["reward"] for record in episode]
        next_values = [*values[1:], 0.0]
        terminated = [False] * (len(episode) - 1) + [True]
        truncated = [False] * len(episode)
        rollout_cut = [False] * len(episode)
        advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, rollout_cut, gamma=gamma, gae_lambda=gae_lambda)
        for row, advantage, target in zip(episode, advantages, returns):
            row["advantage"] = float(advantage)
            row["return"] = float(target)


def _collect_policy_round(policy: CriticAugmentedGraphPolicy, scenarios: Sequence[dict], run_seed: int, round_number: int, supervisor: Any) -> tuple[list[dict], list[tuple[int, int]], list[dict]]:
    records: list[dict] = []
    offsets: list[tuple[int, int]] = []
    summaries = []
    episodes_per_round = PPO_BUDGET["episodes_per_round"]
    first = (round_number - 1) * episodes_per_round
    for episode_index in range(episodes_per_round):
        _check(supervisor)
        scenario = scenarios[first + episode_index]
        variant = "G-WM" if policy.wm_enabled else "G"
        episode_meta = {"scenario_id": scenario["scenario_id"], "round": round_number, "episode_index": episode_index, "variant": variant}
        episode_token = _reserve_work_call(supervisor, "training_episode", 1, episode_meta)
        try:
            env = ReliefEnv(scenario)
            env.reset(policy="LEARNED")
            start = len(records)
            terminated = False
            total_reward = 0.0
            episode_wm_forward_count = 0
            episode_wm_candidate_count = 0
            episode_cache_hit_count = 0
            episode_cache_miss_count = 0
            while not terminated:
                _check(supervisor)
                policy_input = copy.deepcopy(env.policy_input())
                policy_input["proposal_batches"] = _policy_proposals(policy.base_policy, env, policy_input)
                decision_index = len(records) - start + 1
                with _track_wm_forward_calls(supervisor, f"ppo_rollout:{variant}:{round_number}:{episode_index}:{decision_index}", policy.world_model) as forward_call:
                    with torch.no_grad():
                        sample = policy.sample_batch(policy_input)
                cache_usage = _wm_prediction_cache_usage(
                    sample.get("proposal_binding"), forward_call, enabled=policy.wm_enabled,
                )
                episode_wm_forward_count += forward_call["forward_batches"]
                episode_wm_candidate_count += forward_call["candidate_evaluations"]
                episode_cache_hit_count += cache_usage["hit_count"]
                episode_cache_miss_count += cache_usage["miss_count"]
                actions = [str(item) for item in sample["actions"]]
                _, reward, terminated, truncated, info = _step_environment(
                    env,
                    actions,
                    supervisor,
                    {**episode_meta, "decision_index": decision_index},
                )
                if truncated:
                    raise ValueError("N2_POLICY_ROLLOUT_TRUNCATED")
                total_reward += float(reward)
                records.append({
                    "scenario_id": scenario["scenario_id"],
                    "round": round_number,
                    "episode_index": episode_index,
                    "time": int(policy_input["observation"]["time"]),
                    "policy_input": policy_input,
                    "actions": actions,
                    "conditional_masks": copy.deepcopy(sample["masks"]),
                    "old_log_prob": float(sample["log_prob"].detach().item()),
                    "old_value": float(sample["value"].detach().item()),
                    "proposal_binding": copy.deepcopy(sample.get("proposal_binding")),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "transition_identity": info["transition"]["identity"],
                })
            offsets.append((start, len(records)))
            metrics = env.export_trajectory()["summary"]
        except BaseException as exc:
            _fail_work_call(supervisor, episode_token, exc)
            raise
        else:
            _complete_work_call(supervisor, episode_token, 1, episode_meta)
        summaries.append({
            "scenario_id": scenario["scenario_id"],
            "total_reward": total_reward,
            "environment_metrics": metrics,
            "transition_count": len(records) - start,
            "proposal_probability_forward_count": sum(int(item["proposal_binding"]["proposal_count"]) for item in records[start:]),
            "world_model_prediction_forward_count": episode_wm_forward_count,
            "world_model_candidate_evaluation_count": episode_wm_candidate_count,
            "world_model_prediction_cache_hit_count": episode_cache_hit_count,
            "world_model_prediction_cache_miss_count": episode_cache_miss_count,
        })
    _add_rollout_gae(records, policy, offsets)
    return records, offsets, summaries


def _ppo_round_update(policy: CriticAugmentedGraphPolicy, trainer: GppoEngineeringTrainer, records: list[dict], supervisor: Any) -> dict:
    policy.train()
    sample_count = len(records)
    if sample_count != PPO_BUDGET["episodes_per_round"] * HORIZON_MINUTES:
        raise ValueError("N2_PPO_ROLLOUT_COUNT")
    advantages = torch.as_tensor([item["advantage"] for item in records], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.as_tensor([item["return"] for item in records], dtype=torch.float32)
    old_log_probs = torch.as_tensor([item["old_log_prob"] for item in records], dtype=torch.float32)
    paired_minibatch_seed = 8201 + trainer.update_index * 13
    generator = torch.Generator(device="cpu").manual_seed(paired_minibatch_seed)
    actor_heads_before = _tensor_digest(policy.base_policy, ("pair_actor", "noop_actor"))
    actor_losses: list[float] = []
    value_losses: list[float] = []
    entropy_values: list[float] = []
    grad_norms: list[float] = []
    actor_head_gradient_norms: list[float] = []
    adapter_gradient_norms: list[float] = []
    adapter_update_receipts: list[dict[str, Any]] = []
    proposal_update_receipts: list[dict[str, Any]] = []
    proposal_masses: list[float] = []
    initial_ratio_error: float | None = None
    before_model = policy._state_digest(policy.world_model.state_dict())
    before_actor = policy._state_digest(policy.base_policy.state_dict())
    optimizer_updates = 0
    proposal_probability_forward_count = 0
    world_model_prediction_forward_count = 0
    world_model_candidate_evaluation_count = 0
    world_model_prediction_cache_hit_count = 0
    world_model_prediction_cache_miss_count = 0
    cache_lookup_count = 0
    for _epoch in range(PPO_EPOCHS):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, PPO_MINIBATCH_SIZE):
            _check(supervisor)
            indices = permutation[start:start + PPO_MINIBATCH_SIZE].tolist()
            outputs = []
            for index in indices:
                sample = records[index]
                wm_counter_label = f"ppo_update:{policy.variant}:{trainer.update_index + 1}:{index}"
                with _track_wm_forward_calls(supervisor, wm_counter_label, policy.world_model) as forward_call:
                    output = policy.evaluate_batch(sample["policy_input"], sample["actions"], sample["conditional_masks"])
                cache_usage = _wm_prediction_cache_usage(
                    output.get("proposal_binding"), forward_call, enabled=policy.wm_enabled,
                )
                outputs.append(output)
                world_model_prediction_forward_count += forward_call["forward_batches"]
                world_model_candidate_evaluation_count += forward_call["candidate_evaluations"]
                world_model_prediction_cache_hit_count += cache_usage["hit_count"]
                world_model_prediction_cache_miss_count += cache_usage["miss_count"]
                cache_lookup_count += int(policy.wm_enabled)
            update_bindings = []
            for sample_index, output in zip(indices, outputs):
                binding = output.get("proposal_binding")
                if not isinstance(binding, dict) or binding.get("proposal_probability_binding") != "complete_action_batch_sequential_masks":
                    raise ValueError("N2_PPO_PROPOSAL_BINDING_MISSING")
                mass = binding.get("proposal_set_probability_mass")
                candidate_ids = binding.get("candidate_ids")
                if type(mass) not in (int, float) or not 0.0 <= float(mass) <= 1.0 + 1e-5 or not isinstance(candidate_ids, list) or not candidate_ids:
                    raise ValueError("N2_PPO_PROPOSAL_MASS_RECORD")
                proposal_masses.append(float(mass))
                proposal_probability_forward_count += int(binding["proposal_count"])
                update_bindings.append({
                    "scenario_id": records[sample_index]["scenario_id"],
                    "episode_index": records[sample_index]["episode_index"],
                    "time": records[sample_index]["time"],
                    "proposal_binding": copy.deepcopy(binding),
                })
            new_log_probs = torch.stack([item["log_prob"] for item in outputs])
            predicted_values = torch.stack([item["value"] for item in outputs])
            entropy = torch.stack([item["entropy"] for item in outputs]).mean()
            batch = torch.as_tensor(indices, dtype=torch.long)
            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            if initial_ratio_error is None:
                initial_ratio_error = float(torch.max(torch.abs(ratio.detach() - 1.0)).item())
                if initial_ratio_error > 2e-5:
                    raise AssertionError("N2_PPO_PREUPDATE_RATIO")
            adv = advantages[batch].to(new_log_probs.device)
            target = returns[batch].to(predicted_values.device)
            unclipped = ratio * adv
            clipped = ratio.clamp(0.8, 1.2) * adv
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(predicted_values, target)
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError("N2_PPO_NONFINITE_LOSS")
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adapter_before = policy._state_digest(policy.adapter.state_dict()) if policy.wm_enabled else None
            adapter_gradients = [
                parameter.grad.detach()
                for parameter in policy.adapter.parameters()
                if parameter.grad is not None
            ] if policy.wm_enabled else []
            adapter_gradient_norm = (
                float(torch.sqrt(sum(torch.sum(value * value) for value in adapter_gradients)).item())
                if adapter_gradients else 0.0
            )
            if policy.wm_enabled and (
                not adapter_gradients
                or not all(bool(torch.isfinite(value).all()) for value in adapter_gradients)
                or not bool(np.isfinite(adapter_gradient_norm))
            ):
                raise ValueError("N2_GWM_ADAPTER_NONFINITE_OR_MISSING_GRADIENT")
            if policy.wm_enabled:
                adapter_gradient_norms.append(adapter_gradient_norm)
            actor_head_gradients = [
                parameter.grad.detach()
                for name, parameter in policy.base_policy.named_parameters()
                if name.split(".", 1)[0] in {"pair_actor", "noop_actor"} and parameter.grad is not None
            ]
            if actor_head_gradients:
                actor_head_gradient_norms.append(float(torch.sqrt(sum(torch.sum(value * value) for value in actor_head_gradients)).item()))
            grad_norm = torch.nn.utils.clip_grad_norm_([item for item in policy.parameters() if item.requires_grad], 0.5)
            if not bool(torch.isfinite(grad_norm)):
                raise ValueError("N2_PPO_NONFINITE_GRADIENT")
            _run_reserved_work_call(
                supervisor,
                "ppo_optimizer_update",
                1,
                {"variant": policy.variant, "optimizer_update_index": trainer.update_index + optimizer_updates + 1, "sample_count": len(indices)},
                trainer.optimizer.step,
            )
            optimizer_updates += 1
            if policy.wm_enabled:
                adapter_after = policy._state_digest(policy.adapter.state_dict())
                adapter_update_receipts.append({
                    "optimizer_update_index": optimizer_updates,
                    "gradient_norm": adapter_gradient_norm,
                    "state_sha256_before": adapter_before,
                    "state_sha256_after": adapter_after,
                    "parameters_changed": adapter_before != adapter_after,
                })
            proposal_update_receipts.append({
                "optimizer_update_index": optimizer_updates,
                "sample_indices": indices,
                "sample_index_sha256": _sha256_bytes(_canonical(indices)),
                "samples": update_bindings,
            })
            actor_losses.append(float(actor_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            entropy_values.append(float(entropy.detach().item()))
            grad_norms.append(float(grad_norm.detach().item()))
    if optimizer_updates != PPO_BUDGET["updates_per_round"]:
        raise AssertionError("N2_PPO_UPDATE_BUDGET")
    if policy.wm_enabled and (
        len(adapter_update_receipts) != optimizer_updates
        or any(item["gradient_norm"] <= 0.0 or not item["parameters_changed"] for item in adapter_update_receipts)
    ):
        raise AssertionError("N2_GWM_ADAPTER_UPDATE_WITNESS")
    expected_cache_lookups = optimizer_updates * PPO_MINIBATCH_SIZE if policy.wm_enabled else 0
    if cache_lookup_count != expected_cache_lookups:
        raise AssertionError("N2_PPO_CACHE_CALL_COUNT")
    if policy.wm_enabled and (
        world_model_prediction_cache_hit_count != expected_cache_lookups
        or world_model_prediction_cache_miss_count != 0
        or world_model_prediction_forward_count != 0
        or world_model_candidate_evaluation_count != 0
    ):
        raise AssertionError("N2_PPO_WM_CACHE_REUSE")
    if not policy.wm_enabled and (
        world_model_prediction_cache_hit_count != 0
        or world_model_prediction_cache_miss_count != 0
        or world_model_prediction_forward_count != 0
        or world_model_candidate_evaluation_count != 0
    ):
        raise AssertionError("N2_BASELINE_WM_CACHE_REUSE")
    trainer.update_index += optimizer_updates
    trainer.next_batch_index = trainer.update_index
    after_model = policy._state_digest(policy.world_model.state_dict())
    after_actor = policy._state_digest(policy.base_policy.state_dict())
    actor_heads_after = _tensor_digest(policy.base_policy, ("pair_actor", "noop_actor"))
    if before_model != after_model or before_actor == after_actor:
        raise AssertionError("N2_PPO_FROZEN_MODEL_OR_ACTOR_UPDATE")
    return {
        "optimizer_updates": optimizer_updates,
        "rollout_transition_count": sample_count,
        "ppo_sample_evaluation_count": optimizer_updates * PPO_MINIBATCH_SIZE,
        "proposal_probability_forward_count": proposal_probability_forward_count,
        "rollout_proposal_probability_forward_count": sum(int(item["proposal_binding"]["proposal_count"]) for item in records),
        "world_model_prediction_forward_count": world_model_prediction_forward_count,
        "world_model_candidate_evaluation_count": world_model_candidate_evaluation_count,
        "world_model_prediction_cache_lookup_count": cache_lookup_count,
        "world_model_prediction_cache_hit_count": world_model_prediction_cache_hit_count,
        "world_model_prediction_cache_miss_count": world_model_prediction_cache_miss_count,
        "actor_loss": float(np.mean(actor_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_values)),
        "gradient_norm_preclip_mean": float(np.mean(grad_norms)),
        "gradient_norm_preclip_max": float(np.max(grad_norms)),
        "max_initial_probability_ratio_error": float(initial_ratio_error or 0.0),
        "proposal_binding_contract": "complete_action_batch_sequential_masks",
        "proposal_set_probability_mass": {
            "sample_count": len(proposal_masses),
            "mean": float(np.mean(proposal_masses)),
            "minimum": float(np.min(proposal_masses)),
            "maximum": float(np.max(proposal_masses)),
        },
        "proposal_update_receipts": proposal_update_receipts,
        "world_model_state_sha256": after_model,
        "actor_state_sha256": after_actor,
        "actor_head_state_sha256_before": actor_heads_before,
        "actor_head_state_sha256_after": actor_heads_after,
        "actor_head_parameters_changed": actor_heads_before != actor_heads_after,
        "actor_head_gradient_norm_max": max(actor_head_gradient_norms, default=0.0),
        "adapter_gradient_norm_min": min(adapter_gradient_norms, default=0.0),
        "adapter_gradient_norm_max": max(adapter_gradient_norms, default=0.0),
        "adapter_update_receipts": adapter_update_receipts,
        "paired_minibatch_seed": paired_minibatch_seed,
        "world_model_frozen": True,
    }


def run_graph_policy_training(package_root: Path, work_root: Path, supervisor: Any = None) -> dict:
    del package_root
    _require_work_ledger(supervisor)
    root = work_root / "n2_pipeline"
    training_receipt = _json_read(root / "world_model_training_receipt.json")
    restore_receipt = _json_read(root / "checkpoint_restore_receipt.json")
    if training_receipt.get("status") != "complete" or restore_receipt.get("status") != "complete":
        raise PermissionError("N2_WM_CHECKPOINT_RESTORE_REQUIRED")
    scenarios = _json_read(root / "parents.json")["train"]
    if len(scenarios) != COLLECTION_BUDGET["train_parents"]:
        raise ValueError("N2_POLICY_TRAIN_PARENT_COUNT")
    world_model, _ = _load_world_model_checkpoint(
        supervisor,
        work_root / training_receipt["checkpoint_path"],
        data_identity=training_receipt["data_identity"],
        component="graph_policy_training",
    )
    base_g = build_policy("L1", WM_SEED, device="cpu")
    base_gwm = build_policy("L1", WM_SEED, device="cpu")
    policy_g = CriticAugmentedGraphPolicy(base_g, world_model, CriticOnlyGppoAdapter(), wm_enabled=False)
    policy_gwm = CriticAugmentedGraphPolicy(base_gwm, world_model, CriticOnlyGppoAdapter(), wm_enabled=True)
    first_env = ReliefEnv(scenarios[0])
    first_env.reset(policy="LEARNED")
    probe_input = first_env.policy_input()
    probe_input["proposal_batches"] = _policy_proposals(base_g, first_env, probe_input)
    assert_paired_actor_initialization(policy_g, policy_gwm, probe_input)
    paired_actor_initialization_sha256 = _tensor_digest(base_g, ("pair_actor", "noop_actor"))
    if paired_actor_initialization_sha256 != _tensor_digest(base_gwm, ("pair_actor", "noop_actor")):
        raise ValueError("N2_G_GWM_ACTOR_HEAD_INITIALIZATION_MISMATCH")
    trainers = {
        "G": GppoEngineeringTrainer(policy_g, data_identity=training_receipt["data_identity"]),
        "G-WM": GppoEngineeringTrainer(policy_gwm, data_identity=training_receipt["data_identity"]),
    }
    policies = {"G": policy_g, "G-WM": policy_gwm}
    rows = []
    for round_number in range(1, PPO_BUDGET["rounds_per_variant"] + 1):
        for variant in PPO_BUDGET["variants"]:
            torch.manual_seed(WM_SEED + round_number * 101)
            records, _, episodes = _collect_policy_round(policies[variant], scenarios, WM_SEED, round_number, supervisor)
            update = _ppo_round_update(policies[variant], trainers[variant], records, supervisor)
            rollout_pairing_sha256 = _sha256_bytes(_canonical([
                {
                    "scenario_id": row["scenario_id"],
                    "episode_index": row["episode_index"],
                    "time": row["time"],
                    "actions": row["actions"],
                    "conditional_masks": row["conditional_masks"],
                }
                for row in records
            ]))
            rollout_path = root / "rollouts" / variant.replace("-", "_") / f"round_{round_number:02d}.jsonl"
            _tracked_jsonl_write(supervisor, rollout_path, records)
            state = {name: tensor.detach().cpu() for name, tensor in policies[variant].state_dict().items()}
            checkpoint_path = root / "checkpoints" / variant.replace("-", "_") / f"round_{round_number:02d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _tracked_torch_save(
                supervisor,
                checkpoint_path,
                {"schema": "n2-gppo-smoke-checkpoint/1", "variant": variant, "round": round_number, "data_identity": training_receipt["data_identity"], "policy_state": state, "optimizer_state": trainers[variant].optimizer.state_dict(), "updates": trainers[variant].update_index},
                {"variant": variant, "round": round_number},
            )
            rows.append({
                "variant": variant,
                "round": round_number,
                "episode_count": len(episodes),
                "transition_count": len(records),
                "episodes": episodes,
                "rollout_path": rollout_path.relative_to(work_root).as_posix(),
                "rollout_sha256": _sha256_file(rollout_path),
                "rollout_pairing_sha256": rollout_pairing_sha256,
                "checkpoint_path": checkpoint_path.relative_to(work_root).as_posix(),
                "checkpoint_sha256": _tracked_checkpoint_sha256(supervisor, checkpoint_path, f"gppo_checkpoint_verification:{variant}:{round_number}"),
                **update,
            })
    training_episode_count = sum(len(row["episodes"]) for row in rows)
    updates_per_variant = {
        variant: sum(row["optimizer_updates"] for row in rows if row["variant"] == variant)
        for variant in PPO_BUDGET["variants"]
    }
    expected_training_episodes = (
        PPO_BUDGET["rounds_per_variant"]
        * PPO_BUDGET["episodes_per_round"]
        * len(PPO_BUDGET["variants"])
    )
    if (
        training_episode_count != expected_training_episodes
        or sum(updates_per_variant.values()) != PPO_BUDGET["total_updates"]
        or any(
            count != PPO_BUDGET["rounds_per_variant"] * PPO_BUDGET["updates_per_round"]
            for count in updates_per_variant.values()
        )
    ):
        raise AssertionError("N2_TOTAL_PPO_UPDATE_BUDGET")
    rows_by_variant = {row["variant"]: row for row in rows}
    g_row, gwm_row = rows_by_variant["G"], rows_by_variant["G-WM"]
    g_minibatch_ids = [item["sample_index_sha256"] for item in g_row["proposal_update_receipts"]]
    gwm_minibatch_ids = [item["sample_index_sha256"] for item in gwm_row["proposal_update_receipts"]]
    if (
        g_row["rollout_pairing_sha256"] != gwm_row["rollout_pairing_sha256"]
        or g_minibatch_ids != gwm_minibatch_ids
        or g_row["paired_minibatch_seed"] != gwm_row["paired_minibatch_seed"]
    ):
        raise AssertionError("N2_G_GWM_PAIRED_ROLLOUT_OR_MINIBATCH_MISMATCH")
    cache_stats = policy_gwm.prediction_cache.stats()
    rollout_forward_count = sum(episode["world_model_prediction_forward_count"] for row in rows for episode in row["episodes"])
    rollout_candidate_count = sum(episode["world_model_candidate_evaluation_count"] for row in rows for episode in row["episodes"])
    expected_ppo_cache_hits = PPO_BUDGET["rounds_per_variant"] * PPO_BUDGET["updates_per_round"] * PPO_MINIBATCH_SIZE
    if (
        cache_stats["misses"] != rollout_forward_count
        or cache_stats["forward_batches"] != rollout_forward_count
        or cache_stats["candidate_evaluations"] != rollout_candidate_count
        or cache_stats["hits"] < expected_ppo_cache_hits
        or cache_stats["entry_count"] > cache_stats["max_entries"]
    ):
        raise AssertionError("N2_GWM_CACHE_STAT_RECONCILIATION")
    receipt = {
        "schema": SCHEMA,
        "stage": "graph_policy_training",
        "status": "complete",
        "variants": list(PPO_BUDGET["variants"]),
        "fresh_actor_initialization": True,
        "actor_pairing": "same L1 literal graph initialization seed 8201 for G and G-WM",
        "actor_head_initialization_sha256": paired_actor_initialization_sha256,
        "paired_rollout_sha256": g_row["rollout_pairing_sha256"],
        "paired_minibatch_index_sha256": g_minibatch_ids,
        "graph_policy_variant": "L1",
        "rounds_per_variant": PPO_BUDGET["rounds_per_variant"],
        "episodes_per_round": PPO_BUDGET["episodes_per_round"],
        "optimizer_updates": PPO_BUDGET["total_updates"],
        "updates_per_variant": updates_per_variant,
        "training_episode_count": training_episode_count,
        "rows": rows,
        "rollout_sample_count": sum(row["transition_count"] for row in rows),
        "ppo_optimizer_update_count": sum(row["optimizer_updates"] for row in rows),
        "ppo_sample_evaluation_count": sum(row["ppo_sample_evaluation_count"] for row in rows),
        "proposal_probability_forward_count": sum(row["proposal_probability_forward_count"] for row in rows),
        "rollout_proposal_probability_forward_count": sum(row["rollout_proposal_probability_forward_count"] for row in rows),
        "rollout_world_model_prediction_forward_count": sum(episode["world_model_prediction_forward_count"] for row in rows for episode in row["episodes"]),
        "ppo_world_model_prediction_forward_count": sum(row["world_model_prediction_forward_count"] for row in rows),
        "rollout_world_model_prediction_cache_hit_count": sum(episode["world_model_prediction_cache_hit_count"] for row in rows for episode in row["episodes"]),
        "rollout_world_model_prediction_cache_miss_count": sum(episode["world_model_prediction_cache_miss_count"] for row in rows for episode in row["episodes"]),
        "ppo_world_model_prediction_cache_lookup_count": sum(row["world_model_prediction_cache_lookup_count"] for row in rows),
        "ppo_world_model_prediction_cache_hit_count": sum(row["world_model_prediction_cache_hit_count"] for row in rows),
        "ppo_world_model_prediction_cache_miss_count": sum(row["world_model_prediction_cache_miss_count"] for row in rows),
        "world_model_prediction_cache_stats": cache_stats,
        "world_model_prediction_forward_count": sum(
            episode["world_model_prediction_forward_count"] for row in rows for episode in row["episodes"]
        ) + sum(row["world_model_prediction_forward_count"] for row in rows),
        "world_model_candidate_evaluation_count": sum(
            episode["world_model_candidate_evaluation_count"] for row in rows for episode in row["episodes"]
        ) + sum(row["world_model_candidate_evaluation_count"] for row in rows),
    }
    _tracked_json_write(supervisor, root / "graph_policy_training_receipt.json", receipt)
    return receipt


def _load_trained_policies(work_root: Path, supervisor: Any) -> tuple[dict[str, CriticAugmentedGraphPolicy], DynamicConsequenceModel, dict]:
    _require_work_ledger(supervisor)
    root = work_root / "n2_pipeline"
    wm_receipt = _json_read(root / "world_model_training_receipt.json")
    training_policy_receipt = _json_read(root / "graph_policy_training_receipt.json")
    if training_policy_receipt.get("status") != "complete":
        raise PermissionError("N2_POLICY_TRAINING_RECEIPT_REQUIRED")
    model, _ = _load_world_model_checkpoint(
        supervisor,
        work_root / wm_receipt["checkpoint_path"],
        data_identity=wm_receipt["data_identity"],
        component="development_evaluation",
    )
    policies = {}
    final_round = PPO_BUDGET["rounds_per_variant"]
    for variant in PPO_BUDGET["variants"]:
        entry = [row for row in training_policy_receipt["rows"] if row["variant"] == variant and row["round"] == final_round]
        if len(entry) != 1:
            raise ValueError("N2_POLICY_FINAL_CHECKPOINT_ENTRY")
        payload = _load_policy_checkpoint(
            supervisor,
            work_root / entry[0]["checkpoint_path"],
            component=f"development_evaluation:{variant}",
        )
        base = build_policy("L1", WM_SEED, device="cpu")
        wrapper = CriticAugmentedGraphPolicy(base, model, CriticOnlyGppoAdapter(), wm_enabled=variant == "G-WM")
        wrapper.load_state_dict(payload["policy_state"], strict=True)
        wrapper.eval()
        policies[variant] = wrapper
    return policies, model, training_policy_receipt


def run_development_evaluation(package_root: Path, work_root: Path, supervisor: Any = None) -> dict:
    del package_root
    _require_work_ledger(supervisor)
    root = work_root / "n2_pipeline"
    policies, _, policy_training = _load_trained_policies(work_root, supervisor)
    scenarios = _json_read(root / "parents.json")["development"]
    if len(scenarios) != COLLECTION_BUDGET["development_parents"]:
        raise ValueError("N2_DEVELOPMENT_PARENT_COUNT")
    rows = []
    for index, scenario in enumerate(scenarios):
        parent_rows = {}
        for variant in PPO_BUDGET["variants"]:
            _check(supervisor)
            episode_meta = {"scenario_id": scenario["scenario_id"], "parent_index": index, "variant": variant}
            episode_token = _reserve_work_call(supervisor, "development_episode", 1, episode_meta)
            try:
                env = ReliefEnv(scenario)
                env.reset(policy="LEARNED")
                terminated = False
                episode_steps = 0
                proposal_probability_forward_count = 0
                world_model_prediction_forward_count = 0
                world_model_candidate_evaluation_count = 0
                world_model_prediction_cache_hit_count = 0
                world_model_prediction_cache_miss_count = 0
                proposal_diagnostics: list[dict[str, Any]] = []
                while not terminated:
                    policy_input = copy.deepcopy(env.policy_input())
                    policy_input["proposal_batches"] = _policy_proposals(policies[variant].base_policy, env, policy_input)
                    proposal_probability_forward_count += len(policy_input["proposal_batches"])
                    with _track_wm_forward_calls(supervisor, f"development:{variant}:{scenario['scenario_id']}:{episode_steps + 1}", policies[variant].world_model) as forward_call:
                        with torch.no_grad():
                            sample = policies[variant].sample_batch(policy_input, deterministic=True)
                    binding = sample.get("proposal_binding")
                    cache_usage = _wm_prediction_cache_usage(
                        binding, forward_call, enabled=policies[variant].wm_enabled,
                    )
                    world_model_prediction_forward_count += cache_usage["forward_batches"]
                    world_model_candidate_evaluation_count += cache_usage["candidate_evaluations"]
                    world_model_prediction_cache_hit_count += cache_usage["hit_count"]
                    world_model_prediction_cache_miss_count += cache_usage["miss_count"]
                    proposal_diagnostics.append(copy.deepcopy(binding))
                    _, _, terminated, truncated, _ = _step_environment(
                        env,
                        sample["actions"],
                        supervisor,
                        {**episode_meta, "decision_index": episode_steps + 1},
                    )
                    episode_steps += 1
                    if truncated:
                        raise ValueError("N2_DEVELOPMENT_TRUNCATED")
                summary = env.export_trajectory()["summary"]
            except BaseException as exc:
                _fail_work_call(supervisor, episode_token, exc)
                raise
            else:
                _complete_work_call(supervisor, episode_token, 1, {**episode_meta, "environment_step_count": episode_steps})
            parent_rows[variant] = summary
            rows.append({"scenario_id": scenario["scenario_id"], "parent_index": index, "variant": variant, "summary": summary, "environment_step_count": episode_steps, "policy_decision_count": episode_steps, "proposal_probability_forward_count": proposal_probability_forward_count, "world_model_prediction_forward_count": world_model_prediction_forward_count, "world_model_candidate_evaluation_count": world_model_candidate_evaluation_count, "world_model_prediction_cache_hit_count": world_model_prediction_cache_hit_count, "world_model_prediction_cache_miss_count": world_model_prediction_cache_miss_count, "proposal_diagnostics": proposal_diagnostics})
    paired = []
    for scenario in scenarios:
        a = next(row["summary"] for row in rows if row["scenario_id"] == scenario["scenario_id"] and row["variant"] == "G")
        b = next(row["summary"] for row in rows if row["scenario_id"] == scenario["scenario_id"] and row["variant"] == "G-WM")
        paired.append({
            "scenario_id": scenario["scenario_id"],
            "G_minus_G-WM_weighted_on_time_delivery_rate": None if a["weighted_on_time_delivery_rate"] is None or b["weighted_on_time_delivery_rate"] is None else a["weighted_on_time_delivery_rate"] - b["weighted_on_time_delivery_rate"],
            "G_minus_G-WM_inspection_on_time_rate": None if a["inspection_on_time_rate"] is None or b["inspection_on_time_rate"] is None else a["inspection_on_time_rate"] - b["inspection_on_time_rate"],
            "G_minus_G-WM_energy_used": a["energy_used"] - b["energy_used"],
        })
    training_receipt = _json_read(root / "world_model_training_receipt.json")
    restore_receipt = _json_read(root / "checkpoint_restore_receipt.json")
    graph_receipt = _json_read(root / "graph_policy_training_receipt.json")
    if any(item.get("status") != "complete" for item in (training_receipt, restore_receipt, graph_receipt)):
        raise PermissionError("N2_WORLD_MODEL_STAGE_RECEIPTS_REQUIRED")
    development_cache = policies["G-WM"].prediction_cache
    if development_cache is None:
        raise AssertionError("N2_GWM_DEVELOPMENT_CACHE_MISSING")
    development_cache_stats = development_cache.stats()
    development_wm_forward_count = sum(row["world_model_prediction_forward_count"] for row in rows)
    development_wm_candidate_count = sum(row["world_model_candidate_evaluation_count"] for row in rows)
    if (
        development_cache_stats["misses"] != development_wm_forward_count
        or development_cache_stats["forward_batches"] != development_wm_forward_count
        or development_cache_stats["candidate_evaluations"] != development_wm_candidate_count
        or development_cache_stats["hits"] != sum(row["world_model_prediction_cache_hit_count"] for row in rows)
        or development_cache_stats["misses"] != sum(row["world_model_prediction_cache_miss_count"] for row in rows)
    ):
        raise AssertionError("N2_GWM_DEVELOPMENT_CACHE_RECONCILIATION")
    work_upper_bound = _world_model_work_upper_bound()
    actual_wm_forward_count = (
        int(training_receipt["training_prediction_forward_count"])
        + int(restore_receipt["strict_restore_prediction_forward_count"])
        + int(restore_receipt["development_prediction_forward_count"])
        + int(graph_receipt["world_model_prediction_forward_count"])
        + development_wm_forward_count
    )
    actual_wm_candidate_count = (
        int(training_receipt["training_candidate_evaluation_count"])
        + int(restore_receipt["strict_restore_candidate_evaluation_count"])
        + int(restore_receipt["development_candidate_evaluation_count"])
        + int(graph_receipt["world_model_candidate_evaluation_count"])
        + development_wm_candidate_count
    )
    caps = {
        "forward_batch_count": WORK_KIND_LIMITS["wm_forward_batch"],
        "candidate_evaluation_count": WORK_KIND_LIMITS["wm_candidate_evaluation"],
    }
    if (
        actual_wm_forward_count > work_upper_bound["forward_batch_count"]
        or actual_wm_candidate_count > work_upper_bound["candidate_evaluation_count"]
        or actual_wm_forward_count > caps["forward_batch_count"]
        or actual_wm_candidate_count > caps["candidate_evaluation_count"]
    ):
        raise AssertionError("N2_WORLD_MODEL_CUMULATIVE_BUDGET")
    expected_ppo_cache_lookups = work_upper_bound["ppo_cache_lookup_count"]
    if (
        graph_receipt.get("ppo_world_model_prediction_cache_lookup_count") != expected_ppo_cache_lookups
        or graph_receipt.get("ppo_world_model_prediction_cache_hit_count") != expected_ppo_cache_lookups
        or graph_receipt.get("ppo_world_model_prediction_cache_miss_count") != 0
        or graph_receipt.get("ppo_world_model_prediction_forward_count") != 0
    ):
        raise AssertionError("N2_WORLD_MODEL_PPO_CACHE_BUDGET")
    world_model_work = {
        "actual": {
            "forward_batch_count": actual_wm_forward_count,
            "candidate_evaluation_count": actual_wm_candidate_count,
        },
        "static_upper_bound": work_upper_bound,
        "caps": caps,
        "status": "within_bounds",
    }
    receipt = {
        "schema": SCHEMA,
        "stage": "development_evaluation",
        "status": "complete",
        "development_parent_count": COLLECTION_BUDGET["development_parents"],
        "expected_evaluation_episode_count": PPO_BUDGET["development_episodes"],
        "evaluation_episode_count": len(rows),
        "evaluation_environment_step_count": sum(row["environment_step_count"] for row in rows),
        "evaluation_policy_decision_count": sum(row["policy_decision_count"] for row in rows),
        "proposal_probability_forward_count": sum(row["proposal_probability_forward_count"] for row in rows),
        "world_model_prediction_forward_count": sum(row["world_model_prediction_forward_count"] for row in rows),
        "world_model_candidate_evaluation_count": sum(row["world_model_candidate_evaluation_count"] for row in rows),
        "world_model_prediction_cache_hit_count": sum(row["world_model_prediction_cache_hit_count"] for row in rows),
        "world_model_prediction_cache_miss_count": sum(row["world_model_prediction_cache_miss_count"] for row in rows),
        "world_model_prediction_cache_stats": development_cache_stats,
        "world_model_work": world_model_work,
        "matched_variants": list(PPO_BUDGET["variants"]),
        "paired_differences": paired,
        "episodes": rows,
        "policy_training_receipt_sha256": _sha256_bytes(_canonical(policy_training)),
        "development_is_diagnostic_only": True,
    }
    if len(rows) != PPO_BUDGET["development_episodes"]:
        raise AssertionError("N2_DEVELOPMENT_EPISODE_BUDGET")
    _tracked_json_write(supervisor, root / "development_evaluation_receipt.json", receipt)
    return receipt


STAGE_ACTIONS: dict[str, Callable[[Path, Path, Any], dict]] = {
    "paired_consequence_collection": run_paired_consequence_collection,
    "world_model_training": run_world_model_training,
    "checkpoint_restore": run_checkpoint_restore,
    "graph_policy_training": run_graph_policy_training,
    "development_evaluation": run_development_evaluation,
}


def run_stage(stage: str, package_root: str | Path, work_root: str | Path, supervisor: Any = None) -> dict:
    """Run one computational stage; admission and settlement remain driver-owned."""
    try:
        action = STAGE_ACTIONS[stage]
    except KeyError as exc:
        raise ValueError(f"N2_PIPELINE_STAGE_UNSUPPORTED:{stage}") from exc
    return action(require_package_root(package_root), Path(work_root).resolve(), supervisor)
