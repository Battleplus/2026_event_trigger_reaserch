"""Inference-only candidate selection for a frozen PPO actor-critic."""
from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .candidate_features import FEATURE_NAMES as CANDIDATE_FEATURE_NAMES
from .candidate_features import build_candidate_features
from .contracts import PublicHistoryFrame
from .features import FEATURE_NAMES as NODE_FEATURE_NAMES
from .features import collate_graph_sequences


MAX_POLICY_HISTORY = 8


class PolicyUnavailable(RuntimeError):
    """A known deployment prerequisite is unavailable for this decision."""


def _normalization_vector(normalization: Mapping[str, Any], name: str, width: int) -> np.ndarray:
    value = np.asarray(normalization.get(name), dtype=np.float32)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {width} finite train-only values")
    if name.endswith("scale") and (value <= 0).any():
        raise ValueError(f"{name} must contain positive train-only scales")
    return value


def _validate_history(history: Sequence[PublicHistoryFrame], projection: Mapping[str, Any]) -> None:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("public history must be a sequence of PublicHistoryFrame values")
    if len(history) < 2:
        raise PolicyUnavailable("POLICY_HISTORY_UNAVAILABLE: at least two contiguous frames are required")
    if len(history) > MAX_POLICY_HISTORY:
        raise ValueError(f"policy history exceeds {MAX_POLICY_HISTORY} frames")
    if any(not isinstance(frame, PublicHistoryFrame) for frame in history):
        raise ValueError("policy history contains a non-public-history frame")
    times = [float(frame.time_s) for frame in history]
    if any(not math.isfinite(t) for t in times) or any(b - a != 1 for a, b in zip(times, times[1:])):
        raise PolicyUnavailable("POLICY_HISTORY_UNAVAILABLE: history must contain contiguous 1-second frames")
    try:
        current_time = float(projection["sim_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("public projection requires current sim_time_s") from exc
    if not math.isfinite(current_time) or current_time != times[-1]:
        raise PolicyUnavailable("POLICY_HISTORY_UNAVAILABLE: final history frame must match current projection")


def _tensor_graph_batch(batch: Any, torch: Any, device: Any,
                        node_center: np.ndarray, node_scale: np.ndarray) -> Any:
    values = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, np.ndarray):
            if field.name.endswith("mask"):
                dtype = torch.bool
            elif field.name in {"edge_index", "edge_type"}:
                dtype = torch.long
            else:
                dtype = torch.float32
            value = torch.as_tensor(value, dtype=dtype, device=device)
        values[field.name] = value
    values["node_features"] = (values["node_features"] - torch.as_tensor(node_center, device=device)) / torch.as_tensor(node_scale, device=device)
    return type(batch)(**values)


def select_public_candidate(
    model: Any,
    projection: Mapping[str, Any],
    history: Sequence[PublicHistoryFrame],
    catalog: Sequence[Mapping[str, Any]],
    *,
    normalization: Mapping[str, Any],
    model_version: str,
    mode: str,
    device: str,
    synthetic_cpu_fixture: bool = False,
    generator: Any = None,
) -> dict[str, Any]:
    """Run masked stochastic PPO sampling or deterministic evaluation argmax.

    ``normalization`` must explicitly declare ``scope=train_split_only`` and
    carry node/candidate centers and positive scales fitted on that split.
    Real inference is pinned to the project's GPU1 UUID. CPU is only available
    with an explicitly named synthetic fixture and synthetic model version.
    """
    wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    if mode not in {"sample", "evaluate"}:
        raise ValueError("mode must be 'sample' or 'evaluate'")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be explicit")
    if not isinstance(normalization, Mapping) or normalization.get("scope") != "train_split_only":
        raise ValueError("normalization must be explicitly bound to train_split_only")
    if list(normalization.get("node_feature_names", ())) != list(NODE_FEATURE_NAMES):
        raise ValueError("train-only node normalization feature schema mismatch")
    if list(normalization.get("candidate_feature_names", ())) != list(CANDIDATE_FEATURE_NAMES):
        raise ValueError("train-only candidate normalization feature schema mismatch")
    _validate_history(history, projection)
    candidate_batch = build_candidate_features(projection, catalog)
    node_center = _normalization_vector(normalization, "node_center", len(NODE_FEATURE_NAMES))
    node_scale = _normalization_vector(normalization, "node_scale", len(NODE_FEATURE_NAMES))
    candidate_center = _normalization_vector(normalization, "candidate_center", len(CANDIDATE_FEATURE_NAMES))
    candidate_scale = _normalization_vector(normalization, "candidate_scale", len(CANDIDATE_FEATURE_NAMES))

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PolicyUnavailable("PYTORCH_UNAVAILABLE") from exc
    if device == "cpu":
        if not synthetic_cpu_fixture or not model_version.startswith("synthetic:"):
            raise RuntimeError("CPU_POLICY_INFERENCE_IS_SYNTHETIC_FIXTURE_ONLY")
        selected_device = torch.device("cpu")
    elif device == "cuda:0":
        if synthetic_cpu_fixture:
            raise ValueError("synthetic_cpu_fixture requires device='cpu'")
        try:
            from .gain_deployment import _check_gpu1
        except ImportError:  # pragma: no cover
            from wm_ppo_v1.gain_deployment import _check_gpu1
        try:
            selected_device = _check_gpu1()
        except RuntimeError as exc:
            if str(exc) in {"GPU1_UUID_NOT_EXPLICITLY_SELECTED", "UUID_SELECTED_CUDA_DEVICE_UNAVAILABLE",
                            "VISIBLE_DEVICE_UUID_MISMATCH"}:
                raise PolicyUnavailable(str(exc)) from exc
            raise
    else:
        raise RuntimeError("candidate policy inference is restricted to GPU1 cuda:0")

    graph = collate_graph_sequences([history])
    graph = _tensor_graph_batch(graph, torch, selected_device, node_center, node_scale)
    candidates = torch.as_tensor(candidate_batch.features, dtype=torch.float32, device=selected_device)
    candidates = ((candidates - torch.as_tensor(candidate_center, device=selected_device))
                  / torch.as_tensor(candidate_scale, device=selected_device)).unsqueeze(0)
    mask = torch.as_tensor(candidate_batch.candidate_mask, dtype=torch.bool, device=selected_device).unsqueeze(0)
    model = model.to(selected_device).eval()
    with torch.inference_mode():
        outputs = model(graph, candidates)
        logits = outputs["logits"]
        values = outputs["value"]
        if logits.shape != mask.shape or values.numel() != 1 or not torch.isfinite(values).all():
            raise ValueError("actor-critic returned malformed logits or value")
        if not torch.isfinite(logits[mask]).all():
            raise ValueError("actor-critic returned non-finite legal candidate logits")
        from torch.distributions import Categorical
        distribution = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
        if mode == "sample":
            probabilities = distribution.probs[0]
            index_tensor = torch.multinomial(probabilities, 1, generator=generator)[0]
            index = int(index_tensor.item())
        else:
            probs = distribution.probs[0].detach().cpu().numpy()
            best = float(np.max(probs[candidate_batch.candidate_mask]))
            tied = [i for i, value in enumerate(probs) if candidate_batch.candidate_mask[i] and value == best]
            index = min(tied, key=lambda i: candidate_batch.action_keys[i])
            index_tensor = torch.as_tensor(index, dtype=torch.long, device=selected_device)
        log_probability = float(distribution.log_prob(index_tensor.reshape(1)).item())
        value = float(values.reshape(-1)[0].item())
    return {
        "status": "selected", "index": index, "key": candidate_batch.action_keys[index],
        "action": dict(candidate_batch.actions[index]), "old_logprob": log_probability,
        "value": value, "legal_mask": candidate_batch.candidate_mask.tolist(),
        "noop_mask": candidate_batch.noop_mask.tolist(), "model_version": model_version,
        "mode": mode, "device": str(selected_device),
        "timings": {"wall_duration_ns": time.perf_counter_ns() - wall_start,
                    "cpu_duration_ns": time.process_time_ns() - cpu_start},
    }


class PublicCandidatePolicy:
    """Callable runtime adapter around a frozen model and explicit normalization."""

    def __init__(self, model: Any, normalization: Mapping[str, Any], *, model_version: str,
                 mode: str, device: str, synthetic_cpu_fixture: bool = False):
        self.model = model
        self.normalization = dict(normalization)
        self.model_version = model_version
        self.mode = mode
        self.device = device
        self.synthetic_cpu_fixture = synthetic_cpu_fixture

    def __call__(self, *, projection: Mapping[str, Any], history: Sequence[PublicHistoryFrame],
                 catalog: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return select_public_candidate(
            self.model, projection, history, catalog, normalization=self.normalization,
            model_version=self.model_version, mode=self.mode, device=self.device,
            synthetic_cpu_fixture=self.synthetic_cpu_fixture)
