"""Gated streaming trainer for the event-aware world-model stage."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .checkpoint import (
    capture_rng_state,
    save_training_checkpoint,
    verify_training_checkpoint_in_fresh_process,
)
from .contracts import OneStepTarget
from .decision_windows import DecisionWindow, iter_decision_windows
from .features import (
    ACTION_DIM,
    EVENT_DIM,
    EVENT_NAMES,
    EVENT_SCHEMA_VERSION,
    FEATURE_DIM,
    FEATURE_NAMES,
    collate_graph_sequences,
)
from .raw_public_windows import RawPublicWindow, iter_raw_public_windows


MODEL_SEEDS = (8201, 8202, 8203)
EXPECTED_GPU_UUID = "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e"
OPTIMIZER_SPEC = {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0}
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DYNAMIC_TARGET_NAMES = ("task_progress", "energy_delta", "delay", "risk")


@dataclass(frozen=True)
class SceneSource:
    scene_name: str
    scenario_id: str
    split: str
    trajectory: Path
    events: Path
    receipt: Path
    manifest: Path


class RunningNormalizer:
    """Incremental train-only node-feature normalization."""

    def __init__(self, width: int):
        import numpy as np
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values) -> None:
        import numpy as np
        matrix = np.asarray(values, dtype=np.float64).reshape(-1, len(self.mean))
        if not len(matrix):
            return
        batch_count = len(matrix)
        batch_mean = matrix.mean(axis=0)
        batch_m2 = ((matrix - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta ** 2 * self.count * batch_count / total
        self.count = total

    def parameters(self):
        import numpy as np
        if self.count == 0:
            raise ValueError("training split has no public nodes for normalization")
        scale = np.sqrt(self.m2 / self.count)
        scale[scale == 0] = 1.0
        return self.mean.astype(np.float32), scale.astype(np.float32)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str):
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _scene_sources(root: Path, receipt: Mapping[str, Any]) -> list[SceneSource]:
    sources = []
    for row in receipt.get("scenes", []):
        if not isinstance(row, Mapping):
            continue
        scenario_id, split = row.get("scenario_id"), row.get("split")
        scene_name = Path(str(row.get("output_path", ""))).name
        scene_dir = _safe_path(root, row.get("output_path"))
        if (scene_dir is None or not scene_name or not isinstance(scenario_id, str)
                or split not in {"train", "validation"}):
            continue
        rep = scene_dir / "representative"
        run_receipt_path, manifest_path = rep / "run_receipt.json", rep / "manifest.json"
        run_receipt, manifest = _read(run_receipt_path), _read(manifest_path)
        if run_receipt is None or manifest is None:
            continue
        logs = run_receipt.get("logs", {}) if isinstance(run_receipt.get("logs"), Mapping) else {}
        manifest_logs = manifest.get("logs", {}) if isinstance(manifest.get("logs"), Mapping) else {}
        trajectory_name = run_receipt.get("trajectory", logs.get("trajectory"))
        events_name = run_receipt.get("events", logs.get("events"))
        trajectory = _safe_path(rep, trajectory_name)
        events = _safe_path(rep, events_name)
        if (trajectory is None or events is None
                or manifest_logs.get("trajectory") != trajectory.name
                or manifest_logs.get("events") != events.name):
            continue
        sources.append(SceneSource(scene_name, scenario_id, split, trajectory, events,
                                   run_receipt_path, manifest_path))
    return sources


def _dynamic_windows(root: Path, sources: list[SceneSource], split: str | None = None):
    for source in sources:
        if split is not None and source.split != split:
            continue
        for window in iter_raw_public_windows(
                source.trajectory, source.events, source.receipt,
                history_steps=8, manifest_path=source.manifest):
            if window.branch != "representative":
                continue
            if not window.target.valid or not any(window.target.dynamic_mask.values()):
                continue
            yield source, window


def _batches(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _collate(samples, center, scale, device, *, dynamic: bool):
    import numpy as np
    import torch

    histories = [sample.history if isinstance(sample, DecisionWindow) else sample[1].history
                 for sample in samples]
    targets = ([sample[1].target for sample in samples] if dynamic else None)
    batch = collate_graph_sequences(histories, targets=targets)
    nodes = batch.node_features.copy()
    nodes = (nodes - center) / scale
    nodes[~batch.node_mask] = 0.0
    tensors = (
        torch.as_tensor(nodes, dtype=torch.float32, device=device),
        torch.as_tensor(batch.node_mask, dtype=torch.bool, device=device),
        torch.as_tensor(batch.sequence_mask, dtype=torch.bool, device=device),
        torch.as_tensor(batch.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(batch.edge_mask, dtype=torch.bool, device=device),
        torch.as_tensor(batch.edge_type, dtype=torch.long, device=device),
        torch.as_tensor(batch.event_features, dtype=torch.float32, device=device),
        torch.as_tensor(batch.action_features, dtype=torch.float32, device=device),
    )
    if dynamic:
        target = torch.as_tensor(batch.dynamic_target, dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch.dynamic_target_mask, dtype=torch.bool, device=device)
        return tensors, target, mask, batch
    gain = torch.as_tensor([sample.gain_target for sample in samples], dtype=torch.float32, device=device)
    return tensors, gain, batch


def _update_normalizer(normalizer: RunningNormalizer, histories) -> None:
    batch = collate_graph_sequences(histories)
    normalizer.update(batch.node_features[batch.node_mask])


def _fit_normalizer(
    decision_windows: list[DecisionWindow], sources: list[SceneSource], batch_size: int,
) -> tuple[Any, Any, dict[str, int]]:
    normalizer = RunningNormalizer(FEATURE_DIM)
    gain_count = 0
    for group in _batches((row for row in decision_windows if row.split == "train"), batch_size):
        _update_normalizer(normalizer, [row.history for row in group])
        gain_count += len(group)
    dynamic_count = 0
    for group in _batches(_dynamic_windows(Path("."), sources, "train"), batch_size):
        _update_normalizer(normalizer, [window.history for _, window in group])
        dynamic_count += len(group)
    center, scale = normalizer.parameters()
    return center, scale, {"gain_train_count": gain_count, "dynamic_train_count": dynamic_count,
                           "normalizer_node_count": normalizer.count}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_identity(root: Path, dataset_rows, sources: list[SceneSource]) -> str:
    material = {
        "rows": [{key: row.get(key) for key in ("state_id", "scenario_id", "split", "target")}
                 for row in dataset_rows],
        "batch_receipt_sha256": _file_sha256(root / "batch_receipt.json"),
        "aggregate_sha256": _file_sha256(root / "aggregate.json"),
        "source_files": [],
        "feature_schema": list(FEATURE_NAMES), "event_schema": list(EVENT_NAMES),
    }
    for source in sources:
        files = [source.trajectory, source.events, source.receipt, source.manifest,
                 source.trajectory.parent / "state_sampling" / "state_index.json",
                 source.trajectory.parent / "state_sampling" / "state_index.sha256"]
        material["source_files"].append({
            "scene": source.scene_name,
            "files": [{"name": path.name, "sha256": _file_sha256(path)}
                      for path in files if path.is_file()],
        })
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _device_or_reason():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip() != EXPECTED_GPU_UUID:
        return None, {"status": "NOT_RUN", "reason": "GPU1_UUID_NOT_EXPLICITLY_SELECTED",
                      "required_uuid": EXPECTED_GPU_UUID, "cuda_visible_devices": visible}
    try:
        import torch
    except ImportError:
        return None, {"status": "NOT_RUN", "reason": "PYTORCH_UNAVAILABLE"}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return None, {"status": "NOT_RUN", "reason": "UUID_SELECTED_CUDA_DEVICE_UNAVAILABLE",
                      "visible_device_count": torch.cuda.device_count()}
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, "uuid", "unknown"))
    if actual_uuid != EXPECTED_GPU_UUID:
        return None, {"status": "NOT_RUN", "reason": "VISIBLE_DEVICE_UUID_MISMATCH",
                      "required_uuid": EXPECTED_GPU_UUID, "reported_uuid": actual_uuid}
    return torch.device("cuda:0"), {"status": "READY", "device": "cuda:0",
        "physical_gpu_uuid": EXPECTED_GPU_UUID,
        "logical_to_physical_mapping": {"cuda:0": EXPECTED_GPU_UUID},
        "device_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda}


def _model_inputs_from_collate(tensors):
    return tensors


def _dynamic_step(model, optimizer, samples, center, scale, device):
    import torch
    inputs, target, mask, _ = _collate(samples, center, scale, device, dynamic=True)
    if not mask.any():
        return None
    optimizer.zero_grad(set_to_none=True)
    prediction = model(*inputs)["dynamic"]
    loss = torch.nn.functional.mse_loss(prediction[mask], target[mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), mask.sum(dim=(0, 1)).detach().cpu().tolist()


def _gain_step(model, optimizer, samples, center, scale, device):
    import torch
    inputs, target, _ = _collate(samples, center, scale, device, dynamic=False)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(*inputs)["gain"]
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _evaluate(model, decisions, sources, center, scale, device, batch_size):
    import torch
    model.eval()
    dynamic_squared, dynamic_fields, dynamic_batches = 0.0, 0, 0
    dynamic_target_counts = {name: 0 for name in DYNAMIC_TARGET_NAMES}
    gain_squared, gain_count, gain_batches = 0.0, 0, 0
    with torch.no_grad():
        for samples in _batches(_dynamic_windows(Path("."), sources, "validation"), batch_size):
            inputs, target, mask, _ = _collate(samples, center, scale, device, dynamic=True)
            if mask.any():
                error = model(*inputs)["dynamic"][mask] - target[mask]
                dynamic_squared += float((error * error).sum().cpu())
                dynamic_fields += int(mask.sum().item())
                per_target = mask.sum(dim=(0, 1)).detach().cpu().tolist()
                for name, count in zip(DYNAMIC_TARGET_NAMES, per_target):
                    dynamic_target_counts[name] += int(count)
                dynamic_batches += 1
        for samples in _batches((row for row in decisions if row.split == "validation"), batch_size):
            inputs, target, _ = _collate(samples, center, scale, device, dynamic=False)
            error = model(*inputs)["gain"] - target
            gain_squared += float((error * error).sum().cpu())
            gain_count += len(samples)
            gain_batches += 1
    model.train()
    return {
        "dynamic_mse": dynamic_squared / dynamic_fields if dynamic_fields else None,
        "dynamic_valid_field_count": dynamic_fields, "dynamic_batch_count": dynamic_batches,
        "dynamic_valid_target_counts": dynamic_target_counts,
        "gain_mse": gain_squared / gain_count if gain_count else None,
        "gain_sample_count": gain_count, "gain_batch_count": gain_batches,
    }


def run_training(
    batch_output: str | Path,
    output_dir: str | Path,
    *,
    max_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Train only after frozen batch, post-extraction, and decision-row gates pass."""
    import numpy as np
    try:
        from ..light_models_v1.data_gate import check_feature_coverage
        from ..light_models_v1.dataset import load_feature_dataset
    except ImportError:
        from light_models_v1.data_gate import check_feature_coverage
        from light_models_v1.dataset import load_feature_dataset
    from .world_model import EventAwareWorldModel

    start = time.perf_counter()
    root, output = Path(batch_output).resolve(), Path(output_dir).resolve()
    dataset = load_feature_dataset(root)
    receipt = {"schema": "wm_ppo_training_receipt_v1", "stage": "world-model",
        "status": "NOT_RUN", "batch_gate": dataset.get("gate"), "updates": 0,
        "device": None, "models": {}, "g_stage": "NOT_RUN", "g_wm_stage": "NOT_RUN"}
    if not dataset.get("gate", {}).get("ready"):
        receipt["reason"] = dataset.get("gate", {}).get("reason", "TERMINAL_BATCH_GATE_NOT_MET")
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    feature_coverage = check_feature_coverage(dataset.get("rows", []))
    receipt["feature_coverage"] = feature_coverage
    if not feature_coverage.get("met_30_30"):
        receipt["reason"] = "POST_EXTRACTION_30_30_GATE_NOT_MET"
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt

    decisions = list(iter_decision_windows(root))
    decision_coverage = check_feature_coverage([
        {"scenario_id": row.scenario_id, "split": row.split, "target": row.gain_target}
        for row in decisions])
    receipt["decision_gain_coverage"] = decision_coverage
    if not decision_coverage.get("met_30_30"):
        receipt["reason"] = "DECISION_WINDOW_GAIN_30_30_GATE_NOT_MET"
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt

    batch_receipt = _read(root / "batch_receipt.json")
    if batch_receipt is None:
        receipt["reason"] = "BATCH_RECEIPT_CHANGED_AFTER_GATE"
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    sources = _scene_sources(root, batch_receipt)
    split_by_scenario: dict[str, str] = {}
    for source in sources:
        prior = split_by_scenario.setdefault(source.scenario_id, source.split)
        if prior != source.split:
            receipt["reason"] = "SCENARIO_CROSSES_TRAIN_VALIDATION_SPLIT"
            receipt["elapsed_s"] = time.perf_counter() - start
            output.mkdir(parents=True, exist_ok=True)
            (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            return receipt
    train_scenarios = {source.scenario_id for source in sources if source.split == "train"}
    validation_scenarios = {source.scenario_id for source in sources if source.split == "validation"}
    if not train_scenarios or not validation_scenarios or train_scenarios & validation_scenarios:
        receipt["reason"] = "TRAIN_VALIDATION_SCENARIO_SPLIT_INVALID"
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    receipt["scenario_split"] = {"train": sorted(train_scenarios),
                                 "validation": sorted(validation_scenarios),
                                 "disjoint": True}

    device, device_receipt = _device_or_reason()
    if device is None:
        receipt["device_check"] = device_receipt
        receipt["reason"] = device_receipt["reason"]
        receipt["elapsed_s"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    center, scale, sample_counts = _fit_normalizer(decisions, sources, batch_size)
    receipt["sample_counts"] = sample_counts
    receipt["action_known_fraction"] = {
        "decision": _known_action_fraction([row.history for row in decisions if row.split == "train"]),
        "dynamic": _dynamic_action_fraction(sources, batch_size),
        "semantics": "action_features[...,0] is known/applied mask; unknown zero is not interpreted as NOOP",
    }
    import torch
    receipt["device_check"] = device_receipt
    receipt["device"] = str(device)
    receipt["normalization"] = {"scope": "train_split_only", "center": center.tolist(),
                                 "scale": scale.tolist(), "count": sample_counts["normalizer_node_count"]}
    data_hash = _dataset_identity(root, dataset["rows"], sources)
    receipt["dataset_sha256"] = data_hash
    architecture = {"node_dim": FEATURE_DIM, "hidden_dim": 64, "dynamic_dim": len(DYNAMIC_TARGET_NAMES),
                    "event_dim": EVENT_DIM, "action_dim": ACTION_DIM}
    output.mkdir(parents=True, exist_ok=True)
    normalization = (np.asarray(center, dtype=np.float32), np.asarray(scale, dtype=np.float32))
    train_gain = [row for row in decisions if row.split == "train"]
    probe_batch = next(_batches(_dynamic_windows(root, sources, "train"), 1), None)
    if probe_batch:
        probe_inputs = _collate(probe_batch, *normalization, device, dynamic=True)[0]
    else:
        probe_inputs = _collate(train_gain[:1], *normalization, device, dynamic=False)[0]

    total_updates = 0
    for seed in MODEL_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = EventAwareWorldModel(**architecture).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_SPEC["lr"],
                                     weight_decay=OPTIMIZER_SPEC["weight_decay"])
        epoch_rows = []
        for epoch in range(max_epochs):
            model.train()
            dynamic_loss_sum, dynamic_updates, dynamic_fields = 0.0, 0, 0
            dynamic_target_counts = {name: 0 for name in DYNAMIC_TARGET_NAMES}
            for samples in _batches(_dynamic_windows(root, sources, "train"), batch_size):
                result = _dynamic_step(model, optimizer, samples, *normalization, device)
                if result is not None:
                    loss, target_counts = result
                    dynamic_loss_sum += loss
                    dynamic_fields += sum(target_counts)
                    for name, count in zip(DYNAMIC_TARGET_NAMES, target_counts):
                        dynamic_target_counts[name] += int(count)
                    dynamic_updates += 1
            shuffled_gain = train_gain[:]
            random.shuffle(shuffled_gain)
            gain_loss_sum, gain_updates = 0.0, 0
            for samples in _batches(shuffled_gain, batch_size):
                gain_loss_sum += _gain_step(model, optimizer, samples, *normalization, device)
                gain_updates += 1
            total_updates += dynamic_updates + gain_updates
            epoch_rows.append({"epoch": epoch + 1,
                "dynamic_loss_mean": dynamic_loss_sum / dynamic_updates if dynamic_updates else None,
                "dynamic_updates": dynamic_updates, "dynamic_valid_fields": dynamic_fields,
                "dynamic_valid_target_counts": dynamic_target_counts,
                "gain_loss_mean": gain_loss_sum / gain_updates if gain_updates else None,
                "gain_updates": gain_updates})

        validation = _evaluate(model, decisions, sources, *normalization, device, batch_size)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "seed": seed, "dataset_sha256": data_hash, "feature_names": list(FEATURE_NAMES),
            "event_names": list(EVENT_NAMES), "event_schema_version": EVENT_SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM, "event_dim": EVENT_DIM, "action_dim": ACTION_DIM,
            "dynamic_target_names": list(DYNAMIC_TARGET_NAMES), "architecture": architecture,
            "optimizer": dict(OPTIMIZER_SPEC), "normalization": {
                "scope": "train_split_only", "center": center.tolist(), "scale": scale.tolist(),
                "node_count": sample_counts["normalizer_node_count"]},
            "split_scenarios": receipt["scenario_split"], "device": device_receipt,
            "rng_state_sha256": hashlib.sha256(repr(capture_rng_state(torch)).encode()).hexdigest(),
            "epoch_count": max_epochs, "batch_size": batch_size,
            "training_phase_losses": epoch_rows, "validation_metrics": validation,
            "action_known_fraction": receipt["action_known_fraction"],
            "phase_contract": {"dynamic": "adjacent_raw_post_step_history_to_next_post_step_target",
                               "gain": "raw_history_through_t_minus_1_plus_pre_candidate_projection_t"},
        }
        checkpoint_path = seed_dir / "world_model.pt"
        checkpoint_hash = save_training_checkpoint(checkpoint_path, model, optimizer, metadata,
                                                    probe_inputs, device=str(device))
        restore = verify_training_checkpoint_in_fresh_process(checkpoint_path, device=str(device))
        receipt["models"][str(seed)] = {"checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash, "training_updates": sum(
                row["dynamic_updates"] + row["gain_updates"] for row in epoch_rows),
            "epochs": epoch_rows, "validation": validation, "fresh_process_restore": restore}
        if restore.get("status") != "PASS":
            receipt["status"] = "FAILED_CHECKPOINT_RESTORE"
            receipt["reason"] = f"FRESH_PROCESS_RESTORE_FAILED_SEED_{seed}"
            receipt["updates"] = total_updates
            receipt["elapsed_s"] = time.perf_counter() - start
            (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            return receipt
    receipt["status"] = "COMPLETE"
    receipt["updates"] = total_updates
    receipt["elapsed_s"] = time.perf_counter() - start
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def _known_action_fraction(histories) -> float | None:
    if not histories:
        return None
    batch = collate_graph_sequences(histories)
    return float(batch.action_features[..., 0].mean())


def _dynamic_action_fraction(sources: list[SceneSource], batch_size: int) -> float | None:
    import numpy as np
    known, total = 0, 0
    for samples in _batches(_dynamic_windows(Path("."), sources, "train"), batch_size):
        batch = collate_graph_sequences([window.history for _, window in samples])
        mask = batch.sequence_mask
        known += int(batch.action_features[..., 0][mask].sum())
        total += int(mask.sum())
    return known / total if total else None
