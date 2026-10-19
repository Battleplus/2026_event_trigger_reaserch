from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SUPPORT_SCHEMA = "n02-full-wm-training-support/1"
MIN_GAIN_CLASS_STATES = 30
MLP_SEEDS = (8201, 8202, 8203)
EVENT_SEEDS = (8201, 8202, 8203)
EVENT_PATIENCE = 10
EVENT_MAX_EPOCHS = 40
EVENT_BATCH_SIZE = 8
EVENT_LEARNING_RATE = 0.001


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _torch_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _torch_payload_sha256(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _cuda_training_timer_start() -> tuple[float, float]:
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    return time.perf_counter(), time.process_time()


def _cuda_training_timer_stop(start: tuple[float, float]) -> dict[str, Any]:
    torch.cuda.synchronize(0)
    return {
        "elapsed_wall_seconds": time.perf_counter() - start[0],
        "process_cpu_seconds": time.process_time() - start[1],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
    }


def _load_legacy(legacy_source: str | Path) -> tuple[Any, Any, Any]:
    civilian = Path(legacy_source).resolve() / "package" / "civilian"
    if not civilian.is_dir():
        raise FileNotFoundError(f"legacy civilian source not found: {civilian}")
    sys.path.insert(0, str(civilian))
    for name in ("gain_models", "event_models", "scenario_schema", "v3_shards"):
        sys.modules.pop(name, None)
    import gain_models
    import event_models
    import v3_shards
    return gain_models, event_models, v3_shards


def _committed_shards(shard_root: str | Path, v3_shards: Any) -> list[tuple[dict[str, Any], Path]]:
    root = Path(shard_root).resolve(strict=True)
    expected_binding = v3_shards.current_binding()
    markers = sorted(root.glob("**/shards/*/*/*.commit.json"))
    committed: list[tuple[dict[str, Any], Path]] = []
    for marker_path in markers:
        marker = v3_shards.verify_scenario_shard(marker_path, expected_binding=expected_binding)
        if marker.get("split") not in {"train", "validation"}:
            raise ValueError("TRAINING_SUPPORT_SEALED_SPLIT")
        committed.append((marker, marker_path.with_name(str(marker["shard_file"]))))
    return committed


def _read_committed_rows(shards: Sequence[tuple[dict[str, Any], Path]]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    gain_rows: list[dict[str, Any]] = []
    trajectories: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for marker, path in shards:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                wrapper = json.loads(line)
                if not isinstance(wrapper, dict):
                    raise ValueError(f"TRAINING_SUPPORT_ROW_OBJECT:{path}:{line_number}")
                if isinstance(wrapper.get("training_row"), Mapping):
                    row = dict(wrapper["training_row"])
                    if row.get("split") != marker.get("split") or row.get("scenario_id") != marker.get("scenario_id"):
                        raise ValueError("TRAINING_SUPPORT_ROW_MARKER_BINDING")
                    gain_rows.append(row)
                trajectory = wrapper.get("trajectory")
                if isinstance(trajectory, Mapping) and isinstance(trajectory.get("trajectory_trace"), list):
                    trajectories.append((marker, dict(trajectory)))
    return gain_rows, trajectories


def _gain_eligible_rows(rows: Sequence[Mapping[str, Any]], gain_models: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    counts = {"rows_seen": len(rows), "ordinary_rows": 0, "oracle_valid_rows": 0, "eligible_rows": 0}
    for raw in rows:
        label = raw.get("label") if isinstance(raw.get("label"), Mapping) else {}
        oracle = raw.get("oracle_receipt") if isinstance(raw.get("oracle_receipt"), Mapping) else {}
        ordinary = raw.get("forced") is False
        if ordinary:
            counts["ordinary_rows"] += 1
        oracle_valid = oracle.get("valid") is True and oracle.get("label_valid_mask") is True
        label_valid = label.get("valid_mask") is True
        if ordinary and oracle_valid and label_valid:
            counts["oracle_valid_rows"] += 1
        if not ordinary or not oracle_valid or not label_valid:
            continue
        feature_map = gain_models._feature_map(raw.get("public_features"))
        gain = oracle.get("recomputed_gain_mean")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
            continue
        label_gain = label.get("gain_mean")
        if isinstance(label_gain, bool) or not isinstance(label_gain, (int, float)) or not math.isclose(float(label_gain), float(gain), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("TRAINING_SUPPORT_ORACLE_GAIN_MISMATCH")
        row = {
            "state_id": str(raw.get("state_id", "")),
            "scenario_id": str(raw.get("scenario_id", "")),
            "parent_family": str(raw.get("parent_family", raw.get("scenario_id", ""))),
            "split": str(raw.get("split", "")),
            "forced": False,
            "valid_mask": True,
            "public_features": feature_map,
            "public_history": copy.deepcopy(raw.get("public_history")),
            "gain_mean": float(gain),
        }
        for field in (
            "scenario_content_sha256", "scenario_definition_sha256", "scenario_sha256",
            "scenario_hash", "scenario_instance_hash", "event_tape_id", "event_tape_sha256",
            "event_tape_hash", "trajectory_id", "trajectory_sha256", "trajectory_hash",
            "state_id", "state_sha256", "state_hash", "snapshot_sha256",
            "private_snapshot_sha256", "source_snapshot_sha256", "lineage_id", "lineage_sha256",
            "parent_instance_id", "scenario_seed", "scenario_instance_seed", "instance_seed",
            "generation_seed", "trajectory_seed", "seed", "lineage", "scenario",
            "scenario_content", "scenario_definition", "scenario_payload", "raw_scenario",
        ):
            if field in raw:
                row[field] = raw[field]
        if not row["state_id"] or not row["scenario_id"]:
            raise ValueError("TRAINING_SUPPORT_GAIN_ID_REQUIRED")
        eligible.append(row)
    counts["eligible_rows"] = len(eligible)
    return eligible, counts


def _write_not_run(output_dir: str | Path, reason: str, detail: Mapping[str, Any]) -> None:
    _json_write(Path(output_dir) / "TRAINING_STATUS.json", {
        "schema": SUPPORT_SCHEMA,
        "status": "not_run",
        "reason": reason,
        "detail": dict(detail),
        "optimizer_started": False,
    })


def collect_gain_training_rows(
    shard_root: str | Path,
    legacy_source: str | Path,
    output_dir: str | Path,
    *,
    minimum_per_polarity: int = MIN_GAIN_CLASS_STATES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gain_models, _, shard_api = _load_legacy(legacy_source)
    shards = _committed_shards(shard_root, shard_api)
    raw_rows, _ = _read_committed_rows(shards)
    strict_oracle_rows = []
    for raw in raw_rows:
        label = raw.get("label") if isinstance(raw.get("label"), Mapping) else {}
        oracle = raw.get("oracle_receipt") if isinstance(raw.get("oracle_receipt"), Mapping) else {}
        if (
            raw.get("forced") is False
            and label.get("valid_mask") is True
            and oracle.get("valid") is True
            and oracle.get("label_valid_mask") is True
        ):
            strict_oracle_rows.append({"training_row": raw})
    summary = shard_api.summarize_gain_states(strict_oracle_rows)
    split_summary = summary.get("splits", {})
    positive_by_split = {split: int(split_summary.get(split, {}).get("positive", 0)) for split in ("train", "validation")}
    nonpositive_by_split = {split: int(split_summary.get(split, {}).get("non_positive", 0)) for split in ("train", "validation")}
    positive_count = sum(positive_by_split.values())
    nonpositive_count = sum(nonpositive_by_split.values())
    rows, counts = _gain_eligible_rows(raw_rows, gain_models)
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation")}
    detail = {
        "source_shard_count": len(shards),
        "raw_training_row_count": len(raw_rows),
        "eligibility": counts,
        "gain_state_gate": {
            "source": "v3_shards.summarize_gain_states",
            "minimum_per_polarity": int(minimum_per_polarity),
            "distinct_ordinary_positive_state_ids": positive_count,
            "distinct_ordinary_non_positive_state_ids": nonpositive_count,
            "positive_by_split": positive_by_split,
            "non_positive_by_split": nonpositive_by_split,
            "state_count": int(summary.get("state_count", 0)),
        },
        "eligible_fit_rows_by_split": {key: len(value) for key, value in by_split.items()},
        "data_hashes": {
            "source_binding_sha256": _json_sha256(shards[0][0]["binding"]) if shards else None,
            "commit_marker_sha256s": sorted(str(marker.get("marker_sha256", "")) for marker, _ in shards),
            "compressed_shard_sha256s": sorted(str(marker.get("compressed_sha256", "")) for marker, _ in shards),
            "eligible_gain_examples_sha256": _json_sha256([
                {"state_id": row["state_id"], "scenario_id": row["scenario_id"],
                 "split": row["split"], "public_features": row["public_features"], "gain_mean": row["gain_mean"]}
                for row in sorted(rows, key=lambda item: (item["split"], item["state_id"]))
            ]),
        },
        "case_ids_used_as_features": False,
    }
    gate_passed = (
        positive_count >= minimum_per_polarity
        and nonpositive_count >= minimum_per_polarity
        and all(positive_by_split[split] > 0 and nonpositive_by_split[split] > 0 for split in ("train", "validation"))
    )
    detail["gain_state_gate"]["passed"] = gate_passed
    if not gate_passed:
        _write_not_run(output_dir, "INSUFFICIENT_30_POSITIVE_AND_30_NONPOSITIVE_STATES_WITH_SPLIT_COVERAGE", detail)
        raise ValueError("TRAINING_SUPPORT_GAIN_STATE_30_30_GATE")
    gain_models.check_group_leakage(rows)
    return rows, detail


def _gain_arrays(rows: Sequence[Mapping[str, Any]], gm: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = [row for row in rows if row["split"] == "train"]
    validation_model, calibration = gm._validation_parts([row for row in rows if row["split"] == "validation"], 8201)
    normalizer = gm._fit_normalizer(train)
    names = normalizer["feature_names"]
    x_train, y_train = gm._xy(train, names)
    x_model, y_model = gm._xy(validation_model, names)
    x_cal, y_cal = gm._xy(calibration, names)
    return normalizer, gm._transform(x_train, normalizer), y_train, gm._transform(x_model, normalizer), y_model, gm._transform(x_cal, normalizer), y_cal


def train_gain_cpu(
    shard_root: str | Path,
    legacy_source: str | Path,
    output_dir: str | Path,
    *,
    minimum_per_polarity: int = MIN_GAIN_CLASS_STATES,
) -> dict[str, Any]:
    """Train constant/Ridge and, when sklearn is installed, the legacy tree."""
    gm, _, _ = _load_legacy(legacy_source)
    rows, detail = collect_gain_training_rows(shard_root, legacy_source, output_dir, minimum_per_polarity=minimum_per_polarity)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalizer, x_train, y_train, x_val, y_val, x_cal, y_cal = _gain_arrays(rows, gm)
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("constant", gm._fit_constant(y_train), {"selection": "training_mean"}),
        ("ridge", gm._fit_ridge(x_train, y_train, 1.0), {"alpha": 1.0}),
    ]
    tree_state = "unavailable"
    try:
        tree, selection = gm._fit_gbtree(x_train, y_train, x_val, y_val, [3, 5], 8201)
        import joblib
        tree_path = output / "gain_tree.joblib"
        joblib.dump(tree.pop("estimator"), tree_path)
        tree["estimator_file"] = tree_path.name
        selection["estimator_sha256"] = _sha256(tree_path)
        candidates.append(("gradient_boosting", tree, selection))
        tree_state = "complete"
    except RuntimeError as exc:
        if "GAIN_SKLEARN_REQUIRED" not in str(exc):
            raise
        tree_state = "missing_optional_sklearn"

    records = []
    for name, model, fit_info in candidates:
        val_pred = gm._predict_payload(model, x_val, output)
        cal_pred = gm._predict_payload(model, x_cal, output)
        epsilon = gm.select_epsilon(y_val, val_pred, gm._epsilon_candidates({}))
        calibration = gm.calibrate_lower_bound(y_cal, cal_pred, 0.9)
        bundle = {
            "schema": gm.SCHEMA,
            "model_name": name,
            "model": model,
            "normalizer": normalizer,
            "threshold": {**epsilon, "selection_split": "validation_model", "frozen_before_test": True},
            "calibration": {**calibration, "calibration_split": "validation_calibration"},
            "training": {"fit_split": "train", "fit_row_count": int(len(y_train)), "fit_info": fit_info},
            "data_identity": {"fit_row_count": int(len(y_train)), "test_accessed": False, "case_ids_used_as_features": False},
        }
        path = output / f"{name}.json"
        digest = gm._save_bundle(path, bundle)
        records.append({"model_name": name, "bundle_file": path.name, "bundle_sha256": digest,
                        "validation_model_mae": float(np.mean(np.abs(val_pred - y_val))),
                        "calibration": calibration, "epsilon": epsilon["epsilon"], "fit_info": fit_info})
    result = {
        "schema": SUPPORT_SCHEMA, "status": "complete", "stage": "train-cpu",
        "rows": detail, "tree_status": tree_state, "models": records,
    }
    _json_write(output / "CPU_TRAINING.json", result)
    _json_write(output / "TRAINING_STATUS.json", {"schema": SUPPORT_SCHEMA, "status": "complete", "stage": "train-cpu", "optimizer_started": False})
    return result


def _gain_mlp_train_cuda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    checkpoint_path: Path,
    *,
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("TRAINING_SUPPORT_CUDA_REQUIRED")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = _CudaGainMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    val_x = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed)
    best_mae, best_state, best_optimizer, best_epoch, best_updates = math.inf, None, None, 0, 0
    best_rng: dict[str, Any] | None = None
    stale = updates = 0
    resource_timer = _cuda_training_timer_start()
    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(train_x), generator=batch_generator)
        for start in range(0, len(order), 32):
            indexes = order[start:start + 32].to(device)
            prediction, positive_logit = model(train_x[indexes])
            loss = F.huber_loss(prediction, train_y[indexes], delta=1.0)
            loss = loss + 0.25 * F.binary_cross_entropy_with_logits(positive_logit, (train_y[indexes] > 0).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise FloatingPointError("TRAINING_SUPPORT_CUDA_NONFINITE_GRADIENT")
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            prediction, _ = model(val_x)
            mae = float((prediction - val_y).abs().mean().item())
        if mae < best_mae - 1e-12:
            best_mae, best_epoch, stale = mae, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_updates = updates
            best_rng = {
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": [state.clone() for state in torch.cuda.get_rng_state_all()],
                "numpy_rng_state": copy.deepcopy(np.random.get_state()),
                "python_rng_state": copy.deepcopy(random.getstate()),
                "batch_generator_state": batch_generator.get_state().clone(),
            }
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None or best_optimizer is None or best_rng is None:
        raise RuntimeError("TRAINING_SUPPORT_CUDA_NO_BEST_CHECKPOINT")
    resource_usage = _cuda_training_timer_stop(resource_timer)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {"architecture": "mlp64x64", "optimizer": "Adam", "learning_rate": 0.001,
                       "batch_size": 32, "max_epochs": 100, "patience": 10,
                       "loss": "Huber(delta=1)+0.25*BCE(positive_gain)", "device": "cuda:0"}
    optimizer_state_sha256 = _torch_payload_sha256(best_optimizer)
    torch.save({"schema": "gain-mlp-training-checkpoint/1", "seed": seed,
                "model_state": best_state, "optimizer_state": best_optimizer,
                "optimizer_updates_at_best": best_updates, "total_optimizer_updates": updates,
                "best_epoch": best_epoch, "training_device": "cuda:0",
                "normalizer": copy.deepcopy(checkpoint_metadata["normalizer"]),
                "data_hashes": copy.deepcopy(checkpoint_metadata["data_hashes"]),
                "split_identity": copy.deepcopy(checkpoint_metadata["split_identity"]),
                "training_config": training_config,
                "resource_usage": resource_usage,
                "optimizer_state_sha256": optimizer_state_sha256,
                **best_rng}, checkpoint_path)
    payload = {
        "kind": "mlp64x64", "seed": seed,
        "model_state": {key: value.tolist() for key, value in best_state.items()},
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "optimizer_updates": updates, "optimizer_updates_at_best": best_updates,
        "best_epoch": best_epoch, "validation_mae": best_mae,
        "training_config": training_config,
        "optimizer_state_sha256": optimizer_state_sha256,
    }
    return payload, {"seed": seed, "best_epoch": best_epoch, "optimizer_updates": updates,
                     "optimizer_updates_at_best": best_updates, "validation_mae": best_mae,
                     "training_device": "cuda:0", "epochs_completed": epoch,
                     "resource_usage": resource_usage}


class _CudaGainMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.gain_head = nn.Linear(64, 1)
        self.positive_head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.gain_head(hidden).squeeze(-1), self.positive_head(hidden).squeeze(-1)


def _restore_gain_cuda_probe(request_path: str | Path) -> dict[str, Any]:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    gm, _, _ = _load_legacy(request["legacy_source"])
    bundle_path = Path(request["bundle_path"]).resolve(strict=True)
    bundle = gm.load_bundle(bundle_path)
    payload = bundle["model"]
    checkpoint_path = bundle_path.parent / str(payload["checkpoint_file"])
    if _sha256(checkpoint_path) != payload.get("checkpoint_sha256"):
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_CHECKPOINT_HASH")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "gain-mlp-training-checkpoint/1":
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_CHECKPOINT_SCHEMA")
    if checkpoint.get("normalizer") != bundle.get("normalizer"):
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_CHECKPOINT_NORMALIZER")
    if not checkpoint.get("optimizer_state") or checkpoint.get("optimizer_updates_at_best", 0) < 1:
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_OPTIMIZER_STATE_MISSING")
    if not all(key in checkpoint for key in ("torch_rng_state", "cuda_rng_states", "numpy_rng_state", "python_rng_state", "batch_generator_state", "data_hashes", "training_config")):
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_CHECKPOINT_METADATA")
    if not torch.cuda.is_available():
        raise RuntimeError("TRAINING_SUPPORT_GAIN_CUDA_RESTORE_DEVICE_UNAVAILABLE")
    device = torch.device("cuda:0")
    model = gm._GainMLP(len(bundle["normalizer"]["feature_names"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(checkpoint["training_config"]["learning_rate"]))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    normalized = gm._normalizer_matrix(request["features"], bundle["normalizer"])
    input_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    with torch.no_grad():
        gain, _ = model(input_tensor)
    optimizer_devices: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                name = str(value.device)
                optimizer_devices[name] = optimizer_devices.get(name, 0) + 1
    if not any(name.startswith("cuda:0") for name in optimizer_devices):
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_OPTIMIZER_NOT_ON_CUDA")
    return {
        "gain_hat": float(gain[0].item()), "checkpoint_valid": True,
        "optimizer_state_restored": True, "optimizer_updates_at_best": int(checkpoint["optimizer_updates_at_best"]),
        "inference_device": str(device), "gpu_name": torch.cuda.get_device_name(0),
        "optimizer_tensor_devices": optimizer_devices,
    }


def fresh_process_gain_cuda_restore(bundle_path: str | Path, features: Mapping[str, Any], legacy_source: str | Path, tolerance: float = 1e-6) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve(strict=True)
    gm, _, _ = _load_legacy(legacy_source)
    bundle = gm.load_bundle(bundle_path)
    expected = float(gm._predict_payload(bundle["model"], gm._normalizer_matrix(features, bundle["normalizer"]), bundle_path.parent)[0])
    request_path = bundle_path.parent / f"restore_cuda_request_{os.getpid()}.json"
    _json_write(request_path, {"bundle_path": str(bundle_path), "features": dict(features), "legacy_source": str(Path(legacy_source).resolve())})
    try:
        process = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--restore-gain-cuda", str(request_path)],
            cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, timeout=180,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, check=False,
        )
    finally:
        request_path.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(f"TRAINING_SUPPORT_GAIN_CUDA_FRESH_PROCESS_FAILED:{process.stderr[-1200:]}")
    actual = json.loads(process.stdout.strip().splitlines()[-1])
    if not math.isclose(actual["gain_hat"], expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_FRESH_PROCESS_PREDICTION_MISMATCH")
    if actual.get("checkpoint_valid") is not True or actual.get("optimizer_state_restored") is not True or actual.get("inference_device") != "cuda:0":
        raise ValueError("TRAINING_SUPPORT_GAIN_CUDA_FRESH_PROCESS_RESTORE_INVALID")
    return {"fresh_process": True, "same_input_prediction_matches": True, "expected_gain_hat": expected, **actual}


def train_gain_neural_cuda(
    shard_root: str | Path,
    legacy_source: str | Path,
    output_dir: str | Path,
    *,
    minimum_per_polarity: int = MIN_GAIN_CLASS_STATES,
) -> dict[str, Any]:
    """Fit only the neural gain models; this path does not import sklearn."""
    if not torch.cuda.is_available():
        _write_not_run(output_dir, "CUDA_UNAVAILABLE", {"optimizer_started": False})
        raise RuntimeError("TRAINING_SUPPORT_CUDA_REQUIRED")
    gm, _, _ = _load_legacy(legacy_source)
    rows, detail = collect_gain_training_rows(shard_root, legacy_source, output_dir, minimum_per_polarity=minimum_per_polarity)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalizer, x_train, y_train, x_val, y_val, x_cal, y_cal = _gain_arrays(rows, gm)
    candidates = []
    for seed in MLP_SEEDS:
        split_identity = {
            "train_state_ids_sha256": _json_sha256(sorted(row["state_id"] for row in rows if row["split"] == "train")),
            "validation_model_scenario_ids_sha256": _json_sha256(sorted({str(row["scenario_id"]) for row in gm._validation_parts([row for row in rows if row["split"] == "validation"], 8201)[0]})),
            "validation_calibration_scenario_ids_sha256": _json_sha256(sorted({str(row["scenario_id"]) for row in gm._validation_parts([row for row in rows if row["split"] == "validation"], 8201)[1]})),
        }
        model, fit = _gain_mlp_train_cuda(
            x_train, y_train, x_val, y_val, seed, output / f"gain_mlp_seed_{seed}.pt",
            checkpoint_metadata={"normalizer": normalizer, "data_hashes": detail["data_hashes"],
                                 "split_identity": split_identity},
        )
        candidates.append((f"mlp_seed_{seed}", model, fit))
    records = []
    for name, model, fit in candidates:
        pred_val = gm._predict_payload(model, x_val, output)
        pred_cal = gm._predict_payload(model, x_cal, output)
        epsilon = gm.select_epsilon(y_val, pred_val, gm._epsilon_candidates({}))
        calibration = gm.calibrate_lower_bound(y_cal, pred_cal, 0.9)
        bundle = {
            "schema": gm.SCHEMA, "model_name": name, "model": model, "normalizer": normalizer,
            "threshold": {**epsilon, "selection_split": "validation_model", "frozen_before_test": True},
            "calibration": {**calibration, "calibration_split": "validation_calibration"},
            "training": {"fit_split": "train", "fit_row_count": int(len(y_train)), "fit_info": fit},
            "data_identity": {"fit_row_count": int(len(y_train)), "test_accessed": False, "case_ids_used_as_features": False},
        }
        path = output / f"{name}.json"
        digest = gm._save_bundle(path, bundle)
        restore = fresh_process_gain_cuda_restore(path, rows[0]["public_features"], legacy_source)
        records.append({"model_name": name, "bundle_file": path.name, "bundle_sha256": digest,
                        "validation_model_mae": float(np.mean(np.abs(pred_val - y_val))),
                        "calibration": calibration, "epsilon": epsilon["epsilon"],
                        "fit_info": fit, "fresh_process_restore": restore})
    result = {"schema": SUPPORT_SCHEMA, "status": "complete", "stage": "train-neural",
              "rows": detail, "device": "cuda:0", "models": records}
    _json_write(output / "NEURAL_TRAINING.json", result)
    _json_write(output / "TRAINING_STATUS.json", {"schema": SUPPORT_SCHEMA, "status": "complete", "stage": "train-neural", "optimizer_started": True})
    return result


def _map_public_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("TRAINING_SUPPORT_PUBLIC_STATE_REQUIRED")
    return copy.deepcopy(dict(value))


def _state_rollup(before: Mapping[str, Any], after: Mapping[str, Any], reward: Any) -> dict[str, Any]:
    def mapping_rows(state: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
        return {str(row.get("id")): dict(row) for row in state.get(key, []) if isinstance(row, Mapping) and row.get("id") is not None}

    tasks_before, tasks_after = mapping_rows(before, "tasks"), mapping_rows(after, "tasks")
    platforms_before, platforms_after = mapping_rows(before, "platforms"), mapping_rows(after, "platforms")
    common_tasks = sorted(set(tasks_before) & set(tasks_after))
    common_platforms = sorted(set(platforms_before) & set(platforms_after))
    progress_before = before.get("task_progress", {}) if isinstance(before.get("task_progress"), Mapping) else {}
    progress_after = after.get("task_progress", {}) if isinstance(after.get("task_progress"), Mapping) else {}
    task_progress_delta = []
    for key in common_tasks:
        left, right = tasks_before[key], tasks_after[key]
        lp = float(left.get("progress", left.get("completion_fraction", progress_before.get(key, 0.0))) or 0.0)
        rp = float(right.get("progress", right.get("completion_fraction", progress_after.get(key, 0.0))) or 0.0)
        if math.isfinite(lp) and math.isfinite(rp):
            task_progress_delta.append(rp - lp)
    progress_delta = float(sum(task_progress_delta) / len(task_progress_delta)) if task_progress_delta else 0.0
    time_before = float(before.get("time", 0.0) or 0.0)
    time_after = float(after.get("time", time_before) or time_before)
    def delay_at(task: Mapping[str, Any], now: float) -> float:
        deadline = task.get("deadline")
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
            return 0.0
        completed_at = task.get("completed_at")
        status = str(task.get("status", "")).lower()
        reference = float(completed_at) if isinstance(completed_at, (int, float)) and not isinstance(completed_at, bool) else now
        if status in {"failed", "expired", "cancelled", "split_parent"}:
            return 0.0
        return max(0.0, reference - float(deadline))
    delay_changes = []
    for key in common_tasks:
        before_task, after_task = tasks_before[key], tasks_after[key]
        before_status = str(before_task.get("status", "")).lower()
        after_status = str(after_task.get("status", "")).lower()
        completed_statuses = {"completed", "failed", "expired", "cancelled", "split_parent"}
        before_delay = delay_at(before_task, time_before)
        if after_status == "completed" and not isinstance(after_task.get("completed_at"), (int, float)) and before_status in completed_statuses:
            after_delay = before_delay
        else:
            after_delay = delay_at(after_task, time_after)
        delay_changes.append(after_delay - before_delay)
    delay_delta = float(sum(delay_changes) / max(1, len(delay_changes)))
    energy_delta = float(sum(float(platforms_after[key].get("energy", 0.0) or 0.0) - float(platforms_before[key].get("energy", 0.0) or 0.0) for key in common_platforms))
    failed_before = sum(str(task.get("status", "")).lower() in {"failed", "expired"} for task in tasks_before.values()) + sum(bool(platform.get("failed")) for platform in platforms_before.values())
    failed_after = sum(str(task.get("status", "")).lower() in {"failed", "expired"} for task in tasks_after.values()) + sum(bool(platform.get("failed")) for platform in platforms_after.values())
    comm_before = before.get("communication", [])
    comm_after = after.get("communication", [])
    def disconnected(items: Any) -> float:
        links = [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []
        return sum(not bool(item.get("connected", True)) for item in links) / max(1, len(links))
    comm_risk_delta = disconnected(comm_after) - disconnected(comm_before)
    constraint_event = failed_after > failed_before
    reward_value = float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(float(reward)) else 0.0
    return {
        "next_state": {"values": [time_after - time_before, progress_delta, energy_delta, float(failed_after - failed_before)],
                       "valid": [True, bool(common_tasks), bool(common_platforms), True]},
        "progress": {"values": progress_delta, "valid": bool(common_tasks)},
        "delay": {"values": delay_delta, "valid": bool(common_tasks)},
        "energy": {"values": energy_delta, "valid": bool(common_platforms)},
        "failure": {"values": float(failed_after > failed_before), "valid": True},
        "communication_risk": {"values": float(max(0.0, comm_risk_delta)), "valid": isinstance(comm_before, list) and isinstance(comm_after, list)},
        "constraint_risk": {"values": float(constraint_event), "valid": True},
        "utility": {"values": reward_value, "valid": isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(float(reward))},
    }


def event_samples_from_trajectories(
    trajectories: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    event_models: Any,
    *,
    splits: Sequence[str] = ("train",),
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for marker, trajectory in trajectories:
        trace = trajectory.get("trajectory_trace")
        if not isinstance(trace, list):
            continue
        state_history: list[dict[str, Any]] = []
        parent_id = str(marker.get("scenario_id", ""))
        split = str(marker.get("split", ""))
        if not parent_id or split not in splits:
            continue
        for index, step in enumerate(trace):
            if not isinstance(step, Mapping):
                continue
            before = _map_public_state(step.get("public_before"))
            after = _map_public_state(step.get("public_after"))
            action = step.get("action")
            if not isinstance(action, Mapping):
                continue
            history = [*state_history, {"observation": before}][-event_models.HISTORY_LENGTH:]
            candidate = {"candidate_key": f"step-{index}", "full_environment_action": dict(action)}
            targets = _state_rollup(before, after, step.get("reward"))
            sample = {
                "schema": "n02-event-wm-one-step-sample/1",
                "sample_id": f"{parent_id}-t{index:05d}", "parent_id": parent_id,
                "split": split, "history": history, "candidates": [candidate],
                "targets": {head: {"values": [targets[head]["values"]], "valid": [targets[head]["valid"]]} for head in event_models.WORLD_MODEL_HEADS},
                "target_contract": "observed_adjacent_public_before_after_one_tick_v1",
            }
            samples.append(sample)
            state_history.append({"observation": before})
    return samples


class GainAugmentedWorldModel(nn.Module):
    """Legacy graph/GRU action heads plus a shared-history replanner gain head."""

    def __init__(self, event_models: Any, head_dims: Mapping[str, int], hidden_dim: int = 64):
        super().__init__()
        self.base = event_models.TemporalGraphWorldModel(dict(head_dims), hidden_dim=hidden_dim)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("gain_target_mean", torch.zeros(1))
        self.register_buffer("gain_target_scale", torch.ones(1))
        self.target_contract = "observed_adjacent_public_before_after_one_tick_v1"
        self.gain_contract = "fixed_replanner_oracle_gain_from_public_history_v1"

    def set_gain_normalization(self, mean: float, scale: float) -> None:
        if not math.isfinite(float(mean)) or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("TRAINING_SUPPORT_GAIN_NORMALIZATION")
        self.gain_target_mean.fill_(float(mean))
        self.gain_target_scale.fill_(float(scale))

    def predict_gain(self, history_batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        graph_context = self.base._graph_sequence(dict(history_batch))
        return self.gain_head(graph_context).squeeze(-1) * self.gain_target_scale + self.gain_target_mean

    def predict_gain_normalized(self, history_batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        graph_context = self.base._graph_sequence(dict(history_batch))
        return self.gain_head(graph_context).squeeze(-1)

    def predict_action_outcomes(self, action_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.base(dict(action_batch))

    def forward(self, history_batch: Mapping[str, torch.Tensor], action_batch: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        output = {"gain": self.predict_gain(history_batch)}
        if action_batch is not None:
            output.update({f"action_{key}": value for key, value in self.predict_action_outcomes(action_batch).items()})
        return output


def collate_public_histories(histories: Sequence[Sequence[Mapping[str, Any]]], event_models: Any) -> dict[str, torch.Tensor]:
    """Build graph/GRU input without constructing or encoding candidates."""
    encoded = [event_models._encode_sample_graph({"history": list(history)}) for history in histories]
    batch_size = len(encoded)
    steps = event_models.HISTORY_LENGTH
    max_nodes = max(len(encoded_row[0][time_index]) for encoded_row in encoded for time_index in range(steps))
    node_features = torch.zeros((batch_size, steps, max_nodes, event_models.NODE_FEATURE_DIM), dtype=torch.float32)
    node_types = torch.zeros((batch_size, steps, max_nodes), dtype=torch.long)
    node_mask = torch.zeros((batch_size, steps, max_nodes), dtype=torch.bool)
    adjacency = torch.zeros((batch_size, steps, event_models.RELATION_COUNT, max_nodes, max_nodes), dtype=torch.float32)
    history_lengths = torch.zeros((batch_size,), dtype=torch.long)
    for batch_index, (feature_steps, type_steps, edge_steps, lengths) in enumerate(encoded):
        history_lengths[batch_index] = int(lengths[0])
        for time_index in range(steps):
            count = len(feature_steps[time_index])
            node_features[batch_index, time_index, :count] = torch.tensor(feature_steps[time_index], dtype=torch.float32)
            node_types[batch_index, time_index, :count] = torch.tensor(type_steps[time_index], dtype=torch.long)
            node_mask[batch_index, time_index, :count] = True
            adjacency[batch_index, time_index, :, :count, :count] = torch.tensor(edge_steps[time_index], dtype=torch.float32)
    return {"node_features": node_features, "node_types": node_types, "node_mask": node_mask,
            "adjacency": adjacency, "history_lengths": history_lengths}


def gain_history_samples(rows: Sequence[Mapping[str, Any]], *, splits: Sequence[str] = ("train",)) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        history = row.get("public_history")
        if row.get("split") not in splits or not isinstance(history, list) or not history:
            continue
        samples.append({"sample_id": str(row["state_id"]), "parent_id": str(row["scenario_id"]),
                        "history": history, "gain_target": float(row["gain_mean"]),
                        "target_source": "fixed_replanner_oracle"})
    return samples


def _event_action_loss(event_models: Any, model: GainAugmentedWorldModel, batch: Mapping[str, torch.Tensor], packed: Mapping[str, torch.Tensor]) -> torch.Tensor:
    predictions = model.predict_action_outcomes(batch)
    losses = []
    for head in event_models.WORLD_MODEL_HEADS:
        width = model.base.config["head_dims"][head]
        target = packed[head][..., :width]
        valid = packed[head][..., width:].bool()
        if head not in event_models.PROBABILITY_HEADS:
            target = (target - getattr(model.base, f"target_mean_{head}")) / getattr(model.base, f"target_scale_{head}")
        head_loss = event_models._loss(predictions[head], target, valid, head)
        if head_loss is not None:
            losses.append(head_loss)
    if not losses:
        raise ValueError("TRAINING_SUPPORT_EVENT_BATCH_WITHOUT_VALID_LABELS")
    return torch.stack(losses).mean()


def _action_slot_audit(trajectories: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], event_models: Any) -> dict[str, Any]:
    maxima = {"dispatches": 0, "queue_operations": 0}
    observed = {"dispatches": 0, "queue_operations": 0}
    over_limit = {"dispatches": 0, "queue_operations": 0}
    limits = {"dispatches": int(event_models.CANDIDATE_DISPATCH_SLOTS),
              "queue_operations": int(event_models.CANDIDATE_OPERATION_SLOTS)}
    for _, trajectory in trajectories:
        trace = trajectory.get("trajectory_trace")
        if not isinstance(trace, list):
            continue
        for step in trace:
            action = step.get("action") if isinstance(step, Mapping) else None
            if not isinstance(action, Mapping):
                continue
            for key, action_key in (("dispatches", "dispatches"), ("queue_operations", "queue_operations")):
                value = action.get(action_key, [])
                count = len(value) if isinstance(value, list) else 0
                maxima[key] = max(maxima[key], count)
                observed[key] += 1
                over_limit[key] += int(count > limits[key])
    return {"slot_limits": limits, "action_count_maxima": maxima,
            "action_count_observations": observed, "over_limit_action_counts": over_limit,
            "encoder_behavior": "reject_over_limit_no_truncation"}


def _gain_batch_loss(model: GainAugmentedWorldModel, samples: Sequence[Mapping[str, Any]], indices: Sequence[int], event_models: Any, device: torch.device) -> torch.Tensor:
    selected = [samples[index] for index in indices]
    batch = collate_public_histories([row["history"] for row in selected], event_models)
    batch = {key: value.to(device) for key, value in batch.items()}
    target = torch.tensor([row["gain_target"] for row in selected], dtype=torch.float32, device=device)
    normalized_target = (target - model.gain_target_mean) / model.gain_target_scale
    prediction = model.predict_gain_normalized(batch)
    return F.huber_loss(prediction, normalized_target, delta=1.0)


def _validation_loss(
    model: GainAugmentedWorldModel,
    event_models: Any,
    gain_samples: Sequence[Mapping[str, Any]],
    action_samples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    gain_total = action_total = 0.0
    gain_count = action_count = 0
    with torch.no_grad():
        for start in range(0, len(gain_samples), EVENT_BATCH_SIZE):
            selected = gain_samples[start:start + EVENT_BATCH_SIZE]
            batch = collate_public_histories([row["history"] for row in selected], event_models)
            batch = {key: value.to(device) for key, value in batch.items()}
            target = torch.tensor([row["gain_target"] for row in selected], dtype=torch.float32, device=device)
            normalized_target = (target - model.gain_target_mean) / model.gain_target_scale
            loss = F.huber_loss(model.predict_gain_normalized(batch), normalized_target, delta=1.0)
            gain_total += float(loss.item()) * len(selected)
            gain_count += len(selected)
        for start in range(0, len(action_samples), EVENT_BATCH_SIZE):
            selected = list(action_samples[start:start + EVENT_BATCH_SIZE])
            batch, packed = event_models._collate_samples(selected)
            batch = {key: value.to(device) for key, value in batch.items()}
            packed = {key: value.to(device) for key, value in packed.items()}
            loss = _event_action_loss(event_models, model, batch, packed)
            action_total += float(loss.item()) * len(selected)
            action_count += len(selected)
    if not gain_count or not action_count:
        raise ValueError("TRAINING_SUPPORT_JOINT_VALIDATION_SPLIT_EMPTY")
    gain_loss = gain_total / gain_count
    action_loss = action_total / action_count
    return {"gain_huber": gain_loss, "action_masked": action_loss, "joint": 0.5 * (gain_loss + action_loss)}


def _sample_content_hash(samples: Sequence[Mapping[str, Any]], *, gain: bool) -> str:
    if gain:
        payload = [{"sample_id": row["sample_id"], "parent_id": row["parent_id"],
                    "history": row["history"], "gain_target": row["gain_target"]} for row in samples]
    else:
        payload = [{"sample_id": row["sample_id"], "parent_id": row["parent_id"],
                    "history": row["history"], "candidates": row["candidates"], "targets": row["targets"]} for row in samples]
    return _json_sha256(payload)


def _augmented_predictions(
    model: GainAugmentedWorldModel,
    event_models: Any,
    gain_histories: Sequence[Sequence[Mapping[str, Any]]],
    action_samples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    with torch.no_grad():
        history_batch = collate_public_histories(gain_histories, event_models)
        history_batch = {key: value.to(device) for key, value in history_batch.items()}
        result["gain"] = model.predict_gain(history_batch).detach().cpu().tolist()
        if action_samples:
            action_batch, _ = event_models._collate_samples(list(action_samples))
            action_batch = {key: value.to(device) for key, value in action_batch.items()}
            result["action"] = {head: value.detach().cpu().tolist() for head, value in model.predict_action_outcomes(action_batch).items()}
        else:
            result["action"] = {}
    return result


def _load_augmented_checkpoint(checkpoint_path: str | Path, legacy_source: str | Path, device: torch.device) -> tuple[GainAugmentedWorldModel, torch.optim.Optimizer, dict[str, Any]]:
    _, event_models, _ = _load_legacy(legacy_source)
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "n02-gain-augmented-world-model-checkpoint/1":
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_SCHEMA")
    required = ("model_state", "model_state_sha256", "optimizer_state", "optimizer_state_sha256", "normalization", "gain_normalization", "data_hashes", "split_identity", "training_config", "target_contracts", "torch_rng_state", "cuda_rng_states", "numpy_rng_state", "python_rng_state", "batch_generator_state")
    if not all(key in payload for key in required):
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_METADATA")
    if _torch_payload_sha256(payload["optimizer_state"]) != payload["optimizer_state_sha256"]:
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_OPTIMIZER_HASH")
    model = GainAugmentedWorldModel(event_models, payload["head_dims"], hidden_dim=int(payload["hidden_dim"]))
    model.base.set_normalization(payload["normalization"])
    model.set_gain_normalization(*payload["gain_normalization"])
    model.load_state_dict(payload["model_state"], strict=True)
    if _torch_state_sha256(model.state_dict()) != payload.get("model_state_sha256"):
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_STATE_HASH")
    model.to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(payload["training_config"]["learning_rate"]))
    optimizer.load_state_dict(payload["optimizer_state"])
    if int(payload.get("optimizer_updates_at_best", 0)) < 1:
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_OPTIMIZER_EMPTY")
    return model, optimizer, payload


def _restore_augmented_cuda_probe(request_path: str | Path) -> dict[str, Any]:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("TRAINING_SUPPORT_AUGMENTED_RESTORE_CUDA_UNAVAILABLE")
    checkpoint_path = Path(request["checkpoint_path"]).resolve(strict=True)
    if _sha256(checkpoint_path) != request["checkpoint_sha256"]:
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_CHECKPOINT_FILE_HASH")
    model, optimizer, payload = _load_augmented_checkpoint(checkpoint_path, request["legacy_source"], device)
    predictions = _augmented_predictions(model, _load_legacy(request["legacy_source"])[1], request["gain_histories"], request["action_samples"], device)
    optimizer_devices: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                name = str(value.device)
                optimizer_devices[name] = optimizer_devices.get(name, 0) + 1
    if not any(name.startswith("cuda:0") for name in optimizer_devices):
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_OPTIMIZER_NOT_CUDA")
    return {"schema": "n02-gain-augmented-restore-probe/1", "predictions": predictions,
            "checkpoint_valid": True, "optimizer_state_restored": True,
            "optimizer_updates_at_best": int(payload["optimizer_updates_at_best"]),
            "inference_device": str(device), "gpu_name": torch.cuda.get_device_name(0),
            "optimizer_tensor_devices": optimizer_devices,
            "model_state_sha256": payload["model_state_sha256"]}


def fresh_process_augmented_cuda_restore(
    checkpoint_path: str | Path,
    legacy_source: str | Path,
    gain_histories: Sequence[Sequence[Mapping[str, Any]]],
    action_samples: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    device = torch.device("cuda:0")
    model, _, payload = _load_augmented_checkpoint(checkpoint_path, legacy_source, device)
    event_models = _load_legacy(legacy_source)[1]
    expected = _augmented_predictions(model, event_models, gain_histories, action_samples, device)
    request_path = checkpoint_path.parent / f"restore_augmented_request_{os.getpid()}.json"
    _json_write(request_path, {
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "legacy_source": str(Path(legacy_source).resolve()),
        "gain_histories": [list(history) for history in gain_histories],
        "action_samples": list(action_samples),
    })
    try:
        process = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--restore-augmented-cuda", str(request_path)],
            cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, timeout=240,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, check=False,
        )
    finally:
        request_path.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(f"TRAINING_SUPPORT_AUGMENTED_FRESH_PROCESS_FAILED:{process.stderr[-1500:]}")
    actual = json.loads(process.stdout.strip().splitlines()[-1])
    expected_values = [*expected["gain"], *[value for head in sorted(expected["action"]) for row in expected["action"][head] for value in (row if isinstance(row, list) else [row])]]
    actual_values = [*actual["predictions"]["gain"], *[value for head in sorted(actual["predictions"]["action"]) for row in actual["predictions"]["action"][head] for value in (row if isinstance(row, list) else [row])]]
    if len(expected_values) != len(actual_values) or not np.allclose(expected_values, actual_values, rtol=tolerance, atol=tolerance):
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_FRESH_PROCESS_PREDICTION_MISMATCH")
    if actual.get("checkpoint_valid") is not True or actual.get("optimizer_state_restored") is not True or actual.get("inference_device") != "cuda:0":
        raise ValueError("TRAINING_SUPPORT_AUGMENTED_FRESH_PROCESS_RESTORE_INVALID")
    return {"fresh_process": True, "same_input_predictions_match": True,
            "model_state_sha256": payload["model_state_sha256"], **actual}


def _train_augmented_seed(
    train_gain: Sequence[Mapping[str, Any]],
    train_action: Sequence[Mapping[str, Any]],
    validation_gain: Sequence[Mapping[str, Any]],
    validation_action: Sequence[Mapping[str, Any]],
    event_models: Any,
    *,
    seed: int,
    hidden_dim: int,
    head_dims: Mapping[str, int],
    normalization: Mapping[str, Any],
    gain_normalization: tuple[float, float],
    data_hashes: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    model = GainAugmentedWorldModel(event_models, head_dims, hidden_dim=hidden_dim)
    model.base.set_normalization(dict(normalization))
    model.set_gain_normalization(*gain_normalization)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=EVENT_LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    best_loss = math.inf
    best_state = best_optimizer = best_rng = None
    best_epoch = best_updates = updates = stale = 0
    history = []
    resource_timer = _cuda_training_timer_start()
    for epoch in range(1, EVENT_MAX_EPOCHS + 1):
        model.train()
        gain_order = torch.randperm(len(train_gain), generator=generator).tolist()
        action_order = torch.randperm(len(train_action), generator=generator).tolist()
        steps = max(math.ceil(len(train_gain) / EVENT_BATCH_SIZE), math.ceil(len(train_action) / EVENT_BATCH_SIZE))
        epoch_loss_sum = 0.0
        for batch_index in range(steps):
            start = batch_index * EVENT_BATCH_SIZE
            count = min(EVENT_BATCH_SIZE, max(len(train_gain), len(train_action)))
            gain_indices = [gain_order[(start + offset) % len(gain_order)] for offset in range(count)]
            action_indices = [action_order[(start + offset) % len(action_order)] for offset in range(count)]
            losses = [
                _gain_batch_loss(model, train_gain, gain_indices, event_models, device),
            ]
            action_rows = [train_action[index] for index in action_indices]
            action_batch, action_targets = event_models._collate_samples(action_rows, dict(head_dims))
            action_batch = {key: value.to(device) for key, value in action_batch.items()}
            action_targets = {key: value.to(device) for key, value in action_targets.items()}
            losses.append(_event_action_loss(event_models, model, action_batch, action_targets))
            joint_loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(joint_loss)):
                raise FloatingPointError("TRAINING_SUPPORT_JOINT_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            joint_loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError("TRAINING_SUPPORT_JOINT_NONFINITE_GRADIENT")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            epoch_loss_sum += float(joint_loss.detach().item())
        validation = _validation_loss(model, event_models, validation_gain, validation_action, device)
        history.append({"epoch": epoch, "train_joint_loss": epoch_loss_sum / max(1, steps), **validation, "optimizer_updates": steps})
        if validation["joint"] < best_loss - 1e-12:
            best_loss, best_epoch, best_updates, stale = validation["joint"], epoch, updates, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_rng = {
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": [state.clone() for state in torch.cuda.get_rng_state_all()],
                "numpy_rng_state": copy.deepcopy(np.random.get_state()),
                "python_rng_state": copy.deepcopy(random.getstate()),
                "batch_generator_state": generator.get_state().clone(),
            }
        else:
            stale += 1
            if stale >= EVENT_PATIENCE:
                break
    if best_state is None or best_optimizer is None or best_rng is None:
        raise RuntimeError("TRAINING_SUPPORT_JOINT_NO_BEST_CHECKPOINT")
    resource_usage = _cuda_training_timer_stop(resource_timer)
    model.load_state_dict(best_state, strict=True)
    optimizer.load_state_dict(best_optimizer)
    model.eval()
    state_digest = _torch_state_sha256(model.state_dict())
    training_config = {
        "seeds": list(EVENT_SEEDS), "learning_rate": EVENT_LEARNING_RATE,
        "batch_size": EVENT_BATCH_SIZE, "max_epochs": EVENT_MAX_EPOCHS,
        "patience": EVENT_PATIENCE, "hidden_dim": hidden_dim,
        "loss": "mean(Huber(gain),mean(masked_one_step_action_heads))",
        "checkpoint_selection_split": "validation_model", "device": "cuda:0",
        "resource_usage": resource_usage,
    }
    optimizer_state_sha256 = _torch_payload_sha256(best_optimizer)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "n02-gain-augmented-world-model-checkpoint/1", "seed": seed,
        "head_dims": dict(head_dims), "hidden_dim": hidden_dim,
        "model_state": best_state, "model_state_sha256": state_digest,
        "optimizer_state": best_optimizer, "optimizer_state_sha256": optimizer_state_sha256,
        "optimizer_updates_at_best": best_updates, "total_optimizer_updates": updates,
        "best_epoch": best_epoch, "validation_best_loss": best_loss,
        "normalization": copy.deepcopy(dict(normalization)),
        "gain_normalization": [float(gain_normalization[0]), float(gain_normalization[1])],
        "data_hashes": copy.deepcopy(dict(data_hashes)),
        "split_identity": copy.deepcopy(dict(split_identity)),
        "training_config": training_config,
        "target_contracts": {
            "gain": "fixed_replanner_oracle_gain_from_public_history_v1",
            "action": "observed_adjacent_public_before_after_one_tick_v1",
        },
        **best_rng,
    }, checkpoint_path)
    return {
        "seed": seed, "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path), "model_state_sha256": state_digest,
        "optimizer_state_sha256": optimizer_state_sha256,
        "optimizer_updates_at_best": best_updates, "optimizer_updates_total": updates,
        "best_epoch": best_epoch, "epochs_completed": epoch,
        "best_validation_joint_loss": best_loss, "training_history": history,
        "training_config": training_config,
    }


def train_event_wm_cuda(
    shard_root: str | Path,
    legacy_source: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        _write_not_run(output_dir, "CUDA_UNAVAILABLE", {"optimizer_started": False})
        raise RuntimeError("TRAINING_SUPPORT_CUDA_REQUIRED")
    gm, event_models, shard_api = _load_legacy(legacy_source)
    gain_rows, gate_detail = collect_gain_training_rows(shard_root, legacy_source, output_dir)
    shards = _committed_shards(shard_root, shard_api)
    _, trajectories = _read_committed_rows(shards)
    action_slot_detail = _action_slot_audit(trajectories, event_models)
    validation_rows = [row for row in gain_rows if row["split"] == "validation"]
    validation_model_rows, validation_calibration_rows = gm._validation_parts(validation_rows, 8201)
    validation_model_ids = {str(row["state_id"]) for row in validation_model_rows}
    validation_calibration_ids = {str(row["state_id"]) for row in validation_calibration_rows}
    validation_model_parents = {str(row["scenario_id"]) for row in validation_model_rows}
    validation_calibration_parents = {str(row["scenario_id"]) for row in validation_calibration_rows}
    train_gain = gain_history_samples(gain_rows, splits=("train",))
    validation_gain_all = gain_history_samples(gain_rows, splits=("validation",))
    validation_gain_model = [row for row in validation_gain_all if row["sample_id"] in validation_model_ids]
    validation_gain_calibration = [row for row in validation_gain_all if row["sample_id"] in validation_calibration_ids]
    train_action = event_samples_from_trajectories(trajectories, event_models, splits=("train",))
    validation_action_all = event_samples_from_trajectories(trajectories, event_models, splits=("validation",))
    validation_action_model = [row for row in validation_action_all if row["parent_id"] in validation_model_parents]
    validation_action_calibration = [row for row in validation_action_all if row["parent_id"] in validation_calibration_parents]
    counts = {
        "train_gain": len(train_gain), "train_action": len(train_action),
        "validation_model_gain": len(validation_gain_model), "validation_model_action": len(validation_action_model),
        "validation_calibration_gain": len(validation_gain_calibration), "validation_calibration_action": len(validation_action_calibration),
    }
    required_counts = ("train_gain", "train_action", "validation_model_gain", "validation_model_action", "validation_calibration_gain")
    over_limit = any(action_slot_detail["over_limit_action_counts"].values())
    if any(counts[key] < 1 for key in required_counts) or over_limit:
        reason = "ACTION_FEATURE_SLOT_LIMIT_EXCEEDED" if over_limit else "INSUFFICIENT_JOINT_TRAIN_VALIDATION_OR_CALIBRATION_ROWS"
        _write_not_run(output_dir, reason, {
            "gain_state_gate": gate_detail["gain_state_gate"], "sample_counts": counts,
            "required_nonempty_groups": list(required_counts),
            "minimum_per_polarity_global_gain_states": MIN_GAIN_CLASS_STATES, "optimizer_started": False,
            "action_slot_audit": action_slot_detail,
        })
        raise ValueError("TRAINING_SUPPORT_JOINT_ROWS_INSUFFICIENT")

    head_dims = event_models._target_dimensions(train_action)
    normalization = event_models._normalize_statistics(train_action, head_dims)
    gain_values = np.asarray([row["gain_target"] for row in train_gain], dtype=np.float64)
    gain_mean = float(gain_values.mean())
    gain_scale = float(gain_values.std())
    if gain_scale < 1e-12:
        gain_scale = 1.0
    split_identity = {
        "train_gain_state_ids_sha256": _json_sha256(sorted(row["sample_id"] for row in train_gain)),
        "train_action_sample_ids_sha256": _json_sha256(sorted(row["sample_id"] for row in train_action)),
        "validation_model_gain_state_ids_sha256": _json_sha256(sorted(row["sample_id"] for row in validation_gain_model)),
        "validation_model_action_parent_ids_sha256": _json_sha256(sorted(validation_model_parents)),
        "validation_calibration_gain_state_ids_sha256": _json_sha256(sorted(row["sample_id"] for row in validation_gain_calibration)),
        "validation_calibration_action_parent_ids_sha256": _json_sha256(sorted(validation_calibration_parents)),
        "test_accessed": False,
    }
    source_hashes = gate_detail["data_hashes"]
    data_hashes = {
        **source_hashes,
        "train_gain_examples_sha256": _sample_content_hash(train_gain, gain=True),
        "train_action_examples_sha256": _sample_content_hash(train_action, gain=False),
        "validation_model_gain_examples_sha256": _sample_content_hash(validation_gain_model, gain=True),
        "validation_model_action_examples_sha256": _sample_content_hash(validation_action_model, gain=False),
        "validation_calibration_gain_examples_sha256": _sample_content_hash(validation_gain_calibration, gain=True),
        "validation_calibration_action_examples_sha256": _sample_content_hash(validation_action_calibration, gain=False),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_records = []
    restore_inputs_gain = [row["history"] for row in validation_gain_model[:2]]
    restore_inputs_action = validation_action_model[:2]
    for seed in EVENT_SEEDS:
        record = _train_augmented_seed(
            train_gain, train_action, validation_gain_model, validation_action_model,
            event_models, seed=seed, hidden_dim=64, head_dims=head_dims,
            normalization=normalization, gain_normalization=(gain_mean, gain_scale),
            data_hashes=data_hashes, split_identity=split_identity,
            checkpoint_path=output / f"world_model_gain_seed_{seed}.pt",
        )
        record["fresh_process_restore"] = fresh_process_augmented_cuda_restore(
            record["checkpoint_path"], legacy_source, restore_inputs_gain, restore_inputs_action,
        )
        model, _, _ = _load_augmented_checkpoint(record["checkpoint_path"], legacy_source, torch.device("cuda:0"))
        calibration_predictions = []
        for start in range(0, len(validation_gain_calibration), EVENT_BATCH_SIZE):
            selected = validation_gain_calibration[start:start + EVENT_BATCH_SIZE]
            batch_prediction = _augmented_predictions(
                model, event_models, [row["history"] for row in selected], [], torch.device("cuda:0"),
            )
            calibration_predictions.extend(batch_prediction["gain"])
        calibration_truth = [row["gain_target"] for row in validation_gain_calibration]
        gain_calibration = gm.calibrate_lower_bound(calibration_truth, calibration_predictions, 0.9)
        record["gain_calibration"] = {
            **gain_calibration, "calibration_split": "validation_calibration",
            "calibration_sample_count": len(calibration_truth),
            "selection_split": "validation_model",
            "selected_checkpoint_epoch": record["best_epoch"],
            "selected_checkpoint_validation_joint_loss": record["best_validation_joint_loss"],
        }
        checkpoint_records.append(record)
    result = {
        "schema": "n02-full-gain-augmented-wm-training/1", "status": "complete",
        "training_device": "cuda:0", "sample_counts": counts,
        "action_slot_audit": action_slot_detail,
        "gain_state_gate": gate_detail["gain_state_gate"],
        "target_contracts": {
            "gain": "fixed_replanner_oracle_gain_from_public_history_v1",
            "action": "observed_adjacent_public_before_after_one_tick_v1",
        },
        "shared_encoder": True, "gain_head_candidate_free_inference": True,
        "normalization": {"action_target": "train_action_samples_only", "gain_target": "train_gain_rows_only"},
        "data_hashes": data_hashes, "split_identity": split_identity,
        "head_dims": head_dims,
        "training_config": {"seeds": list(EVENT_SEEDS), "hidden_dim": 64, "learning_rate": EVENT_LEARNING_RATE,
                            "batch_size": EVENT_BATCH_SIZE, "max_epochs": EVENT_MAX_EPOCHS,
                            "patience": EVENT_PATIENCE, "selection_split": "validation_model",
                            "calibration_split": "validation_calibration", "test_accessed": False},
        "checkpoints": checkpoint_records,
    }
    _json_write(output / "FULL_WM_TRAINING.json", result)
    _json_write(output / "TRAINING_STATUS.json", {"schema": SUPPORT_SCHEMA, "status": "complete", "stage": "train-event-wm", "optimizer_started": True})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--restore-gain-cuda":
        result = _restore_gain_cuda_probe(argv[1])
        print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
        return 0
    if len(argv) == 2 and argv[0] == "--restore-augmented-cuda":
        result = _restore_augmented_cuda_probe(argv[1])
        print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description="External N02 training support; no smoke-source writes.")
    parser.add_argument("stage", choices=("train-cpu", "train-neural", "train-event-wm"))
    parser.add_argument("--legacy-source", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    functions = {"train-cpu": train_gain_cpu, "train-neural": train_gain_neural_cuda, "train-event-wm": train_event_wm_cuda}
    result = functions[args.stage](args.shard_root, args.legacy_source, args.output_dir)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
