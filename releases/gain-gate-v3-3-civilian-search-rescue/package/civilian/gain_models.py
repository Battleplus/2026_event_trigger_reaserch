"""Public-feature gain predictors and calibration for the N02 gain gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_project_dependencies() -> None:
    for candidate in (
        PROJECT_ROOT / ".deps",
        PROJECT_ROOT / ".deps" / "Lib" / "site-packages",
        PROJECT_ROOT / ".deps" / "site-packages",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_project_dependencies()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = "gain-model-bundle/1"
TRAINING_SCHEMA = "gain-model-training/1"
MLP_SEEDS = (8201, 8202, 8203)
FORBIDDEN_FEATURE_PARTS = frozenset({
    "future", "truth", "label", "gain", "hash", "snapshot", "tape", "seed",
    "scenario", "parent", "identity", "replanner", "candidate_catalog", "oracle",
})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GAIN_MODEL_JSON_OBJECT_REQUIRED")
    return value


def _feature_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("GAIN_MODEL_PUBLIC_FEATURES_REQUIRED")
    result: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        parts = set(name.lower().split("_"))
        if parts & FORBIDDEN_FEATURE_PARTS or any(part in name.lower() for part in ("future", "snapshot", "tape_hash")):
            raise ValueError(f"GAIN_MODEL_PRIVATE_FEATURE_REJECTED:{name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"GAIN_MODEL_NUMERIC_FEATURE_REQUIRED:{name}")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError(f"GAIN_MODEL_NONFINITE_FEATURE:{name}")
        result[name] = number
    return dict(sorted(result.items()))


def _row_features(row: Mapping[str, Any]) -> dict[str, float]:
    features = row.get("public_features", row.get("features"))
    return _feature_map(features)


def _row_gain(row: Mapping[str, Any]) -> float:
    raw = row.get("gain_mean", row.get("gain"))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError("GAIN_MODEL_FINITE_GAIN_REQUIRED")
    return float(raw)


def _is_valid(row: Mapping[str, Any]) -> bool:
    return bool(row.get("valid_mask", row.get("valid", True)))


def _is_forced(row: Mapping[str, Any]) -> bool:
    return bool(row.get("forced", False))


def _row_split(row: Mapping[str, Any]) -> str:
    split = str(row.get("split", ""))
    return {"val": "validation", "development": "validation"}.get(split, split)


def _group_id(row: Mapping[str, Any]) -> str:
    value = row.get("scenario_id", row.get("scenario_instance_id", row.get("parent_id")))
    if value is None or not str(value):
        raise ValueError("GAIN_MODEL_SCENARIO_GROUP_REQUIRED")
    return str(value)


_LINEAGE_ID_FIELDS = frozenset({
    "scenario_content_sha256", "scenario_definition_sha256", "scenario_sha256",
    "scenario_hash", "scenario_instance_hash", "event_tape_id", "event_tape_sha256",
    "event_tape_hash", "trajectory_id", "trajectory_sha256", "trajectory_hash",
    "state_id", "state_sha256", "state_hash", "snapshot_sha256",
    "private_snapshot_sha256", "source_snapshot_sha256", "public_history_sha256",
    "lineage_id", "lineage_sha256", "parent_instance_id",
})
_LINEAGE_SEED_FIELDS = frozenset({
    "scenario_seed", "scenario_instance_seed", "instance_seed", "generation_seed",
    "trajectory_seed", "suffix_seed", "seed",
})
_SCENARIO_IDENTITY_FIELDS = frozenset({
    "id", "parent_id", "parent_family", "scenario_id", "scenario_instance_id",
    "split", "seed", "seed_variant", "scenario_seed", "scenario_instance_seed",
    "instance_seed", "generation_seed", "trajectory_seed", "event_tape_id",
})


def _scenario_content_sha256(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    content = {
        str(key): item for key, item in value.items()
        if str(key).lower() not in _SCENARIO_IDENTITY_FIELDS
    }
    return _sha256({"scenario_content": content})


def _lineage_tokens(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    tokens: set[tuple[str, str]] = set()
    family = str(row.get("parent_family", ""))
    condition = str(row.get("condition", ""))

    def add(field: str, value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(field, item)
            return
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return
        text = str(value).strip()
        if text:
            namespace = f"seed:{family}:{condition}" if field in _LINEAGE_SEED_FIELDS else field
            tokens.add((namespace, text))

    for field in _LINEAGE_ID_FIELDS | _LINEAGE_SEED_FIELDS:
        if field in row:
            add(field, row[field])

    for field in ("scenario", "scenario_content", "scenario_definition", "scenario_payload", "raw_scenario"):
        if field in row:
            digest = _scenario_content_sha256(row[field])
            if digest is not None:
                tokens.add(("scenario_content_sha256", digest))

    lineage = row.get("lineage")
    if isinstance(lineage, Mapping):
        for field in _LINEAGE_ID_FIELDS | _LINEAGE_SEED_FIELDS:
            if field in lineage:
                add(field, lineage[field])

    scenario_instance_id = row.get("scenario_instance_id")
    if scenario_instance_id is not None:
        add("scenario_instance_id", scenario_instance_id)
    return tokens


def check_group_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject scenario groups or shared data lineage spanning multiple splits."""
    assignments: dict[str, set[str]] = {}
    lineage_assignments: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        split = _row_split(row)
        if not split:
            raise ValueError("GAIN_MODEL_SPLIT_REQUIRED")
        assignments.setdefault(_group_id(row), set()).add(split)
        for token in _lineage_tokens(row):
            lineage_assignments.setdefault(token, set()).add(split)
    leaked = sorted(group for group, splits in assignments.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"GAIN_MODEL_GROUP_SPLIT_LEAKAGE:{','.join(leaked[:5])}")
    lineage_leaks = sorted(token for token, splits in lineage_assignments.items() if len(splits) != 1)
    if lineage_leaks:
        fields = sorted({field for field, _ in lineage_leaks})
        raise ValueError(f"GAIN_MODEL_LINEAGE_SPLIT_LEAKAGE:{','.join(fields[:8])}")
    return {
        "scenario_group_count": len(assignments), "lineage_key_count": len(lineage_assignments),
        "split_leakage_count": 0,
    }


def _eligible(rows: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if _row_split(row) == split and _is_valid(row) and not _is_forced(row)]


def _validation_parts(rows: Sequence[Mapping[str, Any]], seed: int = 8201) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split whole validation scenario instances into model and calibration sets."""
    explicit_model = [row for row in rows if _row_split(row) == "validation_model"]
    explicit_calibration = [row for row in rows if _row_split(row) == "validation_calibration"]
    validation = [row for row in rows if _row_split(row) == "validation"]
    if explicit_model or explicit_calibration:
        if not explicit_model or not explicit_calibration or validation:
            raise ValueError("GAIN_MODEL_VALIDATION_SPLIT_INCOMPLETE")
        return explicit_model, explicit_calibration
    if not validation:
        raise ValueError("GAIN_MODEL_VALIDATION_ROWS_REQUIRED")
    by_family: dict[str, list[str]] = {}
    for row in validation:
        family = str(row.get("parent_family", row.get("condition", "all")))
        group = _group_id(row)
        by_family.setdefault(family, [])
        if group not in by_family[family]:
            by_family[family].append(group)
    model_groups: set[str] = set()
    calibration_groups: set[str] = set()
    for family, groups in sorted(by_family.items()):
        ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{family}:{group}".encode("utf-8")).hexdigest())
        if len(ordered) < 2:
            raise ValueError(f"GAIN_MODEL_NEED_TWO_VALIDATION_GROUPS_PER_FAMILY:{family}")
        model_groups.update(ordered[::2])
        calibration_groups.update(ordered[1::2])
    if not model_groups or not calibration_groups:
        raise ValueError("GAIN_MODEL_VALIDATION_PARTITION_EMPTY")
    return (
        [row for row in validation if _group_id(row) in model_groups],
        [row for row in validation if _group_id(row) in calibration_groups],
    )


def _xy(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    features, gains = [], []
    for row in rows:
        values = _row_features(row)
        if tuple(values) != tuple(names):
            raise ValueError("GAIN_MODEL_FEATURE_SCHEMA_MISMATCH")
        features.append([values[name] for name in names])
        gains.append(_row_gain(row))
    if not features:
        raise ValueError("GAIN_MODEL_EMPTY_SPLIT")
    return np.asarray(features, dtype=np.float64), np.asarray(gains, dtype=np.float64)


def _fit_normalizer(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("GAIN_MODEL_TRAIN_ROWS_REQUIRED")
    maps = [_row_features(row) for row in rows]
    names = list(maps[0])
    if any(list(item) != names for item in maps):
        raise ValueError("GAIN_MODEL_FEATURE_SCHEMA_MISMATCH")
    matrix = np.asarray([[item[name] for name in names] for item in maps], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {"feature_names": names, "mean": mean.tolist(), "scale": scale.tolist(), "fit_row_count": len(rows), "fit_split": "train"}


def _transform(matrix: np.ndarray, normalizer: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["scale"], dtype=np.float64)
    return (matrix - mean) / scale


def _epsilon_candidates(config: Mapping[str, Any]) -> list[float]:
    values = config.get("epsilon_grid", [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
    if not isinstance(values, list) or not values:
        raise ValueError("GAIN_MODEL_EPSILON_GRID_REQUIRED")
    result = sorted({float(value) for value in values})
    if any(not math.isfinite(value) for value in result):
        raise ValueError("GAIN_MODEL_EPSILON_GRID_NONFINITE")
    return result


def select_epsilon(y_true: Sequence[float], predictions: Sequence[float], grid: Sequence[float]) -> dict[str, Any]:
    """Choose a fixed mean-gain threshold using model-selection validation only."""
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("GAIN_MODEL_EPSILON_SELECTION_SHAPE")
    choices = []
    for epsilon in grid:
        allowed = predicted > float(epsilon)
        realized = np.where(allowed, truth, 0.0)
        choices.append({
            "epsilon": float(epsilon), "mean_realized_gain": float(realized.mean()),
            "accepted_fraction": float(allowed.mean()),
        })
    best = max(choices, key=lambda row: (row["mean_realized_gain"], row["epsilon"]))
    return {"epsilon": best["epsilon"], "selection_metric": "mean_realized_gain_with_zero_for_suppressed", "choices": choices}


def calibrate_lower_bound(y_true: Sequence[float], predictions: Sequence[float], coverage: float = 0.9) -> dict[str, Any]:
    """Calibrate q for the empirical lower bound prediction - q."""
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("GAIN_MODEL_CALIBRATION_SHAPE")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("GAIN_MODEL_CALIBRATION_COVERAGE")
    residuals = np.sort(predicted - truth)
    rank = int(math.ceil((len(residuals) + 1) * float(coverage)))
    if rank > len(residuals):
        q: float = math.inf
    else:
        q = float(residuals[max(0, rank - 1)])
    return {
        "coverage_target": float(coverage), "q": q if math.isfinite(q) else None,
        "q_finite": math.isfinite(q), "calibration_count": int(len(residuals)),
        "residual_definition": "prediction_minus_observed_gain",
        "finite_sample_rank": rank,
        "claim_limit": "empirical_calibration_under_exchangeability_assumption; not a safety guarantee",
    }


class _GainMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.gain_head = nn.Linear(64, 1)
        self.positive_head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.gain_head(hidden).squeeze(-1), self.positive_head(hidden).squeeze(-1)


def _state_dict_json(model: nn.Module) -> dict[str, list]:
    return {key: tensor.detach().cpu().tolist() for key, tensor in model.state_dict().items()}


def _mlp_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    resource_check: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    model = _GainMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    validation_x = torch.as_tensor(x_validation, dtype=torch.float32)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    max_epochs = min(100, int(config.get("epochs", 100)))
    batch_size = int(config.get("batch_size", 32))
    patience = int(config.get("patience", 10))
    best_metric = math.inf
    best_state = None
    best_optimizer = None
    best_rng = None
    best_optimizer_updates = 0
    best_epoch = 0
    stale = 0
    optimizer_updates = 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            predicted, positive_logits = model(train_x[indices])
            regression = F.huber_loss(predicted, train_y[indices], delta=1.0)
            classification = F.binary_cross_entropy_with_logits(positive_logits, (train_y[indices] > 0).float())
            loss = regression + float(config.get("positive_aux_weight", 0.25)) * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
        if resource_check is not None:
            resource_check()
        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(validation_x)
            metric = float(torch.mean(torch.abs(validation_prediction - validation_y)).item())
        if metric < best_metric - 1e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_optimizer_updates = optimizer_updates
            best_rng = {
                "torch_rng_state": torch.get_rng_state().clone(),
                "numpy_rng_state": copy.deepcopy(np.random.get_state()),
                "python_rng_state": copy.deepcopy(random.getstate()),
                "batch_generator_state": generator.get_state().clone(),
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_optimizer is None or best_rng is None:
        raise RuntimeError("GAIN_MLP_TRAINING_NO_CHECKPOINT")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "gain-mlp-training-checkpoint/1", "seed": int(seed),
        "model_state": best_state, "optimizer_state": best_optimizer,
        "optimizer_updates_at_best": best_optimizer_updates, "total_optimizer_updates": optimizer_updates,
        "best_epoch": best_epoch, **best_rng,
    }, checkpoint_path)
    payload = {
        "kind": "mlp64x64", "seed": int(seed), "model_state": _state_dict_json(model),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "optimizer_updates": optimizer_updates, "optimizer_updates_at_best": best_optimizer_updates,
        "best_epoch": best_epoch,
        "validation_mae": best_metric,
    }
    training = {
        "seed": int(seed), "best_epoch": best_epoch, "optimizer_updates": optimizer_updates,
        "validation_mae": best_metric, "epochs_completed": epoch,
        "training_seconds": time.perf_counter() - start,
    }
    return payload, training


def _fit_constant(y_train: np.ndarray) -> dict[str, Any]:
    return {"kind": "constant", "value": float(y_train.mean())}


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return {"kind": "ridge", "alpha": float(alpha), "weights": weights.tolist()}


def _fit_gbtree(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray,
    depths: Sequence[int], seed: int, resource_check: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib
    except ImportError as exc:
        raise RuntimeError("GAIN_SKLEARN_REQUIRED: install scikit-learn in this project's .deps directory") from exc
    best = None
    rows = []
    for depth in depths:
        estimator = GradientBoostingRegressor(
            loss="huber", n_estimators=100, learning_rate=0.05,
            max_depth=int(depth), random_state=int(seed),
        )
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_validation)
        if resource_check is not None:
            resource_check()
        mae = float(np.mean(np.abs(prediction - y_validation)))
        rows.append({"max_depth": int(depth), "validation_mae": mae})
        if best is None or (mae, int(depth)) < (best[0], best[1]):
            best = (mae, int(depth), estimator)
    assert best is not None
    return {"kind": "gradient_boosting", "max_depth": best[1], "estimator": best[2]}, {"depth_trials": rows, "validation_mae": best[0]}


def _predict_payload(payload: Mapping[str, Any], x: np.ndarray, model_dir: Path | None = None) -> np.ndarray:
    kind = str(payload.get("kind"))
    if kind == "constant":
        return np.full(len(x), float(payload["value"]), dtype=np.float64)
    if kind == "ridge":
        weights = np.asarray(payload["weights"], dtype=np.float64)
        return np.column_stack([np.ones(len(x)), x]) @ weights
    if kind == "mlp64x64":
        rng_state = torch.get_rng_state()
        model = _GainMLP(x.shape[1])
        torch.set_rng_state(rng_state)
        state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in payload["model_state"].items()}
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions, _ = model(torch.as_tensor(x, dtype=torch.float32))
        return predictions.cpu().numpy().astype(np.float64)
    if kind == "gradient_boosting":
        if model_dir is None:
            raise ValueError("GAIN_MODEL_TREE_DIRECTORY_REQUIRED")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("GAIN_SKLEARN_REQUIRED: install scikit-learn in this project's .deps directory") from exc
        artifact = model_dir / str(payload["estimator_file"])
        estimator = joblib.load(artifact)
        return np.asarray(estimator.predict(x), dtype=np.float64)
    raise ValueError(f"GAIN_MODEL_UNKNOWN_KIND:{kind}")


def _normalizer_matrix(features: Mapping[str, Any], normalizer: Mapping[str, Any]) -> np.ndarray:
    values = _feature_map(features)
    names = list(normalizer["feature_names"])
    if list(values) != names:
        raise ValueError("GAIN_MODEL_FEATURE_SCHEMA_MISMATCH")
    raw = np.asarray([[values[name] for name in names]], dtype=np.float64)
    return _transform(raw, normalizer)


def predict_gain(bundle: Mapping[str, Any], features: Mapping[str, Any], model_dir: str | Path | None = None) -> dict[str, Any]:
    if isinstance(bundle, GainRuntime):
        return bundle.predict(features)
    if bundle.get("schema") != SCHEMA:
        raise ValueError("GAIN_MODEL_BUNDLE_SCHEMA")
    normalizer = bundle.get("normalizer")
    if not isinstance(normalizer, Mapping):
        raise ValueError("GAIN_MODEL_NORMALIZER_REQUIRED")
    x = _normalizer_matrix(features, normalizer)
    prediction = float(_predict_payload(bundle["model"], x, Path(model_dir) if model_dir is not None else None)[0])
    calibration = bundle.get("calibration", {})
    q = calibration.get("q") if isinstance(calibration, Mapping) else None
    lower_bound = prediction - float(q) if q is not None and math.isfinite(float(q)) else None
    return {
        "gain_hat": prediction, "lower_bound": lower_bound,
        "epsilon": float(bundle["threshold"]["epsilon"]),
        "model_name": str(bundle.get("model_name", "unknown")),
    }


@dataclass
class GainRuntime:
    """Loaded model handle; weights and tree estimators are constructed once."""

    bundle: dict[str, Any]
    model_dir: Path
    model: Any = None
    cold_load_seconds: float = 0.0
    optimizer_state_restored: bool = False
    forward_calls: int = 0
    forward_seconds: list[float] = field(default_factory=list)

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        normalizer = self.bundle["normalizer"]
        x = _normalizer_matrix(features, normalizer)
        start = time.perf_counter()
        if self.bundle["model"]["kind"] == "mlp64x64":
            self.model.eval()
            with torch.no_grad():
                prediction, _ = self.model(torch.as_tensor(x, dtype=torch.float32))
            gain_hat = float(prediction.item())
        elif self.bundle["model"]["kind"] == "gradient_boosting":
            gain_hat = float(self.model.predict(x)[0])
        else:
            gain_hat = float(_predict_payload(self.bundle["model"], x, self.model_dir)[0])
        elapsed = time.perf_counter() - start
        self.forward_calls += 1
        self.forward_seconds.append(elapsed)
        calibration = self.bundle.get("calibration", {})
        q = calibration.get("q") if isinstance(calibration, Mapping) else None
        lower_bound = gain_hat - float(q) if q is not None and math.isfinite(float(q)) else None
        return {
            "gain_hat": gain_hat, "lower_bound": lower_bound,
            "epsilon": float(self.bundle["threshold"]["epsilon"]),
            "model_name": str(self.bundle.get("model_name", "unknown")),
            "model_forward_seconds": elapsed,
        }


def load_runtime(path: str | Path) -> GainRuntime:
    """Load one immutable predictor and validate its training checkpoint."""
    bundle_path = Path(path).resolve(strict=True)
    start = time.perf_counter()
    bundle = load_bundle(bundle_path)
    model_payload = bundle["model"]
    kind = model_payload.get("kind")
    runtime_model = None
    optimizer_state_restored = kind != "mlp64x64"
    if kind == "mlp64x64":
        checkpoint_path = bundle_path.parent / str(model_payload.get("checkpoint_file", ""))
        if not checkpoint_path.is_file():
            raise ValueError("GAIN_MODEL_TRAINING_CHECKPOINT_MISSING")
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_hash != model_payload.get("checkpoint_sha256"):
            raise ValueError("GAIN_MODEL_TRAINING_CHECKPOINT_HASH_MISMATCH")
        saved_torch_rng = torch.get_rng_state()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("schema") != "gain-mlp-training-checkpoint/1":
                raise ValueError("GAIN_MODEL_TRAINING_CHECKPOINT_SCHEMA")
            expected_state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in model_payload["model_state"].items()}
            saved_state = checkpoint.get("model_state", {})
            if set(expected_state) != set(saved_state) or any(not torch.equal(expected_state[key], saved_state[key]) for key in expected_state):
                raise ValueError("GAIN_MODEL_BUNDLE_CHECKPOINT_STATE_MISMATCH")
            runtime_model = _GainMLP(len(bundle["normalizer"]["feature_names"]))
            runtime_model.load_state_dict(saved_state)
            optimizer_probe = torch.optim.Adam(runtime_model.parameters(), lr=0.001)
            optimizer_probe.load_state_dict(checkpoint["optimizer_state"])
            if not checkpoint.get("optimizer_state") or int(checkpoint.get("optimizer_updates_at_best", 0)) < 1:
                raise ValueError("GAIN_MODEL_OPTIMIZER_CHECKPOINT_INVALID")
            optimizer_state_restored = True
            runtime_model.eval()
        finally:
            torch.set_rng_state(saved_torch_rng)
    elif kind == "gradient_boosting":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("GAIN_SKLEARN_REQUIRED: install scikit-learn in this project's .deps directory") from exc
        estimator_path = bundle_path.parent / str(model_payload["estimator_file"])
        expected = bundle["training"]["fit_info"].get("estimator_sha256")
        if not estimator_path.is_file() or hashlib.sha256(estimator_path.read_bytes()).hexdigest() != expected:
            raise ValueError("GAIN_MODEL_TREE_CHECKPOINT_HASH_MISMATCH")
        runtime_model = joblib.load(estimator_path)
    return GainRuntime(
        bundle, bundle_path.parent, runtime_model, time.perf_counter() - start,
        optimizer_state_restored=optimizer_state_restored,
    )


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve(strict=True)
    bundle = _read_json(bundle_path)
    if bundle.get("schema") != SCHEMA:
        raise ValueError("GAIN_MODEL_BUNDLE_SCHEMA")
    if bundle.get("bundle_sha256") != _sha256({key: value for key, value in bundle.items() if key != "bundle_sha256"}):
        raise ValueError("GAIN_MODEL_BUNDLE_HASH_MISMATCH")
    return bundle


def _save_bundle(path: Path, bundle: dict[str, Any]) -> str:
    payload = copy.deepcopy(bundle)
    payload["schema"] = SCHEMA
    payload["bundle_sha256"] = _sha256(payload)
    _write_json(path, payload)
    return payload["bundle_sha256"]


def train_light_models(
    rows: Sequence[Mapping[str, Any]], output_dir: str | Path,
    config: Mapping[str, Any] | None = None, *, resource_check: Any = None,
) -> dict[str, Any]:
    """Fit light gain baselines and freeze selection/calibration artifacts.

    Training rows must be grouped by independent scenario instance. Rows in
    the validation split are divided by scenario into model-selection and
    residual-calibration groups before any model is fitted.
    """
    config = dict(config or {})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if resource_check is not None:
        resource_check()
    leakage = check_group_leakage(rows)
    train_rows = _eligible(rows, "train")
    validation_model_rows, calibration_rows = _validation_parts(rows, int(config.get("validation_partition_seed", 8201)))
    validation_model_rows = [row for row in validation_model_rows if _is_valid(row) and not _is_forced(row)]
    calibration_rows = [row for row in calibration_rows if _is_valid(row) and not _is_forced(row)]
    if len(train_rows) < 2 or not validation_model_rows or not calibration_rows:
        raise ValueError("GAIN_MODEL_INSUFFICIENT_TRAIN_VALIDATION_CALIBRATION")
    normalizer = _fit_normalizer(train_rows)
    names = normalizer["feature_names"]
    x_train, y_train = _xy(train_rows, names)
    x_model_val, y_model_val = _xy(validation_model_rows, names)
    x_calibration, y_calibration = _xy(calibration_rows, names)
    x_train = _transform(x_train, normalizer)
    x_model_val = _transform(x_model_val, normalizer)
    x_calibration = _transform(x_calibration, normalizer)
    epsilon_grid = _epsilon_candidates(config)
    coverage = float(config.get("calibration_coverage", 0.9))
    model_records: list[dict[str, Any]] = []

    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    constant = _fit_constant(y_train)
    candidates.append(("constant", constant, {"selection": "training_mean"}))
    ridge_alpha = float(config.get("ridge_alpha", 1.0))
    ridge = _fit_ridge(x_train, y_train, ridge_alpha)
    candidates.append(("ridge", ridge, {"alpha": ridge_alpha}))

    tree_config = config.get("gradient_boosting", {})
    tree_depths = list(tree_config.get("depths", [3, 5])) if isinstance(tree_config, Mapping) else [3, 5]
    tree_seed = int(tree_config.get("seed", 8201)) if isinstance(tree_config, Mapping) else 8201
    tree_model, tree_selection = _fit_gbtree(
        x_train, y_train, x_model_val, y_model_val, tree_depths, tree_seed, resource_check,
    )
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("GAIN_SKLEARN_REQUIRED: install scikit-learn in this project's .deps directory") from exc
    tree_path = output / "gain_tree.joblib"
    joblib.dump(tree_model.pop("estimator"), tree_path)
    tree_model["estimator_file"] = tree_path.name
    tree_selection["estimator_sha256"] = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    candidates.append(("gradient_boosting", tree_model, tree_selection))

    mlp_config = dict(config.get("mlp", {}))
    mlp_config.setdefault("learning_rate", 0.001)
    mlp_config.setdefault("batch_size", 32)
    mlp_config.setdefault("epochs", 100)
    mlp_config.setdefault("patience", 10)
    mlp_runs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for seed in config.get("mlp_seeds", list(MLP_SEEDS)):
        ckpt = output / f"gain_mlp_seed_{int(seed)}.pt"
        model, training = _mlp_train(
            x_train, y_train, x_model_val, y_model_val, int(seed), mlp_config, ckpt, resource_check,
        )
        mlp_runs.append((f"mlp_seed_{int(seed)}", model, training))

    all_candidates = candidates + mlp_runs
    for name, model, fit_info in all_candidates:
        if resource_check is not None:
            resource_check()
        val_predictions = _predict_payload(model, x_model_val, output)
        calibration_predictions = _predict_payload(model, x_calibration, output)
        epsilon = select_epsilon(y_model_val, val_predictions, epsilon_grid)
        calibration = calibrate_lower_bound(y_calibration, calibration_predictions, coverage)
        bundle: dict[str, Any] = {
            "schema": SCHEMA, "model_name": name, "model": model,
            "normalizer": normalizer,
            "threshold": {**epsilon, "selection_split": "validation_model", "frozen_before_test": True},
            "calibration": {**calibration, "calibration_split": "validation_calibration"},
            "training": {"fit_split": "train", "fit_row_count": len(train_rows), "fit_info": fit_info},
            "data_identity": {
                "training_state_ids_sha256": _sha256(sorted(str(row.get("state_id", row.get("sample_id", ""))) for row in train_rows)),
                "training_examples_sha256": _sha256([
                    {"state_id": str(row.get("state_id", row.get("sample_id", ""))),
                     "features": _row_features(row), "gain": _row_gain(row)}
                    for row in sorted(train_rows, key=lambda item: str(item.get("state_id", item.get("sample_id", ""))))
                ]),
                "validation_model_group_ids_sha256": _sha256(sorted({_group_id(row) for row in validation_model_rows})),
                "validation_calibration_group_ids_sha256": _sha256(sorted({_group_id(row) for row in calibration_rows})),
                "test_accessed": False,
            },
        }
        bundle_path = output / f"{name}.json"
        bundle_hash = _save_bundle(bundle_path, bundle)
        model_records.append({
            "model_name": name, "bundle_file": bundle_path.name, "bundle_sha256": bundle_hash,
            "model_selection_mae": float(np.mean(np.abs(val_predictions - y_model_val))),
            "calibration": calibration, "epsilon": epsilon["epsilon"],
            "validation_model_count": len(validation_model_rows), "validation_calibration_count": len(calibration_rows),
            "fit_info": fit_info,
        })

    result = {
        "schema": TRAINING_SCHEMA, "status": "complete", "config_sha256": _sha256(config),
        "split_audit": leakage,
        "groups": {
            "train_scenario_groups": sorted({_group_id(row) for row in train_rows}),
            "validation_model_scenario_groups": sorted({_group_id(row) for row in validation_model_rows}),
            "validation_calibration_scenario_groups": sorted({_group_id(row) for row in calibration_rows}),
            "test_rows_loaded": False,
        },
        "forced_excluded_count": sum(1 for row in rows if _is_forced(row)),
        "forced_row_fraction": float(sum(1 for row in rows if _is_forced(row)) / max(1, len(rows))),
        "invalid_excluded_count": sum(1 for row in rows if not _is_valid(row)),
        "normalizer": normalizer, "models": model_records,
    }
    _write_json(output / "GAIN_MODEL_TRAINING.json", result)
    if resource_check is not None:
        resource_check()
    return result


def fresh_process_restore_check(bundle_path: str | Path, features: Mapping[str, Any], tolerance: float = 1e-7) -> dict[str, Any]:
    """Verify a saved bundle by predicting in a separate Python process."""
    bundle_path = Path(bundle_path).resolve(strict=True)
    bundle = load_bundle(bundle_path)
    expected_runtime = load_runtime(bundle_path)
    expected = expected_runtime.predict(features)
    request_path = bundle_path.parent / f"restore_request_{os.getpid()}.json"
    _write_json(request_path, {"bundle": str(bundle_path), "features": dict(features)})
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--restore-probe", str(request_path)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            check=False,
        )
    finally:
        request_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"GAIN_MODEL_FRESH_PROCESS_RESTORE_FAILED:{result.stderr[-1000:]}")
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    child_cpu_seconds = actual.get("process_cpu_seconds")
    if isinstance(child_cpu_seconds, bool) or not isinstance(child_cpu_seconds, (int, float)) or not math.isfinite(float(child_cpu_seconds)) or float(child_cpu_seconds) < 0:
        raise ValueError("GAIN_MODEL_FRESH_PROCESS_CPU_RECEIPT_INVALID")
    if not math.isclose(float(actual["gain_hat"]), float(expected["gain_hat"]), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("GAIN_MODEL_FRESH_PROCESS_PREDICTION_MISMATCH")
    if actual.get("checkpoint_valid") is not True or actual.get("optimizer_state_restored") is not True:
        raise ValueError("GAIN_MODEL_FRESH_PROCESS_OPTIMIZER_RESTORE_MISMATCH")
    return {
        "fresh_process": True, "prediction_matches": True, "checkpoint_valid": True,
        "optimizer_state_restored": True, "gain_hat": expected["gain_hat"],
        "bundle_sha256": bundle["bundle_sha256"], "cold_load_seconds": expected_runtime.cold_load_seconds,
        "fresh_process_cpu_seconds": float(child_cpu_seconds),
    }


def _restore_probe(request_path: str) -> int:
    cpu_started = time.process_time()
    request = _read_json(Path(request_path))
    bundle_path = Path(request["bundle"]).resolve(strict=True)
    runtime = load_runtime(bundle_path)
    result = runtime.predict(request["features"])
    result["checkpoint_valid"] = True
    result["optimizer_state_restored"] = runtime.optimizer_state_restored
    result["process_cpu_seconds"] = max(0.0, time.process_time() - cpu_started)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-probe")
    args = parser.parse_args(argv)
    if args.restore_probe:
        return _restore_probe(args.restore_probe)
    parser.error("use the staged gain_runner.py command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
