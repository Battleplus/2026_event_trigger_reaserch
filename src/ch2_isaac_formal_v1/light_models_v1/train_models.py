"""Guarded light-model training entry point.

The CLI is intentionally inert until the frozen 168-scene batch and the
post-extraction 30/30 gate both pass.  It never installs optional packages.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .data_gate import check_feature_coverage
    from .dataset import load_feature_dataset
    from .features import FEATURE_SCHEMA
except ImportError:  # direct invocation from this directory
    from data_gate import check_feature_coverage
    from dataset import load_feature_dataset
    from features import FEATURE_SCHEMA


MODEL_SEEDS = (8201, 8202, 8203)
EXPECTED_GPU_UUID = "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = [{"scenario_id": row.get("scenario_id"), "state_id": row.get("state_id"),
                "split": row.get("split"), "target": row.get("target"),
                "features": [row["features"][name] for name in FEATURE_SCHEMA]}
               for row in rows]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_validation_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scenarios = sorted({str(row["scenario_id"]) for row in rows if row.get("split") == "validation"})
    if len(scenarios) < 2:
        return {"selection": [], "calibration": []}
    selection_ids = set(scenarios[::2])
    if not selection_ids:
        selection_ids = {scenarios[0]}
    calibration_ids = set(scenarios) - selection_ids
    if not calibration_ids:
        calibration_ids = {scenarios[-1]}
        selection_ids = set(scenarios[:-1])
    return {
        "selection": [row for row in rows if row.get("split") == "validation" and row.get("scenario_id") in selection_ids],
        "calibration": [row for row in rows if row.get("split") == "validation" and row.get("scenario_id") in calibration_ids],
    }


def _metrics(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "regret": None}
    errors = [abs(float(row["target"]) - float(prediction)) for row, prediction in zip(rows, predictions)]
    regrets = []
    for row, prediction in zip(rows, predictions):
        target = float(row["target"])
        predicted_positive = float(prediction) > 0.0
        actual_positive = target > 0.0
        regrets.append(abs(target) if predicted_positive != actual_positive else 0.0)
    return {"mae": sum(errors) / len(errors), "regret": sum(regrets) / len(regrets)}


def _constant_model(train_rows: list[dict[str, Any]], evaluation: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    mean = sum(float(row["target"]) for row in train_rows) / len(train_rows)
    metrics = {}
    for name, rows in evaluation.items():
        metrics[name] = _metrics(rows, [mean] * len(rows))
    return {"status": "PASS", "mean": mean, "metrics": metrics}


def _ridge_model(train_rows: list[dict[str, Any]], evaluation: dict[str, list[dict[str, Any]]], alpha: float = 1.0) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"status": "RIDGE_NOT_RUN_NUMPY_MISSING"}
    x = np.asarray([[row["features"][name] for name in FEATURE_SCHEMA] for row in train_rows], dtype=float)
    y = np.asarray([float(row["target"]) for row in train_rows], dtype=float)
    scale = np.std(x, axis=0)
    scale[scale == 0] = 1.0
    center = np.mean(x, axis=0)
    normalized = (x - center) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def predict(rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        matrix = np.asarray([[row["features"][name] for name in FEATURE_SCHEMA] for row in rows], dtype=float)
        matrix = np.column_stack([np.ones(len(matrix)), (matrix - center) / scale])
        return [float(value) for value in matrix @ coefficients]

    return {"status": "PASS", "alpha": alpha, "normalization": "train_mean_std",
            "coefficients": [float(value) for value in coefficients],
            "metrics": {name: _metrics(rows, predict(rows)) for name, rows in evaluation.items()}}


def _tree_status(train_rows: list[dict[str, Any]], evaluation: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
    except ImportError:
        return {"status": "TREE_NOT_RUN_DEPENDENCY_MISSING", "dependency": "sklearn"}
    try:
        model = ExtraTreesRegressor(n_estimators=128, random_state=MODEL_SEEDS[0], n_jobs=1)
        x = [[row["features"][name] for name in FEATURE_SCHEMA] for row in train_rows]
        y = [float(row["target"]) for row in train_rows]
        model.fit(x, y)
        metrics = {}
        for name, rows in evaluation.items():
            values = [[row["features"][key] for key in FEATURE_SCHEMA] for row in rows]
            metrics[name] = _metrics(rows, [float(value) for value in model.predict(values)]) if rows else _metrics([], [])
        return {"status": "PASS", "metrics": metrics, "estimator": "ExtraTreesRegressor"}
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {"status": "TREE_FAILED", "error": f"{type(exc).__name__}: {exc}"}


def _cpu_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return copy.deepcopy(value)


def _fresh_process_checkpoint_check(checkpoint: Path) -> dict[str, Any]:
    script = (
        "import json,sys,torch; from torch import nn; "
        "p=torch.load(sys.argv[1],map_location='cpu'); "
        "x=torch.tensor(p['probe_raw'],dtype=torch.float32); "
        "center=torch.tensor(p['center'],dtype=torch.float32); scale=torch.tensor(p['scale'],dtype=torch.float32); "
        "m=nn.Sequential(nn.Linear(len(p['center']),64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)); "
        "m.load_state_dict(p['model']); m.eval(); "
        "value=float(m((x-center)/scale).reshape(-1)[0]); "
        "print(json.dumps({'prediction':value,'expected':float(p['probe_expected'])}))"
    )
    process = subprocess.run([sys.executable, "-c", script, str(checkpoint)],
                             capture_output=True, text=True, check=False)
    if process.returncode != 0:
        return {"status": "FAIL", "detail": process.stderr[-1000:]}
    try:
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        error = abs(float(payload["prediction"]) - float(payload["expected"]))
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "detail": f"invalid child output: {exc}"}
    return {"status": "PASS" if error <= 1e-5 else "FAIL",
            "max_abs_error": error, "tolerance": 1e-5}


def _mlp_status(train_rows: list[dict[str, Any]], evaluation: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"status": "MLP_NOT_RUN_DEPENDENCY_MISSING", "dependency": "torch"}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if selectors not in (["1"], [EXPECTED_GPU_UUID]):
        return {"status": "MLP_NOT_RUN_PHYSICAL_GPU1_NOT_EXPLICITLY_SELECTED",
                "required_selector": EXPECTED_GPU_UUID, "cuda_visible_devices": visible}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return {"status": "MLP_NOT_RUN_CUDA_DEVICE_1_UNAVAILABLE",
                "cuda_visible_devices": visible, "visible_device_count": torch.cuda.device_count()}
    if not evaluation.get("validation_selection"):
        return {"status": "MLP_NOT_RUN_VALIDATION_SELECTION_EMPTY"}
    # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical index 0.
    device = torch.device("cuda:0")
    metadata: dict[str, Any] = {
        "status": "PASS", "device": str(device), "physical_gpu_selector": visible,
        "physical_gpu_uuid_expected": EXPECTED_GPU_UUID,
        "logical_to_physical_mapping": {"cuda:0": visible}, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "physical_gpu_uuid_reported": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
        "seeds": list(MODEL_SEEDS),
        "batch_size": 32, "max_epochs": 100, "early_stopping_patience": 10, "runs": [],
    }
    import numpy as np
    train_x = np.asarray([[row["features"][name] for name in FEATURE_SCHEMA] for row in train_rows], dtype=np.float32)
    train_y = np.asarray([[float(row["target"])] for row in train_rows], dtype=np.float32)
    selection_rows = evaluation["validation_selection"]
    selection_x = np.asarray([[row["features"][name] for name in FEATURE_SCHEMA] for row in selection_rows], dtype=np.float32)
    selection_y = np.asarray([[float(row["target"])] for row in selection_rows], dtype=np.float32)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    x_tensor = torch.tensor((train_x - center) / scale, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    selection_tensor = torch.tensor((selection_x - center) / scale, dtype=torch.float32, device=device)
    selection_target = torch.tensor(selection_y, dtype=torch.float32, device=device)
    for seed in MODEL_SEEDS:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(len(FEATURE_SCHEMA), 64), nn.ReLU(),
                              nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        best_model_state = None
        best_optimizer_state = None
        updates = 0
        start = time.perf_counter()
        for epoch in range(100):
            model.train()
            order = torch.randperm(len(train_rows), generator=generator).tolist()
            for offset in range(0, len(order), 32):
                indices = order[offset:offset + 32]
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x_tensor.index_select(0, index_tensor)),
                                              y_tensor.index_select(0, index_tensor))
                loss.backward()
                optimizer.step()
                updates += 1
            model.eval()
            with torch.no_grad():
                validation_loss = float(nn.functional.mse_loss(model(selection_tensor), selection_target).item())
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_model_state = _cpu_state(model.state_dict())
                best_optimizer_state = _cpu_state(optimizer.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    break
        if best_model_state is None or best_optimizer_state is None:
            return {"status": "MLP_FAILED_NO_CHECKPOINT", "seed": seed}
        model.load_state_dict(best_model_state)
        model.eval()
        probe_row = selection_rows[0]
        probe_raw = [float(probe_row["features"][name]) for name in FEATURE_SCHEMA]
        with torch.no_grad():
            probe_tensor = torch.tensor((np.asarray([probe_raw], dtype=np.float32) - center) / scale,
                                        dtype=torch.float32, device=device)
            probe_expected = float(model(probe_tensor).item())
        metrics = {}
        for name, rows in evaluation.items():
            if not rows:
                metrics[name] = _metrics([], [])
                continue
            raw = np.asarray([[row["features"][key] for key in FEATURE_SCHEMA] for row in rows], dtype=np.float32)
            tensor = torch.tensor((raw - center) / scale, dtype=torch.float32, device=device)
            with torch.no_grad():
                predictions = [float(value) for value in model(tensor).reshape(-1).detach().cpu().tolist()]
            metrics[name] = _metrics(rows, predictions)
        checkpoint = output_dir / f"mlp_seed_{seed}.pt"
        torch.save({"model": best_model_state, "optimizer": best_optimizer_state,
                    "center": center.tolist(), "scale": scale.tolist(), "seed": seed,
                    "probe_raw": probe_raw, "probe_expected": probe_expected}, checkpoint)
        fresh_check = _fresh_process_checkpoint_check(checkpoint)
        run = {"seed": seed, "optimizer_updates": updates, "epochs": best_epoch,
               "best_validation_mse": best_loss, "elapsed_s": time.perf_counter() - start,
               "checkpoint": str(checkpoint), "normalization": "train_mean_std",
               "metrics": metrics, "data_sha256": _dataset_hash(train_rows + selection_rows + evaluation.get("validation_calibration", [])),
               "model_sha256": hashlib.sha256(pickle.dumps(best_model_state, protocol=4)).hexdigest(),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "optimizer_sha256": hashlib.sha256(pickle.dumps(best_optimizer_state, protocol=4)).hexdigest(),
               "fresh_process_checkpoint_consistency": fresh_check}
        metadata["runs"].append(run)
        if fresh_check.get("status") != "PASS":
            metadata["status"] = "MLP_FAILED_CHECKPOINT_CONSISTENCY"
    return metadata


def run_training(batch_output: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run approved models or write an explicit NOT_RUN receipt."""
    start = time.perf_counter()
    output_root = Path(output_dir).resolve()
    dataset = load_feature_dataset(batch_output)
    receipt: dict[str, Any] = {
        "schema": "isaac_roi_light_models_receipt_v1",
        "status": "NOT_RUN",
        "batch_gate": dataset["gate"],
        "feature_schema": list(FEATURE_SCHEMA),
        "models": {},
    }
    if not dataset["gate"].get("ready"):
        receipt["reason"] = dataset["gate"].get("reason")
        receipt["elapsed_s"] = time.perf_counter() - start
        _write_json(output_root / "training_receipt.json", receipt)
        return receipt
    rows = dataset.get("rows", [])
    coverage = check_feature_coverage(rows)
    receipt["feature_coverage"] = coverage
    receipt["excluded"] = dataset.get("excluded", [])
    receipt["dataset_sha256"] = _dataset_hash(rows) if rows else None
    if not coverage["met_30_30"]:
        receipt["reason"] = "FEATURE_COVERAGE_INSUFFICIENT"
        receipt["elapsed_s"] = time.perf_counter() - start
        _write_json(output_root / "training_receipt.json", receipt)
        return receipt
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation = _scenario_validation_split(rows)
    evaluation = {"validation_selection": validation["selection"],
                  "validation_calibration": validation["calibration"]}
    receipt["models"]["constant_cpu"] = _constant_model(train_rows, evaluation)
    receipt["models"]["ridge_cpu"] = _ridge_model(train_rows, evaluation)
    receipt["models"]["tree"] = _tree_status(train_rows, evaluation)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt["models"]["mlp_gpu1"] = _mlp_status(train_rows, evaluation, output_root)
    receipt["status"] = "COMPLETE"
    receipt["elapsed_s"] = time.perf_counter() - start
    _write_json(output_root / "training_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_training(args.batch_output, args.output_dir)
    print(json.dumps({"status": receipt["status"], "reason": receipt.get("reason")}, sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
