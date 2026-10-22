"""Load trusted MLP and ExtraTrees artifacts emitted by light_models_v1."""
from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Any

try:
    from ..light_models_v1.features import FEATURE_SCHEMA, extract_features
except ImportError:
    from light_models_v1.features import FEATURE_SCHEMA, extract_features


def load_mlp_gain_predictor(path: str | Path):
    """Load a training-script MLP checkpoint and return a public-state predictor."""
    import torch
    from torch import nn

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not {"model", "center", "scale"} <= set(payload):
        raise ValueError("checkpoint does not match light_models_v1 MLP format")
    if len(payload["center"]) != len(FEATURE_SCHEMA) or len(payload["scale"]) != len(FEATURE_SCHEMA):
        raise ValueError("checkpoint feature width does not match FEATURE_SCHEMA")
    model = nn.Sequential(nn.Linear(len(FEATURE_SCHEMA), 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    model.load_state_dict(payload["model"])
    model.eval()
    center = torch.tensor(payload["center"], dtype=torch.float32)
    scale = torch.tensor(payload["scale"], dtype=torch.float32).clamp_min(1e-12)

    def predict(projection: dict[str, Any]) -> float:
        features = extract_features(projection)
        raw = torch.tensor([features[name] for name in FEATURE_SCHEMA], dtype=torch.float32)
        with torch.no_grad():
            return float(model(((raw - center) / scale).unsqueeze(0)).item())

    return predict


def _tree_predictor(payload: dict[str, Any]):
    if payload.get("schema") != "isaac_roi_tree_checkpoint_v1":
        raise ValueError("checkpoint does not match light_models_v1 tree format")
    if payload.get("estimator") != "ExtraTreesRegressor":
        raise ValueError("unsupported tree checkpoint estimator")
    if payload.get("feature_schema") != list(FEATURE_SCHEMA):
        raise ValueError("checkpoint feature schema does not match FEATURE_SCHEMA")
    model = payload.get("model")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("tree checkpoint has no predictor")

    def predict(projection: dict[str, Any]) -> float:
        features = extract_features(projection)
        raw = [[float(features[name]) for name in FEATURE_SCHEMA]]
        return float(model.predict(raw)[0])

    return predict


def load_tree_gain_predictor(path: str | Path):
    """Load a trusted light_models_v1 ExtraTrees artifact and predict public-state gain."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("tree checkpoint must contain a mapping")
    return _tree_predictor(payload)


def load_gain_artifact(path: str | Path, mode: str = "mean"):
    """Load a light_models_v1 deployment artifact and return (predictor, gate metadata)."""
    from light_models_v1.selection import deployment_mode_metadata

    artifact_path = Path(path)
    if artifact_path.suffix.lower() == ".json":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    elif artifact_path.suffix.lower() == ".pkl":
        with artifact_path.open("rb") as stream:
            payload = pickle.load(stream)
    elif artifact_path.suffix.lower() == ".pt":
        import torch
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("MLP artifact must contain a mapping")
        payload = {**payload, "model_kind": "mlp"}
    else:
        raise ValueError("gain artifact must be .json, .pkl, or .pt")
    if not isinstance(payload, dict):
        raise ValueError("gain artifact must contain a mapping")
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("gain artifact has no deployment selection/calibration metadata")
    metadata = deployment_mode_metadata(deployment, mode)
    if payload.get("feature_schema") != list(FEATURE_SCHEMA):
        raise ValueError("artifact feature schema does not match FEATURE_SCHEMA")
    kind = payload.get("model_kind")
    if payload.get("schema") == "isaac_roi_constant_gain_model_v1":
        kind = "constant"
    elif payload.get("schema") == "isaac_roi_ridge_gain_model_v1":
        kind = "ridge"
    elif payload.get("schema") == "isaac_roi_tree_checkpoint_v1":
        kind = "tree"
    metadata["model_kind"] = kind
    if kind == "constant":
        mean = float(payload["mean"])
        raw_predict = lambda _projection: mean
    elif kind == "ridge":
        center = [float(value) for value in payload["center"]]
        scale = [float(value) for value in payload["scale"]]
        coefficients = [float(value) for value in payload["coefficients"]]
        if not (len(center) == len(scale) == len(FEATURE_SCHEMA)
                and len(coefficients) == len(FEATURE_SCHEMA) + 1):
            raise ValueError("ridge artifact dimensions do not match FEATURE_SCHEMA")
        def raw_predict(projection):
            features = extract_features(projection)
            values = [features[name] for name in FEATURE_SCHEMA]
            return coefficients[0] + sum(coefficients[index + 1] *
                ((values[index] - center[index]) / scale[index]) for index in range(len(values)))
    elif kind == "tree":
        raw_predict = _tree_predictor(payload)
    elif kind == "mlp":
        import torch
        from torch import nn
        center = torch.tensor(payload["center"], dtype=torch.float32)
        scale = torch.tensor(payload["scale"], dtype=torch.float32).clamp_min(1e-12)
        model = nn.Sequential(nn.Linear(len(FEATURE_SCHEMA), 64), nn.ReLU(),
                              nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        model.load_state_dict(payload["model"])
        model.eval()
        def raw_predict(projection):
            features = extract_features(projection)
            raw = torch.tensor([features[name] for name in FEATURE_SCHEMA], dtype=torch.float32)
            with torch.no_grad():
                return float(model(((raw - center) / scale).unsqueeze(0)).item())
    else:
        raise ValueError(f"unsupported deployment model kind: {kind!r}")

    q = float(metadata["q"])
    predictor = lambda projection: float(raw_predict(projection)) + q
    return predictor, metadata
