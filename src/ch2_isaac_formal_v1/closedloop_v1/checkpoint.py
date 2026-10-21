"""Load trusted MLP and ExtraTrees artifacts emitted by light_models_v1."""
from __future__ import annotations

import pickle
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
