"""Public-history WM gain calibration and inference adapter.

This API is not wired into the closed-loop runtime. Validation roles and
calibration use only the validation-selection/validation-calibration groups;
test and OOD data are intentionally outside this module's contract.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import PublicHistoryFrame
from .features import EVENT_NAMES, EVENT_SCHEMA_VERSION, FEATURE_NAMES, collate_graph_sequences


GAIN_VERSION = "wm_gain_deployment_v1"
EXPECTED_HISTORY_PHASE = "post_step_through_t_minus_1_plus_pre_candidate_projection_t"
EXPECTED_GPU_UUID = "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_gain_deployment_metadata(
    validation_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    *,
    checkpoint_sha256: str,
    dataset_sha256: str,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Calibrate validation roles with scenario-level selection/calibration isolation."""
    if len(validation_rows) != len(predictions) or not validation_rows:
        raise ValueError("validation rows and predictions must be non-empty and aligned")
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("gain deployment calibration accepts validation rows only")
    scenarios = [row.get("scenario_id") for row in validation_rows]
    if any(not isinstance(value, str) or not value for value in scenarios):
        raise ValueError("validation rows require scenario_id")
    if len(set(scenarios)) < 2:
        raise ValueError("selection and calibration require at least two validation scenarios")
    if len(checkpoint_sha256) != 64 or len(dataset_sha256) != 64:
        raise ValueError("checkpoint and dataset identities must be SHA-256 digests")

    try:
        from ..light_models_v1.selection import build_deployment_metadata
        from ..light_models_v1.train_models import _scenario_validation_split
    except ImportError:
        from light_models_v1.selection import build_deployment_metadata
        from light_models_v1.train_models import _scenario_validation_split

    rows = [dict(row) for row in validation_rows]
    roles = _scenario_validation_split(rows)
    selection_scenarios = {str(row["scenario_id"]) for row in roles["selection"]}
    calibration_scenarios = {str(row["scenario_id"]) for row in roles["calibration"]}
    row_ids = [str(row.get("state_id", "")) for row in rows]
    if not all(row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("validation rows require unique state_id values")
    prediction_by_id = dict(zip(row_ids, map(float, predictions)))
    selection = roles["selection"]
    calibration = roles["calibration"]
    deployment = build_deployment_metadata(
        selection,
        [prediction_by_id[str(row["state_id"])] for row in selection],
        calibration,
        [prediction_by_id[str(row["state_id"])] for row in calibration],
        data_identity=dataset_sha256,
        feature_schema=FEATURE_NAMES,
        alpha=alpha,
    )
    schema_payload = {
        "feature_names": list(FEATURE_NAMES),
        "event_names": list(EVENT_NAMES),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "history_phase": EXPECTED_HISTORY_PHASE,
    }
    schema_sha256 = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": GAIN_VERSION,
        "version": GAIN_VERSION,
        "integration_status": "NOT_INTEGRATED",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "feature_schema_sha256": deployment["feature_schema_sha256"],
        "input_schema_sha256": schema_sha256,
        "feature_names": list(FEATURE_NAMES),
        "event_names": list(EVENT_NAMES),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "history_phase": EXPECTED_HISTORY_PHASE,
        "validation_selection_scenarios": deployment["selection"]["selection_scenarios"],
        "validation_calibration_scenarios": deployment["lower_bound_calibration"]["calibration_scenarios"],
        "selection_calibration_disjoint": not bool(selection_scenarios & calibration_scenarios),
        "test_or_ood_used": False,
        "deployment": deployment,
    }


def _check_gpu1() -> Any:
    import torch

    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != EXPECTED_GPU_UUID:
        raise RuntimeError("GPU1_UUID_NOT_EXPLICITLY_SELECTED")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("UUID_SELECTED_CUDA_DEVICE_UNAVAILABLE")
    properties = torch.cuda.get_device_properties(0)
    if str(getattr(properties, "uuid", "unknown")) != EXPECTED_GPU_UUID:
        raise RuntimeError("VISIBLE_DEVICE_UUID_MISMATCH")
    return torch.device("cuda:0")


@dataclass(frozen=True)
class GainInference:
    gain_mean: float
    gain_lower_bound: float | None
    threshold: float
    score: float
    mode: str
    version: str
    device: str
    integration_status: str
    timings_ms: Mapping[str, float]


class GainDeploymentAdapter:
    """Predict a gain score from explicit public history; never runs a replanner."""

    def __init__(
        self,
        model: Any,
        center: Sequence[float],
        scale: Sequence[float],
        deployment_artifact: Mapping[str, Any],
        *,
        device: str = "cuda:0",
        checkpoint_sha256: str,
        allow_cpu_fixture: bool = False,
    ) -> None:
        import torch

        if deployment_artifact.get("schema") != GAIN_VERSION:
            raise ValueError("unsupported WM gain deployment schema")
        if deployment_artifact.get("integration_status") != "NOT_INTEGRATED":
            raise ValueError("gain adapter artifact has an unexpected integration status")
        if deployment_artifact.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("gain artifact is not bound to the loaded checkpoint")
        if device == "cpu":
            if not allow_cpu_fixture:
                raise RuntimeError("CPU_GAIN_INFERENCE_IS_SYNTHETIC_FIXTURE_ONLY")
            self.integration_status = "CPU_SYNTHETIC_ONLY_NOT_INTEGRATED"
        elif device == "cuda:0":
            selected = _check_gpu1()
            self.integration_status = "NOT_INTEGRATED"
            device = str(selected)
        else:
            raise RuntimeError("WM gain inference is restricted to GPU1 cuda:0")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.center.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.center.shape:
            raise ValueError("normalization dimensions do not match feature schema")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("normalization parameters must be finite with positive scales")
        normalization = deployment_artifact.get("normalization")
        if not isinstance(normalization, Mapping) or normalization.get("scope") != "train_split_only":
            raise ValueError("gain artifact must carry train-only normalization metadata")
        if list(deployment_artifact.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError("gain artifact feature schema mismatch")
        self.artifact = dict(deployment_artifact)
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, deployment_path: str | Path, *, device: str = "cuda:0"
    ) -> "GainDeploymentAdapter":
        import torch

        checkpoint = Path(checkpoint_path)
        deployment_file = Path(deployment_path)
        checkpoint_hash = _sha256(checkpoint)
        artifact = json.loads(deployment_file.read_text(encoding="utf-8"))
        if artifact.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("deployment artifact checkpoint hash does not match checkpoint")
        if artifact.get("input_schema_sha256") != hashlib.sha256(json.dumps({
                "feature_names": list(FEATURE_NAMES), "event_names": list(EVENT_NAMES),
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "history_phase": EXPECTED_HISTORY_PHASE,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest():
            raise ValueError("deployment artifact input schema hash mismatch")
        expected_feature_hash = hashlib.sha256(
            json.dumps(list(FEATURE_NAMES), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if artifact.get("feature_schema_sha256") != expected_feature_hash:
            raise ValueError("deployment artifact feature schema hash mismatch")
        selected_device = _check_gpu1() if device == "cuda:0" else None
        if selected_device is None:
            raise RuntimeError("checkpoint deployment loading is restricted to GPU1 cuda:0")
        with checkpoint.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        architecture = metadata["architecture"]
        model = _build_model(architecture)
        if (metadata.get("feature_names") != list(FEATURE_NAMES)
                or metadata.get("event_names") != list(EVENT_NAMES)
                or metadata.get("event_schema_version") != EVENT_SCHEMA_VERSION
                or int(metadata.get("feature_dim", -1)) != len(FEATURE_NAMES)):
            raise ValueError("checkpoint feature/event schema mismatch")
        model.load_state_dict(payload["model_state"])
        normalization = metadata["normalization"]
        if normalization.get("scope") != "train_split_only":
            raise ValueError("checkpoint normalization is not train-only")
        artifact["normalization"] = normalization
        if artifact.get("dataset_sha256") != metadata.get("dataset_sha256"):
            raise ValueError("deployment artifact dataset identity mismatch")
        return cls(model, normalization["center"], normalization["scale"], artifact,
                   device=str(selected_device), checkpoint_sha256=checkpoint_hash)

    def predict(
        self,
        history: Sequence[PublicHistoryFrame],
        *,
        history_phase: str,
        mode: str = "lower_bound",
    ) -> GainInference:
        import torch

        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        if history_phase != EXPECTED_HISTORY_PHASE:
            raise ValueError("public history phase does not match the trained gain contract")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not 2 <= len(history) <= 8:
            raise ValueError("gain inference requires 2 to 8 explicit public history frames")
        if any(not isinstance(frame, PublicHistoryFrame) for frame in history):
            raise TypeError("gain inference accepts explicit PublicHistoryFrame values only")
        times = [float(frame.time_s) for frame in history]
        if (any(not np.isfinite(value) or not value.is_integer() for value in times)
                or any(b - a != 1.0 for a, b in zip(times, times[1:]))):
            raise ValueError("history must be contiguous one-second frames ending at decision time t")
        deployment = self.artifact["deployment"]
        selected = deployment.get("modes", {}).get(mode)
        if not isinstance(selected, Mapping):
            raise ValueError(f"gain mode {mode!r} is unavailable")
        sanitized = [replace(frame, action=None, event={}) for frame in history]

        feature_start = time.perf_counter()
        batch = collate_graph_sequences([sanitized])
        nodes = (batch.node_features - self.center) / self.scale
        nodes[~batch.node_mask] = 0.0
        feature_ms = (time.perf_counter() - feature_start) * 1000.0

        transfer_start = time.perf_counter()
        inputs = (
            torch.as_tensor(nodes, dtype=torch.float32),
            torch.as_tensor(batch.node_mask, dtype=torch.bool),
            torch.as_tensor(batch.sequence_mask, dtype=torch.bool),
            torch.as_tensor(batch.edge_index, dtype=torch.long),
            torch.as_tensor(batch.edge_mask, dtype=torch.bool),
            torch.as_tensor(batch.edge_type, dtype=torch.long),
            torch.as_tensor(batch.event_features, dtype=torch.float32),
            torch.zeros_like(torch.as_tensor(batch.action_features, dtype=torch.float32)),
        )
        inputs = tuple(value.to(self.device) for value in inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        forward_start = time.perf_counter()
        with torch.no_grad():
            prediction = self.model(*inputs)["gain"]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0

        return_start = time.perf_counter()
        gain_mean = float(prediction.detach().cpu().item())
        calibration = deployment.get("lower_bound_calibration", {})
        q = calibration.get("q") if calibration.get("status") == "PASS" else None
        gain_lower = gain_mean + float(q) if q is not None else None
        return_ms = (time.perf_counter() - return_start) * 1000.0
        score = gain_lower if mode == "lower_bound" else gain_mean
        if score is None:
            raise ValueError("lower-bound mode has no calibrated residual quantile")
        timings = {
            "validation_and_sanitization_ms": (feature_start - wall_start) * 1000.0,
            "feature_construction_ms": feature_ms,
            "cpu_to_device_ms": transfer_ms,
            "forward_synchronized_ms": forward_ms,
            "device_to_cpu_ms": return_ms,
        }
        end_to_end_ms = (time.perf_counter() - wall_start) * 1000.0
        cpu_total_ms = (time.thread_time() - cpu_start) * 1000.0
        timings["end_to_end_wall_ms"] = end_to_end_ms
        timings["calling_thread_cpu_ms"] = cpu_total_ms
        return GainInference(
            gain_mean=gain_mean, gain_lower_bound=gain_lower,
            threshold=float(selected["epsilon"]), score=float(score), mode=mode,
            version=str(self.artifact["version"]), device=str(self.device),
            integration_status=self.integration_status, timings_ms=timings,
        )


def _build_model(architecture: Mapping[str, Any]) -> Any:
    from .world_model import EventAwareWorldModel

    required = {"node_dim", "hidden_dim", "dynamic_dim", "event_dim", "action_dim"}
    if not required.issubset(architecture):
        raise ValueError("checkpoint architecture metadata is incomplete")
    return EventAwareWorldModel(**{key: int(architecture[key]) for key in required})
