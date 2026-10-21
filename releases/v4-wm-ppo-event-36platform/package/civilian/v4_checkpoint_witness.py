"""Fresh-child restore witnesses for formal V4 checkpoints."""
from __future__ import annotations

import json
import base64
import io
import hashlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"V4_WITNESS_NON_JSON_VALUE:{type(value).__name__}")


def _equal(expected: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(_equal(expected[key], actual[key], tolerance) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_equal(a, b, tolerance) for a, b in zip(expected, actual))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance)
    return expected == actual


def _optimizer_state_matches(saved: dict, restored: dict) -> bool:
    return _equal(_plain(saved), _plain(restored), 1e-12)


def _wm_probe_identity(sample: dict) -> str:
    row = {key: sample.get(key) for key in ("sample_id", "split", "parent_id", "history", "action")}
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def policy_probe_prediction(policy: Any, probe: dict) -> dict:
    from civilian.v4_policy import SCOPE_NAMES, _scope_masks
    with torch.no_grad():
        result = policy.act(probe["observation"], probe["candidates"], probe["high_level_mask"],
                            probe["candidate_mask"], probe["wm_predictions"], deterministic=True)
    high_mask, low_masks = _scope_masks(probe["candidates"], probe["high_level_mask"],
                                        probe["candidate_mask"])
    low_mask = low_masks[SCOPE_NAMES[int(result["scope_index"])]]
    return {"scope": result["scope"], "candidate_key": result["candidate_key"],
            "full_environment_action": _plain(result["full_environment_action"]),
            "high_level_mask": list(high_mask), "low_level_mask": list(low_mask),
            "high_logits_valid": [float(value) for value, valid in zip(result["high_logits"], high_mask) if valid],
            "low_logits_valid": [float(value) for value, valid in zip(result["low_logits"], low_mask) if valid],
            "value": float(result["value"])}


def _run_child(request: dict) -> dict:
    package = Path(__file__).resolve().parents[1]
    payload = io.BytesIO()
    torch.save(request, payload)
    script = ("import base64,io,sys,torch; "
              "from civilian.v4_checkpoint_witness import _child_request; "
              "_child_request(torch.load(io.BytesIO(base64.b64decode(sys.stdin.read())), "
              "map_location='cpu', weights_only=False))")
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "PYTHONPATH": str(package) + os.pathsep + environment.get("PYTHONPATH", "")})
    result = subprocess.run([sys.executable, "-B", "-c", script], input=base64.b64encode(payload.getvalue()).decode("ascii"),
                            cwd=package, env=environment, capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise RuntimeError("V4_FRESH_CHILD_RESTORE_FAILED:" + result.stderr[-3000:])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("V4_FRESH_CHILD_RESTORE_NO_RESULT")
    response = json.loads(lines[-1])
    if response.get("status") != "complete":
        raise RuntimeError("V4_FRESH_CHILD_RESTORE_INCOMPLETE")
    return response


def check_world_models(rows: list[dict], *, tolerance: float = 1e-6) -> dict:
    """Rows contain checkpoint_path, probe sample, presave prediction and data identity."""
    response = _run_child({"kind": "world_model", "rows": rows})
    checks = response["checks"]
    if len(checks) != len(rows):
        raise RuntimeError("V4_WM_RESTORE_CHECK_COUNT")
    for row, check in zip(rows, checks):
        if (check.get("prediction_matches") is not True
                or check.get("data_identity") != row["data_identity"]
                or check.get("probe_sample_identity") != row["probe_sample_identity"]
                or not check.get("normalization_restored")
                or not check.get("rng_restored")
                or not check.get("probe_sample_identity_persisted")
                or check.get("optimizer_state_entries", 0) < 1
                or check.get("optimizer_state_matches") is not True):
            raise RuntimeError("V4_WM_FRESH_CHILD_WITNESS_MISMATCH")
        if not _equal(row["presave_prediction"], check["prediction"], tolerance):
            raise RuntimeError("V4_WM_PRESAVE_PREDICTION_MISMATCH")
        if not check.get("presave_reference_persisted"):
            raise RuntimeError("V4_WM_PRESAVE_REFERENCE_NOT_PERSISTED")
    return {"status": "complete", "fresh_process": True, "child_pid": response["pid"], "checks": checks}


def check_policies(rows: list[dict], *, tolerance: float = 1e-6) -> dict:
    """Rows contain checkpoint_path, deterministic action probe and presave output."""
    response = _run_child({"kind": "policy", "rows": rows})
    checks = response["checks"]
    if len(checks) != len(rows):
        raise RuntimeError("V4_PPO_RESTORE_CHECK_COUNT")
    for row, check in zip(rows, checks):
        if (check.get("prediction_matches") is not True
                or check.get("data_identity") != row["data_identity"]
                or not check.get("rng_restored")
                or check.get("optimizer_state_entries", 0) < 1
                or check.get("optimizer_state_matches") is not True
                or not check.get("presave_reference_persisted")
                or not _equal(row["presave_prediction"], check["prediction"], tolerance)):
            raise RuntimeError("V4_PPO_FRESH_CHILD_WITNESS_MISMATCH")
    return {"status": "complete", "fresh_process": True, "child_pid": response["pid"], "checks": checks}


def _child_request(request: dict) -> None:
    torch.set_num_threads(1)
    checks = []
    if request["kind"] == "world_model":
        from civilian.v4_models import V4DynamicWorldModel, collate_v4_samples
        for row in request["rows"]:
            payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
            model = V4DynamicWorldModel()
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            saved_optimizer = payload["optimizer_state"]
            optimizer = torch.optim.Adam(model.parameters(), lr=float(saved_optimizer["param_groups"][0]["lr"]))
            optimizer.load_state_dict(payload["optimizer_state"])
            optimizer_state_matches = _optimizer_state_matches(saved_optimizer, optimizer.state_dict())
            random.setstate(payload["python_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            rng_restored = (random.getstate() == payload["python_rng_state"]
                            and torch.equal(torch.get_rng_state(), payload["torch_rng_state"]))
            batch, _ = collate_v4_samples([row["probe"]])
            with torch.no_grad():
                prediction = model(batch)
            actual = {key: _plain(value[0]) for key, value in prediction.items()}
            saved_norm = payload.get("normalization", {})
            norm_matches = all(_equal(saved_norm.get(name), _plain(getattr(model, name)), 1e-7)
                               for name in ("node_mean", "node_scale", "action_mean", "action_scale"))
            saved_witness = payload.get("restore_witness", {})
            probe_identity = _wm_probe_identity(row["probe"])
            checks.append({"checkpoint_path": row["checkpoint_path"], "prediction": actual,
                           "prediction_matches": _equal(row["presave_prediction"], actual, 1e-6),
                           "presave_reference_persisted": _equal(saved_witness.get("presave_prediction"),
                                                                  row["presave_prediction"], 1e-12),
                           "probe_sample_identity": probe_identity,
                           "probe_sample_identity_persisted": saved_witness.get("probe_sample_identity") == probe_identity,
                           "data_identity": payload.get("data_identity"),
                           "normalization_restored": norm_matches,
                           "restore_device": "cpu", "rng_restored": rng_restored,
                           "rng_checked_scope": ["python", "torch_cpu"],
                           "cuda_rng_state_status": "saved_not_restored_cpu_child",
                           "optimizer_state_entries": len(optimizer.state),
                           "optimizer_parameter_entries": len(optimizer.state_dict()["param_groups"][0]["params"]),
                           "optimizer_state_matches": optimizer_state_matches})
    elif request["kind"] == "policy":
        from civilian.v4_policy import load_v4_checkpoint
        for row in request["rows"]:
            policy, optimizer, payload = load_v4_checkpoint(row["checkpoint_path"], device="cpu",
                                                            expected_source_identity=row["data_identity"])
            optimizer_state_matches = _optimizer_state_matches(payload["optimizer_state"], optimizer.state_dict())
            rng_restored = (random.getstate() == payload["python_rng_state"]
                            and torch.equal(torch.get_rng_state(), payload["torch_rng_state"]))
            saved_witness = payload.get("restore_witness", {})
            probe = row["probe"]
            prediction = policy_probe_prediction(policy, probe)
            checks.append({"checkpoint_path": row["checkpoint_path"], "prediction": prediction,
                           "prediction_matches": _equal(row["presave_prediction"], prediction, 1e-6),
                           "presave_reference_persisted": _equal(saved_witness.get("presave_prediction"),
                                                                  row["presave_prediction"], 1e-12),
                           "data_identity": payload.get("source_identity"),
                           "restore_device": "cpu", "rng_restored": rng_restored,
                           "rng_checked_scope": ["python", "torch_cpu"],
                           "cuda_rng_state_status": "saved_not_restored_cpu_child",
                           "normalization_restored": "not_applicable",
                           "optimizer_state_entries": len(optimizer.state),
                           "optimizer_parameter_entries": len(optimizer.state_dict()["param_groups"][0]["params"]),
                           "optimizer_state_matches": optimizer_state_matches})
    else:
        raise ValueError("V4_RESTORE_WITNESS_KIND")
    print(json.dumps({"status": "complete", "pid": os.getpid(), "checks": checks}, sort_keys=True, allow_nan=False))
    return None


def _child(request_path: str) -> int:
    _child_request(torch.load(request_path, map_location="cpu", weights_only=False))
    return 0


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    raise SystemExit(_child(sys.argv[2]))
