"""Checkpoint save/restore comparison used after real training is authorised."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .contracts import CheckpointRestoreReceipt


def _digest(value: Any) -> str:
    try:
        import torch
        tensors = value if isinstance(value, dict) else value.state_dict()
        chunks = []
        for key in sorted(tensors):
            chunks.append(key.encode())
            chunks.append(tensors[key].detach().cpu().numpy().tobytes())
        return hashlib.sha256(b"".join(chunks)).hexdigest()
    except (ImportError, AttributeError, TypeError):
        return hashlib.sha256(repr(value).encode()).hexdigest()


def save_checkpoint(path: str | Path, model: Any, metadata: dict[str, Any]) -> str:
    """Save a model state and metadata; no optimizer step is performed."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata}
    # Pass a Python file object so PyTorch's zip writer does not require an
    # ASCII-only Windows path (the workspace path may contain Chinese text).
    with destination.open("wb") as stream:
        torch.save(payload, stream)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


def _tree_digest(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())

    visit(value)
    return digest.hexdigest()


def capture_rng_state(torch: Any) -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def restore_rng_state(torch: Any, state: dict[str, Any], device: str = "cpu") -> None:
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if str(device).startswith("cuda") and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    metadata: dict[str, Any],
    probe_inputs: tuple[Any, ...],
    *,
    device: str,
) -> str:
    """Save complete training state plus a fixed model-output probe."""
    import torch

    required = {"seed", "dataset_sha256", "feature_names", "event_names",
                "event_schema_version", "architecture", "optimizer", "normalization", "device"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata missing required fields: {', '.join(missing)}")
    if not isinstance(metadata["dataset_sha256"], str) or len(metadata["dataset_sha256"]) != 64:
        raise ValueError("dataset_sha256 must be a SHA-256 hex digest")
    if metadata.get("normalization", {}).get("scope") != "train_split_only":
        raise ValueError("checkpoint normalization must be fitted on the train split only")
    if not isinstance(metadata.get("architecture"), dict) or not metadata["architecture"]:
        raise ValueError("checkpoint architecture metadata is required")

    prior_training = model.training
    model.eval()
    with torch.no_grad():
        expected = {name: value.detach().cpu() for name, value in model(*probe_inputs).items()}
    model.train(prior_training)
    payload = {
        "schema": "wm_training_checkpoint_v2",
        "model_state": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_sha256": _tree_digest(optimizer.state_dict()),
        "metadata": dict(metadata),
        "probe_inputs": tuple(_cpu_tree(value) for value in probe_inputs),
        "probe_predictions": expected,
        "rng_state": capture_rng_state(torch),
        "saved_device": str(device),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        torch.save(payload, stream)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def restore_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    device: str = "cpu",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Restore model/optimizer/RNG and compare the saved fixed-input outputs."""
    import torch

    with Path(path).open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wm_training_checkpoint_v2":
        raise ValueError("unsupported world-model checkpoint schema")
    required_rng = {"python", "torch_cpu", "numpy", "torch_cuda"}
    if not required_rng.issubset(payload.get("rng_state", {})):
        raise ValueError("checkpoint RNG state is incomplete")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("architecture"), dict):
        raise ValueError("checkpoint architecture metadata is missing")
    if set(payload.get("probe_predictions", {})) != {"dynamic", "gain"}:
        raise ValueError("checkpoint fixed prediction probe is incomplete")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(torch, payload["rng_state"], device)
    model.eval()
    inputs = tuple(value.to(device) if isinstance(value, torch.Tensor) else value
                   for value in payload["probe_inputs"])
    with torch.no_grad():
        actual = model(*inputs)
    differences = {name: float((actual[name].detach().cpu() - expected).abs().max().item())
                   for name, expected in payload["probe_predictions"].items()}
    predictions_equal = all(torch.allclose(actual[name].detach().cpu(), expected,
                                           atol=atol, rtol=rtol)
                            for name, expected in payload["probe_predictions"].items())
    optimizer_hash = _tree_digest(optimizer.state_dict())
    optimizer_equal = optimizer_hash == payload["optimizer_sha256"]
    return {"status": "PASS" if predictions_equal and optimizer_equal else "FAIL",
            "predictions_equal": predictions_equal, "optimizer_equal": optimizer_equal,
            "optimizer_sha256": optimizer_hash, "max_abs_error": differences,
            "tolerance": {"atol": atol, "rtol": rtol}, "device": str(device),
            "metadata": metadata}


def verify_training_checkpoint_in_fresh_process(
    path: str | Path, *, device: str = "cpu"
) -> dict[str, Any]:
    """Run the restore probe in a distinct Python process."""
    checkpoint = Path(path).resolve()
    package_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(package_root) + (os.pathsep + existing if existing else "")
    process = subprocess.run(
        [sys.executable, "-m", "wm_ppo_v1.restore_worker", "--checkpoint", str(checkpoint),
         "--device", device], cwd=package_root, env=environment,
        capture_output=True, text=True, check=False)
    if process.returncode != 0:
        return {"status": "FAIL", "return_code": process.returncode,
                "stderr": process.stderr[-2000:]}
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "FAIL", "reason": "invalid_fresh_process_receipt",
                "stdout": process.stdout[-1000:]}


def restore_and_compare(
    path: str | Path,
    builder: Callable[[], Any],
    inputs: tuple[Any, ...],
    device: str = "cpu",
) -> CheckpointRestoreReceipt:
    """Load a fresh model and compare deterministic predictions."""

    import torch

    with Path(path).open("rb") as stream:
        payload = torch.load(stream, map_location=device, weights_only=False)
    first = builder().to(device)
    first.load_state_dict(payload["state_dict"])
    second = builder().to(device)
    second.load_state_dict(payload["state_dict"])
    first.eval()
    second.eval()
    with torch.no_grad():
        left = first(*inputs)
        right = second(*inputs)
    equal = all(torch.equal(left[key], right[key]) for key in left)
    return CheckpointRestoreReceipt(
        status="PASS" if equal else "FAIL",
        model_kind=type(first).__name__,
        checkpoint_path=str(path),
        before_digest=_digest(first),
        after_digest=_digest(second),
        predictions_equal=equal,
        device=device,
        reason=None if equal else "PREDICTION_MISMATCH_AFTER_RESTORE",
    )


def write_receipt(path: str | Path, receipt: CheckpointRestoreReceipt) -> None:
    Path(path).write_text(json.dumps(receipt.__dict__, indent=2, sort_keys=True), encoding="utf-8")
