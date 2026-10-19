"""Target-Linux dynamic acceptance probes for the frozen N2 package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import secrets
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from package_layout import require_package_root


AUTHORIZATION_SCHEMA = "n02-target-linux-acceptance-authorization/1"
RECEIPT_SCHEMA = "n02-target-linux-acceptance-receipt/1"
TARGET_HOST = os.environ.get("N02_TARGET_HOST") or "public-target-placeholder"
TARGET_PYTHON = "/opt/n02-runtime/bin/python"
RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _is_target_interpreter(sys_executable: str) -> bool:
    try:
        return Path(TARGET_PYTHON).resolve(strict=True) == Path(sys_executable).resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, _canonical_bytes(value))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_absolute(path: Path, label: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path.resolve(strict=must_exist)


def _source_manifest(package_root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[path.relative_to(package_root).as_posix()] = _sha256_file(path)
    if not files:
        raise FileNotFoundError("package has no Python source files")
    return files


def _source_manifest_sha256(manifest: dict[str, str]) -> str:
    return _sha256_bytes(_canonical_bytes(manifest))


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    inputs = {
        "package_root": require_package_root(
            _require_absolute(args.package_root, "package root", must_exist=True)
        ),
        "runtime_spec_path": _require_absolute(args.runtime_spec, "runtime spec", must_exist=True),
        "scenario_path": _require_absolute(args.scenario, "scenario", must_exist=True),
        "artifact_root": _require_absolute(args.artifact_root, "artifact root", must_exist=True),
        "world_model_checkpoint_path": _require_absolute(args.world_model_checkpoint, "world-model checkpoint", must_exist=True),
        "authorization_path": _require_absolute(args.authorization, "authorization", must_exist=True),
        "work_root": _require_absolute(args.work_root, "work root", must_exist=False),
        "evidence_root": _require_absolute(args.evidence_root, "evidence root", must_exist=False),
    }
    if inputs["runtime_spec_path"] != inputs["package_root"] / "runtime-spec.json":
        raise PermissionError("runtime spec must be the package-bound runtime-spec.json")
    for output_key in ("work_root", "evidence_root"):
        if inputs[output_key].exists():
            raise FileExistsError(f"{output_key} must be a new path; preserving existing state")
    roots = [inputs[name] for name in ("package_root", "artifact_root", "work_root", "evidence_root")]
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("package, artifact, work, and evidence roots must be disjoint")
    if inputs["artifact_root"] not in inputs["world_model_checkpoint_path"].parents:
        raise ValueError("world-model checkpoint must be contained by the explicit artifact root")
    return inputs


def _validate_runtime_spec(path: Path) -> dict[str, Any]:
    spec = _json_read(path)
    expected = spec.get("expected")
    measurement = spec.get("measurement")
    expected_values = {
        "python": "3.11.16",
        "torch": "2.7.0+cu128",
        "numpy": "1.26.0",
        "device": "cpu",
        "affinity": [0],
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "environment": RUNTIME_ENVIRONMENT,
    }
    if (
        spec.get("schema") != "n02-runtime-spec/1"
        or not isinstance(expected, dict)
        or expected.get("interpreter_path") != TARGET_PYTHON
        or expected.get("interpreter_flag") != "-B"
        or any(expected.get(key) != value for key, value in expected_values.items())
        or not isinstance(measurement, dict)
        or measurement.get("status") != "measured"
        or not isinstance(measurement.get("actual"), dict)
    ):
        raise PermissionError("runtime spec is missing a measured match to the frozen target contract")
    if not _is_target_interpreter(sys.executable):
        raise PermissionError("harness interpreter is not the frozen absolute target Python")
    if sys.flags.dont_write_bytecode != 1:
        raise PermissionError("target acceptance must run with Python -B")
    if not sys.platform.startswith("linux"):
        raise PermissionError("dynamic acceptance is target-Linux-only")
    for name, value in RUNTIME_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise PermissionError(f"target runtime environment mismatch: {name}")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise PermissionError("Linux CPU affinity APIs are required")
    os.sched_setaffinity(0, {0})
    if sorted(os.sched_getaffinity(0)) != [0]:
        raise PermissionError("target process affinity is not CPU0-only")
    return spec


def _validate_torch_runtime(spec: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT},
        "cuda_is_initialized": bool(torch.cuda.is_initialized()),
    }
    measured = spec["measurement"]["actual"]
    if any(measured.get(name) != value for name, value in actual.items()):
        raise PermissionError("runtime spec measurement differs from the current target process")
    if actual["cuda_is_initialized"]:
        raise PermissionError("CUDA initialized during CPU-only acceptance")
    return torch, np, actual


def _authorization_bindings(inputs: dict[str, Path], registry_sha256: str, scenario_sha256: str, source_sha256: str, runtime_sha256: str, checkpoint_sha256: str, training_receipt_sha256: str, data_identity: str) -> dict[str, Any]:
    return {
        "package_root": inputs["package_root"].as_posix(),
        "source_manifest_sha256": source_sha256,
        "work_root": inputs["work_root"].as_posix(),
        "evidence_root": inputs["evidence_root"].as_posix(),
        "runtime_spec_path": inputs["runtime_spec_path"].as_posix(),
        "runtime_spec_sha256": runtime_sha256,
        "input_registry_sha256": registry_sha256,
        "scenario_path": inputs["scenario_path"].as_posix(),
        "scenario_sha256": scenario_sha256,
        "artifact_root": inputs["artifact_root"].as_posix(),
        "world_model_checkpoint_path": inputs["world_model_checkpoint_path"].as_posix(),
        "world_model_checkpoint_sha256": checkpoint_sha256,
        "world_model_training_receipt_sha256": training_receipt_sha256,
        "world_model_data_identity": data_identity,
    }


def _validate_authorization(path: Path, expected_sha256: str, bindings: dict[str, Any]) -> dict[str, Any]:
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("authorization SHA-256 must be lowercase hexadecimal")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PermissionError("authorization file digest differs from the out-of-band supplied digest")
    authorization = _json_read(path)
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized"
        or authorization.get("target_host") != TARGET_HOST
        or authorization.get("scope") != "bounded_dynamic_acceptance"
        or not isinstance(authorization.get("authorization_id"), str)
        or not authorization["authorization_id"].strip()
        or authorization.get("bindings") != bindings
    ):
        raise PermissionError("authorization does not bind this target, package, runtime, scenario, artifact, and output paths")
    return {"authorization_id": authorization["authorization_id"], "sha256": actual_sha256}


def _validate_registered_scenario(package_root: Path, scenario_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    from dataset_protocol import validate_registry

    package_root = require_package_root(package_root)
    data_root = package_root / "registered_data"
    validation = validate_registry(data_root, verify_scenarios=True)
    registry_path = data_root / "input_registry.json"
    registry = _json_read(registry_path)
    try:
        relative = scenario_path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise PermissionError("acceptance scenario must be inside registered_data") from exc
    expected_digest = None
    for entry in registry["parents"]:
        if entry["path"] == relative:
            expected_digest = entry["scenario_sha256"]
            break
    if relative == registry["causal_case"]["path"]:
        expected_digest = registry["causal_case"]["sha256"]
    if expected_digest is None:
        raise PermissionError("acceptance scenario path is not listed in the frozen input registry")
    scenario_bytes = scenario_path.read_bytes()
    scenario_digest = _sha256_bytes(scenario_bytes)
    if scenario_digest != expected_digest:
        raise PermissionError("acceptance scenario digest differs from the registry")
    payload = json.loads(scenario_bytes.decode("utf-8"))
    scenario = payload["scenario"] if relative == registry["causal_case"]["path"] else payload
    if scenario.get("synthetic") is not True:
        raise PermissionError("acceptance scenario is not a registered civilian input")
    return scenario, registry, validation["protocol_id"], scenario_digest


def _validate_training_artifact(inputs: dict[str, Path]) -> tuple[dict[str, Any], str, str]:
    receipt_path = inputs["artifact_root"] / "n2_pipeline" / "world_model_training_receipt.json"
    receipt = _json_read(receipt_path)
    expected_checkpoint = (inputs["artifact_root"] / receipt["checkpoint_path"]).resolve(strict=True)
    checkpoint = inputs["world_model_checkpoint_path"]
    if (
        receipt.get("schema") != "n2-civilian-stage-pipeline/1"
        or receipt.get("stage") != "world_model_training"
        or receipt.get("status") != "complete"
        or expected_checkpoint != checkpoint
        or not isinstance(receipt.get("data_identity"), str)
        or not receipt["data_identity"]
        or _sha256_file(checkpoint) != receipt.get("checkpoint_sha256")
    ):
        raise PermissionError("world-model artifact is not the checkpoint bound by a completed training receipt")
    return receipt, _sha256_file(checkpoint), _sha256_file(receipt_path)


def _new_output_roots(inputs: dict[str, Path]) -> None:
    for name in ("work_root", "evidence_root"):
        inputs[name].mkdir(parents=True, exist_ok=False)
    (inputs["work_root"] / "guardian_normal_failure").mkdir()
    (inputs["work_root"] / "guardian_kill").mkdir()


def _state_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _action_mask_probe(scenario: dict[str, Any], receipt_root: Path) -> dict[str, Any]:
    import torch
    from graph_policy import build_policy
    from relief_env import ReliefEnv
    from stable_identity import bind_decision
    from world_model import complete_action_log_probability

    env = ReliefEnv(scenario)
    env.reset(policy="LEARNED")
    policy_input = env.policy_input()
    policy = build_policy("L1", 8201, device="cpu")
    sample = policy.sample_batch(policy_input, deterministic=True)
    masks = copy.deepcopy(sample["masks"])
    full_actions = [record["selected_id"] for record in masks]
    dispatch_actions = [item for item in full_actions if item != "NOOP"]
    if not masks or any(record["candidate_ids"] != [item["id"] for item in policy_input["candidates"]] for record in masks):
        raise AssertionError("policy masks are not bound to the environment candidate order")
    if full_actions != [record["selected_id"] for record in masks] or dispatch_actions != list(sample["actions"]):
        raise AssertionError("sampled complete action batch and conditional masks disagree")
    evaluated = policy.evaluate_batch(policy_input, dispatch_actions, masks)
    complete_log_probability = complete_action_log_probability(policy, policy_input, full_actions, masks)
    if not torch.equal(evaluated["log_prob"], complete_log_probability):
        raise AssertionError("complete-action score differs from residual-mask evaluation")
    identity = bind_decision(policy_input["observation"], policy_input["candidates"])
    _, _, terminated, truncated, info = env.step(dispatch_actions)
    transition = info["transition"]
    executed_ids = [item["id"] for item in transition["selected_actions"]]
    if (
        executed_ids != dispatch_actions
        or transition.get("observation_sha256") != identity["observation_sha256"]
        or transition.get("candidate_catalog_sha256") != identity["candidate_catalog_sha256"]
        or terminated and truncated
    ):
        raise AssertionError("environment execution differs from the scored public action batch")
    tampered = copy.deepcopy(masks)
    tampered[0]["valid_mask"] = list(tampered[0]["valid_mask"])
    tampered[0]["valid_mask"][0] = not tampered[0]["valid_mask"][0]
    try:
        policy.evaluate_batch(policy_input, dispatch_actions, tampered)
    except ValueError:
        tamper_rejected = True
    else:
        raise AssertionError("policy accepted a tampered conditional action mask")
    result = {
        "status": "complete",
        "candidate_ids": [item["id"] for item in policy_input["candidates"]],
        "full_action_batch": full_actions,
        "executed_action_batch": executed_ids,
        "conditional_mask_count": len(masks),
        "complete_action_log_probability": float(complete_log_probability.detach().cpu().item()),
        "tampered_mask_rejected": tamper_rejected,
        "observation_sha256": identity["observation_sha256"],
        "candidate_catalog_sha256": identity["candidate_catalog_sha256"],
    }
    _durable_json(receipt_root / "action-mask-probe.json", result)
    return {"environment": env, "policy": policy, "policy_input": policy_input, "sample": sample, "receipt": result}


def _hidden_input_probe(scenario: dict[str, Any], policy_input: dict[str, Any], policy: Any, world_model: Any, receipt_root: Path) -> dict[str, Any]:
    import torch
    from graph_policy import encode_public_graph
    from relief_env import ReliefEnv
    from world_model import public_policy_tensors

    altered = copy.deepcopy(scenario)
    hidden = [item for item in altered["tasks"] if not item["known_initial"]]
    if not hidden:
        raise ValueError("registered acceptance scenario must include hidden future information")
    for index, task in enumerate(hidden):
        task["position"] = [float(altered["map"]["cols"] - 1), float(altered["map"]["rows"] - 1)]
        task["deadline"] = min(int(altered["horizon"]), int(task["deadline"]) + 3 + index)
        task["priority"] = max(1.0, float(task["priority"]) - 1.0)
        task["report_delay"] = int(task["report_delay"]) + 1
    first_env = ReliefEnv(scenario)
    second_env = ReliefEnv(altered)
    first_env.reset(policy="LEARNED")
    second_env.reset(policy="LEARNED")
    first_input = first_env.policy_input()
    second_input = second_env.policy_input()
    if first_input != second_input or policy_input != first_input:
        raise AssertionError("hidden scenario changes altered the public policy input")
    if any(task["id"] in json.dumps(first_input, sort_keys=True) for task in hidden):
        raise AssertionError("hidden task identity leaked into public policy input")
    first_graph = encode_public_graph(first_input)
    second_graph = encode_public_graph(second_input)
    first_features, first_binding = public_policy_tensors(first_input)
    second_features, second_binding = public_policy_tensors(second_input)
    with torch.no_grad():
        first_logits = policy(first_input)[1]
        second_logits = policy(second_input)[1]
        first_output = world_model(**first_features, binding=first_binding)
        second_output = world_model(**second_features, binding=second_binding)
    if (
        first_graph.candidate_ids != second_graph.candidate_ids
        or not torch.equal(first_graph.node_features, second_graph.node_features)
        or first_binding != second_binding
        or any(not torch.equal(first_features[key], second_features[key]) for key in first_features)
        or not torch.equal(first_logits, second_logits)
        or not torch.equal(first_output["global"], second_output["global"])
        or not torch.equal(first_output["per_task"], second_output["per_task"])
    ):
        raise AssertionError("actor or world-model output depends on hidden scenario fields")
    result = {
        "status": "complete",
        "hidden_task_count": len(hidden),
        "public_input_equal": True,
        "actor_logits_equal": True,
        "world_model_global_equal": True,
        "world_model_task_equal": True,
        "candidate_catalog_equal": True,
    }
    _durable_json(receipt_root / "hidden-input-probe.json", result)
    return result


def _fresh_process_restore_probe(package_root: Path, checkpoint: Path, data_identity: str, policy_input: dict[str, Any], expected_world_model: Any, work_root: Path, evidence_root: Path) -> dict[str, Any]:
    from world_model import public_policy_tensors

    request_path = work_root / "restore-request.json"
    receipt_path = evidence_root / "fresh-process-restore.json"
    _durable_json(request_path, {"policy_input": policy_input})
    features, binding = public_policy_tensors(policy_input)
    with __import__("torch").no_grad():
        expected = expected_world_model(**features, binding=binding)
    expected_hashes = {
        name: _sha256_bytes(expected[name].detach().cpu().contiguous().numpy().tobytes())
        for name in ("global", "per_task")
    }
    child_token = secrets.token_hex(32)
    environment = {**os.environ, "N2_ACCEPTANCE_CHILD_TOKEN": child_token}
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--_restore-child",
        "--_child-token",
        child_token,
        "--_package-root",
        str(package_root),
        "--_checkpoint",
        str(checkpoint),
        "--_data-identity",
        data_identity,
        "--_request",
        str(request_path),
        "--_receipt",
        str(receipt_path),
    ]
    completed = subprocess.run(command, cwd=package_root, env=environment, capture_output=True, text=True, check=False, timeout=300)
    if completed.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError(f"fresh-process restore failed ({completed.returncode}): {completed.stderr[-2000:]}")
    restored = _json_read(receipt_path)
    if (
        restored.get("status") != "complete"
        or restored.get("pid") == os.getpid()
        or restored.get("checkpoint_sha256") != _sha256_file(checkpoint)
        or restored.get("prediction_hashes") != expected_hashes
        or restored.get("torch_threads") != 1
        or restored.get("torch_interop_threads") != 1
        or restored.get("cuda_is_initialized") is not False
    ):
        raise AssertionError("fresh-process restore did not reproduce the actual artifact predictions")
    request_path.unlink()
    return restored


def _restore_child(args: argparse.Namespace) -> int:
    if os.environ.get("N2_ACCEPTANCE_CHILD_TOKEN") != args.child_token or not args.child_token:
        raise PermissionError("fresh-process restore child requires its parent token")
    if not sys.platform.startswith("linux") or not _is_target_interpreter(sys.executable):
        raise PermissionError("fresh-process restore child is outside the accepted target runtime")
    for name, value in RUNTIME_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise PermissionError(f"restore child runtime environment mismatch: {name}")
    os.sched_setaffinity(0, {0})
    args.package_root = require_package_root(args.package_root)
    sys.path.insert(0, str(args.package_root))
    import torch
    from world_model import load_consequence_checkpoint, public_policy_tensors

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, _payload = load_consequence_checkpoint(args.checkpoint, data_identity=args.data_identity)
    request = _json_read(args.request)
    features, binding = public_policy_tensors(request["policy_input"])
    with torch.no_grad():
        output = model(**features, binding=binding)
    receipt = {
        "schema": "n2-fresh-process-wm-restore/1",
        "status": "complete",
        "pid": os.getpid(),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "model_state_sha256": model_state_digest(model),
        "prediction_hashes": {
            name: _sha256_bytes(output[name].detach().cpu().contiguous().numpy().tobytes())
            for name in ("global", "per_task")
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_is_initialized": bool(torch.cuda.is_initialized()),
    }
    _durable_json(args.receipt, receipt)
    return 0


def model_state_digest(model: Any) -> str:
    from world_model import _model_state_digest

    return _model_state_digest(model.state_dict())


def _critic_update_probe(scenario: dict[str, Any], world_model: Any, action_probe: dict[str, Any], receipt_root: Path, data_identity: str) -> dict[str, Any]:
    import torch
    from graph_policy import build_policy
    from n2_stage_pipeline import _policy_proposals
    from world_model import (
        CriticAugmentedGraphPolicy,
        CriticOnlyGppoAdapter,
        GppoEngineeringTrainer,
        assert_paired_actor_initialization,
    )

    base_g = build_policy("L1", 8201, device="cpu")
    base_gwm = build_policy("L1", 8201, device="cpu")
    policy_g = CriticAugmentedGraphPolicy(base_g, world_model, CriticOnlyGppoAdapter(), wm_enabled=False)
    policy_gwm = CriticAugmentedGraphPolicy(base_gwm, world_model, CriticOnlyGppoAdapter(), wm_enabled=True)
    env = action_probe["environment"]
    policy_input = copy.deepcopy(action_probe["policy_input"])
    policy_input["proposal_batches"] = _policy_proposals(base_gwm, env, policy_input)
    assert_paired_actor_initialization(policy_g, policy_gwm, policy_input)
    with torch.no_grad():
        baseline_value = base_gwm.value(policy_input).reshape(())
        initial_value = policy_gwm.evaluate_batch(
            policy_input,
            list(action_probe["sample"]["actions"]),
            copy.deepcopy(action_probe["sample"]["masks"]),
        )["value"].reshape(())
    if not torch.equal(baseline_value, initial_value):
        raise AssertionError("zero-initialized WM adapter changed the initial critic value")
    world_model_before = _state_digest(world_model)
    adapter_before = _state_digest(policy_gwm.adapter)
    trainer = GppoEngineeringTrainer(policy_gwm, data_identity=data_identity)
    dispatch_proposal = next(
        (item for item in policy_input["proposal_batches"] if item["actions"] != ["NOOP"]),
        None,
    )
    if dispatch_proposal is None:
        raise ValueError("acceptance scenario has no registered legal dispatch proposal")
    dispatch_actions = [item for item in dispatch_proposal["actions"] if item != "NOOP"]
    dispatch_masks = copy.deepcopy(dispatch_proposal["masks"])
    old_log_probability = base_gwm.evaluate_batch(
        policy_input,
        dispatch_actions,
        dispatch_masks,
    )["log_prob"].detach()
    target = baseline_value.detach() + 1.0
    update = trainer.update(
        policy_input,
        action_batch=dispatch_actions,
        action_masks=dispatch_masks,
        old_log_probability=old_log_probability,
        advantage=torch.zeros((), dtype=torch.float32),
        return_target=target,
        batch_index=0,
    )
    gradients = [parameter.grad for parameter in policy_gwm.adapter.parameters() if parameter.grad is not None]
    if (
        not gradients
        or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        or sum(float(gradient.abs().sum().item()) for gradient in gradients) <= 0.0
        or adapter_before == _state_digest(policy_gwm.adapter)
        or world_model_before != _state_digest(world_model)
        or any(parameter.requires_grad or parameter.grad is not None for parameter in world_model.parameters())
        or update.get("world_model_frozen") is not True
    ):
        raise AssertionError("critic update did not train the adapter while keeping WM frozen")
    result = {
        "status": "complete",
        "paired_actor_initialization_equal": True,
        "initial_critic_value_equal": True,
        "critic_adapter_updated": True,
        "critic_gradient_l1": sum(float(gradient.abs().sum().item()) for gradient in gradients),
        "world_model_state_sha256_before": world_model_before,
        "world_model_state_sha256_after": _state_digest(world_model),
        "world_model_frozen": True,
        "optimizer_updates": trainer.update_index,
        "proposal_binding": update["proposal_binding"],
    }
    _durable_json(receipt_root / "critic-update-probe.json", result)
    return result


def _run_guarded_child(args: argparse.Namespace) -> int:
    guardian_token = os.environ.get("RELIEF_GRAPH_GUARDIAN_TOKEN")
    if not guardian_token or guardian_token != args.guardian_token or not args.guarded_pipeline:
        raise PermissionError("guardian test child requires its parent guardian token")
    if not sys.platform.startswith("linux") or not _is_target_interpreter(sys.executable):
        raise PermissionError("guardian test child is outside the accepted target runtime")
    args.package_root = require_package_root(args.package_root)
    sys.path.insert(0, str(args.package_root))
    from resource_supervisor import (
        CARRY_FORWARD_RECEIPT_NAME,
        DRIVER_STAGE_SCHEMA,
        EXPORT_DIR_NAME,
        ResourceSupervisor,
        STAGE_ORDER,
        STAGE_PROTOCOL_ID,
    )

    work_root = args.work_root
    ledger_path = work_root / "resource-ledger.jsonl"
    supervisor = ResourceSupervisor(
        STAGE_ORDER[0],
        ledger_path,
        work_root,
        work_root / EXPORT_DIR_NAME,
        args.runtime_identity,
        prior_cost_path=args.package_root / CARRY_FORWARD_RECEIPT_NAME,
    )
    supervisor.start_monitor()
    receipt_path = work_root / "receipts" / f"01-{STAGE_ORDER[0]}.json"
    supervisor.select_stage(STAGE_ORDER[0])
    token = supervisor.reserve_work_call(
        "output_write",
        1,
        {"component": "acceptance_guardian_probe", "probe": args.child_mode},
    )
    if args.child_mode == "normal_failure":
        error = RuntimeError("acceptance_controlled_normal_failure")
        supervisor.fail_work_call(token, error)
        entry = supervisor.finish(status="failed", error=error, metadata={"receipt": receipt_path.as_posix()})
        _durable_json(receipt_path, {
            "schema": DRIVER_STAGE_SCHEMA,
            "stage_protocol": STAGE_PROTOCOL_ID,
            "stage": STAGE_ORDER[0],
            "status": "failed",
            "runtime_identity_sha256": args.runtime_identity,
            "release_digest": args.runtime_identity,
            "first_error": "RuntimeError: acceptance_controlled_normal_failure",
            "partial_checkpoints": [],
            "resource_entry": entry,
        })
        supervisor.close(error=error)
        return 1
    if args.child_mode == "guardian_kill":
        _durable_json(work_root / "guardian-kill-ready.json", {
            "schema": "n2-acceptance-guardian-kill-ready/1",
            "status": "pending_work_call_committed",
            "work_call_token": token,
            "pid": os.getpid(),
            "runtime_identity_sha256": args.runtime_identity,
        })
        while True:
            time.sleep(0.05)
    raise ValueError("unknown guarded child mode")


def _guarded_child_entry(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_acceptance-child-mode", choices=("normal_failure", "guardian_kill"), required=True)
    parser.add_argument("--_acceptance-runtime-identity", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--_guardian-token", required=True)
    parser.add_argument("--_guarded-pipeline", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return _run_guarded_child(args)


def _guarded_probe_arguments(mode: str, inputs: dict[str, Path], identity: str, authorization_sha256: str) -> list[str]:
    return [
        "--run",
        "--package-root", str(inputs["package_root"]),
        "--work-root", str(inputs["work_root"] / ("guardian_normal_failure" if mode == "normal_failure" else "guardian_kill")),
        "--acceptance-evidence-root", str(inputs["evidence_root"]),
        "--acceptance-runtime-spec", str(inputs["runtime_spec_path"]),
        "--acceptance-scenario", str(inputs["scenario_path"]),
        "--acceptance-artifact-root", str(inputs["artifact_root"]),
        "--acceptance-authorization", str(inputs["authorization_path"]),
        "--acceptance-authorization-sha256", authorization_sha256,
        "--_acceptance-child-mode", mode,
        "--_acceptance-runtime-identity", identity,
    ]


def _run_guarded_failure_probe(mode: str, inputs: dict[str, Path], identity: str, authorization_sha256: str) -> dict[str, Any]:
    from process_guardian import run_guarded_pipeline
    from resource_supervisor import STAGE_ORDER

    work_root = inputs["work_root"] / ("guardian_normal_failure" if mode == "normal_failure" else "guardian_kill")
    guardian = __import__("process_guardian")
    original_limits = guardian.STAGE_LIMITS
    if mode == "guardian_kill":
        bounded = dict(original_limits)
        first_wall, first_cpu = bounded[STAGE_ORDER[0]]
        bounded[STAGE_ORDER[0]] = (min(float(first_wall), 10.0), min(float(first_cpu), 60.0))
        guardian.STAGE_LIMITS = bounded
    try:
        return_code = run_guarded_pipeline(
            argv=_guarded_probe_arguments(mode, inputs, identity, authorization_sha256),
            driver_path=Path(__file__).resolve(),
            package_root=inputs["package_root"],
            work_root=work_root,
            runtime_identity_sha256=identity,
            runtime_environment=RUNTIME_ENVIRONMENT,
            preflight_only=False,
        )
    finally:
        guardian.STAGE_LIMITS = original_limits
    receipt_path = work_root / "receipts" / f"01-{STAGE_ORDER[0]}.json"
    guardian_path = work_root / "guardians" / f"01-{STAGE_ORDER[0]}.json"
    receipt = _json_read(receipt_path)
    guardian_receipt = _json_read(guardian_path)
    if return_code == 0 or receipt.get("status") != "failed" or not receipt.get("first_error"):
        raise AssertionError(f"{mode} did not produce the expected failed stage receipt")
    if mode == "normal_failure":
        if receipt.get("first_error") != "RuntimeError: acceptance_controlled_normal_failure" or guardian_receipt.get("outer_guardian_kill_reason") is not None:
            raise AssertionError("normal worker failure was replaced by an outer guardian failure")
    elif not guardian_receipt.get("outer_guardian_kill_reason"):
        raise AssertionError("controlled guardian kill did not record its kill reason")
    database = sqlite3.connect((work_root / "work-calls.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        integrity = database.execute("PRAGMA integrity_check").fetchone()
        statuses = database.execute("SELECT status, COUNT(*) FROM work_calls GROUP BY status").fetchall()
    finally:
        database.close()
    if integrity != ("ok",):
        raise AssertionError("guardian failure left an invalid SQLite ledger")
    counts = {status: count for status, count in statuses}
    if mode == "normal_failure":
        if counts.get("failed") != 1 or counts.get("pending", 0):
            raise AssertionError("normal failure did not durably settle its failed work call")
    else:
        ready_path = work_root / "guardian-kill-ready.json"
        if not ready_path.is_file() or counts.get("pending") != 1 or counts.get("failed", 0):
            raise AssertionError("guardian kill did not preserve the committed pending work call")
    return {
        "status": "expected_failure_recorded",
        "return_code": return_code,
        "first_error": receipt["first_error"],
        "outer_guardian_kill_reason": guardian_receipt.get("outer_guardian_kill_reason"),
        "work_call_status_counts": counts,
        "sqlite_integrity": integrity[0],
        "receipt_path": receipt_path.relative_to(work_root).as_posix(),
        "guardian_path": guardian_path.relative_to(work_root).as_posix(),
        "ready_marker_preserved": (work_root / "guardian-kill-ready.json").is_file(),
    }


def _controlled_failure_export(source: Path, destination: Path) -> dict[str, Any]:
    from infra_io import controlled_export

    result = controlled_export(source, destination)
    manifest = _json_read(destination / "export-manifest.json")
    completion = _json_read(destination / "EXPORT_COMPLETE.json")
    if (
        result.get("verified") is not True
        or manifest.get("verified") is not True
        or completion.get("verified") is not True
        or _sha256_file(destination / "export-manifest.json") != completion.get("export_manifest_sha256")
    ):
        raise AssertionError("controlled failure export did not verify its manifest and completion marker")
    return {
        "status": "complete",
        "file_count": completion["file_count"],
        "payload_bytes": completion["payload_bytes"],
        "manifest_sha256": completion["export_manifest_sha256"],
        "completion_sha256": _sha256_file(destination / "EXPORT_COMPLETE.json"),
    }


def _resource_measurement(start_wall: float, start_cpu: float, child_usage: Any) -> dict[str, float]:
    current_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        current_children.ru_utime + current_children.ru_stime
        - child_usage.ru_utime - child_usage.ru_stime
    )
    own_cpu = time.process_time() - start_cpu
    return {
        "wall_seconds": max(0.0, time.monotonic() - start_wall),
        "cpu_seconds": max(0.0, own_cpu + child_cpu),
        "parent_cpu_seconds": max(0.0, own_cpu),
        "waited_child_cpu_seconds": max(0.0, child_cpu),
        "source": "time.monotonic/process_time plus RUSAGE_CHILDREN; target worker/guardian descendants are included after wait",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--runtime-spec", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    return parser


def _run_parent(args: argparse.Namespace) -> int:
    if not sys.platform.startswith("linux"):
        raise PermissionError("dynamic acceptance is target-Linux-only")
    inputs = _resolve_inputs(args)
    runtime_spec = _validate_runtime_spec(inputs["runtime_spec_path"])
    source_manifest = _source_manifest(inputs["package_root"])
    source_digest = _source_manifest_sha256(source_manifest)
    registry_path = inputs["package_root"] / "registered_data" / "input_registry.json"
    registry_digest = _sha256_file(registry_path)
    scenario_digest = _sha256_file(inputs["scenario_path"])
    training_receipt, checkpoint_digest, training_receipt_digest = _validate_training_artifact(inputs)
    bindings = _authorization_bindings(
        inputs,
        registry_digest,
        scenario_digest,
        source_digest,
        _sha256_file(inputs["runtime_spec_path"]),
        checkpoint_digest,
        training_receipt_digest,
        training_receipt["data_identity"],
    )
    authorization = _validate_authorization(inputs["authorization_path"], args.authorization_sha256, bindings)
    scenario, registry, protocol_id, validated_scenario_digest = _validate_registered_scenario(inputs["package_root"], inputs["scenario_path"])
    if validated_scenario_digest != scenario_digest or registry_digest != _sha256_file(registry_path):
        raise PermissionError("registered inputs changed during acceptance admission")
    torch, np, actual_runtime = _validate_torch_runtime(runtime_spec)
    del np

    runtime_identity = _sha256_bytes(_canonical_bytes({
        "schema": "n02-acceptance-runtime-identity/1",
        "bindings": bindings,
        "authorization_id": authorization["authorization_id"],
    }))
    _new_output_roots(inputs)
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    evidence_root = inputs["evidence_root"]
    results: dict[str, Any] = {}
    try:
        sys.path.insert(0, str(inputs["package_root"]))
        from world_model import load_consequence_checkpoint

        world_model, checkpoint_payload = load_consequence_checkpoint(
            inputs["world_model_checkpoint_path"],
            data_identity=training_receipt["data_identity"],
        )
        strict_restore = training_receipt.get("strict_restore_subprocess", {})
        if checkpoint_payload.get("model_state_sha256") != strict_restore.get("model_state_sha256"):
            raise PermissionError("checkpoint model digest differs from its completed training receipt")
        world_model.eval()

        action_probe = _action_mask_probe(scenario, evidence_root)
        results["action_mask_identity"] = action_probe["receipt"]
        results["hidden_input_invariance"] = _hidden_input_probe(
            scenario,
            action_probe["policy_input"],
            action_probe["policy"],
            world_model,
            evidence_root,
        )
        restore = _fresh_process_restore_probe(
            inputs["package_root"],
            inputs["world_model_checkpoint_path"],
            training_receipt["data_identity"],
            action_probe["policy_input"],
            world_model,
            inputs["work_root"],
            evidence_root,
        )
        results["fresh_process_restore"] = restore
        results["critic_update_frozen_wm"] = _critic_update_probe(
            scenario,
            world_model,
            action_probe,
            evidence_root,
            training_receipt["data_identity"],
        )
        results["normal_failure"] = _run_guarded_failure_probe(
            "normal_failure", inputs, runtime_identity, args.authorization_sha256,
        )
        results["normal_failure_export"] = _controlled_failure_export(
            inputs["work_root"] / "guardian_normal_failure",
            evidence_root / "normal-failure-export",
        )
        results["guardian_kill"] = _run_guarded_failure_probe(
            "guardian_kill", inputs, runtime_identity, args.authorization_sha256,
        )
        results["guardian_kill_export"] = _controlled_failure_export(
            inputs["work_root"] / "guardian_kill",
            evidence_root / "guardian-kill-export",
        )
        post_source_digest = _source_manifest_sha256(_source_manifest(inputs["package_root"]))
        if post_source_digest != source_digest:
            raise PermissionError("package source changed while acceptance probes were running")
        measurement = _resource_measurement(start_wall, start_cpu, child_usage)
        if not all(isinstance(value, dict) and value.get("status") in {"complete", "expected_failure_recorded"} for value in results.values()):
            raise AssertionError("one or more dynamic acceptance probes did not complete")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "target_host": TARGET_HOST,
            "observed_hostname": platform.node(),
            "authorization": authorization,
            "runtime_identity_sha256": runtime_identity,
            "runtime_spec_sha256": _sha256_file(inputs["runtime_spec_path"]),
            "runtime": actual_runtime,
            "protocol_id": protocol_id,
            "input_registry_sha256": registry_digest,
            "scenario_path": inputs["scenario_path"].as_posix(),
            "scenario_id": scenario.get("scenario_id"),
            "scenario_sha256": scenario_digest,
            "source_manifest_sha256": source_digest,
            "source_files": source_manifest,
            "world_model_checkpoint_sha256": checkpoint_digest,
            "world_model_training_receipt_sha256": training_receipt_digest,
            "world_model_data_identity": training_receipt["data_identity"],
            "work_root": inputs["work_root"].as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "artifact_root": inputs["artifact_root"].as_posix(),
            "resource_measurement": measurement,
            "probes": results,
        }
        _durable_json(evidence_root / "acceptance-receipt.json", receipt)
        return 0
    except BaseException as exc:
        _append_jsonl(evidence_root / "acceptance-first-errors.jsonl", {
            "schema": "n2-acceptance-first-error/1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_identity_sha256": runtime_identity,
            "completed_probes": results,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--_acceptance-child-mode" in arguments:
        return _guarded_child_entry(arguments)
    if "--_restore-child" in arguments:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--_restore-child", action="store_true")
        parser.add_argument("--_child-token", required=True)
        parser.add_argument("--_package-root", dest="package_root", type=Path, required=True)
        parser.add_argument("--_checkpoint", type=Path, required=True)
        parser.add_argument("--_data-identity", required=True)
        parser.add_argument("--_request", type=Path, required=True)
        parser.add_argument("--_receipt", type=Path, required=True)
        child_args = parser.parse_args(arguments)
        child_args.package_root = require_package_root(child_args.package_root)
        return _restore_child(child_args)
    args = _parser().parse_args(arguments)
    try:
        return _run_parent(args)
    except BaseException as exc:
        print(f"N2 target-Linux acceptance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
