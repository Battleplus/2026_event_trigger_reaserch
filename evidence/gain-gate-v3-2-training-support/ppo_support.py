"""External paired G/G-WM PPO adapter for committed V3 scenario objects.

The frozen event_policy module supplies the policy, real rollout, GAE and PPO
minibatch update functions. This file owns only the 84/84 data adapter,
two-arm paired schedule, CUDA audits, and fresh-process checkpoint probe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
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

import torch


SUPPORT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SUPPORT_DIR / "ppo_config.json"
SEEDS = (8201, 8202, 8203)
VARIANTS = ("Periodic-G", "Periodic-G-WM")
policy_module: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _torch_payload_hash(value: Any) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _cuda_timer_start() -> tuple[float, float]:
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    return time.perf_counter(), time.process_time()


def _cuda_timer_stop(start: tuple[float, float]) -> dict[str, Any]:
    torch.cuda.synchronize(0)
    return {"elapsed_wall_seconds": time.perf_counter() - start[0],
            "process_cpu_seconds": time.process_time() - start[1],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def _load_source(legacy_source: str | Path) -> tuple[Any, Any]:
    global policy_module
    root = Path(legacy_source).resolve(strict=True) / "package" / "civilian"
    if not root.is_dir():
        raise FileNotFoundError(f"legacy civilian source not found: {root}")
    sys.path.insert(0, str(root))
    for module in ("event_policy", "event_models"):
        sys.modules.pop(module, None)
    policy = importlib.import_module("event_policy")
    policy_module = policy
    models = importlib.import_module("event_models")
    return policy, models


def _load_gain_augmented_ensemble(paths: Sequence[str | Path], legacy_source: str | Path, device: torch.device, event_models: Any) -> Any:
    if len(paths) != 3:
        raise ValueError("PPO_SUPPORT_THREE_TRAINED_WM_CHECKPOINTS_REQUIRED")
    training_support = importlib.import_module("training_support")

    class ActionHeadAdapter(torch.nn.Module):
        def __init__(self, core: torch.nn.Module):
            super().__init__()
            self.core = core

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return self.core.predict_action_outcomes(batch)

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name.startswith(("target_mean_", "target_scale_")):
                    return getattr(self.core.base, name)
                raise

    members = []
    identities = []
    for index, value in enumerate(paths):
        path = Path(value).resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "n02-gain-augmented-world-model-checkpoint/1":
            raise ValueError("PPO_SUPPORT_AUGMENTED_WM_CHECKPOINT_SCHEMA")
        expected_seed = SEEDS[index]
        if int(payload.get("seed", -1)) != expected_seed:
            raise ValueError("PPO_SUPPORT_AUGMENTED_WM_SEED_ORDER")
        model = training_support.GainAugmentedWorldModel(event_models, payload["head_dims"], hidden_dim=int(payload["hidden_dim"]))
        model.base.set_normalization(payload["normalization"])
        model.set_gain_normalization(*payload["gain_normalization"])
        model.load_state_dict(payload["model_state"], strict=True)
        if training_support._torch_state_sha256(model.state_dict()) != payload.get("model_state_sha256"):
            raise ValueError("PPO_SUPPORT_AUGMENTED_WM_STATE_HASH")
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        member = ActionHeadAdapter(model).to(device).eval()
        for parameter in member.parameters():
            parameter.requires_grad_(False)
        members.append(member)
        identities.append(_sha256(path))

    class FrozenEnsemble:
        def __init__(self, models: list[torch.nn.Module], file_hashes: list[str]):
            self.models = models
            self._identity = hashlib.sha256("".join(file_hashes).encode("ascii")).hexdigest()
            self.forward_calls = 0
            self.prediction_calls = 0

        @property
        def ensemble_hash(self) -> str:
            return self._identity

        def __iter__(self):
            return iter(self.models)

    return FrozenEnsemble(members, identities)


def validate_inputs(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only shape/schema gate; performs no model construction or optimizer work."""
    if config.get("schema") != "n02-v3-paired-g-ppo-config/1":
        raise ValueError("PPO_SUPPORT_CONFIG_SCHEMA")
    if tuple(config.get("seeds", ())) != SEEDS or tuple(config.get("variants", ())) != VARIANTS:
        raise ValueError("PPO_SUPPORT_FROZEN_PAIR_CONFIG")
    train, validation = payload.get("train"), payload.get("validation")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("PPO_SUPPORT_SPLIT_ARRAYS_REQUIRED")
    if len(train) != 84 or len(validation) != 84:
        raise ValueError("PPO_SUPPORT_EXPECTED_84_TRAIN_84_VALIDATION")
    if any(not isinstance(row, dict) or row.get("split") != "train" for row in train):
        raise ValueError("PPO_SUPPORT_TRAIN_SCENARIO_SPLIT")
    if any(not isinstance(row, dict) or row.get("split") != "validation" for row in validation):
        raise ValueError("PPO_SUPPORT_VALIDATION_SCENARIO_SPLIT")
    def parent(row: Mapping[str, Any]) -> str:
        return str(row.get("parent_id", row.get("scenario_id", "")))
    train_ids = [parent(row) for row in train]
    val_ids = [parent(row) for row in validation]
    if any(not value for value in train_ids + val_ids):
        raise ValueError("PPO_SUPPORT_PARENT_ID_REQUIRED")
    if len(set(train_ids)) != 84 or len(set(val_ids)) != 84:
        raise ValueError("PPO_SUPPORT_PARENT_IDS_NOT_UNIQUE")
    if set(train_ids) & set(val_ids):
        raise ValueError("PPO_SUPPORT_TRAIN_VALIDATION_PARENT_OVERLAP")
    if not isinstance(payload.get("threshold_manifest"), dict):
        raise ValueError("PPO_SUPPORT_THRESHOLD_MANIFEST_REQUIRED")
    return {
        "train_count": len(train), "validation_count": len(validation),
        "train_parent_ids_sha256": _json_hash(train_ids),
        "validation_parent_ids_sha256": _json_hash(val_ids),
        "train_validation_disjoint": True,
    }


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _update_policy_cuda(
    policy: Any, optimizer: torch.optim.Optimizer, records: list[dict[str, Any]], *,
    seed: int, epochs: int, minibatch_size: int, updates_per_epoch: int,
    learning_rate: float, clip_ratio: float, entropy_coefficient: float,
    update_offset: int = 0,
) -> list[dict[str, Any]]:
    """CUDA-safe adapter of frozen event_policy._update_policy (CPU indices, CUDA tensors)."""
    if len(records) < minibatch_size * updates_per_epoch:
        raise ValueError("PPO_SUPPORT_SHARED_UPDATE_SAMPLE_COUNT")
    device = next(policy.parameters()).device
    advantages = torch.tensor([row["advantage"] for row in records], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([row["return"] for row in records], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([row["old_log_probability"] for row in records], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 491)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(records), generator=generator)
        for batch_index in range(updates_per_epoch):
            indices = permutation[batch_index * minibatch_size:(batch_index + 1) * minibatch_size].tolist()
            evaluated = [policy.evaluate_action(
                records[index]["observation"], records[index]["candidates"],
                scope_index=int(records[index]["scope_index"]), candidate_index=int(records[index]["candidate_index"]),
                high_level_mask=records[index]["high_level_mask"], candidate_mask=None,
                wm_predictions=records[index].get("wm_predictions"),
            ) for index in indices]
            batch_logp = torch.stack([row["log_probability"] for row in evaluated])
            batch_values = torch.stack([row["value"] for row in evaluated])
            batch_aux = torch.stack([row["auxiliary_value"] for row in evaluated])
            batch_index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            ratio = torch.exp(batch_logp - old_log_probs[batch_index_tensor])
            if epoch == 0 and batch_index == 0 and float(torch.max(torch.abs(ratio.detach() - 1.0)).item()) > 2e-5:
                raise AssertionError("PPO_SUPPORT_INITIAL_LOGPROB_RATIO")
            unclipped = ratio * advantages[batch_index_tensor]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_index_tensor]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(batch_values - returns[batch_index_tensor]).mean()
            entropy = torch.stack([row["entropy"] for row in evaluated]).mean()
            loss = actor_loss + value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("PPO_SUPPORT_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError("PPO_SUPPORT_EMPTY_OR_NONFINITE_GRADIENT")
            auxiliary_parameters = list(policy.auxiliary_critic.parameters()) if policy.auxiliary_critic is not None else []
            auxiliary_gradient_norm = math.sqrt(sum(float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in auxiliary_parameters if parameter.grad is not None))
            if policy.use_wm_aux and auxiliary_gradient_norm <= 0:
                raise FloatingPointError("PPO_SUPPORT_AUXILIARY_CRITIC_GRADIENT_MISSING")
            actor_parameters = list(policy.actor_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.high_actor.parameters()) + list(policy.low_actor.parameters())
            actor_gradient_norm = math.sqrt(sum(float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in actor_parameters if parameter.grad is not None))
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("PPO_SUPPORT_NONFINITE_GRADIENT_NORM")
            before = policy_module._tensor_digest(policy.state_dict())
            optimizer.step()
            after = policy_module._tensor_digest(policy.state_dict())
            if before == after:
                raise AssertionError("PPO_SUPPORT_OPTIMIZER_DID_NOT_UPDATE")
            history.append({
                "update": update_offset + len(history) + 1, "epoch": epoch + 1,
                "minibatch": batch_index, "sample_indices": indices,
                "sample_count": len(indices), "actor_loss": float(actor_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "auxiliary_value_abs_mean": float(batch_aux.detach().abs().mean().item()),
                "entropy": float(entropy.detach().item()), "total_loss": float(loss.detach().item()),
                "actor_gradient_norm_preclip": actor_gradient_norm,
                "auxiliary_critic_gradient_norm_preclip": auxiliary_gradient_norm,
                "gradient_norm_preclip": float(gradient_norm.detach().item()),
                "parameter_digest_before": before, "parameter_digest_after": after,
                "optimizer_step_applied": True,
            })
    return history


def _save_cuda_checkpoint(
    path: Path, policy: Any, optimizer: torch.optim.Optimizer, *, variant: str,
    seed: int, config: Mapping[str, Any], source_hash: str, data_hash: str,
    wm_hash: str, actor_init_hash: str, update_log: Sequence[Mapping[str, Any]],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    payload = {
        "schema": "n02-v3-paired-g-ppo-checkpoint/1", "variant": variant,
        "seed": seed, "hidden_dim": policy.hidden_dim, "use_wm_aux": policy.use_wm_aux,
        "model_state": state, "optimizer_state": optimizer_state,
        "optimizer_state_sha256": _torch_payload_hash(optimizer_state),
        "training_config": dict(config), "legacy_event_policy_sha256": source_hash,
        "data_identity_sha256": data_hash, "world_model_ensemble_hash": wm_hash,
        "actor_initialization_sha256": actor_init_hash,
        "optimizer_updates": len(update_log),
        "model_state_sha256": policy_module._tensor_digest(state),
        "update_log_sha256": _json_hash(list(update_log)),
        "restore_probe": dict(probe),
        "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "model_state_sha256": payload["model_state_sha256"], "optimizer_updates": len(update_log)}


def _restore_cuda_probe(request_file: str | Path) -> dict[str, Any]:
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("PPO_SUPPORT_RESTORE_CUDA_UNAVAILABLE")
    policy_module, _ = _load_source(request["legacy_source"])
    checkpoint_path = Path(request["checkpoint_path"]).resolve(strict=True)
    if _sha256(checkpoint_path) != request["checkpoint_sha256"]:
        raise ValueError("PPO_SUPPORT_CHECKPOINT_FILE_HASH")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "n02-v3-paired-g-ppo-checkpoint/1":
        raise ValueError("PPO_SUPPORT_CHECKPOINT_SCHEMA")
    policy = policy_module.build_hierarchical_policy(int(payload["seed"]), bool(payload["use_wm_aux"]), hidden_dim=int(payload["hidden_dim"]))
    policy.load_state_dict(payload["model_state"], strict=True)
    if policy_module._tensor_digest(policy.state_dict()) != payload.get("model_state_sha256"):
        raise ValueError("PPO_SUPPORT_RESTORE_MODEL_HASH")
    policy.to(torch.device("cuda:0")).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(payload["training_config"]["learning_rate"]))
    if _torch_payload_hash(payload["optimizer_state"]) != payload.get("optimizer_state_sha256"):
        raise ValueError("PPO_SUPPORT_RESTORE_OPTIMIZER_HASH")
    optimizer.load_state_dict(payload["optimizer_state"])
    _optimizer_state_to_device(optimizer, torch.device("cuda:0"))
    if payload["optimizer_updates"] < 1:
        raise ValueError("PPO_SUPPORT_CHECKPOINT_OPTIMIZER_EMPTY")
    probe = payload["restore_probe"]
    predictions = probe.get("wm_predictions")
    with torch.no_grad():
        observed = policy.act(probe["observation"], probe["candidates"], probe["high_level_mask"], None, predictions, deterministic=True)
    expected = probe["expected"]
    if observed["scope"] != expected["scope"] or observed["candidate_key"] != expected["candidate_key"]:
        raise ValueError("PPO_SUPPORT_RESTORE_ACTION_MISMATCH")
    for key in ("high_logits", "low_logits"):
        if observed[key] != expected[key]:
            raise ValueError(f"PPO_SUPPORT_RESTORE_LOGITS_MISMATCH:{key}")
    devices = sorted({str(value.device) for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)})
    if not any(device.startswith("cuda:0") for device in devices):
        raise ValueError("PPO_SUPPORT_RESTORED_OPTIMIZER_NOT_CUDA")
    return {"schema": "n02-v3-paired-g-ppo-restore-probe/1", "checkpoint_valid": True,
            "optimizer_restored": True, "optimizer_tensor_devices": devices,
            "inference_device": "cuda:0", "prediction_match": True,
            "model_state_sha256": payload["model_state_sha256"]}


def _fresh_process_restore(checkpoint: Mapping[str, Any], legacy_source: str | Path) -> dict[str, Any]:
    request_path = Path(checkpoint["path"]).parent / f"restore_request_{os.getpid()}.json"
    _write_json(request_path, {"checkpoint_path": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"], "legacy_source": str(Path(legacy_source).resolve())})
    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--restore-cuda", str(request_path)],
            cwd=str(SUPPORT_DIR), capture_output=True, text=True, timeout=180, check=False,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
    finally:
        request_path.unlink(missing_ok=True)
    if completed.returncode:
        raise RuntimeError(f"PPO_SUPPORT_FRESH_PROCESS_RESTORE_FAILED:{completed.stderr[-1500:]}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result.get("checkpoint_valid") is not True or result.get("optimizer_restored") is not True or result.get("prediction_match") is not True:
        raise ValueError("PPO_SUPPORT_FRESH_PROCESS_RESTORE_INVALID")
    return result


def train_paired_g_gpu(
    scenario_file: str | Path, legacy_source: str | Path,
    wm_checkpoint_paths: Sequence[str | Path], output_dir: str | Path,
    *, config_path: str | Path = DEFAULT_CONFIG, allow_frozen_config: bool = False,
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not allow_frozen_config or config.get("freeze_status") != "frozen":
        raise RuntimeError("PPO_SUPPORT_CONFIG_NOT_FROZEN_BY_OWNER")
    if config.get("training_enabled") is not True:
        raise RuntimeError("PPO_SUPPORT_TRAINING_DISABLED")
    if not torch.cuda.is_available():
        raise RuntimeError("PPO_SUPPORT_CUDA_REQUIRED")
    payload = json.loads(Path(scenario_file).read_text(encoding="utf-8"))
    shape = validate_inputs(payload, config)
    policy_module, models_module = _load_source(legacy_source)
    source_dir = Path(legacy_source).resolve() / "package" / "civilian"
    source_hash = _sha256(source_dir / "event_policy.py")
    source_hashes = {name: _sha256(source_dir / name) for name in ("event_policy.py", "event_models.py", "event_environment.py", "event_trigger.py")}
    device = torch.device("cuda:0")
    _, event_models, _ = __import__("training_support")._load_legacy(legacy_source)
    ensemble = _load_gain_augmented_ensemble(wm_checkpoint_paths, legacy_source, device, event_models)
    wm_before = policy_module._world_model_member_state_digests(ensemble)
    if any(parameter.requires_grad for model in ensemble.models for parameter in model.parameters()):
        raise ValueError("PPO_SUPPORT_WM_NOT_FROZEN")
    train_scenarios = payload["train"]
    val_scenarios = payload["validation"]
    thresholds = payload["threshold_manifest"]
    data_identity = _json_hash({"train": train_scenarios, "validation": val_scenarios})
    config_hash = _json_hash(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies: dict[str, dict[int, Any]] = {variant: {} for variant in VARIANTS}
    optimizers: dict[str, dict[int, torch.optim.Optimizer]] = {variant: {} for variant in VARIANTS}
    actor_hashes: dict[int, str] = {}
    for seed in SEEDS:
        reference = policy_module.build_hierarchical_policy(seed, use_wm_aux=False)
        for variant in VARIANTS:
            use_wm = variant.endswith("-WM")
            policy = policy_module.build_hierarchical_policy(seed, use_wm_aux=use_wm)
            policy.load_state_dict(reference.state_dict(), strict=False)
            policy.variant = variant
            policy.to(device).train()
            policies[variant][seed] = policy
            optimizers[variant][seed] = torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))
        digests = {policy_module._actor_digest(policies[variant][seed]) for variant in VARIANTS}
        if len(digests) != 1:
            raise ValueError("PPO_SUPPORT_PAIRED_ACTOR_INITIALIZATION_MISMATCH")
        actor_hashes[seed] = next(iter(digests))

    rollout_summaries: dict[str, dict[int, list[Any]]] = {variant: {seed: [] for seed in SEEDS} for variant in VARIANTS}
    update_logs: dict[str, dict[int, list[Any]]] = {variant: {seed: [] for seed in SEEDS} for variant in VARIANTS}
    round_counts: list[dict[str, int]] = []
    epochs, rounds, batch = int(config["ppo_epochs_per_round"]), int(config["rounds"]), int(config["minibatch_size"])
    updates_per_round_cap = int(config["max_updates_total_per_seed_variant"]) // rounds
    if updates_per_round_cap % epochs:
        raise ValueError("PPO_SUPPORT_UPDATE_CAP_NOT_DIVISIBLE_BY_EPOCHS")
    resource_timer = _cuda_timer_start()
    for round_index in range(rounds):
        round_rollouts: dict[tuple[str, int], list[dict[str, Any]]] = {}
        transition_counts = []
        for seed in SEEDS:
            for variant in VARIANTS:
                policy = policies[variant][seed]
                records_for_round: list[dict[str, Any]] = []
                for episode_index, scenario in enumerate(train_scenarios):
                    episode_seed = seed + 100_003 * round_index + episode_index
                    random.seed(episode_seed)
                    torch.manual_seed(episode_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(episode_seed)
                    with torch.no_grad():
                        records, summary = policy_module._event_episode(
                            scenario, policy, variant=variant, seed=seed,
                            threshold_manifest=thresholds, ensemble=ensemble,
                            config={"collection": {"max_candidates": int(config["max_candidates"])}},
                            training=True, deterministic=False,
                        )
                    policy_module._attach_episode_gae(records, gamma=float(config["gamma"]), gae_lambda=float(config["gae_lambda"]))
                    for record in records:
                        record["round"] = round_index + 1
                        record["episode_index"] = episode_index
                    records_for_round.extend(records)
                    rollout_summaries[variant][seed].append(summary)
                round_rollouts[(variant, seed)] = records_for_round
                transition_counts.append(len(records_for_round))
        common_transitions = min(transition_counts)
        updates_per_epoch = min(updates_per_round_cap // epochs, common_transitions // (batch * epochs))
        if updates_per_epoch < 1:
            raise ValueError("PPO_SUPPORT_INSUFFICIENT_TRANSITIONS_FOR_PAIRED_UPDATES")
        round_counts.append({"round": round_index + 1, "transitions_by_arm_seed_min": common_transitions,
                             "updates_per_epoch_shared": updates_per_epoch,
                             "optimizer_updates_per_arm_seed": updates_per_epoch * epochs})
        for seed in SEEDS:
            for variant in VARIANTS:
                policy = policies[variant][seed]
                log = _update_policy_cuda(
                    policy, optimizers[variant][seed], round_rollouts[(variant, seed)],
                    seed=seed + 100_003 * round_index, epochs=epochs, minibatch_size=batch,
                    updates_per_epoch=updates_per_epoch, learning_rate=float(config["learning_rate"]),
                    clip_ratio=float(config["clip_ratio"]), entropy_coefficient=float(config["entropy_coefficient"]),
                    update_offset=len(update_logs[variant][seed]),
                )
                update_logs[variant][seed].extend(log)
    resource_usage = _cuda_timer_stop(resource_timer)
    result_rows: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    for seed in SEEDS:
        for variant in VARIANTS:
            policy = policies[variant][seed]
            policy.eval()
            validation_summaries = []
            first_probe: dict[str, Any] | None = None
            for scenario in val_scenarios:
                with torch.no_grad():
                    records, summary = policy_module._event_episode(
                        scenario, policy, variant=variant, seed=seed,
                        threshold_manifest=thresholds, ensemble=ensemble,
                        config={"collection": {"max_candidates": int(config["max_candidates"])}},
                        training=False, deterministic=bool(config["validation_deterministic"]),
                    )
                validation_summaries.append(summary)
                if first_probe is None:
                    row = records[0]
                    first_probe = {
                        "observation": row["observation"], "candidates": row["candidates"],
                        "high_level_mask": row["high_level_mask"],
                        "wm_predictions": row["wm_predictions"],
                    }
            assert first_probe is not None
            expected = policy.act(first_probe["observation"], first_probe["candidates"], first_probe["high_level_mask"], None, first_probe["wm_predictions"], deterministic=True)
            first_probe["expected"] = {key: expected[key] for key in ("scope", "candidate_key", "high_logits", "low_logits")}
            checkpoint = _save_cuda_checkpoint(
                output / variant / f"seed_{seed}" / "checkpoint.pt", policy, optimizers[variant][seed],
                variant=variant, seed=seed, config=config, source_hash=source_hash,
                data_hash=data_identity, wm_hash=ensemble.ensemble_hash,
                actor_init_hash=actor_hashes[seed], update_log=update_logs[variant][seed], probe=first_probe,
            )
            restore = _fresh_process_restore(checkpoint, legacy_source)
            result_rows[variant][str(seed)] = {
                "checkpoint": {**checkpoint, "fresh_process_restore": restore},
                "actor_initialization_sha256": actor_hashes[seed],
                "optimizer_updates": len(update_logs[variant][seed]),
                "update_log": update_logs[variant][seed],
                "training_episode_summaries": rollout_summaries[variant][seed],
                "validation_episode_summaries": validation_summaries,
            }
    wm_after = policy_module._world_model_member_state_digests(ensemble)
    if wm_before != wm_after:
        raise ValueError("PPO_SUPPORT_FROZEN_WM_DIGEST_CHANGED")
    result = {
        "schema": "n02-v3-paired-g-gwm-ppo-result/1", "status": "complete",
        "training_device": "cuda:0", "config": config, "config_sha256": config_hash,
        "source_hashes": source_hashes, "data_identity_sha256": data_identity,
        "upstream_update_wrapper_sha256": hashlib.sha256(inspect.getsource(policy_module._update_policy).encode("utf-8")).hexdigest(),
        "cuda_safe_update_adapter": "_update_policy_cuda; source algorithm reused with device-correct batch tensors",
        "resource_usage": resource_usage,
        "input_shape": shape, "world_model_ensemble_hash": ensemble.ensemble_hash,
        "world_model_member_state_digests_before": wm_before,
        "world_model_member_state_digests_after": wm_after,
        "actor_initialization_sha256_by_seed": {str(seed): actor_hashes[seed] for seed in SEEDS},
        "shared_episode_order": True, "round_counts": round_counts,
        "variants": result_rows,
    }
    _write_json(output / "PAIRED_G_PPO.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--restore-cuda":
        print(json.dumps(_restore_cuda_probe(argv[1]), sort_keys=True, allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description="External paired V3 G/G-WM CUDA PPO adapter")
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--wm-checkpoints", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--enable-training", action="store_true")
    args = parser.parse_args(argv)
    result = train_paired_g_gpu(args.scenario_file, args.source, args.wm_checkpoints, args.output_dir,
                                config_path=args.config, allow_frozen_config=args.enable_training)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
