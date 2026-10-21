"""Scalar PPO for minute-stepped, autoregressive relief allocations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from dataset_protocol import validate_registry
from package_layout import require_package_root, resolve_package_layout
from relief_env import ReliefEnv


SEEDS = (8201, 8202, 8203)
EXPECTED_VARIANTS = ("L0", "L1")
EXPECTED_EPISODES_PER_ROUND = 4
EXPECTED_HORIZON = 60


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 60
    episodes_per_round: int = 4
    expected_horizon: int = 60

    def validate(self) -> None:
        if asdict(self) != asdict(PPOConfig()):
            raise ValueError("PPO hyperparameters are frozen by the execution contract")


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    rollout_cut: Sequence[bool],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = {len(rewards), len(values), len(next_values), len(terminated), len(truncated), len(rollout_cut)}
    if len(lengths) != 1:
        raise ValueError("GAE input lengths must match")
    count = len(rewards)
    advantages = np.zeros(count, dtype=np.float64)
    accumulator = 0.0
    for index in range(count - 1, -1, -1):
        is_terminated = bool(terminated[index])
        is_truncated = bool(truncated[index])
        was_cut = bool(rollout_cut[index])
        bootstrap = 0.0 if is_terminated else float(next_values[index])
        delta = float(rewards[index]) + gamma * bootstrap - float(values[index])
        carry = 0.0 if is_terminated or is_truncated or was_cut else 1.0
        accumulator = delta + gamma * gae_lambda * carry * accumulator
        advantages[index] = accumulator
    values_array = np.asarray(values, dtype=np.float64)
    returns = advantages + values_array
    return advantages.astype(np.float32), returns.astype(np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registered_scenarios(
    package_root: str | Path,
    split: str,
    *,
    allow_heldout: bool = False,
) -> tuple[list[dict], dict]:
    if split == "heldout" and not allow_heldout:
        raise PermissionError("heldout scenario loading is closed until the six-checkpoint barrier passes")
    layout = resolve_package_layout(package_root)
    root = layout.root
    if split not in {"train", "development", "heldout"}:
        raise ValueError(f"unknown registered split: {split}")
    validate_registry(layout.registered_data, verify_scenarios=True, verify_splits={split})
    registry_path = layout.registry
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    scenarios: list[dict] = []
    entries = [entry for entry in registry["parents"] if entry["split"] == split]
    for entry in entries:
        path = layout.registered_data / entry["path"]
        if sha256_file(path) != entry["scenario_sha256"]:
            raise ValueError(f"registered input changed: {entry['parent_id']}")
        scenario = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(scenario)
    if len(scenarios) != int(registry["split_sizes"][split]):
        raise ValueError(f"registered {split} count mismatch")
    return scenarios, registry


def source_identity(package_root: str | Path) -> dict[str, str]:
    layout = resolve_package_layout(package_root)
    root = layout.root
    validate_registry(layout.registered_data, verify_scenarios=True)
    source_files = {
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    }
    static_data_files = {
        layout.runtime_spec,
        layout.prior_resource_cost,
        *layout.registered_data.rglob("*.json"),
        *(root / "fixtures").glob("*.json"),
    }
    for relative in (
        "protocol/preregistered_protocol.json",
        "REUSE_IMPLEMENTATION.json",
        "SOURCE_REUSE_TRAINING.json",
    ):
        path = root / relative
        if path.is_file():
            static_data_files.add(path)
    bound_files = sorted(
        (path for path in source_files | static_data_files if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not bound_files:
        raise FileNotFoundError("canonical package source manifest is empty")
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in bound_files
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_state_digest(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for group_index, group in enumerate(state_dict["param_groups"]):
        digest.update(f"group:{group_index}".encode("ascii"))
        digest.update(_canonical_json({"params": list(group["params"])}))
        for key, value in sorted(group.items()):
            if key == "params":
                continue
            digest.update(_canonical_json({key: value}))
    for parameter_id, values in sorted(state_dict["state"].items()):
        digest.update(f"parameter:{parameter_id}".encode("ascii"))
        for name, value in sorted(values.items()):
            digest.update(name.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(_canonical_json(value))
    return digest.hexdigest()


def _optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    return _optimizer_state_digest(optimizer.state_dict())


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class PPOTrainer:
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        variant: str,
        seed: int,
        scenarios: Sequence[dict],
        data_hash: str,
        package_root: str | Path,
        run_dir: str | Path,
        config: PPOConfig | None = None,
        resource_check: Callable[[], None] | None = None,
        allow_engineering_scenarios: bool = False,
    ) -> None:
        self.config = config or PPOConfig()
        self.config.validate()
        if variant not in EXPECTED_VARIANTS:
            raise ValueError("variant must be L0 or L1")
        if len(scenarios) != 8 and not (allow_engineering_scenarios and len(scenarios) >= 4):
            raise ValueError("training split must contain exactly 8 registered scenarios")
        if any(int(scenario["horizon"]) != self.config.expected_horizon for scenario in scenarios):
            raise ValueError("all training episodes must use the frozen 60-minute horizon")
        self.policy = policy
        self.variant = variant
        self.seed = int(seed)
        self.scenarios = tuple(scenarios)
        self.data_hash = str(data_hash)
        self.package_root = require_package_root(package_root)
        self.run_dir = Path(run_dir)
        self.allow_engineering_scenarios = bool(allow_engineering_scenarios)
        self.resource_check = resource_check or (lambda: None)
        self.source_hashes = source_identity(self.package_root)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._minibatch_generator = torch.Generator(device="cpu").manual_seed(self.seed + 109)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.round_number = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _collect_episode(self, scenario: dict, round_number: int, episode_index: int) -> tuple[list[dict], dict]:
        env = ReliefEnv(scenario)
        env.reset(policy=self.variant)
        self.policy.eval()
        policy_input = env.policy_input()
        records: list[dict] = []
        decision_wall = 0.0
        decision_cpu = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            self.resource_check()
            current_input = copy.deepcopy(policy_input)
            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            with torch.no_grad():
                sample = self.policy.sample_batch(current_input, deterministic=False)
            decision_wall += time.perf_counter() - started_wall
            decision_cpu += time.process_time() - started_cpu
            actions = [str(item) for item in sample["actions"]]
            masks = _jsonable(sample["masks"])
            next_observation, reward, terminated, truncated, info = env.step(actions)
            next_input = env.policy_input()
            records.append({
                "scenario_id": scenario["scenario_id"],
                "round": round_number,
                "episode_index": episode_index,
                "time": int(current_input["observation"]["time"]),
                "policy_input": current_input,
                "actions": actions,
                "conditional_masks": masks,
                "conditional_log_probs": [float(value.detach().cpu().item()) for value in sample["conditional_log_probs"]],
                "old_log_prob": float(sample["log_prob"].detach().cpu().item()),
                "old_value": float(sample["value"].detach().cpu().item()),
                "entropy": float(sample["entropy"].detach().cpu().item()),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "rollout_cut": False,
                "next_policy_input": next_input,
                "transition_identity": info["transition"]["identity"],
            })
            policy_input = next_input
        trajectory = env.export_trajectory()
        if len(records) != self.config.expected_horizon:
            raise ValueError(f"scenario {scenario['scenario_id']} yielded {len(records)} ticks")
        self._attach_gae(records)
        episode_summary = {
            "scenario_id": scenario["scenario_id"],
            "episode_index": episode_index,
            "ticks": len(records),
            "total_reward": sum(record["reward"] for record in records),
            "decision_wall_seconds": decision_wall,
            "decision_cpu_seconds": decision_cpu,
            "environment_metrics": trajectory["summary"],
        }
        return records, episode_summary

    def _attach_gae(self, records: list[dict]) -> None:
        values = [record["old_value"] for record in records]
        rewards = [record["reward"] for record in records]
        next_values: list[float] = [0.0] * len(records)
        terminated = [record["terminated"] for record in records]
        truncated = [record["truncated"] for record in records]
        rollout_cut = [record["rollout_cut"] for record in records]
        for index, record in enumerate(records):
            if record["terminated"]:
                next_values[index] = 0.0
            elif index + 1 < len(records):
                next_values[index] = values[index + 1]
            else:
                with torch.no_grad():
                    next_values[index] = float(self.policy.value(record["next_policy_input"]).detach().cpu().item())
                record["rollout_cut"] = True
                rollout_cut[index] = True
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            rollout_cut,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        for index, record in enumerate(records):
            record["next_value"] = next_values[index]
            record["advantage"] = float(advantages[index])
            record["return"] = float(returns[index])

    def _collect_round(self, round_number: int) -> tuple[list[dict], list[dict]]:
        records: list[dict] = []
        episodes: list[dict] = []
        first = (round_number - 1) * self.config.episodes_per_round
        for episode_index in range(self.config.episodes_per_round):
            scenario_index = (first + episode_index) % len(self.scenarios)
            scenario = self.scenarios[scenario_index]
            episode_records, episode_summary = self._collect_episode(scenario, round_number, episode_index)
            records.extend(episode_records)
            episodes.append(episode_summary)
        expected = self.config.episodes_per_round * self.config.expected_horizon
        if len(records) != expected:
            raise ValueError(f"rollout must have {expected} physical minute transitions")
        return records, episodes

    def _update(self, records: list[dict]) -> dict[str, float]:
        self.policy.train()
        sample_count = len(records)
        advantages = torch.as_tensor([record["advantage"] for record in records], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        returns = torch.as_tensor([record["return"] for record in records], dtype=torch.float32)
        old_log_probs = torch.as_tensor([record["old_log_prob"] for record in records], dtype=torch.float32)
        actor_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        grad_norms: list[float] = []
        initial_ratio_error: float | None = None
        optimizer_updates = 0
        for epoch in range(self.config.epochs):
            permutation = torch.randperm(sample_count, generator=self._minibatch_generator)
            for start in range(0, sample_count, self.config.minibatch_size):
                self.resource_check()
                indices = permutation[start:start + self.config.minibatch_size].tolist()
                outputs = [
                    self.policy.evaluate_batch(
                        records[index]["policy_input"],
                        records[index]["actions"],
                        records[index]["conditional_masks"],
                    )
                    for index in indices
                ]
                new_log_probs = torch.stack([output["log_prob"] for output in outputs])
                predicted_values = torch.stack([output["value"] for output in outputs])
                entropy = torch.stack([output["entropy"] for output in outputs]).mean()
                batch_indices = torch.as_tensor(indices, dtype=torch.long)
                batch_advantages = advantages[batch_indices].to(new_log_probs.device)
                batch_returns = returns[batch_indices].to(predicted_values.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(new_log_probs.device)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                if optimizer_updates == 0:
                    initial_ratio_error = float(torch.max(torch.abs(ratios.detach() - 1.0)).cpu().item())
                    if initial_ratio_error > 2e-5:
                        raise AssertionError(f"pre-update PPO probability ratio differs from one by {initial_ratio_error}")
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(predicted_values, batch_returns)
                loss = (
                    actor_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                optimizer_updates += 1
                actor_losses.append(float(actor_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                grad_norms.append(float(grad_norm.detach().cpu().item()))
        expected_updates = self.config.epochs * (sample_count // self.config.minibatch_size)
        if optimizer_updates != expected_updates or optimizer_updates != 16:
            raise AssertionError(f"optimizer update count {optimizer_updates} != frozen 16")
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "gradient_norm_preclip_mean": float(np.mean(grad_norms)),
            "gradient_norm_preclip_max": float(np.max(grad_norms)),
            "max_initial_probability_ratio_error": float(initial_ratio_error or 0.0),
            "optimizer_updates": optimizer_updates,
            "transition_count": sample_count,
        }

    def save_checkpoint(self, path: str | Path, *, training_complete: bool = False, target_rounds: int | None = None) -> dict:
        destination = Path(path)
        payload = {
            "schema": "relief-ppo-checkpoint/1",
            "variant": self.variant,
            "seed": self.seed,
            "round_number": self.round_number,
            "training_complete": bool(training_complete),
            "target_rounds": target_rounds,
            "config": asdict(self.config),
            "data_hash": self.data_hash,
            "source_hashes": self.source_hashes,
            "architecture_evidence": copy.deepcopy(getattr(self.policy, "architecture_evidence", None)),
            "model_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "minibatch_generator_state": self._minibatch_generator.get_state(),
            "model_state_sha256": _state_digest(self.policy.state_dict()),
            "optimizer_state_sha256": _optimizer_digest(self.optimizer),
        }
        _atomic_torch_save(payload, destination)
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "round_number": self.round_number,
            "model_state_sha256": payload["model_state_sha256"],
            "optimizer_state_sha256": payload["optimizer_state_sha256"],
            "training_complete": bool(training_complete),
        }

    def restore_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        expected = {
            "schema": "relief-ppo-checkpoint/1",
            "variant": self.variant,
            "seed": self.seed,
            "config": asdict(self.config),
            "data_hash": self.data_hash,
            "source_hashes": self.source_hashes,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"checkpoint identity mismatch in {name}")
        self.policy.load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.round_number = int(payload["round_number"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        self._minibatch_generator.set_state(payload["minibatch_generator_state"])
        if _state_digest(self.policy.state_dict()) != payload["model_state_sha256"]:
            raise ValueError("restored model state checksum mismatch")
        if _optimizer_digest(self.optimizer) != payload["optimizer_state_sha256"]:
            raise ValueError("restored optimizer state checksum mismatch")
        return payload

    def train_round(self) -> dict:
        round_number = self.round_number + 1
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        records, episodes = self._collect_round(round_number)
        update_metrics = self._update(records)
        self.round_number = round_number
        rollout_path = self.run_dir / "rollouts" / f"round_{round_number:04d}.jsonl"
        _write_jsonl(rollout_path, records)
        row = {
            "schema": "relief-ppo-learning-curve/1",
            "variant": self.variant,
            "seed": self.seed,
            "round": round_number,
            "scenario_ids": [episode["scenario_id"] for episode in episodes],
            "episodes": episodes,
            "mean_episode_reward": float(np.mean([episode["total_reward"] for episode in episodes])),
            "mean_weighted_on_time_delivery_rate": float(np.mean([
                episode["environment_metrics"]["weighted_on_time_delivery_rate"] or 0.0
                for episode in episodes
            ])),
            **update_metrics,
            "rollout_path": str(rollout_path),
            "rollout_sha256": sha256_file(rollout_path),
            "round_wall_seconds_excluding_checkpoint": time.perf_counter() - started_wall,
            "round_cpu_seconds_excluding_checkpoint": time.process_time() - started_cpu,
        }
        checkpoint = self.save_checkpoint(
            self.run_dir / "checkpoints" / f"round_{round_number:04d}.pt",
            training_complete=False,
        )
        row["checkpoint"] = checkpoint
        row["round_wall_seconds_including_checkpoint"] = time.perf_counter() - started_wall
        row["round_cpu_seconds_including_checkpoint"] = time.process_time() - started_cpu
        curve_path = self.run_dir / "learning_curve.jsonl"
        with curve_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row


def verify_next_update_equivalence(
    original: PPOTrainer,
    restore_factory: Callable[[], PPOTrainer],
    checkpoint_path: str | Path,
) -> dict:
    """Compare the next real rollout/update from saved state with uninterrupted training."""

    expected = original.train_round()
    expected_model_hash = _state_digest(original.policy.state_dict())
    expected_optimizer_hash = _optimizer_digest(original.optimizer)
    restore_started_wall = time.perf_counter()
    restore_started_cpu = time.process_time()
    resumed = restore_factory()
    resumed.restore_checkpoint(checkpoint_path)
    restore_setup_wall = time.perf_counter() - restore_started_wall
    restore_setup_cpu = time.process_time() - restore_started_cpu
    actual = resumed.train_round()
    actual_model_hash = _state_digest(resumed.policy.state_dict())
    actual_optimizer_hash = _optimizer_digest(resumed.optimizer)
    def behavior_row(row: dict) -> dict:
        normalized = copy.deepcopy(row)
        for field in (
            "rollout_path",
            "round_wall_seconds_excluding_checkpoint",
            "round_cpu_seconds_excluding_checkpoint",
            "round_wall_seconds_including_checkpoint",
            "round_cpu_seconds_including_checkpoint",
            "checkpoint",
        ):
            normalized.pop(field, None)
        for episode in normalized["episodes"]:
            episode.pop("decision_wall_seconds", None)
            episode.pop("decision_cpu_seconds", None)
        return normalized

    expected_behavior = behavior_row(expected)
    actual_behavior = behavior_row(actual)
    if expected_behavior != actual_behavior:
        raise AssertionError("next-round record differs after strict checkpoint restore")
    if expected_model_hash != actual_model_hash:
        raise AssertionError("next-round model parameters differ after strict checkpoint restore")
    if expected_optimizer_hash != actual_optimizer_hash:
        raise AssertionError("next-round optimizer state differs after strict checkpoint restore")
    return {
        "equivalent": True,
        "compared_round": expected["round"],
        "expected_round_wall_seconds_including_checkpoint": expected["round_wall_seconds_including_checkpoint"],
        "expected_round_cpu_seconds_including_checkpoint": expected["round_cpu_seconds_including_checkpoint"],
        "restored_round_wall_seconds_including_checkpoint": actual["round_wall_seconds_including_checkpoint"],
        "restored_round_cpu_seconds_including_checkpoint": actual["round_cpu_seconds_including_checkpoint"],
        "restore_setup_wall_seconds": restore_setup_wall,
        "restore_setup_cpu_seconds": restore_setup_cpu,
        "model_state_sha256": expected_model_hash,
        "optimizer_state_sha256": expected_optimizer_hash,
        "round_record_sha256": hashlib.sha256(_canonical_json(expected_behavior)).hexdigest(),
    }
