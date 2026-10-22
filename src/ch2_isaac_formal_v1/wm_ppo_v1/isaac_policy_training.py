"""G PPO training on the independent closed-loop Isaac runtime.

This entry is deliberately inert until an explicit call supplies a terminal
feature batch, train-only policy normalization, and a local Isaac source root.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .candidate_features import FEATURE_NAMES as CANDIDATE_FEATURE_NAMES
from .contracts import PublicHistoryFrame
from .features import FEATURE_NAMES as NODE_FEATURE_NAMES, collate_graph_sequences


PPO_SEEDS = (8201, 8202, 8203)
PPO_GAMMA = 1.0
PPO_GAE_LAMBDA = 1.0
OBJECTIVE_KEY = "registered_planner_only_subtotal"
TRAINING_EPISODE_LIMIT = 84


@dataclass(frozen=True)
class GateReceipt:
    status: str
    reasons: tuple[str, ...]
    terminal_batch_ready: bool
    met_30_30: bool
    feature_coverage: Mapping[str, Any] | None


def check_training_gates(batch_output: str | Path) -> tuple[dict[str, Any], GateReceipt]:
    """Run terminal-168 and recomputed feature 30/30 checks before runtime setup."""
    from light_models_v1.dataset import load_feature_dataset

    dataset = load_feature_dataset(batch_output)
    gate = dataset.get("gate", {})
    coverage = dataset.get("coverage")
    reasons = []
    if gate.get("ready") is not True:
        reasons.append(str(gate.get("reason") or "TERMINAL_BATCH_GATE_NOT_READY"))
    if not isinstance(coverage, dict) or coverage.get("met_30_30") is not True:
        reasons.append("POST_EXTRACTION_FEATURE_30_30_GATE_NOT_MET")
    receipt = GateReceipt("PASS" if not reasons else "REFUSED", tuple(reasons),
                          gate.get("ready") is True,
                          isinstance(coverage, dict) and coverage.get("met_30_30") is True,
                          coverage if isinstance(coverage, dict) else None)
    return dataset, receipt


def _jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSON object at {path.name}:{line_no}")
            rows.append(value)
    return rows


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("reward evidence contains a missing or non-finite number")
    return float(value)


def _state_digest(value: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps((key, tensor), sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def save_g_checkpoint(path: str | Path, model: Any, optimizer: Any, metadata: Mapping[str, Any]) -> str:
    """Persist the G actor/critic and optimizer without reusing the WM schema."""
    import torch
    from .checkpoint import capture_rng_state, _cpu_tree, _tree_digest

    if metadata.get("method") != "G" or metadata.get("normalization", {}).get("scope") != "train_split_only":
        raise ValueError("G checkpoint requires method G and train-only normalization evidence")
    payload = {"schema": "g_ppo_training_checkpoint_v1",
        "model_state": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "model_sha256": _state_digest(model.state_dict()),
        "optimizer_sha256": _tree_digest(optimizer.state_dict()),
        "metadata": dict(metadata), "rng_state": capture_rng_state(torch)}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        torch.save(payload, stream)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def restore_g_checkpoint(path: str | Path, model: Any, optimizer: Any, *, device: str = "cpu") -> dict[str, Any]:
    """Restore a synthetic or real G checkpoint and verify model/optimizer state identity."""
    import torch
    from .checkpoint import restore_rng_state, _tree_digest

    with Path(path).open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if payload.get("schema") != "g_ppo_training_checkpoint_v1":
        raise ValueError("unsupported G PPO checkpoint schema")
    metadata = payload.get("metadata", {})
    if metadata.get("method") != "G" or metadata.get("normalization", {}).get("scope") != "train_split_only":
        raise ValueError("G PPO checkpoint metadata is incomplete")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(torch, payload.get("rng_state", {}), device)
    model_equal = _state_digest(model.state_dict()) == payload.get("model_sha256")
    optimizer_equal = _tree_digest(optimizer.state_dict()) == payload.get("optimizer_sha256")
    return {"status": "PASS" if model_equal and optimizer_equal else "FAIL",
            "model_equal": model_equal, "optimizer_equal": optimizer_equal,
            "metadata": metadata}


def reconstruct_tick_rewards(run_dir: str | Path, reduction: Mapping[str, Any]) -> tuple[list[float], float]:
    """Build search-mode planner-only rewards from evidence checked by reducer.

    Returns per-tick rewards for t=1..D and the pre-first-action prefix reward.
    The terminal -1 is included only after a complete independently valid run.
    """
    if reduction.get("status") != "PASS" or reduction.get("valid_mask") is not True:
        raise ValueError("episode reducer is not independently PASS")
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("evaluation_config", {}).get("mode") != "search":
        raise ValueError("registered reducer does not support this mission mode")
    duration = int(reduction.get("scope", {}).get("duration_s", 0))
    target_count = int(reduction.get("discovery", {}).get("target_count", 0))
    if duration < 1 or target_count < 1:
        raise ValueError("independent reducer omitted duration or target denominator")
    logs = manifest.get("logs", {})
    traj_path = root / str(logs.get("trajectory", "trajectory.jsonl.gz"))
    event_path = root / str(logs.get("events", "events.jsonl"))
    trajectory, events = _jsonl(traj_path), _jsonl(event_path)
    per_tick_energy = [0.0] * (duration + 1)
    discovered_by_tick: dict[int, set[str]] = {t: set() for t in range(1, duration + 1)}
    for row in trajectory:
        t = int(row.get("sim_time_s", -1))
        if not 1 <= t <= duration:
            raise ValueError("trajectory tick outside evaluated episode")
        energy = row.get("energy")
        per_tick_energy[t] += _finite(energy.get("incremental_wh") if isinstance(energy, dict) else None)
        ids = row.get("target_discovered_ids", [])
        if not isinstance(ids, list):
            raise ValueError("trajectory discovery evidence is malformed")
        discovered_by_tick[t].update(map(str, ids))
    per_tick_events: dict[int, list[dict[str, Any]]] = {t: [] for t in range(1, duration + 1)}
    for event in events:
        if event.get("event") == "online_timing_tick":
            continue
        raw_t = event.get("sim_time_s")
        if raw_t is None:
            continue
        t = int(raw_t)
        if 1 <= t <= duration:
            per_tick_events[t].append(event)

    messages = {"transmission_sent", "command_dropped"}
    violation = {"safety_violation", "boundary_violation", "constraint_violation"}
    rewards = [0.0] * (duration + 1)
    discovered: set[str] = set()
    changed_actors_seen: set[str] = set()
    for t in range(1, duration + 1):
        tick_events = per_tick_events[t]
        newly_discovered = discovered_by_tick[t] - discovered
        discovered.update(newly_discovered)
        unresolved = 1.0 - len(discovered) / target_count
        event_names = [str(event.get("event")) for event in tick_events]
        reward = (len(newly_discovered) / target_count
                  - 0.2 * unresolved / duration
                  - 0.01 * per_tick_energy[t] / 1_000_000.0
                  - 0.25 * sum(name in violation for name in event_names))
        reward -= 0.01 * sum(name in messages for name in event_names) / 1000.0
        reward -= 0.05 * event_names.count("controller_switch") / 100.0
        planner_calls = [event for event in tick_events if event.get("event") == "candidate_generation"]
        reward -= 0.05 * len(planner_calls) / 100.0
        for event in planner_calls:
            timing = event.get("timing")
            reward -= 0.01 * _finite(timing.get("cpu_duration_ns") if isinstance(timing, dict) else None) / 1e9
        for event in tick_events:
            if event.get("event") != "donor_support_transfer":
                continue
            for key in ("from_aircraft_id", "to_aircraft_id"):
                actor = event.get(key)
                if actor is not None and str(actor) not in changed_actors_seen:
                    changed_actors_seen.add(str(actor))
                    reward -= 0.01 / 36.0
        rewards[t] = reward
    discovered_count = int(reduction.get("discovery", {}).get("discovered_count", -1))
    if len(discovered) != discovered_count:
        raise ValueError("tick discoveries disagree with independent reducer identity")
    rewards[duration] -= 1.0
    decisions = sorted({int(event["decision_created_sim_time_s"]) for event in events
                        if event.get("event") == "decision_created"
                        and isinstance(event.get("decision_created_sim_time_s"), (int, float))})
    if any(not 1 <= t <= duration for t in decisions):
        raise ValueError("decision time lies outside the episode")
    prefix_end = decisions[0] if decisions else duration + 1
    prefix = sum(rewards[1:prefix_end])
    objective = reduction.get("registered_objective", {}).get("absolute_online_episode_cost", {})
    value = objective.get(OBJECTIVE_KEY, {}).get("value") if isinstance(objective, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("registered planner-only subtotal is unknown")
    if not math.isclose(sum(rewards[1:]), -float(value), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("tick reward sum disagrees with independent planner-only subtotal")
    return rewards[1:], prefix


def validate_rollout_records(run_dir: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Bind captured public transitions to selected/created runtime event rows."""
    event_path = Path(run_dir) / "events.jsonl"
    events = _jsonl(event_path)
    selected = {}
    created = {}
    for event in events:
        decision_id = event.get("decision_id")
        if not isinstance(decision_id, str):
            continue
        if event.get("event") == "candidate_policy_selection":
            if event.get("status") == "selected" and event.get("source") == "candidate_policy":
                if decision_id in selected:
                    raise ValueError("duplicate selected-policy event for decision")
                selected[decision_id] = event
        elif event.get("event") == "decision_created":
            if decision_id in created:
                raise ValueError("duplicate created-action event for decision")
            created[decision_id] = event
    seen = set()
    for record in records:
        decision_id = record.get("decision_id")
        if not isinstance(decision_id, str) or decision_id in seen:
            raise ValueError("rollout record decision identity is missing or duplicated")
        seen.add(decision_id)
        selection, action = selected.get(decision_id), created.get(decision_id)
        if (selection is None or action is None or action.get("selected_action") != record.get("action")
                or int(action.get("decision_created_sim_time_s", -1)) != int(record.get("decision_time_s", -2))):
            raise ValueError("captured action differs from runtime public event evidence")
    if seen != set(selected) or seen != set(created):
        raise ValueError("captured transitions differ from successful selected-policy event coverage")


def attribute_decision_intervals(
    tick_rewards: Sequence[float], decision_times: Sequence[int], *, duration_s: int,
) -> dict[str, Any]:
    """Sum tick rewards into [t_i,t_(i+1)) intervals; final interval ends D+1."""
    times = sorted(int(t) for t in decision_times)
    if duration_s < 1 or any(t < 1 or t > duration_s for t in times) or len(set(times)) != len(times):
        raise ValueError("decision times must be unique and within the episode")
    if len(tick_rewards) != duration_s:
        raise ValueError("tick rewards must contain exactly D values")
    prefix_end = times[0] if times else duration_s + 1
    prefix = sum(float(value) for value in tick_rewards[:prefix_end - 1])
    intervals = []
    for index, start in enumerate(times):
        stop = times[index + 1] if index + 1 < len(times) else duration_s + 1
        intervals.append({"start_s": start, "stop_exclusive_s": stop,
                          "reward": sum(float(value) for value in tick_rewards[start - 1:stop - 1]),
                          "elapsed_s": stop - start})
    total = prefix + sum(item["reward"] for item in intervals)
    expected = sum(float(value) for value in tick_rewards)
    if not math.isclose(total, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("prefix and decision intervals do not cover all episode ticks")
    return {"prefix_reward": prefix, "intervals": intervals, "total_reward": total}


def _torch_graph_batch(histories: Sequence[Sequence[PublicHistoryFrame]], normalization: Mapping[str, Any], device: Any):
    import torch

    batch = collate_graph_sequences(histories)
    values = {}
    for name in ("node_features", "node_mask", "sequence_mask", "edge_index", "edge_mask",
                 "edge_type", "event_features", "action_features"):
        value = getattr(batch, name)
        if value is not None:
            dtype = torch.bool if name.endswith("mask") else torch.long if name in {"edge_index", "edge_type"} else torch.float32
            values[name] = torch.as_tensor(value, dtype=dtype, device=device)
    center = torch.as_tensor(normalization["node_center"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(normalization["node_scale"], dtype=torch.float32, device=device)
    values["node_features"] = (values["node_features"] - center) / scale
    return type(batch)(**values)


def _update_from_episode(model: Any, core: Any, records: Sequence[Mapping[str, Any]],
                         interval_data: Mapping[str, Any], normalization: Mapping[str, Any],
                         device: Any, *, epochs: int = 4) -> dict[str, Any]:
    import torch
    from .ppo_core import compute_time_aware_gae
    from .candidate_features import MAX_CANDIDATES

    transitions = interval_data["intervals"]
    if not records or len(records) != len(transitions):
        raise ValueError("successful public policy selections and reward intervals differ")
    histories = [record["history"] for record in records]
    graph_batch = _torch_graph_batch(histories, normalization, device)
    batch_size = len(records)
    candidates = torch.zeros((batch_size, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES)), dtype=torch.float32, device=device)
    legal = torch.zeros((batch_size, MAX_CANDIDATES), dtype=torch.bool, device=device)
    noop = torch.zeros_like(legal)
    actions, old_logprob, old_values = [], [], []
    candidate_center = torch.as_tensor(normalization["candidate_center"], dtype=torch.float32, device=device)
    candidate_scale = torch.as_tensor(normalization["candidate_scale"], dtype=torch.float32, device=device)
    for row, record in enumerate(records):
        feature = torch.as_tensor(record["candidate_features"], dtype=torch.float32, device=device)
        count = feature.shape[0]
        if feature.shape != (len(record["catalog"]), len(CANDIDATE_FEATURE_NAMES)) or count > MAX_CANDIDATES:
            raise ValueError("runtime candidate feature matrix violates the public schema")
        candidates[row, :count] = (feature - candidate_center) / candidate_scale
        legal[row, :count] = torch.as_tensor(record["legal_mask"], dtype=torch.bool, device=device)
        noop[row, :count] = torch.as_tensor(record["noop_mask"], dtype=torch.bool, device=device)
        actions.append(int(record["action_index"]))
        old_logprob.append(float(record["old_logprob"]))
        old_values.append(float(record["value"]))
    rewards = torch.as_tensor([item["reward"] for item in transitions], dtype=torch.float32, device=device)
    values = torch.as_tensor(old_values, dtype=torch.float32, device=device)
    elapsed = torch.as_tensor([item["elapsed_s"] for item in transitions], dtype=torch.float32, device=device)
    terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    truncated = torch.zeros_like(terminated)
    terminated[-1] = True
    next_values = torch.cat((values[1:], values.new_zeros(1)))
    advantages, returns = compute_time_aware_gae(rewards, values, next_values, elapsed,
        terminated, truncated, gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA)
    result = core.update(graph_batch, candidate_features=candidates, candidate_mask=legal, noop_mask=noop,
        actions=torch.as_tensor(actions, dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(old_logprob, dtype=torch.float32, device=device), old_values=values,
        returns=returns, advantages=advantages, epochs=epochs)
    return asdict(result)


def _check_policy_normalization(value: Mapping[str, Any], dataset: Mapping[str, Any],
                                batch_output: str | Path) -> dict[str, Any]:
    normalization = dict(value)
    if normalization.get("scope") != "train_split_only":
        raise ValueError("policy normalization must attest train_split_only")
    for key, width in (("node_center", len(NODE_FEATURE_NAMES)), ("node_scale", len(NODE_FEATURE_NAMES)),
                       ("candidate_center", len(CANDIDATE_FEATURE_NAMES)),
                       ("candidate_scale", len(CANDIDATE_FEATURE_NAMES))):
        values = np.asarray(normalization.get(key), dtype=np.float32)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError(f"normalization {key} has invalid shape or values")
        if key.endswith("scale") and (values <= 0).any():
            raise ValueError(f"normalization {key} must be positive")
        normalization[key] = values.tolist()
    if list(normalization.get("node_feature_names", ())) != list(NODE_FEATURE_NAMES):
        raise ValueError("normalization node schema mismatch")
    if list(normalization.get("candidate_feature_names", ())) != list(CANDIDATE_FEATURE_NAMES):
        raise ValueError("normalization candidate schema mismatch")
    actual = hashlib.sha256((Path(batch_output).resolve() / "aggregate.json").read_bytes()).hexdigest()
    if normalization.get("dataset_sha256") != actual:
        raise ValueError("normalization dataset hash differs from gated aggregate")
    training_scenarios = sorted({str(row["scenario_id"]) for row in dataset.get("rows", [])
                                 if row.get("split") == "train"})
    if normalization.get("training_scenario_ids") != training_scenarios:
        raise ValueError("normalization scenario evidence must exactly bind extracted train scenarios")
    return normalization


def _training_scene_order(batch_output: str | Path) -> tuple[list[dict[str, Any]], str]:
    root = Path(batch_output).resolve()
    receipt = json.loads((root / "batch_receipt.json").read_text(encoding="utf-8"))
    registry_path = Path(receipt.get("registry_file", "")).resolve()
    if not registry_path.is_file():
        raise ValueError("frozen formal registry referenced by batch receipt is unavailable")
    registry_bytes = registry_path.read_bytes()
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    if registry_sha != receipt.get("registry_sha256"):
        raise ValueError("frozen formal registry hash differs from batch receipt")
    registry = json.loads(registry_bytes.decode("utf-8"))
    if registry.get("run_count") != 168 or len(registry.get("runs", [])) != 168:
        raise ValueError("training registry must bind exactly 168 scenes")
    completed = {row.get("run_id"): row for row in receipt.get("scenes", []) if isinstance(row, dict)}
    train_rows = [row for row in registry["runs"] if row.get("split") == "train"]
    if len(train_rows) != TRAINING_EPISODE_LIMIT:
        raise ValueError("frozen training episode budget must use all 84 registry train scenes")
    selected = []
    for row in train_rows:
        scene = completed.get(row.get("run_id"))
        if (scene is None or scene.get("split") != "train"
                or scene.get("scenario_id") != row.get("scenario_id")
                or scene.get("config_sha256") != row.get("config_sha256")):
            raise ValueError("train scene receipt identity/config hash differs from frozen registry")
        config_path = (registry_path.parent / str(row.get("config_path", ""))).resolve()
        try:
            config_path.relative_to(registry_path.parent.resolve())
        except ValueError as exc:
            raise ValueError("registry config path escapes its frozen root") from exc
        if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest() != row.get("config_sha256"):
            raise ValueError("train config is missing or hash-mismatched")
        selected.append({"scenario_id": str(row["scenario_id"]), "condition": str(row["condition"]),
                         "source": str(row["source"]), "config_path": str(config_path),
                         "config_sha256": str(row["config_sha256"]), "scenario_seed": int(row["seed"])})
    return selected, registry_sha


def run_g_training(*, batch_output: str | Path,
                    source_root: str | Path, output: str | Path,
                    normalization: Mapping[str, Any], seeds: Sequence[int] = PPO_SEEDS,
                    epochs: int = 4, learning_rate: float = 3e-4) -> dict[str, Any]:
    """Run explicitly requested G PPO episodes; all heavy setup follows data gates."""
    dataset, gate = check_training_gates(batch_output)
    if gate.status != "PASS":
        return {"status": "REFUSED", "gate": asdict(gate), "g": "NOT_RUN",
                "g_wm": "UNAVAILABLE_G_EQUIVALENT"}
    episode_order, registry_sha = _training_scene_order(batch_output)
    policy_normalization = _check_policy_normalization(normalization, dataset, batch_output)
    if tuple(int(seed) for seed in seeds) != PPO_SEEDS:
        raise ValueError(f"training seed order is frozen as {PPO_SEEDS}")

    from .gain_deployment import _check_gpu1
    import torch
    device = _check_gpu1()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU1 must be the sole visible CUDA device mapped to cuda:0")

    from .ppo_core import MaskedCandidateActorCritic, PPOCore
    from .policy_choice import PublicCandidatePolicy
    from closedloop_v1.evaluate_episode import evaluate_run
    from closedloop_v1.runtime import run as run_runtime

    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_seed_reports = []
    model = core = None
    for seed in PPO_SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        model = MaskedCandidateActorCritic(candidate_dim=len(CANDIDATE_FEATURE_NAMES), hidden_dim=64)
        core = PPOCore(model, device=str(device), learning_rate=learning_rate, auxiliary_lambda=0.0)
        episode_reports = []
        for ordinal, episode in enumerate(episode_order, 1):
            episode_dir = destination / f"seed_{seed}" / f"{ordinal:03d}_{episode['scenario_id']}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            policy = PublicCandidatePolicy(model, policy_normalization,
                model_version=f"g-ppo-seed-{seed}", mode="sample", device="cuda:0")
            torch.manual_seed(seed + ordinal)
            status = run_runtime(episode["config_path"], source_root, episode_dir,
                                 catalogue_policy=policy, rollout_callback=records.append)
            if status != 0:
                episode_reports.append({"scenario_id": episode["scenario_id"], "attempted": True,
                    "completed": False, "failed": True, "status": "RUNTIME_FAILED", "updates": 0})
                continue
            reduction = evaluate_run(episode_dir)
            objective = reduction.get("registered_objective", {}).get("absolute_online_episode_cost", {})
            subtotal = objective.get(OBJECTIVE_KEY, {}).get("value") if isinstance(objective, dict) else None
            if (reduction.get("status") != "PASS" or reduction.get("valid_mask") is not True
                    or reduction.get("scope", {}).get("branch") != "representative"
                    or not isinstance(subtotal, (int, float))):
                episode_reports.append({"scenario_id": episode["scenario_id"], "attempted": True,
                                        "completed": False, "failed": True,
                                        "status": "REDUCER_NOT_PASS", "updates": 0,
                                        "reasons": reduction.get("reasons", [])})
                continue
            try:
                validate_rollout_records(episode_dir, records)
                tick_rewards, _ = reconstruct_tick_rewards(episode_dir, reduction)
                attributed = attribute_decision_intervals(tick_rewards,
                    [int(record["decision_time_s"]) for record in records], duration_s=5400)
                update = _update_from_episode(model, core, records, attributed, policy_normalization, device,
                                              epochs=epochs) if records else None
            except (KeyError, TypeError, ValueError) as exc:
                episode_reports.append({"scenario_id": episode["scenario_id"], "attempted": True,
                                        "completed": False, "failed": True,
                                        "status": "REWARD_IDENTITY_REFUSED", "updates": 0,
                                        "reason": str(exc)})
                continue
            episode_reports.append({"scenario_id": episode["scenario_id"], "attempted": True,
                "completed": True, "failed": False,
                "status": "UPDATED" if update else "NO_POLICY_TRANSITIONS",
                "updates": 1 if update else 0, "transition_count": len(records),
                "planner_only_subtotal": float(subtotal), "reward_sum": attributed["total_reward"],
                "update": update})
        trained = any(item["updates"] for item in episode_reports)
        checkpoint = None
        if trained:
            aggregate_path = Path(batch_output).resolve() / "aggregate.json"
            metadata = {"method": "G", "training_seed": seed,
                "episode_order": [item["scenario_id"] for item in episode_order],
                "episode_config_sha256": [item["config_sha256"] for item in episode_order],
                "registry_sha256": registry_sha,
                "dataset_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
                "feature_names": list(NODE_FEATURE_NAMES), "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
                "normalization": policy_normalization, "architecture": {"candidate_dim": len(CANDIDATE_FEATURE_NAMES),
                "hidden_dim": 64}, "optimizer": {"name": "Adam", "learning_rate": learning_rate},
                "gamma": PPO_GAMMA, "gae_lambda": PPO_GAE_LAMBDA, "objective": OBJECTIVE_KEY,
                "device": str(device)}
            checkpoint = save_g_checkpoint(destination / f"g_ppo_seed_{seed}.pt", model, core.optimizer, metadata)
        complete_count = sum(bool(item["completed"]) for item in episode_reports)
        failed_count = sum(bool(item["failed"]) for item in episode_reports)
        update_count = sum(int(item["updates"]) for item in episode_reports)
        seed_status = ("COMPLETE" if complete_count == TRAINING_EPISODE_LIMIT and update_count > 0 else
                       "PARTIAL" if complete_count or failed_count else "NOT_RUN")
        all_seed_reports.append({"seed": seed, "attempted": True, "status": seed_status,
                                 "attempted_episode_count": len(episode_reports),
                                 "completed_episode_count": complete_count,
                                 "failed_episode_count": failed_count, "update_count": update_count,
                                 "checkpoint_sha256": checkpoint, "episodes": episode_reports})
    complete_seeds = sum(item["status"] == "COMPLETE" for item in all_seed_reports)
    status = "COMPLETE" if complete_seeds == len(PPO_SEEDS) else "PARTIAL" if all_seed_reports else "NOT_RUN"
    return {"status": status,
            "method": "G", "g_wm": "UNAVAILABLE_G_EQUIVALENT",
            "gate": asdict(gate), "seeds": list(PPO_SEEDS),
            "episode_order": [item["scenario_id"] for item in episode_order],
            "episode_config_sha256": [item["config_sha256"] for item in episode_order],
            "training_episode_limit": TRAINING_EPISODE_LIMIT,
            "gamma": PPO_GAMMA, "gae_lambda": PPO_GAE_LAMBDA,
            "objective": OBJECTIVE_KEY, "registry_sha256": registry_sha, "seed_runs": all_seed_reports}


def main() -> int:
    parser = argparse.ArgumentParser(description="G PPO training over gated search episodes")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--normalization-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    normalization = json.loads(Path(args.normalization_json).read_text(encoding="utf-8"))
    result = run_g_training(batch_output=args.batch_output,
        source_root=args.source_root, output=args.output, normalization=normalization)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "g_ppo_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "g_wm": result["g_wm"]}, sort_keys=True))
    return 0 if result["status"] in {"COMPLETE", "NOT_RUN", "REFUSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
