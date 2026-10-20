"""Checkpoint-gated deterministic evaluation and independent paired metrics."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from package_layout import require_package_root, resolve_package_layout

import numpy as np
import torch

from baselines import choose_step_c0_actions, choose_step_c1_actions
from independent_audit import audit_run
from dataset_protocol import validate_registry
from ppo_trainer import (
    EXPECTED_VARIANTS,
    PPOConfig,
    SEEDS,
    _optimizer_state_digest,
    _state_digest,
    load_registered_scenarios,
    sha256_file,
    source_identity,
)
from relief_env import ReliefEnv


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def checkpoint_barrier(package_root: str | Path, training_root: str | Path) -> dict:
    """Verify all six frozen final checkpoints before loading heldout scenarios."""

    layout = resolve_package_layout(package_root)
    package = layout.root
    output = Path(training_root)
    freeze_path = output / "budget_freeze.json"
    manifest_path = output / "final_checkpoints.json"
    if not freeze_path.is_file() or not manifest_path.is_file():
        raise PermissionError("evaluation barrier requires budget freeze and final checkpoint manifest")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rounds = int(freeze.get("common_rounds", 0))
    if rounds < 8 or rounds > 32 or freeze.get("eligible") is not True:
        raise PermissionError("budget freeze did not authorize a complete comparison")
    if manifest.get("training_complete") is not True or int(manifest.get("common_rounds", -1)) != rounds:
        raise PermissionError("final checkpoint manifest does not bind a complete common-round run")
    expected_ids = {f"{variant}-{seed}" for seed in SEEDS for variant in EXPECTED_VARIANTS}
    entries = manifest.get("runs", [])
    if {entry.get("run_id") for entry in entries} != expected_ids or len(entries) != 6:
        raise PermissionError("all six unique final model checkpoints are required")
    current_sources = source_identity(package)
    registry_path = layout.registry
    registry_hash = sha256_file(registry_path)
    registry_validation = validate_registry(layout.registered_data, verify_scenarios=False)
    if manifest.get("input_registry_sha256") != registry_hash:
        raise PermissionError("final checkpoint manifest was trained against a different input registry")
    verified = []
    for entry in entries:
        checkpoint_path = output / entry["path"]
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != entry.get("sha256"):
            raise PermissionError(f"final checkpoint file/hash failed: {entry.get('run_id')}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_id = entry["run_id"]
        variant, seed_text = run_id.rsplit("-", 1)
        if payload.get("schema") != "relief-ppo-checkpoint/1":
            raise PermissionError(f"unsupported checkpoint schema: {run_id}")
        if payload.get("variant") != variant or int(payload.get("seed", -1)) != int(seed_text):
            raise PermissionError(f"checkpoint model identity mismatch: {run_id}")
        if payload.get("training_complete") is not True or int(payload.get("round_number", -1)) != rounds or int(payload.get("target_rounds", -1)) != rounds:
            raise PermissionError(f"checkpoint is partial or has wrong rounds: {run_id}")
        if payload.get("data_hash") != registry_hash or payload.get("source_hashes") != current_sources:
            raise PermissionError(f"checkpoint data/source identity mismatch: {run_id}")
        if payload.get("config") != asdict(PPOConfig()):
            raise PermissionError(f"checkpoint optimizer configuration mismatch: {run_id}")
        state_digest = _state_digest(payload["model_state"])
        if state_digest != payload.get("model_state_sha256"):
            raise PermissionError(f"checkpoint model state checksum mismatch: {run_id}")
        if state_digest != entry.get("model_state_sha256"):
            raise PermissionError(f"final manifest model state checksum mismatch: {run_id}")
        optimizer_digest = _optimizer_state_digest(payload["optimizer_state"])
        if optimizer_digest != payload.get("optimizer_state_sha256"):
            raise PermissionError(f"checkpoint optimizer state checksum mismatch: {run_id}")
        if optimizer_digest != entry.get("optimizer_state_sha256"):
            raise PermissionError(f"final manifest optimizer state checksum mismatch: {run_id}")
        if entry.get("variant") != variant or int(entry.get("seed", -1)) != int(seed_text):
            raise PermissionError(f"final manifest model identity mismatch: {run_id}")
        try:
            policy = _load_policy(package, checkpoint_path, variant, int(seed_text), registry_hash)
        except (RuntimeError, ValueError, KeyError) as exc:
            raise PermissionError(f"checkpoint architecture failed strict model loading: {run_id}") from exc
        architecture = policy.architecture_evidence
        if payload.get("architecture_evidence") != architecture or entry.get("architecture_evidence") != architecture:
            raise PermissionError(f"checkpoint architecture evidence mismatch: {run_id}")
        verified.append({
            "run_id": run_id,
            "path": str(entry["path"]),
            "sha256": entry["sha256"],
            "model_state_sha256": state_digest,
            "optimizer_state_sha256": optimizer_digest,
            "architecture_evidence": architecture,
            "round_number": rounds,
            "training_complete": True,
        })
    receipt = {
        "schema": "evaluation-checkpoint-barrier/1",
        "passed": True,
        "common_rounds": rounds,
        "checkpoint_count": len(verified),
        "registry_sha256": registry_hash,
        "registry_validation": registry_validation,
        "source_hashes": current_sources,
        "checkpoints": sorted(verified, key=lambda item: item["run_id"]),
    }
    return receipt


def _load_policy(package_root: Path, checkpoint_path: Path, variant: str, seed: int, registry_hash: str):
    package_root = require_package_root(package_root)
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from graph_policy import build_policy

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("data_hash") != registry_hash:
        raise ValueError("checkpoint was trained against a different scenario registry")
    policy = build_policy(variant, seed, device="cpu")
    policy.load_state_dict(payload["model_state"], strict=True)
    if payload.get("architecture_evidence") != policy.architecture_evidence:
        raise ValueError("checkpoint architecture evidence does not match the current policy")
    policy.eval()
    return policy


def _evaluate_baseline(scenario: dict, policy_name: str) -> tuple[dict, dict]:
    environment = ReliefEnv(scenario)
    observation = environment.reset(policy=policy_name)
    timings = {"wall_seconds": 0.0, "cpu_seconds": 0.0, "decisions": 0}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        candidates = environment.candidates()
        wall = time.perf_counter()
        cpu = time.process_time()
        try:
            if policy_name == "C0":
                actions = choose_step_c0_actions(observation, candidates)
            else:
                actions = choose_step_c1_actions(observation, candidates)
        finally:
            timings["wall_seconds"] += time.perf_counter() - wall
            timings["cpu_seconds"] += time.process_time() - cpu
            timings["decisions"] += 1
        observation, _, terminated, truncated, _ = environment.step([str(action["id"]) for action in actions])
    result = environment.export_trajectory()
    if len(result["transitions"]) != int(scenario["horizon"]):
        raise ValueError("baseline policy did not export one decision per minute")
    return result, timings


def _evaluate_learned(scenario: dict, policy: torch.nn.Module, policy_name: str) -> tuple[dict, dict]:
    environment = ReliefEnv(scenario)
    environment.reset(policy=policy_name)
    timings = {"wall_seconds": 0.0, "cpu_seconds": 0.0, "decisions": 0}
    policy_traces = []
    observation = environment.public_observation()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        policy_input = environment.policy_input()
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.no_grad():
            sample = policy.sample_batch(policy_input, deterministic=True)
        timings["wall_seconds"] += time.perf_counter() - wall
        timings["cpu_seconds"] += time.process_time() - cpu
        timings["decisions"] += 1
        actions = list(sample["actions"])
        policy_traces.append({
            "actions": actions,
            "conditional_masks": sample["masks"],
            "conditional_log_probs": [float(value.detach().cpu().item()) for value in sample["conditional_log_probs"]],
            "batch_log_prob": float(sample["log_prob"].detach().cpu().item()),
            "value": float(sample["value"].detach().cpu().item()),
        })
        observation, _, terminated, truncated, _ = environment.step(actions)
    result = environment.export_trajectory()
    if len(result["transitions"]) != int(scenario["horizon"]):
        raise ValueError("learned policy did not export one decision per minute")
    for transition, trace in zip(result["transitions"], policy_traces):
        transition["policy_trace"] = trace
    return result, timings


def _outside_commitment_metrics(scenario: dict, task_rows: list[dict]) -> dict[str, float | int]:
    specs = {task["id"]: task for task in scenario["tasks"]}
    outside_ids = {
        task_id
        for task_id, task in specs.items()
        if task["kind"] == "delivery" and not task["local"]
    }
    outside = [row for row in task_rows if row["id"] in outside_ids and row.get("original_assignee") is not None]
    late_minutes = [
        max(0.0, float(row["completed_at"]) - float(row["deadline"]))
        for row in outside
        if row.get("delivered") and row.get("on_time") is False and row.get("completed_at") is not None
    ]
    return {
        "outside_commitment_count": len(outside),
        "outside_commitment_late_orders": len(late_minutes),
        "outside_commitment_delay_minutes": float(sum(late_minutes)),
        "outside_commitment_mean_delay_minutes": float(np.mean(late_minutes)) if late_minutes else 0.0,
    }


def _episode_row(
    scenario: dict,
    method: str,
    seed: int | None,
    result: dict,
    timings: dict,
    stratum: str,
) -> tuple[dict, dict]:
    audited = audit_run(scenario, result, expected_policy=method)
    metrics = dict(audited["metrics"])
    metrics.update(_outside_commitment_metrics(scenario, audited["task_rows"]))
    tick_count = max(1, len(result.get("transitions", [])))
    metrics["decision_wall_seconds"] = float(timings["wall_seconds"])
    metrics["decision_cpu_seconds"] = float(timings["cpu_seconds"])
    metrics["decision_wall_ms_per_tick"] = 1000.0 * float(timings["wall_seconds"]) / tick_count
    metrics["decision_cpu_ms_per_tick"] = 1000.0 * float(timings["cpu_seconds"]) / tick_count
    row = {
        "parent_id": scenario["scenario_id"],
        "stratum": stratum,
        "method": method,
        "seed": seed,
        "metrics": metrics,
        "audit_validation_level": audited["validation_level"],
        "trajectory_event_count": len(result.get("events", [])),
        "decision_count": timings["decisions"],
    }
    return row, audited


def _mean(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row["metrics"][metric] or 0.0) for row in rows])) if rows else math.nan


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    location = (len(sorted_values) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return sorted_values[low]
    weight = location - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _paired_bootstrap(
    differences_by_seed: dict[int, dict[str, float]],
    *,
    seed: int,
    replicates: int = 5000,
) -> dict[str, Any]:
    seed_values = sorted(differences_by_seed)
    parent_ids = sorted(next(iter(differences_by_seed.values())))
    if not parent_ids or not seed_values:
        raise ValueError("paired bootstrap requires parent and seed differences")
    generator = random.Random(seed)
    draws: list[float] = []
    count = len(parent_ids)
    for _ in range(replicates):
        sampled = [parent_ids[generator.randrange(count)] for _ in range(count)]
        per_seed = [
            float(np.mean([differences_by_seed[seed_id][parent] for parent in sampled]))
            for seed_id in seed_values
        ]
        draws.append(float(np.mean(per_seed)))
    draws.sort()
    seed_means = {
        str(seed_id): float(np.mean(list(differences_by_seed[seed_id].values())))
        for seed_id in seed_values
    }
    return {
        "mean_difference": float(np.mean(list(seed_means.values()))),
        "seed_mean_differences": seed_means,
        "paired_parent_bootstrap_95_percentile_interval": [
            _percentile(draws, 0.025),
            _percentile(draws, 0.975),
        ],
        "parent_count": count,
        "training_seed_count": len(seed_values),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interval_scope": "paired parent scenarios over the observed three model seeds; excludes training-seed uncertainty",
    }


def _build_summary(rows: list[dict]) -> dict:
    metrics = (
        "weighted_on_time_delivery_rate",
        "inspection_on_time_rate",
        "unmet_orders",
        "late_orders",
        "outside_commitment_late_orders",
        "outside_commitment_delay_minutes",
        "energy_used",
        "reassignments",
        "new_assignments",
        "decision_wall_ms_per_tick",
        "decision_cpu_ms_per_tick",
    )
    c1_by_parent = {
        row["parent_id"]: float(row["metrics"]["weighted_on_time_delivery_rate"] or 0.0)
        for row in rows if row["method"] == "C1"
    }
    l0_by_seed = {
        seed: {row["parent_id"]: float(row["metrics"]["weighted_on_time_delivery_rate"] or 0.0)
               for row in rows if row["method"] == "L0" and row["seed"] == seed}
        for seed in SEEDS
    }
    l1_by_seed = {
        seed: {row["parent_id"]: float(row["metrics"]["weighted_on_time_delivery_rate"] or 0.0)
               for row in rows if row["method"] == "L1" and row["seed"] == seed}
        for seed in SEEDS
    }
    l1_c1 = {
        seed: {parent: l1_by_seed[seed][parent] - c1_by_parent[parent] for parent in c1_by_parent}
        for seed in SEEDS
    }
    l1_l0 = {
        seed: {parent: l1_by_seed[seed][parent] - l0_by_seed[seed][parent] for parent in c1_by_parent}
        for seed in SEEDS
    }
    by_method: dict[str, dict[str, float]] = {}
    for method in ("C0", "C1"):
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {metric: _mean(method_rows, metric) for metric in metrics}
    for method in ("L0", "L1"):
        seed_summaries = {}
        for seed in SEEDS:
            seed_rows = [row for row in rows if row["method"] == method and row["seed"] == seed]
            seed_summaries[str(seed)] = {metric: _mean(seed_rows, metric) for metric in metrics}
        by_method[method] = {
            metric: float(np.mean([seed_summaries[str(seed)][metric] for seed in SEEDS]))
            for metric in metrics
        }
        by_method[method]["per_seed"] = seed_summaries
    return {
        "parent_macro_metrics": by_method,
        "primary_metric": "parent-macro priority-weighted on-time delivery rate",
        "paired_primary_differences": {
            "L1_minus_C1": {
                "per_seed": {str(seed): float(np.mean(list(l1_c1[seed].values()))) for seed in SEEDS},
                **_paired_bootstrap(l1_c1, seed=20261011),
            },
            "L1_minus_L0": {
                "per_seed": {str(seed): float(np.mean(list(l1_l0[seed].values()))) for seed in SEEDS},
                **_paired_bootstrap(l1_l0, seed=20261012),
            },
        },
        "evaluation_parent_count": len(c1_by_parent),
    }


def run_evaluation(
    package_root: str | Path,
    training_root: str | Path,
    output_root: str | Path,
    *,
    resource_check=None,
) -> dict:
    layout = resolve_package_layout(package_root)
    package = layout.root
    training = Path(training_root)
    output = Path(output_root)
    gate = checkpoint_barrier(package, training)
    registry_hash = sha256_file(layout.registry)
    manifest = json.loads((training / "final_checkpoints.json").read_text(encoding="utf-8"))
    policy_cache = {}
    for seed in SEEDS:
        for method in EXPECTED_VARIANTS:
            run_id = f"{method}-{seed}"
            checkpoint = next(item for item in manifest["runs"] if item["run_id"] == run_id)
            policy_cache[run_id] = _load_policy(package, training / checkpoint["path"], method, seed, registry_hash)
    gate["strict_model_loads_before_heldout"] = sorted(policy_cache)
    _write_json(output / "checkpoint_barrier.json", gate)
    scenarios, registry = load_registered_scenarios(package, "heldout", allow_heldout=True)
    parent_strata = {
        entry["parent_id"]: entry["stratum"]
        for entry in registry["parents"] if entry["split"] == "heldout"
    }
    episodes: list[dict] = []
    rows: list[dict] = []
    for scenario in scenarios:
        if resource_check:
            resource_check()
        for method in ("C0", "C1"):
            result, timings = _evaluate_baseline(scenario, method)
            row, audited = _episode_row(
                scenario, method, None, result, timings, parent_strata[scenario["scenario_id"]]
            )
            rows.append(row)
            episodes.append(row)
            stem = f"{method}/{scenario['scenario_id']}"
            _write_json(output / "trajectories" / f"{stem}.json", result)
            _write_json(output / "audits" / f"{stem}.json", audited)
            _write_jsonl(output / "task_rows" / f"{stem}.jsonl", audited["task_rows"])
        for seed in SEEDS:
            for method in EXPECTED_VARIANTS:
                run_id = f"{method}-{seed}"
                model = policy_cache[run_id]
                result, timings = _evaluate_learned(scenario, model, method)
                row, audited = _episode_row(
                    scenario, method, seed, result, timings, parent_strata[scenario["scenario_id"]]
                )
                rows.append(row)
                episodes.append(row)
                stem = f"{method}-{seed}/{scenario['scenario_id']}"
                _write_json(output / "trajectories" / f"{stem}.json", result)
                _write_json(output / "audits" / f"{stem}.json", audited)
                _write_jsonl(output / "task_rows" / f"{stem}.jsonl", audited["task_rows"])
                if resource_check:
                    resource_check()
    expected_runs = 56 * (2 + 6)
    if len(rows) != expected_runs:
        raise ValueError(f"evaluation emitted {len(rows)} runs, expected {expected_runs}")
    summary = _build_summary(rows)
    summary["checkpoint_barrier"] = gate
    summary["evaluation_registry_sha256"] = registry_hash
    summary["evaluation_run_count"] = len(rows)
    summary["heldout_consumed_for_future_development"] = True
    _write_json(output / "episode_metrics.json", {"runs": episodes})
    _write_json(output / "summary.json", summary)
    report_lines = [
        "# 灾后救灾图策略迁移小规模对比",
        "",
        "本报告使用预先登记的 56 个保留父场景。评价开始前，六个最终模型 checkpoint、训练轮数、数据摘要和源码身份均通过屏障核验。评价动作采用确定性选择；C0/C1 与 L0/L1 使用同一父场景外生事件。",
        "",
        f"主指标为父场景宏平均优先级加权按时交付率。C0={summary['parent_macro_metrics']['C0']['weighted_on_time_delivery_rate']:.4f}，C1={summary['parent_macro_metrics']['C1']['weighted_on_time_delivery_rate']:.4f}，L0（三 seed 均值）={summary['parent_macro_metrics']['L0']['weighted_on_time_delivery_rate']:.4f}，L1（三 seed 均值）={summary['parent_macro_metrics']['L1']['weighted_on_time_delivery_rate']:.4f}。",
        "",
        "## 配对差值",
    ]
    for contrast, values in summary["paired_primary_differences"].items():
        ci = values["paired_parent_bootstrap_95_percentile_interval"]
        report_lines.append(f"- {contrast}：seed 配对宏均值 {values['mean_difference']:.4f}；父场景配对区间 [{ci[0]:.4f}, {ci[1]:.4f}]；逐 seed 差值 {values['per_seed']}。区间仅反映已观察父场景，不涵盖全部训练不确定性。")
    report_lines.extend([
        "",
        "报告同时保存巡查按时率、未满足/迟到订单、块外承诺延误分钟数、合成能耗、改派/新增分配和策略决策 CPU/wall 时间。评价指标由独立事件回放审计复算；PPO reward 仅是训练信号。",
        "",
        "所有保留评价场景的结果已查看，因此这些父场景转入开发资料，不再作为后续正式确认集。样本量仅支持本轮小规模开发比较，不据此宣称数量泛化或覆盖训练 seed 不确定性。",
        "",
    ])
    (output / "report_zh.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary
