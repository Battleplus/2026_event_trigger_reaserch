from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import counterfactual_pairs as pairs
import cost_accounting
import event_scenarios
import gain_features
import gain_gate
import gain_oracle
from event_environment import EventReliefEnv


def _environment():
    scenario = copy.deepcopy(event_scenarios.build_event_scenarios()[0])
    env = EventReliefEnv(scenario)
    env.reset()
    return scenario, env


def _collect(env, *, horizon: int = 2, continuation=pairs.noop_continuation, forced: bool = False):
    scenario = env.scenario
    future = [row for row in scenario["event_tape"] if row["time"] > env.time]
    return pairs.collect_paired_label(
        env,
        state_id="state-test-001",
        scenario_id=scenario["parent_id"],
        parent_family="pilot:test-family",
        split="pilot",
        event_type="ordinary_candidate",
        forced=forced,
        replanner_version="Public-C1-Catalog-Replanner-v1",
        replan_action=copy.deepcopy(pairs.NOOP_ACTION),
        suffix_rows=[{"suffix_seed": 31, "future_events": future}],
        continuation_policy=continuation,
        replanner_cpu_seconds=0.002,
        horizon_minutes=horizon,
    )


def _rehash_label(label: dict) -> None:
    pair = label["pairs"][0]
    trace = pair["replan"]["trace"]
    trace.pop("trace_sha256", None)
    trace["trace_sha256"] = pairs._digest(trace)
    pair.pop("pair_sha256", None)
    pair["pair_sha256"] = pairs._digest(pair)
    label["gain_mean"] = pair["gain"]
    label.pop("label_sha256", None)
    label["label_sha256"] = pairs._digest(label)


def test_snapshot_suffix_hides_future_tape_and_same_pair_suffix_replays() -> None:
    scenario, env = _environment()
    snapshot = pairs.capture_snapshot(
        env,
        state_id="state-test-001",
        scenario_id=scenario["parent_id"],
        parent_family="pilot:test-family",
        split="pilot",
        event_type="ordinary_candidate",
        forced=False,
        replanner_version="Public-C1-Catalog-Replanner-v1",
    )
    public = pairs.model_public_observation(env)
    restored = pairs.restore_snapshot(snapshot)
    assert pairs.model_public_observation(restored) == public

    changed_future = copy.deepcopy(scenario)
    changed_future["event_tape"][0]["time"] += 1
    changed_future["event_tape_id"] = event_scenarios._digest(changed_future["event_tape"])
    changed_env = EventReliefEnv(changed_future)
    changed_env.reset()
    assert pairs.model_public_observation(changed_env) == public
    detection = {"event_type": "task_update", "ordinary": True}
    scope = {"affected_task_ids": ["n02-task-01"], "local_eligible": True}
    features = gain_features.extract_public_features(public, detection, scope)
    changed_features = gain_features.extract_public_features(
        pairs.model_public_observation(changed_env), detection, scope,
    )
    assert features == changed_features
    score = sum(features.values()) * 1e-6
    assert gain_gate.decide_gain(score, threshold=0.01) == gain_gate.decide_gain(
        sum(changed_features.values()) * 1e-6, threshold=0.01,
    )

    suffix = pairs.prepare_suffix_snapshot(
        snapshot,
        future_events=[row for row in scenario["event_tape"] if row["time"] > env.time],
        suffix_seed=31,
    )
    result = pairs.run_paired_counterfactual(
        snapshot,
        replan_action=copy.deepcopy(pairs.NOOP_ACTION),
        suffixes=[suffix],
        replanner_version="Public-C1-Catalog-Replanner-v1",
        replanner_cpu_seconds=0.002,
        horizon_minutes=2,
    )
    pair = result["pairs"][0]
    keep, replan = pair["keep"]["trace"], pair["replan"]["trace"]
    assert keep["event_tape_sha256"] == replan["event_tape_sha256"] == pair["event_tape_sha256"]
    assert keep["start_truth"] == replan["start_truth"]
    assert keep["raw_truth"] == replan["raw_truth"]
    assert [step["raw_events"] for step in keep["steps"]] == [
        step["raw_events"] for step in replan["steps"]
    ]


def test_snapshot_excludes_bound_runtime_methods_and_restores_state() -> None:
    scenario, env = _environment()
    public = pairs.model_public_observation(env)
    # Runtime callbacks can be attached by a runner or temporary probe. They
    # are executable behavior, not simulator state and must not enter JSON.
    env._runtime_probe = env.public_observation
    env._base._compatible = env._base._compatible

    snapshot = pairs.capture_snapshot(
        env,
        state_id="state-bound-method-001",
        scenario_id=scenario["parent_id"],
        parent_family="pilot:test-family",
        split="pilot",
        event_type="ordinary_candidate",
        forced=False,
        replanner_version="Public-C1-Catalog-Replanner-v1",
    )
    restored = pairs.restore_snapshot(snapshot)

    assert pairs.model_public_observation(restored) == public
    assert not hasattr(restored, "_runtime_probe")
    assert "_compatible" not in restored._base.__dict__


def test_positive_compute_delay_and_expired_command_are_executed_and_recorded() -> None:
    scenario, env = _environment()
    snapshot = pairs.capture_snapshot(
        env,
        state_id="state-delay-test-001",
        scenario_id=scenario["parent_id"],
        parent_family="pilot:test-family",
        split="pilot",
        event_type="ordinary_candidate",
        forced=False,
        replanner_version="Public-C1-Catalog-Replanner-v1",
    )
    suffix = pairs.prepare_suffix_snapshot(
        snapshot,
        future_events=[row for row in scenario["event_tape"] if row["time"] > env.time],
        suffix_seed=91,
    )

    delayed = pairs.run_paired_counterfactual(
        snapshot,
        replan_action=copy.deepcopy(pairs.NOOP_ACTION),
        suffixes=[suffix],
        continuation_policy=pairs.noop_continuation,
        replanner_version="Public-C1-Catalog-Replanner-v1",
        replanner_cpu_seconds=0.002,
        horizon_minutes=2,
        action_delay_ticks=1,
    )
    assert delayed["valid_mask"] is True
    delayed_trace = delayed["pairs"][0]["replan"]["trace"]
    assert delayed_trace["action_delay_ticks"] == 1
    assert delayed_trace["action_applied_step"] == 1
    assert delayed_trace["action_expired_reason"] is None
    assert len(delayed_trace["steps"]) == 2

    expired = pairs.run_paired_counterfactual(
        snapshot,
        replan_action=copy.deepcopy(pairs.NOOP_ACTION),
        suffixes=[suffix],
        continuation_policy=pairs.noop_continuation,
        replanner_version="Public-C1-Catalog-Replanner-v1",
        replanner_cpu_seconds=0.002,
        horizon_minutes=2,
        action_delay_ticks=2,
    )
    assert expired["valid_mask"] is True
    expired_trace = expired["pairs"][0]["replan"]["trace"]
    assert expired_trace["action_delay_ticks"] == 2
    assert expired_trace["action_applied_step"] is None
    assert expired_trace["action_expired_reason"] == "command_expires_after_requested_window"
    assert expired_trace["initial_planner_call_count"] == 1
    assert gain_oracle.validate_gain_label(expired)["valid"] is True


def test_explicit_noop_safety_planner_receipt_is_counted_and_oracle_accepts() -> None:
    _, env = _environment()

    def noop_safety_replanner(_env):
        return {
            "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
            "action": copy.deepcopy(pairs.NOOP_ACTION),
            "planner_invoked": True,
            "planner_cpu_seconds": 0.0,
            "safety_monitor_cpu_seconds": 0.0,
            "policy_version": "safety-replanner-test-v1",
        }

    label = _collect(env, horizon=2, continuation=noop_safety_replanner)
    assert label["valid_mask"] is True
    receipt_trace = label["pairs"][0]["replan"]["trace"]
    assert receipt_trace["planner_call_receipts"]
    assert all(row["action"] == pairs.NOOP_ACTION for row in receipt_trace["planner_call_receipts"])
    assert label["pairs"][0]["replan"]["cost"]["replan_components"]["call_count"] == 2.0
    result = gain_oracle.validate_gain_label(label)
    assert result["valid"] is True
    assert result["label_valid_mask"] is True


def test_terminal_remaining_work_is_the_only_completion_penalty() -> None:
    _, env = _environment()
    label = _collect(env, horizon=2)
    pair = label["pairs"][0]
    replan = pair["replan"]
    priorities = pair["task_universe"]["priority_weights"]
    endpoint = {row["id"]: row for row in replan["trace"]["raw_truth"]["tasks"]}
    remaining = sum(
        weight * (1.0 - float(endpoint[task_id]["progress"]))
        for task_id, weight in priorities.items()
    )

    assert label["schema"] == "gain-counterfactual-label/2"
    assert replan["cost"]["cost_accounting_schema"] == "gain-cost-accounting/2"
    assert replan["cost"]["execution_components"]["weighted_remaining_fraction"] == pytest.approx(remaining)
    assert replan["cost"]["execution_components"]["completion_cost"] == pytest.approx(remaining)
    assert gain_oracle.validate_gain_label(label)["valid"] is True


def test_unresolved_safety_evidence_is_retained_and_does_not_mask_defined_outcomes() -> None:
    _, env = _environment()

    def unresolved_safety(_env):
        reason = "public_state_insufficient_for_safety_resolution"
        return {
            "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
            "action": copy.deepcopy(pairs.NOOP_ACTION),
            "planner_invoked": False,
            "planner_cpu_seconds": 0.0,
            "safety_monitor_cpu_seconds": 0.0,
            "policy_version": "unresolved-safety-test-v1",
            "safety_report": {"status": "unknown", "checked_entities": 2},
            "safety_disposition": {
                "schema": "gain-safety-disposition/1",
                "status": "unresolved",
                "resolved": False,
                "unknown_reason": reason,
            },
            "unknown_reason": reason,
            "detection": {"event_type": "communication_degradation", "candidate": True},
        }

    label = _collect(env, horizon=2, continuation=unresolved_safety)
    pair = label["pairs"][0]
    assert label["valid_mask"] is True
    assert pair["valid_mask"] is True
    assert pair["gain"] is not None
    assert pair["unresolved_safety_disposition_count"] == 2
    assert pair["unresolved_safety_unknown_reasons"] == [
        "public_state_insufficient_for_safety_resolution",
        "public_state_insufficient_for_safety_resolution",
    ]
    for arm in ("keep", "replan"):
        trace = pair[arm]["trace"]
        assert trace["unresolved_safety_disposition_count"] == 1
        receipt = trace["continuation_policy_receipts"][0]
        assert all(field in receipt for field in pairs.CONTINUATION_SAFETY_FIELDS)
        evidence = {field: receipt[field] for field in pairs.CONTINUATION_SAFETY_FIELDS}
        assert receipt["safety_evidence_sha256"] == pairs._digest(evidence)
    checked = gain_oracle.validate_gain_label(label)
    assert checked["valid"] is True
    assert checked["label_valid_mask"] is True
    assert checked["recomputed_gain_mean"] == pytest.approx(pair["gain"])


def test_oracle_rejects_changed_safety_evidence_even_after_outer_rehash() -> None:
    _, env = _environment()

    def resolved_safety(_env):
        return {
            "receipt_schema": pairs.CONTINUATION_RECEIPT_SCHEMA,
            "action": copy.deepcopy(pairs.NOOP_ACTION),
            "planner_invoked": False,
            "planner_cpu_seconds": 0.0,
            "safety_monitor_cpu_seconds": 0.0,
            "policy_version": "resolved-safety-test-v1",
            "safety_report": {"status": "resolved", "checked_entities": 2},
            "safety_disposition": {
                "schema": "gain-safety-disposition/1",
                "status": "resolved",
                "resolved": True,
            },
            "unknown_reason": None,
            "detection": {"event_type": "communication_degradation", "candidate": True},
        }

    label = _collect(env, horizon=2, continuation=resolved_safety)
    receipt = label["pairs"][0]["replan"]["trace"]["continuation_policy_receipts"][0]
    receipt["safety_report"]["checked_entities"] = 99
    _rehash_label(label)

    with pytest.raises(ValueError, match="SAFETY_EVIDENCE_HASH_MISMATCH"):
        gain_oracle.validate_gain_label(label)


@pytest.mark.parametrize("tamper", ["completed_without_event", "progress_without_service"])
def test_oracle_rejects_rehashed_endpoint_state_inconsistent_with_raw_events(tamper: str) -> None:
    _, env = _environment()
    label = _collect(env, horizon=2)
    pair = label["pairs"][0]
    trace = pair["replan"]["trace"]
    endpoint = trace["raw_truth"]["tasks"]
    target = next(row for row in endpoint if row["id"] in pair["task_universe"]["task_ids"] and row["status"] != "completed")
    if tamper == "completed_without_event":
        target["status"] = "completed"
        target["completed_at"] = trace["end_time"]
        target["service_progress"] = trace["task_progress_contract"][target["id"]]["service_total"]
        target["progress"] = 1.0
    else:
        target["progress"] = 0.987654

    pair["replan"]["cost"] = cost_accounting.calculate_branch_cost(
        trace,
        task_priorities=pair["task_universe"]["priority_weights"],
        planner_cpu_seconds=trace["initial_planner_cpu_seconds"],
        config=cost_accounting.GainCostConfig(**pair["replan"]["cost"]["cost_config"]),
    )
    pair["gain"] = pair["keep"]["cost"]["J_total"] - pair["replan"]["cost"]["J_total"]
    _rehash_label(label)

    expected = (
        "GAIN_ORACLE_ENDPOINT_COMPLETION_WITHOUT_EVENT"
        if tamper == "completed_without_event"
        else "GAIN_ORACLE_ENDPOINT_PROGRESS_MISMATCH"
    )
    with pytest.raises(ValueError, match=expected):
        gain_oracle.validate_gain_label(label)


def test_forced_safety_state_is_retained_as_invalid_label() -> None:
    _, env = _environment()
    label = _collect(env, forced=True)
    assert label["valid_mask"] is False
    assert label["unknown_reason"] == "forced_safety_state"
    assert gain_oracle.validate_gain_label(label)["label_valid_mask"] is False


def test_oracle_accepts_service_progress_reset_after_task_failure() -> None:
    task = {
        "id": "task-1", "kind": "inspection", "known": True,
        "status": "active", "progress": 0.0, "service_progress": 0.0,
    }
    trace = {
        "start_truth": {"tasks": [copy.deepcopy(task)], "subtasks": []},
        "raw_truth": {"tasks": [{**task, "status": "pending", "failure_reason": "platform_failure"}], "subtasks": []},
        "task_progress_contract": {
            "task-1": {"service_total": 2.0, "workload_share": 1.0, "parent_task_id": None},
        },
    }
    trace["task_progress_contract_sha256"] = gain_oracle._digest(trace["task_progress_contract"])
    gain_oracle._verify_endpoint_against_events(trace, [
        {"type": "service", "task_id": "task-1", "duration": 1.0},
        {"type": "task_failed", "task_id": "task-1"},
    ])
