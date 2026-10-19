from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import cost_accounting
import gain_evaluation


def _temporary_directory() -> Path:
    path = ROOT / "tests" / f".gain-eval-{uuid4().hex}"
    path.mkdir()
    return path


def _remove_temporary_directory(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _modules() -> dict:
    return {
        "pairs": SimpleNamespace(NOOP_ACTION={}),
        "scenarios": SimpleNamespace(validate_gain_scenario=lambda _scenario: None),
        "features": object(), "gate": object(), "env_class": object(),
        "trigger_class": object(), "identify_scope": lambda *_args: {},
    }


def _scenario(scenario_id: str) -> dict:
    return {
        "scenario_id": scenario_id, "parent_id": f"parent-{scenario_id}",
        "split": "test", "condition": "none_or_light", "parent_family": "none_or_light",
        "scenario_content_sha256": (scenario_id.encode("ascii").hex() * 32)[:64],
        "event_tape": [], "base_scenario": {"horizon": 1},
    }


def _summary(scenario: dict, method: str, *, unsafe: bool = False, score: bool = True) -> dict:
    row = {
        "schema": "gain-closed-loop-episode/1", "method": method,
        "scenario_id": scenario["scenario_id"], "split": scenario["split"],
        "episode_status": "complete_unsafe" if unsafe else "complete",
        "safety_unresolved": int(unsafe), "forced_events": 0, "forced_handled": 0,
    }
    if score:
        row.update({"J_total": 1.0, "weighted_completion_rate": 0.5})
    return row


def _evaluation_call(monkeypatch, output_dir, scenarios, summaries, *, resource_check=None):
    loaded = []
    observed_priorities = []

    class Runtime:
        def predict(self, _features):
            return {"gain_hat": 0.0, "epsilon": 0.0}

    def runtime_loader(path):
        loaded.append(str(path))
        return Runtime()

    def observer(scenario, _modules, _helpers):
        priorities = {"task-a": 3.0, "task-b": 1.0}
        return priorities, {"scenario_id": scenario["scenario_id"], "steps": []}

    def one_episode(scenario, method, _runtime, _modules, _helpers, _config, _cost, priorities):
        observed_priorities.append((scenario["scenario_id"], method, dict(priorities)))
        row = summaries(scenario, method)
        trajectory = {"scenario_id": scenario["scenario_id"], "evaluation": {"method": method}}
        return row, trajectory

    monkeypatch.setattr(gain_evaluation, "_observe_task_priorities", observer)
    monkeypatch.setattr(gain_evaluation, "_run_one_episode", one_episode)
    checks = []
    helpers = {"legacy_environment_scenario": lambda row: dict(row)}
    if resource_check is not None:
        helpers["resource_check"] = lambda: (checks.append(True), resource_check())[1]
    result = gain_evaluation.run_closed_loop_evaluation(
        scenarios,
        {key: f"{key}.json" for key in gain_evaluation.MODEL_PATH_KEYS.values()},
        output_dir, {"statistics": {"bootstrap_replicates": 20, "permutation_replicates": 20}},
        runtime_loader=runtime_loader, evaluation_modules=_modules(), runner_api=helpers,
    )
    return result, loaded, observed_priorities, checks


def test_production_and_independent_scores_match_raw_trace_components() -> None:
    trace = {
        "start_truth": {"tasks": [{"id": "task-a"}, {"id": "task-b"}]},
        "raw_truth": {"tasks": [
            {"id": "task-a", "status": "completed", "progress": 1.0},
            {"id": "task-b", "status": "active", "progress": 0.5},
        ]},
        "initial_planner_call_count": 1,
        "continuation_policy_cpu_seconds": 0.0,
        "evaluation_monitor_cpu_seconds": 0.25,
        "steps": [{"stage": "execution", "planner_invoked": False, "raw_events": [
            {"type": "delay_accrued", "minutes": 3.0},
            {"type": "move", "energy": 2.0, "environmental_risk": 0.2},
            {"type": "communication_tx", "message_count": 2},
            {"type": "task_switch", "time": 1, "task_id": "task-b"},
        ]}],
        "planner_call_receipts": [],
    }
    priorities = {"task-a": 3.0, "task-b": 1.0}
    independent = gain_evaluation.recompute_episode_score(
        trace, task_priorities=priorities, planner_cpu_seconds=0.5,
    )
    production = cost_accounting.calculate_branch_cost(
        trace, task_priorities=priorities, planner_cpu_seconds=0.5,
    )
    gain_evaluation._assert_score_parity(production, independent)
    assert production["execution_components"]["weighted_remaining_fraction"] == pytest.approx(0.125)
    assert production["safety_components"]["evaluation_monitor_cpu_seconds"] == pytest.approx(0.25)
    assert independent["online_cpu_seconds"] == pytest.approx(0.75)


def test_attempt_cooldown_counts_noop_attempts_and_forced_events_bypass() -> None:
    assert gain_evaluation._attempt_cooldown_candidate(
        True, forced=False, current_time=1, next_allowed_time=0,
    )
    next_allowed = 1 + 4
    assert not gain_evaluation._attempt_cooldown_candidate(
        True, forced=False, current_time=4, next_allowed_time=next_allowed,
    )
    assert gain_evaluation._attempt_cooldown_candidate(
        True, forced=False, current_time=next_allowed, next_allowed_time=next_allowed,
    )
    assert gain_evaluation._attempt_cooldown_candidate(
        True, forced=True, current_time=next_allowed + 1, next_allowed_time=next_allowed + 4,
    )


def test_noop_observer_freezes_first_public_priority_and_observes_full_task_universe() -> None:
    class ObserverEnv:
        def __init__(self, _scenario):
            self.time = 0
            self.actions = []
            self.public = {"time": 0, "tasks": [
                {"id": "task-a", "known": True, "priority": 3.0},
                {"id": "task-b", "known": False, "priority": 99.0},
            ]}
            self.history = [self.public]

        def reset(self):
            return self.public

        def _raw_truth(self):
            return {"tasks": [{"id": "task-a"}, {"id": "task-b"}]}

        def step(self, action):
            self.actions.append(action)
            self.time += 1
            self.public = {"time": self.time, "tasks": [
                {"id": "task-a", "known": True, "priority": 1.0},
                {"id": "task-b", "known": True, "priority": 2.0 if self.time == 1 else 100.0},
            ]}
            self.history.append(self.public)
            return self.public, 0.0, False, False, {}

        def export_trajectory(self):
            return {"steps": [{"action": action} for action in self.actions], "end_time": self.time}

    noop_action = {"kind": "noop"}
    modules = {
        "pairs": SimpleNamespace(
            NOOP_ACTION=noop_action,
            model_public_observation=lambda env: env.public,
        ),
        "env_class": ObserverEnv,
    }
    helpers = SimpleNamespace(legacy_environment_scenario=lambda scenario: scenario)
    scenario = {
        "scenario_id": "observer-test", "split": "test",
        "base_scenario": {"horizon": 2},
    }

    priorities, trace = gain_evaluation._observe_task_priorities(scenario, modules, helpers)

    assert priorities == {"task-a": 3.0, "task-b": 2.0}
    assert [step["action"] for step in trace["steps"]] == [noop_action, noop_action]
    assert trace["evaluation"]["method"] == "NoopPriorityObserver"
    assert trace["evaluation"]["scenario_id"] == "observer-test"


def test_episode_stops_without_requesting_a_new_decision_at_horizon() -> None:
    public = {"time": 0, "tasks": [{"id": "task-a", "known": True, "priority": 1.0}]}

    class HorizonEnv:
        def __init__(self, _scenario):
            self.time = 0
            self.public = dict(public)
            self._steps = []

        def reset(self):
            return self.public

        def _raw_truth(self):
            return {"tasks": [{"id": "task-a", "status": "active", "progress": 0.0}]}

        def step(self, _action):
            self.time += 1
            self.public = {"time": self.time, "tasks": public["tasks"]}
            self._steps.append({"stage": "execution", "raw_events": [], "planner_invoked": False})
            return self.public, 0.0, False, False, {}

        def export_trajectory(self):
            return {
                "steps": list(self._steps), "raw_truth": self._raw_truth(),
                "end_time": self.time, "summary": {},
            }

    class NeverDetect:
        def __init__(self, *_args):
            pass

        def detect(self, *_args, **_kwargs):
            raise AssertionError("horizon endpoint must not request a new detection")

    modules = {
        "pairs": SimpleNamespace(
            NOOP_ACTION={"kind": "noop"},
            model_public_observation=lambda env: env.public,
            _public_history=lambda env: [env.public],
        ),
        "env_class": HorizonEnv, "trigger_class": NeverDetect,
    }
    helpers = SimpleNamespace(legacy_environment_scenario=lambda scenario: scenario)
    scenario = {
        "scenario_id": "horizon-test", "split": "test", "condition": "none_or_light",
        "parent_family": "none_or_light", "scenario_content_sha256": "b" * 64,
        "event_tape": [], "base_scenario": {"horizon": 1},
    }

    summary, trajectory = gain_evaluation._run_one_episode(
        scenario, "ExistingET", None, modules, helpers, {}, cost_accounting.DEFAULT_CONFIG,
        {"task-a": 1.0},
    )

    assert summary["episode_status"] == "complete"
    assert trajectory["evaluation"]["decisions"] == []
    assert trajectory["initial_planner_call_count"] == 0
    assert trajectory["evaluation"]["timing_stage_semantics"]["planning"].startswith("includes fixed-replanner")


def test_evaluator_loads_each_frozen_model_once_and_shares_priority_denominator(
    monkeypatch,
) -> None:
    scenarios = [_scenario("test-a"), _scenario("test-b")]
    output_dir = _temporary_directory()
    try:
        result, loaded, observed, checks = _evaluation_call(
            monkeypatch, output_dir, scenarios,
            lambda scenario, method: _summary(scenario, method, unsafe=method == "GainTree"),
            resource_check=lambda: None,
        )
        persisted = json.loads((output_dir / "EVALUATION.json").read_text(encoding="utf-8"))
    finally:
        _remove_temporary_directory(output_dir)

    assert len(loaded) == len(gain_evaluation.MODEL_PATH_KEYS)
    assert result["resource_check_count"] == len(scenarios) * (1 + len(gain_evaluation.METHODS))
    assert len(checks) == result["resource_check_count"]
    assert {tuple(sorted(row[2].items())) for row in observed} == {
        (("task-a", 3.0), ("task-b", 1.0)),
    }
    for scenario in scenarios:
        rows = [item for item in observed if item[0] == scenario["scenario_id"]]
        assert len(rows) == len(gain_evaluation.METHODS)
        assert len({tuple(sorted(item[2].items())) for item in rows}) == 1
    unsafe = next(row for row in result["episode_results"] if row["method"] == "GainTree")
    assert unsafe["episode_status"] == "complete_unsafe"
    assert unsafe["J_total"] == 1.0
    assert result["panel_analysis_status"] == "complete"
    assert "ExistingETAttemptCooldown" in result["method_names"]
    assert result["analysis_contract"]["primary_holm_family_size"] == 30
    assert result["method_support"]["GainMLP8201"]["status"] == "INCONCLUSIVE"
    assert len(result["method_support"]["GainMLP8201"]["baseline_criteria"]) == 2
    assert "not a deployed ensemble" in result["mlp_seed_average_analysis"]["claim_limit"]
    assert persisted["scenario_ids"] == ["test-a", "test-b"]
    assert persisted["model_load_receipts"]["GainTree"]["bundle_path"] == "gain_tree.json"


def test_evaluator_rejects_incomplete_panel_without_dropping_unsafe_scores(
    monkeypatch,
) -> None:
    scenario = _scenario("test-only")

    def rows(scenario, method):
        if method == "RiskET":
            return _summary(scenario, method, score=False)
        return _summary(scenario, method, unsafe=method == "Periodic4")

    output_dir = _temporary_directory()
    try:
        result, *_ = _evaluation_call(monkeypatch, output_dir, [scenario], rows)
    finally:
        _remove_temporary_directory(output_dir)

    assert result["status"] == "complete_with_episode_failures"
    assert result["panel_analysis_status"] == "rejected_incomplete_panel"
    unsafe = next(row for row in result["episode_results"] if row["method"] == "Periodic4")
    assert unsafe["episode_status"] == "complete_unsafe"
    assert unsafe["J_total"] == 1.0
    assert "RiskET" in result["panel_analysis_error"]
