from civilian import v4_analysis
from civilian.v4_policy import POLICY_SEEDS, VARIANTS
from civilian.v4_rollout import (
    _classify_new_queue_risks, _external_queue_event_evidence, _step_execution_evidence,
)


def _complete_queue_evidence():
    return {
        "actual_executed_controllable_hard_constraint_assessment": {
            "status": "complete", "scope": "observed_public_task_queue_feasibility",
            "attribution_counts": {"controllable": 0, "external": 0, "unknown": 0}},
        "minute_execution_evidence": [{
            "scope": "observed_public_task_queue_feasibility",
            "pre_action_safety": {}, "post_step_safety": {},
            "attribution_basis": "entity overlap is associative evidence, not proof of causation"}],
        "controllable_violation_scope": (
            "observed_public_task_queue_feasibility; not complete physical or flight safety"),
    }


def test_analysis_averages_paired_training_seeds_and_reports_positive_cost_benefit(monkeypatch):
    monkeypatch.setattr(v4_analysis, "BOOTSTRAP_REPLICATES", 200)
    variants = {}
    improved = "Event-G-WM"
    for variant_index, variant in enumerate(VARIANTS):
        variants[variant] = {}
        for seed_index, seed in enumerate(POLICY_SEEDS):
            variants[variant][str(seed)] = {
                "episodes": [
                    {"parent_id": f"test-{scenario:03d}",
                         "validation_cost": float(10 + (0 if variant == improved else 1) + seed_index / 10),
                         "forced_safety_report_count": 1,
                     "resolved_forced_safety_report_count": 1,
                     "newly_introduced_controllable_hard_constraint_violations": 0,
                         **_complete_queue_evidence(),
                         "environment_metrics": {
                             "task_outcomes": [{"priority": 1.0, "completed": True},
                                               {"priority": 3.0, "completed": False}],
                             "metrics": {"failure_rate": {"value": 0.1},
                                         "critical_failure_rate": {"value": 0.02}},
                     }}
                    for scenario in range(140)
                ]
            }
    report = v4_analysis.analyze_paired_evaluation(
        {"status": "complete", "split": "test", "variants": variants})
    assert report["scenario_count"] == 140
    assert report["seed_repeat_rule"] == "average three paired seeds within each scenario"
    assert len(report["paired_comparisons"]) == 7
    comparison = next(row for row in report["paired_comparisons"] if row["method"] == improved)
    assert comparison["cost"]["mean_difference"] > 0
    assert comparison["cost"]["ci95"][0] > 0
    assert comparison["safety_completion_guardrails"]["passed"]


def test_unknown_execution_safety_evidence_cannot_pass_guardrails(monkeypatch):
    monkeypatch.setattr(v4_analysis, "BOOTSTRAP_REPLICATES", 20)
    variants = {}
    for variant in VARIANTS:
        variants[variant] = {}
        for seed in POLICY_SEEDS:
            variants[variant][str(seed)] = {
                "episodes": [
                    {"parent_id": f"test-{scenario:03d}", "validation_cost": 10.0,
                     "forced_safety_report_count": 0, "resolved_forced_safety_report_count": 0,
                     "newly_introduced_controllable_hard_constraint_violations": 0,
                     **_complete_queue_evidence(),
                     "environment_metrics": {
                         "task_outcomes": [{"priority": 1.0, "completed": True}],
                         "metrics": {"failure_rate": {"value": 0.0},
                                     "critical_failure_rate": {"value": 0.0}},
                     }}
                    for scenario in range(140)
                ]
            }
    affected = variants["Event-G"][str(POLICY_SEEDS[0])]["episodes"][0]
    affected["forced_safety_report_count"] = 1
    affected["resolved_forced_safety_report_count"] = 1
    affected["newly_introduced_controllable_hard_constraint_violations"] = None
    affected["actual_executed_controllable_hard_constraint_assessment"] = {"status": "unassessed"}

    report = v4_analysis.analyze_paired_evaluation(
        {"status": "complete", "split": "test", "variants": variants})
    guardrails = report["paired_comparisons"][0]["safety_completion_guardrails"]
    assert not guardrails["safety_evidence_complete"]
    assert guardrails["newly_introduced_controllable_hard_constraint_violations"] is None
    assert guardrails["zero_new_controllable_hard_constraint_violations"] is None
    assert guardrails["forced_safety_handling_rate"] == 1.0
    assert guardrails["forced_safety_handling_100_percent"] is True
    assert not guardrails["passed"]


def test_queue_risk_attribution_requires_unambiguous_entity_overlap():
    before = {"violations": []}
    action_only = {"violations": [{"type": "overlap", "platform_id": "p1", "task_id": "t1"}]}
    external_only = {"violations": [{"type": "overlap", "platform_id": "p2", "task_id": "t2"}]}
    dual = {"violations": [{"type": "overlap", "platform_id": "p1", "task_id": "t2"}]}
    action = {("platform", "p1"), ("task", "t1")}

    assert _classify_new_queue_risks(before, action_only, action, []) [0]["attribution"] == "controllable"
    external = [{"entities": [{"type": "platform", "id": "p2"}, {"type": "task", "id": "t2"}]}]
    assert _classify_new_queue_risks(before, external_only, action, external)[0]["attribution"] == "external"
    ambiguous = [{"entities": [{"type": "task", "id": "t2"}]}]
    assert _classify_new_queue_risks(before, dual, action, ambiguous)[0]["attribution"] == "unknown"
    unconfirmed = {("task", "t1")}
    assert _classify_new_queue_risks(before, action_only, set(), [], unconfirmed)[0]["attribution"] == "unknown"
    assert _classify_new_queue_risks(before, before, action, external) == []

    observed = _external_queue_event_evidence([
        {"type": "energy_fraction_drop", "platform_id": "p1"},
        {"type": "communication_changed", "links": [{"source": "p1", "target": "p2"}]},
        {"type": "task_arrived", "task_id": "t3"},
    ])
    assert [row["type"] for row in observed] == [
        "energy_fraction_drop", "communication_changed", "task_arrived"]


def test_external_or_unknown_queue_risks_fail_analysis_guardrail():
    row = {"newly_introduced_controllable_hard_constraint_violations": 0,
           **_complete_queue_evidence()}
    assert v4_analysis._valid_hard_constraint_evidence(row)
    row["actual_executed_controllable_hard_constraint_assessment"]["attribution_counts"]["external"] = 2
    assert v4_analysis._valid_hard_constraint_evidence(row)
    row["actual_executed_controllable_hard_constraint_assessment"]["attribution_counts"]["unknown"] = 1
    assert not v4_analysis._valid_hard_constraint_evidence(row)


def test_dispatch_execution_status_requires_public_state_change():
    before = {"time": 4, "tasks": [{"id": "t1", "assignee": None, "assignees": [], "started_at": None}],
              "platforms": [{"id": "p1", "active_task_id": None}]}
    after = {"time": 5, "tasks": [{"id": "t1", "assignee": "p1", "assignees": ["p1"], "started_at": 5}],
             "platforms": [{"id": "p1", "active_task_id": "t1"}]}
    dispatch = {"platform_id": "p1", "task_id": "t1", "source": "reconstruction"}
    info = {"operational_dispatches": [dispatch], "raw_events": [], "operation_receipts": []}
    evidence = _step_execution_evidence(before, {"dispatches": [dispatch]}, info, [],
                                        {"violations": []}, {"violations": []}, after)
    assert evidence["submitted_to_execution_dispatches"][0]["execution_status"] == "confirmed"
    assert "accepted_dispatches" not in evidence

    after["tasks"][0].update({"assignee": None, "assignees": [], "started_at": None})
    after["platforms"][0]["active_task_id"] = None
    risk = {"violations": [{"code": "queued_task_deadline_infeasible", "task_id": "t1", "platform_id": "p1"}]}
    evidence = _step_execution_evidence(before, {"dispatches": [dispatch]}, info, [],
                                        {"violations": []}, risk, after)
    assert evidence["submitted_to_execution_dispatches"][0]["execution_status"] == "unconfirmed"
    assert evidence["new_risks"][0]["attribution"] == "unknown"
