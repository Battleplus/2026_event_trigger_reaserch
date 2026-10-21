"""Scenario-paired bootstrap and Holm analysis for the frozen V4 test set."""
from __future__ import annotations

import math
import random
from typing import Any

try:
    from .v4_policy import VARIANTS, POLICY_SEEDS
except ImportError:
    from v4_policy import VARIANTS, POLICY_SEEDS

BOOTSTRAP_REPLICATES = 10_000
REFERENCE = "Event-G"
METRICS = ("validation_cost", "weighted_completion_rate", "failure_rate", "critical_failure_rate")
CONTRASTS = (("Periodic-G", "Event-G"),
             ("Periodic-G", "Periodic-G-WM"),
             ("Event-G", "Event-G-WM-Decision"),
             ("Periodic-G-WM", "Event-G-WM-Decision"),
             ("Event-G-WM-Decision", "Event-G-WM"),
             ("Event-G-WM-NoU", "Event-G-WM"),
             ("Event-G-WM-Global", "Event-G-WM"))


def _metric(summary: dict, name: str) -> float:
    if name == "validation_cost":
        value = summary.get("validation_cost")
    elif name == "weighted_completion_rate":
        outcomes = summary.get("environment_metrics", {}).get("task_outcomes", [])
        total_weight = sum(max(0.0, float(row.get("priority", 0.0))) for row in outcomes)
        value = (sum(max(0.0, float(row.get("priority", 0.0))) * float(bool(row.get("completed")))
                     for row in outcomes) / total_weight if total_weight else None)
    else:
        metrics = summary.get("environment_metrics", {})
        entry = metrics.get("metrics", {}).get(name)
        if isinstance(entry, dict):
            value = entry.get("value")
        else:
            value = metrics.get(name)
            if isinstance(value, dict):
                value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"V4_ANALYSIS_METRIC_INVALID:{name}")
    return float(value)


def _valid_count(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and float(value) >= 0)


def _valid_hard_constraint_evidence(row: dict) -> bool:
    value = row.get("newly_introduced_controllable_hard_constraint_violations")
    if not _valid_count(value):
        return False
    assessment = row.get("actual_executed_controllable_hard_constraint_assessment")
    if not isinstance(assessment, dict) or assessment.get("status") != "complete":
        return False
    if assessment.get("scope") != "observed_public_task_queue_feasibility":
        return False
    counts = assessment.get("attribution_counts")
    evidence = row.get("minute_execution_evidence")
    if not isinstance(counts, dict) or not isinstance(evidence, list) or not evidence:
        return False
    if counts.get("unknown") != 0:
        return False
    if counts.get("controllable") != value or row.get("controllable_violation_scope") != (
            "observed_public_task_queue_feasibility; not complete physical or flight safety"):
        return False
    return all(isinstance(item, dict)
               and item.get("scope") == "observed_public_task_queue_feasibility"
               and item.get("attribution_basis") == "entity overlap is associative evidence, not proof of causation"
               and isinstance(item.get("pre_action_safety"), dict)
               and isinstance(item.get("post_step_safety"), dict)
               for item in evidence)


def _percentile(sorted_values: list[float], p: float) -> float:
    index = (len(sorted_values)-1)*p
    lower=int(index); upper=min(len(sorted_values)-1,lower+1)
    return sorted_values[lower] + (sorted_values[upper]-sorted_values[lower])*(index-lower)


def _paired_stats(differences: list[float], *, seed: int) -> dict[str, Any]:
    if not differences:
        raise ValueError("V4_ANALYSIS_EMPTY_PAIRED_DIFFERENCES")
    n=len(differences); observed=sum(differences)/n
    rng=random.Random(seed); boot=[]; extreme=0
    for _ in range(BOOTSTRAP_REPLICATES):
        boot_mean=sum(differences[rng.randrange(n)] for _ in range(n))/n
        boot.append(boot_mean)
        null_mean=sum(value if rng.getrandbits(1) else -value for value in differences)/n
        if abs(null_mean) >= abs(observed): extreme += 1
    boot.sort()
    return {"mean_difference":observed,"ci95":[_percentile(boot,.025),_percentile(boot,.975)],
            "paired_randomization_p":(extreme+1)/(BOOTSTRAP_REPLICATES+1),"scenario_count":n,
            "bootstrap_replicates":BOOTSTRAP_REPLICATES}


def _holm(rows: list[dict]) -> None:
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]["cost"]["paired_randomization_p"])
    m=len(ordered); previous=0.0
    for rank,index in enumerate(ordered):
        adjusted=min(1.0,max(previous,(m-rank)*rows[index]["cost"]["paired_randomization_p"]))
        rows[index]["cost"]["holm_adjusted_p"]=adjusted; previous=adjusted


def analyze_paired_evaluation(evaluation: dict) -> dict[str, Any]:
    if evaluation.get("status")!="complete" or evaluation.get("split")!="test":
        raise ValueError("V4_ANALYSIS_TEST_EVALUATION_REQUIRED")
    raw=evaluation.get("variants")
    if not isinstance(raw,dict) or set(raw)!=set(VARIANTS):
        raise ValueError("V4_ANALYSIS_VARIANT_SET")
    by_variant: dict[str,dict[str,dict[str,dict[str,float]]]]={}
    common_ids=None
    means={}
    for variant in VARIANTS:
        by_variant[variant]={}
        for seed in POLICY_SEEDS:
            pack=raw[variant].get(str(seed))
            if not isinstance(pack,dict) or not isinstance(pack.get("episodes"),list):
                raise ValueError("V4_ANALYSIS_SEED_EPISODES_MISSING")
            index={str(row.get("parent_id")):row for row in pack["episodes"]}
            if len(index)!=140 or None in index:
                raise ValueError("V4_ANALYSIS_TEST_PARENT_COVERAGE")
            ids=set(index)
            if common_ids is None: common_ids=ids
            elif ids!=common_ids: raise ValueError("V4_ANALYSIS_UNPAIRED_PARENT_IDENTITIES")
            by_variant[variant][str(seed)]={parent:{metric:_metric(row,metric) for metric in METRICS}
                                             for parent,row in index.items()}
        means[variant]={}
        for parent in sorted(common_ids or []):
            means[variant][parent]={metric:sum(by_variant[variant][str(seed)][parent][metric] for seed in POLICY_SEEDS)/len(POLICY_SEEDS)
                                    for metric in METRICS}
    comparisons=[]
    for reference, method in CONTRASTS:
        stats={metric:_paired_stats([
                    (means[reference][parent][metric]-means[method][parent][metric]) if metric == "validation_cost"
                    else (means[method][parent][metric]-means[reference][parent][metric])
                    for parent in sorted(common_ids or [])],
                                    seed=8201+sum(map(ord,method))+sum(map(ord,metric)))
               for metric in METRICS}
        stats["validation_cost"].update({"direction":"positive_is_lower_cost_for_method",
                                          "hypothesis_family":"seven preregistered paired contrasts"})
        stats["cost"] = stats["validation_cost"]
        episode_rows = [row for variant in (reference, method) for seed in POLICY_SEEDS
                        for row in raw[variant][str(seed)]["episodes"]]
        evidence_fields = ("forced_safety_report_count", "resolved_forced_safety_report_count",
                           "newly_introduced_controllable_hard_constraint_violations")
        forced_counts_complete = all(
            all(field in row and _valid_count(row[field]) for field in evidence_fields[:2])
            for row in episode_rows
        )
        hard_constraint_evidence_complete = all(_valid_hard_constraint_evidence(row) for row in episode_rows)
        safety_evidence_complete = forced_counts_complete and hard_constraint_evidence_complete
        forced = sum(int(row["forced_safety_report_count"]) for row in episode_rows) if forced_counts_complete else None
        resolved = sum(int(row["resolved_forced_safety_report_count"]) for row in episode_rows) if forced_counts_complete else None
        violations = (sum(int(row["newly_introduced_controllable_hard_constraint_violations"])
                          for row in episode_rows) if hard_constraint_evidence_complete else None)
        completion_ok=stats["weighted_completion_rate"]["ci95"][0] > -0.02
        safety_ok = bool(forced_counts_complete and hard_constraint_evidence_complete
                         and resolved == forced and violations == 0)
        stats.update({"method":method,"reference":reference,
                      "safety_completion_guardrails":{"weighted_completion_noninferior_2pp":completion_ok,
                          "forced_safety_handling_rate": (resolved / forced if forced else None),
                          "forced_safety_handling_100_percent": (resolved == forced if forced else None),
                          "safety_evidence_complete": safety_evidence_complete,
                          "forced_safety_counts_complete": forced_counts_complete,
                          "hard_constraint_evidence_complete": hard_constraint_evidence_complete,
                          "newly_introduced_controllable_hard_constraint_violations": violations,
                          "zero_new_controllable_hard_constraint_violations": (violations == 0 if violations is not None else None),
                          "passed":completion_ok and safety_ok},
                      "cost_claim_supported":False})
        comparisons.append(stats)
    _holm(comparisons)
    for row in comparisons:
        row["cost_claim_supported"]=(row["cost"]["ci95"][0]>0 and row["cost"]["holm_adjusted_p"]<0.05
                                      and row["safety_completion_guardrails"]["passed"])
    return {"schema":"n02-v4-paired-analysis/1","status":"complete","reference":REFERENCE,
            "scenario_count":len(common_ids or []),"seed_repeat_rule":"average three paired seeds within each scenario",
            "contrasts":[{"reference":reference,"method":method} for reference,method in CONTRASTS],
            "primary_metric":"validation_cost","bootstrap_replicates":BOOTSTRAP_REPLICATES,
            "method_means_by_scenario":means,"paired_comparisons":comparisons,
            "holm_family_size":len(CONTRASTS),
            "claims_require_cost_benefit_ci_above_zero_holm_p_below_0_05_and_guardrails":True}
