"""Scenario-paired summaries for gain-gate experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


ANALYSIS_SCHEMA = "gain-paired-analysis/1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _seed_from(seed: int, metric: str, method: str) -> int:
    digest = hashlib.sha256(f"{seed}:{metric}:{method}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _scenario_id(row: Mapping[str, Any]) -> str:
    value = row.get("scenario_id", row.get("scenario_instance_id", row.get("parent_id")))
    if value is None or not str(value):
        raise ValueError("GAIN_ANALYSIS_SCENARIO_ID_REQUIRED")
    return str(value)


def _method(row: Mapping[str, Any]) -> str:
    value = row.get("method", row.get("policy"))
    if value is None or not str(value):
        raise ValueError("GAIN_ANALYSIS_METHOD_REQUIRED")
    return str(value)


def _finite_metric(row: Mapping[str, Any], metric: str) -> float:
    raw = row.get(metric)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"GAIN_ANALYSIS_FINITE_METRIC_REQUIRED:{metric}")
    return float(raw)


def aggregate_scenario_units(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Average repeated runs within each scenario/method unit."""
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        scenario = _scenario_id(row)
        method = _method(row)
        for metric in metrics:
            values[(scenario, method, metric)].append(_finite_metric(row, metric))
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (scenario, method, metric), observations in values.items():
        result[scenario][method][metric] = float(np.mean(observations))
    return {scenario: dict(methods) for scenario, methods in result.items()}


def _paired_values(
    units: Mapping[str, Mapping[str, Mapping[str, float]]],
    baseline: str,
    method: str,
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    paired_scenarios = sorted(
        scenario for scenario, methods in units.items()
        if baseline in methods and method in methods
        and metric in methods[baseline] and metric in methods[method]
    )
    if not paired_scenarios:
        raise ValueError(f"GAIN_ANALYSIS_NO_PAIRED_SCENARIOS:{baseline}:{method}:{metric}")
    baseline_values = np.asarray([units[scenario][baseline][metric] for scenario in paired_scenarios], dtype=np.float64)
    method_values = np.asarray([units[scenario][method][metric] for scenario in paired_scenarios], dtype=np.float64)
    return paired_scenarios, baseline_values, method_values, method_values - baseline_values


def _bootstrap_ci(differences: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    if replicates < 1:
        raise ValueError("GAIN_ANALYSIS_BOOTSTRAP_REPLICATES")
    draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_permutation_p(differences: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(differences)))
    if np.allclose(differences, 0.0):
        return 1.0
    if len(differences) <= 18:
        count = 1 << len(differences)
        signs = np.empty((count, len(differences)), dtype=np.float64)
        indices = np.arange(count, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(len(differences), dtype=np.uint64)) & 1
        signs[:] = np.where(bits == 1, 1.0, -1.0)
        null = np.abs((signs * differences[None, :]).mean(axis=1))
        return float(np.mean(null >= observed - 1e-15))
    if permutations < 1:
        raise ValueError("GAIN_ANALYSIS_PERMUTATIONS")
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(differences)))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed - 1e-15)) / (permutations + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values keyed by comparison identity."""
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for _, value in ordered):
        raise ValueError("GAIN_ANALYSIS_P_VALUE_RANGE")
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, p_value) in enumerate(ordered):
        value = min(1.0, (count - index) * float(p_value))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def paired_analysis(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    *,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    seed: int = 8201,
    expected_scenario_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyze paired methods with independent scenario instances as units.

    Repeated rows within a scenario/method are averaged before inference.
    This keeps event windows, suffix branches, and training seeds from being
    counted as independent scenarios.
    """
    if not rows or not comparisons or not metrics:
        raise ValueError("GAIN_ANALYSIS_INPUT_REQUIRED")
    units = aggregate_scenario_units(rows, metrics)
    expected_scenarios = sorted(set(map(str, expected_scenario_ids))) if expected_scenario_ids is not None else sorted(units)
    if not expected_scenarios:
        raise ValueError("GAIN_ANALYSIS_EXPECTED_SCENARIOS_EMPTY")
    reports: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for comparison in comparisons:
        baseline = str(comparison["baseline"])
        method = str(comparison["method"])
        for metric in metrics:
            missing = [
                scenario for scenario in expected_scenarios
                if scenario not in units
                or baseline not in units[scenario] or method not in units[scenario]
                or metric not in units[scenario].get(baseline, {})
                or metric not in units[scenario].get(method, {})
            ]
            if missing:
                raise ValueError(
                    f"GAIN_ANALYSIS_INCOMPLETE_METHOD_SCENARIO_PANEL:{method}:{baseline}:{metric}:"
                    f"{','.join(missing[:8])}"
                )
            scenarios, baseline_values, method_values, differences = _paired_values(units, baseline, method, metric)
            key = f"{method}__vs__{baseline}__{metric}"
            rng = np.random.default_rng(_seed_from(seed, metric, method))
            ci_low, ci_high = _bootstrap_ci(differences, bootstrap_replicates, rng)
            permutation_rng = np.random.default_rng(_seed_from(seed ^ 0x5A17, metric, method))
            p_value = _paired_permutation_p(differences, permutation_replicates, permutation_rng)
            p_values[key] = p_value
            reports.append({
                "comparison_id": key, "baseline": baseline, "method": method, "metric": metric,
                "scenario_count": len(scenarios), "scenario_ids_sha256": hashlib.sha256(_canonical(scenarios)).hexdigest(),
                "baseline_mean": float(np.mean(baseline_values)), "method_mean": float(np.mean(method_values)),
                "paired_difference_mean": float(np.mean(differences)),
                "paired_difference_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
                "paired_difference_ci95": [ci_low, ci_high],
                "paired_permutation_p": p_value,
            })
    adjusted = holm_adjust(p_values)
    for report in reports:
        report["holm_adjusted_p"] = adjusted[report["comparison_id"]]
    return {
        "schema": ANALYSIS_SCHEMA,
        "inference_unit": "independent_scenario_instance",
        "within_scenario_aggregation": "arithmetic_mean_over_repeated_runs",
        "bootstrap_replicates": int(bootstrap_replicates),
        "permutation_replicates": int(permutation_replicates),
        "permutation_method": "exact_sign_flip_if_n_le_18_else_monte_carlo_sign_flip",
        "seed": int(seed), "comparisons": reports,
        "holm_family_size": len(p_values),
    }
