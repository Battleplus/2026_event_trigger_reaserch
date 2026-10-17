"""Training-only diagnostics for paired gain labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gain_models


DIAGNOSTICS_SCHEMA = "gain-learnability-diagnostics/1"
MODEL_NAMES = ("constant", "ridge", "gradient_boosting_depth3")
TIMING_KEYS = ("feature_gate_seconds", "planner_seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GAIN_DIAGNOSTICS_FINITE_NUMBER_REQUIRED:{field}")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"GAIN_DIAGNOSTICS_FINITE_NUMBER_REQUIRED:{field}")
    return number


def _state_id(row: Mapping[str, Any]) -> str:
    value = row.get("state_id", row.get("sample_id"))
    if value is None or not str(value):
        raise ValueError("GAIN_DIAGNOSTICS_STATE_ID_REQUIRED")
    return str(value)


def _features(row: Mapping[str, Any]) -> dict[str, float]:
    return gain_models._row_features(row)


def _validated_gain(row: Mapping[str, Any]) -> float:
    """Read only the independently validated label target, never a model output."""
    oracle_gain = row.get("oracle_gain")
    receipt = row.get("oracle_receipt", row.get("oracle"))
    receipt_gain = None
    if receipt is not None:
        if not isinstance(receipt, Mapping) or receipt.get("valid") is not True or receipt.get("label_valid_mask") is not True:
            raise ValueError("GAIN_DIAGNOSTICS_ORACLE_RECEIPT_INVALID")
        receipt_gain = _finite_number(receipt.get("recomputed_gain_mean"), "oracle_receipt.recomputed_gain_mean")

    if oracle_gain is not None:
        gain = _finite_number(oracle_gain, "oracle_gain")
    elif receipt_gain is not None:
        gain = receipt_gain
    else:
        # gain_mean is accepted for compact runner rows, whose contract sets it
        # only after independent oracle validation.
        gain = _finite_number(row.get("gain_mean"), "gain_mean")
    if receipt_gain is not None and not math.isclose(gain, receipt_gain, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("GAIN_DIAGNOSTICS_ORACLE_GAIN_MISMATCH")
    if row.get("gain_mean") is not None:
        recorded = _finite_number(row["gain_mean"], "gain_mean")
        if not math.isclose(gain, recorded, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_DIAGNOSTICS_RECORDED_GAIN_MISMATCH")
    return gain


def _suffix_gains(row: Mapping[str, Any]) -> list[float] | None:
    raw = row.get("suffix_gains", row.get("per_suffix_gains"))
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("GAIN_DIAGNOSTICS_SUFFIX_GAINS_INVALID")
    values = []
    for index, item in enumerate(raw):
        value = item.get("gain") if isinstance(item, Mapping) else item
        values.append(_finite_number(value, f"suffix_gains[{index}]"))
    if not values:
        return None
    return values


def _paired_variance(row: Mapping[str, Any], gains: Sequence[float] | None, oracle_gain: float) -> float | None:
    suffix_variance = statistics.variance(gains) if gains is not None and len(gains) > 1 else None
    reported = row.get("paired_variance")
    reported_variance = None if reported is None else _finite_number(reported, "paired_variance", nonnegative=True)
    if suffix_variance is not None and reported_variance is not None and not math.isclose(
        suffix_variance, reported_variance, rel_tol=1e-8, abs_tol=1e-10,
    ):
        raise ValueError("GAIN_DIAGNOSTICS_PAIRED_VARIANCE_MISMATCH")
    if gains is not None and not math.isclose(float(np.mean(gains)), oracle_gain, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("GAIN_DIAGNOSTICS_SUFFIX_MEAN_MISMATCH")
    return suffix_variance if suffix_variance is not None else reported_variance


def _state_record(row: Mapping[str, Any]) -> dict[str, Any]:
    valid = row.get("valid_mask", row.get("valid", True)) is True
    forced = row.get("forced", False) is True
    unknown_reason = row.get("unknown_reason")
    if unknown_reason is not None and (not isinstance(unknown_reason, str) or not unknown_reason.strip()):
        raise ValueError("GAIN_DIAGNOSTICS_UNKNOWN_REASON_INVALID")
    state: dict[str, Any] = {
        "state_id": _state_id(row),
        "scenario_id": gain_models._group_id(row),
        "split": gain_models._row_split(row),
        "valid": valid,
        "forced": forced,
        "unknown_reason": unknown_reason,
        "features": None,
        "gain": None,
        "paired_variance": None,
        "suffix_gains": None,
    }
    if valid and not forced:
        state["features"] = _features(row)
        state["gain"] = _validated_gain(row)
        suffix_gains = _suffix_gains(row)
        state["suffix_gains"] = suffix_gains
        state["paired_variance"] = _paired_variance(row, suffix_gains, state["gain"])
    return state


def _collapse_training_states(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    training_rows = [row for row in rows if gain_models._row_split(row) == "train"]
    grouped: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in training_rows:
        state = _state_record(row)
        previous = grouped.get(state["state_id"])
        if previous is None:
            grouped[state["state_id"]] = state
        elif previous != state:
            raise ValueError(f"GAIN_DIAGNOSTICS_CONFLICTING_DUPLICATE_STATE:{state['state_id']}")
        else:
            duplicates += 1
    states = [grouped[key] for key in sorted(grouped)]
    return states, len(training_rows), duplicates, len(rows) - len(training_rows)


def _grouped_folds(states: Sequence[Mapping[str, Any]], fold_count: int, seed: int) -> list[list[str]]:
    groups = sorted({str(state["scenario_id"]) for state in states})
    if len(groups) < fold_count:
        return []
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest())
    return [ordered[index::fold_count] for index in range(fold_count)]


def _normalizer_for(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [{"features": state["features"]} for state in states]
    return gain_models._fit_normalizer(rows)


def _matrix(states: Sequence[Mapping[str, Any]], normalizer: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    names = list(normalizer["feature_names"])
    feature_rows = []
    targets = []
    for state in states:
        features = state["features"]
        if list(features) != names:
            raise ValueError("GAIN_DIAGNOSTICS_FEATURE_SCHEMA_MISMATCH")
        feature_rows.append([features[name] for name in names])
        targets.append(float(state["gain"]))
    raw = np.asarray(feature_rows, dtype=np.float64)
    return gain_models._transform(raw, normalizer), np.asarray(targets, dtype=np.float64)


def _bootstrap_ci(values: Sequence[float], replicates: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = np.asarray(values, dtype=np.float64)[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _timing_summary(planner_timing: Any) -> dict[str, Any]:
    if planner_timing is None:
        return {
            "status": "NOT_MEASURED", "source": "caller_supplied_warmed_measurements",
            "feature_gate_seconds": None, "planner_seconds": None,
            "median_feature_to_planner_ratio": None, "ratio_limit": 0.2,
        }
    if not isinstance(planner_timing, Mapping):
        raise ValueError("GAIN_DIAGNOSTICS_TIMING_MAPPING_REQUIRED")

    summaries: dict[str, Any] = {}
    samples: dict[str, list[float]] = {}
    for name in TIMING_KEYS:
        raw = planner_timing.get(name)
        if raw is None:
            values: list[float] = []
            missing_count = 0
        else:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"GAIN_DIAGNOSTICS_TIMING_ARRAY_REQUIRED:{name}")
            values = []
            missing_count = 0
            for index, item in enumerate(raw):
                if item is None:
                    missing_count += 1
                else:
                    values.append(_finite_number(item, f"{name}[{index}]", nonnegative=True))
        samples[name] = values
        summaries[name] = None if not values else {
            "count": len(values), "missing_count": missing_count,
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
    if not samples["feature_gate_seconds"] or not samples["planner_seconds"]:
        return {
            "status": "NOT_MEASURED", "source": "caller_supplied_warmed_measurements",
            **summaries, "median_feature_to_planner_ratio": None, "ratio_limit": 0.2,
        }
    planner_median = float(np.median(samples["planner_seconds"]))
    feature_median = float(np.median(samples["feature_gate_seconds"]))
    if planner_median <= 0.0:
        return {
            "status": "NOT_MEASURED", "source": "caller_supplied_warmed_measurements",
            **summaries, "median_feature_to_planner_ratio": None, "ratio_limit": 0.2,
            "unknown_reason": "planner_median_is_zero",
        }
    ratio = feature_median / planner_median
    return {
        "status": "WITHIN_20_PERCENT" if ratio <= 0.2 else "OVER_20_PERCENT",
        "source": "caller_supplied_warmed_measurements",
        **summaries, "median_feature_to_planner_ratio": ratio, "ratio_limit": 0.2,
    }


def _learnability_cv(
    states: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, seed: int,
    resource_check: Any = None,
) -> dict[str, Any] | None:
    folds = _grouped_folds(states, 3, seed)
    if not folds:
        return None
    scenario_ids = sorted({str(state["scenario_id"]) for state in states})
    feature_names = list(states[0]["features"])
    if any(list(state["features"]) != feature_names for state in states):
        raise ValueError("GAIN_DIAGNOSTICS_FEATURE_SCHEMA_MISMATCH")
    predictions: dict[str, dict[str, float]] = {name: {} for name in MODEL_NAMES}
    fold_reports = []
    state_by_group = defaultdict(list)
    for state in states:
        state_by_group[str(state["scenario_id"])].append(state)

    for fold_index, validation_groups in enumerate(folds):
        validation_group_set = set(validation_groups)
        training_groups = set(scenario_ids) - validation_group_set
        train_states = [state for state in states if state["scenario_id"] in training_groups]
        validation_states = [state for state in states if state["scenario_id"] in validation_group_set]
        if not train_states or not validation_states or training_groups & validation_group_set:
            raise ValueError("GAIN_DIAGNOSTICS_GROUPED_FOLD_INVALID")

        normalizer = _normalizer_for(train_states)
        x_train, y_train = _matrix(train_states, normalizer)
        x_validation, y_validation = _matrix(validation_states, normalizer)
        constant_value = float(np.mean(y_train))
        constant_prediction = np.full(len(validation_states), constant_value, dtype=np.float64)
        ridge = gain_models._fit_ridge(x_train, y_train, alpha=1.0)
        ridge_prediction = gain_models._predict_payload(ridge, x_validation)
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError as exc:
            raise RuntimeError("GAIN_SKLEARN_REQUIRED: install scikit-learn in this project's .deps directory") from exc
        tree = GradientBoostingRegressor(
            loss="huber", n_estimators=100, learning_rate=0.05,
            max_depth=3, random_state=int(seed + fold_index),
        )
        tree.fit(x_train, y_train)
        tree_prediction = np.asarray(tree.predict(x_validation), dtype=np.float64)
        if resource_check is not None:
            resource_check()

        for name, prediction in (
            ("constant", constant_prediction), ("ridge", ridge_prediction),
            ("gradient_boosting_depth3", tree_prediction),
        ):
            if prediction.shape != y_validation.shape or not np.all(np.isfinite(prediction)):
                raise ValueError(f"GAIN_DIAGNOSTICS_INVALID_PREDICTIONS:{name}")
            for state, value in zip(validation_states, prediction, strict=True):
                predictions[name][str(state["state_id"])] = float(value)

        fold_reports.append({
            "fold": fold_index,
            "train_scenario_ids": sorted(training_groups),
            "validation_scenario_ids": sorted(validation_group_set),
            "train_scenario_count": len(training_groups),
            "validation_scenario_count": len(validation_group_set),
            "train_state_count": len(train_states),
            "validation_state_count": len(validation_states),
            "normalizer_fit_scenario_ids": sorted(training_groups),
            "normalizer_fit_row_count": int(normalizer["fit_row_count"]),
            "normalizer_fit_split": normalizer["fit_split"],
            "normalizer_feature_mean": dict(zip(feature_names, map(float, normalizer["mean"]), strict=True)),
            "normalizer_feature_scale": dict(zip(feature_names, map(float, normalizer["scale"]), strict=True)),
        })

    observed = {str(state["state_id"]): float(state["gain"]) for state in states}
    scenario_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: {name: [] for name in MODEL_NAMES})
    for state in states:
        state_id = str(state["state_id"])
        scenario = str(state["scenario_id"])
        target = observed[state_id]
        for name in MODEL_NAMES:
            scenario_errors[scenario][name].append(abs(predictions[name][state_id] - target))

    per_model: dict[str, Any] = {}
    scenario_maes = {
        scenario: {name: float(np.mean(errors[name])) for name in MODEL_NAMES}
        for scenario, errors in sorted(scenario_errors.items())
    }
    for model_index, name in enumerate(MODEL_NAMES):
        model_values = [scenario_maes[scenario][name] for scenario in scenario_ids]
        per_model[name] = {
            "scenario_weighted_mae": float(np.mean(model_values)),
            "state_weighted_mae": float(np.mean([abs(predictions[name][state_id] - observed[state_id]) for state_id in observed])),
        }
        if name == "constant":
            continue
        improvements = [scenario_maes[scenario]["constant"] - scenario_maes[scenario][name] for scenario in scenario_ids]
        per_model[name].update({
            "comparison_baseline": "constant",
            "scenario_level_improvement_mean": float(np.mean(improvements)),
            "scenario_level_improvement_ci95": _bootstrap_ci(
                improvements, bootstrap_replicates, seed + 10_000 + model_index,
            ),
            "per_scenario_mae_differences": [
                {
                    "scenario_id": scenario,
                    "constant_mae": scenario_maes[scenario]["constant"],
                    "model_mae": scenario_maes[scenario][name],
                    "improvement_over_constant": scenario_maes[scenario]["constant"] - scenario_maes[scenario][name],
                }
                for scenario in scenario_ids
            ],
        })
        ci = per_model[name]["scenario_level_improvement_ci95"]
        per_model[name]["grouped_cv_signal"] = "IMPROVES_OVER_CONSTANT" if ci is not None and ci[0] > 0.0 else "NOT_ESTABLISHED"

    return {
        "status": "COMPUTED",
        "inference_unit": "independent_scenario_instance",
        "fold_count": len(folds), "folds": fold_reports,
        "model_configuration": {
            "constant": "training_fold_mean",
            "ridge_alpha": 1.0,
            "gradient_boosting": {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05, "loss": "huber"},
        },
        "models": per_model,
        "bootstrap_replicates": int(bootstrap_replicates),
        "scenario_mae_table": scenario_maes,
    }


def run_learnability_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 8201,
    planner_timing: Mapping[str, Sequence[float | None] | None] | None = None,
    resource_check: Any = None,
) -> dict[str, Any]:
    """Summarize validated training labels and run three-fold scenario CV.

    Only ``split=train`` rows contribute gains, features, variance summaries,
    model fits, or model evaluation. Other splits are excluded from those
    calculations after the group-leakage check. Repeated suffixes contribute
    within-state variance only; they never become additional model rows.
    """
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("GAIN_DIAGNOSTICS_BOOTSTRAP_REPLICATES")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("GAIN_DIAGNOSTICS_SEED_REQUIRED")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("GAIN_DIAGNOSTICS_ROWS_REQUIRED")
    normalized_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("GAIN_DIAGNOSTICS_ROW_OBJECT_REQUIRED")
        normalized_rows.append(row)
    if resource_check is not None:
        resource_check()
    leakage = gain_models.check_group_leakage(normalized_rows) if normalized_rows else {"scenario_group_count": 0, "split_leakage_count": 0}
    states, train_row_count, duplicate_count, ignored_non_train_count = _collapse_training_states(normalized_rows)

    valid_states = [state for state in states if state["valid"]]
    forced_states = [state for state in states if state["forced"]]
    forced_valid_states = [state for state in valid_states if state["forced"]]
    invalid_states = [state for state in states if not state["valid"]]
    invalid_unknown_reasons = Counter(
        str(state["unknown_reason"]) if state["unknown_reason"] is not None else "unspecified"
        for state in invalid_states
    )
    eligible = [state for state in valid_states if not state["forced"]]
    eligible_scenarios = sorted({str(state["scenario_id"]) for state in eligible})
    gains = [float(state["gain"]) for state in eligible]
    positive_count = sum(value > 0.0 for value in gains)
    nonpositive_count = len(gains) - positive_count
    positive_gains = [max(0.0, value) for value in gains]

    noise_variances = [float(state["paired_variance"]) for state in eligible if state["paired_variance"] is not None]
    suffix_count = sum(len(state["suffix_gains"] or []) for state in eligible)
    scenario_headroom = {}
    for state in eligible:
        scenario_headroom.setdefault(str(state["scenario_id"]), []).append(max(0.0, float(state["gain"])))
    scenario_headroom_means = [float(np.mean(values)) for _, values in sorted(scenario_headroom.items())]
    headroom_ci = _bootstrap_ci(scenario_headroom_means, bootstrap_replicates, seed + 200_000) if scenario_headroom_means else None

    diagnostics_started = time.perf_counter()
    cv = None
    if len(eligible_scenarios) >= 3 and eligible:
        cv = _learnability_cv(
            eligible, bootstrap_replicates=bootstrap_replicates, seed=seed,
            resource_check=resource_check,
        )
    elapsed = max(0.0, time.perf_counter() - diagnostics_started)
    if resource_check is not None:
        resource_check()
    timing = _timing_summary(planner_timing)

    report = {
        "schema": DIAGNOSTICS_SCHEMA,
        "status": "DIAGNOSTICS_ONLY",
        "stage_decision": "NOT_MADE_BY_DIAGNOSTICS",
        "data_scope": "training_split_only",
        "input_row_count": len(normalized_rows),
        "training_input_row_count": train_row_count,
        "ignored_non_training_row_count": ignored_non_train_count,
        "duplicate_training_row_count_collapsed": duplicate_count,
        "group_leakage_check": leakage,
        "counts": {
            "unique_training_state_count": len(states),
            "valid_training_state_count": len(valid_states),
            "invalid_training_state_count": len(invalid_states),
            "forced_training_state_count": len(forced_states),
            "forced_training_state_fraction": len(forced_states) / len(states) if states else None,
            "forced_valid_state_count": len(forced_valid_states),
            "invalid_unknown_reason_counts": dict(sorted(invalid_unknown_reasons.items())),
            "invalid_missing_unknown_reason_count": invalid_unknown_reasons.get("unspecified", 0),
            "ordinary_valid_state_count": len(eligible),
            "ordinary_valid_scenario_count": len(eligible_scenarios),
            "positive_ordinary_state_count": positive_count,
            "nonpositive_ordinary_state_count": nonpositive_count,
            "positive_state_requirement_each_side": 30,
            "class_balance_requirement_met": positive_count >= 30 and nonpositive_count >= 30,
            "suffix_measurement_count_descriptive_only": suffix_count,
        },
        "label_distribution": {
            "status": "COMPUTED" if gains else "NO_ELIGIBLE_TRAINING_LABELS",
            "gain_mean_state_weighted": float(np.mean(gains)) if gains else None,
            "gain_median": float(np.median(gains)) if gains else None,
            "positive_fraction": positive_count / len(gains) if gains else None,
            "oracle_positive_action_mean_gain_state_weighted": float(np.mean(positive_gains)) if positive_gains else None,
            "oracle_positive_action_mean_gain_scenario_weighted": float(np.mean(scenario_headroom_means)) if scenario_headroom_means else None,
            "oracle_positive_action_mean_gain_scenario_bootstrap_ci95": headroom_ci,
        },
        "paired_label_noise": {
            "status": "COMPUTED" if noise_variances else "NOT_MEASURED",
            "inference_unit": "unique_state_id",
            "suffixes_are_repeated_measurements_not_independent_states": True,
            "state_count_with_variance": len(noise_variances),
            "state_count_without_variance": len(eligible) - len(noise_variances),
            "suffix_measurement_count_descriptive_only": suffix_count,
            "mean_within_state_paired_variance": float(np.mean(noise_variances)) if noise_variances else None,
            "median_within_state_paired_variance": float(np.median(noise_variances)) if noise_variances else None,
            "root_mean_within_state_paired_variance": math.sqrt(float(np.mean(noise_variances))) if noise_variances else None,
        },
        "grouped_cross_validation": cv or {
            "status": "INSUFFICIENT_SCENARIO_GROUPS",
            "required_fold_count": 3,
            "available_ordinary_scenario_count": len(eligible_scenarios),
            "fold_count": 0,
        },
        "timing": timing,
        "diagnostics_compute_seconds": elapsed,
        "seed": int(seed),
        "bootstrap_replicates": int(bootstrap_replicates),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "LEARNABILITY_DIAGNOSTICS.json"
    if report_path.exists():
        raise FileExistsError(f"GAIN_DIAGNOSTICS_REFUSE_OVERWRITE:{report_path.name}")
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, report_path)
    return report
