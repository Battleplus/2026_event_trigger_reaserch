from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import gain_diagnostics

OUTPUT_DIR = Path(__file__).resolve().parent
REPORT_PATH = OUTPUT_DIR / "LEARNABILITY_DIAGNOSTICS.json"


def _run(rows: list[dict], **kwargs) -> dict:
    if REPORT_PATH.exists():
        raise AssertionError("test output report already exists")
    try:
        return gain_diagnostics.run_learnability_diagnostics(rows, OUTPUT_DIR, **kwargs)
    finally:
        REPORT_PATH.unlink(missing_ok=True)


def _training_rows(scenario_count: int = 6) -> list[dict]:
    rows = []
    for group in range(scenario_count):
        scenario_id = f"train-{group:02d}"
        for window in range(2):
            gain = float(group + window - 3)
            rows.append({
                "state_id": f"{scenario_id}-state-{window}",
                "scenario_id": scenario_id,
                "split": "train",
                "forced": False,
                "valid_mask": True,
                "features": {
                    "affected_task_count": float(group + window),
                    "time_fraction": float(window) / 2.0,
                },
                "gain_mean": gain,
                "oracle_gain": gain,
                "suffix_gains": [gain - 0.1, gain + 0.1],
                "paired_variance": 0.02,
            })
    return rows


def test_grouped_cv_keeps_scenarios_whole_and_fits_normalization_on_train_only() -> None:
    rows = _training_rows()
    baseline = _run(rows, bootstrap_replicates=200, seed=17)
    extreme_test_row = {
        "state_id": "test-extreme-state", "scenario_id": "test-extreme", "split": "test",
        "forced": False, "valid_mask": True,
        "features": {"affected_task_count": 1e9, "time_fraction": 1e9},
        "gain_mean": 1e9, "oracle_gain": 1e9,
    }
    with_test = _run(rows + [extreme_test_row], bootstrap_replicates=200, seed=17)

    assert baseline["counts"]["ordinary_valid_state_count"] == 12
    assert with_test["ignored_non_training_row_count"] == 1
    assert with_test["grouped_cross_validation"]["scenario_mae_table"] == baseline["grouped_cross_validation"]["scenario_mae_table"]
    for fold in with_test["grouped_cross_validation"]["folds"]:
        train_groups = set(fold["train_scenario_ids"])
        validation_groups = set(fold["validation_scenario_ids"])
        assert train_groups.isdisjoint(validation_groups)
        assert fold["normalizer_fit_scenario_ids"] == sorted(train_groups)
        expected_mean = sum(
            row["features"]["affected_task_count"]
            for row in rows
            if row["scenario_id"] in train_groups
        ) / sum(1 for row in rows if row["scenario_id"] in train_groups)
        assert fold["normalizer_feature_mean"]["affected_task_count"] == pytest.approx(expected_mean)
        assert "test-extreme" not in fold["normalizer_fit_scenario_ids"]


def test_duplicate_states_and_suffixes_do_not_inflate_state_counts() -> None:
    rows = _training_rows()
    duplicate = dict(rows[0])
    duplicate["feature_wall_seconds"] = 999.0
    duplicate["replanner_cpu_seconds"] = 1e9
    try:
        report = gain_diagnostics.run_learnability_diagnostics(
            rows + [duplicate], OUTPUT_DIR, bootstrap_replicates=100, seed=23,
        )

        assert report["counts"]["ordinary_valid_state_count"] == len(rows)
        assert report["counts"]["suffix_measurement_count_descriptive_only"] == 2 * len(rows)
        assert report["paired_label_noise"]["state_count_with_variance"] == len(rows)
        assert report["paired_label_noise"]["suffixes_are_repeated_measurements_not_independent_states"] is True
        assert report["duplicate_training_row_count_collapsed"] == 1
        assert report["timing"]["status"] == "NOT_MEASURED"
        assert report["timing"]["median_feature_to_planner_ratio"] is None
        assert REPORT_PATH.is_file()
    finally:
        REPORT_PATH.unlink(missing_ok=True)


def test_forced_and_invalid_rows_are_excluded_from_label_counts() -> None:
    rows = _training_rows()
    rows.extend([
        {
            "state_id": "train-forced", "scenario_id": "train-extra", "split": "train",
            "forced": True, "valid_mask": True, "gain_mean": 99.0,
        },
        {
            "state_id": "train-invalid", "scenario_id": "train-extra", "split": "train",
            "forced": False, "valid_mask": False, "unknown_reason": "branch_execution_failed", "gain_mean": None,
        },
        {
            "state_id": "train-forced-invalid", "scenario_id": "train-extra", "split": "train",
            "forced": True, "valid_mask": False, "unknown_reason": "forced_safety_state", "gain_mean": None,
        },
    ])
    report = _run(rows, bootstrap_replicates=100)
    assert report["counts"]["forced_valid_state_count"] == 1
    assert report["counts"]["forced_training_state_count"] == 2
    assert report["counts"]["forced_training_state_fraction"] == pytest.approx(2 / len(rows))
    assert report["counts"]["invalid_training_state_count"] == 2
    assert report["counts"]["invalid_unknown_reason_counts"] == {
        "branch_execution_failed": 1,
        "forced_safety_state": 1,
    }
    assert report["counts"]["ordinary_valid_state_count"] == len(rows) - 3
    assert report["counts"]["positive_ordinary_state_count"] + report["counts"]["nonpositive_ordinary_state_count"] == len(rows) - 3


def test_timing_ratio_requires_both_explicit_measurement_arrays() -> None:
    report = _run(
        _training_rows(),
        bootstrap_replicates=100,
        planner_timing={
            "feature_gate_seconds": [0.001, 0.002, 0.001],
            "planner_seconds": [0.01, 0.02, 0.01],
        },
    )
    assert report["timing"]["status"] == "WITHIN_20_PERCENT"
    assert report["timing"]["median_feature_to_planner_ratio"] == pytest.approx(0.1)
    assert report["timing"]["feature_gate_seconds"]["count"] == 3
    assert report["timing"]["planner_seconds"]["count"] == 3

    missing = _run(
        _training_rows(),
        bootstrap_replicates=100,
        planner_timing={"feature_gate_seconds": [0.001], "planner_seconds": None},
    )
    assert missing["timing"]["status"] == "NOT_MEASURED"
    assert missing["timing"]["median_feature_to_planner_ratio"] is None


def test_suffix_mean_mismatch_is_rejected() -> None:
    rows = _training_rows()
    rows[0]["suffix_gains"] = [0.0, 1.0]
    rows[0]["paired_variance"] = 0.5
    with pytest.raises(ValueError, match="SUFFIX_MEAN_MISMATCH"):
        _run(rows, bootstrap_replicates=20)
