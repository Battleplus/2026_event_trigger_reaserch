from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import counterfactual_pairs
import cost_accounting
import event_environment
import gain_oracle
import gain_scenarios
import gain_variance


REPLANNER_VERSION = "Public-C1-Catalog-Replanner-v1"


def test_training_state_selection_is_deterministic_and_gain_blind() -> None:
    rows = [
        {
            "state_id": f"state-{index}", "scenario_id": f"scenario-{index // 2}",
            "snapshot_sha256": f"{index:064x}", "split": "train",
            "forced": False, "valid_mask": True, "gain_mean": float(index),
        }
        for index in range(6)
    ]
    selected, receipt = gain_variance.select_training_records(rows, sample_size=3, selection_seed=37)
    changed_gain_rows = [dict(row, gain_mean=-999999.0) for row in rows]
    changed, changed_receipt = gain_variance.select_training_records(
        changed_gain_rows, sample_size=3, selection_seed=37,
    )

    assert [row["state_id"] for row in selected] == [row["state_id"] for row in changed]
    assert receipt["selected_rank_hashes"] == changed_receipt["selected_rank_hashes"]
    assert receipt["gain_values_used_for_selection"] is False
    assert receipt["selected_state_ids"] == [row["state_id"] for row in selected]


def test_k8_diagnostic_keeps_k2_suffixes_adds_frozen_samples_and_saves_oracle() -> None:
    test_root = ROOT / "tests" / f".gain-variance-test-{uuid4().hex}"
    test_root.mkdir()
    try:
        _run_k8_diagnostic_test(test_root)
    finally:
        for child in sorted(test_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        test_root.rmdir()


def _run_k8_diagnostic_test(test_root: Path) -> None:
    scenario = next(gain_scenarios.iter_gain_scenarios(
        {"train": 1, "validation": 0, "test": 0},
        seed=8201,
        conditions=("none_or_light",),
    ))
    env = event_environment.EventReliefEnv(scenario)
    env.reset()
    state_id = "gain-variance-integration-state"
    snapshot_json = counterfactual_pairs.capture_snapshot(
        env,
        state_id=state_id,
        scenario_id=scenario["parent_id"],
        parent_family=scenario["parent_family"],
        split="train",
        event_type="test_state",
        forced=False,
        replanner_version=REPLANNER_VERSION,
    )
    snapshot = counterfactual_pairs._snapshot_payload(
        snapshot_json, counterfactual_pairs.SNAPSHOT_SCHEMA,
    )
    original_seeds = [
        gain_variance.suffix_seed(scenario["parent_id"], state_id, 0, index)
        for index in range(gain_variance.BASE_SUFFIX_COUNT)
    ]
    suffix_jsons = []
    for seed in original_seeds:
        events = gain_scenarios.sample_future_event_suffix(
            copy.deepcopy(scenario), after_time=0, horizon_minutes=2, suffix_seed=seed,
        )
        suffix_jsons.append(counterfactual_pairs.prepare_suffix_snapshot(
            snapshot_json, future_events=events, suffix_seed=seed,
            env_factory=event_environment.EventReliefEnv,
        ))

    continuation_calls = 0

    def shared_safety_noop(_env):
        nonlocal continuation_calls
        continuation_calls += 1
        return copy.deepcopy(counterfactual_pairs.NOOP_ACTION)

    label = counterfactual_pairs.run_paired_counterfactual(
        snapshot_json,
        replan_action=copy.deepcopy(counterfactual_pairs.NOOP_ACTION),
        suffixes=suffix_jsons,
        continuation_policy=shared_safety_noop,
        replanner_version=REPLANNER_VERSION,
        replanner_cpu_seconds=0.002,
        cost_config=cost_accounting.DEFAULT_CONFIG,
        horizon_minutes=2,
        env_factory=event_environment.EventReliefEnv,
    )
    assert gain_oracle.validate_gain_label(label)["valid"] is True

    row = {
        "schema": "gain-gate-training-row/1", "state_id": state_id,
        "scenario_id": scenario["parent_id"], "parent_id": scenario["parent_id"],
        "split": "train", "condition": scenario["condition"], "forced": False,
        "valid_mask": True, "snapshot_sha256": snapshot["snapshot_sha256"],
        "label_sha256": label["label_sha256"], "suffix_seed": original_seeds,
    }
    bundle = {
        "row": row, "scenario": scenario, "snapshot_json": snapshot_json,
        "label": label, "suffix_jsons": suffix_jsons,
    }
    resource_checks: list[bool] = []
    output_dir = test_root / "variance-run"
    report = gain_variance.run_gain_variance_diagnostic(
        [row], output_dir,
        modules={
            "pairs": counterfactual_pairs,
            "oracle": gain_oracle,
            "scenarios": gain_scenarios,
            "env_class": event_environment.EventReliefEnv,
        },
        continuation_policy=shared_safety_noop,
        record_loader=lambda selected: bundle,
        sample_size=1,
        additional_suffixes=1,
        selection_seed=91,
        resource_check=lambda: resource_checks.append(True),
    )

    assert continuation_calls == 2 * (2 + 3)
    assert len(resource_checks) == 2
    assert report["status"] == "COMPLETE"
    assert report["selection"]["selected_state_ids"] == [state_id]
    assert report["protocol"]["base_suffix_count"] == 2
    assert report["protocol"]["additional_suffix_count"] == 1
    assert report["protocol"]["total_suffix_count"] == 3
    state = report["states"][0]
    assert state["k2"]["valid_mask"] is True
    assert state["k8"]["valid_mask"] is True
    assert state["k2"]["pair_count"] == 2
    assert state["k8"]["pair_count"] == 3
    assert len(state["k2_suffix_replay_comparison"]) == 2
    assert all(
        item["keep_event_tape_sha256_matches"] and item["replan_event_tape_sha256_matches"]
        for item in state["k2_suffix_replay_comparison"]
    )
    assert state["additional_suffix_seeds"] == [gain_variance.suffix_seed(
        scenario["parent_id"], state_id, 0, 2,
    )]

    artifact_path = output_dir / state["raw_artifact_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["k8_label"]["suffix_count"] == 3
    assert len(artifact["additional_pairs"]) == 1
    assert artifact["k8_oracle"]["valid"] is True
    assert artifact["k8_oracle"]["label_sha256"] == artifact["k8_label"]["label_sha256"]
    assert len(artifact["k2_suffix_replay_comparison"]) == 2
    assert (output_dir / "GAIN_VARIANCE_DIAGNOSTIC.json").is_file()


def test_sample_rejects_too_few_eligible_training_states() -> None:
    with pytest.raises(ValueError, match="GAIN_VARIANCE_INSUFFICIENT_ELIGIBLE_TRAIN_STATES"):
        gain_variance.select_training_records(
            [{"state_id": "forced", "scenario_id": "train-1", "split": "train", "forced": True,
              "valid_mask": True}],
            sample_size=1,
        )
