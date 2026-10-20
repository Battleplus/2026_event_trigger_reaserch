from __future__ import annotations

from collections import Counter
from pathlib import Path

from run_capture import periodic_trigger_due, source_random_allow
from run_formal_batch import CONDITIONS, SOURCES, validate_registry


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "formal_batch_registry.json"


def test_frozen_registry_has_balanced_168_scene_grid_and_unique_seed_families():
    registry, rows = validate_registry(REGISTRY)
    assert registry["execution"] == "NOT_RUN"
    assert len(rows) == 168
    assert len({row["seed"] for row in rows}) == 168
    assert len({row["parent_family"] for row in rows}) == 168
    assert len({row["config_path"] for row in rows}) == 168
    for split in ("train", "validation"):
        for condition in CONDITIONS:
            cell = [row for row in rows
                    if row["split"] == split and row["condition"] == condition]
            assert len(cell) == 12
            assert Counter(row["source"] for row in cell) == {source: 4 for source in SOURCES}
            assert all(row["config"]["mode"] == "search" for row in cell)
            assert all(row["config"]["targets_per_zone"] == 3 for row in cell)
            assert all(row["config"]["scenario_id"] == row["scenario_id"] for row in cell)


def test_condition_switches_are_one_at_a_time_or_all_enabled():
    _, rows = validate_registry(REGISTRY)
    for row in rows:
        config = row["config"]
        enabled = {name for name, value in config["mechanisms"].items()
                   if value["enabled"]}
        if row["condition"] == "composite_all_enabled":
            assert enabled == set(config["mechanisms"])
        else:
            assert enabled == {row["condition"]}
    delay = next(row["config"] for row in rows if row["condition"] == "light_delay")
    assert delay["formal"]["baseline_observation_delay_s"] == 1.0
    assert delay["mechanisms"]["light_delay"] == {
        "enabled": True, "platform_id": "uav-0-1", "additional_delay_s": 1.0,
    }
    activation = next(row["config"] for row in rows if row["condition"] == "task_activation")
    schedule = activation["evaluator_schedule"]
    assert schedule["activation_task_ids"] == [f"z{z}-cell{c}" for z in range(3) for c in range(3)]
    assert schedule["activation_times_s"] == [600.0] * 9
    assert "truth" not in schedule and "reward" not in schedule


def test_periodic_source_is_causal_and_does_not_create_eventless_sample_events():
    assert not periodic_trigger_due(0)
    assert periodic_trigger_due(240)
    assert periodic_trigger_due(480)
    assert not periodic_trigger_due(481)
    allowed_a, draw_a = source_random_allow(1234, 720)
    allowed_b, draw_b = source_random_allow(1234, 720)
    assert (allowed_a, draw_a) == (allowed_b, draw_b)
    assert source_random_allow(1234, 721)[1] != draw_a
