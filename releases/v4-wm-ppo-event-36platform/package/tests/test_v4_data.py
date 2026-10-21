from civilian import v4_data


def test_frozen_registry_has_preregistered_counts_and_unique_seeds():
    config = {
        "scenario": {}, "seeds": {"registry": {"test_base": 840000}},
    }
    registry = v4_data.frozen_registry(config)
    from civilian.v3_shards import expected_registry
    legacy = expected_registry(task_mode="search_rescue", targets_per_region=3)
    assert {split: len(rows) for split, rows in registry.items()} == {
        "train": 84, "validation": 84, "test": 140,
    }
    for split, rows in registry.items():
        seeds = [row["spec"]["seed_variant"] for row in rows]
        assert len(seeds) == len(set(seeds))
        assert all(len(row["scenario_sha256"]) == 64 for row in rows)
        assert all(row["spec"]["task_mode"] == "search_rescue" and row["spec"]["targets_per_region"] == 3 for row in rows)
    assert [row["spec"]["seed_variant"] for row in registry["train"][:2]] == [8201, 8202]
    for split in ("train", "validation"):
        assert [row["spec"] for row in registry[split]] == [row for row in legacy if row["split"] == split]
    assert all(sum(row["spec"]["condition"] == condition for row in registry["test"]) == 20
               for condition in v4_data.CONDITIONS)


def test_public_history_strips_private_provenance_and_rejects_future_fields():
    history = [{"time": i, "platforms": [], "provenance": {"snapshot_hash": "x"}} for i in range(5)]
    selected = v4_data.public_history(history)
    assert [row["time"] for row in selected] == [1, 2, 3, 4]
    assert all("provenance" not in row for row in selected)
    try:
        v4_data.public_history([{"time": 1, "future_events": []}])
    except ValueError as error:
        assert "HIDDEN_FIELD" in str(error)
    else:
        raise AssertionError("future event data entered a model history")


def test_branch_minutes_are_seeded_unique_and_capped_at_eight():
    first = v4_data._sample_branch_minutes(842701, 90)
    assert first == v4_data._sample_branch_minutes(842701, 90)
    assert len(first) == 8
    assert len(set(first)) == len(first)
    assert first == tuple(sorted(first))
    assert all(1 <= minute <= 80 for minute in first)
    assert v4_data._sample_branch_minutes(1, 13) == (1, 2, 3)
    assert v4_data._sample_branch_minutes(1, 10) == ()


def test_short_branch_masks_ten_minute_targets():
    history = [{"time": 4, "platforms": [{"id": "p1"}], "tasks": [{"id": "t1", "deadline": 8}]}]
    steps = [{"public_observation": {"platforms": [{"id": "p1", "position": [1, 2], "energy": 10}],
                                     "tasks": [{"id": "t1", "position": [3, 4], "progress": 0}]},
              "next_public_observation": {"platforms": [{"id": "p1", "position": [2, 2], "energy": 9}],
                                          "tasks": [{"id": "t1", "position": [3, 4], "progress": 0.2}]},
              "raw_events": []}]
    truth = {"time": 4, "platforms": [{"id": "p1", "energy_used": 0}],
             "tasks": [{"id": "t1", "deadline": 8, "status": "active", "progress": 0}]}
    result = v4_data._targets(history, {}, steps, truth,
                              {"time": 5, "platforms": [{"id": "p1", "energy_used": 1}],
                               "tasks": [{"id": "t1", "deadline": 8, "status": "active", "progress": 0.2}]})
    assert result["next_minute"]["valid"][0][0]
    assert result["horizon"] == {"steps": 1, "valid": False}
    assert not any(result[name]["valid"] for name in ("progress", "delay", "energy", "failure", "communication_risk"))
