import torch

from civilian import v4_policy


def _candidates():
    return [
        {"candidate_key": "NOOP", "candidate_scope": "NOOP", "legal": True,
         "full_environment_action": {"scope": "NOOP", "noop": True}},
        {"candidate_key": "LOCAL", "candidate_scope": "LOCAL_REPAIR", "legal": True,
         "full_environment_action": {"scope": "LOCAL_REPAIR"}},
    ]


def _predictions(progress=0.1):
    return {
        "progress": torch.tensor([progress, progress + 0.1]),
        "delay": torch.tensor([0.2, 0.3]),
        "energy": torch.tensor([0.4, 0.5]),
        "failure": torch.tensor([0.01, 0.02]),
        "communication_risk": torch.tensor([0.03, 0.04]),
        "next_minute": torch.zeros(2, 1, 4),
    }


def _mock_dynamic_actions(monkeypatch):
    def encode(observation, action):
        row = torch.zeros(1, v4_policy.ACTION_DIM)
        row[0, 0] = float(str(action.get("scope", "")).upper() == "LOCAL_REPAIR")
        return ["mock-node"], row
    monkeypatch.setattr(v4_policy, "encode_v4_action_features", encode)


def test_smdp_gae_discounts_minute_rewards_and_elapsed_time():
    rows = [
        {"minute_rewards": [1.0, 2.0], "dt": 2, "old_value": 0.5, "terminated": False},
        {"reward": 3.0, "dt": 1, "old_value": 1.0, "terminated": True},
    ]
    v4_policy.attach_smdp_gae(rows, gamma=0.5, gae_lambda=0.5)
    # First interval reward is 1 + .5*2; its bootstrap is .5^2 * 1.
    expected_delta = 2.0 + 0.25 - 0.5
    expected_second = 3.0 - 1.0
    expected_advantage = expected_delta + (0.5 * 0.5) ** 2 * expected_second
    assert rows[0]["dt"] == 2
    assert rows[0]["discounted_reward"] == 2.0
    assert rows[1]["advantage"] == expected_second
    assert abs(rows[0]["advantage"] - expected_advantage) < 1e-7
    assert rows[0]["return"] == rows[0]["advantage"] + rows[0]["old_value"]


def test_smdp_terminal_and_truncation_bootstrap_rules():
    rows = [
        {"reward": 2.0, "dt": 3, "old_value": 0.5, "terminated": False,
         "truncated": True, "bootstrap_value": 4.0},
        {"reward": 9.0, "dt": 1, "old_value": 7.0, "terminated": True},
    ]
    v4_policy.attach_smdp_gae(rows, gamma=0.5, gae_lambda=0.9)
    assert rows[0]["next_value"] == 4.0
    assert rows[0]["advantage"] == 2.0 + 0.5**3 * 4.0 - 0.5
    assert rows[1]["advantage"] == 2.0


def test_actor_consumes_candidate_aligned_wm_and_critic_stays_public(monkeypatch):
    monkeypatch.setattr(v4_policy, "_node_features", lambda _: ([[0.0] * v4_policy.NODE_FEATURE_DIM], [0], [(0, 0, 0)]))
    _mock_dynamic_actions(monkeypatch)
    observation = {"public": True}
    candidates = _candidates()
    policy = v4_policy.build_v4_policy(7, use_wm_actor=True)
    first = policy.actor_logits(observation, candidates, _predictions(0.1))
    second = policy.actor_logits(observation, candidates, _predictions(0.9))
    assert not torch.equal(first[1], second[1])
    value_before = policy.value(observation)
    value_after = policy.value(observation)
    assert torch.equal(value_before, value_after)


def test_non_wm_variants_keep_wm_slots_but_reject_wm_inputs(monkeypatch):
    monkeypatch.setattr(v4_policy, "_node_features", lambda _: ([[0.0] * v4_policy.NODE_FEATURE_DIM], [0], [(0, 0, 0)]))
    _mock_dynamic_actions(monkeypatch)
    policy = v4_policy.build_v4_policy(9, use_wm_actor=False)
    assert policy.actor_fusion[0].in_features == v4_policy.POLICY_HIDDEN_DIM + v4_policy.WM_SLOT_DIM
    policy.actor_logits({"public": True}, _candidates(), None)
    try:
        policy.actor_logits({"public": True}, _candidates(), _predictions())
    except ValueError as error:
        assert "MUST_NOT_RECEIVE_WM" in str(error)
    else:
        raise AssertionError("non-WM actor accepted world-model inputs")


def test_dispatch_candidate_logits_bind_recipient_to_public_platform_features():
    observation = {
        "map": {"rows": 100, "cols": 100, "grid_cells": 10000,
                "grid_contract": "relief-grid-100x100-v1"},
        "time": 0, "horizon": 100,
        "platforms": [
            {"id": "p_b", "position": [10, 15], "energy": 20, "capacity_kg": 20,
             "speed_km_min": 1, "energy_per_km": 1},
            {"id": "p_c", "position": [80, 75], "energy": 95, "capacity_kg": 20,
             "speed_km_min": 1, "energy_per_km": 1},
        ],
        "tasks": [{"id": "t1", "position": [50, 50], "quantity": 1,
                   "deadline": 80, "kind": "delivery"}],
        "environment": [], "communication": [], "queues": {},
        "predecessors": {}, "cooperation": {}, "task_progress": {},
    }
    candidates = [
        {"candidate_key": "dispatch-to-b", "candidate_scope": "LOCAL_REPAIR", "legal": True,
         "full_environment_action": {"scope": "LOCAL_REPAIR", "dispatches": [
             {"platform_id": "p_b", "task_id": "t1"}]}},
        {"candidate_key": "dispatch-to-c", "candidate_scope": "LOCAL_REPAIR", "legal": True,
         "full_environment_action": {"scope": "LOCAL_REPAIR", "dispatches": [
             {"platform_id": "p_c", "task_id": "t1"}]}},
    ]
    policy = v4_policy.build_v4_policy(31)
    _, logits = policy.actor_logits(observation, candidates)
    assert not torch.equal(logits[0], logits[1])


def test_selected_scope_log_probability_replays_with_full_candidate_mask(monkeypatch):
    monkeypatch.setattr(v4_policy, "_node_features", lambda _: ([[0.0] * v4_policy.NODE_FEATURE_DIM], [0], [(0, 0, 0)]))
    _mock_dynamic_actions(monkeypatch)
    candidates = [
        {"candidate_key": "NOOP", "candidate_scope": "NOOP", "legal": True,
         "full_environment_action": {"scope": "NOOP", "noop": True}},
        {"candidate_key": "LOCAL", "candidate_scope": "LOCAL_REPAIR", "legal": True,
         "full_environment_action": {"scope": "LOCAL_REPAIR"}},
        {"candidate_key": "GLOBAL", "candidate_scope": "GLOBAL_REPLAN", "legal": True,
         "full_environment_action": {"scope": "GLOBAL_REPLAN"}},
    ]
    policy = v4_policy.build_v4_policy(101).eval()
    observation = {"public": True}
    with torch.no_grad():
        selected = policy.act(observation, candidates, [False, True, False], deterministic=True)
        replay = policy.evaluate_action(observation, candidates,
            scope_index=selected["scope_index"], candidate_index=selected["candidate_index"],
            high_level_mask=selected["high_level_mask"], candidate_mask=selected["candidate_mask"])
    assert selected["candidate_mask"] == [True, True, True]
    assert selected["scope"] == "LOCAL_REPAIR"
    assert torch.allclose(replay["log_probability"], torch.tensor(selected["log_probability"]), atol=1e-7)


def test_no_uncertainty_variant_keeps_full_world_model_actor_inputs():
    assert v4_policy.variant_policy_options("Event-G-WM-NoU") == {
        "use_wm_actor": True, "no_utility": False,
    }


def test_ppo_defaults_match_v4_protocol():
    assert v4_policy.smdp_ppo_defaults() == {
        "seeds": [8201, 8202, 8203], "rounds": 128, "episodes_per_round": 4,
        "learning_rate": 0.0003, "epochs": 4, "minibatch_size": 64,
        "clip_ratio": 0.2, "entropy_coefficient": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
    }


def test_variant_semantics_and_checkpoint_optimizer_restore(tmp_path):
    variant = "Event-G-WM-NoU"
    assert v4_policy.variant_policy_options(variant) == {"use_wm_actor": True, "no_utility": False}
    policy = v4_policy.build_v4_policy(13, **v4_policy.variant_policy_options(variant))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    path = tmp_path / "resume.pt"
    v4_policy.save_v4_checkpoint(path, policy, optimizer, variant=variant, seed=13,
                                 config={"learning_rate": 0.0003}, source_identity="source-abc")
    restored, restored_optimizer, payload = v4_policy.load_v4_checkpoint(path, expected_source_identity="source-abc")
    assert restored.use_wm_actor and not restored.no_utility
    assert restored_optimizer.state_dict()["param_groups"][0]["lr"] == 0.0003
    assert payload["source_identity"] == "source-abc"


def test_update_keeps_singleton_transitions_for_critic_and_uses_partial_batch(monkeypatch):
    monkeypatch.setattr(v4_policy, "_node_features", lambda _: ([[0.0] * v4_policy.NODE_FEATURE_DIM], [0], [(0, 0, 0)]))
    _mock_dynamic_actions(monkeypatch)
    candidates = [_candidates()[0]]
    policy = v4_policy.build_v4_policy(11)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    records = [{
        "observation": {"public": True}, "candidates": candidates, "scope_index": 0,
        "candidate_index": 0, "high_level_mask": [True, False, False],
        "candidate_mask": [True], "wm_predictions": None,
        "old_log_probability": 0.0, "advantage": 1.0, "return": 2.0,
    } for _ in range(3)]
    logs = v4_policy.update_v4_policy(policy, optimizer, records, seed=11, epochs=1, minibatch_size=2)
    assert [row["sample_count"] for row in logs] == [2, 1]
    assert all(row["actor_sample_count"] == 0 for row in logs)
    assert all(row["value_loss"] > 0 for row in logs)
