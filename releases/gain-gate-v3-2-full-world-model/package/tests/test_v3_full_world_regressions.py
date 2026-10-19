from __future__ import annotations

import copy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from civilian import counterfactual_pairs as pairs
from civilian import gain_scenarios
from civilian import v3_runner
from civilian.event_environment import EventReliefEnv
from civilian.v3_scenarios import build_v3_gain_scenario, sample_v3_future_suffix


def test_v3_sampled_suffix_and_runner_continuation_produce_an_executable_pair():
    scenario = build_v3_gain_scenario(
        split="pilot", condition="none_or_light", seed_variant=8201, instance_index=0,
    )
    env = EventReliefEnv(scenario)
    env.reset()
    snapshot = pairs.capture_snapshot(
        env, state_id="v3-pair-regression", scenario_id=scenario["parent_id"],
        parent_family=scenario["parent_family"], split="pilot", event_type="ordinary_candidate",
        forced=False, replanner_version="Public-C1-Catalog-Replanner-v3",
    )
    sampled = sample_v3_future_suffix(
        scenario, after_time=int(env.time), horizon_minutes=30, suffix_seed=8201,
    )
    suffix = pairs.prepare_suffix_snapshot(
        snapshot, future_events=sampled, suffix_seed=8201,
    )
    suffix_payload = pairs._snapshot_payload(suffix, pairs.SUFFIX_SCHEMA)
    suffix_env_state = pairs._decode(suffix_payload["state"])
    assert suffix_env_state["scenario"]["scenario_content_sha256"] == gain_scenarios.scenario_content_sha256(suffix_env_state["scenario"])

    def continuation(_branch_env):
        return v3_runner._paired_continuation_envelope(
            pairs.NOOP_ACTION, planner_invoked=False, planner_cpu_seconds=0.0,
            policy_version="shared-safety-trigger-v1",
        )

    label = pairs.collect_paired_label(
        env, state_id="v3-pair-regression", scenario_id=scenario["parent_id"],
        parent_family=scenario["parent_family"], split="pilot", event_type="ordinary_candidate",
        forced=False, replanner_version="Public-C1-Catalog-Replanner-v3",
        replan_action=copy.deepcopy(pairs.NOOP_ACTION), suffix_rows=[{
            "suffix_seed": 8201, "future_events": sampled,
        }], continuation_policy=continuation, replanner_cpu_seconds=0.0,
        horizon_minutes=2,
    )
    assert label["valid_mask"] is True
    assert label["unknown_reason"] is None
    assert label["gain_mean"] is not None
    assert all(pair["valid_mask"] is True for pair in label["pairs"])
