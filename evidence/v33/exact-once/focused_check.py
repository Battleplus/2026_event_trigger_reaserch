from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "package"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "civilian"))

from civilian.counterfactual_pairs import (  # noqa: E402
    capture_snapshot,
    prepare_suffix_snapshot,
    run_paired_counterfactual,
)
from civilian.event_environment import EventReliefEnv  # noqa: E402
from civilian.fixed_replanner import REPLANNER_VERSION  # noqa: E402
from civilian.gain_oracle import validate_gain_label  # noqa: E402
from civilian.v33_scenarios import (  # noqa: E402
    HORIZON_MINUTES,
    make_v33_scenario,
    sample_v33_future_suffix,
)

NOOP = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}


def _run_mode(mode: str) -> dict:
    scenario = make_v33_scenario(
        split="pilot", condition="none_or_light", seed_variant=930103,
        instance_index=1, task_mode="search_rescue", targets_per_region=3,
    )
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    rescue = next(
        row for row in scenario["base_scenario"]["tasks"]
        if row["kind"] == "delivery" and row.get("discovered_by")
        and next(task for task in scenario["base_scenario"]["tasks"]
                 if task["id"] == row["discovered_by"])["known_initial"]
    )
    rescue_id = str(rescue["id"])
    parent_id = str(rescue["discovered_by"])
    parent = env._base._tasks[parent_id]
    source = str(parent["assignee"])
    source_platform = env._base._platforms[source]
    partner = next(
        row["id"] for row in env._base._platforms.values()
        if row["type"] == "uav" and row["home_region"] == source_platform["home_region"]
        and row["id"] != source
    )
    env._queues = {platform_id: [] for platform_id in env._queues}
    env._queues[source] = [parent_id]
    env._base._queues = copy.deepcopy(env._queues)
    for mapping in (env._task_specs, env._base._task_spec, env._base._tasks):
        mapping[rescue_id]["report_delay"] = 1
    for mapping in (env._task_specs, env._original_task_specs):
        mapping[parent_id]["service_minutes"] = 2.0
    for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
        mapping[parent_id]["service_minutes"] = 2.0
    position = list(parent["position"])
    region_row, region_col = map(int, source_platform["home_region"][1:])
    bounds = (0, 33, 66, 100)
    partner_position = [
        float(bounds[region_col] if position[0] > (bounds[region_col] + bounds[region_col + 1]) / 2
              else bounds[region_col + 1] - 0.1),
        float(bounds[region_row] if position[1] > (bounds[region_row] + bounds[region_row + 1]) / 2
              else bounds[region_row + 1] - 0.1),
    ]
    for platform_id, target_position in ((source, position), (partner, partner_position)):
        for platform_map in (env._base._platforms, env._base._public_platforms, env._reported_platforms):
            platform_map[platform_id]["position"] = copy.deepcopy(target_position)
    env._candidate_validation_context = {
        "time": env.time, "observation": env.public_observation(), "dispatch_catalog": {},
    }
    operation = {
        "op": mode, "task_id": parent_id, "from_platform": source,
        "to_platform": partner, "share": 0.5,
    }
    _, _, _, _, initial_info = env.step({
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [operation], "dispatches": [],
    })
    child_ids = list(env._split_manifests[parent_id]["child_ids"])
    dispatched_children = {row["task_id"] for row in initial_info["operational_dispatches"]}
    if dispatched_children != set(child_ids):
        raise AssertionError(f"{mode}: child dispatch mismatch {dispatched_children}")
    completed_at_first_step = [
        row for row in env._raw_truth()["subtasks"]
        if row["id"] in child_ids and row["status"] == "completed"
    ]
    if len(completed_at_first_step) != 1 or rescue_id in {
        row["id"] for row in env.public_observation()["tasks"]
    }:
        raise AssertionError(f"{mode}: dependent rescue published before all children completed")

    for _ in range(50):
        if parent_id in {row["id"] for row in env.public_observation()["tasks"] if row["status"] == "completed"}:
            break
        env.step(copy.deepcopy(NOOP))
    completed_children = [env._base._tasks[child_id] for child_id in child_ids]
    if any(row["status"] != "completed" for row in completed_children):
        raise AssertionError(f"{mode}: child tasks did not complete")
    if parent_id not in {row["id"] for row in env.public_observation()["tasks"] if row["status"] == "completed"}:
        raise AssertionError(f"{mode}: logical parent did not complete")
    if rescue_id in {row["id"] for row in env.public_observation()["tasks"]}:
        raise AssertionError(f"{mode}: rescue became public before report delay elapsed")
    env.step(copy.deepcopy(NOOP))
    if rescue_id not in {row["id"] for row in env.public_observation()["tasks"]}:
        raise AssertionError(f"{mode}: delayed rescue was not published")

    def discovery_events() -> list[dict]:
        return [
            event for step in env._steps for event in step["raw_events"]
            if event.get("task_id") == rescue_id and event.get("kind") == "task_discovered"
            and event.get("type") in {"report_generated", "report_received"}
        ]

    events_at_publication = discovery_events()
    env.step(copy.deepcopy(NOOP))
    env.step(copy.deepcopy(NOOP))
    events_after_noops = discovery_events()
    generated = [event for event in events_after_noops if event["type"] == "report_generated"]
    received = [event for event in events_after_noops if event["type"] == "report_received"]
    if len(generated) != 1 or len(received) != 1:
        raise AssertionError(f"{mode}: expected one generation and receipt, got {len(generated)}, {len(received)}")
    if len(events_at_publication) != 2 or events_after_noops != events_at_publication:
        raise AssertionError(f"{mode}: report events changed on repeated NOOP steps")
    if generated[0].get("inspection_id") != parent_id or received[0].get("inspection_id") != parent_id:
        raise AssertionError(f"{mode}: report lost logical parent identity")
    if generated[0]["time"] < max(int(row["completed_at"]) for row in completed_children):
        raise AssertionError(f"{mode}: report generated before last child completion")

    snapshot = capture_snapshot(
        env, state_id=f"v33-exact-once-{mode}-930103-1",
        scenario_id=scenario["parent_id"], parent_family=scenario["parent_family"],
        split="pilot", event_type=f"revealed_rescue_after_{mode}", forced=False,
        replanner_version=REPLANNER_VERSION,
    )
    suffix_seed = 930301 if mode == "split" else 930302
    future_events = sample_v33_future_suffix(
        scenario, after_time=int(env.time),
        horizon_minutes=HORIZON_MINUTES - int(env.time), suffix_seed=suffix_seed,
    )
    suffix = prepare_suffix_snapshot(
        snapshot, future_events=future_events, suffix_seed=suffix_seed,
        env_factory=EventReliefEnv,
    )
    label = run_paired_counterfactual(
        snapshot, replan_action=copy.deepcopy(NOOP), suffixes=[suffix],
        replanner_version=REPLANNER_VERSION, replanner_cpu_seconds=0.0,
        horizon_minutes=30,
    )
    oracle = validate_gain_label(label)
    pair = label["pairs"][0]
    oracle_universe = pair.get("task_universe", {})
    if rescue_id not in oracle_universe.get("task_ids", []):
        raise AssertionError(f"{mode}: oracle obligation set omits revealed rescue {rescue_id}")
    return {
        "mode": mode,
        "scenario_id": scenario["parent_id"],
        "parent_task_id": parent_id,
        "child_task_ids": child_ids,
        "rescue_task_id": rescue_id,
        "first_step_completed_child_count": len(completed_at_first_step),
        "all_children_completed_before_logical_completion": all(row["status"] == "completed" for row in completed_children),
        "report_delay_minutes": 1,
        "logical_parent_completion_time": max(int(row["completed_at"]) for row in completed_children),
        "report_generated_time": int(generated[0]["time"]),
        "report_received_time": int(received[0]["time"]),
        "generated_count_after_repeated_noops": len(generated),
        "received_count_after_repeated_noops": len(received),
        "repeated_noop_added_discovery_events": len(events_after_noops) - len(events_at_publication),
        "paired_label_valid_mask": label["valid_mask"],
        "paired_label_unknown_reason": label.get("unknown_reason"),
        "oracle_valid": oracle["valid"],
        "oracle_label_valid_mask": oracle["label_valid_mask"],
        "oracle_recomputed_gain_mean": oracle["recomputed_gain_mean"],
        "oracle_obligation_task_ids": oracle_universe.get("task_ids", []),
        "oracle_includes_revealed_rescue": rescue_id in oracle_universe.get("task_ids", []),
        "pair_sha256": pair["pair_sha256"],
        "label_sha256": label["label_sha256"],
        "raw_paired_label": label,
    }


def main() -> None:
    result = {
        "schema": "v33-exact-once-focused-evidence/1",
        "evidence_type": "focused_simulation_contract_check",
        "modes": [_run_mode(mode) for mode in ("split", "cooperate")],
    }
    out = Path(__file__).with_name("evidence.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence_json": str(out),
        "modes": [
            {key: mode[key] for key in (
                "mode", "generated_count_after_repeated_noops",
                "received_count_after_repeated_noops", "repeated_noop_added_discovery_events",
                "paired_label_valid_mask", "oracle_valid", "oracle_includes_revealed_rescue",
                "oracle_recomputed_gain_mean",
            )}
            for mode in result["modes"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
