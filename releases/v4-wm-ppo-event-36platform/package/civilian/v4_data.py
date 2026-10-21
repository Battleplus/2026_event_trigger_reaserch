"""Frozen V4 registries and public-history/action-conditioned samples."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
    from .event_oracle import _configured_step_reward, _reward_primitives, _reward_weights
    from .v33_scenarios import CONDITIONS, make_v33_scenario
except ImportError:
    from event_environment import EventReliefEnv, generate_reconstruction_candidates, repair_action
    from event_oracle import _configured_step_reward, _reward_primitives, _reward_weights
    from v33_scenarios import CONDITIONS, make_v33_scenario

POLICIES = ("periodic_public_best", "event_public_best", "random_legal")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def frozen_registry(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    try:
        from .v3_shards import expected_registry
    except ImportError:
        from v3_shards import expected_registry
    existing = expected_registry(task_mode="search_rescue", targets_per_region=3)
    registry = {"train": [row for row in existing if row["split"] == "train"],
                "validation": [row for row in existing if row["split"] == "validation"], "test": []}
    test_base = int(config.get("seeds", {}).get("registry", {}).get("test_base", 840000))
    registry["test"] = [{"split": "test", "condition": condition, "task_mode": "search_rescue",
                         "targets_per_region": 3, "instance_index": ci * 20 + replicate,
                         "seed_variant": test_base + ci * 1000 + replicate}
                        for ci, condition in enumerate(CONDITIONS) for replicate in range(20)]
    if tuple(map(len, (registry["train"], registry["validation"], registry["test"]))) != (84, 84, 140):
        raise ValueError("V4_REGISTRY_COUNTS")
    result = {}
    for split, specs in registry.items():
        rows=[]
        for spec in specs:
            scenario = make_v33_scenario(**spec)
            expected_task_count = 54
            if (scenario["split"] != split or len(scenario["base_scenario"]["tasks"]) != expected_task_count
                    or scenario["task_mode"] != "search_rescue" or scenario["targets_per_region"] != 3):
                raise ValueError("V4_FROZEN_SCENARIO_CONTRACT")
            rows.append({"spec": spec, "scenario_sha256": scenario["scenario_content_sha256"]})
        result[split] = rows
    return result


def public_history(history: Sequence[dict]) -> list[dict]:
    result = []
    for observation in history[-4:]:
        row = copy.deepcopy(observation)
        if any(key in row for key in ("truth", "raw_truth", "event_tape", "future_events")):
            raise ValueError("V4_HIDDEN_FIELD_IN_OBSERVATION")
        row.pop("provenance", None)
        result.append(row)
    return result


def _public_best(env: EventReliefEnv) -> dict:
    try:
        from .baselines import choose_step_c1_actions
        from .fixed_replanner import estimate_public_remaining_cost
    except ImportError:
        from baselines import choose_step_c1_actions
        from fixed_replanner import estimate_public_remaining_cost
    observation = env.public_observation()
    catalog = env.candidate_dispatches()
    c1 = choose_step_c1_actions(observation, catalog)
    actions = [row["full_environment_action"] for row in generate_reconstruction_candidates(env, max_candidates=16)
               if row.get("legal") is True]
    action = min(actions, key=lambda candidate: (estimate_public_remaining_cost(env, candidate),
                                                   0 if candidate.get("dispatches") == c1 else 1,
                                                   digest(candidate))) if actions else {
        "action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}
    repaired, receipt = repair_action(env, action)
    if receipt.get("postrepair_legal") is not True:
        raise ValueError("V4_PUBLIC_BEST_ILLEGAL")
    return repaired


def _legal_candidates(env: EventReliefEnv) -> list[dict]:
    rows = generate_reconstruction_candidates(env, max_candidates=16)
    legal = [row["full_environment_action"] for row in rows if row.get("legal") is True]
    noop = next((row for row in legal if row.get("noop") is True), None)
    if noop is None:
        noop = {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}
        legal.insert(0, noop)
    return [copy.deepcopy(row) for row in legal]


def _truth(env: EventReliefEnv) -> dict:
    return copy.deepcopy(env._raw_truth())


def _targets(history: list[dict], action: dict, steps: list[dict], start_truth: dict, end_truth: dict) -> dict:
    before = history[-1]
    first_before = steps[0].get("public_observation", {}) if steps else {}
    first_after = steps[0].get("next_public_observation", {}) if steps else {}
    first_platform_before = {str(row["id"]): row for row in first_before.get("platforms", [])}
    first_platform_after = {str(row["id"]): row for row in first_after.get("platforms", [])}
    first_task_before = {str(row["id"]): row for row in first_before.get("tasks", [])}
    first_task_after = {str(row["id"]): row for row in first_after.get("tasks", [])}
    start_platform = {str(row["id"]): row for row in start_truth.get("platforms", [])}
    end_platform = {str(row["id"]): row for row in end_truth.get("platforms", [])}
    start_task = {str(row["id"]): row for row in start_truth.get("tasks", [])}
    end_task = {str(row["id"]): row for row in end_truth.get("tasks", [])}
    values, valid, entity_ids = [], [], []
    for row in before.get("platforms", []):
        key = str(row["id"]); a, b = start_platform.get(key), end_platform.get(key)
        entity_ids.append(key)
        if a is None or b is None:
            values.append([0.0] * 4); valid.append([False] * 4); continue
        p0, p1 = first_platform_before.get(key, {}).get("position"), first_platform_after.get(key, {}).get("position")
        e0, e1 = first_platform_before.get(key, {}).get("energy"), first_platform_after.get(key, {}).get("energy")
        values.append([float(p1[0])-float(p0[0]) if p0 is not None and p1 is not None else 0.0,
                       float(p1[1])-float(p0[1]) if p0 is not None and p1 is not None else 0.0,
                       float(e1)-float(e0) if e0 is not None and e1 is not None else 0.0, 0.0])
        valid.append([p0 is not None and p1 is not None, p0 is not None and p1 is not None,
                      e0 is not None and e1 is not None, False])
    for row in before.get("tasks", []):
        key = str(row["id"]); a, b = start_task.get(key), end_task.get(key)
        entity_ids.append(key)
        if a is None or b is None:
            values.append([0.0] * 4); valid.append([False] * 4); continue
        public_a, public_b = first_task_before.get(key, {}), first_task_after.get(key, {})
        p0, p1 = public_a.get("position"), public_b.get("position")
        values.append([float(p1[0])-float(p0[0]) if p0 is not None and p1 is not None else 0.0,
                       float(p1[1])-float(p0[1]) if p0 is not None and p1 is not None else 0.0, 0.0,
                       float(public_b.get("progress", 0))-float(public_a.get("progress", 0))])
        valid.append([p0 is not None and p1 is not None, p0 is not None and p1 is not None, False,
                      "progress" in public_a and "progress" in public_b])
    total_tasks = [row for row in start_truth.get("tasks", []) if row.get("deadline") is not None]
    done = [row for row in total_tasks if end_task.get(str(row["id"]), {}).get("status") == "completed"]
    delays = []
    for row in total_tasks:
        endpoint = end_task.get(str(row["id"]), {})
        finish_time = endpoint.get("completed_at") if endpoint.get("status") == "completed" else end_truth.get("time")
        if finish_time is not None:
            delays.append(max(0.0, float(finish_time)-float(row["deadline"])))
    e0 = sum(float(row.get("energy_used", 0)) for row in start_platform.values())
    e1 = sum(float(row.get("energy_used", 0)) for row in end_platform.values())
    fail_valid = bool(start_task) and set(start_task) == set(end_task)
    fail_count = sum(start_task[key].get("status") not in {"failed", "expired"}
                     and end_task[key].get("status") in {"failed", "expired"} for key in start_task if key in end_task)
    comm = [event for step in steps for event in step.get("raw_events", []) if event.get("type") == "communication_tx"]
    risk_valid = bool(steps) and all("loss_rate" in event for event in comm)
    risk = sum(float(event.get("loss_rate", 0)) for event in comm) / max(1, len(comm))
    start_progress = {str(row["id"]): float(row.get("progress", 0)) for row in start_truth.get("tasks", [])}
    progress = sum(max(0.0, float(row.get("progress", 0))-start_progress.get(str(row["id"]), 0.0))
                   for row in end_truth.get("tasks", [])) / max(1, len(total_tasks))
    ten_minute_valid = len(steps) == 10
    return {"next_minute": {"entity_ids": entity_ids, "values": values, "valid": valid},
            "progress": {"value": progress, "valid": bool(total_tasks) and ten_minute_valid},
            "delay": {"value": sum(delays)/max(1,len(delays)), "valid": bool(total_tasks) and ten_minute_valid},
            "energy": {"value": max(0.0,e1-e0), "valid": bool(start_platform) and set(start_platform)==set(end_platform) and ten_minute_valid},
            "failure": {"value": fail_count/max(1,len(start_task)), "valid": fail_valid and ten_minute_valid},
            "communication_risk": {"value": risk, "valid": risk_valid and ten_minute_valid},
            "horizon": {"steps": len(steps), "valid": ten_minute_valid}}


def _noop() -> dict:
    return {"action_type": "reconstruct", "scope": "NOOP", "noop": True,
            "queue_operations": [], "dispatches": []}


def _safety_disposition(env: EventReliefEnv) -> tuple[dict, dict]:
    try:
        from . import gain_safety
    except ImportError:
        import gain_safety
    observation = env.public_observation()
    report = gain_safety.assess_queue_safety(observation)
    applied = gain_safety.apply_safety_disposition(env, observation, report)
    refreshed = env.public_observation()
    verified = gain_safety.verify_live_safety_disposition(env, observation, report, applied, refreshed)
    if verified.get("resolved") is not True:
        raise ValueError(f"V4_SAFETY_DISPOSITION_UNRESOLVED:{verified.get('unknown_reason')}")
    return refreshed, {"report": report, "apply_receipt": applied, "verification": verified}


def _sample_branch_minutes(seed: int, horizon: int, branch_horizon: int = 10,
                           max_minutes: int = 8) -> tuple[int, ...]:
    latest = horizon - branch_horizon
    if latest < 1 or max_minutes < 1:
        return ()
    rng = random.Random(seed ^ 0xB4A6C39D)
    return tuple(sorted(rng.sample(range(1, latest + 1), min(8, max_minutes, latest))))


def _collect_branches(env: EventReliefEnv, history: list[dict], seed: int, trajectory_policy: str,
                      branch_minutes: Sequence[int] | None = None, branch_horizon: int = 10) -> dict | None:
    if branch_minutes is None:
        branch_minutes = _sample_branch_minutes(seed, int(env.scenario["base_scenario"]["horizon"]), branch_horizon)
    if branch_horizon != 10:
        raise ValueError("V4_BRANCH_HORIZON_PROTOCOL_FIXED_AT_TEN")
    if env.time not in branch_minutes or env.time == 0 or env.time + branch_horizon > int(env.scenario["base_scenario"]["horizon"]):
        return None
    # The live trajectory snapshot, public history, exogenous tape, and safety state
    # are deep-copied once and shared by every candidate continuation.
    snapshot = copy.deepcopy(env)
    noop = _noop()
    best = _public_best(snapshot)
    pool = _legal_candidates(snapshot)
    fixed_rng = random.Random(seed ^ (env.time * 0x9E3779B1))
    named = [("NOOP", noop)]
    if digest(best) != digest(noop):
        named.append(("public_best", best))
    excluded = {digest(action) for _, action in named}
    random_pool = []
    seen = set(excluded)
    for candidate in pool:
        key = digest(candidate)
        if key not in seen:
            seen.add(key); random_pool.append(candidate)
    selected = fixed_rng.sample(random_pool, min(2, len(random_pool)))
    named += [(f"random_{i+1}", action) for i, action in enumerate(selected)]
    branches = []
    for name, action in named:
        branch = copy.deepcopy(snapshot)
        start_truth = _truth(branch)
        branch_history = public_history(history)
        steps=[]; done=truncated=False
        for index in range(branch_horizon):
            current, safety = _safety_disposition(branch)
            if index == 0 and branch._public_history and current != branch._public_history[-1]:
                branch._public_history.append(copy.deepcopy(current))
            requested = copy.deepcopy(action) if index == 0 else _noop()
            repaired, legality = repair_action(branch, requested)
            if legality.get("postrepair_legal") is not True:
                raise ValueError("V4_BRANCH_ACTION_LEGALITY")
            next_observation, reward, done, truncated, info = branch.step(repaired)
            steps.append({"time": int(current["time"]), "full_environment_action": repaired,
                "public_observation": current, "next_public_observation": copy.deepcopy(next_observation),
                "reward": float(reward), "reward_primitives": copy.deepcopy(info["reward_primitives"]),
                "reward_terms": copy.deepcopy(info["reward_terms"]), "raw_events": copy.deepcopy(info["raw_events"]),
                "legality_receipt": legality, "safety_receipt": safety})
            if done or truncated: break
        trajectory = _finish_branch(branch_history, action, steps, start_truth, _truth(branch), env.time)
        branches.append({"name":name,"candidate_key":digest(action),"trajectory":trajectory})
    return {"trajectory_policy":trajectory_policy,"window_start":env.time,
            "public_observation":copy.deepcopy(snapshot.public_observation()),
            "event_tape_id":snapshot.event_tape_id,"branches":branches}


def _finish_branch(history: list[dict], action: dict, steps: list[dict], start_truth: dict, end_truth: dict, start: int) -> dict:
    targets = _targets(history, action, steps, start_truth, end_truth)
    recomputed=[]
    for step in steps:
        components, valid = _reward_primitives(step["full_environment_action"], step["raw_events"])
        if not valid: raise ValueError("V4_REWARD_ORACLE_PRIMITIVES_INVALID")
        reward = _configured_step_reward(components, _reward_weights(None))
        if abs(reward-step["reward"]) > 1e-6: raise ValueError("V4_REWARD_ORACLE_MISMATCH")
        recomputed.append(reward)
    try:
        from .cost_accounting import calculate_branch_cost, task_priorities_from_truth
    except ImportError:
        from cost_accounting import calculate_branch_cost, task_priorities_from_truth
    trace = {"steps":steps,"start_truth":start_truth,"raw_truth":end_truth,
             "initial_planner_call_count":int(not action.get("noop",False))}
    cost = calculate_branch_cost(trace, task_priorities=task_priorities_from_truth(start_truth))
    return {"history":history,"action":copy.deepcopy(action),"steps":steps,"targets":targets,
            "start_truth":start_truth,"end_truth":end_truth,"oracle_cost":cost,
            "oracle_cost_recomputed_from_raw_trace":True,
            "reward_oracle":{"valid":True,"step_rewards":recomputed},
            "window_start":start,"window_end":start+len(steps)}


def _trajectory(env: EventReliefEnv, policy: str, seed: int, *, max_branch_minutes: int = 8,
                branch_horizon: int = 10) -> dict:
    rng = random.Random(seed)
    obs = env.reset(); history = [copy.deepcopy(obs)]; steps = []; windows=[]
    branch_minutes = _sample_branch_minutes(seed, int(env.scenario["base_scenario"]["horizon"]),
                                            branch_horizon, max_branch_minutes)
    done = truncated = False
    try:
        from .event_trigger import TriggerEngine
    except ImportError:
        from event_trigger import TriggerEngine
    trigger = TriggerEngine({"delta_min":4,"history_length":4,"threshold":0.5}, {})
    while not (done or truncated):
        obs, safety = _safety_disposition(env)
        if env._public_history and obs != env._public_history[-1]:
            env._public_history.append(copy.deepcopy(obs))
        history = public_history(env._public_history)
        window = _collect_branches(env, history, seed, policy, branch_minutes, branch_horizon)
        if window is not None: windows.append(window)
        if policy == "random_legal":
            action = copy.deepcopy(rng.choice(_legal_candidates(env)))
        elif policy == "periodic_public_best":
            action = _public_best(env) if env.time % 4 == 0 or safety["report"].get("forced") else _noop()
        else:
            previous = steps[-1]["public_observation"] if steps else None
            detection = trigger.detect(previous, obs, history=history[-4:])
            decision = trigger.decide(detection, env.time, mode="event")
            action = _public_best(env) if decision.get("triggered") or safety["report"].get("forced") else _noop()
        action, receipt = repair_action(env, action)
        if receipt.get("postrepair_legal") is not True:
            raise ValueError("V4_TRAJECTORY_ACTION_LEGALITY")
        nxt, reward, done, truncated, info = env.step(action)
        trigger.record_reconstruction(int(obs.get("time", 0)), bool(int(info.get("effective_reconstruction_count", 0)) > 0))
        steps.append({"time": int(obs["time"]), "public_history": copy.deepcopy(history),
                      "full_environment_action": copy.deepcopy(action),
                      "public_observation": copy.deepcopy(obs), "next_public_observation": copy.deepcopy(nxt),
                      "reward": float(reward), "reward_primitives": copy.deepcopy(info["reward_primitives"]),
                      "reward_terms": copy.deepcopy(info["reward_terms"]), "raw_events": copy.deepcopy(info["raw_events"]),
                      "legality_receipt": receipt, "safety_receipt": safety})
        obs = nxt; history.append(copy.deepcopy(obs)); history = history[-4:]
    return {"steps": steps, "windows":windows,"summary": env.episode_metrics()}


def collect_scenario(scenario: dict, *, seed: int, max_branch_minutes: int = 8,
                     branch_horizon: int = 10) -> dict[str, Any]:
    """Collect trajectories and paired legal action branches from one scenario."""
    trajectories = {}
    for offset, policy in enumerate(POLICIES):
        trajectories[policy] = _trajectory(EventReliefEnv(scenario), policy, seed+offset,
                                            max_branch_minutes=max_branch_minutes, branch_horizon=branch_horizon)
    windows=[{**window,"trajectory_policy":policy}
             for policy,result in trajectories.items() for window in result["windows"]]
    return {"parent_id":scenario["parent_id"],"split":scenario["split"],"trajectories":trajectories,"windows":windows}


def make_model_samples(collection: dict) -> list[dict]:
    rows=[]
    for policy, result in collection.get("trajectories", {}).items():
        for step in result.get("steps", []):
            before = step["public_observation"]
            after = step["next_public_observation"]
            before_platforms = {str(row["id"]): row for row in before.get("platforms", [])}
            after_platforms = {str(row["id"]): row for row in after.get("platforms", [])}
            before_tasks = {str(row["id"]): row for row in before.get("tasks", [])}
            after_tasks = {str(row["id"]): row for row in after.get("tasks", [])}
            entity_ids, values, valid = [], [], []
            for row in before.get("platforms", []):
                key = str(row["id"]); nxt = after_platforms.get(key, {})
                p0, p1 = row.get("position"), nxt.get("position")
                e0, e1 = row.get("energy"), nxt.get("energy")
                entity_ids.append(key)
                values.append([float(p1[0])-float(p0[0]) if p0 is not None and p1 is not None else 0.0,
                               float(p1[1])-float(p0[1]) if p0 is not None and p1 is not None else 0.0,
                               float(e1)-float(e0) if e0 is not None and e1 is not None else 0.0, 0.0])
                valid.append([p0 is not None and p1 is not None, p0 is not None and p1 is not None,
                              e0 is not None and e1 is not None, False])
            for row in before.get("tasks", []):
                key = str(row["id"]); nxt = after_tasks.get(key, {})
                p0, p1 = row.get("position"), nxt.get("position")
                entity_ids.append(key)
                values.append([float(p1[0])-float(p0[0]) if p0 is not None and p1 is not None else 0.0,
                               float(p1[1])-float(p0[1]) if p0 is not None and p1 is not None else 0.0,
                               0.0, float(nxt.get("progress", 0))-float(row.get("progress", 0))])
                valid.append([p0 is not None and p1 is not None, p0 is not None and p1 is not None,
                              False, "progress" in row and "progress" in nxt])
            history = public_history(step.get("public_history", [before]))
            rows.append({"sample_id": f"{collection['parent_id']}-{policy}-t{int(step['time']):03d}-trajectory",
                         "split": collection["split"], "parent_id": collection["parent_id"],
                         "history": history, "action": copy.deepcopy(step["full_environment_action"]),
                         "targets": {"next_minute": {"entity_ids": entity_ids, "values": values, "valid": valid},
                                     "progress": {"value": 0.0, "valid": False},
                                     "delay": {"value": 0.0, "valid": False},
                                     "energy": {"value": 0.0, "valid": False},
                                     "failure": {"value": 0.0, "valid": False},
                                     "communication_risk": {"value": 0.0, "valid": False},
                                     "horizon": {"steps": 1, "valid": False}},
                         "sample_kind": "trajectory_one_step"})
    for window in collection["windows"]:
        for branch in window["branches"]:
            trajectory=branch["trajectory"]
            rows.append({"sample_id":f"{collection['parent_id']}-{window['trajectory_policy']}-t{window['window_start']:03d}-{branch['name']}",
                "split":collection["split"],"parent_id":collection["parent_id"],
                "history":copy.deepcopy(trajectory["history"]),"action":copy.deepcopy(trajectory["action"]),
                "targets":copy.deepcopy(trajectory["targets"]),"branch_trace":copy.deepcopy(trajectory["steps"]),
                "reward_oracle":copy.deepcopy(trajectory["reward_oracle"]), "sample_kind": "ten_minute_branch"})
    return rows
