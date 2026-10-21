"""Minimal staged V4 data, model, PPO, evaluation, and analysis runner."""
from __future__ import annotations

import argparse
import bisect
import copy
import gzip
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    from . import v4_data
    from .v33_scenarios import make_v33_scenario
except ImportError:
    import v4_data
    from v33_scenarios import make_v33_scenario

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "V4_CONFIG.json"
PROTOCOL_PATH = ROOT / "protocols" / "V4_PROTOCOL.md"


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _write(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as stream:
        stream.write(_json(payload)); stream.flush(); os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: list[dict], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                        allow_nan=False).encode("ascii") + b"\n")
        raw.flush(); os.fsync(raw.fileno())


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "n02-v4-wm-ppo-config/1":
        raise ValueError("V4_CONFIG_SCHEMA")
    try:
        from .v4_policy import POLICY_SEEDS, VARIANTS
    except ImportError:
        from v4_policy import POLICY_SEEDS, VARIANTS
    if tuple(value.get("ppo", {}).get("variants", ())) != VARIANTS or tuple(value.get("ppo", {}).get("seeds", ())) != POLICY_SEEDS:
        raise ValueError("V4_CONFIG_POLICY_VARIANT_OR_SEED_MISMATCH")
    return value


def _source_identity() -> dict[str, str]:
    names = ("event_environment.py", "event_oracle.py", "v33_scenarios.py", "baselines.py",
             "fixed_replanner.py", "gain_safety.py", "relief_env.py", "scenario_schema.py", "region_contract.py",
             "v4_data.py", "cost_accounting.py", "event_trigger.py", "v3_shards.py")
    return {name: hashlib.sha256((PACKAGE / "civilian" / name).read_bytes()).hexdigest()
            for name in names if (PACKAGE / "civilian" / name).is_file()}


def _identity_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _freeze(output: Path, config: dict) -> dict:
    path = output / "FROZEN_REGISTRY.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("config_sha256") != hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest():
            raise ValueError("V4_FROZEN_REGISTRY_CONFIG_MISMATCH")
        return frozen
    registry = v4_data.frozen_registry(config)
    frozen = {"schema": "n02-v4-frozen-registry/1", "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
              "registry": registry, "counts": {key: len(rows) for key, rows in registry.items()},
              "scenario_identity_sha256": {key: hashlib.sha256(v4_data.canonical([r["scenario_sha256"] for r in rows])).hexdigest()
                                            for key, rows in registry.items()}}
    _write(path, frozen)
    return frozen


def _verify_environment_review(output: Path) -> dict:
    path = output / "environment_review.json"
    if not path.is_file():
        raise ValueError("V4_COLLECTION_BLOCKED_ENVIRONMENT_REVIEW_MISSING")
    review = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "approved_read_only", "reviewed_source_identity": _source_identity(),
                "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()}
    for key, value in expected.items():
        if review.get(key) != value:
            raise ValueError(f"V4_ENVIRONMENT_REVIEW_MISMATCH:{key}")
    return review


def _collect(output: Path, config: dict, frozen: dict) -> dict:
    _verify_environment_review(output)
    if (output / "COLLECTION.json.gz").exists():
        raise ValueError("V4_COLLECTION_IMMUTABLE_OUTPUT_EXISTS")
    samples_count = 0; scenario_count = 0
    for split in ("train", "validation"):
        for row in frozen["registry"][split]:
            scenario = make_v33_scenario(**row["spec"])
            if scenario["scenario_content_sha256"] != row["scenario_sha256"]:
                raise ValueError("V4_SCENARIO_FROZEN_IDENTITY_MISMATCH")
            result = v4_data.collect_scenario(scenario,
                seed=int(config["collection"]["trajectory_seed"]) + int(row["spec"]["seed_variant"]),
                max_branch_minutes=int(config["collection"]["max_branch_minutes"]),
                branch_horizon=int(config["collection"]["branch_horizon_minutes"]))
            samples = v4_data.make_model_samples(result)
            rel = Path("collection") / split / f"{scenario['parent_id']}.json.gz"
            _write_jsonl(output / rel, [result])
            _write_jsonl(output / "model_samples" / split / f"{scenario['parent_id']}.json.gz", samples)
            samples_count += len(samples); scenario_count += 1
            progress_path = output / "COLLECTION_PROGRESS.jsonl"
            with progress_path.open("ab") as stream:
                stream.write(json.dumps({"split":split,"parent_id":scenario["parent_id"],
                    "scenario_sha256":row["scenario_sha256"],"sample_count":len(samples)},sort_keys=True).encode("ascii")+b"\n")
                stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema": "n02-v4-collection-receipt/1", "status": "complete", "scenario_count": scenario_count,
               "trajectory_count": scenario_count*3, "branch_sample_count": samples_count,
               "registry_identity": frozen["scenario_identity_sha256"], "source_identity": _source_identity()}
    _write(output / "COLLECTION_RECEIPT.json", receipt)
    return receipt


def _read_samples(path: Path) -> list[dict]:
    rows=[]
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            if line.strip(): rows.append(json.loads(line))
    return rows


class _ModelSampleShards:
    """Random-access view over compressed shards via line-indexed JSONL files."""
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.ends = []
        self.indexes = []
        total = 0
        for path in paths:
            indexed_path = path.with_suffix(".jsonl")
            offsets = []
            if indexed_path.is_file():
                with indexed_path.open("rb") as stream:
                    while True:
                        offset = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if line.strip():
                            offsets.append(offset)
            else:
                with gzip.open(path, "rb") as source, indexed_path.open("wb") as target:
                    while True:
                        line = source.readline()
                        if not line:
                            break
                        if line.strip():
                            offsets.append(target.tell())
                            target.write(line)
            self.indexes.append((indexed_path, offsets))
            total += len(offsets)
            self.ends.append(total)
        if not total:
            raise ValueError("V4_MODEL_SAMPLE_SHARDS_MISSING")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        indexed_path, offsets = self.indexes[shard_index]
        with indexed_path.open("rb") as stream:
            stream.seek(offsets[index - start])
            return json.loads(stream.readline())


def _all_model_samples(output: Path, split: str) -> _ModelSampleShards:
    paths = sorted((output / "model_samples" / split).glob("*.json.gz"))
    if not paths:
        raise ValueError("V4_MODEL_SAMPLE_SHARDS_MISSING")
    return _ModelSampleShards(paths)


def _train_wm(output: Path, config: dict) -> dict:
    rows = _all_model_samples(output, "train")
    validation = _all_model_samples(output, "validation")
    try:
        from .v4_models import evaluate_v4_predictions, load_v4_ensemble, train_v4_ensemble
    except ImportError:
        from v4_models import evaluate_v4_predictions, load_v4_ensemble, train_v4_ensemble
    model_cfg = dict(config["world_model"])
    model_cfg["device"] = "cuda"
    receipt = train_v4_ensemble(rows, output / "world_models", model_cfg, validation_samples=validation)
    evaluation_models, _ = load_v4_ensemble(receipt["checkpoint_paths"], device="cpu")
    receipt["validation_evaluation"] = evaluate_v4_predictions(
        evaluation_models, validation, split="validation", device="cpu",
        batch_size=int(model_cfg.get("evaluation_batch_size", 16)))
    receipt.update({"source_identity": _source_identity(),
                   "train_sample_count": len(rows), "validation_sample_count": len(validation)})
    _write(output / "WORLD_MODEL_RECEIPT.json", receipt)
    return receipt


def _train_ppo(output: Path, config: dict) -> dict:
    try:
        from .v4_models import load_v4_ensemble
    except ImportError:
        from v4_models import load_v4_ensemble
    try:
        from .v4_rollout import train_v4_methods
    except ImportError:
        from v4_rollout import train_v4_methods
    metadata = json.loads((output / "WORLD_MODEL_RECEIPT.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or len(metadata.get("checkpoint_paths", [])) != 3:
        raise ValueError("V4_WORLD_MODEL_RECEIPT_REQUIRED")
    models, _ = load_v4_ensemble(metadata["checkpoint_paths"], device="cpu")
    frozen = json.loads((output / "FROZEN_REGISTRY.json").read_text(encoding="utf-8"))
    scenarios = {key: [make_v33_scenario(**row["spec"]) for row in frozen["registry"][key]]
                 for key in ("train", "validation")}
    rollout_config = {**config["ppo"], "trigger": config["trigger"]}
    policy_source = {name: hashlib.sha256((PACKAGE / "civilian" / name).read_bytes()).hexdigest()
                     for name in ("v4_models.py", "v4_policy.py", "v4_rollout.py")}
    policy_source_identity = _identity_digest({**_source_identity(), **policy_source})
    receipt = train_v4_methods(scenarios["train"], scenarios["validation"], models,
                               output / "policies", rollout_config, policy_source_identity)
    _write(output / "PPO_RECEIPT.json", receipt)
    return receipt


def _evaluate(output: Path, config: dict) -> dict:
    try:
        from .v4_models import evaluate_v4_prediction_groups, load_v4_ensemble
        from .v4_rollout import evaluate_v4_methods
    except ImportError:
        from v4_models import evaluate_v4_prediction_groups, load_v4_ensemble
        from v4_rollout import evaluate_v4_methods
    model_receipt=json.loads((output/"WORLD_MODEL_RECEIPT.json").read_text(encoding="utf-8"))
    policy_receipt=json.loads((output/"PPO_RECEIPT.json").read_text(encoding="utf-8"))
    if model_receipt.get("status") != "complete" or policy_receipt.get("status") != "complete":
        raise ValueError("V4_MODEL_AND_PPO_RECEIPTS_REQUIRED")
    models,_=load_v4_ensemble(model_receipt["checkpoint_paths"],device="cpu")
    frozen = json.loads((output / "FROZEN_REGISTRY.json").read_text(encoding="utf-8"))
    scenarios=[make_v33_scenario(**row["spec"]) for row in frozen["registry"]["test"]]
    if len(scenarios)!=140 or any(scenario["scenario_content_sha256"]!=row["scenario_sha256"]
                                  for scenario,row in zip(scenarios,frozen["registry"]["test"])):
        raise ValueError("V4_FROZEN_TEST_IDENTITY_MISMATCH")
    transition_groups = _frozen_test_transition_samples(
        scenarios, config["collection"],
        evidence_dir=output / "evaluation" / "test_world_model_transitions")
    world_model_metrics = evaluate_v4_prediction_groups(models, transition_groups, split="test", device="cpu")
    rollout_config = {**config["ppo"], "trigger": config["trigger"]}
    result = evaluate_v4_methods(scenarios, output / "policies", models, rollout_config, output / "evaluation")
    result["world_model_test_metrics"] = world_model_metrics
    result["world_model_test_metrics_contract"] = {
        "evaluated_after_ppo_receipt": True, "split": "test",
        "test_samples_used_for_training_or_calibration": False,
        "coverage_reported_per_head": True,
    }
    _write(output / "EVALUATION_RECEIPT.json", result)
    return result


def _frozen_test_transition_samples(scenarios: list[dict], collection_config: dict, *,
                                    evidence_dir: Path | None = None):
    """Yield one scenario's post-freeze labels under the registered branch protocol."""
    for scenario in scenarios:
        collected = v4_data.collect_scenario(
            scenario,
            seed=int(collection_config["trajectory_seed"]) + int(scenario.get("seed_variant", 0)),
            max_branch_minutes=int(collection_config["max_branch_minutes"]),
            branch_horizon=int(collection_config["branch_horizon_minutes"]))
        if evidence_dir is not None:
            _write_jsonl(evidence_dir / f"{scenario['parent_id']}.raw.json.gz", [collected])
        rows = v4_data.make_model_samples(collected)
        for row in rows:
            row["split"] = "test"
        if rows:
            if evidence_dir is not None:
                _write_jsonl(evidence_dir / f"{scenario['parent_id']}.json.gz", rows)
            yield rows


def _analyze(output: Path) -> dict:
    result = json.loads((output / "EVALUATION_RECEIPT.json").read_text(encoding="utf-8"))
    try:
        from .v4_analysis import analyze_paired_evaluation
    except ImportError:
        from v4_analysis import analyze_paired_evaluation
    report=analyze_paired_evaluation(result)
    report["source_evaluation_sha256"]=hashlib.sha256(_json(result)).hexdigest()
    _write(output / "ANALYSIS.json", report)
    return report


def _event_safety_fixture(scenario: dict) -> dict:
    try:
        from . import gain_safety
        from .event_environment import EventReliefEnv
    except ImportError:
        import gain_safety
        from event_environment import EventReliefEnv
    env=EventReliefEnv(scenario); env.reset()
    noop={"action_type":"reconstruct","scope":"NOOP","noop":True,"queue_operations":[],"dispatches":[]}
    event_kinds=set(); steps=0; lifecycle=None; communication_event_seen=False
    horizon=int(scenario["base_scenario"]["horizon"])
    while env.time < horizon:
        before=env.public_observation()
        report=gain_safety.assess_queue_safety(before)
        applied=gain_safety.apply_safety_disposition(env,before,report)
        refreshed=env.public_observation()
        verified=gain_safety.verify_live_safety_disposition(env,before,report,applied,refreshed)
        if verified.get("resolved") is not True:
            raise ValueError("V4_FIXTURE_SAFETY_HOOK_UNRESOLVED")
        _,_,terminated,truncated,info=env.step(copy.deepcopy(noop)); steps+=1
        events=info.get("raw_events",[])
        event_kinds.update(str(row.get("type")) for row in events)
        communication_event_seen |= any(row.get("type")=="communication_changed" for row in events)
        if lifecycle is None and env.time == 1:
            active=[row for row in env.public_observation()["platforms"]
                    if row.get("active_task_id") is not None and not row.get("failed")]
            if not active:
                raise ValueError("V4_FIXTURE_MINUTE1_ACTIVE_TASK_MISSING")
            original=copy.deepcopy(env._communication)
            for link in env._communication:
                link["quality"]=0.0; link["loss_rate"]=1.0
            held_before=env.public_observation(); held_report=gain_safety.assess_queue_safety(held_before)
            held_apply=gain_safety.apply_safety_disposition(env,held_before,held_report)
            held_after=env.public_observation()
            held_verify=gain_safety.verify_live_safety_disposition(env,held_before,held_report,held_apply,held_after)
            paused=[row["task_id"] for row in held_apply.get("actions",[]) if row.get("type")=="safety_task_paused"]
            env._communication=original
            env._refresh_reported_state([])
            resumed=[row.get("task_id") for row in env._extension_events
                     if row.get("type")=="safety_task_resumed"]
            if held_verify.get("resolved") is not True or not paused or not resumed:
                raise ValueError("V4_FIXTURE_COMMUNICATION_PAUSE_RELEASE")
            lifecycle={"probe_minute":env.time,"paused_task_ids":paused,
                       "resumed_task_ids":sorted(set(resumed)),"safety_receipt_resolved":True}
        if terminated or truncated: break
    if steps != horizon or not {"platform_failure","communication_changed"} <= event_kinds or lifecycle is None or not communication_event_seen:
        raise ValueError(f"V4_FIXTURE_90MIN_EVENT_OR_SAFETY_COVERAGE:{steps}:{sorted(event_kinds)}:{lifecycle}")
    return {"platform_count":len(scenario["base_scenario"]["platforms"]),
            "task_count":len(scenario["base_scenario"]["tasks"]),"horizon_minutes":horizon,
            "noop_step_count":steps,"observed_raw_event_kinds":sorted(event_kinds),"safety_lifecycle":lifecycle}


def environment_check(output: Path) -> dict:
    config = _config(); frozen = _freeze(output, config)
    # One public fixture checks actual constructors, action legality, branch and label schema.
    spec = frozen["registry"]["train"][0]
    try:
        from . import v4_data
    except ImportError:
        import v4_data
    scenario = make_v33_scenario(**spec["spec"])
    if scenario["scenario_content_sha256"] != spec["scenario_sha256"]:
        raise ValueError("V4_FIXTURE_SCENARIO_IDENTITY")
    env = v4_data.EventReliefEnv(scenario); env.reset()
    noop={"action_type":"reconstruct","scope":"NOOP","noop":True,"queue_operations":[],"dispatches":[]}
    while env.time < 8:
        v4_data._safety_disposition(env)
        env.step(copy.deepcopy(noop))
    window=v4_data._collect_branches(env,env._public_history[-4:],int(config["collection"]["trajectory_seed"]),
                                     "interface_fixture",branch_minutes=(8,))
    if window is None: raise ValueError("V4_FIXTURE_BRANCH_MISSING")
    branch=window["branches"][1]["trajectory"]
    samples = [{"sample_id":"interface-fixture", "split":"train", "parent_id":scenario["parent_id"],
                "history":branch["history"], "action":branch["action"], "targets":branch["targets"]}]
    if not samples or any(set(s) & {"truth", "raw_truth", "event_tape", "future_events"} for s in samples):
        raise ValueError("V4_FIXTURE_PUBLIC_INPUT_CONTRACT")
    from .v4_models import collate_v4_samples
    inputs, targets = collate_v4_samples(samples)
    if inputs["node_features"].shape[1] != 4 or not targets["next_minute_valid"].any():
        raise ValueError("V4_FIXTURE_MODEL_SCHEMA")
    branch_names=[row["name"] for row in window["branches"]]
    if sum(name.startswith("random_") for name in branch_names)>2 or len({row["candidate_key"] for row in window["branches"]})!=len(window["branches"]):
        raise ValueError("V4_FIXTURE_PAIRED_BRANCH_IDENTITY")
    if any(not row["trajectory"]["reward_oracle"].get("valid") or
           not row["trajectory"].get("oracle_cost_recomputed_from_raw_trace") or
           not row["trajectory"]["targets"].get("horizon",{}).get("valid") for row in window["branches"]):
        raise ValueError("V4_FIXTURE_BRANCH_ORACLE_OR_HORIZON")
    if any(int(row["time"])>int(window["window_start"]) for row in branch["history"]):
        raise ValueError("V4_FIXTURE_FUTURE_INPUT_LEAKAGE")
    compound=next(row for row in frozen["registry"]["train"] if row["spec"]["condition"]=="compound")
    compound_scenario=make_v33_scenario(**compound["spec"])
    event_safety=_event_safety_fixture(compound_scenario)
    receipt = {"schema": "n02-v4-environment-check/1", "status": "fixture_passed_collection_not_run",
               "registry_counts": frozen["counts"], "fixture_sample_count": len(samples),
               "full_scenario_noop_and_fault_fixture":event_safety,
               "branch_oracle_recomputed":True,"future_input_isolation":_hidden_future_counterfactual(scenario),
               "sample_fields": sorted(samples[0]), "model_input_shapes": {k: list(v.shape) for k,v in inputs.items()},
               "target_fields": sorted(targets), "source_identity": _source_identity()}
    _write(output / "ENVIRONMENT_CHECK.json", receipt)
    return receipt


def _hidden_future_counterfactual(scenario: dict) -> dict:
    try:
        from .event_environment import EventReliefEnv
    except ImportError:
        from event_environment import EventReliefEnv
    first = EventReliefEnv(scenario); first.reset()
    now = first.time
    changed = copy.deepcopy(scenario)
    first_future = [event for event in changed["event_tape"] if int(event.get("time", -1)) > now]
    if not first_future:
        raise ValueError("V4_FIXTURE_HIDDEN_FUTURE_EVENT_MISSING")
    for event in changed["event_tape"]:
        if int(event.get("time", -1)) > now:
            if event.get("kind") == "communication_change":
                event["quality"] = max(0.0, min(1.0, float(event.get("quality", 0.0)) * 0.5))
            elif event.get("kind") == "platform_failure":
                event["time"] = max(now + 1, int(event["time"]))
                event["report_delay"] = int(event.get("report_delay", 0)) + 1
            else:
                task_id = event.get("task_id")
                hidden = next((row for row in changed["base_scenario"]["tasks"]
                               if row.get("id") == task_id and not row.get("known_initial")), None)
                if hidden is not None:
                    hidden["position"] = [min(99.9, float(hidden["position"][0]) + 0.25),
                                           min(99.9, float(hidden["position"][1]) + 0.25)]
                elif "new_energy" in event:
                    event["new_energy"] = max(0.0, float(event["new_energy"]) * 0.5)
    try:
        from .event_scenarios import _digest
        from .gain_scenarios import scenario_content_sha256
    except ImportError:
        from event_scenarios import _digest
        from gain_scenarios import scenario_content_sha256
    changed["event_tape"].sort(key=lambda row: (int(row["time"]), str(row["event_id"])))
    changed["event_tape_id"] = _digest(changed["event_tape"])
    changed["scenario_content_sha256"] = scenario_content_sha256(changed)
    second = EventReliefEnv(changed); second.reset()
    public_a, public_b = first.public_observation(), second.public_observation()
    history_a = v4_data.public_history(first._public_history)
    history_b = v4_data.public_history(second._public_history)
    if public_a != public_b or history_a != history_b:
        raise ValueError("V4_FIXTURE_HIDDEN_FUTURE_CHANGED_PUBLIC_PREFIX")
    action_a, action_b = v4_data._public_best(first), v4_data._public_best(second)
    if action_a != action_b:
        raise ValueError("V4_FIXTURE_HIDDEN_FUTURE_CHANGED_FIXED_PUBLIC_DECISION")
    hidden_ids = {event.get("task_id") for event in first_future if event.get("task_id") is not None}
    hidden_a = [row for row in first.scenario["base_scenario"]["tasks"] if row.get("id") in hidden_ids]
    hidden_b = [row for row in changed["base_scenario"]["tasks"] if row.get("id") in hidden_ids]
    suffix_a = v4_data.digest({"events": first_future, "hidden_tasks": hidden_a})
    suffix_b = v4_data.digest({"events": [event for event in changed["event_tape"] if int(event.get("time", -1)) > now],
                               "hidden_tasks": hidden_b})
    if suffix_a == suffix_b:
        raise ValueError("V4_FIXTURE_HIDDEN_FUTURE_SUFFIX_UNCHANGED")
    sample_a = {"history": history_a, "action": action_a, "targets": {}}
    sample_b = {"history": history_b, "action": action_b, "targets": {}}
    inputs_a, _ = v4_data_collate([sample_a])
    inputs_b, _ = v4_data_collate([sample_b])
    import torch
    tensors_equal = all(torch.equal(inputs_a[key], inputs_b[key]) for key in inputs_a)
    if not tensors_equal:
        raise ValueError("V4_FIXTURE_HIDDEN_FUTURE_CHANGED_MODEL_INPUT")
    return {"future_event_count": len(first_future), "future_suffix_sha256_before": suffix_a,
            "future_suffix_sha256_after": suffix_b, "public_history_equal": True,
            "model_inputs_equal": tensors_equal,
            "fixed_public_decision_equal": True, "current_public_observation_equal": True}


def v4_data_collate(samples: list[dict]):
    try:
        from .v4_models import collate_v4_samples
    except ImportError:
        from v4_models import collate_v4_samples
    return collate_v4_samples(samples)


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("environment-check","collect","train-wm","train-ppo","evaluate","analyze"))
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args(argv); config=_config(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "environment-check": result=environment_check(args.output)
    elif args.stage == "collect": result=_collect(args.output,config,_freeze(args.output,config))
    elif args.stage == "train-wm": result=_train_wm(args.output,config)
    elif args.stage == "train-ppo": result=_train_ppo(args.output,config)
    elif args.stage == "evaluate": result=_evaluate(args.output,config)
    else: result=_analyze(args.output)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
