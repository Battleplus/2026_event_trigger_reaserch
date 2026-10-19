"""One required 90-minute NOOP acceptance; never used as training data."""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import sys
import time

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source / "package" / "civilian"))
import v3_runner
import v3_shards
import counterfactual_pairs as pairs
from event_runner import _modules

v3_runner._enforce_runtime()
scenario = v3_runner._scenario_factory()(
    split="train", condition="none_or_light", instance_index=0, seed_variant=990001,
)
env = _modules()[0](scenario)
initial = env.reset()
initial_queues = copy.deepcopy(env._queues)
started_wall, started_cpu = time.perf_counter(), time.process_time()
output.mkdir(parents=True, exist_ok=True)
trace_path = output / "noop-90min.jsonl.gz"
ticks = 0
with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
    stream.write(json.dumps({"scenario": scenario, "initial_public": initial}) + "\n")
    for tick in range(90):
        observation, reward, terminated, truncated, info = env.step(copy.deepcopy(pairs.NOOP_ACTION))
        ticks += 1
        stream.write(json.dumps({"tick": ticks, "observation": observation, "reward": reward,
                                 "terminated": terminated, "truncated": truncated, "info": info}) + "\n")
        if terminated or truncated:
            break
truth = env._raw_truth()
completed = [row["id"] for row in truth["tasks"] if row["status"] == "completed"]
receipt = {
    "purpose": "required NOOP acceptance, not training data",
    "execution_timing": "supplemented during the single smoke run; not completed beforehand",
    "executor": "coordinator acceptance probe; production implementation by GPT-6 Luna",
    "binding": v3_shards.current_binding(), "scenario_id": scenario["parent_id"],
    "ticks": ticks, "final_time": env.time,
    "platform_count": len(initial["platforms"]), "logical_task_count": len(truth["tasks"]),
    "initial_queue_count": sum(len(queue) for queue in initial_queues.values()),
    "completed_task_ids": completed,
    "cpu_seconds": time.process_time() - started_cpu,
    "wall_seconds": time.perf_counter() - started_wall,
    "peak_rss_bytes": v3_runner._peak_rss(),
    "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
}
receipt["passed"] = ticks == 90 and env.time == 90 and len(completed) > 0
(output / "noop-90min.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
assert receipt["passed"]
