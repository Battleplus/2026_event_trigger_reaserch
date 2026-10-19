"""Read-only coordinator recomputation of committed V3.2 raw shards."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import sys

source = Path(sys.argv[1]).resolve()
run = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(source / "package" / "civilian"))
import gain_oracle

checked = []
states = []
for marker in sorted(run.rglob("*.commit.json")):
    m = json.loads(marker.read_text(encoding="utf-8"))
    shard = marker.with_name(m["shard_file"])
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == m["compressed_sha256"]
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        lines = stream.readlines()
    assert len(lines) == m["row_count"]
    assert hashlib.sha256("".join(lines).encode("utf-8")).hexdigest() == m["decompressed_jsonl_sha256"]
    header = json.loads(lines[0])
    trace = header["trajectory"]["trajectory_trace"]
    assert len(trace) == 90
    for line in lines[1:]:
        envelope = json.loads(line)
        row = envelope.get("training_row", envelope)
        label = row["label"]
        receipt = gain_oracle.validate_gain_label(label)
        assert receipt["valid"] is True
        states.append({"state_id": row["state_id"], "forced": row["forced"],
                       "valid_mask": label.get("valid_mask"), "gain_mean": label.get("gain_mean"),
                       "unknown_reason": label.get("unknown_reason"),
                       "recomputed": receipt})
    checked.append({"scenario_id": m["scenario_id"], "condition": m["condition"],
                    "ticks": len(trace), "labels": len(lines) - 1,
                    "non_noop_actions": sum(not r["action"].get("noop", False) for r in trace),
                    "cpu_seconds": m["runtime_receipt"]["process_cpu_seconds"],
                    "shard_sha256": m["compressed_sha256"]})
valid = [s for s in states if not s["forced"] and s["valid_mask"]]
report = {"scope": "committed raw simulation shards only; no uncommitted partial file is accepted",
          "scenarios": checked, "completed_instances": len(checked), "states": states,
          "ordinary_valid": len(valid), "ordinary_positive": sum(s["gain_mean"] > 0 for s in valid),
          "ordinary_nonpositive": sum(s["gain_mean"] <= 0 for s in valid),
          "forced_excluded": sum(s["forced"] for s in states),
          "unknown_reasons": dict(collections.Counter(s["unknown_reason"] for s in states if not s["valid_mask"])),
          "formal_30_30_gate": "NOT_EVALUATED; these are smoke data"}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k not in {"states", "scenarios"}}, sort_keys=True))
