import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_records import export_capture


def test_export_records_keeps_periodic_decision_event_and_summary(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    static = {"public_task_geometry": {}}
    (capture / "static_config.json").write_text(json.dumps(static) + "\n", encoding="utf-8")
    static_hash = hashlib.sha256(json.dumps(static, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ticks = [
        {"sim_time_s": 0, "static_config_sha256": static_hash, "deltas": {
            "aircraft": {"upsert": {"u1": {"position_m": [0, 0], "available": True}}, "delete": []},
            "public_tasks": {"upsert": {}, "delete": []}, "observations": {"upsert": {}, "delete": []}},
         "events": [], "decisions": [], "costs": []},
        {"sim_time_s": 1, "static_config_sha256": static_hash, "deltas": {
            "aircraft": {"upsert": {}, "delete": []}, "public_tasks": {"upsert": {}, "delete": []},
            "observations": {"upsert": {}, "delete": []}},
         "events": [{"event": "command_applied", "sim_time_s": 1, "aircraft_id": "u1", "decision_id": "d1"}],
         "decisions": [{"event": "decision_created", "sim_time_s": 1, "selected_action": "noop", "decision_id": "d1"}],
         "costs": []},
    ]
    with gzip.open(capture / "ticks.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in ticks:
            stream.write(json.dumps(row) + "\n")
    receipt = capture / "run_receipt.json"
    receipt.write_text(json.dumps({"status": "complete", "branch_metrics": {}}), encoding="utf-8")
    result = export_capture(capture, tmp_path / "records", receipt)
    rows = [json.loads(line) for line in (tmp_path / "records" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["sample_count"] == 2
    assert any(row["kind"] == "event" for row in rows)
    assert rows[-1]["kind"] == "summary"
