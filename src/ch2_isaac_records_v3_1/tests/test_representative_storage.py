import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_capture
from public_trace import PublicTraceWriter


def test_waypoint_command_traffic_is_not_a_routine_event_stream(tmp_path):
    writer = PublicTraceWriter(tmp_path / "scene", {})
    for t in range(1, 7):
        writer.append({"sim_time_s": t, "aircraft": {}, "public_tasks": {},
                       "events": [{"event": "command_applied", "aircraft_id": "u1", "decision_id": None},
                                  {"event": "command_applied", "aircraft_id": "u2", "decision_id": "d1"}]},
                      hidden_audit={})
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / "scene" / "records.jsonl").read_text().splitlines()]
    assert [(r["t"], r["kind"]) for r in rows] == [(1, "event"), (5, "sample")]


class TinyBackend:
    def __init__(self, _source_root, records, _runtime):
        self._positions = {
            row["aircraft_id"]: {
                "position_m": [value * 1000.0 for value in row["initial_position_km"]],
                "velocity_mps": [0.0, 0.0, 0.0],
                "pose_error_m": 0.0,
                "kit_frame": 0,
            }
            for row in records
        }
        self.scene = SimpleNamespace(tactical_time_s=0.0)

    def positions(self):
        return {key: dict(value) for key, value in self._positions.items()}

    def step(self, _commands, dt):
        self.scene.tactical_time_s += dt
        for value in self._positions.values():
            value["kit_frame"] += 1
        return self.positions()

    def snapshot(self):
        return {"positions": self.positions(),
                "scene_time_s": self.scene.tactical_time_s}

    def close(self):
        pass


def test_representative_persists_only_public_trace_and_labels(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    config_source = project / "collection" / "configs" / "capture_v1_representative.json"
    config = json.loads(config_source.read_text(encoding="utf-8"))
    config["representative_duration_s"] = 1
    config_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "scene"
    source_root = tmp_path / "mock_source"

    monkeypatch.setattr(run_capture, "IsaacBackend", TinyBackend)
    original_sha = run_capture.sha
    monkeypatch.setattr(run_capture, "sha", lambda path: original_sha(path)
                        if Path(path).is_file() else "0" * 64)

    assert run_capture.run(config_path, source_root, output, "representative") == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["public_capture_storage"]["records.jsonl"]["bytes"] > 0
    records = [json.loads(line) for line in
               (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [row for row in records if row.get("kind") == "sample"]
    assert any(row["t"] == 0 and row["reason"] == "periodic" for row in samples)
    assert any(row["t"] == 1 and row["reason"] == "decision" for row in samples)
    assert not (output / "run_receipt.json").exists()
    assert not (output / "trajectory.jsonl.gz").exists()
    assert not (output / "events.jsonl").exists()
    assert (output / "labels" / "state_index.json").is_file()
    assert (output / "labels" / "evaluator_truth.json").is_file()
    assert (output / "labels" / "pairs").is_dir()
