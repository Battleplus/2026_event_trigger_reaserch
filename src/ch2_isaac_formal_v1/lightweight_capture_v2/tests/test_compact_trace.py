import gzip
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_trace import DeltaTrajectoryWriter, iter_lossless_rows, public_event
from compact_existing_suffix import convert_suffix
from evaluate_suffix_light import adapt_snapshot_backend, iter_trace_rows
from public_trace import PublicTraceReader, PublicTraceWriter


def test_delta_trace_round_trip_none_and_branch_reset(tmp_path):
    rows = [
        {"branch": "keep", "aircraft_id": "uav-0", "sim_time_s": 1,
         "task_id": None, "public_target_observations": [], "energy": {"remaining_wh": 10},
         "communication": {"safety_hold": False}, "usd_position_m": [1, 2, 3]},
        {"branch": "replan", "aircraft_id": "uav-0", "sim_time_s": 1,
         "task_id": None, "public_target_observations": [], "energy": {"remaining_wh": 10},
         "communication": {"safety_hold": False}, "usd_position_m": [1, 2, 3]},
        {"branch": "keep", "aircraft_id": "uav-0", "sim_time_s": 2,
         "task_id": "task-1", "public_target_observations": [{"observation_id": "obs-1"}],
         "energy": {"remaining_wh": 9}, "communication": {"safety_hold": True},
         "usd_position_m": [2, 2, 3]},
        {"branch": "keep", "aircraft_id": "uav-0", "sim_time_s": 3,
         "task_id": None, "public_target_observations": [], "energy": {"remaining_wh": 8},
         "communication": {"safety_hold": False}, "usd_position_m": [2, 3, 3]},
    ]
    path = tmp_path / "trajectory.jsonl.gz"
    with DeltaTrajectoryWriter(path) as writer:
        for row in rows:
            writer.write(json.dumps(row, separators=(",", ":")))
    assert list(iter_lossless_rows(path)) == rows


def test_public_event_drops_candidate_and_ppo_payload_recursively():
    result = public_event({"event": "candidate_generation", "candidate_count": 2,
                           "candidates": [{"kind": "transfer"}], "ppo_logits": [1, 2],
                           "selected_action": {"kind": "noop"},
                           "details": {"value": 0.3, "status": "ok"}})
    assert result == {"event": "candidate_generation", "candidate_count": 2,
                      "selected_action": {"kind": "noop"}, "details": {"status": "ok"}}


def test_delta_encoding_reduces_repetitive_trace_bytes(tmp_path):
    rows = [{"branch": "keep", "aircraft_id": f"uav-{actor}", "sim_time_s": t,
             "position_m": [float(actor), float(t % 5), 20.0], "task_id": f"task-{actor}",
             "energy": {"remaining_wh": 500.0 - t, "incremental_wh": 0.01},
             "communication": {"command_attempted": False, "command_delivered": False},
             "observations": []}
            for t in range(120) for actor in range(12)]
    raw_path = tmp_path / "raw.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=6, newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    delta_path = tmp_path / "delta.jsonl.gz"
    with DeltaTrajectoryWriter(delta_path) as writer:
        for row in rows:
            writer.write(json.dumps(row, separators=(",", ":")))
    assert delta_path.stat().st_size < raw_path.stat().st_size


def test_suffix_reader_accepts_raw_and_delta_gzip(tmp_path):
    rows = [{"branch": "keep", "aircraft_id": "uav-0", "sim_time_s": 1},
            {"branch": "replan", "aircraft_id": "uav-0", "sim_time_s": 1}]
    raw_path = tmp_path / "raw.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    delta_path = tmp_path / "delta.jsonl.gz"
    with DeltaTrajectoryWriter(delta_path) as writer:
        for row in rows:
            writer.write(json.dumps(row))

    def raw_jsonl(path):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from (json.loads(line) for line in stream if line.strip())

    assert list(iter_trace_rows(raw_path, raw_jsonl)) == rows
    assert list(iter_trace_rows(delta_path, raw_jsonl)) == rows


def test_existing_suffix_conversion_keeps_source_and_copies_evaluator_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [{"branch": branch, "aircraft_id": "uav-0", "sim_time_s": time_s,
             "task_id": None if time_s == 1 else "task-1", "usd_position_m": [time_s, 0, 3]}
            for branch in ("keep", "replan") for time_s in (1, 2)]
    with gzip.open(source / "trajectory.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in ("run_receipt.json", "manifest.json", "events.jsonl"):
        (source / name).write_text("{}\n", encoding="utf-8")
    original_size = (source / "trajectory.jsonl.gz").stat().st_size
    output = tmp_path / "converted"
    evidence = convert_suffix(source, output)
    assert list(iter_lossless_rows(output / "trajectory.jsonl.gz")) == rows
    assert evidence["trajectory_storage"]["reconstructed_row_count"] == len(rows)
    assert (source / "trajectory.jsonl.gz").stat().st_size == original_size
    assert (output / "events.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_public_trace_rebuilds_two_ticks_and_static_task_geometry(tmp_path):
    output = tmp_path / "public"
    static = {"schema": "public_trace_static_v1",
              "public_task_geometry": {"task-1": {"public_bounds_km": [0, 0, 1, 1],
                                                   "sweep_waypoints_km": [[0.5, 0.5]]}}}
    writer = PublicTraceWriter(output, static)
    for t, x in ((1, 1.0), (2, 2.0)):
        writer.append({"sim_time_s": t,
                       "aircraft": {"uav-0": {"position_m": [x, 0, 0]}},
                       "public_tasks": {"task-1": {"completed": t == 2}},
                       "public_observations": [], "events": [],
                       "decisions": [{"decision_id": "d1"}] if t == 2 else [], "costs": []},
                      hidden_audit={})
    writer.file_info()
    states = list(PublicTraceReader(output).iter_public_states())
    assert [state["aircraft"]["uav-0"]["position_m"][0] for state in states] == [1.0, 2.0]
    assert states[1]["public_tasks"]["task-1"]["completed"] is True
    assert states[0]["public_tasks"]["task-1"]["sweep_waypoints_km"] == [[0.5, 0.5]]
    import gzip
    with gzip.open(output / "view_5s.jsonl.gz", "rt", encoding="utf-8") as stream:
        view_rows = [json.loads(line) for line in stream]
    assert view_rows[0]["sim_time_s"] == 2
    assert view_rows[0]["state_sample_phase"] == "poststep"


def test_snapshot_adapter_reads_isaac_motion_state_position():
    import numpy as np
    backend = {"platforms": {"heli-0-0": {"motion_model_state": {
        "state": type("MotionState", (), {"position": np.array([1.0, 2.0, 3.0])})(),
        "max_speed": 4.0}}}}
    adapted = adapt_snapshot_backend(backend)
    assert adapted["platforms"]["heli-0-0"]["motion_model_state"]["position"].tolist() == [1.0, 2.0, 3.0]
    assert adapted["platforms"]["heli-0-0"]["motion_model_state"]["max_speed"] == 4.0
