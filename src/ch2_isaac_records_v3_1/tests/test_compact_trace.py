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
                       "decisions": [{"event": "decision_created", "decision_id": "d1",
                                      "selected_action": {"kind": "noop"}}] if t == 2 else [], "costs": []},
                      hidden_audit={})
    writer.set_summary({"t_end": 2, "status": "complete"})
    info = writer.file_info()
    assert set(info) >= {"config.json", "records.jsonl"}
    assert not (output / "ticks.jsonl.gz").exists()
    assert not (output / "view_5s.jsonl.gz").exists()
    assert not (output / "evaluation_audit.jsonl.gz").exists()
    states = list(PublicTraceReader(output).iter_public_states())
    assert [state["aircraft"]["uav-0"]["pos"][0] for state in states] == [2.0]
    assert states[0]["public_tasks"]["task-1"]["status"] == "completed"
    assert PublicTraceReader(output).static_config["public_task_geometry"]["task-1"]["sweep_waypoints_km"] == [[0.5, 0.5]]
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["kind"] == "summary"
    assert [row["t"] for row in rows if row["kind"] == "sample"] == [2]


def test_public_trace_keeps_all_same_time_decisions_and_drops_static_fields(tmp_path):
    output = tmp_path / "scene"
    writer = PublicTraceWriter(output, {"schema": "scene_config_v1"})
    writer.append({"sim_time_s": 5,
                   "aircraft": {"u1": {"kind": "uav", "home_zone": 1,
                                         "position_m": [1, 2, 3],
                                         "available_energy_wh": 9,
                                         "active_task_ids": ["task-1"]}},
                   "public_tasks": {"task-1": {"kind": "uav_search", "zone": 1,
                                                 "assignment": "u1", "active": True,
                                                 "route_waypoint_count": 2,
                                                 "route_waypoint_cursor": 1}},
                   "public_observations": [], "events": [],
                   "decisions": [
                       {"event": "decision_created", "decision_id": "d1",
                        "selected_action": {"kind": "noop"}},
                       {"event": "decision_created", "decision_id": "d2",
                        "selected_action": {"kind": "noop"}},
                   ]}, hidden_audit={})
    writer.set_summary({"t_end": 5, "energy_source": "analytical_model"})
    writer.file_info()
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    samples = [row for row in rows if row["kind"] == "sample"]
    assert [row["decision_seq"] for row in samples] == [1, 2]
    assert "kind" not in samples[0]["state"]["platforms"][0]
    assert "home_zone" not in samples[0]["state"]["platforms"][0]
    assert samples[0]["state"]["tasks"][0]["status"] == "running"


def test_snapshot_adapter_reads_isaac_motion_state_position():
    import numpy as np
    backend = {"platforms": {"heli-0-0": {"motion_model_state": {
        "state": type("MotionState", (), {"position": np.array([1.0, 2.0, 3.0])})(),
        "max_speed": 4.0}}}}
    adapted = adapt_snapshot_backend(backend)
    assert adapted["platforms"]["heli-0-0"]["motion_model_state"]["position"].tolist() == [1.0, 2.0, 3.0]
    assert adapted["platforms"]["heli-0-0"]["motion_model_state"]["max_speed"] == 4.0


def test_public_decision_keeps_predecision_pose_and_only_actual_action(tmp_path):
    writer = PublicTraceWriter(tmp_path, {"schema": "scene_config_v1"})
    before = {"aircraft": {"u1": {"position_m": [1, 2, 3], "available": True,
                                   "hidden_target_coordinates": [900, 800], "kind": "uav"}},
              "public_tasks": {"t1": {"active": True, "completed": False,
                                      "assignment": "u1", "kind": "search"}},
              "public_observations": []}
    writer.append({**before, "sim_time_s": 0, "decisions": [], "events": []}, hidden_audit={})
    writer.append({**before, "sim_time_s": 1, "decisions": [
        {"event": "candidate_generation", "decision_id": "not-selected", "candidates": [{"kind": "noop"}]}],
        "events": [{"event": "communication_safety_hold", "sim_time_s": 1,
                    "aircraft_id": "u1"}]}, hidden_audit={})
    writer.append({**before, "sim_time_s": 12.3,
                   "aircraft": {"u1": {"position_m": [4, 5, 6], "available": True}},
                   "decision_state": before,
                   "decisions": [{"event": "decision_created", "decision_id": "d1",
                                  "selected_action": {"kind": "uav_transfer", "task_id": "t1",
                                                      "donor": "u1", "recipient": "u2"},
                                  "decision_effect_sim_time_s": 13}],
                   "events": []}, hidden_audit={"target_truth": [9, 9]})
    writer.set_summary({"t_end": 13, "total_energy_wh": 2, "energy_source": "analytical_model"})
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    samples = [row for row in rows if row["kind"] == "sample"]
    assert [row["t"] for row in samples] == [0, 12.3]
    assert samples[0]["action"] is None
    assert samples[1]["state"]["platforms"][0]["pos"] == [1, 2, 3]
    assert samples[1]["action"]["effect_time"] == 13
    assert samples[1]["action"]["to"] == "u2"
    assert not {"ticks.jsonl.gz", "view_5s.jsonl.gz", "evaluation_audit.jsonl.gz"} & {
        path.name for path in tmp_path.iterdir()}
    assert "hidden_target_coordinates" not in json.dumps(rows)
    assert "candidate_generation" not in json.dumps(rows)


def test_public_decision_keeps_predecision_pose_and_only_actual_action(tmp_path):
    writer = PublicTraceWriter(tmp_path, {"schema": "scene_config_v1"})
    before = {"aircraft": {"u1": {"position_m": [1, 2, 3], "available": True,
                                   "hidden_target_coordinates": [900, 800], "kind": "uav"}},
              "public_tasks": {"t1": {"active": True, "completed": False,
                                      "assignment": "u1", "kind": "search"}},
              "public_observations": []}
    writer.append({**before, "sim_time_s": 0, "decisions": [], "events": []}, hidden_audit={})
    writer.append({**before, "sim_time_s": 1, "decisions": [
        {"event": "candidate_generation", "decision_id": "not-selected", "candidates": [{"kind": "noop"}]}],
        "events": [{"event": "communication_safety_hold", "sim_time_s": 1,
                    "aircraft_id": "u1"}]}, hidden_audit={})
    writer.append({**before, "sim_time_s": 12.3,
                   "aircraft": {"u1": {"position_m": [4, 5, 6], "available": True}},
                   "decision_state": before,
                   "decisions": [{"event": "decision_created", "decision_id": "d1",
                                  "selected_action": {"kind": "uav_transfer", "task_id": "t1",
                                                      "donor": "u1", "recipient": "u2"},
                                  "decision_effect_sim_time_s": 13}],
                   "events": []}, hidden_audit={"target_truth": [9, 9]})
    writer.set_summary({"t_end": 13, "total_energy_wh": 2, "energy_source": "analytical_model"})
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    samples = [row for row in rows if row["kind"] == "sample"]
    assert [row["t"] for row in samples] == [0, 12.3]
    assert samples[0]["action"] is None
    assert samples[1]["state"]["platforms"][0]["pos"] == [1, 2, 3]
    assert samples[1]["action"]["effect_time"] == 13
    assert samples[1]["action"]["to"] == "u2"
    assert not {"ticks.jsonl.gz", "view_5s.jsonl.gz", "evaluation_audit.jsonl.gz"} & {
        path.name for path in tmp_path.iterdir()}
    assert "hidden_target_coordinates" not in json.dumps(rows)
    assert "candidate_generation" not in json.dumps(rows)
