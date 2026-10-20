#!/usr/bin/env python3
"""Independent scalar evaluation for one raw, mixed keep/replan suffix trace."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path


class MissingEvidence(Exception):
    pass


class ContradictoryEvidence(Exception):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContradictoryEvidence(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc


def _snapshot_digest(value) -> str:
    """Reproduce the sampler's content digest without importing its scorer."""
    def normalize(item):
        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return {"__ndarray__": item.tolist(), "dtype": str(item.dtype), "shape": list(item.shape)}
            if isinstance(item, np.generic):
                return item.item()
        except ImportError:
            pass
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, set):
            return sorted((normalize(val) for val in item), key=lambda val: json.dumps(val, sort_keys=True))
        return item
    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _index_entry(index_path: Path, state_id: str):
    index_path = index_path.resolve()
    index = _json(index_path)
    sidecar = index_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise MissingEvidence("state_index.sha256 is missing")
    sidecar_hash = sidecar.read_text(encoding="ascii").split()[0]
    index_hash = _sha(index_path)
    if sidecar_hash != index_hash:
        raise ContradictoryEvidence("state index SHA256 differs from its sidecar")
    entries = index.get("states")
    if not isinstance(entries, list):
        raise MissingEvidence("source state index has no states list")
    matches = [entry for entry in entries
               if Path(str(entry.get("private_file", ""))).stem == state_id
               or str(entry.get("sim_time_s")) == state_id]
    if len(matches) != 1:
        raise ContradictoryEvidence(f"state ID {state_id!r} matched {len(matches)} source index entries")
    entry = matches[0]
    source_dir = index_path.parent
    private_path = (source_dir / str(entry["private_file"])).resolve()
    public_path = (source_dir / str(entry["public_file"])).resolve()
    if not _inside(private_path, source_dir) or not _inside(public_path, source_dir):
        raise ContradictoryEvidence("indexed snapshot/public projection path escapes state_sampling")
    if not private_path.is_file() or not public_path.is_file():
        raise MissingEvidence("indexed private snapshot or public projection is missing")
    private_hash = _sha(private_path)
    public_hash = _sha(public_path)
    if private_hash != entry.get("private_sha256"):
        raise ContradictoryEvidence("private snapshot SHA256 differs from source index")
    if public_hash != entry.get("public_sha256"):
        raise ContradictoryEvidence("public projection SHA256 differs from source index")
    return index, entry, private_path, public_path, {
        "state_index_sha256": index_hash,
        "state_index_sidecar_sha256": _sha(sidecar),
        "private_snapshot_sha256": private_hash,
        "public_projection_sha256": public_hash,
    }


def _read_snapshot(private_path: Path, source_root: Path, collector_root: Path):
    if not source_root.is_dir():
        raise MissingEvidence("--source-root must point to the trusted source tree needed by pickle")
    if not collector_root.is_dir() or not (collector_root / "mission.py").is_file():
        raise MissingEvidence("--collector-root must point to formal_collection_v1 containing mission.py")
    source_root = source_root.resolve()
    collector_root = collector_root.resolve()
    for path in (source_root, collector_root):
        normalized = str(path)
        sys.path[:] = [entry for entry in sys.path
                       if str(Path(entry or ".").resolve()) != normalized]
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(collector_root))
    try:
        with private_path.open("rb") as handle:
            value = pickle.load(handle)
    except (ImportError, AttributeError, pickle.UnpicklingError) as exc:
        raise MissingEvidence(f"private snapshot pickle dependencies could not be loaded: {exc}") from exc
    if not isinstance(value, tuple) or len(value) != 2:
        raise ContradictoryEvidence("private snapshot is not a (backend_state, common_state) pair")
    backend, common = value
    if not isinstance(backend, dict) or not isinstance(common, (tuple, list)) or len(common) != 17:
        raise ContradictoryEvidence("snapshot structure does not match the recorded 17-field common state")
    return backend, common, _snapshot_digest(value)


def _activation_schedule(config: dict, public_task_ids: set[str]):
    source = config.get("evaluator_schedule")
    if source is None:
        source = config.get("pilot")
    if source is None:
        source = {}
    if not isinstance(source, dict):
        raise ContradictoryEvidence("evaluator schedule must be an object")
    ids = list(source.get("activation_task_ids", sorted(public_task_ids)))
    times = list(source.get("activation_times_s", [0.0] * len(ids)))
    if len(ids) != len(times) or len(set(ids)) != len(ids) or not set(ids) <= public_task_ids:
        raise ContradictoryEvidence("public activation schedule does not match the task catalog")
    activation = {task_id: math.inf for task_id in public_task_ids}
    if len(ids) < len(public_task_ids):
        activation.update({task_id: 0.0 for task_id in public_task_ids})
    for task_id, time_s in zip(ids, times):
        value = float(time_s)
        if not math.isfinite(value) or value < 0:
            raise ContradictoryEvidence("public activation time must be finite and non-negative")
        activation[task_id] = value
    return activation


def _sweep(task: dict, lane_spacing_km: float, point_spacing_km: float):
    bounds = task.get("public_bounds_km")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise MissingEvidence(f"task {task.get('task_id')} has no public bounds")
    xmin, ymin, xmax, ymax = map(float, bounds)
    lane_count = max(1, math.ceil((ymax - ymin) / lane_spacing_km))
    point_count = max(1, math.ceil((xmax - xmin) / point_spacing_km))
    route = []
    for lane in range(lane_count):
        y = ymin + (lane + 0.5) * (ymax - ymin) / lane_count
        xs = [xmin + (i + 0.5) * (xmax - xmin) / point_count for i in range(point_count)]
        if lane % 2:
            xs.reverse()
        route.extend((x, y) for x in xs)
    return route


def _event_summary(events_path: Path, branches: set[str], state_time: int, horizon: int):
    events = {branch: [] for branch in branches}
    for event in _jsonl(events_path):
        branch = str(event.get("branch", ""))
        if branch not in branches:
            raise ContradictoryEvidence(f"unexpected event branch {branch!r}")
        time_s = event.get("sim_time_s")
        if not isinstance(time_s, (int, float)) or not math.isfinite(float(time_s)):
            raise ContradictoryEvidence("event has no finite simulation time")
        if not state_time <= float(time_s) < state_time + horizon:
            raise ContradictoryEvidence(f"event lies outside suffix window: {branch}@{time_s}")
        events[branch].append(event)
    return events


def _source_context(suffix_root: Path, source_index: Path, source_root: Path,
                    collector_root: Path, receipt: dict, manifest: dict):
    suffix = receipt.get("suffix", {})
    local_id = str(suffix.get("state_id", ""))
    if not local_id:
        raise MissingEvidence("suffix receipt has no state ID")
    index, entry, private_path, public_path, hashes = _index_entry(source_index, local_id)
    source_dir = source_index.resolve().parent.parent
    source_manifest_path = source_dir / "manifest.json"
    source_receipt_path = source_dir / "run_receipt.json"
    truth_path = source_dir / "evaluator_truth.json"
    if not all(path.is_file() for path in (source_manifest_path, source_receipt_path, truth_path)):
        raise MissingEvidence("original sampler manifest, receipt, or evaluator truth is missing")
    source_manifest = _json(source_manifest_path)
    source_receipt = _json(source_receipt_path)
    truth = _json(truth_path)
    hashes.update({
        "source_manifest_sha256": _sha(source_manifest_path),
        "source_receipt_sha256": _sha(source_receipt_path),
        "source_truth_sha256": _sha(truth_path),
        "suffix_receipt_sha256": _sha(suffix_root / "run_receipt.json"),
        "suffix_manifest_sha256": _sha(suffix_root / "manifest.json"),
    })
    trace_path = next((path for path in (suffix_root / "trajectory.jsonl.gz", suffix_root / "trajectory.jsonl")
                       if path.is_file()), None)
    events_path = suffix_root / "events.jsonl"
    if trace_path is None or not events_path.is_file():
        raise MissingEvidence("mixed suffix trajectory (JSONL or gzip) and events JSONL are required")
    hashes.update({"trajectory_sha256": _sha(trace_path), "events_sha256": _sha(events_path)})

    if source_receipt.get("status") != "complete" or source_receipt.get("actual_isaac_run") is not True:
        raise MissingEvidence("original sampling receipt is not a complete Isaac capture")
    if source_manifest.get("config_sha256") != source_receipt.get("config_sha256"):
        raise ContradictoryEvidence("source receipt and source manifest config hashes differ")
    source_config = source_manifest.get("config", {})
    suffix_config = manifest.get("config", {})
    source_config_hashes = [suffix[key] for key in
                            ("source_config_sha256", "source_manifest_config_sha256") if key in suffix]
    if not source_config_hashes:
        raise MissingEvidence("suffix receipt has no source manifest config SHA256")
    if len(set(source_config_hashes)) != 1:
        raise ContradictoryEvidence("suffix source config SHA256 aliases disagree")
    if source_config_hashes[0] != source_manifest.get("config_sha256"):
        raise ContradictoryEvidence("suffix source config SHA differs from original source manifest")
    if (source_manifest.get("config_sha256") != manifest.get("config_sha256")
            or source_config != suffix_config):
        raise ContradictoryEvidence("suffix config does not exactly match the original sampler config")
    if (manifest.get("run_id") != source_manifest.get("run_id")
            or receipt.get("run_id") != source_manifest.get("run_id")):
        raise ContradictoryEvidence("suffix and source run IDs differ")
    if suffix.get("config_sha256") != source_manifest.get("config_sha256"):
        raise ContradictoryEvidence("suffix receipt config SHA differs from source manifest")
    if source_receipt.get("evaluator_truth_sha256") != hashes["source_truth_sha256"]:
        raise ContradictoryEvidence("original evaluator truth SHA differs from original receipt")
    copied_source_manifest = suffix.get("source_manifest_metadata")
    if copied_source_manifest is not None and copied_source_manifest != source_manifest:
        raise ContradictoryEvidence("suffix receipt's copied source manifest differs from the original manifest")
    copied_source_manifest_hash = suffix.get("source_manifest_sha256")
    if (copied_source_manifest_hash is not None
            and copied_source_manifest_hash != hashes["source_manifest_sha256"]):
        raise ContradictoryEvidence("suffix receipt's source manifest SHA differs from the original manifest")
    suffix_truth = suffix_root / "evaluator_truth.json"
    if suffix_truth.is_file():
        hashes["suffix_truth_sha256"] = _sha(suffix_truth)
        if hashes["suffix_truth_sha256"] != hashes["source_truth_sha256"]:
            raise ContradictoryEvidence("suffix-local evaluator truth differs from original sampler truth")

    if source_receipt.get("state_sampling") != index:
        raise ContradictoryEvidence("public source index differs from original receipt's state_sampling record")
    if suffix.get("sim_time_s") != entry.get("sim_time_s") or suffix.get("backend_time_s") != entry.get("backend_time_s"):
        raise ContradictoryEvidence("suffix time metadata differs from selected source index entry")
    if suffix.get("private_snapshot_sha256") != hashes["private_snapshot_sha256"]:
        raise ContradictoryEvidence("suffix receipt snapshot SHA differs from indexed private snapshot")
    state_index_ref = Path(str(suffix.get("state_index", "")))
    if state_index_ref.is_file() and state_index_ref.resolve() != source_index.resolve():
        raise ContradictoryEvidence("suffix receipt references a different source state index")
    private_ref = Path(str(suffix.get("private_snapshot", "")))
    if private_ref.is_file() and private_ref.resolve() != private_path.resolve():
        raise ContradictoryEvidence("suffix receipt references a different private snapshot")
    backend_state, common, snapshot_hash = _read_snapshot(private_path, source_root, collector_root)
    if any(suffix.get(key) != snapshot_hash for key in
           ("snapshot_sha256", "keep_restore_sha256", "replan_restore_sha256")):
        raise ContradictoryEvidence("suffix snapshot digest and all three restore hashes must match")
    decision_time = int(entry.get("decision_time_s", -1))
    backend_time = float(entry.get("backend_time_s", math.nan))
    if (decision_time <= 0 or decision_time != int(suffix.get("decision_time_s", -2))
            or abs(backend_time - (decision_time - 1)) > 1e-9
            or abs(float(backend_state.get("scene_time_s", math.nan)) - backend_time) > 1e-9):
        raise ContradictoryEvidence("decision time must be t with the indexed backend snapshot at t-1")
    if entry.get("continuation_phase") != "post_public_updates_pre_candidate_generation_pre_step":
        raise MissingEvidence("source snapshot continuation phase is unsupported")
    expected_horizon = min(1800, 5400 - (decision_time - 1))
    if expected_horizon <= 0 or int(suffix.get("horizon_steps", -1)) != expected_horizon:
        raise ContradictoryEvidence("suffix horizon does not match frozen protocol v1")
    if suffix.get("audit_interface", {}).get("state_id") not in {local_id, str(entry.get("sim_time_s"))}:
        raise ContradictoryEvidence("suffix audit state ID differs from source index")
    return {
        "source_dir": source_dir, "source_manifest": source_manifest,
        "source_receipt": source_receipt, "truth": truth,
        "trace_path": trace_path, "events_path": events_path,
        "index": index, "entry": entry, "public_projection": _json(public_path),
        "backend_state": backend_state, "common": common, "snapshot_hash": snapshot_hash,
        "decision_time": decision_time, "horizon": expected_horizon, "hashes": hashes,
    }


def _validate_initial_snapshot(context: dict, receipt: dict, manifest: dict):
    common = context["common"]
    backend = context["backend_state"]
    projection = context["public_projection"]
    source_receipt = context["source_receipt"]
    config = manifest["config"]
    decision_time = context["decision_time"]
    truth_rows = context["truth"].get("targets")
    if not isinstance(truth_rows, list) or not truth_rows:
        raise MissingEvidence("original evaluator truth must contain hidden targets")
    truth = {}
    for item in truth_rows:
        target_id = str(item.get("target_id", ""))
        xy = item.get("target_xy_km")
        if not target_id or target_id in truth or int(item.get("zone", -1)) not in range(9):
            raise ContradictoryEvidence("evaluator truth target IDs/zones are missing or duplicated")
        if not isinstance(xy, list) or len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ContradictoryEvidence(f"invalid truth coordinate for {target_id}")
        if item.get("kind") not in {"uav", "helicopter"}:
            raise MissingEvidence(f"evaluator truth platform kind is missing for {target_id}")
        truth[target_id] = {"zone": int(item["zone"]), "xy": tuple(map(float, xy)), "kind": item["kind"]}

    mission = common[16]
    if not isinstance(mission, dict):
        raise ContradictoryEvidence("common snapshot has no mission state")
    initial_discovered = set(map(str, mission.get("discovered", [])))
    initial_completed = set(map(str, mission.get("completed", [])))
    initial_serviced = initial_completed & set(truth)
    interface = receipt.get("suffix", {}).get("audit_interface", {})
    if (set(map(str, interface.get("initial_discovered_target_ids", []))) != initial_discovered
            or set(map(str, interface.get("initial_serviced_target_ids", []))) != initial_serviced):
        raise ContradictoryEvidence("audit interface initial target IDs differ from private common snapshot")
    truth_ids = set(truth)
    if not initial_discovered <= truth_ids:
        raise ContradictoryEvidence("private snapshot discovered IDs include unknown evaluator truth IDs")

    actor_manifest = source_receipt.get("actor_manifest", [])
    actor_by_id = {str(item["aircraft_id"]): item for item in actor_manifest if item.get("aircraft_id")}
    if len(actor_by_id) != len(actor_manifest) or not actor_by_id:
        raise MissingEvidence("source actor manifest is missing or has duplicate IDs")
    platforms = backend.get("platforms", {})
    if set(platforms) != set(actor_by_id):
        raise ContradictoryEvidence("private backend platform IDs differ from source actor manifest")
    queues, _routes, cursors, services, energy = common[:5]
    failed, safety_paused, communication_held = set(common[8]), set(common[9]), set(common[10])
    public_targets = common[11]
    if not isinstance(public_targets, dict):
        raise ContradictoryEvidence("snapshot public observations are malformed")
    expected_public_task_ids = set(map(str, source_receipt.get("expected_task_ids", [])))
    if not expected_public_task_ids:
        raise MissingEvidence("source receipt has no public task IDs")
    activation = _activation_schedule(config, expected_public_task_ids)
    expected_projection = {
        "sim_time_s": decision_time,
        "aircraft": {},
        "public_observations": [],
    }
    meters_per_unit = float(source_receipt.get("meters_per_unit", 0.0))
    if meters_per_unit <= 0:
        raise MissingEvidence("source receipt has no meters_per_unit")
    for aid in sorted(actor_by_id):
        actor = actor_by_id[aid]
        position_scene = platforms[aid].get("motion_model_state", {}).get("position")
        if position_scene is None or len(position_scene) != 3:
            raise MissingEvidence(f"snapshot has no position for {aid}")
        position_m = [float(value) * meters_per_unit for value in position_scene]
        task_index = int(cursors[aid][0])
        active = []
        if task_index < len(queues[aid]):
            candidate = str(queues[aid][task_index])
            if activation.get(candidate, math.inf) <= decision_time:
                active = [candidate]
        expected_projection["aircraft"][aid] = {
            "kind": actor["aircraft_type"], "home_zone": int(actor["home_zone"]),
            "available": bool(mission.get("aircraft_available", {}).get(aid, True)),
            "faulted": aid in failed, "safety_paused": aid in safety_paused,
            "communication_held": aid in communication_held,
            "position_m": position_m,
            "available_energy_wh": float(energy[aid]),
            "active_task_ids": active,
            "queue_task_ids": [str(task) for task in queues[aid]
                               if activation.get(str(task), math.inf) <= decision_time],
        }
    public_fields = ("target_id", "kind", "zone", "target_xy_km", "observed_at_sim_time_s",
                     "available_at_sim_time_s", "observation_id")
    for target_id in sorted(public_targets):
        item = public_targets[target_id]
        expected_projection["public_observations"].append(
            {key: copy.deepcopy(item[key]) for key in public_fields if key in item})
        if target_id not in initial_discovered or float(item.get("available_at_sim_time_s", math.inf)) > decision_time:
            raise ContradictoryEvidence("snapshot public projection exposes an undetected or not-yet-available target")
        target = truth.get(str(target_id))
        if target and (int(item.get("zone", -1)) != target["zone"]
                       or item.get("kind") != target["kind"]
                       or tuple(map(float, item.get("target_xy_km", []))) != target["xy"]):
            raise ContradictoryEvidence("snapshot public observation differs from evaluator truth")
    saved_aircraft = projection.get("aircraft", {})
    if set(saved_aircraft) != set(expected_projection["aircraft"]):
        raise ContradictoryEvidence("source public projection actor IDs differ from private snapshot")
    if projection.get("sim_time_s") != decision_time:
        raise ContradictoryEvidence("source public projection time differs from decision time")
    for aid, expected in expected_projection["aircraft"].items():
        actual = saved_aircraft[aid]
        for field in ("kind", "home_zone", "available", "faulted", "safety_paused", "communication_held",
                      "active_task_ids", "queue_task_ids"):
            if actual.get(field) != expected[field]:
                raise ContradictoryEvidence(f"public source projection {field} differs from private snapshot for {aid}")
        actual_pos = actual.get("position_m", [])
        if len(actual_pos) != 3 or math.dist(list(map(float, actual_pos)), expected["position_m"]) > 0.02:
            raise ContradictoryEvidence(f"public source projection position differs from private snapshot for {aid}")
        if not math.isclose(float(actual.get("available_energy_wh", math.nan)), expected["available_energy_wh"],
                            rel_tol=1e-10, abs_tol=1e-8):
            raise ContradictoryEvidence(f"public source projection energy differs from private snapshot for {aid}")
    if projection.get("public_observations") != expected_projection["public_observations"]:
        raise ContradictoryEvidence("public source projection observations differ from private common snapshot")
    if len(services) == 0 or not isinstance(services, dict):
        raise MissingEvidence("snapshot service queues are missing")
    tasks = source_receipt.get("public_task_manifest", [])
    if not tasks:
        raise MissingEvidence("source receipt public task geometry is missing")
    task_map = {str(task["task_id"]): task for task in tasks}
    if set(task_map) != expected_public_task_ids:
        raise ContradictoryEvidence("source task geometry and expected task IDs differ")
    return {
        "truth": truth, "actor_by_id": actor_by_id, "task_map": task_map,
        "activation": activation, "initial_discovered": initial_discovered,
        "initial_serviced": initial_serviced, "initial_completed": initial_completed,
    }


def _validate_public_rows(row, expected_observations, truth, time_s):
    observations = row.get("public_target_observations")
    if not isinstance(observations, list):
        raise MissingEvidence("trajectory row lacks the public observation view")
    actual = {}
    for item in observations:
        target_id = str(item.get("target_id", ""))
        if not target_id or target_id in actual:
            raise ContradictoryEvidence("trajectory public observation IDs are empty or duplicated")
        actual[target_id] = item
    if set(actual) != set(expected_observations):
        raise ContradictoryEvidence("trajectory public observation view differs from available prefix+events")
    for target_id, item in actual.items():
        expected = expected_observations[target_id]
        if float(item.get("available_at_sim_time_s", math.inf)) > time_s:
            raise ContradictoryEvidence(f"public target {target_id} appears before availability time")
        if (item.get("observation_id") != expected.get("observation_id")
                or float(item.get("detected_at_sim_time_s", math.inf)) != float(expected["detected_at_sim_time_s"])
                or float(item.get("available_at_sim_time_s", math.inf)) != float(expected["available_at_sim_time_s"])):
            raise ContradictoryEvidence(f"public target {target_id} timing/identity differs from its event")
        target = truth.get(target_id)
        if target and (int(item.get("zone", -1)) != target["zone"] or item.get("kind") != target["kind"]
                       or tuple(map(float, item.get("target_xy_km", []))) != target["xy"]):
            raise ContradictoryEvidence(f"public target {target_id} content differs from evaluator truth")


def _evaluate_trace(context: dict, receipt: dict, manifest: dict, validated: dict):
    trace_path = context["trace_path"]
    event_rows = _event_summary(context["events_path"], {"keep", "replan"},
                                context["decision_time"], context["horizon"])
    branch_events = {branch: defaultdict(list) for branch in ("keep", "replan")}
    for branch, events in event_rows.items():
        for event in events:
            branch_events[branch][int(event["sim_time_s"])].append(event)
    config = manifest["config"]
    receipt_suffix = receipt["suffix"]
    truth = validated["truth"]
    actor_by_id = validated["actor_by_id"]
    task_map = validated["task_map"]
    activation = validated["activation"]
    initial_discovered = validated["initial_discovered"]
    initial_serviced = validated["initial_serviced"]
    initial_completed = validated["initial_completed"]
    public_targets_initial = context["common"][11]
    source_receipt = context["source_receipt"]
    task_routes = {task_id: _sweep(task, float(config["sweep_lane_spacing_km"]),
                                  float(config["sensor_radius_km"]))
                   for task_id, task in task_map.items()}
    zones = source_receipt.get("zone_bounds_km", {})
    if not zones:
        raise MissingEvidence("source receipt has no ROI zone bounds")
    sensor_radius = float(config["sensor_radius_km"])
    waypoint_tolerance = float(config["waypoint_tolerance_km"])
    meters_per_unit = float(source_receipt["meters_per_unit"])
    actor_ids = set(actor_by_id)
    energy_rates = config.get("energy_wh_per_km", {})
    if not energy_rates:
        raise MissingEvidence("config has no analytical energy rates")
    anomaly = config.get("energy_anomaly")
    mode = str(config.get("mode", ""))
    if mode not in {"search", "search_strike"}:
        raise MissingEvidence("unsupported mission mode")
    search_strike = mode == "search_strike"
    if not truth:
        raise MissingEvidence("hidden target denominator is empty")

    metrics = {
        branch: {"energy_wh": 0.0, "violations": 0, "messages": 0, "switches": 0,
                 "planner_calls": 0, "planning_cpu_seconds": 0.0, "row_count": 0,
                 "window_steps": 0, "discovered": set(initial_discovered),
                 "serviced": set(initial_serviced), "unresolved_sum": 0.0,
                 "public_observations": {key: dict(value) for key, value in public_targets_initial.items()},
                 "queues": copy.deepcopy(context["common"][0]),
                 "cursors": copy.deepcopy(context["common"][2]),
                 "completed_tasks": set(initial_completed) & set(task_map),
                 "service_queues": copy.deepcopy(context["common"][3]),
                 "strike_started": dict(context["common"][16].get("strike_tasks", {})),
                 "prev_pos": {}, "last_time": None, "counts": defaultdict(int),
                 "last_group_time": None, "row_messages": 0}
        for branch in ("keep", "replan")
    }
    # Control-cost events are independent of trajectory compression.
    for branch in ("keep", "replan"):
        for event in event_rows[branch]:
            kind = event.get("event")
            if kind in {"safety_violation", "boundary_violation", "constraint_violation"}:
                metrics[branch]["violations"] += 1
            if kind == "transmission_sent" or kind == "command_dropped":
                metrics[branch]["messages"] += 1
            if kind == "controller_switch":
                metrics[branch]["switches"] += 1
            if kind == "candidate_generation":
                metrics[branch]["planner_calls"] += 1
                timing = event.get("timing", {})
                cpu_ns = timing.get("cpu_duration_ns")
                if cpu_ns is None:
                    raise MissingEvidence("planner event lacks CPU duration")
                metrics[branch]["planning_cpu_seconds"] += float(cpu_ns) / 1e9

    time_first = context["decision_time"]
    time_end = time_first + context["horizon"]
    expected_branch_order = {"keep": 0, "replan": 1}
    last_order = (-1, -1)
    group = None

    def finish_group(key, rows):
        if key is None:
            return
        branch, time_s = key
        state = metrics[branch]
        if set(rows) != actor_ids:
            raise MissingEvidence(f"{branch}@{time_s} actor coverage is incomplete")
        if len(rows) != len(actor_ids):
            raise ContradictoryEvidence(f"{branch}@{time_s} contains duplicate actor rows")
        state["window_steps"] += 1
        state["last_group_time"] = time_s
        visible = state["public_observations"]
        # Public observations are received in the pre-step phase, except at t where the snapshot
        # already contains that phase's updates.
        for event in branch_events[branch].get(time_s, []):
            if event.get("event") == "public_observation":
                target_id = str(event.get("target_id", ""))
                det_time = event.get("detected_at_sim_time_s")
                avail_time = event.get("available_at_sim_time_s")
                if not target_id or target_id not in truth or det_time is None or avail_time is None:
                    raise MissingEvidence("public observation event lacks evaluator fields")
                expected_emission = max(float(det_time) + 1.0, math.ceil(float(avail_time)))
                if float(time_s) + 1e-9 < expected_emission:
                    raise ContradictoryEvidence("public observation event precedes the configured detection delay")
                observation = {
                    "target_id": target_id, "observation_id": event.get("observation_id"),
                    "detected_at_sim_time_s": float(det_time),
                    "available_at_sim_time_s": float(avail_time),
                    "zone": truth[target_id]["zone"], "kind": truth[target_id]["kind"],
                    "target_xy_km": list(truth[target_id]["xy"]),
                }
                if observation["observation_id"] is None:
                    raise MissingEvidence("public observation has no observation ID")
                if target_id in visible and visible[target_id].get("observation_id") != observation["observation_id"]:
                    raise ContradictoryEvidence("target has conflicting public observation IDs")
                visible[target_id] = observation
                if search_strike:
                    possible = [aid for aid, actor in actor_by_id.items()
                                if actor["aircraft_type"] == truth[target_id]["kind"]
                                and (actor["aircraft_type"] == "uav"
                                     or int(actor["home_zone"]) == truth[target_id]["zone"])]
                    if possible:
                        state["service_queues"][sorted(possible)[0]].append(target_id)
        # Transfers are emitted before the corresponding physical rows.
        for event in branch_events[branch].get(time_s, []):
            if event.get("event") != "donor_support_transfer":
                continue
            donor = str(event.get("from_aircraft_id", "")); recipient = str(event.get("to_aircraft_id", ""))
            task_id = str(event.get("task_id", ""))
            if donor not in actor_ids or recipient not in actor_ids or task_id not in task_map:
                raise ContradictoryEvidence("task transfer references unknown actor or task")
            donor_queue = state["queues"].get(donor, [])
            if task_id not in donor_queue:
                raise ContradictoryEvidence("task transfer donor queue does not contain transferred task")
            donor_queue.remove(task_id)
            state["queues"].setdefault(recipient, []).append(task_id)

        newly_discovered = set()
        row_disc = set()
        for aid, row in rows.items():
            state["row_count"] += 1
            if int(row.get("sim_time_s", -1)) != time_s or float(row.get("dt_s", math.nan)) != 1.0:
                raise ContradictoryEvidence("trajectory rows must have the expected one-second suffix time")
            if row.get("aircraft_type") != actor_by_id[aid]["aircraft_type"]:
                raise ContradictoryEvidence(f"trajectory actor type differs from source actor manifest for {aid}")
            if int(row.get("zone", -1)) != int(actor_by_id[aid]["home_zone"]):
                raise ContradictoryEvidence(f"trajectory home zone differs from source actor manifest for {aid}")
            current = row.get("usd_position_m")
            previous = row.get("previous_usd_position_m")
            mission_xy = row.get("mission_xy_km")
            if not (isinstance(current, list) and len(current) == 3 and isinstance(previous, list)
                    and len(previous) == 3 and isinstance(mission_xy, list) and len(mission_xy) >= 2):
                raise MissingEvidence(f"{branch}/{aid}/{time_s} lacks position fields")
            current = list(map(float, current)); previous = list(map(float, previous))
            mission_xy = list(map(float, mission_xy[:2]))
            if math.dist(mission_xy, [current[0] / 1000.0, current[1] / 1000.0]) > 2e-5:
                raise ContradictoryEvidence(f"mission XY differs from USD position for {branch}/{aid}/{time_s}")
            prior = state["prev_pos"].get(aid)
            if prior is None:
                initial_scene = context["backend_state"]["platforms"][aid]["motion_model_state"]["position"]
                initial_m = [float(value) * meters_per_unit for value in initial_scene]
                if math.dist(previous, initial_m) > 0.02:
                    raise ContradictoryEvidence(f"first row for {branch}/{aid} does not start from snapshot position at t-1")
            elif math.dist(previous, prior) > 0.02:
                raise ContradictoryEvidence(f"position discontinuity for {branch}/{aid}/{time_s}")
            displacement = math.dist(previous, current)
            kind = actor_by_id[aid]["aircraft_type"]
            expected_rate = float(energy_rates[kind])
            if anomaly and aid == anomaly.get("aircraft_id") and time_s >= float(anomaly["start_s"]):
                expected_rate *= float(anomaly["rate_multiplier"])
            energy = row.get("energy")
            if not isinstance(energy, dict) or energy.get("effective_wh_per_km") is None:
                raise MissingEvidence(f"{branch}/{aid}/{time_s} lacks effective energy rate")
            if not math.isclose(float(energy["effective_wh_per_km"]), expected_rate, rel_tol=1e-10, abs_tol=1e-8):
                raise ContradictoryEvidence(f"reported energy rate differs from config at {branch}/{aid}/{time_s}")
            computed_wh = displacement * expected_rate / 1000.0
            if energy.get("incremental_wh") is None or not math.isclose(float(energy["incremental_wh"]), computed_wh,
                                                                        rel_tol=1e-8, abs_tol=1e-8):
                raise ContradictoryEvidence(f"reported energy increment differs from position-derived cost at {branch}/{aid}/{time_s}")
            state["energy_wh"] += computed_wh

            if kind == "helicopter":
                bounds = zones.get(str(actor_by_id[aid]["home_zone"]))
                if not isinstance(bounds, list) or len(bounds) != 4:
                    raise MissingEvidence(f"no home-zone bounds for {aid}")
                x, y = current[0] / 1000.0, current[1] / 1000.0
                xmin, ymin, xmax, ymax = map(float, bounds)
                if x < xmin - 1e-6 or x > xmax + 1e-6 or y < ymin - 1e-6 or y > ymax + 1e-6:
                    raise ContradictoryEvidence(f"helicopter {aid} left its declared home-zone at {time_s}")

            prior_row = state.get("last_row_by_actor", {}).get(aid)
            comm = row.get("communication") or {}
            faulted = bool(row.get("faulted", False))
            if (faulted or bool(comm.get("safety_hold", False))) and displacement > 0.02:
                raise ContradictoryEvidence(f"fault/communication-held actor {aid} moved during {time_s}")
            if prior_row and bool((prior_row.get("energy") or {}).get("safety_pause", False)) and displacement > 0.02:
                raise ContradictoryEvidence(f"energy-paused actor {aid} moved after its safety hold")
            state.setdefault("last_row_by_actor", {})[aid] = row
            state["prev_pos"][aid] = current

            command_events = [event for event in branch_events[branch].get(time_s, [])
                              if event.get("aircraft_id") == aid
                              and event.get("event") in {"transmission_sent", "command_dropped"}]
            if len(command_events) > 1:
                raise ContradictoryEvidence(f"multiple command attempt events for {branch}/{aid}/{time_s}")
            communication = row.get("communication") or {}
            if bool(communication.get("command_attempted", False)) != bool(command_events):
                raise ContradictoryEvidence(f"command attempt flag/event mismatch for {branch}/{aid}/{time_s}")
            if command_events:
                state["row_messages"] += 1
                sent = command_events[0].get("event") == "transmission_sent"
                if bool(communication.get("command_attempt_accepted", False)) != sent:
                    raise ContradictoryEvidence(f"command acceptance flag/event mismatch for {branch}/{aid}/{time_s}")

            queue = state["queues"].get(aid, [])
            cursor_task, waypoint_index = map(int, state["cursors"].get(aid, [0, 0]))
            expected_task = None
            if cursor_task < len(queue):
                candidate_task = str(queue[cursor_task])
                if activation.get(candidate_task, math.inf) <= time_s:
                    expected_task = candidate_task
            declared_task = row.get("task_id")
            if declared_task != expected_task:
                raise ContradictoryEvidence(f"active public task differs from snapshot cursor for {branch}/{aid}/{time_s}")
            active = expected_task is not None and not faulted
            expected_visit = False
            if active:
                route = task_routes.get(expected_task, [])
                if waypoint_index < len(route):
                    expected_visit = math.dist(mission_xy, route[waypoint_index]) <= waypoint_tolerance
                else:
                    raise ContradictoryEvidence(f"snapshot cursor exceeds task route for {aid}/{expected_task}")
            if bool(row.get("search_cell_visited", False)) != expected_visit:
                raise ContradictoryEvidence(f"waypoint visit does not match geometric replay for {branch}/{aid}/{time_s}")
            expected_complete = False
            if expected_visit:
                waypoint_index += 1
                if waypoint_index >= len(task_routes[expected_task]):
                    expected_complete = True
                    state["cursors"][aid] = [cursor_task + 1, 0]
                    state["completed_tasks"].add(expected_task)
                else:
                    state["cursors"][aid] = [cursor_task, waypoint_index]
            if bool(row.get("task_completed", False)) != expected_complete:
                raise ContradictoryEvidence(f"public task completion does not match waypoint replay for {branch}/{aid}/{time_s}")
            complete_events = [event for event in branch_events[branch].get(time_s, [])
                               if event.get("event") == "search_cell_complete"
                               and event.get("aircraft_id") == aid]
            if len(complete_events) != int(expected_complete):
                raise ContradictoryEvidence(f"search completion event differs from waypoint replay for {branch}/{aid}/{time_s}")
            if complete_events and complete_events[0].get("task_id") != expected_task:
                raise ContradictoryEvidence(f"search completion event names wrong task for {branch}/{aid}/{time_s}")

            _validate_public_rows(row, visible, truth, time_s)
            new_ids = row.get("target_discovered_ids")
            if not isinstance(new_ids, list) or len(set(map(str, new_ids))) != len(new_ids):
                raise MissingEvidence(f"{branch}/{aid}/{time_s} lacks a unique target detection list")
            expected_new = set()
            if active and expected_task is not None:
                task_zone = int(task_map[expected_task]["zone"])
                for target_id, target in truth.items():
                    if (target_id not in state["discovered"] and target["zone"] == task_zone
                            and math.dist(mission_xy, target["xy"]) <= sensor_radius):
                        expected_new.add(target_id)
                state["discovered"].update(expected_new)
            actual_new = set(map(str, new_ids))
            if actual_new != expected_new:
                raise ContradictoryEvidence(f"target discovery IDs do not match truth/radius geometry at {branch}/{aid}/{time_s}")
            if set(map(str, row.get("detected_target_ids", new_ids))) != expected_new:
                raise ContradictoryEvidence(f"detected_target_ids differ from truth/radius geometry at {branch}/{aid}/{time_s}")
            newly_discovered.update(expected_new)
            row_disc.update(actual_new)
            target_events = [event for event in branch_events[branch].get(time_s, [])
                             if event.get("event") == "target_discovered" and event.get("aircraft_id") == aid]
            event_ids = set()
            for event in target_events:
                target_id = str(event.get("target_id", ""))
                if (target_id in event_ids or event.get("task_id") != expected_task
                        or int(event.get("zone", -1)) != truth.get(target_id, {}).get("zone")
                        or float(event.get("detected_at_sim_time_s", -1)) != time_s):
                    raise ContradictoryEvidence(f"target discovery event disagrees with geometry at {branch}/{aid}/{time_s}")
                event_ids.add(target_id)
            if event_ids != expected_new:
                raise ContradictoryEvidence(f"target discovery events differ from geometry at {branch}/{aid}/{time_s}")

            if search_strike:
                for target_id in list(state["service_queues"].get(aid, [])):
                    if target_id not in visible or target_id in state["serviced"]:
                        continue
                    target = truth[target_id]
                    eligible = (actor_by_id[aid]["aircraft_type"] == target["kind"]
                                and (target["kind"] != "helicopter"
                                     or int(actor_by_id[aid]["home_zone"]) == target["zone"]))
                    if not eligible:
                        continue
                    if math.dist(mission_xy, target["xy"]) > float(config["abstract_strike_service_radius_km"]):
                        state["strike_started"].pop(target_id, None)
                        continue
                    start = state["strike_started"].get(target_id)
                    if start is None:
                        state["strike_started"][target_id] = time_s
                    elif time_s - float(start) >= float(config["abstract_strike_service_duration_s"]):
                        state["serviced"].add(target_id)
                        state["service_queues"][aid].remove(target_id)
                        service_events = [event for event in branch_events[branch].get(time_s, [])
                                          if event.get("event") == "abstract_strike_complete"
                                          and str(event.get("target_id")) == target_id
                                          and event.get("aircraft_id") == aid]
                        if len(service_events) != 1:
                            raise ContradictoryEvidence(f"abstract service completion event differs from dwell replay for {target_id}")

        discovered_fraction = len(state["discovered"]) / len(truth)
        if search_strike:
            unresolved = 0.5 * (1.0 - discovered_fraction) + 0.5 * (1.0 - len(state["serviced"]) / len(truth))
        else:
            unresolved = 1.0 - discovered_fraction
        state["unresolved_sum"] += unresolved
        state["last_time"] = time_s

    group_key = None
    group_rows = {}
    for row in _jsonl(trace_path):
        branch = str(row.get("branch", ""))
        if branch not in {"keep", "replan"}:
            raise ContradictoryEvidence(f"unexpected trajectory branch {branch!r}")
        time_value = row.get("sim_time_s")
        if not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)):
            raise ContradictoryEvidence("trajectory row has no finite simulation time")
        time_s = int(time_value)
        if float(time_value) != time_s or not time_first <= time_s < time_end:
            raise ContradictoryEvidence("trajectory time is non-integer or outside frozen suffix window")
        order = (expected_branch_order[branch], time_s)
        if order < last_order:
            raise MissingEvidence("streaming evaluator requires trace grouped by keep then replan and time")
        last_order = order
        key = (branch, time_s)
        if group_key is None:
            group_key = key
        elif key != group_key:
            finish_group(group_key, group_rows)
            group_rows = {}
            group_key = key
        aid = str(row.get("aircraft_id", ""))
        if aid not in actor_ids:
            raise ContradictoryEvidence(f"trajectory references unknown actor {aid!r}")
        if aid in group_rows:
            raise ContradictoryEvidence(f"duplicate trajectory row for {branch}/{aid}/{time_s}")
        group_rows[aid] = row
    finish_group(group_key, group_rows)

    expected_rows = context["horizon"] * len(actor_ids)
    for branch, state in metrics.items():
        if state["row_count"] != expected_rows or state["window_steps"] != context["horizon"]:
            raise MissingEvidence(f"{branch} coverage {state['row_count']} rows / {state['window_steps']} seconds; expected {expected_rows} / {context['horizon']}")
        event_attempts = sum(1 for event in event_rows[branch]
                             if event.get("event") in {"transmission_sent", "command_dropped"})
        if state["row_messages"] != event_attempts:
            raise ContradictoryEvidence(f"{branch} command event counter mismatch")
    if search_strike:
        for branch, state in metrics.items():
            expected_completed = state["serviced"] - initial_serviced
            event_completed = {str(event.get("target_id")) for event in event_rows[branch]
                               if event.get("event") == "abstract_strike_complete"}
            if expected_completed != event_completed:
                raise ContradictoryEvidence(f"{branch} abstract service completion events differ from independent dwell replay")
    changed_actors = set()
    initial_decision_id = f"replan-decision-{context['decision_time']}"
    for event in event_rows["replan"]:
        if event.get("event") == "donor_support_transfer" and event.get("decision_id") == initial_decision_id:
            changed_actors.update(filter(None, (event.get("from_aircraft_id"), event.get("to_aircraft_id"))))
    metrics_out = {}
    for branch, state in metrics.items():
        d = len(state["discovered"])
        s = len(state["serviced"])
        q_terminal = (0.5 * (1 - d / len(truth)) + 0.5 * (1 - s / len(truth))) if search_strike else (1 - d / len(truth))
        L = state["unresolved_sum"] / state["window_steps"]
        energy_kwh = state["energy_wh"] / 1000.0
        j_exec = q_terminal + 0.2 * L + 0.01 * (energy_kwh / 1000.0) + 0.25 * state["violations"]
        metrics_out[branch] = {
            "Q_terminal": q_terminal, "L": L, "valid_duration_s": state["window_steps"],
            "energy_wh": state["energy_wh"], "energy_kwh": energy_kwh,
            "J_exec": j_exec, "discovered_count_terminal": d,
            "discovered_count_initial": len(initial_discovered),
            "serviced_count_terminal": s, "serviced_count_initial": len(initial_serviced),
            "controllable_violation_count": state["violations"],
            "messages": state["messages"], "switches": state["switches"],
            "planner_calls": state["planner_calls"],
            "planning_cpu_seconds": state["planning_cpu_seconds"],
            "trajectory_rows": state["row_count"],
        }
    keep, replan = metrics_out["keep"], metrics_out["replan"]
    incremental = {
        "messages": replan["messages"] - keep["messages"],
        "switches": replan["switches"] - keep["switches"],
        "planner_calls": replan["planner_calls"] - keep["planner_calls"],
        "changed_actors": len(changed_actors),
        "planning_cpu_seconds": replan["planning_cpu_seconds"] - keep["planning_cpu_seconds"],
    }
    c_increment = (0.01 * (incremental["messages"] / 1000.0)
                   + 0.05 * (incremental["switches"] / 100.0)
                   + 0.05 * (incremental["planner_calls"] / 100.0)
                   + 0.01 * (incremental["changed_actors"] / 36.0)
                   + 0.01 * incremental["planning_cpu_seconds"])
    return metrics_out, incremental, c_increment, keep["J_exec"] - replan["J_exec"] - c_increment


def _identity_claims(metadata: dict, field: str) -> list[str]:
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    containers = (metadata, metadata.get("formal"), metadata.get("formal_identity"),
                  config, config.get("formal"))
    aliases = (field, "scenario") if field == "scenario_id" else (field,)
    values = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for alias in aliases:
            value = container.get(alias)
            if value is not None and str(value):
                values.append(str(value))
    return values


def _consistent_identity(field: str, source_manifest: dict, source_receipt: dict,
                         suffix_manifest: dict, suffix_receipt: dict):
    source_values = (_identity_claims(source_manifest, field)
                     + _identity_claims(source_receipt, field))
    suffix_values = (_identity_claims(suffix_manifest, field)
                     + _identity_claims(suffix_receipt, field))
    if len(set(source_values)) > 1:
        raise ContradictoryEvidence(f"original source {field} metadata is inconsistent")
    if len(set(suffix_values)) > 1:
        raise ContradictoryEvidence(f"suffix {field} metadata is inconsistent")
    source_value = source_values[0] if source_values else None
    suffix_value = suffix_values[0] if suffix_values else None
    if source_value is not None and suffix_value is not None and source_value != suffix_value:
        raise ContradictoryEvidence(f"suffix {field} differs from the original source manifest")
    return suffix_value or source_value


def evaluate_suffix(suffix_dir: Path, source_index: Path, source_root: Path,
                    collector_root: Path):
    suffix_dir = Path(suffix_dir).resolve()
    reasons = []
    try:
        receipt_path = suffix_dir / "run_receipt.json"
        manifest_path = suffix_dir / "manifest.json"
        if not receipt_path.is_file() or not manifest_path.is_file():
            raise MissingEvidence("suffix run receipt or manifest is missing")
        receipt = _json(receipt_path)
        manifest = _json(manifest_path)
        if receipt.get("status") != "complete" or receipt.get("actual_isaac_run") is not True:
            raise MissingEvidence("suffix run receipt is not complete actual Isaac")
        context = _source_context(suffix_dir, Path(source_index), Path(source_root),
                                  Path(collector_root), receipt, manifest)
        validated = _validate_initial_snapshot(context, receipt, manifest)
        metrics, incremental, c_increment, gain = _evaluate_trace(context, receipt, manifest, validated)
        config = manifest["config"]
        source_manifest = context["source_manifest"]
        suffix_meta = receipt["suffix"]
        state_index_entry = context["entry"]
        local_id = Path(str(state_index_entry["private_file"])).stem
        scenario_id = _consistent_identity("scenario_id", source_manifest,
                                           context["source_receipt"], manifest, receipt)
        scenario_id = str(scenario_id or source_manifest.get("run_id"))
        split = _consistent_identity("split", source_manifest,
                                     context["source_receipt"], manifest, receipt)
        if split not in {"train", "validation"}:
            reasons.append("split_missing_or_unsupported")
        seed = suffix_meta.get("seed", manifest.get("suffix_seed"))
        if seed is None or seed != manifest.get("suffix_seed"):
            reasons.append("suffix_seed_missing_or_mismatched")
        forced = bool(suffix_meta.get("audit_interface", {}).get("forced", False))
        fixture_only = bool(manifest.get("fixture_only", False) or source_manifest.get("fixture_only", False)
                            or receipt.get("fixture_only", False))
        audit_interface = suffix_meta.get("audit_interface", {})
        forced = forced or audit_interface.get("state_kind") != "ordinary_sampled_decision_state"
        snapshot_hash = context["snapshot_hash"]
        config_hash = source_manifest.get("config_sha256")
        status = "PASS" if not reasons and not forced and not fixture_only else "UNKNOWN"
        valid_mask = status == "PASS"
        if forced:
            reasons.append("forced_state")
        if fixture_only:
            reasons.append("fixture_only")
        return {
            "schema_version": 1, "evaluator": "independent_mixed_suffix_scalar_v1",
            "status": status, "valid_mask": valid_mask, "reasons": reasons,
            "fixture_only": fixture_only, "forced": forced,
            "scenario_id": scenario_id, "local_state_id": local_id,
            "state_id": f"{scenario_id}::{local_id}", "split": split,
            "suffix_seed": seed, "decision_time_s": context["decision_time"],
            "horizon_steps": context["horizon"],
            "snapshot_sha256": snapshot_hash, "config_sha256": config_hash,
            "input_sha256": context["hashes"],
            "J_exec_keep": metrics["keep"]["J_exec"],
            "J_exec_replan": metrics["replan"]["J_exec"],
            "C_increment": c_increment, "G": gain,
            "incremental": incremental, "branches": metrics,
            "checks": {
                "source_index_and_sidecar": True,
                "private_snapshot_and_public_projection_binding": True,
                "snapshot_and_three_restore_hashes": True,
                "snapshot_time_t_minus_1": True,
                "initial_public_projection_matches_snapshot": True,
                "public_observation_timing_and_content": True,
                "target_detection_geometry": True,
                "search_waypoint_completion_geometry": True,
                "stationary_safety_holds": True,
                "helicopter_home_zone": True,
                "position_derived_energy": True,
                "command_and_planner_costs": True,
            },
            "limitations": [
                "Checks public observation timing and observed values; does not prove the entire controller code has no future-information path.",
                "Analytical linear energy is recomputed from persisted USD position deltas; it is not physical aircraft energy.",
            ],
        }
    except MissingEvidence as exc:
        reasons.append(str(exc))
        return {"schema_version": 1, "evaluator": "independent_mixed_suffix_scalar_v1",
                "status": "UNKNOWN", "valid_mask": False, "reasons": reasons,
                "input_sha256": _available_hashes(suffix_dir, Path(source_index))}
    except ContradictoryEvidence as exc:
        reasons.append(str(exc))
        return {"schema_version": 1, "evaluator": "independent_mixed_suffix_scalar_v1",
                "status": "FAIL", "valid_mask": False, "reasons": reasons,
                "input_sha256": _available_hashes(suffix_dir, Path(source_index))}
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        reasons.append(f"evaluator_error:{type(exc).__name__}:{exc}")
        return {"schema_version": 1, "evaluator": "independent_mixed_suffix_scalar_v1",
                "status": "UNKNOWN", "valid_mask": False, "reasons": reasons,
                "input_sha256": _available_hashes(suffix_dir, Path(source_index))}


def _available_hashes(suffix_dir: Path, source_index: Path):
    paths = [suffix_dir / "run_receipt.json", suffix_dir / "manifest.json",
             suffix_dir / "trajectory.jsonl.gz", suffix_dir / "trajectory.jsonl",
             suffix_dir / "events.jsonl", source_index, source_index.with_suffix(".sha256")]
    result = {}
    for path in paths:
        if path.is_file():
            result[path.name] = _sha(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--collector-root", type=Path, required=True,
                        help="formal_collection_v1 directory containing mission.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_suffix(args.suffix_dir, args.source_index, args.source_root, args.collector_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "valid_mask": report.get("valid_mask"),
                      "state_id": report.get("state_id"), "suffix_seed": report.get("suffix_seed"),
                      "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
