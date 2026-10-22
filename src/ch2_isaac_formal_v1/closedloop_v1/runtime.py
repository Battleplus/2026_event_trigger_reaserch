#!/usr/bin/env python3
"""Independent 90-minute closed-loop episode runner, derived from formal v1."""
from __future__ import annotations
import argparse, copy, gzip, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
PILOT_ROOT = Path(__file__).resolve().parents[1]
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))
from formal_collection_v1.isaac_backend import IsaacBackend
from formal_collection_v1.mission import (MissionState, abstract_strike, assign_cross_zone_support,
    build_mission, build_public_replan_catalog, discover_at, initial_c1_greedy, mission_snapshot,
    restore_mission, roi_zones, task_sweep, task_activation_schedule, EventTriggerGate, spatial_speed_cap)
from formal_collection_v1.state_sampling import public_projection
from closedloop_v1.decision import DecisionAdapter
from closedloop_v1.compact_capture import (append_evaluation_audit, compact_tick,
    write_evaluation_context)
from compact_collection_v1 import CompactCollectionWriter

PLANNER_CALL_EVENT = "candidate_generation"
TRAJECTORY_NAME = "trajectory.jsonl.gz"
PERIODIC_INTERVAL_S = 240
RUNTIME_SEMANTICS_ID = "closedloop_v1_fault_release_accounting_v2"
ORDINARY_TRIGGER_EVENTS = {
    "task_activated", "public_observation", "search_cell_complete",
    "platform_fault", "communication_safety_hold", "energy_safety_hold",
    "energy_anomaly",
}

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def max_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)

def mechanism_enabled(cfg, name):
    return bool(cfg.get("mechanisms", {}).get(name, {}).get("enabled", True))

def periodic_trigger_due(time_s, interval_s=240):
    return int(time_s) > 0 and int(time_s) % int(interval_s) == 0

def source_random_allow(seed, time_s, probability=0.5):
    token = f"{int(seed)}:{int(time_s)}:random-allow-suppress".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return value < float(probability), value

def persist(path, data):
    with Path(path).open("w",encoding="utf-8") as f:
        json.dump(data,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())

def snapshot_digest(value):
    """Hash nested runtime state by value, including NumPy arrays and RNG tuples."""
    def normalize(item):
        if isinstance(item, np.ndarray):
            return {"__ndarray__": item.tolist(), "dtype": str(item.dtype), "shape": list(item.shape)}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, set):
            return sorted((normalize(val) for val in item), key=lambda val: json.dumps(val, sort_keys=True))
        return item
    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def stable_drop(seed, actor, t, probability):
    token=f"{seed}:{actor}:{t}:comm".encode("utf-8")
    sample=int.from_bytes(hashlib.sha256(token).digest()[:8],"big")/2**64
    return sample < probability


def suffix_horizon(decision_time_s, cap_s=5400, max_steps=1800):
    """Physical suffix length from a decision at t with backend state at t-1."""
    return min(int(max_steps), int(cap_s) - (int(decision_time_s) - 1))


def segment_step_time(base_time, step_index, skip_prepublic_first_step=False):
    """Return the physical clock used by segment() for a zero-based loop index."""
    return int(base_time) + int(step_index) + (0 if skip_prepublic_first_step else 1)


def effective_energy_rate(cfg, aircraft_id, aircraft_kind, sim_time_s):
    rate = float(cfg["energy_wh_per_km"][aircraft_kind])
    anomaly = cfg.get("energy_anomaly") if mechanism_enabled(cfg, "energy_anomaly") else None
    active = (anomaly is not None and aircraft_id == anomaly["aircraft_id"]
              and float(sim_time_s) >= float(anomaly["start_s"]))
    if active:
        rate *= float(anomaly["rate_multiplier"])
    return rate, active

def transfer_task_is_unstarted(queues, cursor, donor, task_id):
    return task_id in queues[donor] and queues[donor].index(task_id) > int(cursor[donor][0])

def queue_replan_action(pending, decision_id, created_time_s, delay_s, selected_action):
    effect_time_s = int(created_time_s) + max(1, int(delay_s))
    item = {"effect_time_s":effect_time_s,"created_time_s":int(created_time_s),
            "decision_id":decision_id,"selected_action":selected_action}
    pending.append(item)
    return item

def run(config_path, source_root, output, phase="representative", resume_state=None, suffix_seed=None,
        suffix_meta=None, strategy_mode="existing_et", gain_predictor=None, checkpoint_path=None,
        checkpoint_model="mlp", gain_threshold=0.0, deployment_metadata=None,
        enable_compact_capture=False):
    if phase != "representative" or resume_state is not None:
        raise ValueError("closedloop_v1 supports one representative episode only")
    run_wall_start_ns=time.perf_counter_ns(); run_cpu_start_ns=time.process_time_ns()
    config_path=Path(config_path).resolve()
    source_root=Path(source_root); output=Path(output)
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if int(cfg.get("representative_duration_s", 0)) != 5400:
        raise ValueError("closedloop_v1 requires a 5400-second representative episode")
    output.mkdir(parents=True,exist_ok=True)
    if checkpoint_model not in {"constant", "ridge", "mlp", "tree"}:
        raise ValueError("checkpoint_model must be constant, ridge, mlp, or tree")
    if strategy_mode.endswith("_gain_gate") and gain_predictor is None:
        raise ValueError(f"strategy mode {strategy_mode} requires a trained {checkpoint_model} checkpoint")
    rec_path=output/"run_receipt.json"; manifest_path=output/"manifest.json"
    traj_path=output/TRAJECTORY_NAME; events_path=output/"events.jsonl"
    receipt={"run_id":cfg["run_id"],"phase":phase,"status":"initializing","actual_isaac_run":False,
             "config_sha256":sha(config_path),"source_archive_sha256":cfg["source_archive_sha256"],
             "energy_model":"linear distance Wh/km, analytical; not physical measurement",
             "communication_model":"time-addressed command delay/drop model; not radio physics",
             "hidden_truth_logged":False,"source_solver_reused":False,"stages":[],"donor_effects":[]}
    receipt["closedloop"] = {"mode": strategy_mode, "checkpoint_model": checkpoint_model,
        "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha(checkpoint_path) if checkpoint_path else None,
        "gain_threshold": float(gain_threshold),
        "deployment_metadata": copy.deepcopy(deployment_metadata),
        "predictor_device": "cpu" if gain_predictor is not None else None,
        "paired_evaluation": "not_performed", "source_runner": "formal_collection_v1/run_capture.py",
        "source_runner_sha256": "9329E4EB69652EC1F8278E8A7DE71953826261AE6C66EEB0601150FD52EA7693",
        "adapted_runtime_sha256": sha(Path(__file__)),
        "controller_source_sha256": {name: sha(PILOT_ROOT / "formal_collection_v1" / name)
            for name in ("mission.py", "isaac_backend.py", "state_sampling.py")}}
    if suffix_seed is not None:
        receipt["suffix_worker_pid"] = os.getpid()
        receipt["suffix_index_metadata"] = copy.deepcopy(suffix_meta or {})
    receipt["resource_usage"] = {"wall_duration_s": None, "process_cpu_s": None,
                                  "max_rss_bytes": max_rss_bytes(), "gpu_worker_count": 1}
    persist(rec_path,receipt); backend=None
    try:
        seed=int(cfg["seed"]); planner_rng=random.Random(seed)
        center=(planner_rng.uniform(-125,125),planner_rng.uniform(-125,125))
        mission=build_mission(seed,center,int(cfg["targets_per_zone"]),cfg["mode"])
        truth_path=output/"evaluator_truth.json"
        persist(truth_path,{"schema":"isaac_roi_evaluator_truth_v1","run_id":cfg["run_id"],
            "targets":[{"target_id":target_id,"zone":zone,"target_xy_km":list(xy),
                        "kind":mission.truth_kind[target_id]}
                       for target_id,(zone,xy) in sorted(mission.truth.items())]})
        public_mission=MissionState(tasks=mission.tasks,aircraft=mission.aircraft,mode=mission.mode)
        assignment=initial_c1_greedy(public_mission)
        mission.assignments=dict(assignment)
        actors={a.aircraft_id:a for a in mission.aircraft}; tasks={t.task_id:t for t in mission.tasks}
        compact_writer = None
        compact_capture_dir = None
        if enable_compact_capture:
            scene = cfg.get("scene", {})
            static_config = {
                "map_size_km": float(cfg["map_size_km"]),
                "meters_per_unit": float(cfg["meters_per_unit"]),
                "roi_width_km": float(cfg["roi_width_km"]),
                "sensor_radius_km": float(cfg["sensor_radius_km"]),
                "sweep_lane_spacing_km": float(cfg["sweep_lane_spacing_km"]),
                "waypoint_tolerance_km": float(cfg["waypoint_tolerance_km"]),
                "terrain_grid": scene.get("terrain_grid"),
                "forest_tree_count": scene.get("forest_tree_count"),
                "aircraft": [{"aircraft_id": a.aircraft_id, "kind": a.kind,
                              "home_zone": int(a.home_zone)} for a in mission.aircraft],
            }
            compact_capture_dir = output / "compact_capture"
            compact_writer = CompactCollectionWriter(compact_capture_dir, static_config)
        pilot_cfg = dict(cfg.get("pilot", {}))
        formal_cfg = dict(cfg.get("formal", {}))
        periodic_interval_s = PERIODIC_INTERVAL_S
        light_delay_cfg = dict(cfg.get("mechanisms", {}).get("light_delay", {}))
        baseline_observation_delay_s = float(formal_cfg.get("baseline_observation_delay_s", 1.0))
        light_delay_platform_id = str(light_delay_cfg.get("platform_id", "uav-0-1"))
        light_delay_additional_s = float(light_delay_cfg.get("additional_delay_s", 0.0))
        evaluator_schedule = dict(cfg.get("evaluator_schedule", {}))
        if mechanism_enabled(cfg, "task_activation"):
            activation_ids = list(evaluator_schedule.get("activation_task_ids", [t.task_id for t in mission.tasks]))
            activation_times = list(evaluator_schedule.get("activation_times_s", [0.0 for _ in activation_ids]))
        else:
            activation_ids = [t.task_id for t in mission.tasks]
            activation_times = [0.0 for _ in activation_ids]
        if len(activation_ids) < len(mission.tasks):
            activation = {task.task_id: 0.0 for task in mission.tasks}
            delayed_schedule = task_activation_schedule(mission.tasks, activation_ids, activation_times)
            activation.update({task_id: delayed_schedule[task_id] for task_id in activation_ids})
        else:
            activation = task_activation_schedule(mission.tasks, activation_ids, activation_times)
        event_schedule = dict(cfg.get("evaluator_schedule", {}).get("disturbances", {}))
        platform_fault_enabled = mechanism_enabled(cfg, "platform_fault")
        communication_outage_enabled = mechanism_enabled(cfg, "communication_outage")
        platform_fault_time = int(event_schedule.get("platform_fault_time_s", 900))
        comm_outage_time = int(event_schedule.get("communication_outage_time_s", 1800))
        comm_outage_duration = int(event_schedule.get("communication_outage_duration_s", 120))
        raw_speed_blocks = (cfg.get("scene", {}).get("spatial_speed_blocks", [])
                            if mechanism_enabled(cfg, "environment_speed_block") else [])
        speed_blocks = []
        for block in raw_speed_blocks:
            resolved = dict(block)
            if "rect_offset_km" in resolved:
                ox0, oy0, ox1, oy1 = map(float, resolved.pop("rect_offset_km"))
                resolved["rect_km"] = [center[0] + ox0, center[1] + oy0,
                                        center[0] + ox1, center[1] + oy1]
            speed_blocks.append(resolved)
        queues={aid:[] for aid in actors}
        for task_id,aid in assignment.items(): queues[aid].append(task_id)
        for aid in queues:
            queues[aid].sort(key=lambda task_id: (activation.get(task_id, math.inf), task_id))
        records=[]
        for a in mission.aircraft:
            cx,cy=a.public_xy_km; ix=int(a.aircraft_id.rsplit("-",1)[1]); ox=(-.45,.45)[ix]
            records.append({"aircraft_id":a.aircraft_id,"kind":a.kind,"home_zone":a.home_zone,
                            "initial_position_km":[cx+ox,cy+ox,20.0],"max_speed_mps":65.0 if a.kind=="uav" else 100.0,
                            "terrain_clearance_m":120.0})
        runtime=dict(cfg["runtime"]); runtime.update(cfg["scene"])
        runtime["spatial_speed_blocks"] = speed_blocks
        source_files=("scenes/air_combat_scene.py","scenes/air_combat_specs.py","scenes/aircraft_motion.py",
                      "sensors/air_combat_specs.py","sensors/air_combat_obstacles.py","weapons/air_combat_specs.py")
        source_hashes={name:sha(source_root/name) for name in source_files}
        backend=IsaacBackend(source_root,records,runtime)
        rb0=backend.positions()
        if compact_writer is not None:
            disturbances = cfg.get("evaluator_schedule", {}).get("disturbances", {})
            evaluation_config = {
                "mode": cfg["mode"], "representative_duration_s": int(cfg["representative_duration_s"]),
                "sensor_radius_km": float(cfg["sensor_radius_km"]),
                "waypoint_tolerance_km": float(cfg["waypoint_tolerance_km"]),
                "energy_wh_per_km": dict(cfg["energy_wh_per_km"]),
                "energy_capacity_wh": dict(cfg["energy_capacity_wh"]),
                "energy_reserve_fraction": float(cfg["energy_reserve_fraction"]),
                "energy_anomaly": (copy.deepcopy(cfg.get("energy_anomaly"))
                    if mechanism_enabled(cfg, "energy_anomaly") else None),
                "mechanisms": {name: {"enabled": mechanism_enabled(cfg, name)}
                    for name in ("platform_fault", "communication_outage")},
                "evaluator_schedule": {"disturbances": {
                    key: disturbances.get(key) for key in
                    ("platform_fault_time_s", "communication_outage_time_s",
                     "communication_outage_duration_s") if key in disturbances}},
            }
            write_evaluation_context(compact_capture_dir, {
                "schema": "closedloop_compact_evaluation_context_v1",
                "run_id": cfg["run_id"], "evaluation_config": evaluation_config,
                "actor_manifest": [{"aircraft_id": a.aircraft_id,
                    "aircraft_type": a.kind, "home_zone": int(a.home_zone)} for a in mission.aircraft],
                "initial_positions_m": {aid: list(value["position_m"]) for aid, value in rb0.items()},
                "zone_bounds_km": {str(z["zone"]): [z["xmin"], z["ymin"], z["xmax"], z["ymax"]]
                                   for z in roi_zones(center, cfg["roi_width_km"])},
                "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
            })
        max_speed_by_id={r["aircraft_id"]:r["max_speed_mps"] for r in records}
        receipt.update({"status":"scene_ready","actual_isaac_run":True,"map_size_km":300.0,"meters_per_unit":100.0,
                       "roi_center_km":list(center),"roi_width_km":cfg["roi_width_km"],"aircraft_count":36,
                       "aircraft_per_zone":{"uav":2,"helicopter":2},"hidden_target_count":len(mission.truth),
                       "public_search_task_count":len(mission.tasks),"initial_assignment_solver":"c1_greedy_proxy",
                       "c1_source_solver_reused":False,"c1_tie_break":"same-zone capable aircraft, minimum load, nearest cell, id",
                       "hidden_truth_exposed_to_planner":False,"initial_usd_pose_error_max_m":max(x["pose_error_m"] for x in rb0.values()),
                       "evaluator_truth_file":truth_path.name,"evaluator_truth_sha256":sha(truth_path),
                       "initial_positions_m":{aid:x["position_m"] for aid,x in rb0.items()},
                       "expected_task_ids":[t.task_id for t in mission.tasks],
                       "spatial_speed_blocks":speed_blocks,
                       "zone_bounds_km":{str(z["zone"]):[z["xmin"],z["ymin"],z["xmax"],z["ymax"]]
                                         for z in roi_zones(center,cfg["roi_width_km"])},
                       "sensor_radius_km":float(cfg["sensor_radius_km"]),
                       "waypoint_tolerance_km":float(cfg["waypoint_tolerance_km"]),
                       "target_discovery_predicate":"active task zone equals evaluator target zone and 2D Euclidean distance from mission_xy_km to target_xy_km is <= sensor_radius_km",
                       "search_cell_completion_predicate":"each configured sweep waypoint is visited within waypoint_tolerance_km; cell completes after all waypoints are visited",
                       "public_task_manifest":[{"task_id":t.task_id,"zone":t.zone,"cell":t.cell,"allowed_kind":t.allowed_kind,
                                                 "public_bounds_km":[t.public_bounds_km[0],t.public_bounds_km[2],
                                                                     t.public_bounds_km[1],t.public_bounds_km[3]],
                                                 "sweep_waypoints_km":[list(p) for p in task_sweep(
                                                     t,cfg["sweep_lane_spacing_km"],cfg["sensor_radius_km"])]}
                                                for t in mission.tasks],
                       "actor_manifest":[{"aircraft_id":a.aircraft_id,"aircraft_type":a.kind,
                                           "home_zone":a.home_zone,"max_speed_mps":max_speed_by_id[a.aircraft_id]} for a in mission.aircraft]})
        persist(rec_path,receipt)
        routes={aid:{tid:task_sweep(tasks[tid],cfg["sweep_lane_spacing_km"],cfg["sensor_radius_km"]) for tid in q}
                for aid,q in queues.items()}
        cursor={aid:[0,0] for aid in actors}; service={aid:[] for aid in actors}
        energy_cap={aid:float(cfg["energy_capacity_wh"][actors[aid].kind]) for aid in actors}
        energy=dict(energy_cap); last={aid:np.array(rb0[aid]["position_m"],float)/1000 for aid in actors}
        active={aid:list(last[aid]) for aid in actors}; pending={aid:[] for aid in actors}
        failed=set(); safety_paused=set(); communication_held=set(); public_target_observations={}; branch_counters={}
        delayed_observations=[]; pending_replan_actions=[]; active_decision_by_actor={}
        ordinary_gate=EventTriggerGate(int(pilot_cfg.get("ordinary_event_threshold", 1)),
                                       float(pilot_cfg.get("ordinary_event_cooldown_s", 240.0)))
        decision_adapter = DecisionAdapter(
            mode=strategy_mode,
            periodic_interval_s=periodic_interval_s,
            event_threshold=int(pilot_cfg.get("ordinary_event_threshold", 1)),
            event_cooldown_s=float(pilot_cfg.get("ordinary_event_cooldown_s", 240.0)),
            risk_energy_fraction=0.25,
            gain_predictor=gain_predictor,
            gain_threshold=float(gain_threshold))
        communication_seed = seed if suffix_seed is None else int(suffix_seed)
        ordinary_events_since_gate=[]; decision_events=[]; forced_event_times=set()
        handled_forced_event_times=set(); ordinary_decision_sequence=0
        def remaining_route_energy(aid, queue, sim_time_s):
            position=np.asarray(backend.positions()[aid]["position_m"],float)[:2]/1000.0
            total_km=0.0
            for index,task_id in enumerate(queue):
                if index < int(cursor[aid][0]) or task_id in mission.completed:
                    continue
                points=routes[aid].get(task_id,[])
                start=int(cursor[aid][1]) if index==int(cursor[aid][0]) else 0
                points=points[start:]
                if not points:
                    continue
                total_km+=math.dist(position,points[0])
                total_km+=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
                position=np.asarray(points[-1],float)
            rate, _ = effective_energy_rate(cfg, aid, actors[aid].kind, sim_time_s)
            return total_km*rate
        pending_event_rows=[]
        def emit(h,branch,t,event,**fields):
            if event in {"task_activated", "public_observation", "search_cell_complete", "energy_anomaly"}:
                ordinary_events_since_gate.append(event)
                decision_events.append(event)
            if event in {"platform_fault", "communication_safety_hold", "energy_safety_hold"}:
                forced_event_times.add(int(t))
                if int(t) not in handled_forced_event_times:
                    decision_events.append(event)
            row={"schema_version":1,"run_id":cfg["run_id"],"phase":phase,"branch":branch,
                 "sim_time_s":t,"event":event,"event_id":f"{branch}:{t}:{event}:{fields.get('aircraft_id','')}:{fields.get('task_id',fields.get('target_id',''))}",**fields}
            pending_event_rows.append(row)
            m=branch_counters.setdefault(branch,{"completed":set(),"discovered":set(),"serviced":set(),
                 "energy_wh":0.0,"command_attempts":0,"switches":0,"planner_calls":0,"violations":0,"faulted":set(),"rows":0,
                 "speed_block_actor_steps":0})
            if event=="search_cell_complete":m["completed"].add(fields.get("task_id"))
            elif event=="target_discovered":m["discovered"].add(fields.get("target_id"))
            elif event=="abstract_strike_complete":m["serviced"].add(fields.get("target_id"))
            elif event=="controller_switch":m["switches"]+=1
            elif event==PLANNER_CALL_EVENT:m["planner_calls"]+=1
            elif event in {"safety_violation","boundary_violation","constraint_violation"}:m["violations"]+=1
            elif event=="platform_fault":m["faulted"].add(fields.get("aircraft_id"))
            if event in ORDINARY_TRIGGER_EVENTS:
                ordinary_gate.observe_event()
        def common_state():
            return copy.deepcopy((queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,
                                  communication_held,public_target_observations,delayed_observations,
                                  pending_replan_actions,active_decision_by_actor,ordinary_gate,mission_snapshot(mission)))
        def restore_common(snapshot):
            nonlocal queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused
            nonlocal communication_held,public_target_observations,delayed_observations
            nonlocal pending_replan_actions,active_decision_by_actor,ordinary_gate
            (queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,
             communication_held,public_target_observations,delayed_observations,
             pending_replan_actions,active_decision_by_actor,ordinary_gate,ms)=copy.deepcopy(snapshot)
            restore_mission(mission,ms)
        def fail_actor(aid,t,branch,ev):
            if aid in failed:return
            failed.add(aid); actors[aid].available=False; pending[aid].clear()
            current=backend.positions()[aid]["position_m"]; active[aid]=[v/1000 for v in current]
            pre_fault_queue=list(queues[aid]); pre_fault_cursor=list(cursor[aid])
            rest=queues[aid][cursor[aid][0]:]; queues[aid]=queues[aid][:cursor[aid][0]]
            for tid in rest:
                choices=[]
                for recipient,actor in actors.items():
                    if (actor.kind!=tasks[tid].allowed_kind or not actor.available
                            or recipient in failed or recipient in safety_paused
                            or recipient in communication_held):
                        continue
                    current_energy=remaining_route_energy(recipient,queues[recipient],t)
                    endpoint=np.asarray(backend.positions()[recipient]["position_m"],float)[:2]/1000.0
                    for q_index,q_task in enumerate(queues[recipient]):
                        if q_index<int(cursor[recipient][0]) or q_task in mission.completed:
                            continue
                        q_points=routes[recipient].get(q_task,[])
                        q_start=int(cursor[recipient][1]) if q_index==int(cursor[recipient][0]) else 0
                        if q_points[q_start:]: endpoint=np.asarray(q_points[-1],float)
                    points=routes[aid][tid]
                    added_km=math.dist(endpoint,points[0])+sum(math.dist(a,b) for a,b in zip(points,points[1:]))
                    recipient_rate, _ = effective_energy_rate(cfg, recipient, actor.kind, t)
                    projected=current_energy+added_km*recipient_rate
                    reserve=energy_cap[recipient]*float(cfg["energy_reserve_fraction"])
                    energy_margin=energy[recipient]-projected-reserve
                    if energy_margin>=0:
                        choices.append((recipient,energy_margin))
                choices.sort(key=lambda item:(actors[item[0]].home_zone!=actors[aid].home_zone,
                                              len(queues[item[0]]),-item[1],item[0]))
                if choices:
                    donor,_=choices[0]
                    if assign_cross_zone_support(mission,actors[donor].home_zone,donor):
                        queues[donor].append(tid); routes[donor][tid]=routes[aid][tid]
                        mission.assignments[tid]=donor
                        support_kind=("local_uav_support" if actors[donor].home_zone==actors[aid].home_zone
                                      else "cross_zone_uav_support")
                        effect={"time_s":t,"branch":branch,"donor":aid,"recipient":donor,"task_id":tid,"kind":support_kind}
                        receipt["donor_effects"].append(effect)
                        emit(ev,branch,t,"donor_support_transfer",from_aircraft_id=aid,to_aircraft_id=donor,
                             task_id=tid,support_kind=support_kind)
                        continue
                mission.assignments.pop(tid, None)
                emit(ev,branch,t,"fault_task_unrecovered",aircraft_id=aid,task_id=tid,
                     details={"failure_cost_unit":"one_unrecovered_public_task"})
            release_state={
                "schema":"fault_queue_release_state_v1",
                "source":"executor_queues_cursor_and_mission_assignments_after_release",
                "pre_fault":{"queue_task_ids":pre_fault_queue,"cursor":pre_fault_cursor,
                             "unfinished_task_ids":list(rest)},
                "post_fault":{"queues_by_aircraft":{key:list(value) for key,value in queues.items()},
                              "cursors_by_aircraft":{key:list(value) for key,value in cursor.items()},
                              "assignments_by_task":dict(mission.assignments),
                              "completed_task_ids":sorted(mission.completed),
                              "actor_state_by_id":{key:{"kind":actor.kind,"home_zone":int(actor.home_zone),
                                  "available":bool(actor.available),"faulted":key in failed,
                                  "safety_paused":key in safety_paused,
                                  "communication_held":key in communication_held}
                                  for key,actor in actors.items()}},
            }
            emit(ev,branch,t,"platform_fault",aircraft_id=aid,details={
                "action":"local_hold","unfinished_tasks_released":rest,
                "queue_release_state":release_state})
        def segment(branch,duration,ev,base_time=0,replan_candidates=None,replan_delay_s=1,
                    replan_selected_action=None,skip_prepublic_first_step=False):
            nonlocal ordinary_decision_sequence
            rows=0
            pending_trajectory_rows=[]
            branch_counters.setdefault(branch,{"completed":set(),"discovered":set(),"serviced":set(),"energy_wh":0.0,
                "command_attempts":0,"switches":0,"planner_calls":0,"violations":0,"faulted":set(),"rows":0,
                "speed_block_actor_steps":0})
            if branch in {"keep","replan"}:
                for aid in sorted(failed): emit(ev,branch,base_time,"platform_fault",aircraft_id=aid,details={"restored_from_common_prefix":True})
            if branch=="replan":
                chosen=replan_selected_action or {"kind":"noop","estimated_gain_s":0.0}
                decision_id=f"{branch}-decision-{base_time}"
                scheduled=queue_replan_action(pending_replan_actions,decision_id,base_time,replan_delay_s,chosen)
                delay_s=scheduled["effect_time_s"]-base_time
                emit(ev,branch,base_time,"decision_created",decision_id=decision_id,
                     decision_created_sim_time_s=base_time,decision_complete_sim_time_s=base_time+delay_s,
                     decision_effect_sim_time_s=base_time+delay_s,selected_action=chosen,
                     timing={"planning_delay_s":delay_s,"source":"measured_catalog_timing"})
                emit(ev,branch,base_time,"replan_catalog_choice",details={"catalog":replan_candidates,"chosen":chosen})
            for j in range(duration):
                t=segment_step_time(base_time,j,skip_prepublic_first_step)
                if branch=="representative" and j==0:
                    receipt["closedloop"].setdefault("timing", {})["startup_load"]={
                        "wall_duration_ns":time.perf_counter_ns()-run_wall_start_ns,
                        "cpu_duration_ns":time.process_time_ns()-run_cpu_start_ns,
                        "scope":"run() entry through backend/mission/config setup and first representative tick start"}
                loop_wall_start=time.perf_counter_ns(); loop_cpu_start=time.process_time_ns()
                current=backend.positions(); commands={}; attempts={}; delivered_ids=set()
                due_replans=([item for item in pending_replan_actions if item["effect_time_s"] <= t]
                             if not (skip_prepublic_first_step and j == 0) else [])
                pending_replan_actions[:]=[item for item in pending_replan_actions if item["effect_time_s"] > t]
                for item in due_replans:
                    chosen=item["selected_action"]
                    if chosen.get("kind") != "uav_transfer":
                        continue
                    donor,recipient,transfer_task = chosen["donor"],chosen["recipient"],chosen["task_id"]
                    stale_reason = None
                    if not transfer_task_is_unstarted(queues,cursor,donor,transfer_task):
                        stale_reason = "task_started_or_missing"
                    elif not actors[recipient].available or recipient in failed:
                        stale_reason = "recipient_failed"
                    elif recipient in communication_held or recipient in safety_paused:
                        stale_reason = "recipient_held"
                    elif (communication_outage_enabled and recipient == "uav-1-0"
                          and comm_outage_time <= t < comm_outage_time + comm_outage_duration):
                        stale_reason = "recipient_held"
                    elif activation.get(transfer_task, math.inf) > t:
                        stale_reason = "task_not_activated"
                    else:
                        endpoint = np.asarray(current[recipient]["position_m"],float)[:2]/1000.0
                        for q_index,q_task in enumerate(queues[recipient]):
                            if q_index < int(cursor[recipient][0]) or q_task in mission.completed:
                                continue
                            q_points = routes[recipient].get(q_task,[])
                            q_start = int(cursor[recipient][1]) if q_index == int(cursor[recipient][0]) else 0
                            if q_points[q_start:]: endpoint = np.asarray(q_points[-1],float)
                        task_points = routes[donor][transfer_task]
                        added_km = math.dist(endpoint,task_points[0]) + sum(
                            math.dist(a,b) for a,b in zip(task_points,task_points[1:]))
                        rate,_ = effective_energy_rate(cfg,recipient,actors[recipient].kind,t)
                        projected = remaining_route_energy(recipient,queues[recipient],t) + added_km*rate
                        reserve = energy_cap[recipient]*float(cfg["energy_reserve_fraction"])
                        if energy[recipient]-projected < reserve:
                            stale_reason = "recipient_energy_infeasible"
                    if stale_reason is None:
                        queues[donor].remove(transfer_task); queues[recipient].append(transfer_task)
                        routes[recipient][transfer_task] = routes[donor][transfer_task]
                        mission.assignments[transfer_task] = recipient
                        active_decision_by_actor[recipient] = item["decision_id"]
                        effect={"time_s":t,"branch":branch,"donor":donor,"recipient":recipient,
                                "task_id":transfer_task,"kind":"ordinary_event_replan",
                                "decision_id":item["decision_id"],"decision_created_sim_time_s":item["created_time_s"]}
                        receipt["donor_effects"].append(effect)
                        emit(ev,branch,t,"donor_support_transfer",decision_id=item["decision_id"],
                             from_aircraft_id=donor,to_aircraft_id=recipient,task_id=transfer_task,
                             support_kind="ordinary_event_replan")
                    else:
                        emit(ev,branch,t,"replan_action_rejected",decision_id=item["decision_id"],
                             task_id=transfer_task,from_aircraft_id=donor,to_aircraft_id=recipient,
                             reason=stale_reason,decision_created_sim_time_s=item["created_time_s"],
                             decision_complete_sim_time_s=item["effect_time_s"])
                do_prepublic = not (skip_prepublic_first_step and j == 0)
                if do_prepublic:
                    for task_id, activation_time in activation.items():
                        if activation_time == t:
                            emit(ev,branch,t,"task_activated",task_id=task_id,activation_sim_time_s=t)
                newly_available=([item for item in delayed_observations if item["available_at_sim_time_s"] <= t]
                                 if do_prepublic else [])
                if do_prepublic:
                    delayed_observations[:]=[item for item in delayed_observations if item["available_at_sim_time_s"] > t]
                for item in newly_available:
                    public_target_observations[item["target_id"]]=item
                    emit(ev,branch,t,"public_observation",target_id=item["target_id"],
                         detected_at_sim_time_s=item["detected_at_sim_time_s"],
                         available_at_sim_time_s=item["available_at_sim_time_s"],
                         observation_id=item["observation_id"])
                    if cfg["mode"] == "search_strike":
                        possible=[x for x in actors if actors[x].kind==item["kind"]
                                  and (actors[x].kind=="uav" or actors[x].home_zone==item["zone"])]
                        if possible:
                            service[sorted(possible)[0]].append(item["target_id"])
                anomaly=cfg.get("energy_anomaly") if mechanism_enabled(cfg, "energy_anomaly") else None
                if do_prepublic and anomaly is not None and t==int(anomaly["start_s"]):
                    emit(ev,branch,t,"energy_anomaly",aircraft_id=anomaly["aircraft_id"],
                         details={"rate_multiplier":float(anomaly["rate_multiplier"])})
                forced_outage_holds=set()
                if do_prepublic and branch == "representative":
                    if platform_fault_enabled and t == platform_fault_time:
                        fail_actor("uav-0-0", t, branch, ev)
                    for aid in actors:
                        if aid in failed:
                            continue
                        if energy[aid] <= energy_cap[aid]*float(cfg["energy_reserve_fraction"]):
                            position=current[aid]["position_m"]
                            active[aid]=[v/1000.0 for v in position]
                            pending[aid].clear()
                            if aid not in safety_paused:
                                safety_paused.add(aid)
                                emit(ev,branch,t,"energy_safety_hold",aircraft_id=aid,
                                     details={"remaining_wh":energy[aid],"pending_commands_cleared":True,
                                              "local_hold":True})
                        in_outage = (communication_outage_enabled and
                            comm_outage_time <= t < comm_outage_time + comm_outage_duration and
                            aid == "uav-1-0")
                        if in_outage:
                            if aid not in communication_held:
                                pending[aid].clear()
                                position=current[aid]["position_m"]
                                active[aid]=[v/1000.0 for v in position]
                                communication_held.add(aid)
                                emit(ev,branch,t,"communication_safety_hold",aircraft_id=aid,
                                     details={"reason":"forced_command_outage","pending_commands_cleared":True,
                                              "hold_at_current_position":True,"hold_position_m":position})
                            forced_outage_holds.add(aid)
                        elif aid in communication_held:
                            communication_held.remove(aid)
                            emit(ev,branch,t,"communication_safety_resume",aircraft_id=aid,
                                 details={"reason":"forced_command_outage_ended"})
                sample_events = list(ordinary_events_since_gate)
                forced_now = int(t) in forced_event_times
                if platform_fault_enabled and t == platform_fault_time and "uav-0-0" not in failed:
                    forced_now = True
                    decision_events.append("platform_fault")
                if (communication_outage_enabled and t == comm_outage_time
                        and "uav-1-0" not in communication_held):
                    forced_now = True
                    decision_events.append("communication_safety_hold")
                energy_emergency = any(aid not in safety_paused and energy[aid] <=
                    energy_cap[aid] * float(cfg["energy_reserve_fraction"]) for aid in actors)
                forced_now |= energy_emergency
                if energy_emergency:
                    decision_events.append("energy_safety_hold")
                decision_wall_start=time.perf_counter_ns()
                decision_cpu_start=time.process_time_ns()
                active_task_ids = {aid: ([queues[aid][cursor[aid][0]]] if cursor[aid][0] < len(queues[aid])
                    and activation.get(queues[aid][cursor[aid][0]], math.inf) <= t else []) for aid in actors}
                public_queues = {aid: [task_id for task_id in queue if activation.get(task_id, math.inf) <= t]
                                 for aid, queue in queues.items()}
                platform_status = {aid: {"kind": actors[aid].kind, "home_zone": int(actors[aid].home_zone),
                    "available": bool(actors[aid].available), "faulted": aid in failed,
                    "safety_paused": aid in safety_paused, "communication_held": aid in communication_held}
                    for aid in actors}
                projection = public_projection(t, current, energy, active_task_ids, public_queues,
                    [public_target_observations[x] for x in sorted(public_target_observations)], platform_status)
                if do_prepublic and branch == "representative":
                    ordinary_decision_sequence += 1
                    eligible_task_ids = {aid: {task_id for task_id in queue
                        if activation.get(task_id, math.inf) <= t} for aid, queue in queues.items()}
                    positions_km = {aid: [float(value) / 1000.0 for value in current[aid]["position_m"]]
                                    for aid in actors}
                    catalog_result = {}
                    def invoke_catalog():
                        result = build_public_replan_catalog(
                            queues, routes, cursor, actors, mission.completed, failed, energy,
                            {aid: energy_cap[aid] * float(cfg["energy_reserve_fraction"]) for aid in actors},
                            positions_km, max_speed_by_id,
                            {aid: effective_energy_rate(cfg, aid, actors[aid].kind, t)[0] for aid in actors},
                            float(cfg.get("pair_duration_s", 1800)), int(cfg["replan_catalog_limit"]),
                            eligible_task_ids=eligible_task_ids)
                        catalog_result["candidates"], catalog_result["selected"] = result
                        return result
                    decision = decision_adapter.decide(
                        backend, t, projection, list(decision_events), invoke_catalog,
                        readback=current)
                    decision_wall_ns=time.perf_counter_ns()-decision_wall_start
                    decision_cpu_ns=time.process_time_ns()-decision_cpu_start
                    counters=branch_counters.setdefault("decision_timing",{"wall_duration_ns":0,"cpu_duration_ns":0,"decision_count":0})
                    counters["wall_duration_ns"]+=decision_wall_ns
                    counters["cpu_duration_ns"]+=decision_cpu_ns
                    counters["decision_count"]+=1
                    decision_events.clear()
                    if forced_now:
                        handled_forced_event_times.add(int(t))
                    if decision["triggered"] or decision.get("gate_suppressed"):
                        ordinary_events_since_gate.clear()
                        decision_id = f"closedloop-{decision['mode']}-decision-{t}"
                        emit(ev, branch, t, "decision_gate", decision_id=decision_id,
                             mode=decision["mode"], trigger_reasons=decision["trigger_reasons"],
                             gate_suppressed=decision.get("gate_suppressed", False),
                             predicted_gain=decision.get("predicted_gain"), timing=decision["timing"])
                    if decision["triggered"] and not decision.get("gate_suppressed"):
                        if decision["planner_called"]:
                            candidates = catalog_result["candidates"]
                            selected = decision["selected_action"]
                            planning_wall = int(decision["timing"]["planner_wall_duration_ns"])
                            planning_cpu = int(decision["timing"]["planner_cpu_duration_ns"])
                            planning_delay_s = max(int(pilot_cfg.get("planning_delay_s", 1)),
                                                   int(math.ceil(decision_wall_ns / 1_000_000_000.0)))
                            emit(ev, branch, t, "candidate_generation", decision_id=decision_id,
                                 candidate_count=len(candidates), candidates=candidates,
                                 timing={"wall_duration_ns": planning_wall,"cpu_duration_ns": planning_cpu})
                            queue_replan_action(pending_replan_actions, decision_id, t, planning_delay_s, selected)
                            emit(ev, branch, t, "decision_created", decision_id=decision_id,
                                 decision_created_sim_time_s=t, decision_complete_sim_time_s=t+planning_delay_s,
                                 decision_effect_sim_time_s=t+planning_delay_s, selected_action=selected,
                                 timing={"planning":{"wall_duration_ns":planning_wall,
                                     "cpu_duration_ns":planning_cpu,"status":"measured"},
                                     "decision_wall_duration_ns":decision_wall_ns,
                                     "decision_cpu_duration_ns":decision_cpu_ns,
                                     "planning_delay_s":planning_delay_s,"gate_timing_scope":"partial_adapter"})
                if platform_fault_enabled and t==platform_fault_time:
                    fail_actor("uav-0-0",t,branch,ev)
                command_prep_wall_start=time.perf_counter_ns(); command_prep_cpu_start=time.process_time_ns()
                for aid in actors:
                    if aid in failed:
                        p=current[aid]["position_m"]; commands[aid]=[p[0]/1000,p[1]/1000,p[2]/1000]; continue
                    if energy[aid]<=energy_cap[aid]*float(cfg["energy_reserve_fraction"]):
                        p=current[aid]["position_m"]; active[aid]=[p[0]/1000,p[1]/1000,p[2]/1000]
                        pending[aid].clear(); commands[aid]=active[aid]
                        if aid not in safety_paused:
                            safety_paused.add(aid); emit(ev,branch,t,"energy_safety_hold",aircraft_id=aid,details={"remaining_wh":energy[aid],"pending_commands_cleared":True,"local_hold":True})
                        continue
                    forced=(communication_outage_enabled and comm_outage_time<=t
                            <comm_outage_time+comm_outage_duration and aid=="uav-1-0")
                    if forced:
                        if aid not in communication_held:
                            pending[aid].clear()
                            p=current[aid]["position_m"]
                            active[aid]=[p[0]/1000,p[1]/1000,p[2]/1000]
                            communication_held.add(aid)
                            emit(ev,branch,t,"communication_safety_hold",aircraft_id=aid,
                                 details={"reason":"forced_command_outage","pending_commands_cleared":True,
                                          "hold_at_current_position":True,"hold_position_m":p})
                        forced_outage_holds.add(aid)
                        commands[aid]=active[aid]
                        continue
                    if aid in communication_held:
                        communication_held.remove(aid)
                        emit(ev,branch,t,"communication_safety_resume",aircraft_id=aid,
                             details={"reason":"forced_command_outage_ended"})
                    due=[msg for msg in pending[aid] if msg[0]<=t]
                    pending[aid]=[msg for msg in pending[aid] if msg[0]>t]
                    if due:
                        delivered_ids.add(aid)
                        active[aid]=due[-1][1]
                        command_decision_id=due[-1][2] if len(due[-1]) > 2 else active_decision_by_actor.get(aid)
                        if command_decision_id is not None:
                            active_decision_by_actor[aid]=command_decision_id
                        emit(ev,branch,t,"command_delivered",aircraft_id=aid,decision_id=command_decision_id,
                             transmission_received_sim_time_s=t,details={"configured_delay_s":int(cfg["communication"]["delay_s"])})
                        emit(ev,branch,t,"command_applied",aircraft_id=aid,decision_id=command_decision_id,
                             command_applied_sim_time_s=t)
                    ti,wi=cursor[aid]; tid=queues[aid][ti] if ti<len(queues[aid]) and activation.get(queues[aid][ti], math.inf) <= t else None
                    desired=active[aid]
                    if service[aid] and service[aid][0] in public_target_observations:
                        target_id=service[aid][0]
                        observed=public_target_observations[target_id]
                        x,y=observed["target_xy_km"]; desired=[x,y,20.0]
                    elif tid is not None and wi<len(routes[aid][tid]):
                        x,y=routes[aid][tid][wi]; desired=[x,y,20.0]
                    if desired!=active[aid] and not pending[aid]:
                        forced=(communication_outage_enabled and comm_outage_time<=t
                                <comm_outage_time+comm_outage_duration and aid=="uav-1-0")
                        drop=forced or stable_drop(communication_seed,aid,t,float(cfg["communication"]["dropout_probability"]))
                        attempts[aid]=(not drop,forced)
                        if not drop:
                            command_decision_id=active_decision_by_actor.get(aid)
                            pending[aid].append((t+int(cfg["communication"]["delay_s"]),list(desired),command_decision_id))
                            emit(ev,branch,t,"transmission_sent",aircraft_id=aid,decision_id=command_decision_id,
                                 transmission_sent_sim_time_s=t,
                                 command_effect_sim_time_s=t+int(cfg["communication"]["delay_s"]))
                        else: emit(ev,branch,t,"command_dropped",aircraft_id=aid,details={"forced_outage":forced})
                    commands[aid]=active[aid]
                command_prep_wall_ns=time.perf_counter_ns()-command_prep_wall_start
                command_prep_cpu_ns=time.process_time_ns()-command_prep_cpu_start
                decision_phase_wall_ns=command_prep_wall_start-decision_wall_start
                decision_phase_cpu_ns=command_prep_cpu_start-decision_cpu_start
                timing_counters=branch_counters.setdefault("online_timing", {
                    "event_state_safety_wall_ns":0,"event_state_safety_cpu_ns":0,
                    "projection_policy_planning_wall_ns":0,"projection_policy_planning_cpu_ns":0,
                    "command_ready_wall_ns":0,"command_ready_cpu_ns":0,
                    "post_step_online_state_wall_ns":0,"post_step_online_state_cpu_ns":0,
                    "post_step_evaluator_telemetry_wall_ns":0,"post_step_evaluator_telemetry_cpu_ns":0,
                    "raw_artifact_logging_wall_ns":0,"raw_artifact_logging_cpu_ns":0,
                    "timing_event_write_wall_ns":0,"timing_event_write_cpu_ns":0,
                    "artifact_flush_wall_ns":0,"artifact_flush_cpu_ns":0,
                    "simulator_step_wall_ns":0,"simulator_step_cpu_ns":0,"tick_count":0})
                timing_counters["event_state_safety_wall_ns"]+=max(0,decision_wall_start-loop_wall_start)
                timing_counters["event_state_safety_cpu_ns"]+=max(0,decision_cpu_start-loop_cpu_start)
                timing_counters["projection_policy_planning_wall_ns"]+=decision_phase_wall_ns
                timing_counters["projection_policy_planning_cpu_ns"]+=decision_phase_cpu_ns
                timing_counters["command_ready_wall_ns"]+=command_prep_wall_ns
                timing_counters["command_ready_cpu_ns"]+=command_prep_cpu_ns
                timing_counters["tick_count"]+=1
                measured_ready_delay_s=max(int(pilot_cfg.get("planning_delay_s",1)),
                    int(math.ceil((decision_phase_wall_ns+command_prep_wall_ns)/1_000_000_000.0)))
                for queued in pending_replan_actions:
                    if queued.get("created_time_s")==t and queued.get("decision_id","").startswith("closedloop-"):
                        queued["effect_time_s"]=t+measured_ready_delay_s
                        emit(ev,branch,t,"decision_timing_finalized",decision_id=queued["decision_id"],
                             decision_complete_sim_time_s=queued["effect_time_s"],
                             decision_effect_sim_time_s=queued["effect_time_s"],
                             timing={"online_compute_wall_duration_ns":decision_phase_wall_ns+command_prep_wall_ns,
                                     "command_ready_delay_s":measured_ready_delay_s,"clock_quantum_s":1})
                sim_wall_start=time.perf_counter_ns(); sim_cpu_start=time.process_time_ns()
                readbacks=backend.step(commands,1.0)
                sim_step_wall_ns=time.perf_counter_ns()-sim_wall_start
                sim_step_cpu_ns=time.process_time_ns()-sim_cpu_start
                timing_counters["simulator_step_wall_ns"]+=sim_step_wall_ns
                timing_counters["simulator_step_cpu_ns"]+=sim_step_cpu_ns
                post_step_online_wall_start=time.perf_counter_ns(); post_step_online_cpu_start=time.process_time_ns()
                post_step_eval_wall_ns=0; post_step_eval_cpu_ns=0
                for aid,rb in readbacks.items():
                    pos=np.asarray(rb["position_m"],float)/1000.0; xy=(pos[0],pos[1]); previous_m=(last[aid]*1000.0).tolist(); dist=math.dist(last[aid],pos)
                    effective_rate, anomaly_active = effective_energy_rate(cfg,aid,actors[aid].kind,t)
                    use=dist*1000.0*effective_rate/1000.0
                    energy[aid]-=use; last[aid]=pos
                    ti,wi=cursor[aid]; tid=queues[aid][ti] if (ti<len(queues[aid])
                         and activation.get(queues[aid][ti], math.inf) <= t) else None
                    found=[]
                    if tid and aid not in failed:
                        before=set(mission.discovered); discover_at(mission,tasks[tid],xy,float(cfg["sensor_radius_km"]))
                        found=sorted(mission.discovered-before)
                    for target_id in found:
                        zone,(target_x,target_y)=mission.truth[target_id]
                        observation={"target_id":target_id,"zone":tasks[tid].zone,
                                     "target_xy_km":[target_x,target_y],
                                     "kind":mission.truth_kind[target_id],"observed_at_sim_time_s":t,
                                     "detected_at_sim_time_s":t,
                                     "available_at_sim_time_s":t+baseline_observation_delay_s+
                                         (light_delay_additional_s if mechanism_enabled(cfg, "light_delay")
                                          and aid == light_delay_platform_id else 0.0),
                                     "observation_id":f"obs-{branch}-{target_id}-{t}",
                                     "observation_source":"simulated_sensor_radius"}
                        delayed_observations.append(observation)
                        emit(ev,branch,t,"target_discovered",aircraft_id=aid,target_id=target_id,
                             task_id=tid,zone=observation["zone"],detected_at_sim_time_s=t,
                             available_at_sim_time_s=observation["available_at_sim_time_s"],
                             observation_id=observation["observation_id"])
                        if cfg["mode"]=="search_strike":
                            mission.event_log.append({"event":"strike_task_revealed","target_id":target_id,
                                                      "allowed_kind":observation["kind"],"time_s":t})
                    for target_id in list(service[aid]):
                        if abstract_strike(mission,target_id,actors[aid].kind,actors[aid].home_zone,xy,t,
                               float(cfg["abstract_strike_service_radius_km"]),float(cfg["abstract_strike_service_duration_s"])):
                            service[aid].remove(target_id); emit(ev,branch,t,"abstract_strike_complete",aircraft_id=aid,target_id=target_id)
                    visited=False; completed=False
                    if tid and wi<len(routes[aid][tid]) and aid not in failed:
                        if math.dist(xy,routes[aid][tid][wi])<=float(cfg["waypoint_tolerance_km"]): cursor[aid][1]+=1; visited=True
                        if cursor[aid][1]>=len(routes[aid][tid]):
                            mission.completed.add(tid); cursor[aid]=[ti+1,0]; completed=True
                            emit(ev,branch,t,"search_cell_complete",aircraft_id=aid,task_id=tid)
                            if cursor[aid][0] < len(queues[aid]): emit(ev,branch,t,"controller_switch",aircraft_id=aid,task_id=queues[aid][cursor[aid][0]])
                    eval_row_wall_start=time.perf_counter_ns(); eval_row_cpu_start=time.process_time_ns()
                    attempt=attempts.get(aid,(True,False))
                    pre_step_xy=(float(current[aid]["position_m"][0])/1000.0,
                                 float(current[aid]["position_m"][1])/1000.0)
                    active_speed_cap=spatial_speed_cap(pre_step_xy, float(t-1), speed_blocks)
                    speed_block_active=active_speed_cap is not None and active_speed_cap < max_speed_by_id[aid]
                    if speed_block_active:
                        branch_counters[branch]["speed_block_actor_steps"] += 1
                    row={"schema_version":3,"run_id":cfg["run_id"],"phase":phase,"branch":branch,"seed":seed,
                         "sim_time_s":t,"dt_s":1.0,"kit_frame":rb["kit_frame"],"aircraft_id":aid,"aircraft_type":actors[aid].kind,
                         "zone":actors[aid].home_zone,"task_id":tid,"search_cell_visited":visited,"task_completed":completed,"faulted":aid in failed,
                         "target_discovered_ids":found,
                         "detected_target_ids":found,
                         "mission_xy_km":[float(xy[0]),float(xy[1])],"usd_position_m":rb["position_m"],
                         "public_target_observations":[public_target_observations[x] for x in sorted(public_target_observations)],
                         "public_observation_ids":sorted(public_target_observations),
                         "baseline_max_speed_mps":float(max_speed_by_id[aid]),
                         "speed_block_definitions":speed_blocks,
                         "speed_cap_active":speed_block_active,
                         "speed_cap_mps":float(active_speed_cap) if active_speed_cap is not None else float(max_speed_by_id[aid]),
                         "previous_usd_position_m":previous_m,
                         "velocity_mps":rb["velocity_mps"],"usd_pose_error_m":rb["pose_error_m"],
                         "energy":{"status":"analytical_linear","incremental_wh":use,"remaining_wh":max(0.0,energy[aid]),
                                   "effective_wh_per_km":effective_rate,"anomaly_active":anomaly_active,
                                   "safety_pause":energy[aid]<=energy_cap[aid]*float(cfg["energy_reserve_fraction"])},
                         "communication":{"status":"simulated_command_queue","command_delivered":aid in delivered_ids,
                                           "command_dropped":not attempt[0],"forced_outage":aid in forced_outage_holds or attempt[1],
                                           "safety_hold":aid in communication_held or aid in forced_outage_holds,
                                           "command_attempt_accepted":attempt[0],"command_attempted":aid in attempts,
                                           "delay_s":int(cfg["communication"]["delay_s"])},
                         "hidden_truth_logged":False}
                    pending_trajectory_rows.append(row); rows+=1
                    counter=branch_counters[branch]; counter["rows"]+=1; counter["energy_wh"]+=use
                    counter["command_attempts"]+=int(aid in attempts)
                    post_step_eval_wall_ns+=time.perf_counter_ns()-eval_row_wall_start
                    post_step_eval_cpu_ns+=time.process_time_ns()-eval_row_cpu_start
                post_step_total_wall_ns=time.perf_counter_ns()-post_step_online_wall_start
                post_step_total_cpu_ns=time.process_time_ns()-post_step_online_cpu_start
                post_step_online_wall_ns=max(0,post_step_total_wall_ns-post_step_eval_wall_ns)
                post_step_online_cpu_ns=max(0,post_step_total_cpu_ns-post_step_eval_cpu_ns)
                compact_rows = list(pending_trajectory_rows) if compact_writer is not None and branch == "representative" else None
                compact_events_before_timing = list(pending_event_rows) if compact_rows is not None else None
                raw_log_wall_start=time.perf_counter_ns(); raw_log_cpu_start=time.process_time_ns()
                traj.writelines(json.dumps(row,separators=(",",":"))+"\n" for row in pending_trajectory_rows)
                pending_trajectory_rows.clear()
                ev.writelines(json.dumps(row,separators=(",",":"))+"\n" for row in pending_event_rows)
                pending_event_rows.clear()
                raw_log_wall_ns=time.perf_counter_ns()-raw_log_wall_start
                raw_log_cpu_ns=time.process_time_ns()-raw_log_cpu_start
                emit(ev,branch,t,"online_timing_tick",timing={
                    "phases":{
                        "event_state_safety":{"wall_duration_ns":max(0,decision_wall_start-loop_wall_start),
                                               "cpu_duration_ns":max(0,decision_cpu_start-loop_cpu_start)},
                        "projection_policy_planning":{"wall_duration_ns":decision_phase_wall_ns,
                                                       "cpu_duration_ns":decision_phase_cpu_ns},
                        "command_ready":{"wall_duration_ns":command_prep_wall_ns,
                                         "cpu_duration_ns":command_prep_cpu_ns},
                        "post_step_online_state":{"wall_duration_ns":post_step_online_wall_ns,
                                                   "cpu_duration_ns":post_step_online_cpu_ns},
                        "post_step_evaluator_telemetry":{"wall_duration_ns":post_step_eval_wall_ns,
                                                          "cpu_duration_ns":post_step_eval_cpu_ns},
                        "simulator_step":{"wall_duration_ns":sim_step_wall_ns,
                                           "cpu_duration_ns":sim_step_cpu_ns},
                        "raw_artifact_logging":{"wall_duration_ns":raw_log_wall_ns,
                                                 "cpu_duration_ns":raw_log_cpu_ns}}})
                timing_event_row=pending_event_rows.pop()
                timing_event_wall_start=time.perf_counter_ns(); timing_event_cpu_start=time.process_time_ns()
                ev.write(json.dumps(timing_event_row,separators=(",",":"))+"\n")
                timing_event_write_wall_ns=time.perf_counter_ns()-timing_event_wall_start
                timing_event_write_cpu_ns=time.process_time_ns()-timing_event_cpu_start
                if compact_rows is not None:
                    compact_state, compact_events, compact_decisions, compact_costs, compact_events_complete = compact_tick(
                        compact_rows, tasks, queues, cursor, activation, routes,
                        mission.completed, t, compact_events_before_timing + [timing_event_row],
                        {aid: {"available": actor.available, "safety_paused": aid in safety_paused}
                         for aid, actor in actors.items()}, mission.assignments)
                    compact_writer.append_tick(compact_state, events=compact_events,
                        decisions=compact_decisions, costs=compact_costs,
                        event_log_complete=compact_events_complete,
                        hidden_truth=({"targets": [{"target_id": target_id, "zone": zone,
                            "target_xy_km": list(xy), "kind": mission.truth_kind[target_id]}
                            for target_id, (zone, xy) in sorted(mission.truth.items())]}
                            if t == 1 else None))
                    append_evaluation_audit(compact_capture_dir, t, compact_rows,
                        compact_events_before_timing + [timing_event_row])
                timing_counters=branch_counters[branch]["online_timing"]
                timing_counters["post_step_online_state_wall_ns"]+=post_step_online_wall_ns
                timing_counters["post_step_online_state_cpu_ns"]+=post_step_online_cpu_ns
                timing_counters["post_step_evaluator_telemetry_wall_ns"]+=post_step_eval_wall_ns
                timing_counters["post_step_evaluator_telemetry_cpu_ns"]+=post_step_eval_cpu_ns
                timing_counters["raw_artifact_logging_wall_ns"]+=raw_log_wall_ns
                timing_counters["raw_artifact_logging_cpu_ns"]+=raw_log_cpu_ns
                timing_counters["timing_event_write_wall_ns"]+=timing_event_write_wall_ns
                timing_counters["timing_event_write_cpu_ns"]+=timing_event_write_cpu_ns
                if j%60==59:
                    flush_wall_start=time.perf_counter_ns(); flush_cpu_start=time.process_time_ns()
                    traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
                    timing_counters["artifact_flush_wall_ns"]+=time.perf_counter_ns()-flush_wall_start
                    timing_counters["artifact_flush_cpu_ns"]+=time.process_time_ns()-flush_cpu_start
                    receipt["stages"].append({"branch":branch,"sim_time_s":t,"kit_frame":next(iter(readbacks.values()))["kit_frame"],"rows_written":rows})
                    persist(rec_path,receipt)
            return rows
        with gzip.open(traj_path,"wt",encoding="utf-8",compresslevel=1,newline="\n") as traj, events_path.open("w",encoding="utf-8",buffering=1) as ev:
            if resume_state is not None:
                state_time = int(suffix_meta["sim_time_s"])
                backend_state, common = copy.deepcopy(resume_state)
                expected_digest = snapshot_digest((backend_state, common))
                indexed_backend_time = float(suffix_meta["backend_time_s"])
                saved_backend_time = float(backend_state["scene_time_s"])
                expected_backend_time = state_time - 1
                if (abs(indexed_backend_time - expected_backend_time) > 1e-9
                        or abs(saved_backend_time - indexed_backend_time) > 1e-9):
                    raise ValueError("snapshot backend time must equal indexed time t-1")
                horizon = suffix_horizon(state_time)
                if horizon <= 0:
                    raise ValueError(f"state time {state_time} leaves no suffix horizon")

                def restore_and_check():
                    backend.restore(backend_state)
                    restore_common(common)
                    actual = (backend.snapshot(), common_state())
                    digest = snapshot_digest(actual)
                    if digest != expected_digest:
                        raise RuntimeError(f"restored state hash mismatch: {digest} != {expected_digest}")
                    return digest

                # Every branch starts from the exact pre-candidate, pre-step state.
                keep_restore_hash = restore_and_check()
                initial_mission_state = copy.deepcopy(common[-1])
                truth_target_ids = set(mission.truth)
                public_task_ids = {task.task_id for task in mission.tasks}
                initial_completed = set(initial_mission_state["completed"])
                initial_remaining_public_task_ids = sorted(
                    task_id for task_id in public_task_ids
                    if task_id not in initial_completed and activation.get(task_id, math.inf) <= state_time)
                audit_interface = {
                    "state_id": suffix_meta["state_id"],
                    "state_kind": "ordinary_sampled_decision_state",
                    "forced": False,
                    "snapshot_position": "restored snapshot before keep/replan branches",
                    "initial_discovered_target_ids": sorted(set(initial_mission_state["discovered"])),
                    "initial_serviced_target_ids": sorted(initial_completed & truth_target_ids),
                    "initial_completed_public_task_ids": sorted(initial_completed & public_task_ids),
                    "initial_remaining_public_task_ids": initial_remaining_public_task_ids,
                }
                keep_start_backend_time = float(backend.scene.tactical_time_s)
                keep_rows = segment("keep", horizon, ev, state_time,
                                     skip_prepublic_first_step=True)
                keep_end_backend_time = float(backend.scene.tactical_time_s)

                replan_restore_hash = restore_and_check()
                replan_start_backend_time = float(backend.scene.tactical_time_s)
                positions = backend.positions()
                current_km = {aid: [v / 1000.0 for v in positions[aid]["position_m"]]
                              for aid in actors}
                energy_reserve = {aid: energy_cap[aid] * float(cfg["energy_reserve_fraction"])
                                  for aid in actors}
                eligible = {aid: {tid for tid in queue if activation.get(tid, math.inf) <= state_time}
                            for aid, queue in queues.items()}
                wall0 = time.perf_counter_ns(); cpu0 = time.process_time_ns()
                candidates, choice = build_public_replan_catalog(
                    queues, routes, cursor, actors, mission.completed, failed, energy,
                    energy_reserve, current_km, max_speed_by_id,
                    {aid: effective_energy_rate(cfg, aid, actors[aid].kind, state_time)[0]
                     for aid in actors}, horizon, int(cfg["replan_catalog_limit"]),
                    eligible_task_ids=eligible)
                catalog_timing = {"wall_duration_ns": time.perf_counter_ns() - wall0,
                                  "cpu_duration_ns": time.process_time_ns() - cpu0}
                planning_delay = max(1, int(math.ceil(catalog_timing["wall_duration_ns"] / 1e9)))
                emit(ev, "replan", state_time, PLANNER_CALL_EVENT,
                     decision_id=f"replan-decision-{state_time}",
                     candidate_count=len(candidates), candidates=candidates, timing=catalog_timing)
                replan_rows = segment("replan", horizon, ev, state_time, candidates,
                                      planning_delay, choice, skip_prepublic_first_step=True)
                replan_end_backend_time = float(backend.scene.tactical_time_s)
                receipt.update({"status": "complete", "actual_isaac_run": True,
                    "suffix": {**suffix_meta, "seed": int(suffix_seed), "horizon_steps": horizon,
                        "decision_time_s": state_time, "snapshot_backend_time_s": float(backend_state["scene_time_s"]),
                        "keep_restore_sha256": keep_restore_hash,
                        "replan_restore_sha256": replan_restore_hash,
                        "snapshot_sha256": expected_digest,
                        "audit_interface": audit_interface,
                        "keep_rows_written": keep_rows, "replan_rows_written": replan_rows,
                        "keep_backend_time_start_s": keep_start_backend_time,
                        "keep_backend_time_end_s": keep_end_backend_time,
                        "replan_backend_time_start_s": replan_start_backend_time,
                        "replan_backend_time_end_s": replan_end_backend_time,
                        "catalog_timing": catalog_timing, "planning_delay_s": planning_delay,
                        "selected_action": choice,
                        "conditioned_randomness": "only future stable_drop communication draws use suffix seed; all other disturbance schedules remain frozen by sim time"},
                    "rows_written": keep_rows + replan_rows,
                    "simulated_steps_by_branch": {"keep": keep_rows // len(actors),
                                                   "replan": replan_rows // len(actors)},
                    "events": events_path.name, "trajectory": traj_path.name,
                    "raw_physical_trace": True, "scalar_status": "NOT_COMPUTED"})
                traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
            else:
                for task_id, activation_time in activation.items():
                    if activation_time == 0:
                        emit(ev, "representative" if phase == "representative" else "pair_prefix", 0,
                             "task_activated", task_id=task_id, activation_sim_time_s=0)
                if phase=="representative":
                    total_rows=segment("representative",int(cfg["representative_duration_s"]),ev)
                    seconds={"representative":int(cfg["representative_duration_s"])}
                else:
                    prefix=int(cfg["pair_snapshot_time_s"])
                    prefix_rows=segment("pair_prefix",prefix,ev)
                    # The paired decision starts from the persisted prefix state; an
                    # unfinished ordinary planner action must not leak into keep.
                    pending_replan_actions.clear()
                    pair_backend=backend.snapshot(); pair_common=common_state()
                    energy_reserve={aid:energy_cap[aid]*float(cfg["energy_reserve_fraction"]) for aid in actors}
                    pair_start_positions={aid:rb["position_m"] for aid,rb in backend.positions().items()}
                    current_positions_km={aid:[p[0]/1000.0,p[1]/1000.0,p[2]/1000.0]
                                          for aid,p in pair_start_positions.items()}
                    catalog_wall_start=time.perf_counter_ns(); catalog_cpu_start=time.process_time_ns()
                    energy_rate_by_id={aid:effective_energy_rate(cfg,aid,actors[aid].kind,prefix)[0] for aid in actors}
                    pair_eligible={aid:{task_id for task_id in queue if activation.get(task_id, math.inf) <= prefix}
                                   for aid,queue in queues.items()}
                    candidates,choice=build_public_replan_catalog(queues,routes,cursor,actors,mission.completed,
                        failed,energy,energy_reserve,current_positions_km,max_speed_by_id,energy_rate_by_id,
                        int(cfg["pair_duration_s"]),int(cfg["replan_catalog_limit"]),
                        eligible_task_ids=pair_eligible)
                    catalog_timing={"wall_duration_ns":time.perf_counter_ns()-catalog_wall_start,
                                    "cpu_duration_ns":time.process_time_ns()-catalog_cpu_start}
                    pair_planning_delay_s=max(1,int(math.ceil(catalog_timing["wall_duration_ns"]/1_000_000_000.0)))
                    emit(ev,"replan",prefix,PLANNER_CALL_EVENT,decision_id=f"replan-decision-{prefix}",
                         decision_created_sim_time_s=prefix,candidate_count=len(candidates),candidates=candidates,
                         timing=catalog_timing)
                    receipt["pair_snapshot"]={"sim_time_s":prefix,"ordinary_decision":"post-1Hz observation/task queue state",
                         "backend_snapshot":True,"mission_queues_energy_comms":True,"frozen_candidate_catalog":candidates}
                    saved_full_state=(pair_backend,pair_common)
                    pair_state_digest=snapshot_digest(saved_full_state)
                    receipt["pair_snapshot"].update({"state_sha256":pair_state_digest,"positions_m":pair_start_positions,
                        "selected_candidate":choice,"catalog_policy":"max public ETA reduction minus 30s switch cost; NOOP included",
                        "catalog_deadline":"not checked; horizon is metadata only, candidate ETAs are not deadline-feasibility claims"})
                    keep_rows=segment("keep",int(cfg["pair_duration_s"]),ev,prefix,[{"kind":"noop","estimated_gain_s":0.0}])
                    backend.restore(pair_backend); restore_common(pair_common)
                    restored_positions={aid:rb["position_m"] for aid,rb in backend.positions().items()}
                    max_restore_error=max(math.dist(pair_start_positions[aid],restored_positions[aid]) for aid in actors)
                    restored_full_state=(backend.snapshot(),common_state())
                    restored_state_digest=snapshot_digest(restored_full_state)
                    receipt["pair_snapshot"]["restore_max_position_error_m"]=max_restore_error
                    receipt["pair_snapshot"]["restore_state_match"]=max_restore_error<=0.02
                    receipt["pair_snapshot"]["restored_state_sha256"]=restored_state_digest
                    receipt["pair_snapshot"]["restore_full_state_match"]=pair_state_digest==restored_state_digest
                    replan_rows=segment("replan",int(cfg["pair_duration_s"]),ev,prefix,candidates,
                                        pair_planning_delay_s,choice)
                    total_rows=prefix_rows+keep_rows+replan_rows
                    seconds={"pair_prefix":prefix,"keep":int(cfg["pair_duration_s"]),"replan":int(cfg["pair_duration_s"])}
            traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
        if resume_state is not None:
            persist(manifest_path, {"run_id": cfg["run_id"],
                "config_sha256": receipt["config_sha256"], "source_hashes": source_hashes,
                "logs": {"trajectory": traj_path.name, "events": events_path.name},
                "suffix_seed": int(suffix_seed), "source_solver_reused": False})
            persist(rec_path, receipt)
            return 0

        decision_costs = branch_counters.pop("decision_timing", {
            "wall_duration_ns": 0, "cpu_duration_ns": 0, "decision_count": 0})
        online_timing = {key: branch_counters.pop(key) for key in list(branch_counters)
                         if isinstance(branch_counters.get(key), dict) and "event_state_safety_wall_ns" in branch_counters[key]}
        receipt["closedloop"]["decision_costs"] = {
            **decision_costs,
            "scope": "projection through policy and planning adapter; this nested diagnostic is not added to phase totals",
            "simulator_clock_quantum_s": 1,
        }
        receipt["closedloop"]["timing"] = {
            "online_by_branch": {"representative": online_timing.get("online_timing")} if online_timing else {},
            "online_total_rule": "sum of event_state_safety, projection_policy_planning, command_ready, and post_step_online_state; evaluator telemetry, raw artifact logging, timing-record writes, artifact flushes, and simulator step are separate",
            "effect_delay_rule": "new replans use measured projection_policy_planning wall plus command_ready wall, rounded up to the 1-second simulator clock and bounded by configured minimum; absent phase costs are unknown",
            "startup_load": receipt["closedloop"].get("timing", {}).get("startup_load", {"status":"unknown"}),
            "post_step_scope": "post_step_online_state measures energy, detection, observation scheduling, and mission state updates; post_step_evaluator_telemetry measures row materialization and runtime summary counters; raw_artifact_logging measures JSON serialization and trajectory/event writes",
            "timing_record_write_scope": "online_timing_tick record writes are timed separately and included in receipt aggregates",
            "artifact_flush_scope": "periodic gzip/event flush and fsync wall/CPU are aggregated separately",
            "simulator_clock_quantum_s": 1,
            "millisecond_responsiveness_claim": False,
        }
        def public_metrics(branch):
            m=branch_counters.get(branch,{})
            public_task_ids={task.task_id for task in mission.tasks}
            window_complete=set(m.get("completed",set())) & public_task_ids
            window_discovered=set(m.get("discovered",set()))
            window_serviced=set(m.get("serviced",set()))
            prefix_complete=set(branch_counters.get("pair_prefix",{}).get("completed",set())) if branch in {"keep","replan"} else set()
            prefix_discovered=set(branch_counters.get("pair_prefix",{}).get("discovered",set())) if branch in {"keep","replan"} else set()
            prefix_serviced=set(branch_counters.get("pair_prefix",{}).get("serviced",set())) if branch in {"keep","replan"} else set()
            terminal_complete=prefix_complete|window_complete
            terminal_discovered=prefix_discovered|window_discovered
            terminal_serviced=prefix_serviced|window_serviced
            return {"public_task_denominator":len(mission.tasks),
                "window_new_completed_public_tasks":len(window_complete),
                "cumulative_completed_public_tasks":len(terminal_complete),
                "terminal_remaining_public_tasks":len(mission.tasks)-len(terminal_complete),
                "window_new_discovered_targets":len(window_discovered),
                "cumulative_discovered_targets":len(terminal_discovered),
                "window_new_serviced_targets":len(window_serviced),
                "cumulative_serviced_targets":len(terminal_serviced),
                 "analytical_energy_kwh":m.get("energy_wh",0.0)/1000.0,"command_attempts":m.get("command_attempts",0),
                 "switches":m.get("switches",0),"planner_calls":m.get("planner_calls",0),"violations":m.get("violations",0),
                 "faulted_aircraft_count":len(m.get("faulted",set())),"speed_block_actor_steps":m.get("speed_block_actor_steps",0),
                 "row_count":m.get("rows",0),"actor_count":36}
        metrics={branch:public_metrics(branch) for branch in seconds}
        pair_summary=None
        if "keep" in metrics and "replan" in metrics:
            keys=("window_new_completed_public_tasks","window_new_discovered_targets","window_new_serviced_targets","analytical_energy_kwh",
                  "command_attempts","switches","planner_calls","violations","faulted_aircraft_count")
            pair_summary={"gain_keep_minus_replan":{f"{key}_keep_minus_replan":metrics["keep"][key]-metrics["replan"][key] for key in keys},
                          "scalar_status":"NOT_COMPUTED","scalar_reason":"no independent full-episode cost reducer or frozen total-cost evaluation is implemented; no scalar J claim"}
        receipt.update({"status":"complete","rows_written":total_rows,"simulated_seconds_by_branch":seconds,
                       "branch_metrics":metrics,"pair_summary":pair_summary,
                    "independent_audit":"pending_external_audit","scalar_status":"INDEPENDENT_EVALUATOR_REQUIRED",
                    "events":events_path.name,"trajectory":traj_path.name})
        persist(manifest_path,{"run_id":cfg["run_id"],"config_sha256":receipt["config_sha256"],
                               "closedloop":receipt["closedloop"],"source_hashes":source_hashes,
                               "evaluation_config":cfg,
                               "initial_positions_m":receipt.get("initial_positions_m"),
                               "actor_manifest":receipt.get("actor_manifest"),
                               "zone_bounds_km":receipt.get("zone_bounds_km"),
                               "public_task_manifest":receipt.get("public_task_manifest"),
                               "evaluation_truth_file":truth_path.name,
                               "evaluator_truth":{"file":truth_path.name,"sha256":sha(truth_path)},
                               "logs":{"trajectory":traj_path.name,"events":events_path.name},
                               "runtime_gpu_request":cfg["runtime"],"source_solver_reused":False})
        persist(rec_path,receipt)
        return 0
    except Exception as exc:
        receipt.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"}); persist(rec_path,receipt); return 2
    finally:
        if backend is not None:
            close_wall_start=time.perf_counter_ns(); close_cpu_start=time.process_time_ns()
            backend.close()
            receipt["closedloop"].setdefault("timing", {})["backend_close"]={
                "wall_duration_ns":time.perf_counter_ns()-close_wall_start,
                "cpu_duration_ns":time.process_time_ns()-close_cpu_start,
                "status":"measured"}
            backend=None
        receipt["resource_usage"].update({
            "wall_duration_s": (time.perf_counter_ns()-run_wall_start_ns)/1e9,
            "process_cpu_s": (time.process_time_ns()-run_cpu_start_ns)/1e9,
            "max_rss_bytes": max_rss_bytes(),
            "gpu_worker_count": 1,
        })
        receipt["backend_close_pending"]=False
        try:
            persist(rec_path,receipt)
        except OSError:
            print(f"could not finalize receipt because storage is unavailable: {rec_path}", file=sys.stderr)
        if compact_writer is not None:
            context_path = compact_capture_dir / "evaluation_context.json"
            try:
                context = json.loads(context_path.read_text(encoding="utf-8"))
                context["runtime_receipt"] = copy.deepcopy(receipt)
                write_evaluation_context(compact_capture_dir, context)
            except OSError as exc:
                print(f"could not finalize compact evaluation context: {exc}", file=sys.stderr)

def main():
    p=argparse.ArgumentParser(description="Run one 90-minute closed-loop ROI Isaac episode.")
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/a1_representative.json")
    p.add_argument("--source-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=sorted(DecisionAdapter.MODES),default="existing_et")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--model",choices=("constant","ridge","mlp","tree"),default=None,
                   help="optional artifact type assertion; type is otherwise read from the artifact")
    p.add_argument("--deployment-mode", choices=("mean", "lower_bound"), default="mean")
    p.add_argument("--compact-capture", action="store_true",
                   help="write a separate compact public per-second capture")
    a=p.parse_args()
    predictor = None
    gain_threshold = 0.0
    deployment_metadata = None
    checkpoint_model = "mlp"
    if a.mode.endswith("_gain_gate"):
        if a.checkpoint is None:
            p.error("gain-gated modes require --checkpoint pointing to a trained model checkpoint")
        try:
            from . import checkpoint as checkpoint_loader
        except ImportError:
            import checkpoint as checkpoint_loader
        try:
            predictor, deployment_metadata = checkpoint_loader.load_gain_artifact(
                a.checkpoint, mode=a.deployment_mode)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            p.error(f"could not load deployment artifact: {exc}")
        checkpoint_model = str(deployment_metadata["model_kind"])
        if a.model is not None and a.model != checkpoint_model:
            p.error(f"--model={a.model} conflicts with artifact model type {checkpoint_model}")
        gain_threshold = float(deployment_metadata["epsilon"])
    elif a.checkpoint is not None:
        p.error("--checkpoint is only valid for a gain-gated mode")
    elif a.model is not None:
        p.error("--model is only valid with a gain-gated mode and --checkpoint")
    rc=run(a.config,a.source_root,a.output,strategy_mode=a.mode,
           gain_predictor=predictor,checkpoint_path=a.checkpoint,checkpoint_model=checkpoint_model,
           gain_threshold=gain_threshold,deployment_metadata=deployment_metadata,
           enable_compact_capture=a.compact_capture)
    raise SystemExit(rc)

if __name__=="__main__": main()
