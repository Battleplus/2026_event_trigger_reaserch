#!/usr/bin/env python3
"""A1 90-minute Isaac run or 30-minute snapshot-paired keep/replan run."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, sys
from pathlib import Path
import numpy as np
from isaac_backend import IsaacBackend
from mission import (MissionState, abstract_strike, assign_cross_zone_support, build_mission,
                     build_public_replan_catalog, discover_at, initial_c1_greedy, mission_snapshot,
                     restore_mission, roi_zones, task_sweep)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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

def run(config_path, source_root, output, phase):
    config_path=Path(config_path); source_root=Path(source_root); output=Path(output)
    cfg=json.loads(config_path.read_text(encoding="utf-8")); output.mkdir(parents=True,exist_ok=True)
    rec_path=output/"run_receipt.json"; manifest_path=output/"manifest.json"
    traj_path=output/"trajectory.jsonl"; events_path=output/"events.jsonl"
    receipt={"run_id":cfg["run_id"],"phase":phase,"status":"initializing","actual_isaac_run":False,
             "config_sha256":sha(config_path),"source_archive_sha256":cfg["source_archive_sha256"],
             "energy_model":"linear distance Wh/km, analytical; not physical measurement",
             "communication_model":"time-addressed command delay/drop model; not radio physics",
             "hidden_truth_logged":False,"source_solver_reused":False,"stages":[],"donor_effects":[]}
    persist(rec_path,receipt); backend=None
    try:
        seed=int(cfg["seed"]); planner_rng=random.Random(seed)
        center=(planner_rng.uniform(-125,125),planner_rng.uniform(-125,125))
        mission=build_mission(seed,center,int(cfg["targets_per_zone"]),cfg["mode"])
        truth_path=output/"evaluator_truth.json"
        persist(truth_path,{"targets":[{"target_id":target_id,"zone":zone,"target_xy_km":list(xy),
                                        "kind":mission.truth_kind[target_id]}
                           for target_id,(zone,xy) in sorted(mission.truth.items())]})
        public_mission=MissionState(tasks=mission.tasks,aircraft=mission.aircraft,mode=mission.mode)
        assignment=initial_c1_greedy(public_mission)
        mission.assignments=dict(assignment)
        actors={a.aircraft_id:a for a in mission.aircraft}; tasks={t.task_id:t for t in mission.tasks}
        queues={aid:[] for aid in actors}
        for task_id,aid in assignment.items(): queues[aid].append(task_id)
        records=[]
        for a in mission.aircraft:
            cx,cy=a.public_xy_km; ix=int(a.aircraft_id.rsplit("-",1)[1]); ox=(-.45,.45)[ix]
            records.append({"aircraft_id":a.aircraft_id,"kind":a.kind,"home_zone":a.home_zone,
                            "initial_position_km":[cx+ox,cy+ox,20.0],"max_speed_mps":65.0 if a.kind=="uav" else 100.0,
                            "terrain_clearance_m":120.0})
        runtime=dict(cfg["runtime"]); runtime.update(cfg["scene"])
        source_files=("scenes/air_combat_scene.py","scenes/air_combat_specs.py","scenes/aircraft_motion.py",
                      "sensors/air_combat_specs.py","sensors/air_combat_obstacles.py","weapons/air_combat_specs.py")
        source_hashes={name:sha(source_root/name) for name in source_files}
        backend=IsaacBackend(source_root,records,runtime)
        rb0=backend.positions()
        max_speed_by_id={r["aircraft_id"]:r["max_speed_mps"] for r in records}
        receipt.update({"status":"scene_ready","actual_isaac_run":True,"map_size_km":300.0,"meters_per_unit":100.0,
                       "roi_center_km":list(center),"roi_width_km":cfg["roi_width_km"],"aircraft_count":36,
                       "aircraft_per_zone":{"uav":2,"helicopter":2},"hidden_target_count":len(mission.truth),
                       "public_search_task_count":len(mission.tasks),"initial_assignment_solver":"c1_greedy_proxy",
                       "c1_source_solver_reused":False,"c1_tie_break":"same-zone capable aircraft, minimum load, nearest cell, id",
                       "hidden_truth_exposed_to_planner":False,"initial_usd_pose_error_max_m":max(x["pose_error_m"] for x in rb0.values()),
                       "initial_positions_m":{aid:x["position_m"] for aid,x in rb0.items()},
                       "expected_task_ids":[t.task_id for t in mission.tasks],
                       "zone_bounds_km":{str(z["zone"]):[z["xmin"],z["ymin"],z["xmax"],z["ymax"]]
                                         for z in roi_zones(center,cfg["roi_width_km"])},
                       "sensor_radius_km":float(cfg["sensor_radius_km"]),
                       "waypoint_tolerance_km":float(cfg["waypoint_tolerance_km"]),
                       "target_discovery_predicate":"active task zone equals evaluator target zone and 2D Euclidean distance from mission_xy_km to target_xy_km is <= sensor_radius_km",
                       "search_cell_completion_predicate":"each configured sweep waypoint is visited within waypoint_tolerance_km; cell completes after all waypoints are visited",
                       "evaluator_truth_file":"evaluator_truth.json","evaluator_truth_sha256":sha(truth_path),
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
        def remaining_route_energy(aid, queue):
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
            return total_km*float(cfg["energy_wh_per_km"][actors[aid].kind])
        def emit(h,branch,t,event,**fields):
            row={"schema_version":1,"run_id":cfg["run_id"],"phase":phase,"branch":branch,
                 "sim_time_s":t,"event":event,"event_id":f"{branch}:{t}:{event}:{fields.get('aircraft_id','')}:{fields.get('task_id',fields.get('target_id',''))}",**fields}
            h.write(json.dumps(row,separators=(",",":"))+"\n")
            m=branch_counters.setdefault(branch,{"completed":set(),"discovered":set(),"serviced":set(),
                 "energy_wh":0.0,"command_attempts":0,"switches":0,"planner_calls":0,"violations":0,"faulted":set(),"rows":0})
            if event=="search_cell_complete":m["completed"].add(fields.get("task_id"))
            elif event=="target_discovered":m["discovered"].add(fields.get("target_id"))
            elif event=="abstract_strike_complete":m["serviced"].add(fields.get("target_id"))
            elif event=="controller_switch":m["switches"]+=1
            elif event=="planner_call" or event=="replan_catalog_choice":m["planner_calls"]+=1
            elif event in {"safety_violation","boundary_violation","constraint_violation"}:m["violations"]+=1
            elif event=="platform_fault":m["faulted"].add(fields.get("aircraft_id"))
        def common_state():
            return copy.deepcopy((queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,
                                  communication_held,public_target_observations,mission_snapshot(mission)))
        def restore_common(snapshot):
            nonlocal queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused
            nonlocal communication_held,public_target_observations
            (queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,
             communication_held,public_target_observations,ms)=copy.deepcopy(snapshot)
            restore_mission(mission,ms)
        def fail_actor(aid,t,branch,ev):
            if aid in failed:return
            failed.add(aid); actors[aid].available=False; pending[aid].clear()
            current=backend.positions()[aid]["position_m"]; active[aid]=[v/1000 for v in current]
            rest=queues[aid][cursor[aid][0]:]; queues[aid]=queues[aid][:cursor[aid][0]]
            for tid in rest:
                choices=[]
                for recipient,actor in actors.items():
                    if actor.kind!=tasks[tid].allowed_kind or not actor.available:
                        continue
                    current_energy=remaining_route_energy(recipient,queues[recipient])
                    endpoint=np.asarray(backend.positions()[recipient]["position_m"],float)[:2]/1000.0
                    for q_index,q_task in enumerate(queues[recipient]):
                        if q_index<int(cursor[recipient][0]) or q_task in mission.completed:
                            continue
                        q_points=routes[recipient].get(q_task,[])
                        q_start=int(cursor[recipient][1]) if q_index==int(cursor[recipient][0]) else 0
                        if q_points[q_start:]: endpoint=np.asarray(q_points[-1],float)
                    points=routes[aid][tid]
                    added_km=math.dist(endpoint,points[0])+sum(math.dist(a,b) for a,b in zip(points,points[1:]))
                    projected=current_energy+added_km*float(cfg["energy_wh_per_km"][actor.kind])
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
            emit(ev,branch,t,"platform_fault",aircraft_id=aid,details={"action":"local_hold","unfinished_tasks_released":rest})
        def segment(branch,duration,ev,base_time=0,replan_candidates=None):
            rows=0
            branch_counters.setdefault(branch,{"completed":set(),"discovered":set(),"serviced":set(),"energy_wh":0.0,
                "command_attempts":0,"switches":0,"planner_calls":0,"violations":0,"faulted":set(),"rows":0})
            if branch in {"keep","replan"}:
                for aid in sorted(failed): emit(ev,branch,base_time,"platform_fault",aircraft_id=aid,details={"restored_from_common_prefix":True})
            if branch=="replan":
                chosen=(replan_candidates or [{"kind":"noop"}])[0]
                if chosen["kind"]=="uav_transfer":
                    donor,recipient,tid=chosen["donor"],chosen["recipient"],chosen["task_id"]
                    if tid in queues[donor] and actors[recipient].available:
                        queues[donor].remove(tid); queues[recipient].append(tid); routes[recipient][tid]=routes[donor][tid]
                        mission.assignments[tid]=recipient
                        effect={"time_s":base_time,"branch":branch,"donor":donor,"recipient":recipient,"task_id":tid,"kind":"frozen_catalog_uav_transfer"}
                        receipt["donor_effects"].append(effect)
                        emit(ev,branch,base_time,"donor_support_transfer",from_aircraft_id=donor,to_aircraft_id=recipient,task_id=tid)
                emit(ev,branch,base_time,"replan_catalog_choice",details={"catalog":replan_candidates,"chosen":chosen})
            for j in range(duration):
                t=base_time+j+1; current=backend.positions(); commands={}; attempts={}; delivered_ids=set()
                forced_outage_holds=set()
                if t==int(cfg["representative_events"]["uav_fault_time_s"]): fail_actor("uav-0-0",t,branch,ev)
                for aid in actors:
                    if aid in failed:
                        p=current[aid]["position_m"]; commands[aid]=[p[0]/1000,p[1]/1000,p[2]/1000]; continue
                    if energy[aid]<=energy_cap[aid]*float(cfg["energy_reserve_fraction"]):
                        p=current[aid]["position_m"]; active[aid]=[p[0]/1000,p[1]/1000,p[2]/1000]
                        pending[aid].clear(); commands[aid]=active[aid]
                        if aid not in safety_paused:
                            safety_paused.add(aid); emit(ev,branch,t,"energy_safety_hold",aircraft_id=aid,details={"remaining_wh":energy[aid],"pending_commands_cleared":True,"local_hold":True})
                        continue
                    outage_start=int(cfg["representative_events"]["comm_outage_time_s"])
                    outage_dur=int(cfg["representative_events"]["comm_outage_duration_s"])
                    forced=(outage_start<=t<outage_start+outage_dur and aid=="uav-1-0")
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
                        emit(ev,branch,t,"command_delivered",aircraft_id=aid,details={"configured_delay_s":int(cfg["communication"]["delay_s"])})
                    ti,wi=cursor[aid]; tid=queues[aid][ti] if ti<len(queues[aid]) else None
                    desired=active[aid]
                    if service[aid]:
                        target_id=service[aid][0]
                        observed=public_target_observations[target_id]
                        x,y=observed["target_xy_km"]; desired=[x,y,20.0]
                    elif tid is not None and wi<len(routes[aid][tid]):
                        x,y=routes[aid][tid][wi]; desired=[x,y,20.0]
                    if desired!=active[aid] and not pending[aid]:
                        outage_start=int(cfg["representative_events"]["comm_outage_time_s"])
                        outage_dur=int(cfg["representative_events"]["comm_outage_duration_s"])
                        forced=(outage_start<=t<outage_start+outage_dur and aid=="uav-1-0")
                        drop=forced or stable_drop(seed,aid,t,float(cfg["communication"]["dropout_probability"]))
                        attempts[aid]=(not drop,forced)
                        if not drop: pending[aid].append((t+int(cfg["communication"]["delay_s"]),list(desired)))
                        else: emit(ev,branch,t,"command_dropped",aircraft_id=aid,details={"forced_outage":forced})
                    commands[aid]=active[aid]
                readbacks=backend.step(commands,1.0)
                for aid,rb in readbacks.items():
                    pos=np.asarray(rb["position_m"],float)/1000.0; xy=(pos[0],pos[1]); previous_m=(last[aid]*1000.0).tolist(); dist=math.dist(last[aid],pos)
                    use=dist*1000.0*float(cfg["energy_wh_per_km"][actors[aid].kind])/1000.0
                    energy[aid]-=use; last[aid]=pos
                    ti,wi=cursor[aid]; tid=queues[aid][ti] if ti<len(queues[aid]) else None
                    found=[]
                    if tid and aid not in failed:
                        before=set(mission.discovered); discover_at(mission,tasks[tid],xy,float(cfg["sensor_radius_km"]))
                        found=sorted(mission.discovered-before)
                    for target_id in found:
                        zone,(target_x,target_y)=mission.truth[target_id]
                        observation={"target_id":target_id,"zone":tasks[tid].zone,
                                     "target_xy_km":[target_x,target_y],
                                     "kind":mission.truth_kind[target_id],"observed_at_sim_time_s":t,
                                     "observation_source":"simulated_sensor_radius"}
                        public_target_observations[target_id]=observation
                        emit(ev,branch,t,"target_discovered",aircraft_id=aid,target_id=target_id,
                             task_id=tid,zone=observation["zone"],target_xy_km=observation["target_xy_km"],
                             kind=observation["kind"],observation_source=observation["observation_source"])
                        if cfg["mode"]=="search_strike":
                            mission.event_log.append({"event":"strike_task_revealed","target_id":target_id,
                                                      "allowed_kind":observation["kind"],"time_s":t})
                            possible=[x for x in actors if actors[x].kind==observation["kind"]
                                      and (actors[x].kind=="uav" or actors[x].home_zone==observation["zone"])]
                            if possible: service[sorted(possible)[0]].append(target_id)
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
                    attempt=attempts.get(aid,(True,False))
                    row={"schema_version":3,"run_id":cfg["run_id"],"phase":phase,"branch":branch,"seed":seed,
                         "sim_time_s":t,"dt_s":1.0,"kit_frame":rb["kit_frame"],"aircraft_id":aid,"aircraft_type":actors[aid].kind,
                         "zone":actors[aid].home_zone,"task_id":tid,"search_cell_visited":visited,"task_completed":completed,"faulted":aid in failed,
                         "target_discovered_ids":found,"mission_xy_km":[float(xy[0]),float(xy[1])],"usd_position_m":rb["position_m"],
                         "public_target_observations":[public_target_observations[x] for x in found],
                         "previous_usd_position_m":previous_m,
                         "velocity_mps":rb["velocity_mps"],"usd_pose_error_m":rb["pose_error_m"],
                         "energy":{"status":"analytical_linear","incremental_wh":use,"remaining_wh":max(0.0,energy[aid]),
                                   "safety_pause":energy[aid]<=energy_cap[aid]*float(cfg["energy_reserve_fraction"])},
                         "communication":{"status":"simulated_command_queue","command_delivered":aid in delivered_ids,
                                           "command_dropped":not attempt[0],"forced_outage":aid in forced_outage_holds or attempt[1],
                                           "safety_hold":aid in communication_held or aid in forced_outage_holds,
                                           "command_attempt_accepted":attempt[0],"command_attempted":aid in attempts,
                                           "delay_s":int(cfg["communication"]["delay_s"])},
                         "hidden_truth_logged":False}
                    traj.write(json.dumps(row,separators=(",",":"))+"\n"); rows+=1
                    counter=branch_counters[branch]; counter["rows"]+=1; counter["energy_wh"]+=use
                    counter["command_attempts"]+=int(aid in attempts)
                if j%60==59:
                    traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
                    receipt["stages"].append({"branch":branch,"sim_time_s":t,"kit_frame":next(iter(readbacks.values()))["kit_frame"],"rows_written":rows})
                    persist(rec_path,receipt)
            return rows
        with traj_path.open("w",encoding="utf-8",buffering=1) as traj, events_path.open("w",encoding="utf-8",buffering=1) as ev:
            if phase=="representative":
                total_rows=segment("representative",int(cfg["representative_duration_s"]),ev)
                seconds={"representative":int(cfg["representative_duration_s"])}
            else:
                prefix=int(cfg["pair_snapshot_time_s"])
                prefix_rows=segment("pair_prefix",prefix,ev)
                pair_backend=backend.snapshot(); pair_common=common_state()
                energy_reserve={aid:energy_cap[aid]*float(cfg["energy_reserve_fraction"]) for aid in actors}
                pair_start_positions={aid:rb["position_m"] for aid,rb in backend.positions().items()}
                current_positions_km={aid:[p[0]/1000.0,p[1]/1000.0,p[2]/1000.0]
                                      for aid,p in pair_start_positions.items()}
                energy_rate_by_id={aid:float(cfg["energy_wh_per_km"][actors[aid].kind]) for aid in actors}
                candidates,choice=build_public_replan_catalog(queues,routes,cursor,actors,mission.completed,
                    failed,energy,energy_reserve,current_positions_km,max_speed_by_id,energy_rate_by_id,
                    int(cfg["pair_duration_s"]),int(cfg["replan_catalog_limit"]))
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
                replan_rows=segment("replan",int(cfg["pair_duration_s"]),ev,prefix,[choice])
                total_rows=prefix_rows+keep_rows+replan_rows
                seconds={"pair_prefix":prefix,"keep":int(cfg["pair_duration_s"]),"replan":int(cfg["pair_duration_s"])}
            traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
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
                "faulted_aircraft_count":len(m.get("faulted",set())),"row_count":m.get("rows",0),"actor_count":36}
        metrics={branch:public_metrics(branch) for branch in seconds}
        pair_summary=None
        if "keep" in metrics and "replan" in metrics:
            keys=("window_new_completed_public_tasks","window_new_discovered_targets","window_new_serviced_targets","analytical_energy_kwh",
                  "command_attempts","switches","planner_calls","violations","faulted_aircraft_count")
            pair_summary={"gain_keep_minus_replan":{f"{key}_keep_minus_replan":metrics["keep"][key]-metrics["replan"][key] for key in keys},
                          "scalar_status":"NOT_COMPUTED","scalar_reason":"planning CPU time is not instrumented; no scalar J claim"}
        receipt.update({"status":"complete","rows_written":total_rows,"simulated_seconds_by_branch":seconds,
                       "branch_metrics":metrics,"pair_summary":pair_summary,
                       "independent_audit":"pending_external_audit","events":"events.jsonl","trajectory":"trajectory.jsonl"})
        persist(manifest_path,{"run_id":cfg["run_id"],"config":cfg,"config_sha256":receipt["config_sha256"],
                               "source_hashes":source_hashes,"logs":{"trajectory":"trajectory.jsonl","events":"events.jsonl"},
                               "runtime_gpu_request":cfg["runtime"],"source_solver_reused":False})
        persist(rec_path,receipt)
        return 0
    except Exception as exc:
        receipt.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"}); persist(rec_path,receipt); return 2
    finally:
        receipt["backend_close_pending"]=backend is not None; persist(rec_path,receipt)
        if backend is not None: backend.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/a1_representative.json")
    p.add_argument("--source-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase",choices=("representative","pair"),default="representative"); a=p.parse_args()
    rc=run(a.config,a.source_root,a.output,a.phase)
    raise SystemExit(rc)

if __name__=="__main__": main()
