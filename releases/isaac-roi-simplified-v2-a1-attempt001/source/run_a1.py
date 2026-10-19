#!/usr/bin/env python3
"""A1 90-minute Isaac run or 30-minute snapshot-paired keep/replan run."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, sys
from pathlib import Path
import numpy as np
from isaac_backend import IsaacBackend
from mission import (abstract_strike, assign_cross_zone_support, build_mission,
                     discover_at, initial_c1_greedy, mission_snapshot,
                     restore_mission, task_sweep)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def persist(path, data):
    with Path(path).open("w",encoding="utf-8") as f:
        json.dump(data,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())

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
        assignment=initial_c1_greedy(mission)
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
        receipt.update({"status":"scene_ready","actual_isaac_run":True,"map_size_km":300.0,"meters_per_unit":100.0,
                       "roi_center_km":list(center),"roi_width_km":cfg["roi_width_km"],"aircraft_count":36,
                       "aircraft_per_zone":{"uav":2,"helicopter":2},"hidden_target_count":len(mission.truth),
                       "public_search_task_count":len(mission.tasks),"initial_assignment_solver":"c1_greedy_proxy",
                       "c1_source_solver_reused":False,"c1_tie_break":"same-zone capable aircraft, minimum load, nearest cell, id",
                       "hidden_truth_exposed_to_planner":False,"initial_usd_pose_error_max_m":max(x["pose_error_m"] for x in rb0.values()),
                       "initial_positions_m":{aid:x["position_m"] for aid,x in rb0.items()},
                       "expected_task_ids":[t.task_id for t in mission.tasks],
                       "public_task_manifest":[{"task_id":t.task_id,"zone":t.zone,"cell":t.cell,"allowed_kind":t.allowed_kind} for t in mission.tasks],
                       "actor_manifest":[{"aircraft_id":a.aircraft_id,"type":a.kind,"home_zone":a.home_zone} for a in mission.aircraft]})
        persist(rec_path,receipt)
        routes={aid:{tid:task_sweep(tasks[tid],cfg["sweep_lane_spacing_km"],cfg["sensor_radius_km"]) for tid in q}
                for aid,q in queues.items()}
        cursor={aid:[0,0] for aid in actors}; service={aid:[] for aid in actors}
        energy_cap={aid:float(cfg["energy_capacity_wh"][actors[aid].kind]) for aid in actors}
        energy=dict(energy_cap); last={aid:np.array(rb0[aid]["position_m"],float)/1000 for aid in actors}
        active={aid:list(last[aid]) for aid in actors}; pending={aid:[] for aid in actors}
        failed=set(); safety_paused=set()
        def emit(h,branch,t,event,**fields):
            row={"schema_version":1,"run_id":cfg["run_id"],"phase":phase,"branch":branch,
                 "sim_time_s":t,"event":event,"event_id":f"{branch}:{t}:{event}:{fields.get('aircraft_id','')}:{fields.get('task_id',fields.get('target_id',''))}",**fields}
            h.write(json.dumps(row,separators=(",",":"))+"\n")
        def common_state():
            return copy.deepcopy((queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,mission_snapshot(mission)))
        def restore_common(snapshot):
            nonlocal queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused
            (queues,routes,cursor,service,energy,last,active,pending,failed,safety_paused,ms)=copy.deepcopy(snapshot)
            restore_mission(mission,ms)
        def fail_actor(aid,t,branch,ev):
            if aid in failed:return
            failed.add(aid); actors[aid].available=False; pending[aid].clear()
            current=backend.positions()[aid]["position_m"]; active[aid]=[v/1000 for v in current]
            rest=queues[aid][cursor[aid][0]:]; queues[aid]=queues[aid][:cursor[aid][0]]
            for tid in rest:
                choices=[x for x in actors if actors[x].kind==tasks[tid].allowed_kind and actors[x].home_zone!=actors[aid].home_zone and actors[x].available]
                choices.sort(key=lambda x:(len(queues[x]),x))
                if choices:
                    donor=choices[0]
                    if assign_cross_zone_support(mission,actors[donor].home_zone,donor):
                        queues[donor].append(tid); routes[donor][tid]=routes[aid][tid]
                        mission.assignments[tid]=donor
                        effect={"time_s":t,"branch":branch,"donor":aid,"recipient":donor,"task_id":tid,"kind":"cross_zone_uav_support"}
                        receipt["donor_effects"].append(effect)
                        emit(ev,branch,t,"donor_support_transfer",from_aircraft_id=aid,to_aircraft_id=donor,task_id=tid)
            emit(ev,branch,t,"platform_fault",aircraft_id=aid,details={"action":"local_hold","unfinished_tasks_released":rest})
        def segment(branch,duration,ev,base_time=0,replan_candidates=None):
            rows=0
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
                    due=[msg for msg in pending[aid] if msg[0]<=t]
                    pending[aid]=[msg for msg in pending[aid] if msg[0]>t]
                    if due:
                        delivered_ids.add(aid)
                        active[aid]=due[-1][1]
                        emit(ev,branch,t,"command_delivered",aircraft_id=aid,details={"configured_delay_s":int(cfg["communication"]["delay_s"])})
                    ti,wi=cursor[aid]; tid=queues[aid][ti] if ti<len(queues[aid]) else None
                    desired=active[aid]
                    if service[aid]:
                        target_id=service[aid][0]; _,(x,y)=mission.truth[target_id]; desired=[x,y,20.0]
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
                        for target_id in found: emit(ev,branch,t,"target_discovered",aircraft_id=aid,target_id=target_id,task_id=tid)
                    for target_id in found:
                        if cfg["mode"]=="search_strike":
                            mission.event_log.append({"event":"strike_task_revealed","target_id":target_id,"allowed_kind":mission.truth_kind[target_id],"time_s":t})
                            possible=[x for x in actors if actors[x].kind==mission.truth_kind[target_id] and actors[x].home_zone==mission.truth[target_id][0]]
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
                    attempt=attempts.get(aid,(True,False))
                    row={"schema_version":2,"run_id":cfg["run_id"],"phase":phase,"branch":branch,"seed":seed,
                         "sim_time_s":t,"dt_s":1.0,"kit_frame":rb["kit_frame"],"aircraft_id":aid,"aircraft_type":actors[aid].kind,
                         "zone":actors[aid].home_zone,"task_id":tid,"search_cell_visited":visited,"task_completed":completed,"faulted":aid in failed,
                         "target_discovered_ids":found,"mission_xy_km":[float(xy[0]),float(xy[1])],"usd_position_m":rb["position_m"],
                         "previous_usd_position_m":previous_m,
                         "velocity_mps":rb["velocity_mps"],"usd_pose_error_m":rb["pose_error_m"],
                         "energy":{"status":"analytical_linear","incremental_wh":use,"remaining_wh":max(0.0,energy[aid]),
                                   "safety_pause":energy[aid]<=energy_cap[aid]*float(cfg["energy_reserve_fraction"])},
                         "communication":{"status":"simulated_command_queue","command_delivered":aid in delivered_ids,
                                           "command_dropped":not attempt[0],"forced_outage":attempt[1],
                                           "command_attempt_accepted":attempt[0],"delay_s":int(cfg["communication"]["delay_s"])},
                         "hidden_truth_logged":False}
                    traj.write(json.dumps(row,separators=(",",":"))+"\n"); rows+=1
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
                candidates=[{"kind":"noop"}]
                donors=sorted((aid for aid in actors if actors[aid].kind=="uav" and queues[aid] and aid not in failed),key=lambda a:(-len(queues[a]),a))
                for donor in donors:
                    for recipient in sorted(actors):
                        if actors[recipient].kind=="uav" and actors[recipient].home_zone!=actors[donor].home_zone and recipient not in failed:
                            tid=queues[donor][-1]
                            candidates.append({"kind":"uav_transfer","donor":donor,"recipient":recipient,"task_id":tid})
                candidates=candidates[:int(cfg["replan_catalog_limit"])]
                choice=next((c for c in candidates if c["kind"]=="uav_transfer"),candidates[0])
                # Freeze the candidate list and choice before outcomes are observed.
                candidates=[choice]+[c for c in candidates if c!=choice]
                receipt["pair_snapshot"]={"sim_time_s":prefix,"ordinary_decision":"post-1Hz observation/task queue state",
                     "backend_snapshot":True,"mission_queues_energy_comms":True,"frozen_candidate_catalog":candidates}
                keep_rows=segment("keep",int(cfg["pair_duration_s"]),ev,prefix,candidates[:1] if False else [{"kind":"noop"}])
                backend.restore(pair_backend); restore_common(pair_common)
                replan_rows=segment("replan",int(cfg["pair_duration_s"]),ev,prefix,candidates[:1])
                total_rows=prefix_rows+keep_rows+replan_rows
                seconds={"pair_prefix":prefix,"keep":int(cfg["pair_duration_s"]),"replan":int(cfg["pair_duration_s"])}
            traj.flush(); os.fsync(traj.fileno()); ev.flush(); os.fsync(ev.fileno())
        receipt.update({"status":"complete","rows_written":total_rows,"simulated_seconds_by_branch":seconds,
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
