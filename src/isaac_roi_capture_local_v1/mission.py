"""Civil ROI mission rules shared by the Isaac runner and trajectory audit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import time


@dataclass(frozen=True)
class Task:
    task_id: str
    zone: int
    cell: int
    kind: str
    allowed_kind: str
    public_xy_km: tuple[float, float]
    public_bounds_km: tuple[float, float, float, float]
    private_xy_km: tuple[float, float] | None = None


@dataclass
class Aircraft:
    aircraft_id: str
    kind: str
    home_zone: int
    public_xy_km: tuple[float, float]
    available: bool = True


@dataclass
class MissionState:
    tasks: list[Task]
    aircraft: list[Aircraft]
    assignments: dict[str, str] = field(default_factory=dict)
    discovered: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    truth: dict[str, tuple[int, tuple[float, float]]] = field(default_factory=dict)
    truth_kind: dict[str, str] = field(default_factory=dict)
    mode: str = "search"
    strike_tasks: dict[str, str] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EventTriggerGate:
    """Open a planning gate after a fixed count of public operational events."""
    threshold: int = 1
    cooldown_s: float = 240.0
    pending_events: int = 0
    last_open_s: float = -math.inf

    def observe_event(self) -> None:
        self.pending_events += 1

    def open(self, time_s: float) -> bool:
        if self.pending_events < self.threshold or time_s - self.last_open_s < self.cooldown_s:
            return False
        self.pending_events = 0
        self.last_open_s = float(time_s)
        return True


def task_activation_schedule(tasks: list[Task], active_task_ids: list[str],
                             activation_times_s: list[float]) -> dict[str, float]:
    """Return deterministic public activation times; unscheduled catalog tasks stay deferred."""
    if len(active_task_ids) != len(activation_times_s):
        raise ValueError("active_task_ids and activation_times_s must have equal length")
    known = {task.task_id for task in tasks}
    if len(set(active_task_ids)) != len(active_task_ids) or not set(active_task_ids) <= known:
        raise ValueError("activation IDs must be unique members of the public task catalog")
    schedule = {task.task_id: math.inf for task in tasks}
    for task_id, time_s in zip(active_task_ids, activation_times_s):
        if not math.isfinite(float(time_s)) or float(time_s) < 0:
            raise ValueError("activation times must be finite and non-negative")
        schedule[task_id] = float(time_s)
    return schedule


def spatial_speed_cap(position_km: tuple[float, float], time_s: float,
                      blocks: list[dict[str, Any]]) -> float | None:
    """Return the minimum configured speed cap for matching local spatial/time blocks."""
    x, y = position_km
    caps = []
    for block in blocks:
        xmin, ymin, xmax, ymax = block["rect_km"]
        start_s, end_s = block["interval_s"]
        if (start_s <= time_s < end_s and xmin <= x <= xmax and ymin <= y <= ymax):
            caps.append(float(block["max_speed_mps"]))
    return min(caps) if caps else None


def roi_zones(center_km: tuple[float, float], width_km: float = 50.0) -> list[dict[str, Any]]:
    """3x3 square zones in native map XY, returned in km from map center."""
    cx, cy = center_km
    edge = width_km / 3.0
    return [
        {"zone": z, "xmin": cx-width_km/2 + col*edge, "xmax": cx-width_km/2+(col+1)*edge,
         "ymin": cy-width_km/2 + row*edge, "ymax": cy-width_km/2+(row+1)*edge,
         "center_xy_km": [cx-width_km/2+(col+.5)*edge, cy-width_km/2+(row+.5)*edge]}
        for z in range(9) for row, col in [(z//3, z%3)]
    ]


def build_mission(seed: int, center_km: tuple[float, float], targets_per_zone: int,
                  mode: str = "search", offset_xy_km: float = 0.8) -> MissionState:
    """Create public cell tasks separately from hidden truth coordinates."""
    import random
    if targets_per_zone not in (1, 3, 5) or mode not in ("search", "search_strike"):
        raise ValueError("targets_per_zone must be 1/3/5 and mode search/search_strike")
    tasks: list[Task] = []
    truth: dict[str, tuple[int, tuple[float, float]]] = {}
    mission_truth_kind: dict[str, str] = {}
    for z, zone in enumerate(roi_zones(center_km)):
        # Nine fixed public search cells per zone, independent of target load.
        for cell in range(9):
            task_id = f"z{z}-cell{cell}"
            row, col = divmod(cell, 3)
            cell_w = (zone["xmax"]-zone["xmin"])/3
            cell_h = (zone["ymax"]-zone["ymin"])/3
            bounds = (zone["xmin"]+col*cell_w, zone["xmin"]+(col+1)*cell_w,
                      zone["ymin"]+row*cell_h, zone["ymin"]+(row+1)*cell_h)
            cell_xy = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2)
            # Stable task specialization: UAV searches columns 0-1, heli column 2.
            allowed = "uav" if col < 2 else "helicopter"
            tasks.append(Task(task_id, z, cell, f"{allowed}_search", allowed, cell_xy, bounds))
        # Truth cardinality and coordinates are generated separately and never enter tasks.
        for target_index in range(targets_per_zone):
            target_rng = random.Random(seed * 1_000_003 + z * 10_007 + target_index)
            x = target_rng.uniform(zone["xmin"]+0.4, zone["xmax"]-0.4)
            y = target_rng.uniform(zone["ymin"]+0.4, zone["ymax"]-0.4)
            target_id = f"target-z{z}-{target_index}"
            truth[target_id] = (z, (x, y))
            mission_kind = "uav" if target_rng.random() < 0.5 else "helicopter"
            # Type is hidden until detection; it does not affect public search routes.
            mission_truth_kind[target_id] = mission_kind
    aircraft = []
    for z, zone in enumerate(roi_zones(center_km)):
        x, y = zone["center_xy_km"]
        for n in range(2):
            aircraft.append(Aircraft(f"uav-{z}-{n}", "uav", z, (x, y)))
        for n in range(2):
            aircraft.append(Aircraft(f"heli-{z}-{n}", "helicopter", z, (x, y)))
    return MissionState(tasks, aircraft, truth=truth, truth_kind=mission_truth_kind, mode=mode)


def initial_c1_greedy(mission: MissionState) -> dict[str, str]:
    """Public-information C1 proxy: nearest capable platform, load tie-break."""
    load = {a.aircraft_id: 0 for a in mission.aircraft}
    assignments: dict[str, str] = {}
    for task in sorted(mission.tasks, key=lambda t: (t.zone, t.cell)):
        candidates = [a for a in mission.aircraft if a.available and a.kind == task.allowed_kind
                      and a.home_zone == task.zone]
        if not candidates:
            raise RuntimeError(f"no capable aircraft for {task.task_id}")
        tx, ty = _cell_center(mission, task)
        chosen = min(candidates, key=lambda a: (
            load[a.aircraft_id], math.dist(a.public_xy_km, (tx, ty)), a.aircraft_id))
        assignments[task.task_id] = chosen.aircraft_id
        load[chosen.aircraft_id] += 1
    mission.assignments = assignments
    mission.event_log.append({"event":"initial_assignment", "solver":"c1_greedy_proxy",
                              "tie_break":"minimum current load, then distance, then aircraft id",
                              "source_solver_reused":False, "assignment_count":len(assignments)})
    return assignments


def build_public_replan_catalog(queues, routes, cursors, actors, completed,
                                failed, energy_remaining_wh, energy_reserve_wh,
                                current_positions_km, max_speed_mps, energy_wh_per_km,
                                horizon_s: float, max_candidates: int = 32,
                                switch_cost_s: float = 30.0,
                                eligible_task_ids=None):
    """Estimate legal unstarted UAV transfers from public route ETA only.

    The catalog includes NOOP. It never reads target truth or branch outcomes.
    """
    eligible_task_ids = eligible_task_ids or {}

    def _eligible(aircraft_id, extra=None):
        allowed = eligible_task_ids.get(aircraft_id)
        if allowed is None:
            return None
        result = set(allowed)
        if extra is not None:
            result.update(extra)
        return result

    def _is_allowed(aircraft_id, task_id):
        allowed = _eligible(aircraft_id)
        return allowed is None or task_id in allowed

    def queue_distance_km(aircraft_id, queue, route_map=None, eligible_override=None):
        cursor_task, cursor_waypoint = map(int, cursors[aircraft_id])
        position = tuple(float(value) for value in current_positions_km[aircraft_id][:2])
        distance = 0.0
        route_map = routes[aircraft_id] if route_map is None else route_map
        for index, task_id in enumerate(queue):
            if index < cursor_task or task_id in completed:
                continue
            if eligible_override is not None and task_id not in eligible_override:
                continue
            points = route_map.get(task_id, [])
            start = cursor_waypoint if index == cursor_task else 0
            points = points[start:]
            if not points:
                continue
            distance += math.dist(position, points[0])
            distance += sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            position = tuple(points[-1])
        return distance

    def queue_eta_s(aircraft_id, queue, route_map=None, eligible_override=None):
        return queue_distance_km(aircraft_id, queue, route_map, eligible_override) * 1000.0 / float(max_speed_mps[aircraft_id])

    def queue_energy_wh(aircraft_id, queue, route_map=None, eligible_override=None):
        return queue_distance_km(aircraft_id, queue, route_map, eligible_override) * float(energy_wh_per_km[aircraft_id])

    workloads = {aid: queue_eta_s(aid, queue, eligible_override=_eligible(aid)) for aid, queue in queues.items()}
    candidates = [{"kind":"noop", "estimated_gain_s":0.0, "reason":"retain current allocation"}]
    donors = [aid for aid, actor in actors.items()
              if actor.kind == "uav" and aid not in failed
              and any(_is_allowed(aid, task_id) for task_id in queues[aid][int(cursors[aid][0])+1:])]
    for donor in sorted(donors):
        # Never transfer the active route; only untouched public tasks are eligible.
        for task_index in range(int(cursors[donor][0])+1, len(queues[donor])):
            task_id = queues[donor][task_index]
            if task_id in completed or not _is_allowed(donor, task_id):
                continue
            for recipient, actor in sorted(actors.items()):
                if (recipient == donor or actor.kind != "uav" or recipient in failed):
                    continue
                donor_after = list(queues[donor]); donor_after.remove(task_id)
                recipient_after = list(queues[recipient]) + [task_id]
                recipient_routes_after = dict(routes[recipient])
                recipient_routes_after[task_id] = routes[donor][task_id]
                donor_eta_after = queue_eta_s(donor, donor_after, eligible_override=_eligible(donor))
                recipient_eta_after = queue_eta_s(recipient, recipient_after, recipient_routes_after,
                                                  _eligible(recipient, {task_id}))
                donor_energy_after = queue_energy_wh(donor, donor_after, eligible_override=_eligible(donor))
                recipient_energy_before = queue_energy_wh(recipient, queues[recipient], eligible_override=_eligible(recipient))
                recipient_energy_after = queue_energy_wh(recipient, recipient_after, recipient_routes_after,
                                                         _eligible(recipient, {task_id}))
                recipient_distance_before = queue_distance_km(recipient, queues[recipient], eligible_override=_eligible(recipient))
                recipient_distance_after = queue_distance_km(recipient, recipient_after, recipient_routes_after,
                                                             _eligible(recipient, {task_id}))
                if (energy_remaining_wh.get(donor, 0.0)-donor_energy_after < energy_reserve_wh.get(donor, 0.0)
                        or energy_remaining_wh.get(recipient, 0.0)-recipient_energy_after < energy_reserve_wh.get(recipient, 0.0)):
                    continue
                before = max(workloads[donor], workloads[recipient])
                after = max(donor_eta_after, recipient_eta_after)
                gain = before-after-switch_cost_s
                route_distance_km = recipient_distance_after-recipient_distance_before
                task_eta = route_distance_km * 1000.0 / float(max_speed_mps[recipient])
                energy_cost = recipient_energy_after-recipient_energy_before
                candidates.append({"kind":"uav_transfer", "donor":donor, "recipient":recipient,
                                   "task_id":task_id,"estimated_task_eta_s":task_eta,
                                   "estimated_route_distance_km":route_distance_km,
                                   "estimated_energy_wh":energy_cost,"estimated_gain_s":gain,
                                   "horizon_deadline_checked":False,
                                   "switch_cost_s":switch_cost_s})
    sort_key = lambda c:(-c["estimated_gain_s"], c["kind"]!="noop",
                         c.get("donor",""),c.get("recipient",""),c.get("task_id",""))
    noop = next(candidate for candidate in candidates if candidate["kind"] == "noop")
    alternatives = sorted((candidate for candidate in candidates if candidate["kind"] != "noop"), key=sort_key)
    limit = min(16, max(1, int(max_candidates)))
    catalog = sorted([noop, *alternatives[:limit-1]], key=sort_key)
    selected = catalog[0]
    return catalog, selected


def _cell_center(mission: MissionState, task: Task) -> tuple[float, float]:
    return task.public_xy_km


def task_waypoint(mission: MissionState, task: Task, meters_per_unit: float,
                  map_center_scene: tuple[float, float]) -> tuple[float, float, float]:
    """Waypoint depends only on public task cell, never hidden target truth."""
    x, y = _cell_center(mission, task)
    scale = 1000.0 / meters_per_unit
    return ((x-map_center_scene[0])*scale, (y-map_center_scene[1])*scale, 1.2)


def task_sweep(task: Task, lane_spacing_km: float = 0.8,
               point_spacing_km: float = 0.4) -> list[tuple[float, float]]:
    """Public lawnmower sweep with spacing no wider than the declared sensor radius."""
    xmin, xmax, ymin, ymax = task.public_bounds_km
    count = max(1, math.ceil((ymax-ymin)/lane_spacing_km))
    ys = [ymin+(i+0.5)*(ymax-ymin)/count for i in range(count)]
    path = []
    for i, y in enumerate(ys):
        n = max(1, math.ceil((xmax-xmin)/point_spacing_km))
        xs = [xmin+(j+0.5)*(xmax-xmin)/n for j in range(n)]
        if i % 2: xs.reverse()
        path.extend((x,y) for x in xs)
    return path


def discover_at(mission: MissionState, task: Task, xy_km: tuple[float, float],
                radius_km: float = 0.4) -> bool:
    found = []
    for target_id, (zone, truth_xy) in mission.truth.items():
        if target_id not in mission.discovered and zone == task.zone and math.dist(xy_km, truth_xy) <= radius_km:
            mission.discovered.add(target_id)
            mission.event_log.append({"event":"target_discovered", "target_id":target_id,
                                      "search_task_id":task.task_id,
                                      "time_source":"simulated_sensor_radius"})
            found.append(target_id)
    return bool(found)


def abstract_strike(mission: MissionState, target_id: str, platform_kind: str,
                    platform_zone: int, position_km: tuple[float,float], time_s: float,
                    service_radius_km: float = 0.3, service_duration_s: float = 20.0,
                    available: bool = True, paused: bool = False) -> bool:
    """Complete service after continuous eligible dwell; no weapon/trajectory interface."""
    if (mission.mode != "search_strike" or target_id not in mission.discovered
            or target_id in mission.completed):
        return False
    zone, truth_xy = mission.truth[target_id]
    allowed_kind = mission.truth_kind[target_id]
    if allowed_kind != platform_kind or (platform_kind == "helicopter" and zone != platform_zone):
        return False
    start = mission.strike_tasks.get(target_id)
    if not available or paused or math.dist(position_km, truth_xy) > service_radius_km:
        if mission.strike_tasks.pop(target_id, None) is not None:
            reason = "unavailable" if not available else "paused" if paused else "left_service_radius"
            mission.event_log.append({"event":"abstract_strike_service_reset", "target_id":target_id,
                                      "time_s":time_s, "reason":reason})
        return False
    if start is None:
        mission.strike_tasks[target_id] = time_s
        mission.event_log.append({"event":"abstract_strike_service_start", "target_id":target_id,
                                  "platform_kind":platform_kind, "time_s":time_s})
        return False
    if time_s - start < service_duration_s:
        return False
    mission.completed.add(target_id)
    mission.event_log.append({"event":"abstract_strike_complete", "target_id":target_id,
                              "time_s":time_s, "weapon_control":"none", "ballistics":"none"})
    return True


def assign_cross_zone_support(mission: MissionState, requester_zone: int,
                              candidate_id: str) -> bool:
    aircraft = next(a for a in mission.aircraft if a.aircraft_id == candidate_id)
    if aircraft.kind == "helicopter" and aircraft.home_zone != requester_zone:
        mission.event_log.append({"event":"support_rejected", "aircraft_id":candidate_id,
                                  "reason":"helicopter_zone_lock"})
        return False
    return aircraft.kind == "uav"


def mission_snapshot(mission: MissionState) -> dict[str, Any]:
    return {"assignments":dict(mission.assignments), "discovered":sorted(mission.discovered),
            "completed":sorted(mission.completed), "mode":mission.mode,
            "event_log":list(mission.event_log), "strike_tasks":dict(mission.strike_tasks),
            "aircraft_available":{a.aircraft_id:a.available for a in mission.aircraft}}


def restore_mission(mission: MissionState, snap: dict[str, Any]) -> None:
    mission.assignments = dict(snap["assignments"])
    mission.discovered = set(snap["discovered"])
    mission.completed = set(snap["completed"])
    mission.mode = snap["mode"]
    mission.event_log = list(snap["event_log"])
    mission.strike_tasks = dict(snap["strike_tasks"])
    for aircraft in mission.aircraft:
        aircraft.available = bool(snap["aircraft_available"][aircraft.aircraft_id])


def recalculate_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent finite-difference checks over persisted USD readback rows."""
    by_actor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_actor.setdefault(row["aircraft_id"], []).append(row)
    speed_max = 0.0
    covered_tasks: set[str] = set()
    branch_energy: dict[str, float] = {}
    displacements = {}
    for actor, trace in by_actor.items():
        trace.sort(key=lambda r: r["sim_time_s"])
        first = trace[0]["usd_position_m"]
        last = trace[-1]["usd_position_m"]
        displacements[actor] = math.dist(first, last)
        for a, b in zip(trace, trace[1:]):
            dt = b["sim_time_s"] - a["sim_time_s"]
            if dt > 0:
                speed_max = max(speed_max, math.dist(a["usd_position_m"], b["usd_position_m"])/dt)
        for row in trace:
            if row.get("search_cell_visited") and row.get("task_id"):
                covered_tasks.add(f"{row.get('branch')}:{row['task_id']}")
            energy = row.get("energy", {}).get("incremental_wh")
            if energy is not None:
                key = str(row.get("branch", "main"))
                branch_energy[key] = branch_energy.get(key, 0.0) + float(energy)
    return {"aircraft_count":len(by_actor), "row_count":len(rows),
            "max_readback_speed_mps":speed_max, "net_displacement_m":displacements,
            "covered_task_keys":sorted(covered_tasks),"recalculated_energy_wh_by_branch":branch_energy}
