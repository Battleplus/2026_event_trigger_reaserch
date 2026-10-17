"""Deterministic discrete-time civilian disaster-relief simulator.

The policy sees only ``_public``.  Operational truth, delayed reports, and
resource changes are held separately and are exported only as run evidence.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

try:
    from .baselines import choose_actions
    from .chapter1_adapter import normalize_initial_assignment
    from .scenario_schema import validate_scenario
    from .stable_identity import bind_decision
except ImportError:
    from baselines import choose_actions
    from chapter1_adapter import normalize_initial_assignment
    from scenario_schema import validate_scenario
    from stable_identity import bind_decision


_EPS = 1e-9


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _clean_number(value: float) -> float:
    if abs(value) < _EPS:
        return 0.0
    return float(value)


def _travel_ticks(distance: float, speed: float) -> int:
    if distance <= _EPS:
        return 0
    return math.ceil(distance / speed - _EPS)


class ReliefEnv:
    """Run baseline policies or learned batches over one synthetic scenario."""

    def __init__(self, scenario: dict):
        validate_scenario(scenario)
        self.scenario = copy.deepcopy(scenario)
        self.initial_handoff = normalize_initial_assignment(self.scenario)
        self._platform_spec = {item["id"]: copy.deepcopy(item) for item in self.scenario["platforms"]}
        self._depot_spec = {item["id"]: copy.deepcopy(item) for item in self.scenario["depots"]}
        self._task_spec = {item["id"]: copy.deepcopy(item) for item in self.scenario["tasks"]}
        self._original_owner: dict[str, str | None] = {}
        for platform_id, queue in self.scenario["chapter1_snapshot"]["queues"].items():
            for task_id in queue:
                self._original_owner[task_id] = platform_id
        self._reset()

    def _reset(self) -> None:
        self._step_api_active = False
        self._terminal = False
        self._trajectory_cache = None
        self.time = 0
        self.events: list[dict] = []
        self.transitions: list[dict] = []
        self._event_seq = 0
        self._policy: str | None = None
        self._platforms: dict[str, dict] = {}
        for platform_id, spec in self._platform_spec.items():
            self._platforms[platform_id] = {
                **copy.deepcopy(spec),
                "position": list(spec["position"]),
                "energy": float(spec["energy_initial"]),
                "energy_used": 0.0,
                "failed": False,
                "active_task_id": None,
            }
        self._depots = {
            depot_id: {**copy.deepcopy(spec), "stock": {key: float(value) for key, value in spec["stock"].items()}}
            for depot_id, spec in self._depot_spec.items()
        }
        self._tasks: dict[str, dict] = {}
        for task_id, spec in self._task_spec.items():
            known = bool(spec["known_initial"])
            self._tasks[task_id] = {
                **copy.deepcopy(spec),
                "original_assignee": self._original_owner.get(task_id),
                "assignee": self._original_owner.get(task_id),
                "known": known,
                "discovered": known,
                "discovery_time": 0 if known else None,
                "status": "pending" if known else "undiscovered",
                "phase": "pending" if known else "undiscovered",
                "started_at": None,
                "completed_at": None,
                "deadline_missed": False,
                "deadline_event_emitted": False,
                "report_generated_at": None,
                "report_received_at": None,
                "failure_reason": None,
                "service_progress": 0.0,
                "cargo_onboard": 0.0,
                "delivered": False,
                "on_time": None,
                "_reassigned": False,
                "_new_assignment_counted": False,
                "active_platform_id": None,
            }
        self._cargo: dict[tuple[str, str, str, str], float] = defaultdict(float)
        self._queues = {platform_id: list(queue) for platform_id, queue in self.scenario["chapter1_snapshot"]["queues"].items()}
        for platform_id in self._platform_spec:
            self._queues.setdefault(platform_id, [])
        self._scheduled_reports: list[dict] = []

        self._public_platforms = {
            platform_id: {
                "id": platform_id,
                "type": spec["type"],
                "position": list(spec["position"]),
                "local": spec["local"],
                "capabilities": list(spec["capabilities"]),
                "capacity_kg": spec["capacity_kg"],
                "speed_km_min": spec["speed_km_min"],
                "energy_per_km": spec["energy_per_km"],
                "service_energy_per_min": spec["service_energy_per_min"],
                "energy": float(spec["energy_initial"]),
                "failed": False,
                "busy": False,
                "active_task_id": None,
            }
            for platform_id, spec in self._platform_spec.items()
        }
        self._public_depots = {
            depot_id: {
                "id": depot_id,
                "position": list(spec["position"]),
                "stock": {key: float(value) for key, value in spec["stock"].items()},
            }
            for depot_id, spec in self._depot_spec.items()
        }
        self._public_tasks = {
            task_id: self._public_task_from_truth(task)
            for task_id, task in self._tasks.items()
            if task["known"]
        }

    @staticmethod
    def _public_task_from_truth(task: dict) -> dict:
        fields = (
            "id", "kind", "position", "local", "deadline", "priority", "commodity", "quantity",
            "depot_id", "load_minutes", "unload_minutes", "service_minutes", "status", "phase",
            "assignee", "original_assignee", "discovered", "started_at", "completed_at",
            "deadline_missed", "report_generated_at", "report_received_at",
        )
        return {key: copy.deepcopy(task[key]) for key in fields if key in task}

    def _log(self, event_type: str, time: int | float, **fields: Any) -> dict:
        event = {"seq": self._event_seq, "time": _clean_number(float(time)), "type": event_type}
        self._event_seq += 1
        event.update(copy.deepcopy(fields))
        self.events.append(event)
        return event

    def public_observation(self) -> dict:
        """Return a detached view containing only information received so far."""

        return {
            "schema": "relief-public-observation/1",
            "time": self.time,
            "horizon": self.scenario["horizon"],
            "map": copy.deepcopy(self.scenario["map"]),
            "platforms": [copy.deepcopy(self._public_platforms[key]) for key in sorted(self._public_platforms)],
            "depots": [copy.deepcopy(self._public_depots[key]) for key in sorted(self._public_depots)],
            "tasks": [copy.deepcopy(self._public_tasks[key]) for key in sorted(self._public_tasks)],
            "chapter1_snapshot": {
                "source": self.scenario["chapter1_snapshot"]["source"],
                "time": self.scenario["chapter1_snapshot"]["time"],
                "queues": copy.deepcopy(self.scenario["chapter1_snapshot"]["queues"]),
            },
        }

    def reset(self, policy: str = "LEARNED") -> dict:
        """Start a minute-stepped episode and return its public observation.

        Learned policies may reassign local tasks and retry released failures,
        while the existing block-external commitment rule remains enforced.
        """

        if policy not in {"C0", "C1", "L0", "L1", "LEARNED"}:
            raise ValueError("policy must be C0, C1, L0, L1, or LEARNED")
        self._reset()
        self._policy = policy
        self._rewards_paid: set[tuple[str, str]] = set()
        self._step_api_active = True
        self._terminal = False
        self._trajectory_cache: dict | None = None
        for task_id, owner in sorted(self._original_owner.items()):
            self._log(
                "assignment", 0, task_id=task_id,
                previous_assignee=owner, new_assignee=owner,
                initial=True, reason="chapter1_snapshot",
            )
        self._open_minute(0)
        return self.public_observation()

    def _open_minute(self, time: int) -> None:
        self.time = time
        self._apply_external_events(time)
        self._receive_reports(time)
        self._expire_deadlines(time)

    def _step_reward(self, events: list[dict]) -> float:
        reward = -0.01 * sum(
            float(event.get("energy", 0.0))
            for event in events if event.get("type") in {"move", "service"}
        )
        for event in events:
            event_type = event.get("type")
            task_id = event.get("task_id")
            if not isinstance(task_id, str) or task_id not in self._tasks:
                continue
            task = self._tasks[task_id]
            if event_type == "delivery_confirmed" and task["on_time"] is True:
                marker = ("delivery", task_id)
                if marker not in self._rewards_paid:
                    reward += float(task["priority"]) / 5.0
                    self._rewards_paid.add(marker)
            elif event_type == "report_received" and event.get("kind") == "inspection_result":
                marker = ("inspection_report", task_id)
                if task["kind"] == "inspection" and task["on_time"] is True and marker not in self._rewards_paid:
                    reward += 0.1
                    self._rewards_paid.add(marker)
            elif event_type == "assignment" and event.get("initial") is False:
                original = task["original_assignee"]
                if original is not None and event.get("new_assignee") != original:
                    marker = ("initial_reassignment", task_id)
                    if marker not in self._rewards_paid:
                        reward -= 0.05
                        self._rewards_paid.add(marker)
        return _clean_number(reward)

    def step(self, action_batch: list[str] | tuple[str, ...]) -> tuple[dict, float, bool, bool, dict]:
        """Execute one simultaneous assignment batch, then advance one minute."""

        if not getattr(self, "_step_api_active", False):
            raise RuntimeError("call reset() before step()")
        if self._terminal:
            raise RuntimeError("episode is terminated")
        if not isinstance(action_batch, (list, tuple)) or any(type(item) is not str for item in action_batch):
            raise TypeError("action_batch must be a list or tuple of candidate ID strings")
        time = int(self.time)
        observation = self.public_observation()
        catalog = self.candidates()
        candidates_by_id = {candidate["id"]: candidate for candidate in catalog}
        selected: list[dict] = []
        selected_platforms: set[str] = set()
        selected_tasks: set[str] = set()
        visible_tasks = {item["id"]: item for item in observation["tasks"]}
        visible_depots = {item["id"]: item for item in observation["depots"]}
        reserved_stock = self._public_stock_commitments()
        stopped = False
        for candidate_id in action_batch:
            if stopped:
                raise ValueError("NOOP must end an action batch")
            if candidate_id == "NOOP":
                stopped = True
                continue
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None or candidate.get("type") != "dispatch":
                raise ValueError(f"unknown or non-dispatch candidate ID: {candidate_id!r}")
            platform_id = candidate["platform_id"]
            task_id = candidate["task_id"]
            if platform_id in selected_platforms or task_id in selected_tasks:
                raise ValueError("an action batch may use each platform and task at most once")
            task = visible_tasks[task_id]
            if task["kind"] == "delivery":
                depot_id = task["depot_id"]
                commodity = task["commodity"]
                key = (depot_id, commodity)
                quantity = float(task["quantity"])
                available = float(visible_depots[depot_id]["stock"].get(commodity, 0.0))
                if reserved_stock.get(key, 0.0) + quantity > available + _EPS:
                    raise ValueError("an action batch exceeds publicly available depot stock")
                reserved_stock[key] = reserved_stock.get(key, 0.0) + quantity
            selected_platforms.add(platform_id)
            selected_tasks.add(task_id)
            selected.append(copy.deepcopy(candidate))

        identity = bind_decision(observation, catalog)
        transition = {
            "time": time,
            "public_observation": observation,
            "candidates": copy.deepcopy(catalog),
            "selected_actions": copy.deepcopy(selected),
            "identity": identity,
            "observation_sha256": identity["observation_sha256"],
            "candidate_catalog_sha256": identity["candidate_catalog_sha256"],
        }
        self.transitions.append(transition)
        event_start = len(self.events)
        for action in selected:
            self._dispatch(action, time)
        self._advance(time)
        self._open_minute(time + 1)
        reward_events = self.events[event_start:]
        reward = self._step_reward(reward_events)
        self._terminal = self.time >= int(self.scenario["horizon"])
        info = {
            "transition": copy.deepcopy(transition),
            "reward_events": copy.deepcopy(reward_events),
            "policy": self._policy,
        }
        return self.public_observation(), reward, self._terminal, False, info

    def export_trajectory(self) -> dict:
        """Export a baseline-shaped full trajectory after the horizon."""

        if not getattr(self, "_step_api_active", False) or not self._terminal:
            raise RuntimeError("trajectory export requires a terminated episode")
        if self._trajectory_cache is None:
            final_state = self._close_state()
            self._trajectory_cache = {
                "scenario_id": self.scenario["scenario_id"],
                "policy": self._policy,
                "initial_handoff": copy.deepcopy(self.initial_handoff),
                "events": copy.deepcopy(self.events),
                "transitions": copy.deepcopy(self.transitions),
                "final_state": final_state,
                "summary": self._summary(final_state),
            }
        return copy.deepcopy(self._trajectory_cache)

    @property
    def episode_metrics(self) -> dict:
        """Independent evaluation metrics for a completed minute-stepped run."""

        return copy.deepcopy(self.export_trajectory()["summary"])

    def _compatible(self, platform: dict, task: dict) -> bool:
        needed = "inspect" if task["kind"] == "inspection" else "deliver"
        if needed not in platform["capabilities"] or platform["local"] != task["local"]:
            return False
        if task["kind"] == "delivery" and float(task["quantity"]) > float(platform["capacity_kg"]) + _EPS:
            return False
        return True

    def _public_stock_commitments(self) -> dict[tuple[str, str], float]:
        committed: dict[tuple[str, str], float] = {}
        for task in self._public_tasks.values():
            if (
                task.get("kind") != "delivery"
                or task.get("status") not in {"active", "assigned"}
                or task.get("phase") not in {"assigned", "to_depot", "loading"}
            ):
                continue
            key = (str(task["depot_id"]), str(task["commodity"]))
            committed[key] = committed.get(key, 0.0) + float(task["quantity"])
        return committed

    def _public_stock_remaining(self, depot_id: str, commodity: str) -> float:
        stock = float(self._public_depots[depot_id]["stock"].get(commodity, 0.0))
        committed = self._public_stock_commitments().get((depot_id, commodity), 0.0)
        return max(0.0, stock - committed)

    def _estimated_mission(self, platform: dict, task: dict, now: int) -> tuple[float, float]:
        speed = float(platform["speed_km_min"])
        energy_per_km = float(platform["energy_per_km"])
        service_per_min = float(platform["service_energy_per_min"])
        position = platform["position"]
        if task["kind"] == "inspection":
            distance = _distance(position, task["position"])
            completion = now + _travel_ticks(distance, speed) + int(task["service_minutes"])
            energy = distance * energy_per_km + float(task["service_minutes"]) * service_per_min
        else:
            depot = self._public_depots.get(task["depot_id"], {"position": self._depot_spec[task["depot_id"]]["position"]})
            depot_position = depot["position"]
            outbound = _distance(position, depot_position)
            onward = _distance(depot_position, task["position"])
            completion = now + _travel_ticks(outbound, speed) + int(task["load_minutes"])
            completion += _travel_ticks(onward, speed) + int(task["unload_minutes"])
            service = int(task["load_minutes"]) + int(task["unload_minutes"])
            energy = (outbound + onward) * energy_per_km + service * service_per_min
        return float(completion), float(energy)

    def candidates(self) -> list[dict]:
        """Build a stable public candidate catalog plus NOOP."""

        candidates: list[dict] = []
        tasks = self._public_tasks
        for platform_id in sorted(self._public_platforms):
            platform = self._public_platforms[platform_id]
            if platform["busy"] or platform["failed"]:
                continue
            for task_id in sorted(tasks):
                task = tasks[task_id]
                if task["status"] not in {"pending", "assigned", "released"}:
                    continue
                if not self._compatible(platform, task):
                    continue
                # C1 may alter local snapshot queues.  Work outside the local
                # block is an external commitment: keep its snapshot owner.
                if self._policy in {"C1", "L0", "L1", "LEARNED"} and not task["local"]:
                    if task["original_assignee"] is None or task["original_assignee"] != platform_id:
                        continue
                if self._policy == "C0" and task["original_assignee"] != platform_id:
                    continue
                if task["kind"] == "delivery":
                    depot = self._public_depots[task["depot_id"]]
                    available = (
                        self._public_stock_remaining(task["depot_id"], task["commodity"])
                        if self._step_api_active
                        else float(depot["stock"].get(task["commodity"], 0.0))
                    )
                    if available + _EPS < float(task["quantity"]):
                        continue
                completion, required_energy = self._estimated_mission(platform, task, self.time)
                if float(platform["energy"]) + _EPS < required_energy:
                    continue
                candidates.append({
                    "id": f"dispatch:{platform_id}:{task_id}",
                    "type": "dispatch",
                    "platform_id": platform_id,
                    "task_id": task_id,
                    "kind": task["kind"],
                    "estimated_completion": completion,
                    "required_energy": required_energy,
                    "locked": bool(not task["local"] and task["original_assignee"] is not None),
                })
        candidates.sort(key=lambda item: (item["platform_id"], item["task_id"], item["id"]))
        candidates.append({"id": "NOOP", "type": "noop"})
        return candidates

    def policy_input(self) -> dict:
        """Return the public state and its separately bound candidate catalog."""

        observation = self.public_observation()
        return {"observation": observation, "candidates": self.candidates()}

    def _schedule_report(self, kind: str, time: int, delay: int, **fields: Any) -> None:
        payload = {
            "kind": kind,
            "subject_id": fields.get("subject_id"),
            "task_id": fields.get("task_id"),
            "platform_id": fields.get("platform_id"),
            "depot_id": fields.get("depot_id"),
            "commodity": fields.get("commodity"),
            "quantity": fields.get("quantity"),
            "value": fields.get("value"),
            "new_energy": fields.get("new_energy"),
            "old_energy": fields.get("old_energy"),
            "requested_quantity": fields.get("requested_quantity"),
            "failure_reason": fields.get("failure_reason"),
            "report_delay": int(delay),
            "released_task_ids": list(fields.get("released_task_ids", [])),
            "failed_task_ids": list(fields.get("failed_task_ids", [])),
            "inspection_id": fields.get("inspection_id"),
            "target_type": fields.get("target_type"),
            "target_id": fields.get("target_id"),
            "old_position": copy.deepcopy(fields.get("old_position")),
            "new_position": copy.deepcopy(fields.get("new_position")),
        }
        self._log(
            "report_generated",
            time,
            kind=kind,
            subject_id=payload["subject_id"],
            task_id=payload["task_id"],
            platform_id=payload["platform_id"],
            depot_id=payload["depot_id"],
            commodity=payload["commodity"],
            quantity=payload["quantity"],
            value=payload["value"],
            new_energy=payload["new_energy"],
            old_energy=payload["old_energy"],
            requested_quantity=payload["requested_quantity"],
            failure_reason=payload["failure_reason"],
            target_type=payload["target_type"],
            target_id=payload["target_id"],
            old_position=copy.deepcopy(payload["old_position"]),
            new_position=copy.deepcopy(payload["new_position"]),
            report_delay=int(delay),
            inspection_id=payload["inspection_id"],
        )
        payload["due"] = time + int(delay)
        payload["generated_at"] = time
        self._scheduled_reports.append(payload)

    def _report_receive(self, report: dict, time: int) -> None:
        kind = report["kind"]
        subject = report.get("subject_id")
        task_id = report.get("task_id")
        fields = {key: copy.deepcopy(value) for key, value in report.items() if key not in {"due", "generated_at", "kind", "subject_id", "task_id"}}
        if kind == "platform_failure":
            platform_id = report["platform_id"]
            self._public_platforms[platform_id]["failed"] = True
            self._public_platforms[platform_id]["busy"] = False
            self._public_platforms[platform_id]["active_task_id"] = None
            for released_id in report.get("released_task_ids", []):
                if released_id in self._tasks:
                    self._public_tasks[released_id] = self._public_task_from_truth(self._tasks[released_id])
            for failed_id in report.get("failed_task_ids", []):
                if failed_id in self._tasks:
                    self._public_tasks[failed_id] = self._public_task_from_truth(self._tasks[failed_id])
        elif kind == "energy_change":
            public = self._public_platforms[report["platform_id"]]
            public["energy"] = max(
                0.0,
                float(public["energy"]) + float(report["new_energy"]) - float(report["old_energy"]),
            )
        elif kind == "stock_loss":
            depot = self._public_depots[report["depot_id"]]
            commodity = report["commodity"]
            depot["stock"][commodity] = max(0.0, float(depot["stock"].get(commodity, 0.0)) - float(report["quantity"]))
        elif kind == "spatial_perturbation":
            target_type = report["target_type"]
            target_id = report["target_id"]
            new_position = list(report["new_position"])
            if target_type == "platform":
                self._public_platforms[target_id]["position"] = new_position
            elif target_id in self._tasks and self._tasks[target_id].get("known"):
                self._public_tasks[target_id] = self._public_task_from_truth(self._tasks[target_id])
        elif kind == "task_discovered":
            hidden_id = task_id or subject
            if hidden_id in self._tasks:
                truth = self._tasks[hidden_id]
                truth["known"] = True
                truth["discovered"] = True
                truth["discovery_time"] = time
                truth["status"] = "pending"
                truth["phase"] = "report_received"
                truth["report_received_at"] = time
                self._public_tasks[hidden_id] = self._public_task_from_truth(truth)
        elif kind in {"inspection_result", "task_failure"}:
            if task_id in self._tasks:
                truth = self._tasks[task_id]
                truth["report_received_at"] = time
                if kind == "inspection_result":
                    truth["phase"] = "report_received"
                elif truth["status"] == "pending":
                    truth["phase"] = "released"
                self._public_tasks[task_id] = self._public_task_from_truth(truth)
            if kind == "task_failure" and report.get("failure_reason") not in {"platform_failure", "platform_failed_before_start"}:
                platform_id = report.get("platform_id")
                if platform_id in self._public_platforms and self._public_platforms[platform_id]["active_task_id"] == task_id:
                    self._public_platforms[platform_id]["busy"] = False
                    self._public_platforms[platform_id]["active_task_id"] = None
        if kind == "inspection_result" and task_id in self._public_tasks:
            self._public_tasks[task_id]["report_received_at"] = time
            self._public_tasks[task_id]["phase"] = "report_received"
        if kind in {"platform_failure", "energy_change"}:
            fields["platform_id"] = report["platform_id"]
        if kind == "energy_change":
            fields["old_energy"] = report["old_energy"]
            fields["new_energy"] = self._public_platforms[report["platform_id"]]["energy"]
        if kind == "stock_loss":
            fields["depot_id"] = report["depot_id"]
            fields["commodity"] = report["commodity"]
            fields["quantity"] = report["quantity"]
            fields["requested_quantity"] = report["requested_quantity"]
        self._log(
            "report_received",
            time,
            kind=kind,
            subject_id=subject,
            task_id=task_id,
            **fields,
        )

    def _apply_external_events(self, time: int) -> None:
        for disturbance in self.scenario.get("disturbances", []):
            if disturbance["time"] != time:
                continue
            kind = disturbance["kind"]
            delay = int(disturbance.get("report_delay", 0))
            if kind == "failure":
                platform_id = disturbance["platform_id"]
                platform = self._platforms[platform_id]
                if platform["failed"]:
                    continue
                platform["failed"] = True
                released: list[str] = []
                failed: list[str] = []
                active_id = platform["active_task_id"]
                if active_id is not None:
                    task = self._tasks[active_id]
                    self._fail_task(task, platform_id, "platform_failure", time, releasable=True)
                    if task["status"] == "pending":
                        released.append(active_id)
                    elif task["status"] == "failed":
                        failed.append(active_id)
                for task_id in self._queues.get(platform_id, []):
                    task = self._tasks[task_id]
                    if task["status"] in {"completed", "failed"} or task["active_platform_id"] is not None:
                        continue
                    if task["assignee"] != platform_id:
                        continue
                    self._fail_task(task, platform_id, "platform_failed_before_start", time, releasable=True)
                    if task["status"] == "pending":
                        released.append(task_id)
                    elif task["status"] == "failed":
                        failed.append(task_id)
                self._log("platform_failed", time, platform_id=platform_id, disturbance_id=disturbance["id"])
                self._schedule_report(
                    "platform_failure", time, delay, platform_id=platform_id,
                    released_task_ids=released, failed_task_ids=failed,
                )
            elif kind == "energy_change":
                platform_id = disturbance["platform_id"]
                platform = self._platforms[platform_id]
                old = float(platform["energy"])
                new = float(disturbance["value"])
                platform["energy"] = new
                self._log(
                    "energy_changed", time, platform_id=platform_id,
                    old_energy=old, new_energy=new, delta=new - old,
                    disturbance_id=disturbance["id"],
                )
                self._schedule_report(
                    "energy_change", time, delay, platform_id=platform_id,
                    value=new, old_energy=old, new_energy=new,
                )
            elif kind == "stock_loss":
                depot_id = disturbance["depot_id"]
                commodity = disturbance["commodity"]
                requested_quantity = float(disturbance["quantity"])
                depot = self._depots[depot_id]
                available = max(0.0, float(depot["stock"].get(commodity, 0.0)))
                quantity = min(requested_quantity, available)
                depot["stock"][commodity] = max(0.0, available - quantity)
                self._log(
                    "stock_lost", time, depot_id=depot_id, commodity=commodity,
                    quantity=quantity, requested_quantity=requested_quantity,
                    disturbance_id=disturbance["id"],
                )
                self._schedule_report(
                    "stock_loss", time, delay, depot_id=depot_id,
                    commodity=commodity, quantity=quantity,
                    requested_quantity=requested_quantity,
                )
            else:
                target_type = disturbance["target_type"]
                target_id = disturbance["target_id"]
                new_position = list(disturbance["position"])
                if target_type == "platform":
                    target = self._platforms[target_id]
                else:
                    target = self._tasks[target_id]
                old_position = list(target["position"])
                target["position"] = new_position
                self._log(
                    "spatial_perturbation", time,
                    target_type=target_type, target_id=target_id,
                    old_position=old_position, new_position=new_position,
                    disturbance_id=disturbance["id"],
                )
                self._schedule_report(
                    "spatial_perturbation", time, delay,
                    target_type=target_type, target_id=target_id,
                    old_position=old_position, new_position=new_position,
                )

    def _receive_reports(self, time: int) -> None:
        due = [report for report in self._scheduled_reports if report["due"] <= time]
        self._scheduled_reports = [report for report in self._scheduled_reports if report["due"] > time]
        for report in due:
            self._report_receive(report, time)

    def _expire_deadlines(self, time: int) -> None:
        for task_id in sorted(self._tasks):
            task = self._tasks[task_id]
            if task["deadline_event_emitted"] or time < task["deadline"]:
                continue
            if task["status"] in {"completed", "failed"}:
                continue
            task["deadline_missed"] = True
            task["deadline_event_emitted"] = True
            task["phase"] = task["phase"]
            self._log("task_expired", time, task_id=task_id, deadline=task["deadline"], terminal=False)
            if task_id in self._public_tasks:
                self._public_tasks[task_id]["deadline_missed"] = True

    def _mission_feasible_in_truth(self, platform: dict, task: dict) -> tuple[bool, str | None, float]:
        if platform["failed"]:
            return False, "platform_failed", 0.0
        if platform["active_task_id"] is not None:
            return False, "platform_busy", 0.0
        if task["status"] in {"completed", "failed", "undiscovered"} or task["active_platform_id"] is not None:
            return False, "task_unavailable", 0.0
        if not self._compatible(platform, task):
            return False, "capability_or_locality", 0.0
        completion, required_energy = self._estimated_mission(platform, task, self.time)
        del completion
        if float(platform["energy"]) + _EPS < required_energy:
            return False, "insufficient_energy", required_energy
        if task["kind"] == "delivery":
            if float(task["quantity"]) > float(platform["capacity_kg"]) + _EPS:
                return False, "capacity_exceeded", required_energy
            stock = float(self._depots[task["depot_id"]]["stock"].get(task["commodity"], 0.0))
            if stock + _EPS < float(task["quantity"]):
                return False, "insufficient_stock", required_energy
        return True, None, required_energy

    def _dispatch(self, action: dict, time: int) -> None:
        platform_id = action["platform_id"]
        task_id = action["task_id"]
        platform = self._platforms[platform_id]
        task = self._tasks[task_id]
        feasible, reason, _ = self._mission_feasible_in_truth(platform, task)
        if not feasible:
            self._log(
                "dispatch_rejected", time, task_id=task_id, platform_id=platform_id,
                reason=reason or "infeasible", candidate_id=action["id"],
            )
            return

        old_owner = task["assignee"]
        if old_owner != platform_id or task["original_assignee"] is None:
            reason = "C1_new_task" if task["original_assignee"] is None and old_owner is None else "C1_dispatch"
            if self._policy == "C0":
                reason = "C0_snapshot_dispatch"
            elif self._policy not in {"C1", "C0"}:
                reason = "learned_new_task" if task["original_assignee"] is None and old_owner is None else "learned_dispatch"
            self._log(
                "assignment", time, task_id=task_id,
                previous_assignee=old_owner, new_assignee=platform_id,
                initial=False, reason=reason,
            )
            if task["original_assignee"] is None:
                task["_new_assignment_counted"] = True
            elif platform_id != task["original_assignee"]:
                task["_reassigned"] = True
        task["assignee"] = platform_id
        task["active_platform_id"] = platform_id
        task["status"] = "active"
        task["phase"] = "assigned"
        platform["active_task_id"] = task_id
        public_task = self._public_tasks[task_id]
        public_task.update({"assignee": platform_id, "status": "active", "phase": "assigned"})
        public_platform = self._public_platforms[platform_id]
        public_platform.update({"busy": True, "active_task_id": task_id})

        if task["kind"] == "inspection":
            task["phase"] = "to_inspection"
            public_task["phase"] = "to_inspection"
            target = task["position"]
            stage = "inspection_site"
        else:
            task["phase"] = "to_depot"
            public_task["phase"] = "to_depot"
            target = self._depots[task["depot_id"]]["position"]
            stage = "depot"
        if _distance(platform["position"], target) <= _EPS:
            self._arrive(platform_id, task_id, stage, time)

    def _arrive(self, platform_id: str, task_id: str, stage: str, time: int) -> None:
        self._mark_started(self._tasks[task_id], time)
        self._log("arrival", time, platform_id=platform_id, task_id=task_id, stage=stage, position=list(self._platforms[platform_id]["position"]))
        task = self._tasks[task_id]
        if task["kind"] == "inspection":
            task["phase"] = "observe"
        elif stage == "depot":
            task["phase"] = "loading"
        elif stage == "delivery_site":
            task["phase"] = "unloading"
        if task_id in self._public_tasks:
            self._public_tasks[task_id]["phase"] = task["phase"]

    def _free_platform(self, platform_id: str, *, update_public: bool = True) -> None:
        self._platforms[platform_id]["active_task_id"] = None
        if update_public:
            self._public_platforms[platform_id]["busy"] = False
            self._public_platforms[platform_id]["active_task_id"] = None

    def _fail_task(self, task: dict, platform_id: str, reason: str, time: int, *, releasable: bool) -> None:
        task_id = task["id"]
        cargo = float(task["cargo_onboard"])
        if cargo > _EPS:
            self._log(
                "cargo_lost", time, task_id=task_id, platform_id=platform_id,
                depot_id=task["depot_id"], commodity=task["commodity"], quantity=cargo,
            )
            self._cargo[(task["depot_id"], platform_id, task_id, task["commodity"])] = 0.0
            task["cargo_onboard"] = 0.0
            releasable = False
        terminal = not (self._policy in {"C1", "L0", "L1", "LEARNED"} and releasable and cargo <= _EPS)
        self._log(
            "task_failed", time, task_id=task_id, platform_id=platform_id,
            reason=reason, terminal=terminal, released=not terminal,
        )
        task["failure_reason"] = reason
        task["active_platform_id"] = None
        if terminal:
            task["status"] = "failed"
            task["phase"] = "failed"
            task["assignee"] = platform_id
        else:
            task["status"] = "pending"
            task["phase"] = "released"
            task["assignee"] = None
            task["service_progress"] = 0.0
        self._free_platform(platform_id, update_public=False)
        self._schedule_report(
            "task_failure", time, 1, task_id=task_id, subject_id=task_id,
            platform_id=platform_id, released_task_ids=[task_id] if not terminal else [],
            failed_task_ids=[task_id] if terminal else [],
            failure_reason=reason,
        )

    def _consume_energy(self, platform: dict, amount: float) -> bool:
        amount = max(0.0, float(amount))
        if float(platform["energy"]) + _EPS < amount:
            return False
        platform["energy"] = max(0.0, float(platform["energy"]) - amount)
        platform["energy_used"] = float(platform["energy_used"]) + amount
        public = self._public_platforms[platform["id"]]
        public["energy"] = max(0.0, float(public["energy"]) - amount)
        return True

    def _move_toward(self, platform_id: str, task_id: str, target: list[float], stage: str, end_time: int) -> None:
        platform = self._platforms[platform_id]
        start = list(platform["position"])
        distance_left = _distance(start, target)
        if distance_left <= _EPS:
            self._arrive(platform_id, task_id, stage, end_time)
            return
        speed = float(platform["speed_km_min"])
        distance = min(distance_left, speed)
        energy_per_km = float(platform["energy_per_km"])
        planned_energy = distance * energy_per_km
        if planned_energy > float(platform["energy"]) + _EPS:
            if energy_per_km <= _EPS:
                distance = 0.0
            else:
                distance = min(distance, float(platform["energy"]) / energy_per_km)
            planned_energy = distance * energy_per_km
        if distance <= _EPS:
            self._fail_task(self._tasks[task_id], platform_id, "insufficient_energy", end_time - 1, releasable=True)
            return
        ratio = distance / distance_left
        end = [start[0] + (target[0] - start[0]) * ratio, start[1] + (target[1] - start[1]) * ratio]
        self._consume_energy(platform, planned_energy)
        platform["position"] = end
        self._public_platforms[platform_id]["position"] = list(end)
        duration = distance / speed
        self._mark_started(self._tasks[task_id], end_time)
        self._log(
            "move", end_time, platform_id=platform_id, task_id=task_id,
            **{"from": start, "to": list(end)}, distance=distance,
            duration=duration, energy=planned_energy,
        )
        if distance + _EPS >= distance_left:
            self._arrive(platform_id, task_id, stage, end_time)
        elif float(platform["energy"]) <= _EPS:
            self._fail_task(self._tasks[task_id], platform_id, "insufficient_energy", end_time, releasable=True)

    def _service(self, platform_id: str, task_id: str, stage: str, end_time: int) -> bool:
        platform = self._platforms[platform_id]
        task = self._tasks[task_id]
        energy = float(platform["service_energy_per_min"])
        if float(platform["energy"]) + _EPS < energy:
            self._fail_task(task, platform_id, "insufficient_energy", end_time, releasable=True)
            return False
        if not self._consume_energy(platform, energy):
            self._fail_task(task, platform_id, "insufficient_energy", end_time, releasable=True)
            return False
        task["service_progress"] += 1.0
        self._mark_started(task, end_time)
        self._log(
            "service", end_time, task_id=task_id, platform_id=platform_id,
            stage=stage, duration=1.0, energy=energy,
        )
        return True

    def _mark_started(self, task: dict, time: int) -> None:
        if task["started_at"] is not None:
            return
        task["started_at"] = time
        if task["id"] in self._public_tasks:
            self._public_tasks[task["id"]]["started_at"] = time

    def _complete_inspection(self, task: dict, platform_id: str, time: int) -> None:
        task_id = task["id"]
        task["status"] = "completed"
        task["phase"] = "inspection_completed"
        task["completed_at"] = time
        task["active_platform_id"] = None
        task["on_time"] = time <= task["deadline"]
        self._log("inspection_completed", time, task_id=task_id, platform_id=platform_id)
        public_task = self._public_tasks[task_id]
        public_task.update({"status": "completed", "phase": "inspection_completed", "completed_at": time})
        self._free_platform(platform_id)
        task["report_generated_at"] = time
        self._schedule_report(
            "inspection_result", time, int(task.get("report_delay", 0)),
            subject_id=task_id, task_id=task_id, platform_id=platform_id,
        )
        for hidden_id, hidden in sorted(self._tasks.items()):
            if hidden.get("discovered_by") != task_id or hidden["discovered"]:
                continue
            self._schedule_report(
                "task_discovered", time, int(hidden.get("report_delay", 0)),
                subject_id=hidden_id, task_id=hidden_id, inspection_id=task_id,
            )

    def _load_cargo(self, task: dict, platform_id: str, time: int) -> bool:
        depot = self._depots[task["depot_id"]]
        commodity = task["commodity"]
        quantity = float(task["quantity"])
        available = float(depot["stock"].get(commodity, 0.0))
        if available + _EPS < quantity:
            self._fail_task(task, platform_id, "insufficient_stock", time, releasable=True)
            return False
        depot["stock"][commodity] = max(0.0, available - quantity)
        public_depot = self._public_depots[task["depot_id"]]
        public_depot["stock"][commodity] = max(0.0, float(public_depot["stock"].get(commodity, 0.0)) - quantity)
        self._cargo[(task["depot_id"], platform_id, task["id"], commodity)] += quantity
        task["cargo_onboard"] = quantity
        task["phase"] = "in_transit"
        task["service_progress"] = 0.0
        self._log(
            "loaded", time, task_id=task["id"], platform_id=platform_id,
            depot_id=task["depot_id"], commodity=commodity, quantity=quantity,
        )
        self._public_tasks[task["id"]]["phase"] = "in_transit"
        return True

    def _confirm_delivery(self, task: dict, platform_id: str, time: int) -> None:
        task_id = task["id"]
        quantity = float(task["cargo_onboard"])
        self._cargo[(task["depot_id"], platform_id, task_id, task["commodity"])] = 0.0
        task["cargo_onboard"] = 0.0
        task["status"] = "completed"
        task["phase"] = "delivery_confirmed"
        task["completed_at"] = time
        task["delivered"] = True
        task["on_time"] = time <= task["deadline"]
        task["active_platform_id"] = None
        self._log(
            "delivery_confirmed", time, task_id=task_id, platform_id=platform_id,
            depot_id=task["depot_id"], commodity=task["commodity"], quantity=quantity,
        )
        self._public_tasks[task_id].update({"status": "completed", "phase": "delivery_confirmed", "completed_at": time})
        self._free_platform(platform_id)

    def _advance(self, start_time: int) -> None:
        end_time = start_time + 1
        for platform_id in sorted(self._platforms):
            platform = self._platforms[platform_id]
            task_id = platform["active_task_id"]
            if task_id is None or platform["failed"]:
                continue
            task = self._tasks[task_id]
            phase = task["phase"]
            if phase == "to_inspection":
                self._move_toward(platform_id, task_id, task["position"], "inspection_site", end_time)
            elif phase == "to_depot":
                self._move_toward(platform_id, task_id, self._depots[task["depot_id"]]["position"], "depot", end_time)
            elif phase == "to_delivery_site":
                self._move_toward(platform_id, task_id, task["position"], "delivery_site", end_time)
            elif phase == "observe":
                if self._service(platform_id, task_id, "observe", end_time):
                    if task["service_progress"] + _EPS >= float(task["service_minutes"]):
                        self._complete_inspection(task, platform_id, end_time)
            elif phase == "loading":
                if self._service(platform_id, task_id, "loading", end_time):
                    if task["service_progress"] + _EPS >= float(task["load_minutes"]):
                        if self._load_cargo(task, platform_id, end_time):
                            task["phase"] = "in_transit"
                            self._tasks[task_id]["phase"] = "to_delivery_site"
                            self._public_tasks[task_id]["phase"] = "to_delivery_site"
                            if _distance(platform["position"], task["position"]) <= _EPS:
                                self._arrive(platform_id, task_id, "delivery_site", end_time)
            elif phase == "in_transit":
                # A loaded task enters in_transit on the loading-completion
                # event; movement begins on the next interval.
                task["phase"] = "to_delivery_site"
                self._public_tasks[task_id]["phase"] = "to_delivery_site"
            elif phase == "unloading":
                if self._service(platform_id, task_id, "unloading", end_time):
                    if task["service_progress"] + _EPS >= float(task["unload_minutes"]):
                        self._confirm_delivery(task, platform_id, end_time)

    def _apply_arrival_phase(self, time: int) -> None:
        # Arrival events set the physical next phase; the transition is kept
        # here so each interval advances one spatial or service stage only.
        for task in self._tasks.values():
            if task["active_platform_id"] is None:
                continue
            if task["phase"] == "loading" and task["service_progress"] == 0 and task["kind"] == "delivery":
                continue

    def _close_state(self) -> dict:
        for task in self._tasks.values():
            if task["status"] in {"completed", "failed", "undiscovered"}:
                continue
            task["status"] = "expired"
        tasks = []
        for task_id in sorted(self._tasks):
            task = self._tasks[task_id]
            tasks.append({
                "id": task_id,
                "kind": task["kind"],
                "status": task["status"],
                "phase": task["phase"],
                "assignee": task["assignee"],
                "original_assignee": task["original_assignee"],
                "known": bool(task["known"]),
                "discovered": bool(task["discovered"]),
                "started_at": task["started_at"],
                "completed_at": task["completed_at"],
                "deadline": task["deadline"],
                "deadline_missed": bool(task["deadline_missed"]),
                "report_generated_at": task["report_generated_at"],
                "report_received_at": task["report_received_at"],
                "failure_reason": task["failure_reason"],
                "cargo_onboard": _clean_number(float(task["cargo_onboard"])),
                "delivered": bool(task["delivered"]),
                "on_time": task["on_time"],
            })
        platforms = []
        for platform_id in sorted(self._platforms):
            platform = self._platforms[platform_id]
            platforms.append({
                "id": platform_id,
                "position": list(platform["position"]),
                "energy": _clean_number(float(platform["energy"])),
                "energy_used": _clean_number(float(platform["energy_used"])),
                "failed": bool(platform["failed"]),
                "active_task_id": platform["active_task_id"],
                "capacity_kg": platform["capacity_kg"],
            })
        depots = [
            {"id": depot_id, "position": list(self._depots[depot_id]["position"]),
             "stock": {commodity: _clean_number(float(amount)) for commodity, amount in sorted(self._depots[depot_id]["stock"].items())}}
            for depot_id in sorted(self._depots)
        ]
        onboard = []
        for (depot_id, platform_id, task_id, commodity), quantity in sorted(self._cargo.items()):
            if quantity > _EPS:
                onboard.append({
                    "depot_id": depot_id, "platform_id": platform_id,
                    "task_id": task_id, "commodity": commodity,
                    "quantity": _clean_number(float(quantity)),
                })
        return {"time": self.scenario["horizon"], "platforms": platforms, "tasks": tasks, "depots": depots, "onboard": onboard}

    def _summary(self, final_state: dict) -> dict:
        tasks = final_state["tasks"]
        inspections = [task for task in tasks if task["kind"] == "inspection"]
        orders = [task for task in tasks if task["kind"] == "delivery"]
        delivery_weight = sum(float(self._task_spec[task["id"]]["priority"]) for task in orders)
        on_time_weight = sum(
            float(self._task_spec[task["id"]]["priority"])
            for task in orders if task["delivered"] and task["on_time"]
        )
        energy_used = sum(float(platform["energy_used"]) for platform in final_state["platforms"])
        return {
            "weighted_on_time_delivery_rate": None if not orders else on_time_weight / delivery_weight,
            "inspection_completion_rate": None if not inspections else sum(task["status"] == "completed" for task in inspections) / len(inspections),
            "inspection_on_time_rate": None if not inspections else sum(task["status"] == "completed" and task["on_time"] for task in inspections) / len(inspections),
            "delivered_orders": sum(task["delivered"] for task in orders),
            "late_orders": sum(task["delivered"] and not task["on_time"] for task in orders),
            "unmet_orders": sum(not task["delivered"] for task in orders),
            "undiscovered_orders": sum(not task["discovered"] for task in orders),
            "energy_used": _clean_number(energy_used),
            "reassignments": sum(bool(task.get("_reassigned")) for task in self._tasks.values()),
            "new_assignments": sum(bool(task.get("_new_assignment_counted")) for task in self._tasks.values()),
        }

    def run(self, policy: str) -> dict:
        """Simulate through the fixed horizon and return event-level evidence."""

        if policy not in {"C0", "C1"}:
            raise ValueError("policy must be C0 or C1")
        self._reset()
        self._policy = policy
        self._reassignments = 0
        self._new_assignments = 0
        for task_id, owner in sorted(self._original_owner.items()):
            self._log(
                "assignment", 0, task_id=task_id,
                previous_assignee=owner, new_assignee=owner,
                initial=True, reason="chapter1_snapshot",
            )

        horizon = int(self.scenario["horizon"])
        for time in range(horizon + 1):
            self.time = time
            self._apply_external_events(time)
            self._receive_reports(time)
            self._expire_deadlines(time)
            if time == horizon:
                break
            observation = self.public_observation()
            catalog = self.candidates()
            identity = bind_decision(observation, catalog)
            actions = choose_actions(policy, observation, catalog)
            self.transitions.append({
                "time": time,
                "public_observation": observation,
                "candidates": catalog,
                "selected_actions": copy.deepcopy(actions),
                "identity": identity,
                "observation_sha256": identity["observation_sha256"],
                "candidate_catalog_sha256": identity["candidate_catalog_sha256"],
            })
            assigned_platforms: set[str] = set()
            assigned_tasks: set[str] = set()
            for action in actions:
                platform_id = action["platform_id"]
                task_id = action["task_id"]
                if platform_id in assigned_platforms or task_id in assigned_tasks:
                    continue
                assigned_platforms.add(platform_id)
                assigned_tasks.add(task_id)
                self._dispatch(action, time)
            self._advance(time)

        final_state = self._close_state()
        summary = self._summary(final_state)
        return {
            "scenario_id": self.scenario["scenario_id"],
            "policy": policy,
            "initial_handoff": copy.deepcopy(self.initial_handoff),
            "events": copy.deepcopy(self.events),
            "transitions": copy.deepcopy(self.transitions),
            "final_state": final_state,
            "summary": summary,
        }
