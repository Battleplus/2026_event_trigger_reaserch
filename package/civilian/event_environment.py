"""Executable public-information simulator extension for event-trigger experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .baselines import choose_step_c1_actions
    from .relief_env import ReliefEnv
    from .scenario_schema import validate_scenario
    from .event_scenarios import validate_event_scenario
    from .gain_scenarios import validate_gain_scenario
except ImportError:
    from baselines import choose_step_c1_actions
    from relief_env import ReliefEnv
    from scenario_schema import validate_scenario
    from event_scenarios import validate_event_scenario
    from gain_scenarios import validate_gain_scenario


_EPS = 1e-9
_MAX_CANDIDATES = 16
_COORDINATOR = "n02-uav-1"
_REWARD_KEYS = ("completion", "delay", "energy", "risk", "communication", "switch", "reconstruction")


def validate_supported_event_scenario(scenario: dict) -> dict:
    """Dispatch gain study scenarios to their expanded split validator."""
    if isinstance(scenario, dict) and "parent_family" in scenario:
        return validate_gain_scenario(scenario)
    return validate_event_scenario(scenario)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _new_action(scope: str = "LOCAL_REPAIR", *, noop: bool = False,
                queue_operations: list[dict] | None = None, dispatches: list[dict] | None = None) -> dict:
    return {
        "action_type": "reconstruct", "scope": "NOOP" if noop else scope, "noop": bool(noop),
        "queue_operations": copy.deepcopy(queue_operations or []),
        "dispatches": copy.deepcopy(dispatches or []),
    }


def _link_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((str(source), str(target))))


def _communication_metrics(observation: dict, start: str, end: str) -> dict[str, float | bool]:
    """Return the best public connected path from ``start`` to ``end``.

    Communication metadata is public state, so this helper is also used by
    candidate ranking and command-delivery receipts.  Path quality is
    multiplicative, loss is the probability of at least one lost packet, and
    latency is additive.  Ties are resolved by the first path discovered from
    sorted public links, keeping the result deterministic.
    """
    if start == end:
        return {"connected": True, "quality": 1.0, "latency_minutes": 0.0, "loss_rate": 0.0}
    nodes = {str(row.get("id")) for row in observation.get("platforms", []) if isinstance(row, dict)}
    if start not in nodes or end not in nodes:
        return {"connected": False, "quality": 0.0, "latency_minutes": 0.0, "loss_rate": 1.0}
    adjacency: dict[str, list[tuple[str, float, float, float]]] = {node: [] for node in nodes}
    for row in sorted(observation.get("communication", []), key=lambda item: _link_key(item.get("source", ""), item.get("target", ""))):
        if not isinstance(row, dict):
            continue
        left, right = str(row.get("source")), str(row.get("target"))
        if left not in nodes or right not in nodes:
            continue
        quality = max(0.0, min(1.0, float(row.get("quality", 0.0))))
        if not bool(row.get("connected", quality >= 0.3)) or quality <= _EPS:
            continue
        latency = max(0.0, float(row.get("latency_minutes", 0.0)))
        loss = max(0.0, min(1.0, float(row.get("loss_rate", 1.0 - quality))))
        adjacency[left].append((right, quality, latency, loss))
        adjacency[right].append((left, quality, latency, loss))
    best: dict[str, tuple[float, float, float]] = {start: (1.0, 0.0, 0.0)}
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        quality, latency, loss = best[current]
        for neighbor, edge_quality, edge_latency, edge_loss in adjacency[current]:
            candidate = (
                quality * edge_quality,
                latency + edge_latency,
                1.0 - (1.0 - loss) * (1.0 - edge_loss),
            )
            prior = best.get(neighbor)
            if prior is None or candidate[0] > prior[0] + _EPS or (
                abs(candidate[0] - prior[0]) <= _EPS and candidate[1] < prior[1] - _EPS
            ):
                best[neighbor] = candidate
                frontier.append(neighbor)
    if end not in best:
        return {"connected": False, "quality": 0.0, "latency_minutes": 0.0, "loss_rate": 1.0}
    quality, latency, loss = best[end]
    return {"connected": True, "quality": quality, "latency_minutes": latency, "loss_rate": loss}


def _communication_delay_minutes(metrics: dict[str, float | bool]) -> int:
    """Convert public latency/loss into a deterministic command delay.

    Sub-minute links remain immediate, matching the minute-step simulator.
    Whole-minute latency and expected packet retries delay delivery without
    introducing stochastic outcomes into otherwise deterministic episodes.
    """
    latency = max(0.0, float(metrics.get("latency_minutes", 0.0)))
    loss = max(0.0, min(1.0, float(metrics.get("loss_rate", 0.0))))
    latency_ticks = max(0, int(math.ceil(latency - 0.1 - _EPS)))
    retry_ticks = int(loss / max(1.0 - loss, _EPS))
    return latency_ticks + retry_ticks


def _public_route_distance(observation: dict, platform: dict, task: dict) -> float:
    """Estimate route distance using only visible platform/task/depot fields."""
    start = list(platform.get("position", [0.0, 0.0]))
    target = list(task.get("position", start))
    if task.get("kind") != "delivery":
        return _distance(start, target)
    depot = next((row for row in observation.get("depots", []) if row.get("id") == task.get("depot_id")), None)
    depot_position = list(depot.get("position", start)) if depot else start
    return _distance(start, depot_position) + _distance(depot_position, target)


def _dispatch_rank_key(observation: dict, candidate: dict) -> tuple:
    """Stable public-information ordering for feasible dispatch candidates."""
    tasks = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
    platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    task = tasks.get(str(candidate.get("task_id")), {})
    platform = platforms.get(str(candidate.get("platform_id")), {})
    deadline = min(float(observation.get("horizon", 10**9)), float(task.get("deadline", observation.get("horizon", 10**9))))
    completion = float(candidate.get("estimated_completion", observation.get("horizon", 10**9)))
    slack = deadline - completion
    needed = "inspect" if task.get("kind") == "inspection" else "deliver"
    capability_penalty = int(needed not in platform.get("capabilities", []))
    capacity_margin = float(platform.get("capacity_kg", 0.0)) - float(task.get("quantity", 0.0))
    owner = task.get("assignee")
    source = str(owner) if isinstance(owner, str) else _COORDINATOR
    communication = _communication_metrics(observation, source, str(candidate.get("platform_id", "")))
    queue = observation.get("queues", {}).get(str(candidate.get("platform_id", "")), [])
    queue_switch = int(owner != candidate.get("platform_id")) + int(str(candidate.get("task_id")) in queue and queue.index(str(candidate.get("task_id"))) > 0)
    return (
        -float(task.get("priority", 0.0)), slack, capability_penalty, max(0.0, capacity_margin),
        _public_route_distance(observation, platform, task), float(candidate.get("required_energy", 10**9)),
        1.0 - float(communication.get("quality", 0.0)), float(communication.get("latency_minutes", 10**9)),
        float(communication.get("loss_rate", 1.0)), queue_switch,
        str(candidate.get("platform_id", "")), str(candidate.get("task_id", "")), str(candidate.get("id", "")),
    )


class EventReliefEnv:
    """Wrap the frozen minute-step relief simulator with public graph events."""

    def __init__(self, scenario: dict, reward_config: dict | None = None):
        validate_supported_event_scenario(scenario)
        self.scenario = copy.deepcopy(scenario)
        self.reward_config = self._read_reward_config(reward_config)
        base_scenario = copy.deepcopy(scenario["base_scenario"])
        for row in scenario["event_tape"]:
            if row["kind"] == "platform_failure":
                base_scenario["disturbances"].append({
                    "id": row["event_id"], "time": row["time"], "kind": "failure", "report_delay": 0,
                    "platform_id": row["platform_id"],
                })
            elif row["kind"] == "energy_drop":
                base_scenario["disturbances"].append({
                    "id": row["event_id"], "time": row["time"], "kind": "energy_change", "report_delay": 0,
                    "platform_id": row["platform_id"], "value": row["new_energy"],
                })
        validate_scenario(base_scenario)
        self._base = ReliefEnv(base_scenario)
        self._original_task_specs = {task["id"]: copy.deepcopy(task) for task in base_scenario["tasks"]}
        self._task_specs = copy.deepcopy(self._original_task_specs)
        self._task_predecessors: dict[str, list[str]] = {
            "n02-task-04": ["n02-task-03"], "n02-task-06": ["n02-task-04"],
            "n02-task-08": ["n02-task-07"], "n02-task-12": ["n02-task-10"],
        }
        self._cooperation: dict[str, list[str]] = {"n02-task-01": ["n02-task-02"], "n02-task-02": ["n02-task-01"]}
        self._base_environment = [
            {"id": f"env-cell-{x}-{y}", "kind": "cell", "grid_cell": [x, y],
             "wind_vector": [0.0, 0.0], "risk_level": 0.05, "blocked": False}
            for y in range(40, 42) for x in range(40, 42)
        ]
        self._initial_communication = [
            {"source": "n02-uav-1", "target": "n02-uav-2", "quality": 0.96, "latency_minutes": 0.05, "loss_rate": 0.02},
            {"source": "n02-uav-2", "target": "n02-uav-3", "quality": 0.95, "latency_minutes": 0.06, "loss_rate": 0.03},
            {"source": "n02-uav-3", "target": "n02-uav-4", "quality": 0.95, "latency_minutes": 0.06, "loss_rate": 0.03},
            {"source": "n02-uav-4", "target": "n02-heli-1", "quality": 0.92, "latency_minutes": 0.08, "loss_rate": 0.04},
            {"source": "n02-heli-1", "target": "n02-heli-2", "quality": 0.94, "latency_minutes": 0.08, "loss_rate": 0.04},
        ]
        self._reset_fields()

    @staticmethod
    def _read_reward_config(reward_config: dict | None) -> dict:
        if reward_config is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "EVENT_TRIGGER_CONFIG.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            reward_config = config.get("reward")
        if not isinstance(reward_config, dict) or set(reward_config) != set(_REWARD_KEYS) | {"reconstruction_scale"}:
            raise ValueError("reward configuration must match EVENT_TRIGGER_CONFIG.reward")
        normalized = {key: float(reward_config[key]) for key in _REWARD_KEYS + ("reconstruction_scale",)}
        if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
            raise ValueError("reward weights must be finite and nonnegative")
        return normalized

    def _reset_fields(self) -> None:
        self._task_specs = copy.deepcopy(self._original_task_specs)
        self._environment = copy.deepcopy(self._base_environment)
        self._communication = copy.deepcopy(self._initial_communication)
        self._event_ids_applied: set[str] = set()
        self._extension_events: list[dict] = []
        self._steps: list[dict] = []
        self._public_history: list[dict] = []
        self._queues: dict[str, list[str]] = {}
        self._split_manifests: dict[str, dict] = {}
        self._reported_platforms: dict[str, dict] = {}
        self._reported_tasks: dict[str, dict] = {}
        self._reported_depots: dict[str, dict] = {}
        self._reported_progress_counts: dict[str, float] = {}
        self._reported_energy_used: dict[str, float] = {}
        self._pending_communications: list[dict] = []
        # Current-time safety holds are explicit simulator state.  A held
        # task remains an unfinished obligation with its physical assignment
        # and progress intact; only its minute-by-minute execution is paused.
        self._safety_paused_tasks: dict[str, dict] = {}
        self.last_candidate_rejections: list[dict] = []
        self._has_reset = False

    def reset(self) -> dict:
        self._reset_fields()
        self._base._task_spec = copy.deepcopy(self._original_task_specs)
        self._base.reset("LEARNED")
        initial_public = self._base.public_observation()
        self._reported_platforms = {row["id"]: copy.deepcopy(row) for row in initial_public["platforms"]}
        self._reported_tasks = {row["id"]: copy.deepcopy(row) for row in initial_public["tasks"]}
        self._reported_depots = {row["id"]: copy.deepcopy(row) for row in initial_public["depots"]}
        self._reported_progress_counts = {task_id: 0.0 for task_id in self._reported_tasks}
        self._reported_energy_used = {platform_id: 0.0 for platform_id in self._reported_platforms}
        for platform_id, row in self._reported_platforms.items():
            row["energy_used"] = 0.0
        self._queues = {key: list(value) for key, value in self.scenario["queue_plan"].items()}
        for platform_id, queue in self._queues.items():
            for task_id in queue:
                task = self._base._tasks[task_id]
                task["assignee"] = platform_id
                task["status"] = "pending"
                task["phase"] = "pending"
                public_task = self._base._public_tasks[task_id]
                public_task.update({"assignee": platform_id, "status": "pending", "phase": "pending"})
                self._reported_tasks[task_id].update({"assignee": platform_id, "status": "pending", "phase": "pending"})
        self._base._queues = copy.deepcopy(self._queues)
        self._apply_extension_events(int(self._base.time))
        self._has_reset = True
        observation = self.public_observation()
        self._public_history.append(copy.deepcopy(observation))
        return observation

    @property
    def time(self) -> int:
        return int(self._base.time)

    @property
    def event_tape_id(self) -> str:
        return str(self.scenario["event_tape_id"])

    def _platform_load(self, platform_id: str) -> dict:
        active = self._reported_platforms.get(platform_id, {}).get("active_task_id")
        return {"active_tasks": int(active is not None), "queued_tasks": len(self._queues.get(platform_id, []))}

    def public_observation(self) -> dict:
        """Return only current/reported state, public history hashes, and provenance."""
        if not self._has_reset:
            raise RuntimeError("call reset() before requesting public observations")
        # Snapshots produced before the V2 safety extension may not carry the
        # optional hold map.  Normalize that state on first read so replay of
        # historical trajectories remains compatible.
        if not isinstance(getattr(self, "_safety_paused_tasks", None), dict):
            self._safety_paused_tasks = {}
        observation = {
            "schema": "event-relief-public-observation/1", "time": self.time,
            "horizon": self._base.scenario["horizon"], "map": copy.deepcopy(self._base.scenario["map"]),
            "platforms": [copy.deepcopy(self._reported_platforms[key]) for key in sorted(self._reported_platforms)],
            "depots": [copy.deepcopy(self._reported_depots[key]) for key in sorted(self._reported_depots)],
            "tasks": [copy.deepcopy(self._reported_tasks[key]) for key in sorted(self._reported_tasks)],
            "chapter1_snapshot": {
                "source": self._base.scenario["chapter1_snapshot"]["source"],
                "time": self._base.scenario["chapter1_snapshot"]["time"],
                "queues": copy.deepcopy(self._base.scenario["chapter1_snapshot"]["queues"]),
            },
        }
        tasks = {row["id"]: row for row in observation["tasks"]}
        for task_id, row in tasks.items():
            spec = self._task_specs[task_id]
            total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
            row["service_progress"] = float(self._reported_progress_counts.get(task_id, 0.0))
            row["progress"] = min(1.0, row["service_progress"] / max(1.0, total))
            row["predecessors"] = list(self._task_predecessors.get(task_id, []))
            row["cooperative_with"] = list(self._cooperation.get(task_id, []))
            parent_id = spec.get("parent_task_id")
            if parent_id is not None:
                row["parent_task_id"] = str(parent_id)
                row["workload_share"] = float(spec["workload_share"])
                if spec.get("cooperation_group"):
                    row["cooperation_group"] = str(spec["cooperation_group"])
            hold = self._safety_paused_tasks.get(task_id)
            row["safety_paused"] = bool(hold)
            if hold and row.get("status") == "active":
                row["status"] = "paused"
        for parent_id, manifest in self._split_manifests.items():
            parent = tasks.get(parent_id)
            if parent is None:
                continue
            children = [tasks.get(child_id) for child_id in manifest["child_ids"]]
            present = [child for child in children if child is not None]
            completed = bool(present) and len(present) == len(children) and all(child.get("status") == "completed" for child in present)
            progress = sum(float(child.get("progress", 0.0)) * float(child.get("workload_share", 0.0)) for child in present)
            parent.update({"status": "completed" if completed else "split_parent",
                           "phase": "completed" if completed else "split_parent",
                           "progress": min(1.0, progress), "service_progress": progress,
                           "subtask_ids": list(manifest["child_ids"]),
                           "team_owners": [self._task_specs[child_id]["assigned_owner"] for child_id in manifest["child_ids"]]})
        platform_rows = observation["platforms"]
        for row in platform_rows:
            platform_id = row["id"]
            row["health"] = "failed" if row.get("failed") else "healthy"
            row["load"] = self._platform_load(platform_id)
            row["energy_used"] = float(row.get("energy_used", 0.0))
        observation["platforms"] = platform_rows
        observation["queues"] = copy.deepcopy(self._queues)
        observation["predecessors"] = {key: list(value) for key, value in self._task_predecessors.items() if key in tasks}
        observation["cooperation"] = {key: list(value) for key, value in self._cooperation.items() if key in tasks}
        observation["task_progress"] = {key: float(value["progress"]) for key, value in tasks.items()}
        observation["environment"] = copy.deepcopy(self._environment)
        for node in observation["environment"]:
            node["risk"] = float(node.get("risk_level", 0.0))
        observation["communication"] = self._public_communication()
        # Scenario identities and the complete tape hash expose information
        # about events that have not happened yet. Keep those in private
        # trajectory metadata; the public projection is content-addressed only.
        provenance: dict[str, str] = {}
        provenance["snapshot_hash"] = _digest(observation)
        prefix = [{key: value for key, value in item.items() if key != "provenance"} for item in self._public_history]
        if prefix and prefix[-1] == observation:
            provenance["public_prefix_hash"] = _digest(prefix)
        else:
            provenance["public_prefix_hash"] = _digest(prefix + [observation])
        observation["provenance"] = provenance
        return observation

    def _public_communication(self) -> list[dict]:
        return [
            {**copy.deepcopy(row), "connected": bool(float(row["quality"]) >= 0.3)}
            for row in sorted(self._communication, key=lambda item: _link_key(item["source"], item["target"]))
        ]

    def _refresh_reported_state(self, base_events: list[dict]) -> None:
        """Advance central telemetry only from public baseline state reachable now."""
        live = self._base.public_observation()
        live_platforms = {row["id"]: row for row in live["platforms"]}
        live_tasks = {row["id"]: row for row in live["tasks"]}
        energy_deltas: dict[str, float] = {}
        for event in base_events:
            if event.get("type") in {"move", "service"}:
                platform_id = event.get("platform_id")
                if isinstance(platform_id, str):
                    energy_deltas[platform_id] = energy_deltas.get(platform_id, 0.0) + float(event.get("energy", 0.0))
        for event in base_events:
            if event.get("type") != "service":
                continue
            task_id = event.get("task_id")
            task = live_tasks.get(task_id)
            if task is None:
                continue
            owner = task.get("assignee") or task.get("active_platform_id") or event.get("platform_id")
            if isinstance(owner, str) and self._connected(self._public_graph_for_refresh(), _COORDINATOR, owner):
                self._reported_progress_counts[task_id] = float(self._reported_progress_counts.get(task_id, 0.0)) + float(event.get("duration", 1.0))
        for platform_id, row in live_platforms.items():
            if self._connected(self._public_graph_for_refresh(), _COORDINATOR, platform_id):
                self._reported_energy_used[platform_id] = float(self._reported_energy_used.get(platform_id, 0.0)) + energy_deltas.get(platform_id, 0.0)
                self._reported_platforms[platform_id] = copy.deepcopy(row)
                self._reported_platforms[platform_id]["energy_used"] = self._reported_energy_used[platform_id]
        for depot in live["depots"]:
            self._reported_depots[depot["id"]] = copy.deepcopy(depot)
        # A safety hold freezes the physical task while its control link is
        # unavailable.  Keep the public obligation visibly paused until the
        # current public link is restored; stale live telemetry must not erase
        # the hold marker.
        for task_id, row in live_tasks.items():
            hold = self._safety_paused_tasks.get(task_id)
            if hold is not None:
                platform_id = str(hold.get("platform_id") or row.get("active_platform_id") or row.get("assignee") or "")
                platform = live_platforms.get(platform_id, {})
                required_capability = "inspect" if row.get("kind") == "inspection" else "deliver"
                capability_ready = required_capability in platform.get("capabilities", []) and not bool(platform.get("failed"))
                link_ready = bool(platform_id) and self._connected(self._public_graph_for_refresh(), _COORDINATOR, platform_id)
                if platform_id and link_ready and capability_ready:
                    before = self._safety_task_state(task_id, platform_id)
                    self._safety_paused_tasks.pop(task_id, None)
                    self._reported_tasks[task_id] = copy.deepcopy(row)
                    self._reported_tasks[task_id]["safety_paused"] = False
                    self._reported_progress_counts[task_id] = float(row.get("service_progress", self._reported_progress_counts.get(task_id, 0.0)))
                    after = self._safety_task_state(task_id, platform_id)
                    capability_hold = str(hold.get("hold_kind", "")) == "capability"
                    self._safety_transition_event(
                        "safety_task_capability_resumed" if capability_hold else "safety_task_resumed",
                        task_id, platform_id,
                        "active_task_capability_restored" if capability_hold else "active_task_control_link_restored",
                        before, after,
                    )
                else:
                    preserved = copy.deepcopy(self._reported_tasks.get(task_id, row))
                    preserved.update({
                        "status": "paused", "safety_paused": True,
                        "assignee": row.get("assignee", preserved.get("assignee")),
                        "active_platform_id": row.get("active_platform_id", preserved.get("active_platform_id")),
                    })
                    self._reported_tasks[task_id] = preserved
                continue
            owner = row.get("assignee") or row.get("active_platform_id")
            if owner is None or self._connected(self._public_graph_for_refresh(), _COORDINATOR, str(owner)):
                self._reported_tasks[task_id] = copy.deepcopy(row)
                if row.get("status") == "completed":
                    spec = self._task_specs[task_id]
                    self._reported_progress_counts[task_id] = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
                elif row.get("phase") == "released":
                    self._reported_progress_counts[task_id] = 0.0
        for parent_id, manifest in self._split_manifests.items():
            child_rows = [self._reported_tasks.get(child_id) for child_id in manifest["child_ids"]]
            if any(row is None for row in child_rows):
                continue
            parent = self._reported_tasks[parent_id]
            child_progress = 0.0
            for child_id, row in zip(manifest["child_ids"], child_rows):
                spec = self._task_specs[child_id]
                total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
                fraction = min(1.0, float(self._reported_progress_counts.get(child_id, 0.0)) / max(1.0, total))
                child_progress += float(spec["workload_share"]) * fraction
            complete = all(row.get("status") == "completed" for row in child_rows)
            parent.update({"status": "completed" if complete else "split_parent",
                           "phase": "completed" if complete else "split_parent",
                           "progress": min(1.0, child_progress), "service_progress": child_progress,
                           "subtask_ids": list(manifest["child_ids"]),
                           "team_owners": [self._task_specs[child_id]["assigned_owner"] for child_id in manifest["child_ids"]]})

    def _public_graph_for_refresh(self) -> dict:
        return {"platforms": list(self._reported_platforms.values()), "communication": self._public_communication()}

    @staticmethod
    def _safety_stock_commitment(task: dict | None) -> dict | None:
        """Return the queue reservation represented by one delivery task."""
        if not isinstance(task, dict) or task.get("kind") != "delivery":
            return None
        if str(task.get("phase", "")) in {"pending", "released", "in_transit", "to_delivery_site", "unloading", "delivery_confirmed"}:
            return None
        return {
            "depot_id": str(task.get("depot_id")),
            "commodity": str(task.get("commodity")),
            "quantity": float(task.get("quantity", 0.0)),
        }

    @staticmethod
    def _required_task_capability(task: dict | None) -> str:
        """Return the public capability required by a logical task."""
        return "inspect" if isinstance(task, dict) and task.get("kind") == "inspection" else "deliver"

    def _safety_task_state(self, task_id: str, platform_id: str | None = None) -> dict:
        """Build a transition receipt from current physical/reportable state."""
        task = self._base._tasks.get(task_id)
        public_task = self._reported_tasks.get(task_id, {})
        if task is None:
            raise KeyError(task_id)
        # Historical V1 snapshots can contain a stale private task state after
        # the old safety fallback marked the public task active.  Transition
        # evidence must describe the state visible to the current safety hook,
        # while retaining private physical counters when they are available.
        visible_task = {**task, **public_task}
        owner = visible_task.get("assignee")
        queue_owner = next((owner_id for owner_id, queue in self._queues.items() if task_id in queue), None)
        total = float(self._task_specs.get(task_id, {}).get(
            "service_minutes", self._task_specs.get(task_id, {}).get("load_minutes", 1)
            + self._task_specs.get(task_id, {}).get("unload_minutes", 1),
        ))
        progress = min(1.0, max(0.0, float(visible_task.get("service_progress", 0.0)) / max(1.0, total)))
        return {
            "status": "paused" if task_id in self._safety_paused_tasks else str(public_task.get("status", task.get("status"))),
            "phase": str(visible_task.get("phase", "")),
            "progress": progress,
            "service_progress": float(visible_task.get("service_progress", 0.0)),
            "cargo_onboard": float(visible_task.get("cargo_onboard", 0.0)),
            "assignee": owner,
            "active_platform_id": visible_task.get("active_platform_id"),
            "queue_owner": queue_owner,
            "stock_commitment": self._safety_stock_commitment(visible_task),
            "safety_paused": task_id in self._safety_paused_tasks,
        }

    def _safety_transition_event(
        self, event_type: str, task_id: str, platform_id: str | None, reason: str,
        before: dict, after: dict,
    ) -> dict:
        event = {
            "schema": "gain-safety-task-transition/1", "type": event_type,
            "time": int(self.time), "task_id": str(task_id),
            "platform_id": platform_id, "reason": str(reason),
            "before": copy.deepcopy(before), "after": copy.deepcopy(after),
        }
        event["event_sha256"] = _digest(event)
        self._extension_events.append(copy.deepcopy(event))
        return event

    def _safety_transition_remaining(self) -> dict:
        """Recompute only the public queue disposition after a same-time change."""
        try:
            from .gain_safety import assess_queue_safety
        except ImportError:
            from gain_safety import assess_queue_safety
        try:
            return assess_queue_safety(copy.deepcopy(self.public_observation()))
        except (TypeError, ValueError, KeyError) as exc:
            return {
                "schema": "gain-public-queue-safety/1", "forced": True,
                "safe_to_suppress": False, "violations": [],
                "unknown_reason": f"post_disposition_safety_check_failed:{type(exc).__name__}",
            }

    def apply_current_safety_disposition(self, public_observation: dict, safety_report: dict) -> dict:
        """Apply a deterministic safety hold/release at the current tick.

        This hook never advances the simulator, samples the event tape, or
        calls the planner.  It acts only on violations represented in the
        supplied public observation and records the concrete state transition
        in the extension event stream.
        """
        schema = "gain-live-safety-disposition/1"
        if not self._has_reset:
            return {"schema": schema, "status": "unknown", "resolved": False,
                    "actions": [], "remaining_report": {}, "unknown_reason": "environment_not_reset"}
        current = self.public_observation()
        supplied_hash = (public_observation or {}).get("provenance", {}).get("snapshot_hash") if isinstance(public_observation, dict) else None
        current_hash = current["provenance"]["snapshot_hash"]
        base_receipt = {"schema": schema, "pre_public_sha256": supplied_hash,
                        "post_public_sha256": current_hash, "actions": [],
                        "actions_sha256": _digest([])}
        if supplied_hash != current_hash:
            return {**base_receipt, "status": "unknown", "resolved": False,
                    "remaining_report": {}, "unknown_reason": "public_observation_stale"}
        if not isinstance(safety_report, dict):
            return {**base_receipt, "status": "unknown", "resolved": False,
                    "remaining_report": {}, "unknown_reason": "safety_report_malformed"}
        violations = safety_report.get("violations", [])
        if not isinstance(violations, list):
            return {**base_receipt, "status": "unknown", "resolved": False,
                    "remaining_report": {}, "unknown_reason": "safety_report_violations_malformed"}
        if not safety_report.get("forced", False) and not violations:
            return {**base_receipt, "status": "not_required", "resolved": True,
                    "remaining_report": safety_report, "unknown_reason": None}

        public_tasks = {str(row.get("id")): row for row in current.get("tasks", []) if isinstance(row, dict)}
        public_platforms = {str(row.get("id")): row for row in current.get("platforms", []) if isinstance(row, dict)}
        actions: list[dict] = []
        unsupported: list[str] = []
        release_ids: set[str] = set()
        pause_ids: list[tuple[str, str]] = []
        capability_pause_ids: list[tuple[str, str]] = []
        releasable_codes = {
            "queued_task_capability_infeasible", "queued_task_capacity_infeasible",
            "committed_queue_energy_infeasible",
        }
        for violation in violations:
            if not isinstance(violation, dict):
                unsupported.append("violation_must_be_object")
                continue
            code = str(violation.get("code", ""))
            task_id = violation.get("task_id")
            platform_id = violation.get("platform_id")
            if code == "active_task_control_link_lost" and isinstance(task_id, str) and isinstance(platform_id, str):
                task = public_tasks.get(task_id, {})
                platform = public_platforms.get(platform_id, {})
                link = self._connected(current, _COORDINATOR, platform_id)
                if task.get("status") not in {"active", "paused"} or platform.get("active_task_id") != task_id or link:
                    unsupported.append(f"active_pause_precondition_failed:{task_id}")
                elif task_id not in self._safety_paused_tasks:
                    pause_ids.append((task_id, platform_id))
                continue
            if code == "queued_task_capability_infeasible" and isinstance(task_id, str) and isinstance(platform_id, str):
                task = public_tasks.get(task_id, {})
                platform = public_platforms.get(platform_id, {})
                required_capability = self._required_task_capability(task)
                is_active = (
                    task.get("status") in {"active", "paused"}
                    or platform.get("active_task_id") == task_id
                )
                if is_active:
                    if (
                        platform.get("active_task_id") != task_id
                        or task.get("assignee") not in {None, platform_id}
                        or required_capability in platform.get("capabilities", [])
                    ):
                        unsupported.append(f"active_capability_pause_precondition_failed:{task_id}")
                    elif task_id not in self._safety_paused_tasks and not any(
                        queued_task_id == task_id for queued_task_id, _ in pause_ids
                    ):
                        capability_pause_ids.append((task_id, platform_id))
                    continue
            if code in releasable_codes and isinstance(task_id, str) and isinstance(platform_id, str):
                task = public_tasks.get(task_id, {})
                physical = self._base._tasks.get(task_id, {})
                if task.get("status") not in {"pending", "assigned", "released"} or public_platforms.get(platform_id, {}).get("active_task_id") == task_id:
                    unsupported.append(f"release_requires_pre_start:{task_id}")
                elif str(physical.get("status")) == "active" or physical.get("active_platform_id") is not None:
                    unsupported.append(f"release_active_task:{task_id}")
                elif float(physical.get("cargo_onboard", 0.0)) > _EPS:
                    unsupported.append(f"release_loaded_task:{task_id}")
                else:
                    release_ids.add(task_id)
                continue
            unsupported.append(f"unsupported_safety_violation:{code or 'missing_code'}")

        # Never split a cooperative group while releasing one half.  A group
        # is released only when every member is a public, pre-start, unloaded
        # queue item covered by this same disposition.
        for task_id in sorted(release_ids):
            group = public_tasks.get(task_id, {}).get("cooperation_group")
            if not group:
                continue
            members = sorted(task_key for task_key, row in public_tasks.items() if row.get("cooperation_group") == group)
            if not set(members) <= release_ids:
                unsupported.append(f"cooperation_release_would_be_partial:{task_id}")
                release_ids.difference_update(members)

        for task_id, platform_id in pause_ids:
            before = self._safety_task_state(task_id, platform_id)
            self._safety_paused_tasks[task_id] = {"platform_id": platform_id, "paused_at": int(self.time)}
            self._reported_tasks.setdefault(task_id, {}).update({"status": "paused", "safety_paused": True})
            after = self._safety_task_state(task_id, platform_id)
            actions.append(self._safety_transition_event(
                "safety_task_paused", task_id, platform_id, "active_task_control_link_lost", before, after,
            ))

        for task_id, platform_id in capability_pause_ids:
            before = self._safety_task_state(task_id, platform_id)
            self._safety_paused_tasks[task_id] = {
                "platform_id": platform_id,
                "paused_at": int(self.time),
                "hold_kind": "capability",
                "reason": "active_task_capability_lost",
            }
            self._reported_tasks.setdefault(task_id, {}).update({"status": "paused", "safety_paused": True})
            after = self._safety_task_state(task_id, platform_id)
            actions.append(self._safety_transition_event(
                "safety_task_capability_paused", task_id, platform_id,
                "active_task_capability_lost", before, after,
            ))

        for task_id in sorted(release_ids):
            task = self._base._tasks.get(task_id)
            if task is None:
                unsupported.append(f"release_task_missing:{task_id}")
                continue
            platform_id = str(task.get("assignee") or next((owner for owner, queue in self._queues.items() if task_id in queue), ""))
            before = self._safety_task_state(task_id, platform_id or None)
            for queue in self._queues.values():
                while task_id in queue:
                    queue.remove(task_id)
            task["assignee"] = None
            task["status"] = "pending"
            task["phase"] = "released"
            task["active_platform_id"] = None
            self._reported_progress_counts[task_id] = 0.0
            if task_id in self._base._public_tasks:
                self._base._public_tasks[task_id].update({"assignee": None, "status": "pending", "phase": "released"})
            self._reported_tasks.setdefault(task_id, {}).update({"assignee": None, "status": "pending", "phase": "released", "safety_paused": False})
            self._base._queues = copy.deepcopy(self._queues)
            after = self._safety_task_state(task_id, platform_id or None)
            actions.append(self._safety_transition_event(
                "safety_task_released", task_id, platform_id or None,
                "queued_public_infeasibility", before, after,
            ))

        remaining = self._safety_transition_remaining()
        unresolved = bool(unsupported) or bool(remaining.get("forced", False))
        post = self.public_observation()
        return {
            **base_receipt, "post_public_sha256": post["provenance"]["snapshot_hash"],
            "status": "unresolved" if unresolved else "resolved", "resolved": not unresolved,
            "actions": copy.deepcopy(actions), "actions_sha256": _digest(actions),
            "remaining_report": remaining,
            "unknown_reason": ";".join(sorted(set(unsupported))) if unsupported else None,
        }

    def _advance_with_safety_holds(self, start_time: int) -> None:
        """Advance ordinary physics while leaving current safety holds frozen."""
        end_time = start_time + 1
        for platform_id in sorted(self._base._platforms):
            platform = self._base._platforms[platform_id]
            task_id = platform["active_task_id"]
            if task_id is None or platform["failed"] or task_id in self._safety_paused_tasks:
                continue
            task = self._base._tasks[task_id]
            required_capability = self._required_task_capability(task)
            if required_capability not in platform.get("capabilities", []):
                # Capability-loss events are held at the current physical
                # state until the public safety disposition records the hold.
                continue
            phase = task["phase"]
            if phase == "to_inspection":
                self._base._move_toward(platform_id, task_id, task["position"], "inspection_site", end_time)
            elif phase == "to_depot":
                self._base._move_toward(platform_id, task_id, self._base._depots[task["depot_id"]]["position"], "depot", end_time)
            elif phase == "to_delivery_site":
                self._base._move_toward(platform_id, task_id, task["position"], "delivery_site", end_time)
            elif phase == "observe":
                if self._base._service(platform_id, task_id, "observe", end_time) and task["service_progress"] + _EPS >= float(task["service_minutes"]):
                    self._base._complete_inspection(task, platform_id, end_time)
            elif phase == "loading":
                if self._base._service(platform_id, task_id, "loading", end_time) and task["service_progress"] + _EPS >= float(task["load_minutes"]):
                    if self._base._load_cargo(task, platform_id, end_time):
                        task["phase"] = "in_transit"
                        self._base._tasks[task_id]["phase"] = "to_delivery_site"
                        self._base._public_tasks[task_id]["phase"] = "to_delivery_site"
                        if _distance(platform["position"], task["position"]) <= _EPS:
                            self._base._arrive(platform_id, task_id, "delivery_site", end_time)
            elif phase == "in_transit":
                task["phase"] = "to_delivery_site"
                self._base._public_tasks[task_id]["phase"] = "to_delivery_site"
            elif phase == "unloading":
                if self._base._service(platform_id, task_id, "unloading", end_time) and task["service_progress"] + _EPS >= float(task["unload_minutes"]):
                    self._base._confirm_delivery(task, platform_id, end_time)

    def candidate_dispatches(self) -> list[dict]:
        """Return all publicly feasible dispatches, including global cross-owner moves."""
        if not self._has_reset:
            raise RuntimeError("call reset() before requesting candidates")
        observation = self.public_observation()
        public_tasks = {row["id"]: row for row in observation["tasks"]}
        result = []
        reserved: dict[tuple[str, str], float] = {}
        for task in observation["tasks"]:
            if task.get("kind") == "delivery" and task.get("status") in {"active", "assigned"} and task.get("phase") in {"assigned", "to_depot", "loading"}:
                key = (str(task.get("depot_id")), str(task.get("commodity")))
                reserved[key] = reserved.get(key, 0.0) + float(task.get("quantity", 0.0))
        for platform in observation["platforms"]:
            platform_id = platform["id"]
            if platform.get("failed") or platform.get("busy"):
                continue
            if not self._connected(observation, _COORDINATOR, platform_id):
                continue
            for task_id, task in sorted(public_tasks.items()):
                if task.get("status") not in {"pending", "assigned", "released"}:
                    continue
                owner = task.get("assignee")
                if task.get("cooperation_group") and owner != platform_id:
                    continue
                if owner is not None and owner != platform_id and not self._connected(observation, str(owner), platform_id):
                    continue
                if not self._public_compatible(platform, task):
                    continue
                if any(next((row for row in observation["tasks"] if row["id"] == predecessor), {}).get("status") != "completed"
                       for predecessor in task.get("predecessors", [])):
                    continue
                mission = self._estimate_public_mission(platform, task, observation)
                source = str(owner) if isinstance(owner, str) else _COORDINATOR
                communication = _communication_metrics(observation, source, platform_id)
                communication_delay = _communication_delay_minutes(communication)
                estimated_completion = float(mission["estimated_completion"]) + float(communication_delay)
                if mission["blocked_route"] or estimated_completion > min(
                    float(observation["horizon"]), float(task.get("deadline", observation["horizon"]))
                ) + _EPS or mission["required_energy"] > float(platform.get("energy", 0.0)) + _EPS:
                    continue
                if task.get("kind") == "delivery":
                    key = (str(task["depot_id"]), str(task["commodity"]))
                    depot = next((row for row in observation["depots"] if row["id"] == key[0]), {})
                    available = float(depot.get("stock", {}).get(key[1], 0.0)) - float(reserved.get(key, 0.0))
                    if float(task.get("quantity", 0.0)) > available + _EPS:
                        continue
                queues, task_rows = self._planned_queue_state(
                    observation, [], [{"platform_id": platform_id, "task_id": task_id}],
                )
                if self._new_queue_issues(observation, queues, task_rows):
                    continue
                result.append({
                    "id": f"dispatch:{platform_id}:{task_id}", "type": "dispatch",
                    "platform_id": platform_id, "task_id": task_id, "kind": task["kind"],
                    "estimated_completion": estimated_completion,
                    "required_energy": mission["required_energy"],
                    "environmental_risk": mission["environmental_risk"],
                    "communication_quality": float(communication["quality"]),
                    "communication_latency": float(communication["latency_minutes"]),
                    "communication_loss": float(communication["loss_rate"]),
                    "communication_delay": int(communication_delay), "locked": False,
                })
        by_group: dict[str, set[str]] = {}
        for task_id, task in public_tasks.items():
            group = task.get("cooperation_group")
            if group:
                by_group.setdefault(str(group), set()).add(task_id)
        usable_groups = {
            group for group, task_ids in by_group.items()
            if len(task_ids) == 2 and all(any(row["task_id"] == task_id for row in result) for task_id in task_ids)
        }
        result = [row for row in result
                  if not public_tasks[row["task_id"]].get("cooperation_group")
                  or str(public_tasks[row["task_id"]]["cooperation_group"]) in usable_groups]
        # Keep the complete feasible catalog deterministic while exposing
        # urgent, low-slack, cheap public actions first.  The generator uses
        # this same key before applying its local/cross quotas.
        return sorted(result, key=lambda row: _dispatch_rank_key(observation, row))

    def _components(self, observation: dict) -> dict[str, int]:
        nodes = {str(row["id"]) for row in observation.get("platforms", [])}
        adjacency = {node: set() for node in nodes}
        for link in observation.get("communication", []):
            if link.get("connected"):
                left, right = str(link["source"]), str(link["target"])
                if left in nodes and right in nodes:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        components: dict[str, int] = {}
        component = 0
        for node in sorted(nodes):
            if node in components:
                continue
            stack = [node]
            components[node] = component
            while stack:
                current = stack.pop()
                for neighbor in sorted(adjacency[current]):
                    if neighbor not in components:
                        components[neighbor] = component
                        stack.append(neighbor)
            component += 1
        return components

    def _connected(self, observation: dict, start: str, end: str) -> bool:
        if start == end:
            return True
        components = self._components(observation)
        return start in components and end in components and components[start] == components[end]

    def _dispatch_legality(self, observation: dict, dispatch: dict, allowed_ids: dict[tuple[str, str], dict]) -> list[str]:
        reasons = []
        platform_id, task_id = dispatch.get("platform_id"), dispatch.get("task_id")
        if not isinstance(platform_id, str) or not isinstance(task_id, str):
            return ["invalid_dispatch_identity"]
        if (platform_id, task_id) not in allowed_ids:
            reasons.append("capability_stock_time_or_energy")
            return reasons
        platform = next((row for row in observation["platforms"] if row["id"] == platform_id), None)
        task = next((row for row in observation["tasks"] if row["id"] == task_id), None)
        if platform is None or task is None:
            return ["unknown_public_entity"]
        if platform.get("failed") or platform.get("busy"):
            reasons.append("platform_unavailable")
        for predecessor_id in task.get("predecessors", []):
            predecessor = next((row for row in observation["tasks"] if row["id"] == predecessor_id), None)
            if predecessor is None or predecessor.get("status") != "completed":
                reasons.append("predecessor_incomplete")
        owner = task.get("assignee")
        communication_source = owner if isinstance(owner, str) else _COORDINATOR
        if not self._connected(observation, communication_source, platform_id):
            reasons.append("communication_unreachable")
        base_candidate = allowed_ids.get((platform_id, task_id))
        if base_candidate is not None and float(base_candidate.get("estimated_completion", 0.0)) > min(
            float(observation["horizon"]), float(task.get("deadline", observation["horizon"]))
        ) + _EPS:
            reasons.append("time_window_exceeded")
        return reasons

    def _operation_legality(self, observation: dict, operation: dict, dispatch_pairs: set[tuple[str, str]]) -> list[str]:
        reasons = []
        kind = operation.get("op")
        task_ids = [operation.get("task_id")]
        if kind == "swap":
            task_ids.append(operation.get("other_task_id"))
        task_by_id = {row["id"]: row for row in observation["tasks"]}
        platform_by_id = {row["id"]: row for row in observation["platforms"]}
        source, target = operation.get("from_platform"), operation.get("to_platform")
        if kind not in {"transfer", "swap", "insert", "split", "cooperate"}:
            return ["unknown_queue_operation"]
        if not isinstance(source, str) and source is not None:
            reasons.append("invalid_source_platform")
        if not isinstance(target, str) or target not in platform_by_id:
            return reasons + ["unknown_target_platform"]
        if source is not None and source not in platform_by_id:
            reasons.append("unknown_source_platform")
        if kind in {"split", "cooperate"}:
            if source is None or source not in platform_by_id:
                reasons.append("split_requires_source_platform")
            if source == target:
                reasons.append("split_requires_distinct_platforms")
            if len(task_ids) != 1 or not isinstance(task_ids[0], str):
                reasons.append("split_requires_one_parent_task")
            share = operation.get("share")
            if type(share) not in {int, float} or not math.isfinite(float(share)) or not 0.0 < float(share) < 1.0:
                reasons.append("invalid_workload_share")
            parent_id = task_ids[0] if task_ids and isinstance(task_ids[0], str) else None
            parent = task_by_id.get(parent_id) if parent_id is not None else None
            if parent is not None:
                if parent_id in self._split_manifests or self._task_specs.get(parent_id, {}).get("parent_task_id"):
                    reasons.append("task_already_split")
                if parent.get("assignee") != source or parent_id not in self._queues.get(source, []):
                    reasons.append("split_source_owner_mismatch")
                if parent.get("status") not in {"pending", "assigned", "released"}:
                    reasons.append("task_not_reassignable")
                if source is not None and target is not None and not self._connected(observation, source, target):
                    reasons.append("communication_unreachable")
                if source in platform_by_id and platform_by_id[source].get("failed"):
                    reasons.append("source_platform_unavailable")
                if target in platform_by_id and platform_by_id[target].get("failed"):
                    reasons.append("target_platform_unavailable")
                if parent is not None and type(share) in {int, float} and math.isfinite(float(share)) and 0.0 < float(share) < 1.0:
                    shares = (float(share), 1.0 - float(share))
                    child_rows = []
                    for workload_share in shares:
                        child = copy.deepcopy(parent)
                        if child.get("kind") == "delivery":
                            child["quantity"] = float(parent["quantity"]) * workload_share
                        else:
                            child["service_minutes"] = float(parent["service_minutes"]) * workload_share
                        child_rows.append(child)
                    if source in platform_by_id and not self._public_compatible(platform_by_id[source], child_rows[0]):
                        reasons.append("source_child_incompatible")
                    if target in platform_by_id and not self._public_compatible(platform_by_id[target], child_rows[1]):
                        reasons.append("target_child_incompatible")
                if kind == "cooperate" and (parent.get("kind") not in {"inspection", "delivery"}):
                    reasons.append("cooperation_requires_executable_workload")
            before = operation.get("before_task_id")
            if parent_id is not None and before == parent_id:
                reasons.append("insert_anchor_conflicts_with_task")
            elif before is not None and before not in self._queues.get(target, []):
                reasons.append("insert_anchor_not_in_target_queue")
            return sorted(set(reasons))
        if source is not None and source == target and kind != "insert":
            reasons.append("no_op_queue_operation")
        for index, task_id in enumerate(task_ids):
            task = task_by_id.get(task_id) if isinstance(task_id, str) else None
            if task is None:
                reasons.append("task_not_public")
                continue
            if task.get("safety_paused") or task_id in self._safety_paused_tasks:
                reasons.append("task_safety_paused")
            if task.get("status") in {"active", "completed", "failed", "expired", "cancelled"}:
                reasons.append("task_not_reassignable")
            expected_owner = source if kind != "swap" or index == 0 else target
            destination = target if kind != "swap" or index == 0 else source
            if expected_owner is not None and task.get("assignee") != expected_owner:
                reasons.append("source_owner_mismatch")
            if expected_owner is not None and task_id not in self._queues.get(expected_owner, []):
                reasons.append("source_task_not_in_queue")
            if destination is not None and not self._public_compatible(platform_by_id[destination], task):
                reasons.append("capability_or_capacity_mismatch")
            if expected_owner is not None and destination is not None and not self._connected(observation, expected_owner, destination):
                reasons.append("communication_unreachable")
        before = operation.get("before_task_id")
        if kind == "insert":
            if operation.get("task_id") is not None and before == operation.get("task_id"):
                reasons.append("insert_anchor_conflicts_with_task")
            elif before is not None and before not in self._queues.get(target, []):
                reasons.append("insert_anchor_not_in_target_queue")
        if kind == "swap" and (len(task_ids) != 2 or task_ids[0] == task_ids[1]):
            reasons.append("invalid_swap_pair")
        return reasons

    @staticmethod
    def _public_compatible(platform: dict, task: dict) -> bool:
        needed = "inspect" if task.get("kind") == "inspection" else "deliver"
        return (
            needed in platform.get("capabilities", [])
            and (task.get("kind") != "delivery" or float(task.get("quantity", 0.0)) <= float(platform.get("capacity_kg", 0.0)) + _EPS)
        )

    @staticmethod
    def _cell_intersection_length(start: list[float], end: list[float], cell: list[int]) -> float:
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        distance = math.hypot(dx, dy)
        if distance <= _EPS:
            return 0.0
        x, y = float(cell[0]), float(cell[1])
        low, high = 0.0, 1.0
        for p, q in ((-dx, float(start[0]) - x), (dx, x + 1.0 - float(start[0])),
                     (-dy, float(start[1]) - y), (dy, y + 1.0 - float(start[1]))):
            if abs(p) <= _EPS:
                if q < 0.0:
                    return 0.0
                continue
            ratio = q / p
            if p < 0.0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
            if high <= low:
                return 0.0
        return max(0.0, min(1.0, high) - max(0.0, low)) * distance

    def _segment_environment(self, start: list[float], end: list[float], speed: float,
                             energy_per_km: float) -> dict:
        distance = _distance(start, end)
        if distance <= _EPS:
            return {"minutes": 0.0, "energy": 0.0, "risk": 0.0, "blocked": False}
        unit = [(float(end[0]) - float(start[0])) / distance,
                (float(end[1]) - float(start[1])) / distance]
        affected_length = 0.0
        minutes = 0.0
        energy = 0.0
        risk = 0.0
        blocked = False
        for node in self._environment:
            length = self._cell_intersection_length(start, end, node["grid_cell"])
            if length <= _EPS:
                continue
            affected_length += length
            wind = node.get("wind_vector", [0.0, 0.0])
            along_wind = float(wind[0]) * unit[0] + float(wind[1]) * unit[1]
            wind_magnitude = math.hypot(float(wind[0]), float(wind[1]))
            nominal_cell_speed = float(speed) * float(node.get("speed_factor", 1.0)) + along_wind
            cell_speed = max(float(speed) * 0.1, min(float(speed) * 1.5, nominal_cell_speed))
            cell_energy_factor = float(node.get("energy_per_km_factor", 1.0)) * (1.0 + 0.1 * wind_magnitude)
            if node.get("blocked"):
                blocked = True
                cell_speed = max(float(speed) * 0.1, 0.01)
                cell_energy_factor += 1.0
            minutes += length / max(cell_speed, _EPS)
            energy += length * float(energy_per_km) * cell_energy_factor
            risk += length * float(node.get("risk_level", 0.0)) + (length if node.get("blocked") else 0.0)
        outside = max(0.0, distance - affected_length)
        minutes += outside / max(float(speed), _EPS)
        energy += outside * float(energy_per_km)
        return {"minutes": minutes, "energy": energy, "risk": risk, "blocked": blocked}

    def _estimate_route_mission(self, platform: dict, task: dict, observation: dict,
                                position: list[float], start_time: float,
                                stops: list[list[float]], service_minutes: float) -> dict:
        speed = float(platform["speed_km_min"])
        energy_rate = float(platform["energy_per_km"])
        service_rate = float(platform["service_energy_per_min"])
        travel_minutes, travel_energy, risk, blocked = 0.0, 0.0, 0.0, False
        route_start = list(position)
        for end in stops:
            end = list(end)
            segment = self._segment_environment(route_start, end, speed, energy_rate)
            travel_minutes += segment["minutes"]
            travel_energy += segment["energy"]
            risk += segment["risk"]
            blocked = blocked or segment["blocked"]
            route_start = end
        service_minutes = max(0.0, float(service_minutes))
        completion = float(start_time) + math.ceil(travel_minutes - _EPS) + math.ceil(service_minutes - _EPS)
        energy = travel_energy + service_minutes * service_rate
        return {"estimated_completion": float(completion), "required_energy": float(energy),
                "environmental_risk": float(risk), "blocked_route": bool(blocked)}

    def _estimate_public_mission(self, platform: dict, task: dict, observation: dict,
                                 *, position: list[float] | None = None,
                                 start_time: float | None = None) -> dict:
        position = list(platform["position"] if position is None else position)
        start_time = float(self.time if start_time is None else start_time)
        if task["kind"] == "inspection":
            stops = [list(task["position"])]
            service_minutes = float(task["service_minutes"])
        else:
            depot = next(row for row in observation["depots"] if row["id"] == task["depot_id"])
            stops = [list(depot["position"]), list(task["position"])]
            service_minutes = float(task["load_minutes"]) + float(task["unload_minutes"])
        return self._estimate_route_mission(platform, task, observation, position, start_time, stops, service_minutes)

    def _estimate_active_remaining(self, platform: dict, task: dict, observation: dict,
                                   position: list[float], start_time: float) -> dict:
        phase = task.get("phase")
        progress = max(0.0, float(task.get("service_progress", 0.0)))
        stops: list[list[float]] = []
        if task["kind"] == "inspection":
            service = max(0.0, float(task["service_minutes"]) - progress) if phase == "observe" else float(task["service_minutes"])
            if phase != "observe":
                stops = [list(task["position"])]
        else:
            depot = next(row for row in observation["depots"] if row["id"] == task["depot_id"])
            if phase in {"loading"}:
                stops = [list(depot["position"]), list(task["position"])]
                service = max(0.0, float(task["load_minutes"]) - progress) + float(task["unload_minutes"])
            elif phase in {"in_transit", "to_delivery_site"}:
                stops = [list(task["position"])]
                service = float(task["unload_minutes"])
            elif phase == "unloading":
                unloaded = max(0.0, progress - float(task["load_minutes"]))
                service = max(0.0, float(task["unload_minutes"]) - unloaded)
            else:
                stops = [list(depot["position"]), list(task["position"])]
                service = float(task["load_minutes"]) + float(task["unload_minutes"])
        return self._estimate_route_mission(platform, task, observation, position, start_time, stops, service)

    def _queue_schedule_issues(self, observation: dict, queues: dict[str, list[str]],
                               task_rows: dict[str, dict] | None = None) -> list[tuple[str, str, str]]:
        """Project committed queues using only reported platform/task state."""
        tasks = {row["id"]: copy.deepcopy(row) for row in observation["tasks"]}
        if task_rows:
            tasks.update({key: copy.deepcopy(value) for key, value in task_rows.items()})
        platforms = {row["id"]: row for row in observation["platforms"]}
        depots = {row["id"]: row for row in observation["depots"]}
        issues: set[tuple[str, str, str]] = set()
        sequences: dict[str, list[str]] = {}
        owner_by_task: dict[str, str] = {}
        terminal = {"completed", "failed", "expired", "cancelled", "split_parent"}

        for platform_id in sorted(platforms):
            platform = platforms[platform_id]
            sequence = [task_id for task_id in queues.get(platform_id, [])
                        if task_id in tasks and tasks[task_id].get("status") not in terminal]
            active_id = platform.get("active_task_id")
            if isinstance(active_id, str) and active_id in tasks and tasks[active_id].get("status") not in terminal:
                sequence = [active_id] + [task_id for task_id in sequence if task_id != active_id]
            sequences[platform_id] = sequence
            seen = set()
            for task_id in sequence:
                if task_id in seen:
                    issues.add(("duplicate_queue_task", task_id, platform_id))
                seen.add(task_id)
                if task_id in owner_by_task and owner_by_task[task_id] != platform_id:
                    issues.add(("duplicate_queue_task", task_id, platform_id))
                owner_by_task[task_id] = platform_id
                if tasks[task_id].get("assignee") not in {None, platform_id}:
                    issues.add(("queue_owner_mismatch", task_id, platform_id))
                if platform.get("failed"):
                    issues.add(("platform_unavailable", task_id, platform_id))

        units: dict[str, list[str]] = {}
        unit_for_task: dict[str, str] = {}
        for task_id in owner_by_task:
            group = tasks[task_id].get("cooperation_group")
            unit_id = f"cooperation:{group}" if group else f"task:{task_id}"
            unit_for_task[task_id] = unit_id
            units.setdefault(unit_id, []).append(task_id)
        successors: dict[str, set[str]] = {unit_id: set() for unit_id in units}
        indegree = {unit_id: 0 for unit_id in units}
        incoming_tasks: dict[str, set[str]] = {unit_id: set() for unit_id in units}

        def add_edge(before: str, after: str) -> None:
            before_unit = unit_for_task.get(before)
            after_unit = unit_for_task.get(after)
            if before_unit is None or after_unit is None or before_unit == after_unit:
                return
            incoming_tasks[after_unit].add(before)
            if after_unit not in successors[before_unit]:
                successors[before_unit].add(after_unit)
                indegree[after_unit] += 1

        for platform_id, sequence in sequences.items():
            for before, after in zip(sequence, sequence[1:]):
                add_edge(before, after)
            for task_id in sequence:
                task = tasks[task_id]
                if platforms[platform_id].get("active_task_id") == task_id:
                    continue
                for predecessor_id in task.get("predecessors", self._task_predecessors.get(task_id, [])):
                    predecessor_id = str(predecessor_id)
                    predecessor = tasks.get(predecessor_id)
                    if predecessor and predecessor.get("status") == "completed":
                        continue
                    predecessor_children = self._split_manifests.get(predecessor_id, {}).get("child_ids", [])
                    predecessor_ids = predecessor_children or [predecessor_id]
                    for dependency_id in predecessor_ids:
                        dependency = tasks.get(dependency_id)
                        if dependency and dependency.get("status") == "completed":
                            continue
                        if dependency_id not in owner_by_task:
                            issues.add(("predecessor_not_scheduled", task_id, platform_id))
                            continue
                        add_edge(dependency_id, task_id)

        for unit_id, members in units.items():
            if len(members) > 1:
                platforms_for_group = {owner_by_task[task_id] for task_id in members}
                if len(members) != 2 or len(platforms_for_group) != 2:
                    issues.add(("cooperation_pair_invalid", members[0], owner_by_task[members[0]]))
            elif tasks[members[0]].get("cooperation_group"):
                issues.add(("cooperation_pair_incomplete", members[0], owner_by_task[members[0]]))

        now = float(observation["time"])
        state = {
            platform_id: {"time": now, "position": list(platform["position"]), "energy": float(platform.get("energy", 0.0))}
            for platform_id, platform in platforms.items()
        }
        completion_time: dict[str, float] = {}
        ready = sorted(unit_id for unit_id, degree in indegree.items() if degree == 0)
        processed = set()
        while ready:
            unit_id = ready.pop(0)
            processed.add(unit_id)
            members = sorted(units[unit_id])
            member_platforms = [owner_by_task[task_id] for task_id in members]
            dependencies = [completion_time[task_id] for task_id in incoming_tasks[unit_id]
                            if task_id in completion_time]
            start_time = max([now, *dependencies, *(state[key]["time"] for key in member_platforms)])
            estimates = []
            for task_id, platform_id in zip(members, member_platforms):
                task = tasks[task_id]
                platform = platforms[platform_id]
                if platform.get("failed"):
                    estimates.append(None)
                    continue
                if platform.get("active_task_id") == task_id:
                    estimate = self._estimate_active_remaining(platform, task, observation,
                                                                state[platform_id]["position"], start_time)
                else:
                    estimate = self._estimate_public_mission(platform, task, observation,
                                                              position=state[platform_id]["position"], start_time=start_time)
                estimates.append(estimate)
            for task_id, platform_id, estimate in zip(members, member_platforms, estimates):
                if estimate is None:
                    continue
                task = tasks[task_id]
                platform = platforms[platform_id]
                if not self._public_compatible(platform, task):
                    issues.add(("capability_or_capacity_mismatch", task_id, platform_id))
                if estimate["blocked_route"]:
                    issues.add(("route_blocked", task_id, platform_id))
                if estimate["required_energy"] > state[platform_id]["energy"] + _EPS:
                    issues.add(("cumulative_energy_exceeded", task_id, platform_id))
                if estimate["estimated_completion"] > min(float(observation["horizon"]),
                                                            float(task.get("deadline", observation["horizon"]))) + _EPS:
                    issues.add(("cumulative_deadline_exceeded", task_id, platform_id))
                state[platform_id]["time"] = estimate["estimated_completion"]
                state[platform_id]["position"] = list(task["position"])
                state[platform_id]["energy"] = max(0.0, state[platform_id]["energy"] - estimate["required_energy"])
                completion_time[task_id] = estimate["estimated_completion"]
            for successor in sorted(successors[unit_id]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()
        for unit_id, degree in indegree.items():
            if degree > 0 and unit_id not in processed:
                for task_id in units[unit_id]:
                    issues.add(("predecessor_or_queue_cycle", task_id, owner_by_task[task_id]))

        stock_commitments: dict[tuple[str, str], float] = {}
        for task_id in owner_by_task:
            task = tasks[task_id]
            if task.get("kind") != "delivery" or task.get("status") in terminal:
                continue
            phase = task.get("phase")
            if platforms[owner_by_task[task_id]].get("active_task_id") == task_id and phase in {"in_transit", "to_delivery_site", "unloading", "delivery_confirmed"}:
                continue
            key = (str(task.get("depot_id")), str(task.get("commodity")))
            stock_commitments[key] = stock_commitments.get(key, 0.0) + float(task.get("quantity", 0.0))
        for key, quantity in stock_commitments.items():
            depot = depots.get(key[0], {})
            available = float(depot.get("stock", {}).get(key[1], 0.0))
            if quantity > available + _EPS:
                issues.add(("cumulative_inventory_exceeded", key[0], key[1]))
        return sorted(issues)

    def _planned_queue_state(self, observation: dict, queue_operations: list[dict],
                             dispatches: list[dict]) -> tuple[dict[str, list[str]], dict[str, dict]]:
        queues = copy.deepcopy(self._queues)
        tasks = {row["id"]: copy.deepcopy(row) for row in observation["tasks"]}
        tasks.update({key: copy.deepcopy(value) for key, value in self._task_specs.items() if key not in tasks})
        for operation in queue_operations:
            op = operation["op"]
            task_id = operation["task_id"]
            if op in {"split", "cooperate"}:
                source = operation["from_platform"]
                target = operation["to_platform"]
                parent = tasks[task_id]
                source_queue = queues[source]
                parent_index = source_queue.index(task_id)
                source_queue.pop(parent_index)
                ordinal = 1 + sum(manifest["parent_task_id"] == task_id for manifest in self._split_manifests.values())
                child_tag = "team" if op == "cooperate" else "part"
                child_ids = [f"{task_id}::{child_tag}-{ordinal:02d}-{index}" for index in (1, 2)]
                share = float(operation["share"])
                shares = [share, 1.0 - share]
                group_id = f"{task_id}::cooperation-{ordinal:02d}" if op == "cooperate" else None
                parent_spec = self._task_specs[task_id]
                for child_id, workload_share, owner in zip(child_ids, shares, (source, target)):
                    child = copy.deepcopy(parent)
                    child.update({
                        "id": child_id, "parent_task_id": task_id, "workload_share": workload_share,
                        "assignee": owner, "status": "pending", "phase": "pending",
                        "progress": 0.0, "service_progress": 0.0,
                    })
                    if child["kind"] == "delivery":
                        child["quantity"] = float(parent_spec["quantity"]) * workload_share
                    else:
                        child["service_minutes"] = float(parent_spec["service_minutes"]) * workload_share
                    if group_id is not None:
                        child["cooperation_group"] = group_id
                    tasks[child_id] = child
                source_queue.insert(parent_index, child_ids[0])
                target_queue = queues[target]
                anchor = operation.get("before_task_id")
                if anchor in target_queue:
                    target_queue.insert(target_queue.index(anchor), child_ids[1])
                else:
                    target_queue.append(child_ids[1])
                parent.update({"assignee": None, "status": "split_parent", "phase": "split_parent"})
                continue
            if op == "swap":
                other_id = operation["other_task_id"]
                left_owner = tasks[task_id].get("assignee")
                right_owner = tasks[other_id].get("assignee")
                for queue in queues.values():
                    for index, value in enumerate(queue):
                        if value == task_id:
                            queue[index] = other_id
                        elif value == other_id:
                            queue[index] = task_id
                tasks[task_id]["assignee"] = right_owner
                tasks[other_id]["assignee"] = left_owner
                continue
            source = tasks[task_id].get("assignee")
            target = operation["to_platform"]
            for queue in queues.values():
                while task_id in queue:
                    queue.remove(task_id)
            destination = queues.setdefault(target, [])
            anchor = operation.get("before_task_id") if op == "insert" else None
            if anchor in destination:
                destination.insert(destination.index(anchor), task_id)
            else:
                destination.append(task_id)
            tasks[task_id]["assignee"] = target
        for dispatch in dispatches:
            task_id = dispatch["task_id"]
            platform_id = dispatch["platform_id"]
            if task_id not in tasks:
                continue
            for queue in queues.values():
                while task_id in queue:
                    queue.remove(task_id)
            queues.setdefault(platform_id, []).insert(0, task_id)
            tasks[task_id]["assignee"] = platform_id
        return queues, tasks

    def _new_queue_issues(self, observation: dict, queues: dict[str, list[str]],
                          task_rows: dict[str, dict] | None = None) -> list[tuple[str, str, str]]:
        before = set(self._queue_schedule_issues(observation, self._queues))
        after = set(self._queue_schedule_issues(observation, queues, task_rows))
        return sorted(after - before)

    def _check_action(self, action: dict) -> tuple[dict, list[str]]:
        if not isinstance(action, dict):
            return _new_action(noop=True), ["action_must_be_object"]
        scope = action.get("scope")
        if scope not in {"NOOP", "LOCAL_REPAIR", "GLOBAL_REPLAN"}:
            return _new_action(noop=True), ["invalid_scope"]
        noop = action.get("noop")
        queue_operations = action.get("queue_operations", [])
        dispatches = action.get("dispatches", [])
        if type(noop) is not bool or not isinstance(queue_operations, list) or not isinstance(dispatches, list):
            return _new_action(noop=True), ["invalid_action_fields"]
        normalized = _new_action(scope, noop=noop, queue_operations=queue_operations, dispatches=dispatches)
        issues = []
        if action.get("action_type") != "reconstruct":
            issues.append("invalid_action_type")
        if noop and (queue_operations or dispatches or scope != "NOOP"):
            issues.append("noop_must_be_explicit_and_empty")
        if not noop and scope == "NOOP":
            issues.append("non_noop_action_requires_reconstruction_scope")
        if noop and not queue_operations and not dispatches and scope == "NOOP" and not issues:
            return normalized, []
        observation = self.public_observation()
        dispatch_catalog = {(row["platform_id"], row["task_id"]): row for row in self.candidate_dispatches()}
        dispatch_pairs = set()
        used_platforms: set[str] = set()
        used_tasks: set[str] = set()
        kept_dispatches = []
        for dispatch in dispatches:
            if not isinstance(dispatch, dict):
                issues.append("dispatch_must_be_object")
                continue
            reasons = self._dispatch_legality(observation, dispatch, dispatch_catalog)
            pair = (dispatch.get("platform_id"), dispatch.get("task_id"))
            if pair[0] in used_platforms or pair[1] in used_tasks:
                reasons.append("duplicate_platform_or_task")
            if reasons:
                issues.extend(reasons)
                continue
            used_platforms.add(pair[0]); used_tasks.add(pair[1]); dispatch_pairs.add(pair)
            kept_dispatches.append({"platform_id": pair[0], "task_id": pair[1]})
        reserved_stock: dict[tuple[str, str], float] = {}
        for task in observation["tasks"]:
            if (task.get("kind") == "delivery" and task.get("status") in {"active", "assigned"}
                    and task.get("phase") in {"assigned", "to_depot", "loading"}):
                stock_key = (str(task.get("depot_id")), str(task.get("commodity")))
                reserved_stock[stock_key] = reserved_stock.get(stock_key, 0.0) + float(task.get("quantity", 0.0))
        for dispatch in kept_dispatches:
            task = next((row for row in observation["tasks"] if row["id"] == dispatch["task_id"]), None)
            if task is None or task.get("kind") != "delivery":
                continue
            stock_key = (str(task.get("depot_id")), str(task.get("commodity")))
            depot = next((row for row in observation["depots"] if row["id"] == stock_key[0]), {})
            committed = float(reserved_stock.get(stock_key, 0.0)) + float(task.get("quantity", 0.0))
            if committed > float(depot.get("stock", {}).get(stock_key[1], 0.0)) + _EPS:
                issues.append("batch_stock_exceeded")
            reserved_stock[stock_key] = committed
        kept_operations = []
        mutated_tasks: set[str] = set()
        for operation in queue_operations:
            if not isinstance(operation, dict):
                issues.append("queue_operation_must_be_object")
                continue
            operation_task_ids = {operation.get("task_id")}
            if operation.get("op") == "swap":
                operation_task_ids.add(operation.get("other_task_id"))
            operation_task_ids.discard(None)
            if operation_task_ids & mutated_tasks:
                issues.append("task_has_multiple_queue_operations")
                continue
            reasons = self._operation_legality(observation, operation, dispatch_pairs)
            if reasons:
                issues.extend(reasons)
                continue
            mutated_tasks.update(operation_task_ids)
            kept_operations.append(copy.deepcopy(operation))
        group_dispatches: dict[str, list[tuple[str, str]]] = {}
        for platform_id, task_id in dispatch_pairs:
            spec = self._task_specs.get(task_id, {})
            group = spec.get("cooperation_group")
            if group:
                group_dispatches.setdefault(str(group), []).append((platform_id, task_id))
        for group, selected in group_dispatches.items():
            group_tasks = {task_id for task_id, spec in self._task_specs.items()
                           if str(spec.get("cooperation_group", "")) == group}
            if len(group_tasks) != 2 or {task_id for _, task_id in selected} != group_tasks:
                issues.append("cooperative_dispatch_requires_pair")
            elif len({platform_id for platform_id, _ in selected}) != 2:
                issues.append("cooperative_dispatch_requires_distinct_platforms")
        planned_queues, planned_tasks = self._planned_queue_state(observation, kept_operations, kept_dispatches)
        issues.extend(f"{code}:{task_id}:{platform_id}"
                      for code, task_id, platform_id in self._new_queue_issues(observation, planned_queues, planned_tasks))
        if issues:
            normalized["dispatches"] = kept_dispatches
            normalized["queue_operations"] = kept_operations
            if not normalized["dispatches"] and not normalized["queue_operations"]:
                normalized = _new_action(noop=True)
        return normalized, issues

    def _apply_queue_operation(self, operation: dict, time: int) -> dict:
        op = operation["op"]
        task_ids = [operation["task_id"]]
        if op == "swap":
            task_ids.append(operation["other_task_id"])
        owners = {task_id: self._base._tasks[task_id].get("assignee") for task_id in task_ids}
        before = copy.deepcopy(self._queues)
        if op in {"split", "cooperate"}:
            parent_id = operation["task_id"]
            parent_task = self._base._tasks[parent_id]
            parent_spec = self._task_specs[parent_id]
            source, target = operation["from_platform"], operation["to_platform"]
            source_queue = self._queues[source]
            parent_index = source_queue.index(parent_id)
            source_queue.pop(parent_index)
            ordinal = 1 + sum(manifest["parent_task_id"] == parent_id for manifest in self._split_manifests.values())
            operation_name = "team" if op == "cooperate" else "part"
            child_ids = [f"{parent_id}::{operation_name}-{ordinal:02d}-{index}" for index in (1, 2)]
            share = float(operation["share"])
            shares = [share, 1.0 - share]
            group_id = f"{parent_id}::cooperation-{ordinal:02d}" if op == "cooperate" else None
            for child_id, workload_share, owner in zip(child_ids, shares, (source, target)):
                child_spec = copy.deepcopy(parent_spec)
                child_spec.update({"id": child_id, "known_initial": True, "parent_task_id": parent_id,
                                   "workload_share": workload_share, "assigned_owner": owner})
                if child_spec["kind"] == "delivery":
                    child_spec["quantity"] = float(parent_spec["quantity"]) * workload_share
                else:
                    child_spec["service_minutes"] = float(parent_spec["service_minutes"]) * workload_share
                if group_id is not None:
                    child_spec["cooperation_group"] = group_id
                child_task = copy.deepcopy(parent_task)
                child_task.update({
                    **child_spec, "original_assignee": parent_task.get("original_assignee"),
                    "assignee": owner, "known": True, "discovered": True,
                    "discovery_time": time, "status": "pending", "phase": "pending",
                    "started_at": None, "completed_at": None, "deadline_missed": False,
                    "deadline_event_emitted": False, "report_generated_at": None,
                    "report_received_at": None, "failure_reason": None,
                    "service_progress": 0.0, "cargo_onboard": 0.0, "delivered": False,
                    "on_time": None, "_reassigned": False, "_new_assignment_counted": False,
                    "active_platform_id": None,
                })
                self._task_specs[child_id] = child_spec
                self._base._task_spec[child_id] = copy.deepcopy(child_spec)
                self._base._tasks[child_id] = child_task
                self._base._public_tasks[child_id] = self._base._public_task_from_truth(child_task)
                self._reported_tasks[child_id] = copy.deepcopy(self._base._public_tasks[child_id])
                self._reported_progress_counts[child_id] = 0.0
                self._task_predecessors[child_id] = list(self._task_predecessors.get(parent_id, []))
            source_queue.insert(parent_index, child_ids[0])
            target_queue = self._queues.setdefault(target, [])
            anchor = operation.get("before_task_id")
            if anchor in target_queue:
                target_queue.insert(target_queue.index(anchor), child_ids[1])
            else:
                target_queue.append(child_ids[1])
            self._reported_tasks[parent_id].update({"assignee": None, "status": "split_parent", "phase": "split_parent"})
            parent_task.update({"assignee": None, "status": "split_parent", "phase": "split_parent"})
            self._base._public_tasks[parent_id].update({"assignee": None, "status": "split_parent", "phase": "split_parent"})
            self._split_manifests[parent_id] = {
                "parent_task_id": parent_id, "mode": op, "child_ids": child_ids,
                "shares": shares, "cooperation_group": group_id,
            }
            if group_id is not None:
                self._cooperation[child_ids[0]] = [child_ids[1]]
                self._cooperation[child_ids[1]] = [child_ids[0]]
        elif op == "swap":
            left, right = task_ids
            left_owner, right_owner = owners[left], owners[right]
            for platform_id, queue in self._queues.items():
                self._queues[platform_id] = [right if task_id == left else left if task_id == right else task_id for task_id in queue]
            self._base._tasks[left]["assignee"] = right_owner
            self._base._tasks[right]["assignee"] = left_owner
            if left in self._base._public_tasks:
                self._base._public_tasks[left]["assignee"] = right_owner
            if right in self._base._public_tasks:
                self._base._public_tasks[right]["assignee"] = left_owner
        else:
            task_id = task_ids[0]
            source, target = owners[task_id], operation["to_platform"]
            if source is not None:
                self._queues[source] = [item for item in self._queues.get(source, []) if item != task_id]
            target_queue = self._queues.setdefault(target, [])
            target_queue = [item for item in target_queue if item != task_id]
            before_id = operation.get("before_task_id") if op == "insert" else None
            if before_id in target_queue:
                target_queue.insert(target_queue.index(before_id), task_id)
            else:
                target_queue.append(task_id)
            self._queues[target] = target_queue
            self._base._tasks[task_id]["assignee"] = target
            if task_id in self._base._public_tasks:
                self._base._public_tasks[task_id]["assignee"] = target
        self._base._queues = copy.deepcopy(self._queues)
        receipt = {"time": time, "operation": copy.deepcopy(operation), "before_queues": before,
                   "after_queues": copy.deepcopy(self._queues), "owners_before": owners}
        self._extension_events.append({"time": time, "type": "queue_operation", **copy.deepcopy(receipt)})
        return receipt

    def _sync_queues_from_assignments(self) -> None:
        for task_id, task in self._reported_tasks.items():
            owner = task.get("assignee")
            for platform_id in self._queues:
                if platform_id != owner and task_id in self._queues[platform_id]:
                    self._queues[platform_id].remove(task_id)
            if owner in self._queues and task.get("status") not in {"completed", "failed", "expired", "cancelled", "split_parent"}:
                if task_id not in self._queues[owner]:
                    self._queues[owner].append(task_id)
            if task.get("status") in {"completed", "failed", "expired", "cancelled", "split_parent"}:
                for queue in self._queues.values():
                    if task_id in queue:
                        queue.remove(task_id)
        self._base._queues = copy.deepcopy(self._queues)

    def _queue_head_dispatches(self, explicit_dispatches: list[dict]) -> list[dict]:
        explicit_platforms = {row["platform_id"] for row in explicit_dispatches}
        explicit_tasks = {row["task_id"] for row in explicit_dispatches}
        live_tasks = self._base._public_tasks
        heads: dict[str, str] = {}
        for platform_id in sorted(self._queues):
            platform = self._base._platforms[platform_id]
            if platform.get("failed") or platform.get("active_task_id") is not None or platform_id in explicit_platforms:
                continue
            for task_id in self._queues[platform_id]:
                task = live_tasks.get(task_id)
                if task is None or task.get("status") in {"completed", "failed", "expired", "cancelled", "split_parent"}:
                    continue
                if task.get("assignee") != platform_id or task.get("status") not in {"pending", "assigned", "released"}:
                    break
                if any(live_tasks.get(predecessor, {}).get("status") != "completed"
                       for predecessor in self._task_predecessors.get(task_id, [])):
                    break
                original_compatible = self._base._compatible
                self._base._compatible = lambda current_platform, current_task: self._public_compatible(current_platform, current_task)
                try:
                    feasible, _, _ = self._base._mission_feasible_in_truth(platform, self._base._tasks[task_id])
                finally:
                    self._base._compatible = original_compatible
                observed = self.public_observation()
                task_row = next((row for row in observed["tasks"] if row["id"] == task_id), None)
                platform_row = next((row for row in observed["platforms"] if row["id"] == platform_id), None)
                if task_row is not None and platform_row is not None and self._estimate_public_mission(platform_row, task_row, observed)["blocked_route"]:
                    feasible = False
                if feasible and task_id not in explicit_tasks:
                    heads[platform_id] = task_id
                break
        groups: dict[str, list[str]] = {}
        for task_id, spec in self._task_specs.items():
            group = spec.get("cooperation_group")
            if group:
                groups.setdefault(str(group), []).append(task_id)
        held = set()
        for group_tasks in groups.values():
            if len(group_tasks) != 2:
                continue
            platforms = [self._task_specs[task_id]["assigned_owner"] for task_id in group_tasks]
            pair_ready = all(heads.get(platform_id) == task_id for task_id, platform_id in zip(group_tasks, platforms))
            if not pair_ready:
                for task_id, platform_id in zip(group_tasks, platforms):
                    if heads.get(platform_id) == task_id:
                        heads.pop(platform_id, None)
                        held.add(task_id)
        return [{"platform_id": platform_id, "task_id": task_id, "source": "committed_queue"}
                for platform_id, task_id in sorted(heads.items())]

    def _mission_route_target(self, task: dict, phase: str | None = None) -> list[float] | None:
        phase = phase or task.get("phase")
        if task["kind"] == "inspection":
            return list(task["position"]) if phase in {"to_inspection", "assigned", "pending", "released"} else None
        if phase in {"to_depot", "assigned", "pending", "released"}:
            return list(self._base._depots[task["depot_id"]]["position"])
        if phase in {"to_delivery_site", "in_transit"}:
            return list(task["position"])
        return None

    def _prepare_step_physics(self, dispatches: list[dict], time: int) -> tuple[dict[str, tuple[float, float]], list[dict]]:
        original: dict[str, tuple[float, float]] = {}
        blocked_failures = []
        by_platform = {row["platform_id"]: row["task_id"] for row in dispatches}
        for platform_id in sorted(self._base._platforms):
            platform = self._base._platforms[platform_id]
            task_id = platform.get("active_task_id") or by_platform.get(platform_id)
            if task_id is None:
                continue
            task = self._base._tasks[task_id]
            target = self._mission_route_target(task, task.get("phase") if platform.get("active_task_id") else None)
            if target is None:
                continue
            start = list(platform["position"])
            distance = _distance(start, target)
            nominal_speed = float(platform["speed_km_min"])
            segment_end = target
            if distance > nominal_speed > _EPS:
                fraction = nominal_speed / distance
                segment_end = [start[0] + (target[0] - start[0]) * fraction,
                               start[1] + (target[1] - start[1]) * fraction]
            metrics = self._segment_environment(start, segment_end, nominal_speed, float(platform["energy_per_km"]))
            if metrics["blocked"]:
                if platform.get("active_task_id") is not None:
                    self._base._fail_task(task, platform_id, "blocked_route", time, releasable=True)
                    blocked_failures.append({"time": time, "type": "route_blocked", "task_id": task_id,
                                             "platform_id": platform_id, "environmental_risk": metrics["risk"]})
                continue
            segment_distance = _distance(start, segment_end)
            if segment_distance <= _EPS or not self._environment:
                continue
            effective_speed = segment_distance / max(metrics["minutes"], _EPS)
            effective_energy_rate = metrics["energy"] / max(segment_distance, _EPS)
            if not math.isclose(effective_speed, nominal_speed, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(effective_energy_rate, float(platform["energy_per_km"]), rel_tol=1e-9, abs_tol=1e-9):
                original[platform_id] = (float(platform["speed_km_min"]), float(platform["energy_per_km"]))
                platform["speed_km_min"] = effective_speed
                platform["energy_per_km"] = effective_energy_rate
        return original, blocked_failures

    def _restore_step_physics(self, original: dict[str, tuple[float, float]]) -> None:
        for platform_id, values in original.items():
            platform = self._base._platforms[platform_id]
            platform["speed_km_min"], platform["energy_per_km"] = values

    def _augment_movement_risk(self, events: list[dict]) -> None:
        for event in events:
            if event.get("type") != "move":
                continue
            segment = self._segment_environment(list(event["from"]), list(event["to"]), 1.0, 0.0)
            event["environmental_risk"] = float(segment["risk"])
            event["blocked_route"] = bool(segment["blocked"])

    def _apply_extension_events(self, time: int) -> None:
        for row in self.scenario["event_tape"]:
            if row["time"] != time or row["event_id"] in self._event_ids_applied:
                continue
            kind = row["kind"]
            self._event_ids_applied.add(row["event_id"])
            if kind in {"platform_failure", "energy_drop"}:
                continue
            if kind == "energy_fraction_drop":
                platform_id = str(row["platform_id"])
                platform = self._base._platforms[platform_id]
                public = self._base._public_platforms[platform_id]
                previous_energy = max(0.0, float(platform.get("energy", 0.0)))
                new_energy = max(0.0, previous_energy * (1.0 - float(row["loss_fraction"])))
                platform["energy"] = new_energy
                public["energy"] = new_energy
                event = {
                    "time": time, "type": "energy_fraction_drop", "event_id": row["event_id"],
                    "platform_id": platform_id, "old_energy": previous_energy,
                    "new_energy": new_energy, "loss_fraction": float(row["loss_fraction"]),
                }
                self._extension_events.append(copy.deepcopy(event))
                continue
            if kind == "task_arrival":
                task_id = row["task_id"]
                task = self._base._tasks[task_id]
                task.update({"known": True, "discovered": True, "discovery_time": time,
                             "status": "pending", "phase": "pending"})
                self._base._public_tasks[task_id] = self._base._public_task_from_truth(task)
                event = {"time": time, "type": "task_arrived", "event_id": row["event_id"], "task_id": task_id}
            elif kind in {"task_cancel", "priority_change", "deadline_change"}:
                task_id = row["task_id"]
                task = self._base._tasks[task_id]
                if kind == "task_cancel":
                    platform_id = task.get("active_platform_id")
                    if platform_id is not None:
                        if float(task.get("cargo_onboard", 0.0)) > _EPS:
                            self._base._log("cargo_lost", time, task_id=task_id, platform_id=platform_id,
                                            quantity=float(task["cargo_onboard"]))
                            task["cargo_onboard"] = 0.0
                        self._base._free_platform(platform_id)
                        task["active_platform_id"] = None
                    task.update({"status": "cancelled", "phase": "cancelled", "failure_reason": "task_cancelled"})
                    for queue in self._queues.values():
                        if task_id in queue:
                            queue.remove(task_id)
                elif kind == "priority_change":
                    task["priority"] = float(row["new_priority"])
                else:
                    task["deadline"] = int(row["new_deadline"])
                # Future split/cooperation children must inherit the current
                # public mutable task fields, rather than the reset template.
                if task_id in self._task_specs:
                    self._task_specs[task_id].update({
                        "priority": task.get("priority"), "deadline": task.get("deadline"),
                    })
                if task_id in self._base._task_spec:
                    self._base._task_spec[task_id].update({
                        "priority": task.get("priority"), "deadline": task.get("deadline"),
                    })
                if task_id in self._base._public_tasks:
                    self._base._public_tasks[task_id].update({
                        key: task[key] for key in ("status", "phase", "failure_reason", "priority", "deadline") if key in task
                    })
                event = {"time": time, "type": kind, "event_id": row["event_id"], "task_id": task_id,
                         "new_value": row.get("new_priority", row.get("new_deadline"))}
            elif kind in {"capability_loss", "load_change"}:
                platform_id = row["platform_id"]
                platform = self._base._platforms[platform_id]
                public = self._base._public_platforms[platform_id]
                if kind == "capability_loss":
                    capability = row["capability"]
                    platform["capabilities"] = [item for item in platform["capabilities"] if item != capability]
                    public["capabilities"] = list(platform["capabilities"])
                    active_task_id = platform.get("active_task_id")
                    if active_task_id is not None:
                        task = self._base._tasks[active_task_id]
                        required = "inspect" if task["kind"] == "inspection" else "deliver"
                        if required not in platform["capabilities"]:
                            # Leave an already-started mission physically intact
                            # until the same-tick public safety hook applies a
                            # capability hold.  Failing here would erase
                            # progress/cargo before the disposition can choose
                            # pause or an explicitly accounted terminal loss.
                            task.setdefault("safety_event_markers", []).append({
                                "reason": "active_task_capability_lost",
                                "time": int(time),
                                "required_capability": required,
                            })
                else:
                    new_capacity = float(row["capacity_kg"])
                    platform["capacity_kg"] = new_capacity
                    public["capacity_kg"] = new_capacity
                    active_task_id = platform.get("active_task_id")
                    if active_task_id is not None:
                        task = self._base._tasks[active_task_id]
                        if task["kind"] == "delivery" and float(task["quantity"]) > new_capacity + _EPS:
                            self._base._fail_task(task, platform_id, "payload_capacity_lost", time, releasable=True)
                event = {"time": time, "type": kind, "event_id": row["event_id"],
                         "platform_id": platform_id, "capability": row.get("capability"),
                         "capacity_kg": row.get("capacity_kg")}
            elif kind in {"communication_drop", "packet_loss", "communication_delay", "communication_split"}:
                changed = []
                for endpoints in row["links"]:
                    key = _link_key(*endpoints)
                    link = next((item for item in self._communication if _link_key(item["source"], item["target"]) == key), None)
                    if link is None:
                        continue
                    old_quality = float(link["quality"])
                    if "quality" in row:
                        link["quality"] = float(row["quality"])
                    elif "loss_rate" in row:
                        link["loss_rate"] = float(row["loss_rate"])
                        link["quality"] = 1.0 - float(row["loss_rate"])
                    if kind == "communication_split":
                        link["quality"] = 0.0
                    if "latency_minutes" in row:
                        link["latency_minutes"] = float(row["latency_minutes"])
                    link["loss_rate"] = max(float(link["loss_rate"]), 1.0 - float(link["quality"]))
                    changed.append({"source": link["source"], "target": link["target"],
                                    "old_quality": old_quality, "new_quality": link["quality"],
                                    "latency_minutes": link["latency_minutes"], "loss_rate": link["loss_rate"]})
                event = {"time": time, "type": "communication_changed", "event_id": row["event_id"], "links": changed}
            elif kind in {"environment_perturbation", "wind_change", "obstacle_change", "threat_change", "passability_change"}:
                x, y, width, height = row["patch"]
                cells = {(x + dx, y + dy) for dx in range(width) for dy in range(height)}
                speed_factor = float(row.get("speed_factor", 0.65 if kind in {"environment_perturbation", "wind_change"} else 1.0))
                energy_factor = float(row.get("energy_per_km_factor", 1.25 if kind in {"environment_perturbation", "wind_change"} else 1.0))
                for node in self._environment:
                    if tuple(node["grid_cell"]) in cells:
                        node["speed_factor"] = speed_factor
                        node["energy_per_km_factor"] = energy_factor
                        if "wind_vector" in row:
                            node["wind_vector"] = list(row["wind_vector"])
                        risk_delta = float(row.get("risk_delta", 0.0))
                        if kind == "threat_change":
                            risk_delta = float(row.get("risk_level", risk_delta)) - float(node["risk_level"])
                        node["risk_level"] = min(1.0, max(0.0, float(node["risk_level"]) + risk_delta))
                        node["risk"] = float(node["risk_level"])
                        if kind in {"obstacle_change", "passability_change", "environment_perturbation"}:
                            if "blocked_cells" in row:
                                node["blocked"] = node["grid_cell"] in row["blocked_cells"]
                            else:
                                node["blocked"] = node["grid_cell"] == row.get("blocked_cell") if row.get("blocked_cell") is not None else bool(row.get("blocked", node["blocked"]))
                event = {"time": time, "type": "environment_changed", "event_id": row["event_id"],
                         "patch": list(row["patch"]), "changed_cells": [list(cell) for cell in sorted(cells)],
                         "wind_vector": copy.deepcopy(row.get("wind_vector")), "risk_delta": float(row.get("risk_delta", 0.0)),
                         "blocked_cell": copy.deepcopy(row.get("blocked_cell")), "local_speed_factor": speed_factor,
                         "energy_per_km_factor": energy_factor}
            else:
                continue
            self._extension_events.append(copy.deepcopy(event))

    def _logical_owner_set(self, task_id: str) -> list[str]:
        manifest = self._split_manifests.get(task_id)
        if manifest is None:
            owner = self._base._tasks.get(task_id, {}).get("assignee")
            return [] if owner is None else [str(owner)]
        return sorted({str(self._base._tasks[child_id].get("assignee")) for child_id in manifest["child_ids"]
                       if self._base._tasks[child_id].get("assignee") is not None})

    def _logical_completed(self, task_id: str) -> tuple[bool, int | None]:
        manifest = self._split_manifests.get(task_id)
        if manifest is None:
            task = self._base._tasks[task_id]
            return task.get("status") == "completed", task.get("completed_at")
        children = [self._base._tasks[child_id] for child_id in manifest["child_ids"]]
        complete = all(row.get("status") == "completed" for row in children)
        completed_at = max((int(row["completed_at"]) for row in children if row.get("completed_at") is not None), default=None)
        return complete, completed_at if complete else None

    def _record_reward_primitives(self, action: dict, start_time: int, base_events: list[dict], extension_events: list[dict],
                                  operation_receipts: list[dict], owners_before: dict[str, list[str]],
                                  completion_before: dict[str, bool], known_before: dict[str, bool]) -> tuple[list[dict], dict, dict]:
        events = copy.deepcopy(base_events)
        self._augment_movement_risk(events)
        events.extend(copy.deepcopy(extension_events))
        for event in list(events):
            if event.get("type") == "platform_failed":
                events.append({"time": event.get("time"), "type": "platform_failure",
                               "platform_id": event.get("platform_id"), "event_id": event.get("disturbance_id")})
        end_time = int(self.time)
        logical_action_ids = set()
        for dispatch in action["dispatches"]:
            task_id = str(dispatch["task_id"])
            logical_action_ids.add(str(self._task_specs.get(task_id, {}).get("parent_task_id", task_id)))
        for operation in action["queue_operations"]:
            for key in ("task_id", "other_task_id"):
                task_id = operation.get(key)
                if isinstance(task_id, str):
                    logical_action_ids.add(str(self._task_specs.get(task_id, {}).get("parent_task_id", task_id)))
        for task_id in sorted(logical_action_ids):
            previous = owners_before.get(task_id, [])
            current = self._logical_owner_set(task_id)
            if previous == current:
                continue
            events.append({"time": start_time, "type": "task_owner_change", "task_id": task_id,
                           "previous_owners": list(previous), "new_owners": list(current)})
            if previous:
                events.append({"time": start_time, "type": "task_switch", "task_id": task_id,
                               "previous_owners": list(previous), "new_owners": list(current)})
        reorder_count = 0
        for receipt in operation_receipts:
            operation = receipt["operation"]
            if operation.get("op") != "insert" or operation.get("from_platform") != operation.get("to_platform"):
                continue
            before = receipt["before_queues"].get(operation["to_platform"], [])
            after = receipt["after_queues"].get(operation["to_platform"], [])
            if before != after:
                task_id = str(operation["task_id"])
                logical_id = str(self._task_specs.get(task_id, {}).get("parent_task_id", task_id))
                events.append({"time": start_time, "type": "queue_reorder", "task_id": logical_id,
                               "platform_id": operation["to_platform"], "before_queues": list(before),
                               "after_queues": list(after)})
                reorder_count += 1
        for task_id in sorted(self._original_task_specs):
            was_complete = completion_before.get(task_id, False)
            is_complete, completed_at = self._logical_completed(task_id)
            if is_complete and not was_complete:
                events.append({"time": completed_at if completed_at is not None else end_time,
                               "type": "logical_task_completed", "task_id": task_id})
            task = self._base._tasks[task_id]
            if not known_before.get(task_id, bool(task.get("known"))):
                continue
            if task.get("status") == "cancelled":
                finish = end_time
            else:
                finish = min(end_time, completed_at) if is_complete and completed_at is not None else end_time
            deadline = float(task.get("deadline", end_time))
            before_late = max(0.0, min(float(start_time), float(finish)) - deadline)
            after_late = max(0.0, float(finish) - deadline)
            delay_minutes = max(0.0, after_late - before_late)
            if delay_minutes > _EPS:
                events.append({"time": end_time, "type": "delay_accrued", "task_id": task_id,
                               "minutes": float(delay_minutes)})
        communication_messages = sum(int(event.get("message_count", 0)) for event in events
                                     if event.get("type") == "communication_tx")
        completion_count = sum(event.get("type") == "logical_task_completed" for event in events)
        delay_minutes = sum(float(event.get("minutes", 0.0)) for event in events if event.get("type") == "delay_accrued")
        energy_units = sum(float(event.get("energy", 0.0)) for event in events
                           if event.get("type") in {"move", "service"})
        risk_units = sum(float(event.get("environmental_risk", 0.0)) for event in events if event.get("type") == "move")
        risk_units += sum(event.get("type") in {"task_failed", "platform_failure"} for event in events)
        switch_count = len({event.get("task_id") for event in events if event.get("type") == "task_switch"})
        owner_changes = len({event.get("task_id") for event in events if event.get("type") == "task_owner_change"})
        reconstruction_size = owner_changes + reorder_count
        reconstruction_count = int(reconstruction_size > 0)
        primitives = {
            "completion_count": int(completion_count), "delay_minutes": float(delay_minutes),
            "energy_units": float(energy_units), "risk_units": float(risk_units),
            "communication_messages": int(communication_messages), "switch_count": int(switch_count),
            "reconstruction_count": reconstruction_count, "reconstruction_size": int(reconstruction_size),
        }
        weights = self.reward_config
        terms = {
            "completion": weights["completion"] * primitives["completion_count"],
            "delay": -weights["delay"] * primitives["delay_minutes"],
            "energy": -weights["energy"] * primitives["energy_units"],
            "risk": -weights["risk"] * primitives["risk_units"],
            "communication": -weights["communication"] * primitives["communication_messages"],
            "switch": -weights["switch"] * primitives["switch_count"],
            "reconstruction": -weights["reconstruction"] * primitives["reconstruction_count"],
            "reconstruction_scale": -weights["reconstruction_scale"] * primitives["reconstruction_size"],
        }
        return events, primitives, terms

    def _reward_formula(self) -> dict:
        return {"schema": "event-relief-reward/1", "weights": copy.deepcopy(self.reward_config),
                "formula": "completion*w_completion-delay_minutes*w_delay-energy_units*w_energy-risk_units*w_risk-communication_messages*w_communication-switch_count*w_switch-reconstruction_count*w_reconstruction-reconstruction_size*reconstruction_scale",
                "reconstruction_count": "1 iff at least one task owner set or same-platform queue order changes",
                "reconstruction_size": "unique logical tasks with owner changes plus same-platform queue reorder mutations",
                "delay_minutes": "positive lateness accrued by known original tasks until completion or cancellation",
                "risk_units": "sum move.environmental_risk plus one per task_failed and platform_failure primitive"}

    def step(self, full_environment_action: dict) -> tuple[dict, float, bool, bool, dict]:
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        action, issues = self._check_action(full_environment_action)
        if issues:
            raise ValueError(f"illegal event environment action: {sorted(set(issues))}")
        time = self.time
        observation = self.public_observation()
        base_event_start = len(self._base.events)
        extension_start = len(self._extension_events)
        pending_due = [row for row in self._pending_communications if int(row["delivery_time"]) <= time]
        self._pending_communications = [row for row in self._pending_communications if int(row["delivery_time"]) > time]
        pending_dispatches = []
        for row in pending_due:
            pending_dispatches.append({
                "platform_id": row["platform_id"], "task_id": row["task_id"],
                "source": "communication_pending",
            })
            self._extension_events.append({
                "time": time, "type": "communication_delivered", "reason": row.get("reason", "dispatch"),
                "platform_id": row["platform_id"], "task_id": row["task_id"],
                "delivery_time": time, "retry_count": int(row.get("retry_count", 0)),
                "message_count": 1,
            })
        owners_before = {task_id: self._logical_owner_set(task_id) for task_id in self._original_task_specs}
        completion_before = {task_id: self._logical_completed(task_id)[0] for task_id in self._original_task_specs}
        known_before = {task_id: bool(self._base._tasks[task_id].get("known")) for task_id in self._original_task_specs}
        operation_receipts = [self._apply_queue_operation(operation, time) for operation in action["queue_operations"]]
        communication_receipts = []

        def command_receipt(reason: str, source: str, target: str, *, task_id: str | None = None,
                            operation: str | None = None) -> dict:
            metrics = _communication_metrics(observation, source, target)
            delay = _communication_delay_minutes(metrics)
            latency_ticks = int(math.ceil(max(0.0, float(metrics.get("latency_minutes", 0.0)) - 0.1 - _EPS)))
            retry_count = int(max(0, delay - latency_ticks))
            receipt = {
                "reason": reason, "source": source, "target": target, "task_id": task_id,
                "operation": operation, "connected": bool(metrics.get("connected")),
                "quality": float(metrics.get("quality", 0.0)),
                "latency_minutes": float(metrics.get("latency_minutes", 0.0)),
                "loss_rate": float(metrics.get("loss_rate", 1.0)),
                "retry_count": retry_count, "delay_minutes": int(delay),
                "sent_time": time, "delivery_time": time + int(delay),
                "delivered": int(delay) == 0,
            }
            communication_receipts.append(copy.deepcopy(receipt))
            self._extension_events.append({
                "time": time, "type": "communication_tx", "reason": reason,
                "source": source, "target": target, "task_id": task_id,
                "operation": operation, "quality": receipt["quality"],
                "latency_minutes": receipt["latency_minutes"], "loss_rate": receipt["loss_rate"],
                "retry_count": retry_count, "delivery_time": time + int(delay),
                "delivered": int(delay) == 0, "message_count": 1,
            })
            return receipt

        for operation in action["queue_operations"]:
            task_id = operation.get("task_id")
            task_row = next((row for row in observation.get("tasks", []) if row.get("id") == task_id), {})
            source = operation.get("from_platform") or task_row.get("assignee") or _COORDINATOR
            target = operation.get("to_platform") or source
            command_receipt("queue_operation", str(source), str(target), task_id=task_id, operation=operation.get("op"))

        explicit_dispatches = []
        explicit_for_heads = list(pending_dispatches)
        explicit_for_heads.extend(action["dispatches"])
        for dispatch in action["dispatches"]:
            task_row = next((row for row in observation.get("tasks", []) if row.get("id") == dispatch.get("task_id")), {})
            source = task_row.get("assignee") or _COORDINATOR
            receipt = command_receipt("dispatch", str(source), str(dispatch["platform_id"]), task_id=dispatch["task_id"])
            if receipt["delay_minutes"] > 0:
                self._pending_communications.append({
                    "platform_id": dispatch["platform_id"], "task_id": dispatch["task_id"],
                    "delivery_time": receipt["delivery_time"], "retry_count": receipt["retry_count"],
                    "reason": "dispatch",
                })
            else:
                explicit_dispatches.append({**copy.deepcopy(dispatch), "source": "reconstruction"})
        automatic_dispatches = self._queue_head_dispatches(explicit_for_heads)
        immediate_automatic = []
        for dispatch in automatic_dispatches:
            task_row = next((row for row in observation.get("tasks", []) if row.get("id") == dispatch.get("task_id")), {})
            source = task_row.get("assignee") or _COORDINATOR
            receipt = command_receipt("queue_head_dispatch", str(source), str(dispatch["platform_id"]), task_id=dispatch["task_id"])
            if receipt["delay_minutes"] > 0:
                self._pending_communications.append({
                    "platform_id": dispatch["platform_id"], "task_id": dispatch["task_id"],
                    "delivery_time": receipt["delivery_time"], "retry_count": receipt["retry_count"],
                    "reason": "queue_head_dispatch",
                })
            else:
                immediate_automatic.append(dispatch)
        dispatches = pending_dispatches + explicit_dispatches + immediate_automatic
        used_platforms, used_tasks = set(), set()
        operational_dispatches = []
        deferred_pending_dispatches: list[dict] = []
        for row in dispatches:
            if row["platform_id"] in used_platforms or row["task_id"] in used_tasks:
                # A pending communication was already delivered to this step
                # and must not disappear merely because another dispatch won
                # the platform/task de-duplication.  Keep it queued for a
                # deterministic retry; explicit commands remain one-shot.
                if row.get("source") == "communication_pending":
                    deferred_pending_dispatches.append(row)
                continue
            used_platforms.add(row["platform_id"])
            used_tasks.add(row["task_id"])
            operational_dispatches.append(row)
            if row["source"] == "committed_queue":
                self._extension_events.append({"time": time, "type": "queue_head_dispatch",
                                               "platform_id": row["platform_id"], "task_id": row["task_id"]})
        pending_keys = {
            (row["platform_id"], row["task_id"])
            for row in self._pending_communications
        }
        for row in deferred_pending_dispatches:
            key = (row["platform_id"], row["task_id"])
            if key in pending_keys:
                continue
            self._pending_communications.append({
                "platform_id": row["platform_id"], "task_id": row["task_id"],
                "delivery_time": time + 1,
                "retry_count": int(row.get("retry_count", 0)) + 1,
                "reason": "dispatch_dedup_retry",
            })
            self._extension_events.append({
                "time": time, "type": "dispatch_dedup_retry",
                "platform_id": row["platform_id"], "task_id": row["task_id"],
                "delivery_time": time + 1,
            })
        # Cooperative workloads are an atomic dispatch.  A command can reach
        # one child before its partner (or one queue head can become ready
        # first), so never execute a partial pair.  Put the ready child back
        # into the pending queue and retry after one minute.  This preserves
        # the pair constraint without converting a valid candidate branch into
        # a runtime exception.
        cooperative_groups: dict[str, list[dict]] = {}
        for row in operational_dispatches:
            group = self._task_specs.get(row["task_id"], {}).get("cooperation_group")
            if group:
                cooperative_groups.setdefault(str(group), []).append(row)
        held_dispatches: list[dict] = []
        for group_rows in cooperative_groups.values():
            platforms = {str(row["platform_id"]) for row in group_rows}
            if len(group_rows) != 2 or len(platforms) != 2:
                held_dispatches.extend(group_rows)
        if held_dispatches:
            held_keys = {(row["platform_id"], row["task_id"]) for row in held_dispatches}
            operational_dispatches = [
                row for row in operational_dispatches
                if (row["platform_id"], row["task_id"]) not in held_keys
            ]
            pending_keys = {
                (row["platform_id"], row["task_id"])
                for row in self._pending_communications
            }
            for row in held_dispatches:
                key = (row["platform_id"], row["task_id"])
                if key in pending_keys:
                    continue
                self._pending_communications.append({
                    "platform_id": row["platform_id"], "task_id": row["task_id"],
                    "delivery_time": time + 1, "retry_count": 0,
                    "reason": "cooperative_pair_wait",
                })
                self._extension_events.append({
                    "time": time, "type": "cooperative_pair_wait",
                    "platform_id": row["platform_id"], "task_id": row["task_id"],
                    "delivery_time": time + 1,
                })
        physical_overrides, blocked_events = self._prepare_step_physics(operational_dispatches, time)
        self._extension_events.extend(blocked_events)
        selected_rows = []
        for row in operational_dispatches:
            task = self._base._public_tasks[row["task_id"]]
            platform = self._base._public_platforms[row["platform_id"]]
            completion, required_energy = self._base._estimated_mission(platform, task, time)
            selected_rows.append({"id": f"dispatch:{row['platform_id']}:{row['task_id']}", "type": "dispatch",
                                  "platform_id": row["platform_id"], "task_id": row["task_id"],
                                  "kind": task["kind"], "estimated_completion": completion,
                                  "required_energy": required_energy, "locked": False})
        old_policy = self._base._policy
        old_compatible = self._base._compatible
        old_candidates = self._base.candidates
        self._base._policy = "LEARNED"
        self._base._compatible = lambda platform, task: self._public_compatible(platform, task)
        self._base.candidates = lambda: copy.deepcopy(selected_rows) + [{"id": "NOOP", "type": "noop"}]
        action_ids = [row["id"] for row in selected_rows] or ["NOOP"]
        old_advance = self._base._advance
        # The wrapper owns safety holds, while the base environment still
        # performs its normal minute-step physics for every other platform.
        self._base._advance = self._advance_with_safety_holds
        try:
            _, _, terminated, truncated, _ = self._base.step(action_ids)
        finally:
            self._base._advance = old_advance
            self._base._policy = old_policy
            self._base._compatible = old_compatible
            self._base.candidates = old_candidates
            self._restore_step_physics(physical_overrides)
        self._apply_extension_events(self.time)
        base_events = copy.deepcopy(self._base.events[base_event_start:])
        self._refresh_reported_state(base_events)
        self._sync_queues_from_assignments()
        extension_events = copy.deepcopy(self._extension_events[extension_start:])
        raw_events, reward_primitives, reward_terms = self._record_reward_primitives(
            action, time, base_events, extension_events, operation_receipts,
            owners_before, completion_before, known_before,
        )
        reward = float(sum(reward_terms.values()))
        next_observation = self.public_observation()
        step_switches = sum(event.get("type") == "task_switch" for event in raw_events)
        owner_changes = sum(event.get("type") == "task_owner_change" for event in raw_events)
        reorder_count = sum(event.get("type") == "queue_reorder" for event in raw_events)
        step_row = {
            "time": time, "full_environment_action": copy.deepcopy(action),
            "public_observation": copy.deepcopy(observation), "next_public_observation": copy.deepcopy(next_observation),
            "operation_receipts": copy.deepcopy(operation_receipts),
            "communication_receipts": copy.deepcopy(communication_receipts),
            "operational_dispatches": copy.deepcopy(operational_dispatches),
            "reward": reward, "reward_primitives": copy.deepcopy(reward_primitives),
            "reward_terms": copy.deepcopy(reward_terms), "reward_formula": self._reward_formula(),
            "raw_events": copy.deepcopy(raw_events),
        }
        self._steps.append(step_row)
        self._public_history.append(copy.deepcopy(next_observation))
        info = {
            "full_environment_action": copy.deepcopy(action),
            "operation_receipts": copy.deepcopy(operation_receipts), "raw_events": copy.deepcopy(raw_events),
            "communication_receipts": copy.deepcopy(communication_receipts),
            "operational_dispatches": copy.deepcopy(operational_dispatches),
            "requested_reconstruction_count": int(not action["noop"]),
            "effective_reconstruction_count": int(reward_primitives["reconstruction_count"] > 0),
            "reconstruction_size": int(reward_primitives["reconstruction_size"]),
            "explicit_dispatch_count": len(action["dispatches"]),
            "operational_dispatch_count": len(operational_dispatches),
            "task_switch_count": int(step_switches), "task_owner_change_count": int(owner_changes),
            "queue_reorder_count": int(reorder_count),
            "reward_primitives": copy.deepcopy(reward_primitives), "reward_terms": copy.deepcopy(reward_terms),
            "reward_formula": self._reward_formula(),
        }
        return next_observation, float(reward), bool(terminated), bool(truncated), info

    def _raw_truth(self) -> dict:
        if not isinstance(getattr(self, "_safety_paused_tasks", None), dict):
            self._safety_paused_tasks = {}
        tasks = []
        for task_id in sorted(self._original_task_specs):
            task = self._base._tasks[task_id]
            spec = self._original_task_specs[task_id]
            manifest = self._split_manifests.get(task_id)
            if manifest is None:
                children = [task]
                workload_progress = float(task.get("service_progress", 0.0))
                service_total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
                progress = min(1.0, workload_progress / max(1.0, service_total))
                status = task["status"]
                completed_at = task.get("completed_at")
                delivered = bool(task.get("delivered"))
                on_time = task.get("on_time")
                failure_reason = task.get("failure_reason")
                cargo = float(task.get("cargo_onboard", 0.0))
                assignees = [] if task.get("assignee") is None else [str(task["assignee"])]
                safety_paused = task_id in self._safety_paused_tasks
                safety_hold = copy.deepcopy(self._safety_paused_tasks.get(task_id))
                if safety_paused and status not in {"completed", "failed", "cancelled", "expired"}:
                    status = "paused"
            else:
                children = [self._base._tasks[child_id] for child_id in manifest["child_ids"]]
                complete = all(child.get("status") == "completed" for child in children)
                failed = any(child.get("status") == "failed" for child in children)
                safety_paused = any(child_id in self._safety_paused_tasks for child_id in manifest["child_ids"])
                safety_hold = {
                    child_id: copy.deepcopy(self._safety_paused_tasks[child_id])
                    for child_id in manifest["child_ids"] if child_id in self._safety_paused_tasks
                }
                status = "completed" if complete else ("failed" if failed else ("paused" if safety_paused else "pending"))
                service_total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
                workload_progress = sum(float(child.get("service_progress", 0.0)) for child in children)
                progress = sum(float(self._task_specs[child["id"]]["workload_share"]) * min(
                    1.0, float(child.get("service_progress", 0.0)) / max(1.0, float(self._task_specs[child["id"]].get("service_minutes", self._task_specs[child["id"]].get("load_minutes", 1) + self._task_specs[child["id"]].get("unload_minutes", 1)))))
                    for child in children)
                completed_at = max((int(child["completed_at"]) for child in children if child.get("completed_at") is not None), default=None) if complete else None
                delivered = all(bool(child.get("delivered")) for child in children) if spec["kind"] == "delivery" else False
                on_time = all(child.get("on_time") is True for child in children) if complete else None
                failure_reason = next((child.get("failure_reason") for child in children if child.get("failure_reason")), None)
                cargo = sum(float(child.get("cargo_onboard", 0.0)) for child in children)
                assignees = sorted({str(child.get("assignee")) for child in children if child.get("assignee") is not None})
            demand_quantity = float(spec["quantity"]) if spec["kind"] == "delivery" else None
            delivered_quantity = (
                sum(float(child.get("quantity", 0.0)) for child in children if child.get("delivered"))
                if demand_quantity is not None else None
            )
            tasks.append({
                "id": task_id, "kind": spec["kind"], "position": copy.deepcopy(spec["position"]),
                "deadline": task["deadline"], "priority": task["priority"], "status": status,
                "assignee": assignees[0] if len(assignees) == 1 else None, "assignees": assignees,
                "original_assignee": task.get("original_assignee"),
                "known": bool(task.get("known")), "discovered": bool(task.get("discovered")),
                "progress": min(1.0, progress), "service_progress": float(workload_progress),
                "started_at": min((child.get("started_at") for child in children if child.get("started_at") is not None), default=None),
                "completed_at": completed_at, "deadline_missed": bool(task.get("deadline_missed")),
                "failure_reason": failure_reason, "delivered": delivered, "on_time": on_time,
                "demand_quantity": demand_quantity,
                "delivered_quantity": delivered_quantity,
                "unmet_demand_quantity": (
                    max(0.0, demand_quantity - float(delivered_quantity))
                    if demand_quantity is not None and delivered_quantity is not None else None
                ),
                "cargo_onboard": float(cargo), "split_mode": manifest["mode"] if manifest else None,
                "subtask_ids": list(manifest["child_ids"]) if manifest else [],
                "safety_paused": bool(safety_paused), "safety_hold": safety_hold,
            })
        subtasks = []
        for parent_id, manifest in sorted(self._split_manifests.items()):
            for child_id in manifest["child_ids"]:
                child = self._base._tasks[child_id]
                spec = self._task_specs[child_id]
                total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
                subtasks.append({
                    "id": child_id, "parent_task_id": parent_id, "mode": manifest["mode"],
                    "cooperation_group": spec.get("cooperation_group"), "workload_share": float(spec["workload_share"]),
                    "workload_total": float(spec.get("quantity", spec.get("service_minutes", total))),
                    "quantity": float(spec["quantity"]) if spec["kind"] == "delivery" else None,
                    "service_minutes": float(spec["service_minutes"]) if spec["kind"] == "inspection" else None,
                    "assignee": child.get("assignee"), "status": child.get("status"),
                    "progress": min(1.0, float(child.get("service_progress", 0.0)) / max(1.0, total)),
                    "service_progress": float(child.get("service_progress", 0.0)),
                    "started_at": child.get("started_at"), "completed_at": child.get("completed_at"),
                    "delivered": bool(child.get("delivered")), "on_time": child.get("on_time"),
                    "failure_reason": child.get("failure_reason"),
                    "safety_paused": child_id in self._safety_paused_tasks,
                    "safety_hold": copy.deepcopy(self._safety_paused_tasks.get(child_id)),
                })
        platforms = []
        for platform_id in sorted(self._base._platforms):
            platform = self._base._platforms[platform_id]
            platforms.append({
                "id": platform_id, "type": platform["type"], "position": copy.deepcopy(platform["position"]),
                "energy_initial": float(self._base._platform_spec[platform_id]["energy_initial"]),
                "energy": float(platform["energy"]), "energy_used": float(platform["energy_used"]),
                "failed": bool(platform["failed"]), "active_task_id": platform.get("active_task_id"),
                "capacity_kg": float(platform["capacity_kg"]),
            })
        depots = [
            {"id": depot_id, "stock": {key: float(value) for key, value in sorted(depot["stock"].items())}}
            for depot_id, depot in sorted(self._base._depots.items())
        ]
        return {"time": self.time, "tasks": tasks, "subtasks": subtasks, "platforms": platforms, "depots": depots}

    def episode_metrics(self) -> dict:
        """Summarize current partial or terminal state without mutating the simulator."""
        truth = self._raw_truth()
        tasks = truth["tasks"]
        platforms = truth["platforms"]
        steps = self._steps
        events = self._base.events + self._extension_events
        completed = [task for task in tasks if task["status"] == "completed"]
        completed_ids = {task["id"] for task in completed}
        delayed_by_task: dict[str, float] = {}
        on_time_by_task: dict[str, bool | None] = {}
        critical_outcomes: list[bool | None] = []
        critical_count = 0
        for task in tasks:
            deadline = task.get("deadline")
            if type(deadline) not in {int, float} or not math.isfinite(float(deadline)):
                delayed_by_task[task["id"]] = 0.0
                on_time_by_task[task["id"]] = None
            else:
                finished = task["status"] == "completed"
                finish_time = task.get("completed_at") if finished else truth["time"]
                delayed_by_task[task["id"]] = max(0.0, float(finish_time) - float(deadline))
                if finished:
                    on_time_by_task[task["id"]] = float(finish_time) <= float(deadline)
                elif task["status"] == "cancelled":
                    on_time_by_task[task["id"]] = None
                elif truth["time"] >= deadline:
                    on_time_by_task[task["id"]] = False
                else:
                    on_time_by_task[task["id"]] = None
            if float(task.get("priority", 0.0)) >= 5.0:
                critical_count += 1
                complete_on_time = on_time_by_task[task["id"]]
                critical_outcomes.append(
                    False if task["status"] in {"failed", "cancelled"} else
                    (None if complete_on_time is None else bool(complete_on_time))
                )

        completed_on_time = sum(on_time_by_task[task["id"]] is True for task in tasks)
        on_time_known = [value for value in on_time_by_task.values() if value is not None]
        critical_known = [value for value in critical_outcomes if value is not None]
        total_demand = sum(float(task["demand_quantity"]) for task in tasks if task["demand_quantity"] is not None)
        unmet_demand = sum(float(task["unmet_demand_quantity"]) for task in tasks if task["unmet_demand_quantity"] is not None)

        def metric(value: Any, numerator: float | int | None, denominator: float | int,
                   *, valid: bool, unit: str | None = None) -> dict:
            result = {
                "value": value if valid else None, "valid": bool(valid),
                "numerator": numerator if valid else None, "denominator": denominator,
            }
            if unit is not None:
                result["unit"] = unit
            return result

        task_count = len(tasks)
        completed_count = len(completed)
        delay_values = list(delayed_by_task.values())
        deadline_count = sum(
            type(task.get("deadline")) in {int, float} and math.isfinite(float(task["deadline"]))
            for task in tasks
        )
        deadlines_valid = task_count > 0 and deadline_count == task_count
        delay_total = sum(delay_values)
        task_metric_rows = []
        for task in tasks:
            task_metric_rows.append({
                "task_id": task["id"], "kind": task["kind"], "status": task["status"],
                "progress": float(task["progress"]), "completed": task["id"] in completed_ids,
                "on_time": on_time_by_task[task["id"]], "delay_minutes": delayed_by_task[task["id"]],
                "deadline": task["deadline"], "priority": task["priority"],
                "critical": float(task.get("priority", 0.0)) >= 5.0,
                "demand_quantity": task["demand_quantity"],
                "delivered_quantity": task["delivered_quantity"],
                "unmet_demand_quantity": task["unmet_demand_quantity"],
            })

        reward_values = [row.get("reward") for row in steps]
        utility_valid = all(type(value) in {int, float} and math.isfinite(float(value)) for value in reward_values)
        utility = sum(float(value) for value in reward_values) if utility_valid else None
        reward_terms = [row.get("reward_terms") for row in steps]
        reward_terms_valid = all(
            isinstance(terms, dict)
            and all(type(value) in {int, float} and math.isfinite(float(value)) for value in terms.values())
            for terms in reward_terms
        )
        component_totals = {}
        if reward_terms_valid:
            for terms in reward_terms:
                for name, value in terms.items():
                    component_totals[name] = component_totals.get(name, 0.0) + float(value)
        reward_total_from_terms = sum(component_totals.values()) if reward_terms_valid else None
        if utility_valid and reward_terms_valid and not math.isclose(utility, reward_total_from_terms, rel_tol=1e-9, abs_tol=1e-7):
            utility_valid = False
            utility = None

        energy_values = [platform.get("energy_used") for platform in platforms]
        energy_valid = bool(energy_values) and all(
            type(value) in {int, float} and math.isfinite(float(value)) for value in energy_values
        )
        energy = sum(float(value) for value in energy_values) if energy_valid else None

        risk_components = {"environmental_risk": 0.0, "task_failures": 0.0, "platform_failures": 0.0}
        risk_valid = True
        for event in events:
            kind = event.get("type")
            if kind == "move":
                value = event.get("environmental_risk")
                if type(value) not in {int, float} or not math.isfinite(float(value)):
                    risk_valid = False
                else:
                    risk_components["environmental_risk"] += max(0.0, float(value))
            elif kind == "task_failed":
                risk_components["task_failures"] += 1.0
            elif kind in {"platform_failure", "platform_failed"}:
                risk_components["platform_failures"] += 1.0
        risk_units = sum(risk_components.values()) if risk_valid else None

        violation_reasons = {
            "capability": {"capability_lost", "payload_capacity_lost", "capability_or_locality", "capacity_exceeded"},
            "energy": {"insufficient_energy"},
            "time": set(),
            "communication": {"communication_unreachable"},
        }
        violation_counts = {name: set() for name in violation_reasons}

        def original_task_id(task_id: Any) -> str:
            value = str(task_id)
            return str(self._task_specs.get(value, {}).get("parent_task_id", value))

        for event in events:
            kind = event.get("type")
            if kind == "task_failed":
                reason = event.get("reason")
                if reason in violation_reasons["capability"]:
                    violation_counts["capability"].add((original_task_id(event.get("task_id")), str(reason)))
                elif reason in violation_reasons["energy"]:
                    violation_counts["energy"].add((original_task_id(event.get("task_id")), str(reason)))
            elif kind == "dispatch_rejected":
                reason = event.get("reason")
                for name, reasons in violation_reasons.items():
                    if reason in reasons:
                        violation_counts[name].add((original_task_id(event.get("task_id")), str(reason)))
            elif kind == "task_expired":
                violation_counts["time"].add((original_task_id(event.get("task_id", "")), "deadline_expired"))
        for task in tasks:
            if task["status"] != "completed" and task.get("deadline") is not None and truth["time"] >= task["deadline"]:
                violation_counts["time"].add((task["id"], "deadline_expired"))
        dispatch_attempts = sum(len(row.get("operational_dispatches", [])) for row in steps)
        communication_attempts = sum(
            int(event.get("message_count", 0)) for event in events if event.get("type") == "communication_tx"
        )

        platform_loads = {platform["id"]: 0.0 for platform in platforms}
        for task in tasks:
            manifest = self._split_manifests.get(task["id"])
            if manifest is not None:
                for child_id in manifest["child_ids"]:
                    child = self._base._tasks[child_id]
                    owner = child.get("assignee")
                    if owner not in platform_loads or child.get("status") in {"completed", "failed", "cancelled"}:
                        continue
                    child_spec = self._task_specs[child_id]
                    total = float(child_spec.get("service_minutes", child_spec.get("load_minutes", 1) + child_spec.get("unload_minutes", 1)))
                    progress = min(1.0, max(0.0, float(child.get("service_progress", 0.0)) / max(1.0, total)))
                    platform_loads[owner] += float(child_spec["workload_share"]) * (1.0 - progress)
                continue
            if task["status"] in {"completed", "failed", "cancelled"}:
                continue
            if manifest is None:
                owner = task.get("assignee")
                if owner in platform_loads:
                    platform_loads[owner] += max(0.0, 1.0 - float(task["progress"]))
        load_values = list(platform_loads.values())
        load_valid = bool(load_values) and all(math.isfinite(value) for value in load_values)
        mean_load = sum(load_values) / len(load_values) if load_values else 0.0
        load_cv = (
            math.sqrt(sum((value - mean_load) ** 2 for value in load_values) / len(load_values)) / mean_load
            if load_values and mean_load > _EPS else 0.0
        )

        metrics = {
            "completion_rate": metric(completed_count / task_count if task_count else None, completed_count, task_count,
                                      valid=task_count > 0, unit="fraction"),
            "on_time_rate": metric(sum(value is True for value in on_time_known) / len(on_time_known) if on_time_known else None,
                                   sum(value is True for value in on_time_known), len(on_time_known),
                                   valid=bool(on_time_known), unit="fraction"),
            "mean_delay_minutes": metric(delay_total / task_count if task_count and deadlines_valid else None,
                                         delay_total, task_count, valid=deadlines_valid, unit="minutes_per_original_task"),
            "unmet_demand": metric(unmet_demand, unmet_demand, total_demand,
                                   valid=any(task["kind"] == "delivery" for task in tasks) and
                                   all(task["demand_quantity"] is not None and task["unmet_demand_quantity"] is not None
                                       for task in tasks if task["kind"] == "delivery"), unit="kg"),
            "unmet_demand_rate": metric(unmet_demand / total_demand if total_demand > _EPS else None,
                                         unmet_demand, total_demand,
                                         valid=total_demand > _EPS, unit="fraction"),
            "critical_failures": metric(sum(value is False for value in critical_known),
                                        sum(value is False for value in critical_known), critical_count,
                                        valid=critical_count > 0 and len(critical_known) == critical_count, unit="tasks"),
            "critical_failure_rate": metric(sum(value is False for value in critical_known) / len(critical_known) if critical_known else None,
                                            sum(value is False for value in critical_known), len(critical_known),
                                            valid=bool(critical_known) and len(critical_known) == critical_count, unit="fraction"),
            "failure_rate": metric(sum(task["status"] == "failed" for task in tasks) / task_count if task_count else None,
                                   sum(task["status"] == "failed" for task in tasks), task_count,
                                   valid=task_count > 0, unit="fraction"),
            "global_utility": metric(utility, utility, len(steps), valid=utility_valid, unit="reward_units"),
            "mean_step_utility": metric(utility / len(steps) if utility_valid and steps else None,
                                         utility, len(steps), valid=utility_valid and bool(steps), unit="reward_units_per_step"),
            "energy_used": metric(energy, energy, len(platforms), valid=energy_valid, unit="synthetic_energy_units"),
            "safety_risk": metric(risk_units, risk_units, task_count, valid=risk_valid, unit="risk_units"),
            "load_balance": metric(load_cv, None, len(platform_loads), valid=load_valid, unit="coefficient_of_variation"),
        }
        for name, identities in violation_counts.items():
            denominator = (
                communication_attempts if name == "communication" else
                deadline_count if name == "time" else dispatch_attempts
            )
            violations = len(identities)
            metric_name = f"{name}_violations"
            metrics[metric_name] = metric(violations, violations, denominator, valid=True, unit="incidents")
            metrics[f"{name}_violation_rate"] = metric(
                violations / denominator if denominator else None, violations, denominator,
                valid=denominator > 0, unit="fraction",
            )

        effective_scope_counts = {scope: 0 for scope in ("LOCAL_REPAIR", "GLOBAL_REPLAN")}
        requested_scope_counts = {scope: 0 for scope in ("LOCAL_REPAIR", "GLOBAL_REPLAN")}
        effective_reconstruction_count = 0
        reconstruction_size = 0
        for row in steps:
            action = row["full_environment_action"]
            scope = action.get("scope")
            if scope in requested_scope_counts and not action.get("noop"):
                requested_scope_counts[scope] += 1
            primitives = row.get("reward_primitives", {})
            effective = int(primitives.get("reconstruction_count", 0)) > 0
            if effective:
                effective_reconstruction_count += 1
                if scope in effective_scope_counts:
                    effective_scope_counts[scope] += 1
            reconstruction_size += int(primitives.get("reconstruction_size", 0))
        effective_scope_ratios = {
            scope: metric(
                count / effective_reconstruction_count if effective_reconstruction_count else None,
                count, effective_reconstruction_count,
                valid=effective_reconstruction_count > 0, unit="fraction",
            )
            for scope, count in effective_scope_counts.items()
        }
        validity_mask = {name: bool(row["valid"]) for name, row in metrics.items()}
        denominators = {name: row["denominator"] for name, row in metrics.items()}
        for scope, row in effective_scope_ratios.items():
            validity_mask[f"effective_scope_ratio_{scope}"] = bool(row["valid"])
            denominators[f"effective_scope_ratio_{scope}"] = row["denominator"]
        communications = communication_attempts
        return {
            "time": truth["time"], "parent_id": self.scenario["parent_id"], "task_count": task_count,
            "known_task_count": sum(task["known"] for task in tasks), "completed_tasks": completed_count,
            "failed_tasks": sum(task["status"] == "failed" for task in tasks),
            "completion_rate": metrics["completion_rate"]["value"],
            "on_time_rate": metrics["on_time_rate"]["value"],
            "mean_delay": metrics["mean_delay_minutes"]["value"],
            "energy_used": metrics["energy_used"]["value"],
            "metrics": metrics, "validity_mask": validity_mask, "denominators": denominators,
            "task_outcomes": task_metric_rows,
            "critical_failures": metrics["critical_failures"]["value"],
            "critical_failure_rate": metrics["critical_failure_rate"]["value"],
            "unmet_demand": metrics["unmet_demand"]["value"],
            "global_utility": metrics["global_utility"]["value"],
            "safety_risk": metrics["safety_risk"]["value"],
            "load_balance": metrics["load_balance"]["value"],
            "outcomes": metrics,
            "constraint_violations": {name: len(items) for name, items in violation_counts.items()},
            "platform_loads": platform_loads,
            "risk_components": risk_components,
            "reward_component_totals": component_totals,
            "reconstruction_count": effective_reconstruction_count,
            "requested_reconstruction_count": sum(requested_scope_counts.values()),
            "effective_reconstruction_count": effective_reconstruction_count,
            "reconstruction_size": reconstruction_size,
            "operational_dispatch_count": dispatch_attempts,
            "explicit_dispatch_count": sum(len(row["full_environment_action"]["dispatches"]) for row in steps),
            "task_switch_count": sum(int(row["reward_primitives"]["switch_count"]) for row in steps),
            "communication_messages": communications,
            "scope_counts": requested_scope_counts,
            "effective_scope_counts": effective_scope_counts,
            "effective_scope_ratios": effective_scope_ratios,
            "terminal": bool(self._base._terminal),
        }

    @property
    def metrics(self) -> dict:
        return self.episode_metrics()

    def export_trajectory(self) -> dict:
        """Return raw action, public-prefix, event, and endpoint truth evidence."""
        raw_events = copy.deepcopy(self._base.events) + copy.deepcopy(self._extension_events)
        raw_events.sort(key=lambda row: (float(row.get("time", 0)), str(row.get("type", "")), str(row.get("event_id", row.get("task_id", "")))))
        return {
            "schema": "event-relief-trajectory/1", "parent_id": self.scenario["parent_id"],
            "split": self.scenario["split"], "event_tape_id": self.event_tape_id,
            "start_time": 0, "end_time": self.time, "steps": copy.deepcopy(self._steps),
            "raw_events": raw_events, "raw_truth": self._raw_truth(), "summary": self.episode_metrics(),
        }

    def export_assignment_handoff(self) -> dict:
        """Export the current public queue plan, subworkload ownership, and constraints."""
        observation = self.public_observation()
        task_by_id = {row["id"]: row for row in observation["tasks"]}
        platform_by_id = {row["id"]: row for row in observation["platforms"]}
        allocations = []
        cooperative_groups: dict[str, list[dict]] = {}
        for task_id in sorted(self._original_task_specs):
            task = task_by_id.get(task_id, {})
            manifest = self._split_manifests.get(task_id)
            if manifest:
                subworkloads = []
                for child_id in manifest["child_ids"]:
                    child = task_by_id.get(child_id, {})
                    spec = self._task_specs[child_id]
                    row = {
                        "task_id": child_id, "owner": child.get("assignee"),
                        "workload_share": float(spec["workload_share"]),
                        "progress": float(child.get("progress", 0.0)),
                        "remaining_workload_share": float(spec["workload_share"]) * max(0.0, 1.0 - float(child.get("progress", 0.0))),
                        "quantity": float(spec["quantity"]) if spec["kind"] == "delivery" else None,
                        "service_minutes": float(spec["service_minutes"]) if spec["kind"] == "inspection" else None,
                        "status": child.get("status"), "phase": child.get("phase"),
                    }
                    subworkloads.append(row)
                    group = child.get("cooperation_group")
                    if group:
                        cooperative_groups.setdefault(str(group), []).append({
                            "task_id": child_id, "owner": child.get("assignee"),
                            "workload_share": float(spec["workload_share"]),
                        })
                owners = sorted({str(row["owner"]) for row in subworkloads if row["owner"] is not None})
                allocations.append({
                    "task_id": task_id, "mode": manifest["mode"], "owners": owners,
                    "status": task.get("status"), "progress": float(task.get("progress", 0.0)),
                    "subworkloads": subworkloads,
                })
            else:
                owner = task.get("assignee")
                queues = self._queues.get(str(owner), []) if owner is not None else []
                allocations.append({
                    "task_id": task_id, "mode": "single_owner", "owners": [] if owner is None else [str(owner)],
                    "status": task.get("status"), "progress": float(task.get("progress", 0.0)),
                    "queue_position": queues.index(task_id) if task_id in queues else None,
                })
        task_constraints = []
        for task_id, task in sorted(task_by_id.items()):
            if task.get("status") in {"completed", "failed", "cancelled", "expired", "split_parent"}:
                continue
            task_constraints.append({
                "task_id": task_id, "deadline": task.get("deadline"),
                "predecessors": list(task.get("predecessors", [])),
                "capability": "inspect" if task.get("kind") == "inspection" else "deliver",
                "quantity": float(task["quantity"]) if task.get("kind") == "delivery" else None,
                "capacity_kg_required": float(task["quantity"]) if task.get("kind") == "delivery" else 0.0,
                "depot_id": task.get("depot_id"), "commodity": task.get("commodity"),
                "cooperation_group": task.get("cooperation_group"),
                "remaining_progress": max(0.0, 1.0 - float(task.get("progress", 0.0))),
            })
        platform_constraints = []
        for platform_id, platform in sorted(platform_by_id.items()):
            platform_constraints.append({
                "platform_id": platform_id, "capabilities": list(platform.get("capabilities", [])),
                "capacity_kg": float(platform.get("capacity_kg", 0.0)),
                "energy_remaining": float(platform.get("energy", 0.0)),
                "energy_used_reported": float(platform.get("energy_used", 0.0)),
                "failed": bool(platform.get("failed")), "busy": bool(platform.get("busy")),
                "active_task_id": platform.get("active_task_id"),
                "connected_to_coordinator": self._connected(observation, _COORDINATOR, platform_id),
            })
        schedule_issues = self._queue_schedule_issues(observation, self._queues)
        return {
            "schema": "event-assignment-handoff/1", "time": self.time,
            "parent_id": self.scenario["parent_id"], "event_tape_id": self.event_tape_id,
            "snapshot_hash": observation["provenance"]["snapshot_hash"],
            "public_prefix_hash": observation["provenance"]["public_prefix_hash"],
            "queues": copy.deepcopy(self._queues), "allocations": allocations,
            "cooperative_groups": [
                {"group_id": group, "members": sorted(members, key=lambda row: row["task_id"])}
                for group, members in sorted(cooperative_groups.items())
            ],
            "remaining_constraints": {
                "horizon": float(observation["horizon"]),
                "platforms": platform_constraints, "tasks": task_constraints,
                "depots": copy.deepcopy(observation["depots"]),
                "schedule_infeasibilities": [
                    {"constraint": code, "task_id": task_id, "resource_id": resource_id}
                    for code, task_id, resource_id in schedule_issues
                ],
            },
        }


def _scope_name(scope: Any, observation: dict) -> str:
    if isinstance(scope, str) and scope in {"NOOP", "LOCAL_REPAIR", "GLOBAL_REPLAN"}:
        return str(scope)
    if isinstance(scope, dict):
        value = scope.get("scope", scope.get("candidate_scope"))
        if value in {"LOCAL_REPAIR", "GLOBAL_REPLAN"}:
            return value
    if isinstance(scope, str) and scope == "local":
        return "LOCAL_REPAIR"
    if isinstance(scope, str) and scope == "global":
        return "GLOBAL_REPLAN"
    return "GLOBAL_REPLAN"


def _action_within_local_domain(env: EventReliefEnv, action: dict, scope: Any) -> bool:
    if action.get("noop"):
        return True
    if not isinstance(scope, dict):
        return False
    affected_tasks = {str(value) for value in scope.get("affected_task_ids", []) if isinstance(value, str)}
    affected_platforms = {str(value) for value in scope.get("affected_platform_ids", []) if isinstance(value, str)}
    if not affected_tasks or not affected_platforms:
        return False
    touched_tasks: set[str] = set()
    touched_platforms: set[str] = set()
    for operation in action.get("queue_operations", []):
        for key in ("task_id", "other_task_id", "before_task_id"):
            task_id = operation.get(key)
            if isinstance(task_id, str):
                parent_id = env._task_specs.get(task_id, {}).get("parent_task_id")
                touched_tasks.add(str(parent_id or task_id))
        for key in ("from_platform", "to_platform"):
            platform_id = operation.get(key)
            if isinstance(platform_id, str):
                touched_platforms.add(platform_id)
    for dispatch in action.get("dispatches", []):
        task_id = dispatch.get("task_id")
        platform_id = dispatch.get("platform_id")
        if isinstance(task_id, str):
            parent_id = env._task_specs.get(task_id, {}).get("parent_task_id")
            touched_tasks.add(str(parent_id or task_id))
        if isinstance(platform_id, str):
            touched_platforms.add(platform_id)
    return bool(touched_tasks) and touched_tasks <= affected_tasks and touched_platforms <= affected_platforms


def _candidate_record(env: EventReliefEnv, action: dict, observation: dict, legal: bool = True,
                      reason: str | None = None) -> dict:
    full_hash = _digest(action)
    return {
        "candidate_key": f"event-action-{full_hash[:24]}", "full_environment_action": copy.deepcopy(action),
        "candidate_scope": "global" if action["scope"] == "GLOBAL_REPLAN" else ("noop" if action["noop"] else "local"),
        "public_prefix_hash": observation["provenance"]["public_prefix_hash"],
        "snapshot_hash": observation["provenance"]["snapshot_hash"], "event_tape_id": env.event_tape_id,
        "legal": bool(legal), "infeasibility_reason": reason,
    }


def _action_rank_key(observation: dict, action: dict, allowed: dict[tuple[str, str], dict]) -> tuple:
    """Rank a complete alternative using only its public affected tasks."""
    tasks = {str(row.get("id")): row for row in observation.get("tasks", []) if isinstance(row, dict)}
    platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    rows: list[dict] = []
    touched: list[tuple[str, str | None]] = []
    for dispatch in action.get("dispatches", []):
        pair = (str(dispatch.get("platform_id")), str(dispatch.get("task_id")))
        if pair in allowed:
            rows.append(allowed[pair])
        touched.append((pair[1], pair[0]))
    for operation in action.get("queue_operations", []):
        task_id = operation.get("task_id")
        if isinstance(task_id, str):
            touched.append((task_id, operation.get("to_platform")))
        other_id = operation.get("other_task_id")
        if isinstance(other_id, str):
            touched.append((other_id, operation.get("to_platform")))
    for task_id, platform_id in touched:
        if (platform_id, task_id) in allowed:
            rows.append(allowed[(platform_id, task_id)])
            continue
        task = tasks.get(task_id, {})
        platform = platforms.get(str(platform_id), {})
        if not task or not platform:
            continue
        deadline = min(float(observation.get("horizon", 10**9)), float(task.get("deadline", observation.get("horizon", 10**9))))
        owner = task.get("assignee")
        source = str(owner) if isinstance(owner, str) else _COORDINATOR
        communication = _communication_metrics(observation, source, str(platform_id))
        rows.append({
            "platform_id": platform_id, "task_id": task_id,
            "estimated_completion": deadline, "required_energy": 10**9,
            "id": f"operation:{platform_id}:{task_id}",
            "_rank_distance": _public_route_distance(observation, platform, task),
            "_rank_communication": communication,
        })
    if not rows:
        return (10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, "", "", _digest(action))
    ranked = []
    for row in rows:
        rank = _dispatch_rank_key(observation, row)
        if "_rank_distance" in row:
            rank = (*rank[:4], row["_rank_distance"], *rank[5:])
        ranked.append(rank)
    primary = min(ranked)
    return (*primary, len(action.get("queue_operations", [])), len(action.get("dispatches", [])), _digest(action))


def generate_reconstruction_candidates(env: EventReliefEnv, scope: Any = None,
                                       max_candidates: int = 16) -> list[dict]:
    """Create a deterministic catalog of executable public-information actions."""
    if not 1 <= int(max_candidates) <= _MAX_CANDIDATES:
        raise ValueError("max_candidates must be between 1 and 16")
    observation = env.public_observation()
    scope_name = _scope_name(scope, observation)
    tasks = {row["id"]: row for row in observation["tasks"]}
    platform_by_id = {row["id"]: row for row in observation["platforms"]}
    candidates = [row for row in env.candidate_dispatches() if row.get("type") == "dispatch"]
    allowed = {(row["platform_id"], row["task_id"]): row for row in candidates}
    c1_rows = choose_step_c1_actions(observation, candidates)
    c1_dispatches = []
    for row in c1_rows:
        if not env._dispatch_legality(observation, row, allowed):
            c1_dispatches.append({"platform_id": row["platform_id"], "task_id": row["task_id"]})
    noop = _new_action(noop=True)
    action_pool: list[tuple[str, dict]] = [("base", noop)]
    if c1_dispatches:
        c1_action = _new_action(scope_name, dispatches=c1_dispatches)
        if scope_name != "LOCAL_REPAIR" or _action_within_local_domain(env, c1_action, scope):
            action_pool.append(("base", c1_action))

    local_alternatives: list[dict] = []
    cross_alternatives: list[dict] = []
    workload_alternatives: list[dict] = []
    paired_dispatch_alternatives: list[dict] = []
    queues = observation["queues"]

    def add_queue_alternative(action: dict, bucket: list[dict]) -> None:
        """Keep only complete, public-legal queue mutations.

        Queue mutations are deliberately independent of the dispatch catalog:
        a target platform may be busy with its current task while accepting a
        legal future queue item.  The final action check remains the source of
        truth for schedule, stock, communication, and cooperative constraints.
        """
        if scope_name == "LOCAL_REPAIR" and not _action_within_local_domain(env, action, scope):
            return
        normalized, reasons = env._check_action(action)
        if reasons or normalized != action:
            return
        bucket.append(action)

    ranked_pairs = sorted(allowed, key=lambda pair: _dispatch_rank_key(observation, allowed[pair]))
    for pair in ranked_pairs:
        platform_id, task_id = pair
        row = allowed[pair]
        task = tasks[task_id]
        if env._dispatch_legality(observation, row, allowed):
            continue
        owner = task.get("assignee")
        if owner == platform_id:
            queue = queues.get(platform_id, [])
            if task_id in queue and queue.index(task_id) > 0:
                active_id = platform_by_id.get(platform_id, {}).get("active_task_id")
                anchors = [item for item in queue if item != active_id and item != task_id]
                if not anchors:
                    continue
                action = _new_action(scope_name, queue_operations=[{
                    "op": "insert", "task_id": task_id, "from_platform": platform_id,
                    "to_platform": platform_id, "before_task_id": anchors[0],
                }])
            else:
                continue
            add_queue_alternative(action, local_alternatives)
        else:
            action = _new_action(scope_name, queue_operations=[{
                "op": "transfer" if owner is not None else "insert", "task_id": task_id,
                "from_platform": owner, "to_platform": platform_id,
            }])
            add_queue_alternative(action, cross_alternatives)

    # Extend queue alternatives to busy targets and unassigned tasks.  These
    # rows never become immediate dispatches and therefore cannot interrupt an
    # active task; they only change the committed future queue.
    for task_id, task in sorted(tasks.items()):
        if task.get("status") not in {"pending", "assigned", "released"} or task.get("safety_paused"):
            continue
        source = task.get("assignee")
        if source is not None and (source not in queues or task_id not in queues.get(source, [])):
            continue
        for target, platform in sorted(platform_by_id.items()):
            if platform.get("failed") or not env._public_compatible(platform, task):
                continue
            source_node = str(source) if isinstance(source, str) else _COORDINATOR
            if source is not None and not env._connected(observation, source_node, target):
                continue
            if source == target:
                queue = queues.get(target, [])
                if task_id not in queue:
                    continue
                active_id = platform.get("active_task_id")
                ordered = [item for item in queue if item != active_id]
                if task_id not in ordered or ordered.index(task_id) == 0:
                    continue
                anchor = ordered[0]
                operation = {
                    "op": "insert", "task_id": task_id, "from_platform": target,
                    "to_platform": target, "before_task_id": anchor,
                }
                action = _new_action(scope_name, queue_operations=[operation])
                add_queue_alternative(action, local_alternatives if task.get("local") else cross_alternatives)
                continue
            if (target, task_id) in allowed:
                # The ranked dispatch pair above already generated this
                # transfer; only synthesize pairs omitted because the target
                # was busy in candidate_dispatches().
                continue
            operation = {
                "op": "transfer" if source is not None else "insert", "task_id": task_id,
                "from_platform": source, "to_platform": target,
            }
            action = _new_action(scope_name, queue_operations=[operation])
            add_queue_alternative(action, local_alternatives if task.get("local") else cross_alternatives)

    # A pair of executable task transfers is a real swap, not just a label.
    by_owner: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        owner = task.get("assignee")
        if owner in queues and task_id in queues[owner] and task.get("status") in {"pending", "assigned", "released"}:
            by_owner.setdefault(owner, []).append(task_id)
    owners = sorted(by_owner)
    for index, left_owner in enumerate(owners):
        for right_owner in owners[index + 1:]:
            for left_task in sorted(by_owner[left_owner]):
                for right_task in sorted(by_owner[right_owner]):
                    left_scope = tasks[left_task]["local"]
                    action = _new_action(scope_name, queue_operations=[{
                        "op": "swap", "task_id": left_task, "other_task_id": right_task,
                        "from_platform": left_owner, "to_platform": right_owner,
                    }])
                    add_queue_alternative(action, local_alternatives if left_scope else cross_alternatives)
                    # Keep enumerating; ranking and quotas are applied after
                    # every publicly feasible alternative is known.

    groups: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        group = task.get("cooperation_group")
        if group:
            groups.setdefault(str(group), []).append(task_id)
    for group, task_ids in sorted(groups.items()):
        if len(task_ids) != 2:
            continue
        rows = []
        for task_id in sorted(task_ids):
            owner = tasks[task_id].get("assignee")
            candidate = allowed.get((owner, task_id))
            if candidate is None or env._dispatch_legality(observation, candidate, allowed):
                rows = []
                break
            rows.append({"platform_id": owner, "task_id": task_id})
        if len(rows) != 2 or rows[0]["platform_id"] == rows[1]["platform_id"]:
            continue
        action = _new_action(scope_name, dispatches=rows)
        if scope_name == "LOCAL_REPAIR" and not _action_within_local_domain(env, action, scope):
            continue
        paired_dispatch_alternatives.append(action)

    for task_id, task in sorted(tasks.items()):
        source = task.get("assignee")
        if source not in queues or task_id not in queues[source] or task.get("status") not in {"pending", "assigned", "released"}:
            continue
        target_order = sorted(
            queues,
            key=lambda platform_id: (
                platform_by_id[platform_id].get("local") != task.get("local"),
                _distance(platform_by_id[platform_id]["position"], task["position"]),
                platform_id,
            ),
        )
        for target in target_order:
            if target == source:
                continue
            target_queue = queues[target]
            anchor = target_queue[0] if target_queue else None
            for op in ("split", "cooperate"):
                operation = {
                    "op": op, "task_id": task_id, "from_platform": source,
                    "to_platform": target, "share": 0.5,
                }
                if op == "cooperate" and task.get("kind") not in {"inspection", "delivery"}:
                    continue
                if env._operation_legality(observation, operation, set()):
                    continue
                if anchor is not None:
                    operation["before_task_id"] = anchor
                action = _new_action(scope_name, queue_operations=[operation])
                if scope_name == "LOCAL_REPAIR" and not _action_within_local_domain(env, action, scope):
                    continue
                workload_alternatives.append(action)

    # Classify all alternatives before applying the seven-per-domain quota.
    # This prevents lexical enumeration order from hiding urgent public tasks.
    for action in paired_dispatch_alternatives:
        task_ids = [row.get("task_id") for row in action.get("dispatches", [])]
        target = local_alternatives if task_ids and all(tasks.get(task_id, {}).get("local") for task_id in task_ids) else cross_alternatives
        target.append(action)
    for action in workload_alternatives:
        operation = action.get("queue_operations", [{}])[0]
        task_id = operation.get("task_id")
        target = local_alternatives if tasks.get(task_id, {}).get("local") else cross_alternatives
        target.append(action)
    local_alternatives = sorted(local_alternatives, key=lambda action: _action_rank_key(observation, action, allowed))[:7]
    cross_alternatives = sorted(cross_alternatives, key=lambda action: _action_rank_key(observation, action, allowed))[:7]
    action_pool.extend(("local", action) for action in local_alternatives)
    action_pool.extend(("cross", action) for action in cross_alternatives)
    records = []
    seen = set()
    rejected = []
    for _, action in action_pool:
        key = _digest(action)
        if key in seen:
            continue
        seen.add(key)
        normalized, reasons = env._check_action(action)
        if reasons or normalized != action:
            rejected.append({"candidate_key": f"event-action-{key[:24]}", "full_environment_action": action,
                             "legal": False, "infeasibility_reason": sorted(set(reasons))})
            continue
        records.append(_candidate_record(env, action, observation))
        if len(records) >= int(max_candidates):
            break
    env.last_candidate_rejections = rejected
    return records[:int(max_candidates)]


def repair_action(env: EventReliefEnv, action: dict) -> tuple[dict, dict]:
    """Deterministically remove publicly illegal operations and dispatches."""
    before, initial_issues = env._check_action(action)
    if not isinstance(action, dict):
        repaired = _new_action(noop=True)
        issues = initial_issues
    else:
        repaired = copy.deepcopy(before)
        issues = list(initial_issues)
        # Invalid issue markers from the complete check can be redundant; a second pass certifies the result.
        if issues:
            observation = env.public_observation()
            dispatch_catalog = {
                (row["platform_id"], row["task_id"]): row
                for row in env.candidate_dispatches() if row.get("type") == "dispatch"
            }
            clean_dispatches = []
            used_platforms, used_tasks = set(), set()
            for dispatch in action.get("dispatches", []) if isinstance(action.get("dispatches", []), list) else []:
                if not isinstance(dispatch, dict):
                    continue
                pair = (dispatch.get("platform_id"), dispatch.get("task_id"))
                if pair[0] in used_platforms or pair[1] in used_tasks or env._dispatch_legality(observation, dispatch, dispatch_catalog):
                    continue
                used_platforms.add(pair[0]); used_tasks.add(pair[1])
                clean_dispatches.append({"platform_id": pair[0], "task_id": pair[1]})
            clean_ops = []
            for operation in action.get("queue_operations", []) if isinstance(action.get("queue_operations", []), list) else []:
                if isinstance(operation, dict) and not env._operation_legality(observation, operation, set()):
                    clean_ops.append(copy.deepcopy(operation))
            scope = action.get("scope") if action.get("scope") in {"LOCAL_REPAIR", "GLOBAL_REPLAN"} else "LOCAL_REPAIR"
            repaired = _new_action(scope, queue_operations=clean_ops, dispatches=clean_dispatches)
            if not clean_ops and not clean_dispatches:
                repaired = _new_action(noop=True)
    _, post_issues = env._check_action(repaired)
    receipt = {
        "schema": "event-action-repair/1", "legal_before": not bool(initial_issues),
        "issues_before": sorted(set(issues)), "repaired": repaired != action,
        "postrepair_legal": not bool(post_issues), "postrepair_issues": sorted(set(post_issues)),
        "public_snapshot_hash": env.public_observation()["provenance"]["snapshot_hash"],
        "event_tape_id": env.event_tape_id,
    }
    return repaired, receipt
