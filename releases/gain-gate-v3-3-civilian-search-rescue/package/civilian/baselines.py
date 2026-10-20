"""Deterministic, non-learning public-information baselines."""

from __future__ import annotations

from typing import Any


_PRE_LOAD_PHASES = {"assigned", "to_depot", "loading"}


def public_stock_commitments(observation: dict) -> dict[tuple[str, str], float]:
    """Return visible delivery stock already committed before cargo is loaded."""

    committed: dict[tuple[str, str], float] = {}
    for task in observation.get("tasks", []):
        if (
            not isinstance(task, dict)
            or task.get("kind") != "delivery"
            or task.get("status") not in {"active", "assigned"}
            or task.get("phase") not in _PRE_LOAD_PHASES
        ):
            continue
        key = (str(task.get("depot_id")), str(task.get("commodity")))
        committed[key] = committed.get(key, 0.0) + float(task.get("quantity", 0.0))
    return committed


def _task_key(candidate: dict, task_by_id: dict[str, dict]) -> tuple:
    task = task_by_id.get(candidate.get("task_id"), {})
    return (-float(task.get("priority", 0)), int(task.get("deadline", 10**9)), str(task.get("id", "")))


def _is_assignable(candidate: dict) -> bool:
    return candidate.get("id") != "NOOP" and candidate.get("task_id") is not None and candidate.get("platform_id") is not None


def _choose_actions(
    policy: str,
    observation: dict,
    candidates: list[dict],
    *,
    reserve_visible_stock: bool,
) -> list[dict]:
    """Choose at most one assignment per platform and task from public data.

    ``observation`` is intentionally the only source of policy information;
    callers should pass the environment's public mirror, never internal state.
    """

    if policy not in {"C0", "C1"}:
        raise ValueError("policy must be C0 or C1")
    task_by_id = {str(task["id"]): task for task in observation.get("tasks", []) if isinstance(task, dict) and task.get("id") is not None}
    actions = [candidate for candidate in candidates if _is_assignable(candidate)]
    if policy == "C0":
        queues = observation.get("chapter1_snapshot", {}).get("queues", {})
        queue_head: dict[str, str] = {}
        for platform_id, queue in queues.items():
            for task_id in queue:
                task = task_by_id.get(str(task_id), {})
                if task.get("status") not in {"completed", "failed"}:
                    queue_head[str(platform_id)] = str(task_id)
                    break
        actions = [candidate for candidate in actions if str(candidate.get("task_id")) == queue_head.get(str(candidate.get("platform_id")))]
        actions.sort(key=lambda item: str(item.get("platform_id")))
    else:
        actions.sort(key=lambda item: (_task_key(item, task_by_id), float(item.get("estimated_completion", 10**9)), str(item.get("platform_id", "")), str(item.get("task_id", ""))))

    selected: list[dict] = []
    used_platforms: set[str] = set()
    used_tasks: set[str] = set()
    reserved_stock = public_stock_commitments(observation) if reserve_visible_stock else {}
    depots = {} if not reserve_visible_stock else {
        str(depot.get("id")): depot
        for depot in observation.get("depots", [])
        if isinstance(depot, dict) and depot.get("id") is not None
    }
    for candidate in actions:
        platform_id = str(candidate.get("platform_id"))
        task_id = str(candidate.get("task_id"))
        if platform_id in used_platforms or task_id in used_tasks:
            continue
        task = task_by_id.get(task_id, {})
        if reserve_visible_stock and task.get("kind") == "delivery":
            depot_id = str(task.get("depot_id"))
            depot = depots.get(depot_id)
            if depot is not None:
                commodity = str(task.get("commodity"))
                key = (depot_id, commodity)
                quantity = float(task.get("quantity", 0.0))
                available = float(depot.get("stock", {}).get(commodity, 0.0))
                if reserved_stock.get(key, 0.0) + quantity > available + 1e-9:
                    continue
                reserved_stock[key] = reserved_stock.get(key, 0.0) + quantity
        selected.append(dict(candidate))
        used_platforms.add(platform_id)
        used_tasks.add(task_id)
    return selected


def choose_actions(policy: str, observation: dict, candidates: list[dict]) -> list[dict]:
    """Choose a legacy C0/C1 assignment without step-batch stock reservation."""

    return _choose_actions(policy, observation, candidates, reserve_visible_stock=False)


def choose_step_c1_actions(observation: dict, candidates: list[dict]) -> list[dict]:
    """Choose a step-mode C1 batch while reserving visible stock within it."""

    return _choose_actions("C1", observation, candidates, reserve_visible_stock=True)


def choose_step_c0_actions(observation: dict, candidates: list[dict]) -> list[dict]:
    """Choose a step-mode C0 batch while reserving visible stock within it."""

    return _choose_actions("C0", observation, candidates, reserve_visible_stock=True)


def choose_action(policy: str, observation: dict, candidates: list[dict]) -> dict | None:
    """Compatibility helper returning the first selected action."""

    selected = choose_actions(policy, observation, candidates)
    return selected[0] if selected else None


class C0:
    """Stateless adapter exposing the C0 rule."""

    name = "C0"

    def __call__(self, observation: dict, candidates: list[dict]) -> list[dict]:
        return choose_actions(self.name, observation, candidates)


class C1:
    """Stateless adapter exposing the C1 public-priority rule."""

    name = "C1"

    def __call__(self, observation: dict, candidates: list[dict]) -> list[dict]:
        return choose_actions(self.name, observation, candidates)

