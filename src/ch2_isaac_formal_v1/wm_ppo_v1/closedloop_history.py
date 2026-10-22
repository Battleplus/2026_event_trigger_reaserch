"""Bounded live WM history assembled from public runner rows."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .contracts import PublicHistoryFrame
from .public_dynamics import public_projection_to_frame
from .raw_public_windows import PUBLIC_EVENT_TYPES, _tick_frame
try:
    from ..closedloop_v1.decision import WMHistoryUnavailable
except ImportError:  # pragma: no cover
    from closedloop_v1.decision import WMHistoryUnavailable


class LivePublicHistory:
    def __init__(self, actor_ids: Sequence[str], max_frames: int = 7):
        self.actor_ids = frozenset(map(str, actor_ids))
        self.frames: deque[PublicHistoryFrame] = deque(maxlen=max_frames)
        self.activated: set[str] = set()
        self.completed: set[str] = set()

    def append_post_step(self, tick: int, rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
        row_ids = [str(row.get("aircraft_id", "")) for row in rows]
        if len(row_ids) != len(set(row_ids)) or set(row_ids) != self.actor_ids:
            self.frames.clear()
            self.activated.clear()
            self.completed.clear()
            return
        newly_visible = []
        for row in events:
            name = row.get("event")
            if name not in PUBLIC_EVENT_TYPES:
                continue
            try:
                event_time = int(row["sim_time_s"])
                available = int(row.get("available_at_sim_time_s", event_time))
            except (KeyError, TypeError, ValueError):
                continue
            if max(event_time, available) > tick:
                continue
            item = {key: row[key] for key in ("event", "sim_time_s", "available_at_sim_time_s", "aircraft_id", "task_id", "decision_id") if key in row}
            newly_visible.append(item)
            if name == "task_activated" and row.get("task_id") is not None:
                self.activated.add(str(row["task_id"]))
            elif name == "search_cell_complete" and row.get("task_id") is not None:
                self.completed.add(str(row["task_id"]))
        previous = self.frames[-1].time_s if self.frames else None
        if previous is not None and previous != tick - 1:
            self.frames.clear()
        frame = _tick_frame(tick, rows, {}, self.activated, self.completed, newly_visible)
        self.frames.append(frame)

    def decision_history(self, projection: Mapping[str, Any], tick: int, ordinary_events: Sequence[str]) -> tuple[PublicHistoryFrame, ...]:
        if not self.frames or self.frames[-1].time_s != tick - 1:
            raise WMHistoryUnavailable("WM_HISTORY_UNAVAILABLE")
        allowed = {key: projection[key] for key in ("sim_time_s", "aircraft", "public_observations") if key in projection}
        base = public_projection_to_frame(allowed)
        public_events = tuple({"event": name, "sim_time_s": tick, "available_at_sim_time_s": tick}
                              for name in sorted(set(ordinary_events)) if name in {"task_activated", "public_observation", "search_cell_complete", "energy_anomaly"})
        frame = replace(base, public_events=public_events, action=None)
        return tuple(self.frames) + (frame,)
