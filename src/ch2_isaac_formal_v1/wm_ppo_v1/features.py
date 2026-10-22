"""Public-history to dynamic graph tensors.

The encoder uses stable numeric attributes and aggregate categorical counts.
Identifiers are used only to determine node type and are never emitted as
feature values.  The node set is dynamic: every frame can contain a different
number of platform, task and observation nodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import DynamicGraphBatch, PublicHistoryFrame, OneStepTarget


FEATURE_NAMES = (
    "kind_platform",
    "kind_task",
    "kind_observation",
    "x_norm",
    "y_norm",
    "z_norm",
    "speed_norm",
    "energy_fraction",
    "available",
    "faulted",
    "communication_held",
    "active",
    "priority_norm",
    "age_norm",
    "queue_position_norm",
    "queue_length_norm",
    "active_task_count_norm",
    "energy_log_norm",
    "platform_uav",
    "platform_helicopter",
    "home_zone_norm",
    "energy_fraction_known",
    "queue_data_known",
    "availability_delay_norm",
    "priority_known",
    "task_public_data_known",
    "completed",
    "availability_known",
)
FEATURE_DIM = len(FEATURE_NAMES)
EVENT_NAMES = ("search_cell_complete", "platform_fault", "communication_safety_hold",
               "energy_safety_hold", "safety_violation", "boundary_violation",
               "constraint_violation", "transmission_sent", "command_applied",
               "task_activated", "public_observation", "energy_anomaly")
EVENT_SCHEMA_VERSION = "public_events_v2"
EVENT_DIM = len(EVENT_NAMES)
ACTION_DIM = 5  # known, applied, normalized magnitude, normalized duration, held
EDGE_PLATFORM_TASK = 1
EDGE_GEOGRAPHIC = 2


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _maybe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _node(kind: str, item: Mapping[str, Any]) -> np.ndarray:
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    row[{"platform": 0, "task": 1, "observation": 2}[kind]] = 1.0
    row[3] = np.clip(_number(item.get("x_norm", item.get("x"))), -1.0, 1.0)
    row[4] = np.clip(_number(item.get("y_norm", item.get("y"))), -1.0, 1.0)
    row[5] = np.clip(_number(item.get("z_norm", item.get("z"))), -1.0, 1.0)
    row[6] = max(0.0, _number(item.get("speed_norm", item.get("speed"))))
    row[7] = np.clip(_number(item.get("energy_fraction", item.get("energy", 0.0))), 0.0, 1.0)
    row[8] = float(bool(item.get("available", False)))
    row[9] = float(bool(item.get("faulted", False)))
    row[10] = float(bool(item.get("communication_held", False)))
    row[11] = float(bool(item.get("active", True)))
    row[12] = max(0.0, _number(item.get("priority_norm", item.get("priority"))))
    row[13] = max(0.0, _number(item.get("age_norm", item.get("age_s"))))
    row[14] = np.clip(_number(item.get("queue_position_norm")), 0.0, 1.0)
    row[15] = max(0.0, _number(item.get("queue_length_norm")))
    row[16] = max(0.0, _number(item.get("active_task_count_norm")))
    row[17] = max(0.0, _number(item.get("energy_log_norm")))
    row[18] = float(item.get("platform_kind") == "uav")
    row[19] = float(item.get("platform_kind") == "helicopter")
    row[20] = np.clip(_number(item.get("home_zone_norm")), 0.0, 1.0)
    row[21] = float("energy_fraction" in item)
    row[22] = float(bool(item.get("queue_data_known", False)))
    row[23] = max(0.0, _number(item.get("availability_delay_norm")))
    row[24] = float("priority_norm" in item or "priority" in item)
    row[25] = float(bool(item.get("task_public_data_known", False)))
    row[26] = float(bool(item.get("completed", False)))
    row[27] = float(isinstance(item.get("available"), bool))
    return row


def _coerce_frame(frame: PublicHistoryFrame | Mapping[str, Any]) -> PublicHistoryFrame:

    if not isinstance(frame, PublicHistoryFrame):
        frame = PublicHistoryFrame(
            time_s=_number(frame.get("time_s")),
            platforms=tuple(frame.get("platforms", ())),
            tasks=tuple(frame.get("tasks", ())),
            observations=tuple(frame.get("observations", ())),
            event=dict(frame.get("event", {})),
            public_events=tuple(frame.get("public_events", ())),
            action=frame.get("action"),
            node_keys=tuple(frame.get("node_keys", ())),
            task_owners=tuple(frame.get("task_owners", ())),
        )
    return frame


def public_history_to_graph(frame: PublicHistoryFrame | Mapping[str, Any]) -> np.ndarray:
    """Return ``[num_dynamic_nodes, FEATURE_DIM]`` public features."""
    frame = _coerce_frame(frame)
    rows = [_node("platform", x) for x in frame.platforms]
    rows.extend(_node("task", x) for x in frame.tasks)
    rows.extend(_node("observation", x) for x in frame.observations)
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def _frame_graph(frame: PublicHistoryFrame | Mapping[str, Any]):
    frame = _coerce_frame(frame)
    nodes = [*frame.platforms, *frame.tasks, *frame.observations]
    features = public_history_to_graph(frame)
    keys = list(frame.node_keys)
    owners = list(frame.task_owners)
    if len(keys) != len(nodes):
        keys = [None] * len(frame.platforms) + [x.get("_public_id") for x in frame.tasks] + [None] * len(frame.observations)
    if len(owners) != len(nodes):
        owners = [None] * len(frame.platforms) + [x.get("_owner_id") for x in frame.tasks] + [None] * len(frame.observations)
    edges: list[tuple[int, int, int]] = []
    platform_ids = {str(key): i for i, key in enumerate(keys[:len(frame.platforms)]) if key is not None}
    task_offset = len(frame.platforms)
    for j, owner in enumerate(owners[task_offset:task_offset + len(frame.tasks)]):
        if owner is not None and str(owner) in platform_ids:
            p, t = platform_ids[str(owner)], task_offset + j
            edges.extend(((p, t, EDGE_PLATFORM_TASK), (t, p, EDGE_PLATFORM_TASK)))
    # Public zone IDs define same-zone and four-neighbor geographic relations.
    zones = [_maybe_number(x.get("_public_zone")) for x in nodes]
    def adjacent_zone(left: float | None, right: float | None) -> bool:
        if left is None or right is None or not left.is_integer() or not right.is_integer():
            return False
        a, b = int(left), int(right)
        if not (0 <= a < 9 and 0 <= b < 9):
            return False
        ar, ac = divmod(a, 3)
        br, bc = divmod(b, 3)
        return abs(ar - br) + abs(ac - bc) <= 1
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adjacent_zone(zones[i], zones[j]):
                edges.extend(((i, j, EDGE_GEOGRAPHIC), (j, i, EDGE_GEOGRAPHIC)))
    event = np.zeros(EVENT_DIM, dtype=np.float32)
    for item in frame.public_events:
        if not isinstance(item, Mapping):
            continue
        t = _number(item.get("sim_time_s"), float("inf"))
        available_at = _number(item.get("available_at_sim_time_s"), t)
        name = item.get("event")
        if t <= frame.time_s and available_at <= frame.time_s and name in EVENT_NAMES:
            event[EVENT_NAMES.index(name)] += 1.0
    event = np.clip(event, 0.0, 10.0) / 10.0
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    if isinstance(frame.action, Mapping):
        action[:] = (1.0, float(bool(frame.action.get("applied", False))),
                     np.clip(abs(_number(frame.action.get("magnitude"))), 0.0, 1.0),
                     np.clip(_number(frame.action.get("duration_norm")), 0.0, 1.0),
                     float(bool(frame.action.get("held", False))))
    return features, np.asarray(edges, dtype=np.int64).reshape(-1, 3), event, action


def collate_graph_sequences(
    histories: Sequence[Sequence[PublicHistoryFrame | Mapping[str, Any]]],
    targets: Sequence[OneStepTarget | None] | None = None,
) -> DynamicGraphBatch:
    """Pad variable-length graph sequences and retain a node/step mask."""

    if not histories:
        raise ValueError("histories must contain at least one sequence")
    frames = [[_frame_graph(frame) for frame in seq] for seq in histories]
    graphs = [[item[0] for item in seq] for seq in frames]
    max_steps = max(len(seq) for seq in graphs)
    max_nodes = max((graph.shape[0] for seq in graphs for graph in seq), default=0)
    features = np.zeros((len(graphs), max_steps, max_nodes, FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros((len(graphs), max_steps, max_nodes), dtype=bool)
    sequence_mask = np.zeros((len(graphs), max_steps), dtype=bool)
    max_edges = max((item[1].shape[0] for seq in frames for item in seq), default=0)
    edge_index = np.zeros((len(graphs), max_steps, 2, max_edges), dtype=np.int64)
    edge_mask = np.zeros((len(graphs), max_steps, max_edges), dtype=bool)
    edge_type = np.zeros((len(graphs), max_steps, max_edges), dtype=np.int64)
    event_features = np.zeros((len(graphs), max_steps, EVENT_DIM), dtype=np.float32)
    action_features = np.zeros((len(graphs), max_steps, ACTION_DIM), dtype=np.float32)
    for batch_index, seq in enumerate(graphs):
        for step_index, graph in enumerate(seq):
            count = graph.shape[0]
            features[batch_index, step_index, :count] = graph
            node_mask[batch_index, step_index, :count] = True
            sequence_mask[batch_index, step_index] = True
            _, edges, event, action = frames[batch_index][step_index]
            if len(edges):
                edge_index[batch_index, step_index, :, :len(edges)] = edges[:, :2].T
                edge_type[batch_index, step_index, :len(edges)] = edges[:, 2]
                edge_mask[batch_index, step_index, :len(edges)] = True
            event_features[batch_index, step_index] = event
            action_features[batch_index, step_index] = action

    dynamic_target = gain_target = None
    dynamic_target_mask = None
    if targets is not None:
        if len(targets) != len(histories):
            raise ValueError("targets and histories must have equal length")
        dynamic_target = np.zeros((len(targets), 4), dtype=np.float32)
        dynamic_target_mask = np.zeros((len(targets), 4), dtype=bool)
        gain_target = np.full((len(targets),), np.nan, dtype=np.float32)
        names = ("task_progress", "energy_delta", "delay", "risk")
        for index, target in enumerate(targets):
            if target is None or not target.valid:
                continue
            for col, name in enumerate(names):
                is_valid = (target.dynamic_mask is None or bool(target.dynamic_mask.get(name, False)))
                if is_valid and name in target.dynamic:
                    dynamic_target[index, col] = _number(target.dynamic[name])
                    dynamic_target_mask[index, col] = True
            if target.gain is not None:
                gain_target[index] = _number(target.gain)
    return DynamicGraphBatch(node_features=features, node_mask=node_mask,
                             sequence_mask=sequence_mask, dynamic_target=dynamic_target,
                             gain_target=gain_target, dynamic_target_mask=dynamic_target_mask,
                             edge_index=edge_index, edge_mask=edge_mask, edge_type=edge_type,
                             event_features=event_features, action_features=action_features)
