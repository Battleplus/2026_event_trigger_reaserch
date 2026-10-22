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
)
FEATURE_DIM = len(FEATURE_NAMES)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _node(kind: str, item: Mapping[str, Any]) -> np.ndarray:
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    row[{"platform": 0, "task": 1, "observation": 2}[kind]] = 1.0
    row[3] = np.clip(_number(item.get("x_norm", item.get("x"))), -1.0, 1.0)
    row[4] = np.clip(_number(item.get("y_norm", item.get("y"))), -1.0, 1.0)
    row[5] = np.clip(_number(item.get("z_norm", item.get("z"))), -1.0, 1.0)
    row[6] = max(0.0, _number(item.get("speed_norm", item.get("speed"))))
    row[7] = np.clip(_number(item.get("energy_fraction", item.get("energy", 0.0))), 0.0, 1.0)
    row[8] = float(bool(item.get("available", True)))
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
    return row


def public_history_to_graph(frame: PublicHistoryFrame | Mapping[str, Any]) -> np.ndarray:
    """Return ``[num_dynamic_nodes, FEATURE_DIM]`` public features."""

    if not isinstance(frame, PublicHistoryFrame):
        frame = PublicHistoryFrame(
            time_s=_number(frame.get("time_s")),
            platforms=tuple(frame.get("platforms", ())),
            tasks=tuple(frame.get("tasks", ())),
            observations=tuple(frame.get("observations", ())),
            event=dict(frame.get("event", {})),
        )
    rows = [_node("platform", x) for x in frame.platforms]
    rows.extend(_node("task", x) for x in frame.tasks)
    rows.extend(_node("observation", x) for x in frame.observations)
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def collate_graph_sequences(
    histories: Sequence[Sequence[PublicHistoryFrame | Mapping[str, Any]]],
    targets: Sequence[OneStepTarget | None] | None = None,
) -> DynamicGraphBatch:
    """Pad variable-length graph sequences and retain a node/step mask."""

    if not histories:
        raise ValueError("histories must contain at least one sequence")
    graphs = [[public_history_to_graph(frame) for frame in seq] for seq in histories]
    max_steps = max(len(seq) for seq in graphs)
    max_nodes = max((graph.shape[0] for seq in graphs for graph in seq), default=0)
    features = np.zeros((len(graphs), max_steps, max_nodes, FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros((len(graphs), max_steps, max_nodes), dtype=bool)
    sequence_mask = np.zeros((len(graphs), max_steps), dtype=bool)
    for batch_index, seq in enumerate(graphs):
        for step_index, graph in enumerate(seq):
            count = graph.shape[0]
            features[batch_index, step_index, :count] = graph
            node_mask[batch_index, step_index, :count] = True
            sequence_mask[batch_index, step_index] = True

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
                             gain_target=gain_target, dynamic_target_mask=dynamic_target_mask)
