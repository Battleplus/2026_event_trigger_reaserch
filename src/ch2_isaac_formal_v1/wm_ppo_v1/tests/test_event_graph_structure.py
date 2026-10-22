from __future__ import annotations

import copy
import unittest

import numpy as np

from wm_ppo_v1.features import EDGE_GEOGRAPHIC, collate_graph_sequences
from wm_ppo_v1.public_dynamics import public_projection_to_frame
from wm_ppo_v1.world_model import EventAwareWorldModel, require_torch


def _compact_tick():
    return {
        "sim_time_s": 5,
        "hidden_targets": [{"id": "secret", "xy": [999, 999]}],
        "aircraft": {
            "a0": {"kind": "uav", "home_zone": 0, "available": True, "faulted": False,
                   "safety_paused": False, "communication_held": False,
                   "position_m": [0, 0, 0], "velocity_mps": [0, 0, 0],
                   "available_energy_wh": 20, "active_task_ids": [], "queue_task_ids": []},
            "a1": {"kind": "uav", "home_zone": 8, "available": True, "faulted": False,
                   "safety_paused": False, "communication_held": False,
                   "position_m": [0, 0, 0], "velocity_mps": [0, 0, 0],
                   "available_energy_wh": 20, "active_task_ids": [], "queue_task_ids": []},
        },
        "public_tasks": {
            "unassigned": {"zone": 0, "cell": 0, "kind": "search", "allowed_kind": "uav",
                "public_xy_km": [1.0, 1.0], "public_bounds_km": [0, 2, 0, 2],
                "assignment": None, "queue_position": None, "active": False, "completed": False},
            "same-zone": {"zone": 0, "cell": 1, "kind": "search", "allowed_kind": "uav",
                "public_xy_km": [3.0, 1.0], "assignment": "a0", "queue_position": 0,
                "active": True, "completed": False},
            "unknown-zone": {"zone": None, "cell": 2, "kind": "search", "allowed_kind": "uav",
                "public_xy_km": None, "assignment": None, "queue_position": None,
                "active": False, "completed": False},
        },
        "public_observations": [],
        "applied_action": {"applied": True, "magnitude": 0.2, "duration_norm": 0.1},
    }


class EventGraphStructureTests(unittest.TestCase):
    def test_compact_contract_visibility_graph_and_action_paths(self):
        torch = require_torch()
        projection = _compact_tick()
        events = [
            {"event": "platform_fault", "sim_time_s": 4, "available_at_sim_time_s": 5},
            {"event": "search_cell_complete", "sim_time_s": 4, "available_at_sim_time_s": 6},
            {"event": "search_cell_complete", "sim_time_s": 6, "available_at_sim_time_s": 6},
        ]
        frame = public_projection_to_frame(projection)
        self.assertEqual(len(frame.tasks), 3)
        self.assertIsNone(frame.task_owners[0])
        self.assertEqual(frame.tasks[0]["task_public_data_known"], True)
        sparse = copy.deepcopy(projection)
        del sparse["public_tasks"]
        sparse["aircraft"]["a0"]["queue_task_ids"] = ["legacy-task"]
        sparse_frame = public_projection_to_frame(sparse)
        self.assertEqual(len(sparse_frame.tasks), 1)
        self.assertFalse(sparse_frame.tasks[0]["task_public_data_known"])
        clean = copy.deepcopy(projection)
        clean["hidden_targets"] = [{"id": "other", "xy": [-100, -100]}]
        np.testing.assert_array_equal(
            collate_graph_sequences([[frame]]).node_features,
            collate_graph_sequences([[public_projection_to_frame(clean)]]).node_features)
        frame = type(frame)(time_s=frame.time_s, platforms=frame.platforms, tasks=frame.tasks,
            observations=frame.observations, node_keys=frame.node_keys, task_owners=frame.task_owners,
            action=frame.action, public_events=tuple(events))
        batch = collate_graph_sequences([[frame]])
        edges = batch.edge_index[0, 0][:, batch.edge_mask[0, 0]].T
        edge_types = batch.edge_type[0, 0, batch.edge_mask[0, 0]]
        task_offset = len(frame.platforms)
        # Unknown-zone task is deliberately excluded from all geographic edges.
        unknown_task_index = next(i for i, task in enumerate(frame.tasks) if task["_public_id"] == "unknown-zone")
        same_task_index = next(i for i, task in enumerate(frame.tasks) if task["_public_id"] == "same-zone")
        self.assertFalse(any(edge_types[i] == EDGE_GEOGRAPHIC and task_offset + unknown_task_index in edges[i]
                             for i in range(len(edges))))
        # Same-zone task and platform have a relation.
        self.assertTrue(any(edge_types[i] == EDGE_GEOGRAPHIC and task_offset + same_task_index in edges[i]
                            and 0 in edges[i] for i in range(len(edges))))
        self.assertEqual(batch.event_features[0, 0, 1], 0.1)
        self.assertEqual(batch.event_features[0, 0, 0], 0.0)
        altered_events = (events[0], {"event": "platform_fault", "sim_time_s": 3,
                          "available_at_sim_time_s": 10})
        altered_frame = type(frame)(time_s=frame.time_s, platforms=frame.platforms,
            tasks=frame.tasks, observations=frame.observations, node_keys=frame.node_keys,
            task_owners=frame.task_owners, action=frame.action, public_events=altered_events)
        np.testing.assert_array_equal(batch.event_features,
            collate_graph_sequences([[altered_frame]]).event_features)

        # Add padding: even an explicitly supplied edge to the padded node is masked out.
        padded_features = torch.zeros((1, 1, batch.node_features.shape[2] + 1, batch.node_features.shape[3]))
        padded_features[:, :, :-1] = torch.tensor(batch.node_features)
        padded_mask = torch.zeros((1, 1, padded_features.shape[2]), dtype=torch.bool)
        padded_mask[:, :, :-1] = torch.tensor(batch.node_mask)
        padded_edge_index = torch.tensor(batch.edge_index)
        padded_edge_index = torch.cat((padded_edge_index, torch.tensor([[[[0], [padded_features.shape[2] - 1]]]])), dim=-1)
        padded_edge_mask = torch.cat((torch.tensor(batch.edge_mask), torch.ones((1, 1, 1), dtype=torch.bool)), dim=-1)
        padded_edge_type = torch.cat((torch.tensor(batch.edge_type), torch.ones((1, 1, 1), dtype=torch.long)), dim=-1)
        model = EventAwareWorldModel(batch.node_features.shape[-1], hidden_dim=8)
        base_args = [padded_features, padded_mask, torch.tensor(batch.sequence_mask)]
        common = dict(edge_index=padded_edge_index, edge_mask=padded_edge_mask,
            edge_type=padded_edge_type, event_features=torch.tensor(batch.event_features))
        with_edges = model(*base_args, **common, action_features=torch.tensor(batch.action_features))
        baseline_edge_mask = torch.cat((torch.tensor(batch.edge_mask), torch.zeros((1, 1, 1), dtype=torch.bool)), dim=-1)
        baseline = model(*base_args, **{**common, "edge_mask": baseline_edge_mask},
            action_features=torch.tensor(batch.action_features))
        torch.testing.assert_close(with_edges["dynamic"], baseline["dynamic"])
        torch.testing.assert_close(with_edges["gain"], baseline["gain"])
        changed_action = torch.tensor(batch.action_features)
        changed_action[:, :, 2] = 0.95
        changed = model(*base_args, **common, action_features=changed_action)
        torch.testing.assert_close(with_edges["gain"], changed["gain"])
        self.assertFalse(torch.allclose(with_edges["dynamic"], changed["dynamic"]))


if __name__ == "__main__":
    unittest.main()
