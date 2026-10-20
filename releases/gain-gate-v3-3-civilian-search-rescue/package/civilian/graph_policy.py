"""Dynamic-entity graph policy for the copied disaster-relief simulator."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

try:
    from .baselines import public_stock_commitments
    from .scenario_schema import MAP_CONTRACT, MAP_COLS, MAP_ROWS, GRID_CELLS, validate_map_contract
except ImportError:
    from baselines import public_stock_commitments
    from scenario_schema import MAP_CONTRACT, MAP_COLS, MAP_ROWS, GRID_CELLS, validate_map_contract

try:
    from .literal_graph_ops import SELF_EDGE, UAV_TASK_EDGE, PaperFaithfulModuleSet
except ImportError:  # Allows direct execution from the package root.
    from literal_graph_ops import SELF_EDGE, UAV_TASK_EDGE, PaperFaithfulModuleSet


HIDDEN_DIM = 64
NODE_FEATURE_DIM = 60
PAIR_FEATURE_DIM = 16
STRUCTURAL_ZERO_NODE_COLUMNS = (21, 56, 57, 58, 59)
PHASES = (
    "pending", "undiscovered", "report_received", "released", "assigned",
    "to_inspection", "observe", "inspection_completed", "to_depot", "loading",
    "in_transit", "unloading", "delivery_confirmed", "failed",
)
STATUSES = ("undiscovered", "pending", "assigned", "active", "completed", "failed", "expired")
COMMODITIES = ("water", "food", "medical", "other")


@dataclass(frozen=True)
class EncodedGraph:
    node_features: torch.Tensor
    node_types: torch.Tensor
    node_mask: torch.Tensor
    platform_mask: torch.Tensor
    task_mask: torch.Tensor
    edge_types: torch.Tensor
    edge_features: torch.Tensor
    candidate_ids: tuple[str, ...]
    candidate_pairs: tuple[tuple[int, int] | None, ...]
    candidate_stock_requirements: tuple[tuple[str, str, float] | None, ...]
    stock_limits: tuple[tuple[str, str, float], ...]
    platform_slots: int
    task_slots: int


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _travel_ticks(distance: float, speed: float) -> float:
    if distance <= 1e-9:
        return 0.0
    return float(math.ceil(distance / speed - 1e-9))


def _one_hot(value: str, levels: Sequence[str]) -> list[float]:
    return [1.0 if value == level else 0.0 for level in levels]


def _map_extent(observation: dict) -> float:
    map_data = observation["map"]
    validate_map_contract(map_data)
    return max(float(map_data["rows"]) * float(map_data["cell_km"]), float(map_data["cols"]) * float(map_data["cell_km"]), 1.0)


def _platform_features(platform: dict, active_task: dict | None, observation: dict, extent: float) -> list[float]:
    x, y = map(float, platform["position"])
    caps = set(platform.get("capabilities", []))
    stage = "" if active_task is None else str(active_task.get("phase", ""))
    values = [
        1.0, 0.0,
        float(platform.get("type") == "uav"), float(platform.get("type") == "helicopter"),
        0.0, 0.0,
        x / extent, y / extent, float(bool(platform.get("local"))),
        float(bool(platform.get("failed"))), float(bool(platform.get("busy"))),
        float(platform.get("energy", 0.0)) / 100.0,
        float(platform.get("capacity_kg", 0.0)) / 100.0,
        float(platform.get("speed_km_min", 0.0)) / 2.0,
        float(platform.get("energy_per_km", 0.0)) / 10.0,
        float(platform.get("service_energy_per_min", 0.0)) / 10.0,
        float("inspect" in caps), float("deliver" in caps),
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        *[0.0] * len(PHASES),
        0.0, 0.0, 0.0, 0.0,
        float(stage == "to_inspection"), float(stage == "observe"),
        float(stage == "to_depot"), float(stage in {"loading", "in_transit", "unloading"}),
    ]
    return values


def _task_features(task: dict, observation: dict, extent: float) -> list[float]:
    x, y = map(float, task["position"])
    horizon = max(1.0, float(observation["horizon"]))
    now = float(observation["time"])
    received = task.get("report_received_at")
    generated = task.get("report_generated_at")
    observed_report_delay = 0.0
    if received is not None and generated is not None:
        observed_report_delay = max(0.0, float(received) - float(generated)) / horizon
    quantity = float(task.get("quantity") or 0.0)
    service = float(task.get("service_minutes") or 0.0)
    load = float(task.get("load_minutes") or 0.0)
    unload = float(task.get("unload_minutes") or 0.0)
    commodity = task.get("commodity") or "other"
    return [
        0.0, 1.0,
        0.0, 0.0,
        float(task.get("kind") == "inspection"), float(task.get("kind") == "delivery"),
        x / extent, y / extent, float(bool(task.get("local"))),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        max(0.0, float(task.get("deadline", now)) - now) / horizon,
        float(task.get("priority", 0.0)) / 5.0,
        quantity / 20.0,
        0.0,
        service / horizon, load / horizon, unload / horizon,
        float(generated is not None), float(received is not None), observed_report_delay,
        float(bool(task.get("discovered", True))),
        *_one_hot(str(task.get("status", "pending")), STATUSES),
        *_one_hot(str(task.get("phase", task.get("status", "pending"))), PHASES),
        *_one_hot(str(commodity), COMMODITIES),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]


def _mission_estimate(platform: dict, task: dict, depots: dict[str, dict], now: float) -> tuple[float, float, float, float]:
    speed = max(float(platform["speed_km_min"]), 1e-9)
    energy_per_km = float(platform.get("energy_per_km", 0.0))
    service_energy = float(platform.get("service_energy_per_min", 0.0))
    if task["kind"] == "inspection":
        distance = _distance(platform["position"], task["position"])
        service = float(task["service_minutes"])
        completion = now + _travel_ticks(distance, speed) + service
        required = distance * energy_per_km + service * service_energy
        return completion, required, 0.0, 0.0
    depot = depots[task["depot_id"]]
    to_depot = _distance(platform["position"], depot["position"])
    to_task = _distance(depot["position"], task["position"])
    load, unload = float(task["load_minutes"]), float(task["unload_minutes"])
    completion = now + _travel_ticks(to_depot, speed) + load + _travel_ticks(to_task, speed) + unload
    required = (to_depot + to_task) * energy_per_km + (load + unload) * service_energy
    return completion, required, to_depot, to_task


def _pair_features(platform: dict, task: dict, observation: dict, depots: dict[str, dict], candidate: dict | None, extent: float) -> list[float]:
    now = float(observation["time"])
    horizon = max(1.0, float(observation["horizon"]))
    completion, required, to_depot, to_task = _mission_estimate(platform, task, depots, now)
    if candidate is not None:
        completion = float(candidate["estimated_completion"])
        required = float(candidate["required_energy"])
    depot = depots.get(task.get("depot_id"), {})
    depot_stock = float(depot.get("stock", {}).get(task.get("commodity"), 0.0))
    quantity = float(task.get("quantity") or 0.0)
    distance = _distance(platform["position"], task["position"])
    if task["kind"] == "inspection":
        to_depot = 0.0
        to_task = 0.0
    stock_ratio = 0.0 if quantity <= 1e-9 else min(2.0, depot_stock / quantity) / 2.0
    capacity = max(1.0, float(platform.get("capacity_kg", 0.0)))
    return [
        (float(task["position"][0]) - float(platform["position"][0])) / extent,
        (float(task["position"][1]) - float(platform["position"][1])) / extent,
        distance / extent,
        max(0.0, completion - now) / horizon,
        required / max(1.0, float(platform.get("energy", 0.0))),
        float(bool(candidate and candidate.get("locked"))),
        float(task.get("kind") == "delivery"),
        float(task.get("kind") == "inspection"),
        float(task.get("priority", 0.0)) / 5.0,
        max(0.0, float(task.get("deadline", now)) - now) / horizon,
        min(2.0, quantity / capacity) / 2.0,
        stock_ratio,
        to_task / extent,
        to_depot / extent,
        float(bool(platform.get("local")) == bool(task.get("local"))),
        max(0.0, float(platform.get("energy", 0.0)) - required) / max(1.0, float(platform.get("energy", 0.0))),
    ]


def encode_public_graph(policy_input: dict, *, pad_to: tuple[int, int] | None = None) -> EncodedGraph:
    """Encode explicit public entities, types, masks and stable candidate IDs."""

    if not isinstance(policy_input, dict) or not isinstance(policy_input.get("observation"), dict):
        raise ValueError("policy input must contain public 'observation' and 'candidates'")
    observation = policy_input["observation"]
    validate_map_contract(observation.get("map"))
    candidates = policy_input.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("policy input candidates must be a non-empty catalog")
    platforms = sorted(observation.get("platforms", []), key=lambda item: str(item["id"]))
    tasks = sorted(observation.get("tasks", []), key=lambda item: str(item["id"]))
    platform_index = {item["id"]: index for index, item in enumerate(platforms)}
    task_index = {item["id"]: index for index, item in enumerate(tasks)}
    if pad_to is None:
        platform_slots, task_slots = len(platforms), len(tasks)
    else:
        platform_slots, task_slots = map(int, pad_to)
        if platform_slots < len(platforms) or task_slots < len(tasks):
            raise ValueError("padding dimensions cannot be smaller than the entity counts")
    extent = _map_extent(observation)
    tasks_by_id = {item["id"]: item for item in tasks}
    platforms_by_id = {item["id"]: item for item in platforms}
    active_tasks = {item["id"]: item for item in tasks}
    node_features: list[list[float]] = []
    node_types: list[int] = []
    platform_mask: list[bool] = []
    task_mask: list[bool] = []
    for platform in platforms:
        active = active_tasks.get(platform.get("active_task_id"))
        node_features.append(_platform_features(platform, active, observation, extent))
        node_types.append(1)
        platform_mask.append(True)
        task_mask.append(False)
    for _ in range(platform_slots - len(platforms)):
        node_features.append([0.0] * NODE_FEATURE_DIM)
        node_types.append(0)
        platform_mask.append(False)
        task_mask.append(False)
    for task in tasks:
        node_features.append(_task_features(task, observation, extent))
        node_types.append(2)
        platform_mask.append(False)
        task_mask.append(True)
    for _ in range(task_slots - len(tasks)):
        node_features.append([0.0] * NODE_FEATURE_DIM)
        node_types.append(0)
        platform_mask.append(False)
        task_mask.append(False)

    node_count = platform_slots + task_slots
    edge_types = torch.zeros((node_count, node_count), dtype=torch.long)
    for index, valid in enumerate(platform_mask):
        if valid:
            edge_types[index, index] = SELF_EDGE
    for index, valid in enumerate(task_mask[platform_slots:platform_slots + task_slots], start=platform_slots):
        if valid:
            edge_types[index, index] = SELF_EDGE
    depots = {item["id"]: item for item in observation.get("depots", [])}
    stock_limits: dict[tuple[str, str], float] = {}
    committed_stock = public_stock_commitments(observation)
    for depot_id, depot in depots.items():
        for commodity, amount in depot.get("stock", {}).items():
            amount = float(amount)
            if not math.isfinite(amount) or amount < 0.0:
                raise ValueError("public depot stock must be finite and non-negative")
            key = (str(depot_id), str(commodity))
            stock_limits[key] = max(0.0, amount - committed_stock.get(key, 0.0))
    edge_features = torch.zeros((node_count, node_count, PAIR_FEATURE_DIM), dtype=torch.float32)
    pair_candidate: dict[tuple[int, int], dict] = {}
    ids: list[str] = []
    pairs: list[tuple[int, int] | None] = []
    stock_requirements: list[tuple[str, str, float] | None] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or type(candidate.get("id")) is not str:
            raise ValueError("candidate catalog contains an invalid candidate")
        candidate_id = candidate["id"]
        ids.append(candidate_id)
        if candidate_id == "NOOP":
            if candidate.get("type") != "noop":
                raise ValueError("NOOP candidate type mismatch")
            pairs.append(None)
            stock_requirements.append(None)
            continue
        platform_id, task_id = candidate.get("platform_id"), candidate.get("task_id")
        if candidate.get("type") != "dispatch" or platform_id not in platform_index or task_id not in task_index:
            raise ValueError(f"candidate does not map to visible entities: {candidate_id!r}")
        p_i, t_i = platform_index[platform_id], task_index[task_id]
        if (p_i, t_i) in pair_candidate:
            raise ValueError("candidate catalog contains a duplicate entity pair")
        pair_candidate[(p_i, t_i)] = candidate
        pairs.append((p_i, t_i))
        task = tasks_by_id[task_id]
        if task.get("kind") == "delivery":
            depot_id = task.get("depot_id")
            commodity = task.get("commodity")
            quantity = float(task.get("quantity", 0.0))
            if type(depot_id) is not str or type(commodity) is not str or not math.isfinite(quantity) or quantity < 0.0:
                raise ValueError("public delivery task has an invalid stock requirement")
            if depot_id not in depots:
                raise ValueError("public delivery task refers to a depot absent from the observation")
            stock_limits.setdefault((depot_id, commodity), 0.0)
            stock_requirements.append((depot_id, commodity, quantity))
        else:
            stock_requirements.append(None)
    if len(set(ids)) != len(ids) or ids.count("NOOP") != 1:
        raise ValueError("candidate catalog must have unique IDs and one NOOP")

    for p_i, platform in enumerate(platforms):
        for t_i, task in enumerate(tasks):
            candidate = pair_candidate.get((p_i, t_i))
            features = _pair_features(platform, task, observation, depots, candidate, extent)
            t_node = platform_slots + t_i
            edge_types[p_i, t_node] = UAV_TASK_EDGE
            edge_types[t_node, p_i] = UAV_TASK_EDGE
            edge_features[p_i, t_node] = torch.tensor(features, dtype=torch.float32)
            edge_features[t_node, p_i] = torch.tensor(features, dtype=torch.float32)

    padded_features = []
    for features in node_features:
        if len(features) > NODE_FEATURE_DIM:
            raise ValueError(f"node feature schema overflow: {len(features)} > {NODE_FEATURE_DIM}")
        padded_features.append(features + [0.0] * (NODE_FEATURE_DIM - len(features)))
    node_tensor = torch.tensor(padded_features, dtype=torch.float32).reshape(node_count, NODE_FEATURE_DIM)
    node_type_tensor = torch.tensor(node_types, dtype=torch.long)
    node_mask_tensor = torch.tensor([p or t for p, t in zip(platform_mask, task_mask)], dtype=torch.bool)
    platform_mask_tensor = torch.tensor(platform_mask, dtype=torch.bool)
    task_mask_tensor = torch.tensor(task_mask, dtype=torch.bool)
    return EncodedGraph(
        node_features=node_tensor,
        node_types=node_type_tensor,
        node_mask=node_mask_tensor,
        platform_mask=platform_mask_tensor,
        task_mask=task_mask_tensor,
        edge_types=edge_types,
        edge_features=edge_features,
        candidate_ids=tuple(ids),
        candidate_pairs=tuple(pairs),
        candidate_stock_requirements=tuple(stock_requirements),
        stock_limits=tuple((depot_id, commodity, quantity) for (depot_id, commodity), quantity in sorted(stock_limits.items())),
        platform_slots=platform_slots,
        task_slots=task_slots,
    )


def _linear_parameters(input_dim: int, output_dim: int, bias: bool = True) -> int:
    return input_dim * output_dim + (output_dim if bias else 0)


def _common_parameter_count() -> int:
    count = (
        _linear_parameters(NODE_FEATURE_DIM, HIDDEN_DIM)
        + _linear_parameters(HIDDEN_DIM, HIDDEN_DIM)
        + 2 * HIDDEN_DIM
        + _linear_parameters(2 * HIDDEN_DIM + PAIR_FEATURE_DIM, HIDDEN_DIM)
        + _linear_parameters(HIDDEN_DIM, 1)
        + _linear_parameters(HIDDEN_DIM, 1)
        + _linear_parameters(HIDDEN_DIM, HIDDEN_DIM)
        + _linear_parameters(HIDDEN_DIM, 1)
    )
    return count - HIDDEN_DIM * len(STRUCTURAL_ZERO_NODE_COLUMNS)


def _literal_parameter_count() -> int:
    attention = (
        _linear_parameters(HIDDEN_DIM, HIDDEN_DIM, bias=False) * 2
        + _linear_parameters(HIDDEN_DIM + PAIR_FEATURE_DIM, HIDDEN_DIM, bias=False)
        + 2 * HIDDEN_DIM
        + _linear_parameters(2 * HIDDEN_DIM, HIDDEN_DIM)
        + _linear_parameters(HIDDEN_DIM, 1)
    )
    relation_mlp = 2 * _linear_parameters(HIDDEN_DIM, HIDDEN_DIM)
    task_update = 4 * relation_mlp + _linear_parameters(4 * HIDDEN_DIM, HIDDEN_DIM) + _linear_parameters(HIDDEN_DIM, HIDDEN_DIM)
    # The current public environment exposes no task predecessor/successor
    # edges, so the first relation projections always receive an all-zero sum.
    frozen_relation_weights = 2 * HIDDEN_DIM * HIDDEN_DIM
    return attention + task_update - frozen_relation_weights


def _entity_adapter_parameters(width: int, blocks: int) -> int:
    return _linear_parameters(HIDDEN_DIM, width) + blocks * _linear_parameters(width, width) + _linear_parameters(width, HIDDEN_DIM)


def _l0_architecture() -> tuple[int, int, int, int, float]:
    target = _common_parameter_count() + _literal_parameter_count()
    best: tuple[int, int, int, int, float] | None = None
    for width in range(80, 161):
        for blocks in range(1, 9):
            count = _common_parameter_count() + _entity_adapter_parameters(width, blocks)
            relative = abs(count - target) / target
            choice = (width, blocks, count, target, relative)
            if best is None or (relative, width, blocks) < (best[4], best[0], best[1]):
                best = choice
    assert best is not None and best[4] <= 0.05
    return best


def _tensor_digest(module: nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.split(".", 1)[0] not in prefixes:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GraphPolicy(nn.Module):
    """Shared-encoder actor-critic with Literal message passing for L1."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in {"L0", "L1"}:
            raise ValueError("variant must be L0 or L1")
        self.variant = variant
        # The local constructor preserves the reviewed module topology and
        # initialization order while keeping deployment independent of src/.
        source = PaperFaithfulModuleSet(NODE_FEATURE_DIM, PAIR_FEATURE_DIM, HIDDEN_DIM)
        self.node_encoder = source.node_encoder
        self.pair_actor = source.pair_actor
        self.noop_actor = source.noop_actor
        self.critic = source.critic
        self.node_type_embedding = nn.Embedding(2, HIDDEN_DIM)
        if variant == "L1":
            self.literal_attention = source.literal_attention
            self.task_update = source.task_update
            # No public task-predecessor or task-successor relation is encoded.
            # Freeze only the two first-layer matrices whose inputs are therefore
            # structurally zero; their relation biases remain active.
            self.task_update.predecessor_mlp[0].weight.requires_grad_(False)
            self.task_update.successor_mlp[0].weight.requires_grad_(False)
        else:
            width, blocks, _, _, _ = _l0_architecture()
            layers: list[nn.Module] = [nn.Linear(HIDDEN_DIM, width), nn.ELU()]
            for _ in range(blocks):
                layers.extend((nn.Linear(width, width), nn.ELU()))
            layers.append(nn.Linear(width, HIDDEN_DIM))
            self.entity_mlp = nn.Sequential(*layers)

    def _graph_outputs(self, graph: EncodedGraph) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        nodes = graph.node_features.to(device)
        node_mask = graph.node_mask.to(device)
        platform_mask = graph.platform_mask.to(device)
        task_mask = graph.task_mask.to(device)
        edge_types = graph.edge_types.to(device)
        edge_features = graph.edge_features.to(device)
        node_types = graph.node_types.to(device)
        type_indices = node_types.clamp(min=1) - 1
        encoded = self.node_encoder(nodes) + self.node_type_embedding(type_indices)
        if self.variant == "L0":
            encoded = encoded + self.entity_mlp(encoded)
        else:
            original_tasks = encoded[graph.platform_slots:]
            if graph.platform_slots > 0 and encoded.shape[0] > 0:
                with_batch = self.literal_attention(
                    encoded.unsqueeze(0), edge_types.unsqueeze(0), edge_features.unsqueeze(0), graph.platform_slots
                ).squeeze(0)
                encoded = torch.cat((with_batch[:graph.platform_slots], original_tasks), dim=0)
            if graph.task_slots > 0:
                encoded = self.task_update(
                    encoded.unsqueeze(0), edge_types.unsqueeze(0), graph.platform_slots, graph.task_slots
                ).squeeze(0)
        encoded = encoded * node_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = torch.sum(encoded, dim=0) / node_mask.to(encoded.dtype).sum().clamp_min(1.0)
        pair_scores: list[torch.Tensor] = []
        for pair in graph.candidate_pairs:
            if pair is None:
                continue
            p_i, t_i = pair
            task_node = graph.platform_slots + t_i
            pair_input = torch.cat((encoded[p_i], encoded[task_node], edge_features[p_i, task_node]), dim=-1)
            pair_scores.append(self.pair_actor(pair_input).squeeze(-1))
        pair_iter = iter(pair_scores)
        candidate_logits = []
        noop_logit = self.noop_actor(pooled).squeeze(-1)
        for pair in graph.candidate_pairs:
            candidate_logits.append(noop_logit if pair is None else next(pair_iter))
        logits = torch.stack(candidate_logits)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    def forward(self, policy_input: dict) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
        graph = encode_public_graph(policy_input)
        logits, value = self._graph_outputs(graph)
        return graph.candidate_ids, logits, value

    def value(self, policy_input: dict) -> torch.Tensor:
        return self.forward(policy_input)[2]

    @staticmethod
    def _mask_at_step(
        candidate_ids: tuple[str, ...],
        pairs: tuple[tuple[int, int] | None, ...],
        stock_requirements: tuple[tuple[str, str, float] | None, ...],
        stock_limits: dict[tuple[str, str], float],
        used_platforms: set[int],
        used_tasks: set[int],
        reserved_stock: dict[tuple[str, str], float],
    ) -> list[bool]:
        mask = []
        for candidate_id, pair, requirement in zip(candidate_ids, pairs, stock_requirements):
            if candidate_id == "NOOP":
                mask.append(True)
            elif pair is None:
                mask.append(False)
            else:
                entities_available = pair[0] not in used_platforms and pair[1] not in used_tasks
                stock_available = True
                if requirement is not None:
                    depot_id, commodity, quantity = requirement
                    key = (depot_id, commodity)
                    stock_available = reserved_stock.get(key, 0.0) + quantity <= stock_limits.get(key, 0.0) + 1e-9
                mask.append(entities_available and stock_available)
        return mask

    @staticmethod
    def _distribution(logits: torch.Tensor, valid_mask: list[bool]) -> Categorical:
        mask = torch.tensor(valid_mask, dtype=torch.bool, device=logits.device)
        if not bool(mask.any()):
            raise ValueError("a conditional action mask must retain at least one candidate")
        return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))

    def sample_batch(self, policy_input: dict, deterministic: bool = False) -> dict[str, Any]:
        graph = encode_public_graph(policy_input)
        logits, value = self._graph_outputs(graph)
        candidate_indices = {candidate_id: index for index, candidate_id in enumerate(graph.candidate_ids)}
        used_platforms: set[int] = set()
        used_tasks: set[int] = set()
        stock_limits = {(depot_id, commodity): quantity for depot_id, commodity, quantity in graph.stock_limits}
        reserved_stock: dict[tuple[str, str], float] = {}
        actions: list[str] = []
        masks: list[dict[str, Any]] = []
        conditional_log_probs: list[torch.Tensor] = []
        conditional_entropies: list[torch.Tensor] = []
        max_assignments = min(int(graph.platform_mask.sum()), int(graph.task_mask.sum()))

        while True:
            valid_mask = self._mask_at_step(
                graph.candidate_ids, graph.candidate_pairs, graph.candidate_stock_requirements,
                stock_limits, used_platforms, used_tasks, reserved_stock,
            )
            dispatch_available = any(valid and candidate_id != "NOOP" for candidate_id, valid in zip(graph.candidate_ids, valid_mask))
            if not dispatch_available:
                if not masks:
                    distribution = self._distribution(logits, valid_mask)
                    noop_index = candidate_indices["NOOP"]
                    masks.append({"candidate_ids": list(graph.candidate_ids), "valid_mask": valid_mask, "selected_id": "NOOP"})
                    conditional_log_probs.append(distribution.log_prob(torch.tensor(noop_index, device=logits.device)))
                    conditional_entropies.append(distribution.entropy())
                break
            distribution = self._distribution(logits, valid_mask)
            chosen_index = torch.argmax(distribution.logits) if deterministic else distribution.sample()
            chosen_id = graph.candidate_ids[int(chosen_index)]
            masks.append({"candidate_ids": list(graph.candidate_ids), "valid_mask": valid_mask, "selected_id": chosen_id})
            conditional_log_probs.append(distribution.log_prob(chosen_index))
            conditional_entropies.append(distribution.entropy())
            if chosen_id == "NOOP":
                break
            actions.append(chosen_id)
            pair = graph.candidate_pairs[int(chosen_index)]
            assert pair is not None
            used_platforms.add(pair[0])
            used_tasks.add(pair[1])
            requirement = graph.candidate_stock_requirements[int(chosen_index)]
            if requirement is not None:
                depot_id, commodity, quantity = requirement
                key = (depot_id, commodity)
                reserved_stock[key] = reserved_stock.get(key, 0.0) + quantity
            if len(actions) >= max_assignments:
                break

        zero = logits.sum() * 0.0
        return {
            "actions": actions,
            "masks": masks,
            "conditional_log_probs": conditional_log_probs,
            "log_prob": torch.stack(conditional_log_probs).sum() if conditional_log_probs else zero,
            "entropy": torch.stack(conditional_entropies).sum() if conditional_entropies else zero,
            "value": value,
        }

    def evaluate_batch(self, policy_input: dict, actions: Sequence[str], masks: Sequence[dict[str, Any]]) -> dict[str, Any]:
        graph = encode_public_graph(policy_input)
        logits, value = self._graph_outputs(graph)
        used_platforms: set[int] = set()
        used_tasks: set[int] = set()
        stock_limits = {(depot_id, commodity): quantity for depot_id, commodity, quantity in graph.stock_limits}
        reserved_stock: dict[tuple[str, str], float] = {}
        selected_actions: list[str] = []
        conditional_log_probs: list[torch.Tensor] = []
        conditional_entropies: list[torch.Tensor] = []
        saw_noop = False
        for record in masks:
            if saw_noop:
                raise ValueError("conditional records cannot follow NOOP")
            if tuple(record.get("candidate_ids", ())) != graph.candidate_ids:
                raise ValueError("stored conditional candidate order does not match the public catalog")
            current_mask = self._mask_at_step(
                graph.candidate_ids, graph.candidate_pairs, graph.candidate_stock_requirements,
                stock_limits, used_platforms, used_tasks, reserved_stock,
            )
            stored_mask = record.get("valid_mask")
            if not isinstance(stored_mask, list) or stored_mask != current_mask:
                raise ValueError("stored conditional mask does not match the residual entity constraints")
            selected_id = record.get("selected_id")
            try:
                selected_index = graph.candidate_ids.index(selected_id)
            except ValueError as exc:
                raise ValueError("stored selection is absent from its candidate catalog") from exc
            if not current_mask[selected_index]:
                raise ValueError("stored selection is masked")
            distribution = self._distribution(logits, current_mask)
            selected_tensor = torch.tensor(selected_index, device=logits.device)
            conditional_log_probs.append(distribution.log_prob(selected_tensor))
            conditional_entropies.append(distribution.entropy())
            if selected_id == "NOOP":
                saw_noop = True
                continue
            selected_actions.append(selected_id)
            pair = graph.candidate_pairs[selected_index]
            assert pair is not None
            used_platforms.add(pair[0])
            used_tasks.add(pair[1])
            requirement = graph.candidate_stock_requirements[selected_index]
            if requirement is not None:
                depot_id, commodity, quantity = requirement
                key = (depot_id, commodity)
                reserved_stock[key] = reserved_stock.get(key, 0.0) + quantity
        if list(actions) != selected_actions:
            raise ValueError("stored dispatch action IDs disagree with conditional selections")
        if not masks:
            raise ValueError("a PPO transition must retain at least one conditional selection")
        if not saw_noop and any(
            valid and candidate_id != "NOOP"
            for candidate_id, valid in zip(
                graph.candidate_ids,
                self._mask_at_step(
                    graph.candidate_ids, graph.candidate_pairs, graph.candidate_stock_requirements,
                    stock_limits, used_platforms, used_tasks, reserved_stock,
                ),
            )
        ):
            raise ValueError("batch stopped while an unmasked dispatch candidate remained")
        zero = logits.sum() * 0.0
        return {
            "conditional_log_probs": conditional_log_probs,
            "log_prob": torch.stack(conditional_log_probs).sum() if conditional_log_probs else zero,
            "entropy": torch.stack(conditional_entropies).sum() if conditional_entropies else zero,
            "value": value,
        }

    def active_parameter_count(self) -> int:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        zero_feature_weights = HIDDEN_DIM * len(STRUCTURAL_ZERO_NODE_COLUMNS)
        return trainable - zero_feature_weights

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_policy(variant: str, seed: int, device: str | torch.device = "cpu") -> GraphPolicy:
    """Build one freshly initialized policy and attach static architecture evidence."""

    if variant not in {"L0", "L1"}:
        raise ValueError("variant must be L0 or L1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        policy = GraphPolicy(variant)
    common = ("node_encoder", "node_type_embedding", "pair_actor", "noop_actor", "critic")
    policy.to(device)
    policy.seed = int(seed)
    policy.common_init_sha256 = _tensor_digest(policy, common)
    if variant == "L0":
        width, blocks, active, target, difference = _l0_architecture()
        policy.architecture_evidence = {
            "variant": variant,
            "map_contract": MAP_CONTRACT,
            "map_rows": MAP_ROWS,
            "map_cols": MAP_COLS,
            "grid_cells": GRID_CELLS,
            "hidden_dim": HIDDEN_DIM,
            "entity_mlp_width": width,
            "entity_mlp_blocks": blocks,
            "active_parameters": policy.active_parameter_count(),
            "trainable_parameters": policy.trainable_parameter_count(),
            "structurally_zero_node_feature_weights": HIDDEN_DIM * len(STRUCTURAL_ZERO_NODE_COLUMNS),
            "target_L1_parameters": target,
            "relative_parameter_difference": difference,
            "common_init_sha256": policy.common_init_sha256,
        }
    else:
        policy.architecture_evidence = {
            "variant": variant,
            "map_contract": MAP_CONTRACT,
            "map_rows": MAP_ROWS,
            "map_cols": MAP_COLS,
            "grid_cells": GRID_CELLS,
            "hidden_dim": HIDDEN_DIM,
            "active_parameters": policy.active_parameter_count(),
            "trainable_parameters": policy.trainable_parameter_count(),
            "structurally_inactive_parameters": {
                "task_relation_input_weights": 2 * HIDDEN_DIM * HIDDEN_DIM,
                "node_feature_weights": HIDDEN_DIM * len(STRUCTURAL_ZERO_NODE_COLUMNS),
            },
            "common_init_sha256": policy.common_init_sha256,
        }
    return policy


def build_paired_policies(seed: int, device: str | torch.device = "cpu") -> tuple[GraphPolicy, GraphPolicy, dict[str, Any]]:
    """Build paired L0/L1 policies and verify common modules match exactly."""

    l0 = build_policy("L0", seed, device)
    l1 = build_policy("L1", seed, device)
    if l0.common_init_sha256 != l1.common_init_sha256:
        raise AssertionError("paired common modules did not receive identical initialization")
    evidence = {
        "seed": int(seed),
        "common_init_sha256": l0.common_init_sha256,
        "L0": dict(l0.architecture_evidence),
        "L1": dict(l1.architecture_evidence),
        "parameter_match_within_5_percent": l0.architecture_evidence["relative_parameter_difference"] <= 0.05,
    }
    return l0, l1, evidence
