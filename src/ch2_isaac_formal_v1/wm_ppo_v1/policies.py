"""Fixed method definitions and incremental cost accounting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .contracts import SixMethod


@dataclass(frozen=True)
class MethodSpec:
    name: SixMethod
    trigger: str
    gate: str
    replanner: str


def method_specs() -> tuple[MethodSpec, ...]:
    return (
        MethodSpec(SixMethod.A_EVENT_G, "event", "none", "G"),
        MethodSpec(SixMethod.B_LIGHT_G, "event", "light_gain", "G"),
        MethodSpec(SixMethod.C_EVENT_G_WM, "event", "none", "G-WM"),
        MethodSpec(SixMethod.D_LIGHT_G_WM, "event", "light_gain", "G-WM"),
        MethodSpec(SixMethod.E_WM_GAIN_G_WM, "event", "wm_gain", "G-WM"),
        MethodSpec(SixMethod.F_PERIODIC_G_WM, "periodic_4min", "none", "G-WM"),
    )


@dataclass(frozen=True)
class CostLedger:
    """Raw quantities used by the frozen incremental cost formula."""

    messages_increment: float = 0.0
    switches_increment: float = 0.0
    planner_calls_increment: float = 0.0
    changed_actors: float = 0.0
    planning_cpu_seconds_increment: float = 0.0
    cpu_inference_seconds: float = 0.0
    gpu_inference_seconds: float = 0.0
    transfer_seconds: float = 0.0
    command_wait_seconds: float = 0.0
    execution_energy_kwh: float = 0.0
    communication_seconds: float = 0.0
    safety_risk_units: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def incremental_cost(ledger: CostLedger, actor_count: int = 36) -> float:
    """Compute only the frozen C_increment terms, without double charging J_exec."""

    if actor_count <= 0:
        raise ValueError("actor_count must be positive")
    return (
        0.01 * (ledger.messages_increment / 1000.0)
        + 0.05 * (ledger.switches_increment / 100.0)
        + 0.05 * (ledger.planner_calls_increment / 100.0)
        + 0.01 * (ledger.changed_actors / actor_count)
        + 0.01 * ledger.planning_cpu_seconds_increment
    )
