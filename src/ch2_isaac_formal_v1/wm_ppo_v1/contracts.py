"""Small, serialisable contracts shared by the conditional model stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class SixMethod(str, Enum):
    """The six preregistered closed-loop comparisons."""

    A_EVENT_G = "A_event_G"
    B_LIGHT_G = "B_light_gain_G"
    C_EVENT_G_WM = "C_event_G_WM"
    D_LIGHT_G_WM = "D_light_gain_G_WM"
    E_WM_GAIN_G_WM = "E_wm_gain_G_WM"
    F_PERIODIC_G_WM = "F_periodic_G_WM"


@dataclass(frozen=True)
class PublicHistoryFrame:
    """Public observation at one time step.

    ``platforms``, ``tasks`` and ``observations`` contain only public values.
    Hidden targets, future tapes, snapshot hashes and scenario identifiers must
    remain in the surrounding receipt and are never copied into features.
    """

    time_s: float
    platforms: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tasks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    observations: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    event: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OneStepTarget:
    """Targets for one transition from a public history window.

    ``dynamic`` is a one-step target at ``t+1``.  It is intentionally separate
    from any 30-minute scalar endpoint.  The endpoint gain is optional and is
    used only as a supervised gain head target when an independently reduced
    paired label is available.
    """

    dynamic: Mapping[str, float]
    gain: float | None = None
    valid: bool = True
    reason: str | None = None
    dynamic_mask: Mapping[str, bool] | None = None


@dataclass(frozen=True)
class DynamicGraphBatch:
    """Padded graph sequences and masks produced from public history."""

    node_features: Any
    node_mask: Any
    sequence_mask: Any
    dynamic_target: Any | None = None
    gain_target: Any | None = None
    dynamic_target_mask: Any | None = None
    metadata: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class CheckpointRestoreReceipt:
    """Machine-readable evidence for a model restore comparison."""

    status: str
    model_kind: str
    checkpoint_path: str
    before_digest: str | None
    after_digest: str | None
    predictions_equal: bool
    device: str
    reason: str | None = None
