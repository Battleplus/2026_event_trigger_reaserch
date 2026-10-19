"""Single source of truth for the N02 release and seven-stage identity."""

from __future__ import annotations

PACKAGE_RELEASE_ID = "ch2_disaster_relief_n02_minimal_closure_v1_100x100_repair1"
PROTOCOL_ID = "n02-minimal-closure-v1-100x100-repair1"
STAGE_PROTOCOL_ID = f"{PROTOCOL_ID}/seven-stage"
N2_STAGE_ORDER = (
    "input_admission",
    "paired_consequence_collection",
    "world_model_training",
    "checkpoint_restore",
    "graph_policy_training",
    "development_evaluation",
    "settlement_and_export",
)


def validate_protocol_contract(
    *,
    package_release: str,
    protocol_id: str,
    stage_protocol: str,
    stage_order: tuple[str, ...] | list[str],
) -> None:
    if (
        package_release != PACKAGE_RELEASE_ID
        or protocol_id != PROTOCOL_ID
        or stage_protocol != STAGE_PROTOCOL_ID
        or tuple(stage_order) != N2_STAGE_ORDER
    ):
        raise ValueError("N02_PROTOCOL_CONTRACT_MISMATCH")
