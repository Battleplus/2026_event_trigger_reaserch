"""V3 responsibility-region and support rules for the 100 x 100 map."""

from __future__ import annotations

import math
from typing import Any


REGION_BOUNDARIES = (0, 33, 66, 100)
REGION_CONTRACT = "relief-regions-3x3-v1"
VALID_REGION_IDS = frozenset(f"r{row}{col}" for row in range(3) for col in range(3))


def region_index(position: Any) -> tuple[int, int]:
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        raise ValueError("REGION_POSITION_SHAPE")
    x, y = float(position[0]), float(position[1])
    if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < 100 and 0 <= y < 100):
        raise ValueError("REGION_POSITION_OUT_OF_BOUNDS")
    col = 0 if x < 33 else 1 if x < 66 else 2
    row = 0 if y < 33 else 1 if y < 66 else 2
    return row, col


def region_id(position: Any) -> str:
    row, col = region_index(position)
    return f"r{row}{col}"


def region_neighbors(left: str, right: str) -> bool:
    if left not in VALID_REGION_IDS or right not in VALID_REGION_IDS:
        return False
    try:
        lr, lc = int(left[1]), int(left[2])
        rr, rc = int(right[1]), int(right[2])
    except (TypeError, ValueError, IndexError):
        return False
    return abs(lr - rr) + abs(lc - rc) <= 1


def platform_can_support(platform: dict, task: dict, *, mode: str = "replan") -> bool:
    """Check region assignment eligibility; physical limits remain elsewhere."""
    kind = str(task.get("kind", ""))
    capability = "inspect" if kind == "inspection" else "deliver" if kind == "delivery" else None
    if capability is None or capability not in platform.get("capabilities", []):
        return False
    has_platform_region = "home_region" in platform
    has_task_region = "region_id" in task
    if has_platform_region != has_task_region:
        return False
    if not has_platform_region:
        # V2 compatibility path. V3 scenarios always carry explicit region IDs.
        return bool(platform.get("local")) == bool(task.get("local"))
    home, target = str(platform["home_region"]), str(task["region_id"])
    if home not in VALID_REGION_IDS or target not in VALID_REGION_IDS:
        return False
    if mode == "initial":
        return home == target
    if mode != "replan":
        raise ValueError("REGION_SUPPORT_MODE_INVALID")
    return True
