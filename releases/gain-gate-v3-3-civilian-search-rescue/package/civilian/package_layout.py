"""Canonical path contract for the N2 civilian package root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from protocol_contract import PACKAGE_RELEASE_ID


CANONICAL_PACKAGE_ROOT = Path(__file__).resolve().parent
SMOKE_FIXTURE_NAMES = frozenset({
    "combined.json",
    "control.json",
    "energy_change.json",
    "external_commitment.json",
    "failure.json",
    "report_delay.json",
    "stock_loss.json",
})


@dataclass(frozen=True, slots=True)
class PackageLayout:
    root: Path
    runtime_spec: Path
    prior_resource_cost: Path
    registered_data: Path
    registry: Path


def resolve_package_layout(package_root: str | Path) -> PackageLayout:
    requested = Path(package_root)
    if not requested.is_absolute():
        raise PermissionError("package_root must be an absolute path to package/civilian")
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise PermissionError("package_root does not exist") from exc
    if root != CANONICAL_PACKAGE_ROOT:
        raise PermissionError(
            "canonical package_root is the civilian source/data root; pass the absolute "
            f"package/civilian path, not the outer package directory: {root}"
        )

    runtime_spec = root / "runtime-spec.json"
    prior_resource_cost = root / "PRIOR_RESOURCE_COST.json"
    registered_data = root / "registered_data"
    registry = registered_data / "input_registry.json"
    required_files = (runtime_spec, prior_resource_cost, registry)
    if any(path.is_symlink() or not path.is_file() for path in required_files):
        raise PermissionError("canonical package_root is missing a regular bound runtime, cost, or registry file")
    if registered_data.is_symlink() or not registered_data.is_dir():
        raise PermissionError("canonical package_root is missing its registered_data directory")
    fixtures = root / "fixtures"
    if fixtures.is_symlink() or not fixtures.is_dir():
        raise PermissionError("canonical package_root is missing its smoke fixtures directory")
    fixture_entries = tuple(fixtures.iterdir())
    if (
        {path.name for path in fixture_entries} != SMOKE_FIXTURE_NAMES
        or any(path.is_symlink() or not path.is_file() for path in fixture_entries)
    ):
        raise PermissionError("canonical package_root must contain the exact seven regular smoke fixtures")
    return PackageLayout(root, runtime_spec, prior_resource_cost, registered_data, registry)


def require_package_root(package_root: str | Path) -> Path:
    return resolve_package_layout(package_root).root


def require_registered_data_root(path: str | Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        raise PermissionError("registered_data root must be an absolute path under canonical package_root")
    layout = resolve_package_layout(CANONICAL_PACKAGE_ROOT)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PermissionError("registered_data root does not exist") from exc
    if resolved != layout.registered_data:
        raise PermissionError("registered_data must be read from canonical package_root/registered_data")
    return resolved


def require_registered_data_files_root(path: str | Path) -> Path:
    """Resolve the canonical data directory for read-only registry validation."""
    requested = Path(path)
    if not requested.is_absolute():
        raise PermissionError("registered_data root must be an absolute path under canonical package_root")
    canonical_data = CANONICAL_PACKAGE_ROOT / "registered_data"
    if canonical_data.is_symlink() or not canonical_data.is_dir():
        raise PermissionError("canonical package_root is missing its registered_data directory")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PermissionError("registered_data root does not exist") from exc
    if resolved != canonical_data.resolve(strict=True):
        raise PermissionError("registered_data must be read from canonical package_root/registered_data")
    registry = resolved / "input_registry.json"
    if registry.is_symlink() or not registry.is_file():
        raise PermissionError("registered_data is missing its regular input registry")
    return resolved


def require_prior_resource_cost_path(path: str | Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        raise PermissionError("prior resource-cost path must be absolute and bound to canonical package_root")
    layout = resolve_package_layout(CANONICAL_PACKAGE_ROOT)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PermissionError("prior resource-cost file does not exist") from exc
    if resolved != layout.prior_resource_cost:
        raise PermissionError("prior resource-cost file must be read from canonical package_root")
    return resolved
