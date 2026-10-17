#!/usr/bin/env python
"""Prepare a source-only public archive without modifying the frozen work tree.

Preview is the default. Use --apply only after the source freeze and review; it
adds allowlisted source files and redacted evidence projections to an empty
archive checkout. It never copies raw outputs, checkpoints, dependencies, or
temporary directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_FILES = (
    ".gitignore",
    "README.md",
    "ROADMAP.md",
    "STATUS.md",
    "requirements.txt",
    "MANIFEST.md",
    "REPO_AUDIT.md",
    "REUSE_INVENTORY.json",
    "BASELINE_PROVENANCE.json",
    "SOURCE_FILE_INVENTORY.json",
    "SOURCE_BOUNDARY_AUDIT.json",
    "SHA256SUMS.txt",
    "EXPERIMENT_PROTOCOL.md",
    "EXPERIMENT_PLAN.md",
)
CONFIG_FILES = (
    "GAIN_GATE_CONFIG.json",
    "EVENT_TRIGGER_CONFIG.json",
    "ENVIRONMENT_SPEC.json",
)
EVIDENCE_FILES = (
    "evidence/CONTROLLED_HALF_DISPATCH_RECEIPT.json",
    "evidence/P1_FOUNDATION_CHECKS_20261017.md",
    "evidence/p1_foundation_probes.py",
    "reports/P2_R1_COLLECTION_DIAGNOSTIC_20261017.md",
    "reports/P2_R1_PUBLIC_EVIDENCE_RECEIPT_20261017.json",
    "reviews/INDEPENDENT_FOCUSED_CHECK_20261017c.json",
    "reviews/INDEPENDENT_INTEGRATION_CHECK_20261017b.json",
    "reviews/P1_REVIEW.json",
    "reviews/P1_REVIEW_BINDING_FIX.json",
    "reviews/P1_ARCHIVE_REPRODUCTION_REVIEW.json",
    "reviews/P2_REVIEW_20261017.json",
    "reviews/P2_R1_REVIEW_20261017.json",
    "reviews/P0_P1_PRELIMINARY_REPAIR_20261017.md",
    "reviews/PAIR_TEST_RESULTS_20261017a.txt",
    "reviews/PRECOLLECTION_COST_CLARIFICATION_20261017.md",
    "reviews/PRE_FREEZE_REVIEW_20261017.json",
    "reviews/V2_SMOKE_REVIEW_20261017_R3.json",
    "reports/V2_INTERRUPTED_COLLECTION_AUDIT_20261017.json",
    "reviews/PRELIMINARY_MODEL_CHECK.json",
)

SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?key|token|password|passwd|secret|"
    r"authorization|credential|private[_-]?key)\b\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
SECRET_LITERAL_ASSIGNMENT = re.compile(
    r"(?i)([\"'](?:api[_-]?key|access[_-]?key|token|password|passwd|secret|"
    r"authorization|credential|private[_-]?key)[\"']\s*[:=]\s*)"
    r"([\"'][^\"']*[\"'])"
)
PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
COMMON_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16})\b"
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![\w])(?:[A-Za-z]:\\[^\s\"'<>|,;\)\]}]+)")
POSIX_USER_PATH = re.compile(r"(?<![\w])/(?:home|Users)/[^\s\"'<>|,;\)\]}]+")


@dataclass(frozen=True)
class ArchiveFile:
    source: Path
    relative_path: str
    category: str

    @property
    def is_evidence(self) -> bool:
        return self.category == "evidence"


class ArchiveError(RuntimeError):
    pass


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _add_file(root: Path, entries: dict[str, ArchiveFile], relative: str, category: str) -> None:
    source = root / Path(relative)
    if not source.exists():
        return
    if source.is_symlink() or not source.is_file():
        raise ArchiveError(f"allowlisted path is not a regular file: {relative}")
    resolved = source.resolve()
    if root.resolve() not in resolved.parents:
        raise ArchiveError(f"allowlisted path escapes source root: {relative}")
    entries[relative.replace("\\", "/")] = ArchiveFile(source, relative.replace("\\", "/"), category)


def collect_allowlisted_files(source_root: Path) -> list[ArchiveFile]:
    root = source_root.resolve()
    entries: dict[str, ArchiveFile] = {}

    for relative in ROOT_FILES:
        _add_file(root, entries, relative, "source")
    for relative in CONFIG_FILES:
        _add_file(root, entries, f"configs/{relative}", "source")

    for directory in ("package/civilian", "package/tests"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.py")):
            _add_file(root, entries, path.relative_to(root).as_posix(), "source")

    for relative in (
        "staging_tools/prepare_public_archive.py",
    "staging_tools/test_prepare_public_archive.py",
    "staging_tools/audit_interrupted_collection.py",
    ):
        _add_file(root, entries, relative, "source")

    protocols = root / "protocols"
    if protocols.is_dir():
        for path in sorted(protocols.glob("*.md")):
            _add_file(root, entries, path.relative_to(root).as_posix(), "source")

    for directory in (
        "package/civilian/fixtures",
        "package/civilian/registered_data/fixtures",
        "package/civilian/registered_data/data/train",
        "package/civilian/registered_data/data/development",
    ):
        base = root / directory
        if base.is_dir():
            for path in sorted(base.glob("*.json")):
                _add_file(root, entries, path.relative_to(root).as_posix(), "source")
    _add_file(root, entries, "package/civilian/registered_data/input_registry.json", "source")

    for relative in EVIDENCE_FILES:
        _add_file(root, entries, relative, "evidence")

    return [entries[key] for key in sorted(entries)]


def _local_path_markers(source_root: Path, destination: Path) -> list[str]:
    candidates = [source_root.resolve(), destination.resolve(strict=False)]
    profile = os.environ.get("USERPROFILE")
    if profile:
        candidates.append(Path(profile))
    try:
        candidates.append(Path.home())
    except RuntimeError:
        pass

    markers: set[str] = set()
    for candidate in candidates:
        raw = str(candidate)
        if len(raw) < 8:
            continue
        for variant in (raw, raw.replace("\\", "/"), raw.replace("\\", "\\\\")):
            markers.add(variant.casefold())
    return sorted(markers, key=len, reverse=True)


def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def apply(name: str, pattern: re.Pattern[str], replacement: str) -> None:
        nonlocal text
        text, count = pattern.subn(replacement, text)
        if count:
            counts[name] = counts.get(name, 0) + count

    apply("private_key_block", PRIVATE_KEY_BLOCK, "<REDACTED_PRIVATE_KEY>")
    apply("credential_assignment", SECRET_ASSIGNMENT, r"\1<REDACTED_SECRET>")
    apply("bearer_token", BEARER_TOKEN, "<REDACTED_BEARER_TOKEN>")
    apply("known_token", COMMON_TOKEN, "<REDACTED_TOKEN>")
    apply("windows_absolute_path", WINDOWS_ABSOLUTE_PATH, "<REDACTED_LOCAL_PATH>")
    apply("posix_user_path", POSIX_USER_PATH, "<REDACTED_LOCAL_PATH>")
    return text, counts


def _check_unprojected_source(entry: ArchiveFile, payload: bytes, local_markers: Iterable[str]) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveError(f"allowlisted text is not UTF-8: {entry.relative_path}") from error

    restricted = (
        ("private_key_block", PRIVATE_KEY_BLOCK),
        ("credential_assignment", SECRET_LITERAL_ASSIGNMENT),
        ("known_token", COMMON_TOKEN),
        ("bearer_token", BEARER_TOKEN),
    )
    for label, pattern in restricted:
        if pattern.search(text):
            raise ArchiveError(f"redaction required in non-evidence source file ({label}): {entry.relative_path}")
    normalized = text.casefold().replace("\\\\", "\\")
    for marker in local_markers:
        if marker in normalized:
            raise ArchiveError(f"personal absolute path found in non-evidence source file: {entry.relative_path}")


def _destination_is_stageable(destination: Path) -> bool:
    if not destination.exists():
        return True
    return all(child.name == ".git" and child.is_dir() and not child.is_symlink() for child in destination.iterdir())


def _validate_paths(source_root: Path, destination: Path, workspace: Path) -> None:
    source = source_root.resolve()
    target = destination.resolve(strict=False)
    workspace = workspace.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ArchiveError("source and archive destination must be separate directories")
    if workspace != target and workspace not in target.parents:
        raise ArchiveError("archive destination must stay inside the workspace")


def build_plan(source_root: Path, destination: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entries = collect_allowlisted_files(source_root)
    if not entries:
        raise ArchiveError("no allowlisted files found")

    local_markers = _local_path_markers(source_root, destination)
    plan: list[dict[str, object]] = []
    projections: list[dict[str, object]] = []
    for entry in entries:
        source_bytes = entry.source.read_bytes()
        if entry.is_evidence:
            try:
                raw_text = source_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ArchiveError(f"evidence projection is not UTF-8: {entry.relative_path}") from error
            public_text, redactions = _redact_text(raw_text)
            public_bytes = public_text.encode("utf-8")
            output_path = entry.relative_path
            projections.append(
                {
                    "source_path": entry.relative_path,
                    "public_path": output_path,
                    "source_sha256": sha256(source_bytes),
                    "public_sha256": sha256(public_bytes),
                    "redactions": redactions,
                }
            )
        else:
            _check_unprojected_source(entry, source_bytes, local_markers)
            public_bytes = source_bytes
            output_path = entry.relative_path
        plan.append(
            {
                "source_path": entry.relative_path,
                "public_path": output_path,
                "category": entry.category,
                "source_sha256": sha256(source_bytes),
                "public_sha256": sha256(public_bytes),
                "size_bytes": len(public_bytes),
                "redactions": {},
            }
        )
        plan[-1]["redactions"] = next(
            (item["redactions"] for item in projections if item["source_path"] == entry.relative_path),
            {},
        )
    return plan, projections


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def apply_plan(source_root: Path, destination: Path, plan: list[dict[str, object]], projections: list[dict[str, object]]) -> None:
    if not _destination_is_stageable(destination):
        raise ArchiveError("archive destination contains files outside .git; refusing to overwrite")

    destination.mkdir(parents=True, exist_ok=True)
    checksums: list[tuple[str, str]] = []
    for item in plan:
        relative = str(item["source_path"])
        if item["category"] == "evidence":
            payload = (source_root / Path(relative)).read_bytes()
            text, _ = _redact_text(payload.decode("utf-8"))
            payload = text.encode("utf-8")
        else:
            payload = (source_root / Path(relative)).read_bytes()
        output_path = destination / Path(str(item["public_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        if sha256(payload) != item["public_sha256"]:
            raise ArchiveError(f"source changed while staging: {relative}")
        checksums.append((str(item["public_sha256"]), str(item["public_path"])))

    manifest = {
        "schema": "public-archive-manifest/1",
        "source_root_label": "development/ch2_disaster_relief_n02_gain_gate_v1_100x100",
        "files": plan,
        "excluded_roots": [".deps", "outputs", "_pytest-temp", "__pycache__"],
        "excluded_artifacts": ["checkpoints", "raw run outputs", "credentials", "private databases"],
    }
    manifest_bytes = _json_bytes(manifest)
    manifest_path = destination / "PUBLIC_ARCHIVE_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    checksums.append((sha256(manifest_bytes), "PUBLIC_ARCHIVE_MANIFEST.json"))

    redaction_map = {
        "schema": "public-evidence-redaction-map/1",
        "source_files_unchanged": True,
        "files": projections,
    }
    redaction_bytes = _json_bytes(redaction_map)
    redaction_path = destination / "evidence" / "public_projection" / "REDACTION_MAP.json"
    redaction_path.parent.mkdir(parents=True, exist_ok=True)
    redaction_path.write_bytes(redaction_bytes)
    checksums.append((sha256(redaction_bytes), "evidence/public_projection/REDACTION_MAP.json"))

    checksum_lines = [f"{digest}  {relative}" for digest, relative in sorted(checksums, key=lambda pair: pair[1])]
    (destination / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--apply", action="store_true", help="write the staged archive; default is a read-only preview")
    args = parser.parse_args(argv)

    source_root = args.source.resolve()
    workspace = source_root.parent.parent
    destination = (args.destination or (workspace / "github-event-trigger-20261017")).resolve(strict=False)
    try:
        _validate_paths(source_root, destination, workspace)
        plan, projections = build_plan(source_root, destination)
        stageable = _destination_is_stageable(destination)
        if args.apply:
            apply_plan(source_root, destination, plan, projections)
        result = {
            "mode": "applied" if args.apply else "dry-run",
            "destination_name": destination.name,
            "destination_stageable": stageable,
            "file_count": len(plan),
            "evidence_projection_count": len(projections),
            "files": plan,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ArchiveError, OSError) as error:
        print(f"archive preparation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
