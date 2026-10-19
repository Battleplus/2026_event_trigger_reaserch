"""Native-filesystem state I/O, status observation, staging, and export.

This module is standard-library-only.  It never imports the experiment,
environment, model, or torch modules.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRetryPolicy:
    attempts: int = 3
    max_elapsed_seconds: float = 2.0
    delay_seconds: float = 0.05

    def validate(self) -> None:
        if self.attempts < 1:
            raise ValueError("file retry attempts must be positive")
        if self.max_elapsed_seconds < 0 or self.delay_seconds < 0:
            raise ValueError("file retry timing must be non-negative")


DEFAULT_FILE_POLICY = FileRetryPolicy()


def _bounded_file_operation(
    operation: Callable[[], Any], policy: FileRetryPolicy = DEFAULT_FILE_POLICY
) -> Any:
    """Retry only an individual file operation, within a fixed bound."""

    policy.validate()
    started = time.monotonic()
    failure: OSError | None = None
    for attempt in range(policy.attempts):
        try:
            return operation()
        except OSError as exc:
            failure = exc
            elapsed = time.monotonic() - started
            if attempt + 1 >= policy.attempts or elapsed >= policy.max_elapsed_seconds:
                raise
            time.sleep(min(policy.delay_seconds, policy.max_elapsed_seconds - elapsed))
    assert failure is not None
    raise failure


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_atomic_json(
    path: os.PathLike[str] | str,
    value: Any,
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> None:
    """Durably replace a JSON file using a unique same-directory temporary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(value) + "\n").encode("utf-8")

    def write_once() -> None:
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    _bounded_file_operation(write_once, policy)


def durable_append_jsonl(
    path: os.PathLike[str] | str,
    value: Any,
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(value) + "\n").encode("utf-8")

    def append_once() -> None:
        descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("append write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    _bounded_file_operation(append_once, policy)


def read_json(path: os.PathLike[str] | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unescape_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_identity(path: os.PathLike[str] | str) -> dict[str, str]:
    """Return the most-specific Linux mount containing path."""

    resolved = Path(path).resolve()
    if os.name != "posix" or not Path("/proc/self/mountinfo").is_file():
        raise RuntimeError("The active execution directory requires Linux mount metadata")
    best: tuple[int, dict[str, str]] | None = None
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        trailing = right.split()
        mount = Path(_unescape_mount_field(fields[4]))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        record = {
            "resolved_path": str(resolved),
            "mount_point": str(mount),
            "filesystem_type": trailing[0],
            "mount_source": _unescape_mount_field(trailing[1]),
        }
        score = len(str(mount))
        if best is None or score > best[0]:
            best = (score, record)
    if best is None:
        raise RuntimeError(f"No Linux mount identity found for {resolved}")
    return best[1]


def require_native_linux_filesystem(path: os.PathLike[str] | str) -> dict[str, str]:
    identity = filesystem_identity(path)
    resolved = Path(identity["resolved_path"])
    rejected_types = {"9p", "drvfs", "fuseblk", "cifs", "smb3", "ntfs", "ntfs3"}
    if str(resolved).startswith("/mnt/") or identity["filesystem_type"].lower() in rejected_types:
        raise RuntimeError(
            "Active experiment state must use the WSL native Linux filesystem; "
            f"observed {identity}"
        )
    return identity


def process_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def classify_status(
    status: dict[str, Any],
    *,
    now_monotonic: float | None = None,
    stale_after_seconds: float = 3.0,
    alive: Callable[[int | None], bool] = process_alive,
) -> str:
    """Distinguish completed, stopped, live, exited, and stale supervisors."""

    state = status.get("status")
    if state in {"complete", "stopped"}:
        return state
    if state != "running":
        return "invalid"
    if not alive(status.get("supervisor_pid")):
        return "exited_without_final_status"
    now = time.monotonic() if now_monotonic is None else now_monotonic
    updated = status.get("updated_monotonic")
    if not isinstance(updated, (int, float)) or now - float(updated) > stale_after_seconds:
        return "stale_while_process_alive"
    return "running"


def _copy_file_durable(source: Path, destination: Path, policy: FileRetryPolicy) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def copy_once() -> None:
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    _bounded_file_operation(copy_once, policy)


def verify_manifest(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("Frozen identity changed: " + relative)


def stage_sealed_package(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    manifest: dict[str, Any],
    *,
    extra_files: Iterable[str] = ("execution-manifest.json", "hashes.json"),
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    destination_root = Path(destination)
    if destination_root.exists():
        raise RuntimeError("Native attempt path already exists; no retry")
    destination_root.mkdir(parents=True, exist_ok=False)
    copied: dict[str, str] = dict(manifest["files"])
    for relative in extra_files:
        copied[relative] = sha256_file(source_root / relative)
    try:
        for relative in sorted(copied):
            _copy_file_durable(source_root / relative, destination_root / relative, policy)
        verify_manifest(destination_root, copied)
        _fsync_directory(destination_root)
    except BaseException:
        # A partial directory intentionally blocks reuse of this one-shot attempt.
        raise
    return {
        "source": str(source_root),
        "destination": str(destination_root.resolve()),
        "files": len(copied),
        "bytes": sum((destination_root / name).stat().st_size for name in copied),
        "filesystem": require_native_linux_filesystem(destination_root),
    }


def stage_sealed_package_into_initialized_root(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    manifest: dict[str, Any],
    *,
    required_existing: Iterable[str] = ("launcher-status.json",),
    extra_files: Iterable[str] = ("execution-manifest.json", "hashes.json"),
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> dict[str, Any]:
    """Stage after Linux has durably published the pre-copy launcher state."""

    source_root = Path(source).resolve()
    destination_root = Path(destination).resolve()
    expected = set(required_existing)
    observed = {
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
    }
    if not destination_root.is_dir() or observed != expected:
        raise RuntimeError(
            f"Initialized native attempt root has unexpected files: {sorted(observed)}"
        )
    require_native_linux_filesystem(destination_root)
    copied: dict[str, str] = dict(manifest["files"])
    for relative in extra_files:
        copied[relative] = sha256_file(source_root / relative)
    for relative in sorted(copied):
        _copy_file_durable(source_root / relative, destination_root / relative, policy)
    verify_manifest(destination_root, copied)
    _fsync_directory(destination_root)
    return {
        "source": str(source_root),
        "destination": str(destination_root),
        "files": len(copied),
        "bytes": sum((destination_root / name).stat().st_size for name in copied),
        "filesystem": require_native_linux_filesystem(destination_root),
        "pre_copy_status_files": sorted(expected),
    }


def tree_manifest(root: os.PathLike[str] | str, excluded: set[str] | None = None) -> dict[str, dict[str, Any]]:
    root_path = Path(root)
    excluded = excluded or set()
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root_path.rglob("*")):
        relative = path.relative_to(root_path).as_posix()
        if not path.is_file() or relative in excluded or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        output[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return output


def controlled_export(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> dict[str, Any]:
    """Export a stopped/finished native attempt once and verify every byte."""

    source_root = Path(source).resolve()
    destination_root = Path(destination)
    if destination_root.exists():
        raise RuntimeError("Export destination already exists; no overwrite or retry")
    source_manifest = tree_manifest(source_root, {"export-status.json"})
    destination_root.mkdir(parents=True, exist_ok=False)
    try:
        for relative in sorted(source_manifest):
            _copy_file_durable(source_root / relative, destination_root / relative, policy)
        for relative, identity in source_manifest.items():
            output = destination_root / relative
            if output.stat().st_size != identity["bytes"] or sha256_file(output) != identity["sha256"]:
                raise RuntimeError("Export verification failed: " + relative)
        export_manifest = {
            "schema": "verified-controlled-export/1.0.0",
            "source": str(source_root),
            "destination": str(destination_root.resolve()),
            "files": source_manifest,
            "verified": True,
        }
        durable_atomic_json(destination_root / "export-manifest.json", export_manifest, policy)
        complete = {
            "schema": "controlled-export-completion/1.0.0",
            "verified": True,
            "file_count": len(source_manifest),
            "payload_bytes": sum(item["bytes"] for item in source_manifest.values()),
            "export_manifest_sha256": sha256_file(destination_root / "export-manifest.json"),
        }
        durable_atomic_json(destination_root / "EXPORT_COMPLETE.json", complete, policy)
        return complete
    except BaseException as exc:
        try:
            durable_append_jsonl(
                source_root / "export-failures.jsonl",
                {"error": type(exc).__name__ + ": " + str(exc), "worker_relaunched": False},
                policy,
            )
        except BaseException:
            pass
        raise


def append_verified_export_file(
    source_file: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    relative: str,
    *,
    policy: FileRetryPolicy = DEFAULT_FILE_POLICY,
) -> dict[str, Any]:
    """Finalize one settlement file inside the same controlled export phase."""

    source = Path(source_file)
    destination_root = Path(destination)
    manifest_path = destination_root / "export-manifest.json"
    complete_path = destination_root / "EXPORT_COMPLETE.json"
    manifest = read_json(manifest_path)
    complete = read_json(complete_path)
    if not complete.get("verified") or relative in manifest.get("files", {}):
        raise RuntimeError("Controlled export cannot append settlement metadata")
    target = destination_root / relative
    if target.exists():
        raise RuntimeError("Controlled export settlement destination already exists")
    _copy_file_durable(source, target, policy)
    identity = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    if identity["sha256"] != sha256_file(source):
        raise RuntimeError("Controlled export settlement verification failed")
    manifest["files"][relative] = identity
    durable_atomic_json(manifest_path, manifest, policy)
    complete.update({
        "file_count": len(manifest["files"]),
        "payload_bytes": sum(item["bytes"] for item in manifest["files"].values()),
        "export_manifest_sha256": sha256_file(manifest_path),
        "final_settlement_included": relative,
    })
    durable_atomic_json(complete_path, complete, policy)
    return complete
