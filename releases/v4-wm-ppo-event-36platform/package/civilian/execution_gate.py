"""One-shot authorization and systemd handoff for the v5 formal-auth repair."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


AUTHORIZATION_SCHEMA = "n02-v5-formal-run-authorization/1"
FORMAL_UPGRADE_SCHEMA = "n02-v5-formal-authorization-upgrade/1"
PREFLIGHT_SCHEMA = "n02-v4-target-runtime-preflight/1"
CONSUMED_SCHEMA = "n02-v4-authorization-consumption/1"
SUBMISSION_SCHEMA = "n02-v4-systemd-submission/1"
SYSTEMD_CONTEXT_SCHEMA = "n02-v4-systemd-context/1"
TARGET_HOST = os.environ.get("N02_TARGET_HOST") or "public-target-placeholder"
RUNTIME_ROOT = Path(os.environ.get("N02_RUNTIME_ROOT", "/opt/n02-runtime"))
RUNS_ROOT = Path(os.environ.get("N02_RUNS_ROOT", RUNTIME_ROOT / "runs"))
DIAGNOSTICS_ROOT = Path(os.environ.get("N02_DIAGNOSTICS_ROOT", RUNTIME_ROOT / "diagnostics"))
AUTHORIZATIONS_ROOT = Path(os.environ.get("N02_AUTHORIZATIONS_ROOT", RUNTIME_ROOT / "authorizations"))
FORMAL_SCOPE = "n02-v5-full-pipeline-once"
try:
    from .package_layout import PACKAGE_RELEASE_ID
except ImportError:
    from package_layout import PACKAGE_RELEASE_ID

PACKAGE_RELEASE = PACKAGE_RELEASE_ID
CANDIDATE_SCOPE = "candidate_target_runtime_preflight_only"
OUTER_RELEASE_ID = "n02-minimal-closure-formal-execution-20261016-02"
FORMAL_BUDGET = {
    "server_wall_seconds": 1800,
    "server_cpu_seconds": 900,
    "rss_bytes": 4 * 1024**3,
    "work_storage_bytes": 1024**3,
    "export_storage_bytes": 1024**3,
    "environment_steps": 4000,
    "candidate_branches": 128,
    "world_model_updates": 8,
    "ppo_updates": 16,
    "policy_training_episodes": 4,
    "development_evaluation_episodes": 2,
    "world_model_forward_batches": 512,
    "world_model_candidate_evaluations": 2048,
}
SYSTEMD_PROPERTIES = (
    "Restart=no",
    "KillMode=control-group",
    "CPUAccounting=yes",
    "MemoryAccounting=yes",
    "MemoryMax=4G",
    "TasksMax=64",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}\Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"required identity file is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PermissionError(f"identity file is not a JSON object: {path}")
    return value


def _body_digest(document: dict[str, Any], digest_key: str, label: str) -> str:
    claimed = document.get(digest_key)
    body = {key: value for key, value in document.items() if key != digest_key}
    if not isinstance(claimed, str) or not _SHA256.fullmatch(claimed) or digest_json(body) != claimed:
        raise PermissionError(f"{label} digest mismatch")
    return claimed


def load_release_binding(package_root: Path) -> dict[str, Any]:
    """Validate the static candidate chain without treating it as formal approval."""
    package_root = package_root.resolve(strict=True)
    release_root = package_root.parent.parent
    manifest = _read_json(release_root / "SOURCE_MANIFEST.json")
    hash_bundle = _read_json(release_root / "HASH_BUNDLE.json")
    request = _read_json(release_root / "REQUEST_CANDIDATE.json")
    attempt = _read_json(release_root / "ATTEMPT_CANDIDATE.json")

    manifest_sha = _body_digest(manifest, "manifest_sha256", "release manifest")
    bundle_sha = _body_digest(hash_bundle, "bundle_sha256", "hash bundle")
    request_sha = _body_digest(request, "candidate_request_identity_sha256", "request identity")
    attempt_sha = _body_digest(attempt, "candidate_attempt_identity_sha256", "attempt identity")
    if (
        manifest.get("outer_release_id") != OUTER_RELEASE_ID
        or hash_bundle.get("outer_release_id") != OUTER_RELEASE_ID
        or request.get("outer_release_id") != OUTER_RELEASE_ID
        or attempt.get("outer_release_id") != OUTER_RELEASE_ID
        or manifest.get("package_release") != PACKAGE_RELEASE
        or hash_bundle.get("source_manifest_sha256") != manifest_sha
        or request.get("source_manifest_sha256") != manifest_sha
        or request.get("hash_bundle_sha256") != bundle_sha
        or attempt.get("candidate_request_identity_sha256") != request_sha
        or hash_bundle.get("complete_package_file_hashes") is None
        or request.get("authorized") is not False
        or request.get("request_frozen") is not False
        or request.get("execution_started") is not False
        or request.get("training_or_rollout") != "not_run"
        or attempt.get("attempt_started") is not False
        or attempt.get("training_or_rollout") != "not_run"
    ):
        raise PermissionError("v5 manifest, hash bundle, request, or attempt identity chain is inconsistent")

    manifest_files = {
        item.get("path"): item.get("sha256")
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    bundle_files = hash_bundle["complete_package_file_hashes"]
    if not isinstance(bundle_files, dict) or manifest_files != bundle_files:
        raise PermissionError("release manifest file list differs from the complete hash bundle")
    for relative, expected in manifest_files.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith("package/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
        ):
            raise PermissionError("release manifest contains an invalid package path or digest")
        path = release_root / Path(relative)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise PermissionError(f"release package hash mismatch: {relative}")

    return {
        "package_release": manifest["package_release"],
        "outer_release_id": OUTER_RELEASE_ID,
        "protocol_id": manifest.get("protocol_id"),
        "map_contract": manifest.get("map_contract"),
        "stage_protocol": request.get("stage_protocol"),
        "release_manifest_sha256": manifest_sha,
        "hash_bundle_sha256": bundle_sha,
        "candidate_request_identity_sha256": request_sha,
        "candidate_attempt_identity_sha256": attempt_sha,
        "request_scope": request.get("request_scope"),
        "budget": request.get("budget"),
        "authorized": request.get("authorized"),
        "request_frozen": request.get("request_frozen"),
    }


def validate_engineering_id(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise ValueError("engineering id must be a simple path component")
    return value


def diagnostic_path(engineering_id: str, root: Path = DIAGNOSTICS_ROOT) -> Path:
    return root / validate_engineering_id(engineering_id)


def _completed(argv: list[str], runner: Callable[..., Any]) -> Any:
    return runner(argv, capture_output=True, text=True, timeout=20, check=False)


def _unit_properties(package_root: Path, *, stdout: str = "journal", stderr: str = "journal") -> tuple[str, ...]:
    return (
        *SYSTEMD_PROPERTIES,
        "Type=exec",
        f"WorkingDirectory={package_root}",
        f"StandardOutput={stdout}",
        f"StandardError={stderr}",
    )


def systemd_preflight(
    package_root: Path,
    driver_path: Path,
    engineering_id: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
    platform_name: str | None = None,
    uid: int | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check manager, linger, and systemd-run option support without submitting a unit."""
    platform_name = sys.platform if platform_name is None else platform_name
    uid = os.getuid() if uid is None else uid
    environment = os.environ if environment is None else environment
    checks: dict[str, Any] = {"user_manager": False, "linger": False, "unit_configuration": False}
    errors = []
    if not platform_name.startswith("linux"):
        errors.append("target is not Linux")
    runtime_dir = environment.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        errors.append("XDG_RUNTIME_DIR is unset")
    elif not Path(runtime_dir).is_dir():
        errors.append("XDG_RUNTIME_DIR is not an existing directory")

    try:
        manager = _completed(["systemctl", "--user", "show-environment"], runner)
        checks["user_manager"] = manager.returncode == 0
        if manager.returncode != 0:
            errors.append("systemctl --user could not reach the user manager")
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks["user_manager_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("systemctl --user could not reach the user manager")

    try:
        linger = _completed(
            ["loginctl", "show-user", str(uid), "--property=Linger", "--value"],
            runner,
        )
        checks["linger"] = linger.returncode == 0 and linger.stdout.strip().lower() == "yes"
        if not checks["linger"]:
            errors.append("user linger is not enabled")
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks["linger_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("loginctl could not verify user linger")

    try:
        validate_engineering_id(engineering_id)
        help_result = _completed(["systemd-run", "--help"], runner)
        help_text = f"{help_result.stdout}\n{help_result.stderr}"
        required_options = ("--user", "--no-block", "--unit", "--property")
        checks["systemd_run_available"] = help_result.returncode == 0
        checks["required_options_supported"] = (
            help_result.returncode == 0
            and all(option in help_text for option in required_options)
        )
        checks["unit_configuration"] = checks["required_options_supported"]
        if not checks["required_options_supported"]:
            errors.append("systemd-run is unavailable or does not document required options")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        checks["unit_configuration_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("systemd-run is unavailable or does not document required options")

    return {
        "schema": "n02-v4-systemd-preflight/1",
        "status": "accepted" if not errors else "rejected",
        "host": TARGET_HOST,
        "checks": checks,
        "errors": errors,
        "unit_properties": list(SYSTEMD_PROPERTIES),
        "unit_created": False,
    }


def validate_authorization_document(
    authorization: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Validate a package-external formal upgrade over a static candidate identity.

    The release sidecars intentionally remain unauthorized and unfrozen.  The
    one-time envelope supplies the formal scope and budget after preflight and
    binds them to every candidate digest, so static metadata cannot be mutated
    to manufacture approval.
    """
    authorization_sha = _body_digest(authorization, "authorization_sha256", "authorization")
    attempt = authorization.get("attempt")
    if not isinstance(attempt, dict):
        raise PermissionError("authorization attempt binding is missing")
    attempt_id = validate_engineering_id(attempt.get("id"))
    attempt_body = {
        "attempt_id": attempt_id,
        "candidate_attempt_identity_sha256": binding.get("candidate_attempt_identity_sha256"),
        "candidate_request_identity_sha256": binding.get("candidate_request_identity_sha256"),
        "release_manifest_sha256": binding.get("release_manifest_sha256"),
        "hash_bundle_sha256": binding.get("hash_bundle_sha256"),
    }
    attempt_sha = digest_json(attempt_body)
    request = authorization.get("request")
    preflight = authorization.get("preflight")
    upgrade = authorization.get("formal_upgrade")
    if not isinstance(request, dict) or not isinstance(preflight, dict) or not isinstance(upgrade, dict):
        raise PermissionError("authorization request or preflight binding is missing")
    static_candidate = (
        binding.get("authorized") is False
        and binding.get("request_frozen") is False
        and binding.get("request_scope") == CANDIDATE_SCOPE
        and binding.get("budget") is None
    )
    upgrade_matches_candidate = (
        upgrade.get("schema") == FORMAL_UPGRADE_SCHEMA
        and upgrade.get("state") == "approved"
        and upgrade.get("authorized") is True
        and upgrade.get("request_frozen") is True
        and upgrade.get("candidate_request_identity_sha256") == binding.get("candidate_request_identity_sha256")
        and upgrade.get("candidate_attempt_identity_sha256") == binding.get("candidate_attempt_identity_sha256")
        and upgrade.get("release_manifest_sha256") == binding.get("release_manifest_sha256")
        and upgrade.get("hash_bundle_sha256") == binding.get("hash_bundle_sha256")
        and upgrade.get("scope") == FORMAL_SCOPE
        and upgrade.get("budget") == FORMAL_BUDGET
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("state") != "authorized"
        or authorization.get("one_time") is not True
        or not static_candidate
        or not upgrade_matches_candidate
        or attempt.get("candidate_identity_sha256") != binding.get("candidate_attempt_identity_sha256")
        or attempt.get("identity_sha256") != attempt_sha
        or authorization.get("release_manifest_sha256") != binding.get("release_manifest_sha256")
        or authorization.get("hash_bundle_sha256") != binding.get("hash_bundle_sha256")
        or request.get("identity_sha256") != binding.get("candidate_request_identity_sha256")
        or request.get("scope") != FORMAL_SCOPE
        or request.get("budget") != FORMAL_BUDGET
        or authorization.get("scope") != FORMAL_SCOPE
        or authorization.get("budget") != FORMAL_BUDGET
        or not isinstance(preflight.get("engineering_id"), str)
        or not isinstance(preflight.get("path"), str)
        or not isinstance(preflight.get("sha256"), str)
        or not _SHA256.fullmatch(preflight["sha256"])
    ):
        raise PermissionError("formal authorization upgrade does not match the static v5 candidate, preflight, scope, or budget")
    return {
        "authorization_sha256": authorization_sha,
        "attempt_id": attempt_id,
        "attempt_identity_sha256": attempt_sha,
        "request_identity_sha256": request["identity_sha256"],
        "authorized": True,
        "request_frozen": True,
        "request_scope": FORMAL_SCOPE,
        "scope": FORMAL_SCOPE,
        "budget": FORMAL_BUDGET,
        "preflight": preflight,
    }


def validate_preflight_receipt(
    preflight_path: Path,
    authorization_preflight: dict[str, Any],
    binding: dict[str, Any],
    *,
    diagnostics_root: Path = DIAGNOSTICS_ROOT,
) -> dict[str, Any]:
    if not preflight_path.is_absolute():
        raise PermissionError("authorization preflight path must be absolute")
    path = preflight_path.resolve(strict=True)
    root = diagnostics_root.resolve(strict=False)
    expected_id = validate_engineering_id(authorization_preflight.get("engineering_id"))
    if (
        path.name != "preflight_receipt.json"
        or path.parent.name != expected_id
        or path.parent.parent != root
        or path.is_symlink()
        or sha256_file(path) != authorization_preflight.get("sha256")
        or authorization_preflight.get("path") != str(path)
    ):
        raise PermissionError("authorization preflight path or digest is invalid")
    receipt = _read_json(path)
    if (
        receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("status") != "accepted"
        or receipt.get("engineering_id") != expected_id
        or receipt.get("package_release") != binding.get("package_release")
        or receipt.get("release_manifest_sha256") != binding.get("release_manifest_sha256")
        or receipt.get("hash_bundle_sha256") != binding.get("hash_bundle_sha256")
        or receipt.get("candidate_request_identity_sha256") != binding.get("candidate_request_identity_sha256")
        or receipt.get("candidate_attempt_identity_sha256") != binding.get("candidate_attempt_identity_sha256")
        or receipt.get("systemd", {}).get("status") != "accepted"
        or receipt.get("systemd", {}).get("unit_created") is not False
    ):
        raise PermissionError("accepted preflight does not bind the current v4 release and systemd checks")
    return receipt


def validate_authorization_path(path: Path, attempt_id: str, root: Path = AUTHORIZATIONS_ROOT) -> Path:
    if not path.is_absolute():
        raise PermissionError("authorization path must be absolute")
    if path.is_symlink():
        raise PermissionError("authorization path cannot be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.parent != root.resolve(strict=False) or resolved.name != f"{attempt_id}.json":
        raise PermissionError("authorization must be the package-external attempt file in the authorized root")
    return resolved


def run_root_for_attempt(attempt_id: str, root: Path = RUNS_ROOT) -> Path:
    return root / validate_engineering_id(attempt_id)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _unit_details(unit_name: str, runner: Callable[..., Any]) -> dict[str, str]:
    result = _completed([
        "systemctl", "--user", "show", unit_name,
        "--property=InvocationID", "--property=MainPID", "--property=ExecMainPID", "--property=ControlGroup",
        "--property=ActiveState", "--property=SubState",
        *[f"--property={name}" for name in ("Restart", "KillMode", "CPUAccounting", "MemoryAccounting", "MemoryMax")],
    ], runner)
    if result.returncode != 0:
        return {}
    return dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )


def _wait_for_invocation(unit_name: str, runner: Callable[..., Any], timeout: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    while True:
        details = _unit_details(unit_name, runner)
        main_pid = details.get("MainPID", "0")
        exec_main_pid = details.get("ExecMainPID", "0")
        if (
            _INVOCATION_ID.fullmatch(details.get("InvocationID", ""))
            and main_pid.isdigit()
            and exec_main_pid.isdigit()
            and max(int(main_pid), int(exec_main_pid)) > 0
            and details.get("ControlGroup")
        ):
            return details
        if time.monotonic() >= deadline:
            raise RuntimeError("submitted systemd unit did not expose InvocationID, PID, and cgroup")
        time.sleep(0.25)


def build_systemd_run_argv(
    *,
    unit_name: str,
    package_root: Path,
    driver_path: Path,
    run_root: Path,
    authorization_path: Path,
    submission_path: Path,
    python_path: str,
    runtime_environment: dict[str, str],
) -> list[str]:
    if not unit_name.endswith(".service") or not _IDENTIFIER.fullmatch(unit_name[:-8]):
        raise ValueError("invalid one-shot systemd unit name")
    console_path = run_root / "console.log"
    argv = ["systemd-run", "--user", "--no-block", f"--unit={unit_name}"]
    argv.extend(f"--property={item}" for item in _unit_properties(
        package_root,
        stdout=f"append:{console_path}",
        stderr=f"append:{console_path}",
    ))
    argv.append(f"--property=Environment=RELIEF_GRAPH_SYSTEMD_UNIT={unit_name}")
    argv.extend(f"--property=Environment={key}={value}" for key, value in runtime_environment.items())
    argv.extend([
        "--", python_path, "-B", str(driver_path), "--_systemd-managed", "--run",
        "--package-root", str(package_root), "--work-root", str(run_root),
        "--authorization", str(authorization_path), "--_submission-receipt", str(submission_path),
    ])
    return argv


def submit_once(
    *,
    run_root: Path,
    attempt_id: str,
    authorization_sha256: str,
    preflight_sha256: str,
    unit_name: str,
    argv: list[str],
    runner: Callable[..., Any] = subprocess.run,
    invocation_timeout: float = 15.0,
) -> dict[str, Any]:
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    consumed_path = run_root / "authorization_consumed.json"
    submission_path = run_root / "systemd_submission.json"
    _write_exclusive(consumed_path, {
        "schema": CONSUMED_SCHEMA,
        "attempt_id": attempt_id,
        "authorization_sha256": authorization_sha256,
        "preflight_sha256": preflight_sha256,
        "state": "submitting",
    })
    submission = {
        "schema": SUBMISSION_SCHEMA,
        "attempt_id": attempt_id,
        "authorization_sha256": authorization_sha256,
        "preflight_sha256": preflight_sha256,
        "unit_name": unit_name,
        "work_root": str(run_root),
        "state": "submitting",
    }
    _write_exclusive(submission_path, submission)
    try:
        result = _completed(argv, runner)
        if result.returncode != 0:
            raise RuntimeError(f"systemd-run rejected the one-shot service: {result.stderr.strip()}")
        details = _wait_for_invocation(unit_name, runner, invocation_timeout)
        if details.get("ActiveState") not in {"active", "inactive", "failed", "activating"}:
            raise RuntimeError("systemd unit state could not be verified after submission")
        handoff = {
            "schema": SYSTEMD_CONTEXT_SCHEMA,
            "attempt_id": attempt_id,
            "authorization_sha256": authorization_sha256,
            "preflight_sha256": preflight_sha256,
            "unit_name": unit_name,
            "invocation_id": details["InvocationID"],
            "main_pid": int(details["MainPID"] or details["ExecMainPID"]),
            "active_main_pid": int(details["MainPID"]),
            "exec_main_pid": int(details["ExecMainPID"]),
            "control_group": details["ControlGroup"],
            "active_state": details.get("ActiveState"),
            "sub_state": details.get("SubState"),
            "console_log": str(run_root / "console.log"),
            "systemd_run_stdout": result.stdout,
            "systemd_run_stderr": result.stderr,
        }
        _write_exclusive(run_root / "systemd_handoff.json", handoff)
        submission["state"] = "submitted"
        submission["invocation_id"] = handoff["invocation_id"]
        _write_atomic(submission_path, submission)
        consumed = json.loads(consumed_path.read_text(encoding="utf-8"))
        consumed["state"] = "submitted"
        consumed["invocation_id"] = handoff["invocation_id"]
        _write_atomic(consumed_path, consumed)
        return handoff
    except BaseException as exc:
        try:
            consumed = json.loads(consumed_path.read_text(encoding="utf-8"))
            consumed["state"] = "submission_failed_no_replay"
            consumed["error"] = f"{type(exc).__name__}: {exc}"
            _write_atomic(consumed_path, consumed)
        except OSError:
            pass
        raise


def verify_managed_invocation(
    *,
    authorization_path: Path,
    run_root: Path,
    submission_path: Path,
    environment: dict[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    process_cgroup: str | None = None,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    authorization = _read_json(authorization_path)
    authorization_sha = sha256_file(authorization_path)
    attempt_id = authorization.get("attempt", {}).get("id")
    expected_unit = f"n02-v4-{validate_engineering_id(attempt_id)}.service"
    invocation_id = environment.get("INVOCATION_ID", "")
    if (
        environment.get("RELIEF_GRAPH_SYSTEMD_UNIT") != expected_unit
        or not _INVOCATION_ID.fullmatch(invocation_id)
        or submission_path.resolve(strict=True) != (run_root / "systemd_submission.json").resolve(strict=True)
    ):
        raise PermissionError("formal run was not started by its authorized one-shot systemd unit")
    consumed = _read_json(run_root / "authorization_consumed.json")
    submission = _read_json(submission_path)
    if (
        consumed.get("schema") != CONSUMED_SCHEMA
        or consumed.get("attempt_id") != attempt_id
        or consumed.get("authorization_sha256") != authorization_sha
        or consumed.get("state") not in {"submitting", "submitted"}
        or submission.get("schema") != SUBMISSION_SCHEMA
        or submission.get("attempt_id") != attempt_id
        or submission.get("authorization_sha256") != authorization_sha
        or submission.get("unit_name") != expected_unit
        or submission.get("work_root") != str(run_root)
        or (submission.get("invocation_id") is not None and submission.get("invocation_id") != invocation_id)
        or (consumed.get("invocation_id") is not None and consumed.get("invocation_id") != invocation_id)
    ):
        raise PermissionError("systemd submission does not match the consumed authorization")
    details = _unit_details(expected_unit, runner)
    if (
        details.get("InvocationID") != invocation_id
        or details.get("MainPID") != str(os.getpid())
        or not details.get("ControlGroup")
        or any(details.get(name) != expected for name, expected in (
            ("Restart", "no"),
            ("KillMode", "control-group"),
            ("CPUAccounting", "yes"),
            ("MemoryAccounting", "yes"),
            ("MemoryMax", str(4 * 1024**3)),
        ))
    ):
        raise PermissionError("systemd InvocationID, process group, or required resource properties do not match this supervisor")
    process_cgroup = process_cgroup or Path("/proc/self/cgroup").read_text(encoding="ascii")
    if expected_unit not in process_cgroup or expected_unit not in details["ControlGroup"]:
        raise PermissionError("supervisor is outside its authorized systemd control group")
    return {
        "schema": SYSTEMD_CONTEXT_SCHEMA,
        "attempt_id": attempt_id,
        "unit_name": expected_unit,
        "invocation_id": invocation_id,
        "main_pid": os.getpid(),
        "control_group": details["ControlGroup"],
        "active_state": details.get("ActiveState"),
        "sub_state": details.get("SubState"),
        "unit_properties": {name: details[name] for name in ("Restart", "KillMode", "CPUAccounting", "MemoryAccounting", "MemoryMax")},
        "console_log": str(run_root / "console.log"),
    }
