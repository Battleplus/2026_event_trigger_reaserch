from __future__ import annotations

import copy
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import tarfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OLD_PROJECT = PROJECT.parent / "ch2_disaster_relief_n02_event_trigger_v1_100x100"
ARCHIVE = OLD_PROJECT / "evidence" / "n02-event-v1-7e2445a08ceb479daaaffc9eecb79ad5.tar.gz"
LEGACY_PREFIX = "gain_p1_legacy"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_legacy_sources() -> tuple[dict[str, bytes], type, type]:
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        sources = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".py")
        }
        members = {member.name for member in archive.getmembers() if member.isfile()}
        config_bytes = archive.extractfile("configs/EVENT_TRIGGER_CONFIG.json").read()

    config_path = OLD_PROJECT / "configs" / "EVENT_TRIGGER_CONFIG.json"
    if _sha256(config_bytes) != _sha256(config_path.read_bytes()):
        raise RuntimeError("archived event config differs from the read-only source tree")

    class ArchiveLoader(importlib.abc.Loader):
        def __init__(self, member: str):
            self.member = member

        def create_module(self, spec):
            return None

        def exec_module(self, module) -> None:
            module.__file__ = str(OLD_PROJECT / Path(self.member))
            code = compile(sources[self.member], f"<{ARCHIVE}:{self.member}>", "exec")
            exec(code, module.__dict__)

    class ArchiveFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path=None, target=None):
            if fullname in {LEGACY_PREFIX, f"{LEGACY_PREFIX}.civilian"}:
                return importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            prefix = f"{LEGACY_PREFIX}.civilian."
            if not fullname.startswith(prefix):
                return None
            member = "package/civilian/" + fullname[len(prefix):].replace(".", "/") + ".py"
            if member not in sources:
                return None
            return importlib.util.spec_from_loader(fullname, ArchiveLoader(member))

    sys.meta_path.insert(0, ArchiveFinder())
    scenarios_module = importlib.import_module(f"{LEGACY_PREFIX}.civilian.event_scenarios")
    environment_module = importlib.import_module(f"{LEGACY_PREFIX}.civilian.event_environment")
    return sources, scenarios_module.build_event_scenarios, environment_module.EventReliefEnv


def _cooperative_pair_probe(env_type: type, scenario: dict, *, inject_half_dispatch: bool) -> dict:
    env = env_type(copy.deepcopy(scenario))
    env.reset()
    operation = {
        "op": "cooperate",
        "task_id": "n02-task-05",
        "from_platform": "n02-uav-3",
        "to_platform": "n02-uav-2",
        "share": 0.5,
        "before_task_id": "n02-task-02",
    }
    env.step({
        "action_type": "reconstruct",
        "scope": "GLOBAL_REPLAN",
        "noop": False,
        "queue_operations": [operation],
        "dispatches": [],
    })
    children = next(
        row["subtask_ids"]
        for row in env.public_observation()["tasks"]
        if row["id"] == "n02-task-05"
    )
    target_child = next(
        child_id for child_id in children
        if env._task_specs[child_id]["assigned_owner"] == "n02-uav-2"
    )
    if inject_half_dispatch:
        env._queue_head_dispatches = lambda explicit: [{
            "platform_id": "n02-uav-2",
            "task_id": target_child,
            "source": "committed_queue",
        }]

    noop = {
        "action_type": "reconstruct",
        "scope": "NOOP",
        "noop": True,
        "queue_operations": [],
        "dispatches": [],
    }
    try:
        _, _, _, _, info = env.step(noop)
    except ValueError as exc:
        return {
            "status": "raised",
            "exception": type(exc).__name__,
            "message": str(exc),
            "target_child": target_child,
        }
    return {
        "status": "returned",
        "target_child": target_child,
        "operational_dispatches": [
            [row["task_id"], row["platform_id"]]
            for row in info["operational_dispatches"]
        ],
        "pair_waits": [
            row for row in info["raw_events"]
            if row.get("type") == "cooperative_pair_wait"
        ],
    }


def main() -> None:
    sources, build_legacy_scenarios, LegacyEnv = _load_legacy_sources()
    failure_path = OLD_PROJECT / "evidence" / "n02-event-v1-7e2445a08ceb479daaaffc9eecb79ad5-result" / "FIRST_FAILURE.json"
    historical_failure = json.loads(failure_path.read_text(encoding="utf-8"))

    sys.path.insert(0, str(PROJECT / "package" / "civilian"))
    current_scenarios = importlib.import_module("event_scenarios").build_event_scenarios()
    CurrentEnv = importlib.import_module("event_environment").EventReliefEnv

    legacy = _cooperative_pair_probe(LegacyEnv, build_legacy_scenarios()[0], inject_half_dispatch=True)
    current = _cooperative_pair_probe(CurrentEnv, current_scenarios[0], inject_half_dispatch=True)
    sys.path.insert(0, str(PROJECT / "package" / "civilian"))
    package_layout = importlib.import_module("package_layout")
    dataset_protocol = importlib.import_module("dataset_protocol")
    original_layout_gate = package_layout.resolve_package_layout
    package_layout.resolve_package_layout = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("strict package execution/layout gate was invoked")
    )
    try:
        registry = dataset_protocol.validate_registry(PROJECT / "package" / "civilian" / "registered_data")
    finally:
        package_layout.resolve_package_layout = original_layout_gate

    result = {
        "archive": ARCHIVE.relative_to(PROJECT.parent).as_posix(),
        "archive_event_environment_sha256": _sha256(sources["package/civilian/event_environment.py"]),
        "historical_first_failure": {
            "stage": historical_failure["stage"],
            "message": historical_failure["message"],
            "type": historical_failure["type"],
        },
        "controlled_half_dispatch_injection": True,
        "archived_pre_fix": legacy,
        "current_atomic_guard": current,
        "registry_validation_without_execution_gate": {
            "passed": registry["scenarios_verified"],
            "scenario_count": registry["scenario_count"],
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
