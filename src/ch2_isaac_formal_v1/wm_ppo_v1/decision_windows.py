"""Join ordinary pre-candidate decision projections to independent gain labels.

Decision examples combine raw public history through t-1 with the separately
hashed public projection sampled at decision time t. Adjacent physical targets
remain available through ``iter_raw_dynamic_windows`` and are never attached to
these gain examples.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from .public_dynamics import public_projection_to_frame
from .raw_public_windows import (
    RawPublicWindow,
    _events_by_branch,
    _tick_frame,
    _ticks,
    iter_raw_public_windows,
)


EXPECTED_DECISION_PHASE = "post_public_updates_pre_candidate_generation_pre_step"
ORDINARY_EVENT_TYPES = frozenset({
    "task_activated", "public_observation", "search_cell_complete", "energy_anomaly",
})


@dataclass(frozen=True)
class DecisionWindow:
    history: tuple[Any, ...]
    gain_target: float
    state_id: str
    scenario_id: str
    split: str
    decision_time_s: int
    backend_time_s: float
    raw_history_phase: str = "post_step_through_decision_time_minus_one"
    decision_projection_phase: str = EXPECTED_DECISION_PHASE
    gain_target_phase: str = "independent_two_suffix_aggregate"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _safe_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str):
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _report_scene_state(reports: Any) -> tuple[str, str] | None:
    if not isinstance(reports, list) or len(reports) != 2:
        return None
    scenes, states, suffixes = set(), set(), set()
    for report in reports:
        if not isinstance(report, str):
            return None
        path = PurePosixPath(report.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or path.name != "reduction.json" or len(path.parts) < 4:
            return None
        if not path.parts[-2].startswith("suffix_"):
            return None
        scenes.add(path.parts[-4])
        states.add(path.parts[-3])
        suffixes.add(path.parts[-2])
    if len(scenes) != 1 or len(states) != 1 or len(suffixes) != 2:
        return None
    return next(iter(scenes)), next(iter(states))


def _load_indexed_projection(state_root: Path, local_state_id: str) -> tuple[dict[str, Any], dict[str, Any], str, str] | None:
    index_path = state_root / "state_index.json"
    index = _read_json(index_path)
    sidecar = index_path.with_suffix(".sha256")
    index_hash = _sha256(index_path)
    try:
        sidecar_hash = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, UnicodeError, IndexError):
        return None
    if index is None or not isinstance(index.get("states"), list) or index_hash != sidecar_hash:
        return None
    matches = [entry for entry in index["states"]
               if isinstance(entry, dict)
               and Path(str(entry.get("private_file", ""))).stem == local_state_id]
    if len(matches) != 1:
        return None
    entry = matches[0]
    public_path = _safe_path(state_root, entry.get("public_file"))
    if public_path is None or not public_path.is_file():
        return None
    public_hash = _sha256(public_path)
    if not isinstance(entry.get("public_sha256"), str) or public_hash != entry["public_sha256"]:
        return None
    projection = _read_json(public_path)
    if projection is None:
        return None
    decision_time = _finite(entry.get("decision_time_s"))
    indexed_time = _finite(entry.get("sim_time_s"))
    backend_time = _finite(entry.get("backend_time_s"))
    projection_time = _finite(projection.get("sim_time_s"))
    if (decision_time is None or not decision_time.is_integer() or indexed_time != decision_time
            or projection_time != decision_time or backend_time is None
            or backend_time != decision_time - 1
            or entry.get("continuation_phase") != EXPECTED_DECISION_PHASE):
        return None
    events = entry.get("ordinary_events")
    if (not isinstance(events, list) or not events
            or any(not isinstance(event, str) or event not in ORDINARY_EVENT_TYPES for event in events)):
        return None
    return entry, projection, public_hash, index_hash


def _decision_frame(projection: Mapping[str, Any], ordinary_events: list[str]):
    allowed_projection = {key: projection[key] for key in
                         ("sim_time_s", "aircraft", "public_observations", "public_tasks")
                         if key in projection}
    frame = public_projection_to_frame(allowed_projection)
    time_s = int(frame.time_s)
    public_events = tuple({"event": name, "sim_time_s": time_s,
                           "available_at_sim_time_s": time_s}
                          for name in sorted(set(ordinary_events)))
    return replace(frame, public_events=public_events, action=None)


def _raw_history_at_decisions(
    trajectory: Path,
    events_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    decision_times: set[int],
    history_steps: int,
) -> dict[int, tuple[Any, ...]]:
    """Stream representative frames and snapshot history at t-1 without t rows."""
    receipt = _read_json(receipt_path)
    manifest = _read_json(manifest_path)
    if (receipt is None or receipt.get("status") != "complete"
            or receipt.get("actual_isaac_run") is not True or manifest is None):
        return {}
    event_map, _ = _events_by_branch(events_path)
    events = event_map.get("representative", [])
    expected_actors = {str(item["aircraft_id"]) for item in receipt.get("actor_manifest", [])
                       if isinstance(item, Mapping) and item.get("aircraft_id") is not None}
    desired = {time_s - 1 for time_s in decision_times}
    snapshots: dict[int, tuple[Any, ...]] = {}
    history = deque(maxlen=history_steps)
    active_branch = None
    previous_tick = None
    event_pointer = 0
    activated: set[str] = set()
    completed: set[str] = set()
    with gzip.open(trajectory, "rt", encoding="utf-8") as stream:
        for branch, tick, rows in _ticks(stream):
            if branch != "representative":
                if active_branch == "representative":
                    break
                continue
            if active_branch != branch:
                active_branch = branch
                previous_tick = None
                event_pointer = 0
                activated.clear()
                completed.clear()
                history.clear()
            actor_ids = [str(row.get("aircraft_id", "")) for row in rows]
            if len(actor_ids) != len(set(actor_ids)) or (expected_actors and set(actor_ids) != expected_actors):
                history.clear()
                previous_tick = tick
                continue
            if previous_tick is not None and tick != previous_tick + 1:
                history.clear()
            visible = []
            while event_pointer < len(events) and events[event_pointer]["_visible_at_s"] <= tick:
                event = events[event_pointer]
                visible.append(event)
                task_id = event.get("task_id")
                if task_id is not None and event.get("event") == "task_activated":
                    activated.add(str(task_id))
                elif task_id is not None and event.get("event") == "search_cell_complete":
                    completed.add(str(task_id))
                    activated.add(str(task_id))
                event_pointer += 1
            frame = _tick_frame(tick, rows, {}, activated, completed, visible)
            history.append(frame)
            previous_tick = tick
            if tick in desired:
                snapshots[tick + 1] = tuple(history)
            if len(snapshots) == len(decision_times):
                break
    return snapshots


def iter_decision_windows(batch_output: str | Path, *, history_steps: int = 8) -> Iterator[DecisionWindow]:
    """Yield only independently labeled ordinary states with verified public inputs.

    The existing light-model dataset entry point is called first, so its
    terminal 168-scene/30-30 gate and source-report/projection checks remain
    authoritative. No private snapshot is opened here.
    """
    if history_steps < 2:
        raise ValueError("history_steps must be at least two (prior raw history plus decision frame)")
    try:
        from ..light_models_v1.dataset import load_feature_dataset
    except ImportError:
        from light_models_v1.dataset import load_feature_dataset

    root = Path(batch_output).resolve()
    dataset = load_feature_dataset(root)
    if not dataset.get("gate", {}).get("ready"):
        return
    receipt = _read_json(root / "batch_receipt.json")
    aggregate = _read_json(root / "aggregate.json")
    if receipt is None or aggregate is None or not isinstance(aggregate.get("states"), list):
        return
    accepted = {str(row.get("state_id")): row for row in dataset.get("rows", [])
                if isinstance(row, dict) and row.get("state_id") is not None}
    if not accepted:
        return
    scene_by_name = {}
    for scene in receipt.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        output_path = str(scene.get("output_path", "")).replace("\\", "/").strip("/")
        if output_path:
            scene_by_name[output_path.split("/")[-1]] = scene

    selected: dict[str, list[dict[str, Any]]] = {}
    for state in aggregate["states"]:
        if not isinstance(state, dict):
            continue
        state_id = state.get("state_id")
        if state_id is None or str(state_id) not in accepted:
            continue
        accepted_row = accepted[str(state_id)]
        if (accepted_row.get("target") != state.get("gain_mean")
                or accepted_row.get("split") != state.get("split")
                or state.get("valid_mask") is not True):
            continue
        binding = _report_scene_state(state.get("source_reports"))
        scenario_id, local_state_id = state.get("scenario_id"), state.get("local_state_id")
        if (binding is None or binding[1] != local_state_id
                or state_id != f"{scenario_id}::{local_state_id}"):
            continue
        scene_name, bound_local_id = binding
        scene = scene_by_name.get(scene_name)
        if (scene is None or scene.get("scenario_id") != scenario_id
                or scene.get("split") != state.get("split")):
            continue
        scene_dir = _safe_path(root, scene.get("output_path"))
        if scene_dir is None:
            continue
        state_root = scene_dir / "representative" / "state_sampling"
        loaded = _load_indexed_projection(state_root, bound_local_id)
        if loaded is None:
            continue
        entry, projection, public_hash, index_hash = loaded
        scene_receipt = _read_json(scene_dir / "representative" / "run_receipt.json")
        manifest_path = scene_dir / "representative" / "manifest.json"
        manifest = _read_json(manifest_path)
        if (scene_receipt is None or scene_receipt.get("status") != "complete"
                or scene_receipt.get("actual_isaac_run") is not True or manifest is None):
            continue
        logs = scene_receipt.get("logs", {}) if isinstance(scene_receipt.get("logs"), Mapping) else {}
        manifest_logs = manifest.get("logs", {}) if isinstance(manifest.get("logs"), Mapping) else {}
        trajectory_name = scene_receipt.get("trajectory", logs.get("trajectory"))
        events_name = scene_receipt.get("events", logs.get("events"))
        trajectory = _safe_path(scene_dir / "representative", trajectory_name)
        events = _safe_path(scene_dir / "representative", events_name)
        if trajectory is None or events is None:
            continue
        if manifest_logs.get("trajectory") != trajectory.name or manifest_logs.get("events") != events.name:
            continue
        state_id_text = str(state_id)
        selected.setdefault(scene_name, []).append({
            "state_id": state_id_text, "scenario_id": str(scenario_id),
            "split": state["split"], "gain": float(accepted_row["target"]),
            "entry": entry, "projection": projection, "public_hash": public_hash,
            "index_hash": index_hash, "trajectory": trajectory, "events": events,
            "receipt": scene_dir / "representative" / "run_receipt.json",
            "manifest": manifest_path,
        })

    for scene_name, states in selected.items():
        first = states[0]
        state_times = {int(state["entry"]["decision_time_s"]): state for state in states}
        raw_histories = _raw_history_at_decisions(
            first["trajectory"], first["events"], first["receipt"], first["manifest"],
            set(state_times), history_steps - 1)
        for decision_time, state in sorted(state_times.items()):
            raw_history = raw_histories.get(decision_time)
            if not raw_history or raw_history[-1].time_s != decision_time - 1:
                continue
            if any(frame.time_s >= decision_time for frame in raw_history):
                continue
            decision_frame = _decision_frame(state["projection"], state["entry"]["ordinary_events"])
            history = tuple(raw_history) + (decision_frame,)
            yield DecisionWindow(
                history=history, gain_target=state["gain"], state_id=state["state_id"],
                scenario_id=state["scenario_id"], split=state["split"],
                decision_time_s=decision_time,
                backend_time_s=float(state["entry"]["backend_time_s"]),
            )


def iter_raw_dynamic_windows(
    trajectory_path: str | Path,
    events_path: str | Path,
    receipt_path: str | Path,
    *,
    history_steps: int = 8,
    manifest_path: str | Path | None = None,
) -> Iterator[RawPublicWindow]:
    """Yield raw adjacent-state supervision separately from decision gain rows."""
    for window in iter_raw_public_windows(
            trajectory_path, events_path, receipt_path, history_steps=history_steps,
            manifest_path=manifest_path):
        if window.branch == "representative" and window.target.gain is not None:
            raise ValueError("raw dynamic supervision must not carry paired gain labels")
        if window.branch == "representative":
            yield window
