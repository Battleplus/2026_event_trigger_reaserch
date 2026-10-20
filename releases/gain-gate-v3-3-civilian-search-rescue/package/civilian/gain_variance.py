"""K=8 paired-label variance diagnostics for frozen training states."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


DIAGNOSTIC_SCHEMA = "gain-gate-variance-diagnostic/1"
STATE_SCHEMA = "gain-gate-state-variance/1"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_ADDITIONAL_SUFFIXES = 6
BASE_SUFFIX_COUNT = 2
TOTAL_SUFFIX_COUNT = BASE_SUFFIX_COUNT + DEFAULT_ADDITIONAL_SUFFIXES


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def suffix_seed(scenario_id: str, state_id: str, timepoint: int, suffix_index: int) -> int:
    """Match the runner's counterfactual suffix seed namespace."""
    key = {
        "namespace": "counterfactual-suffix-v1", "scenario_id": scenario_id,
        "state_id": state_id, "time": int(timepoint), "suffix_index": int(suffix_index),
    }
    return int.from_bytes(hashlib.sha256(_canonical(key)).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _record_row(record: Mapping[str, Any]) -> Mapping[str, Any]:
    row = record.get("row", record)
    if not isinstance(row, Mapping):
        raise ValueError("GAIN_VARIANCE_ROW_INVALID")
    return row


def _selection_key(row: Mapping[str, Any], selection_seed: int) -> str:
    identity = {
        "schema": "gain-variance-sample-key/1",
        "selection_seed": int(selection_seed),
        "scenario_id": str(row.get("scenario_id", row.get("parent_id", ""))),
        "state_id": str(row.get("state_id", "")),
        "snapshot_sha256": str(row.get("snapshot_sha256", "")),
    }
    return hashlib.sha256(_canonical(identity)).hexdigest()


def select_training_records(
    records: Sequence[Mapping[str, Any]], *, sample_size: int = DEFAULT_SAMPLE_SIZE,
    selection_seed: int = 8201,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Select ordinary, valid train states without consulting any gain value."""
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("GAIN_VARIANCE_SAMPLE_SIZE_INVALID")
    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("GAIN_VARIANCE_SELECTION_SEED_INVALID")

    eligible: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    seen_state_ids: set[str] = set()
    training_row_count = 0
    excluded_forced = 0
    excluded_invalid = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("GAIN_VARIANCE_RECORD_INVALID")
        row = _record_row(record)
        if row.get("split") != "train":
            continue
        training_row_count += 1
        state_id = row.get("state_id")
        scenario_id = row.get("scenario_id", row.get("parent_id"))
        if not isinstance(state_id, str) or not state_id or not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("GAIN_VARIANCE_TRAIN_IDENTITY_MISSING")
        if state_id in seen_state_ids:
            raise ValueError(f"GAIN_VARIANCE_DUPLICATE_STATE_ID:{state_id}")
        seen_state_ids.add(state_id)
        if row.get("forced") is not False:
            excluded_forced += 1
            continue
        if row.get("valid_mask") is not True:
            excluded_invalid += 1
            continue
        eligible.append((_selection_key(row, selection_seed), record, row))

    if len(eligible) < sample_size:
        raise ValueError(
            f"GAIN_VARIANCE_INSUFFICIENT_ELIGIBLE_TRAIN_STATES:{len(eligible)}<{sample_size}"
        )
    eligible.sort(key=lambda item: (item[0], str(item[2]["state_id"])))
    selected = [item[1] for item in eligible[:sample_size]]
    receipt = {
        "algorithm": "sha256_rank_of_scenario_state_snapshot_identity/1",
        "selection_seed": selection_seed,
        "training_row_count": training_row_count,
        "eligible_ordinary_valid_state_count": len(eligible),
        "excluded_forced_state_count": excluded_forced,
        "excluded_invalid_state_count": excluded_invalid,
        "requested_sample_size": sample_size,
        "selected_state_ids": [str(_record_row(item)["state_id"]) for item in selected],
        "selected_rank_hashes": [item[0] for item in eligible[:sample_size]],
        "gain_values_used_for_selection": False,
    }
    return selected, receipt


def _unwrap_label(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and value.get("schema") is None and isinstance(value.get("label"), Mapping):
        value = value["label"]
    if not isinstance(value, Mapping):
        raise ValueError("GAIN_VARIANCE_LABEL_MISSING")
    return value


def _load_bundle(
    record: Mapping[str, Any], record_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    loaded = record_loader(record) if record_loader is not None else record
    if not isinstance(loaded, Mapping):
        raise ValueError("GAIN_VARIANCE_LOADED_RECORD_INVALID")
    row = _record_row(loaded)
    scenario = loaded.get("scenario")
    snapshot_json = loaded.get("snapshot_json")
    suffix_jsons = loaded.get("suffix_jsons")
    label = _unwrap_label(loaded.get("label"))
    if not isinstance(scenario, Mapping) or not isinstance(snapshot_json, str):
        raise ValueError("GAIN_VARIANCE_STATE_ARTIFACTS_MISSING")
    if not isinstance(suffix_jsons, Sequence) or isinstance(suffix_jsons, (str, bytes)):
        raise ValueError("GAIN_VARIANCE_SUFFIX_ARTIFACTS_INVALID")
    if len(suffix_jsons) != BASE_SUFFIX_COUNT or any(not isinstance(item, str) for item in suffix_jsons):
        raise ValueError("GAIN_VARIANCE_REQUIRES_ORIGINAL_K2_SUFFIXES")
    return {
        "row": row, "scenario": scenario, "snapshot_json": snapshot_json,
        "suffix_jsons": list(suffix_jsons), "label": label,
    }


def _build_cost_config(raw: Mapping[str, Any], modules: Mapping[str, Any]) -> Any:
    cost_module = modules.get("cost_accounting")
    config_type = getattr(cost_module, "GainCostConfig", None) if cost_module is not None else None
    if config_type is None:
        try:
            from .cost_accounting import GainCostConfig
        except ImportError:
            from cost_accounting import GainCostConfig
        config_type = GainCostConfig
    expected_fields = set(config_type.__dataclass_fields__)
    if set(raw) != expected_fields:
        raise ValueError("GAIN_VARIANCE_COST_CONFIG_INVALID")
    return config_type(**{key: float(value) for key, value in raw.items()})


def _sign(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _gain_summary(recomputed: Mapping[str, Any]) -> dict[str, Any]:
    if recomputed.get("valid_mask") is not True:
        return {
            "valid_mask": False, "pair_count": int(recomputed.get("pair_count", 0)),
            "gain_mean": None, "paired_variance": None, "mean_sign": None,
            "positive_suffix_count": 0, "negative_suffix_count": 0, "zero_suffix_count": 0,
        }
    gains = [
        float(row["gain"]) for row in recomputed.get("pairs", [])
        if isinstance(row, Mapping) and row.get("valid_mask") is True and row.get("gain") is not None
    ]
    if not gains or any(not math.isfinite(value) for value in gains):
        raise ValueError("GAIN_VARIANCE_ORACLE_GAIN_SUMMARY_INVALID")
    mean = float(recomputed["gain_mean"])
    variance = recomputed.get("paired_variance")
    return {
        "valid_mask": True,
        "pair_count": len(gains),
        "gain_mean": mean,
        "paired_variance": None if variance is None else float(variance),
        "mean_sign": _sign(mean),
        "positive_suffix_count": sum(value > 0.0 for value in gains),
        "negative_suffix_count": sum(value < 0.0 for value in gains),
        "zero_suffix_count": sum(value == 0.0 for value in gains),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_file(path)


def _validate_source_bundle(bundle: Mapping[str, Any], modules: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = bundle["row"]
    label = bundle["label"]
    pairs_module = modules["pairs"]
    oracle_module = modules["oracle"]
    scenario = copy.deepcopy(dict(bundle["scenario"]))
    snapshot = pairs_module._snapshot_payload(bundle["snapshot_json"], pairs_module.SNAPSHOT_SCHEMA)

    if row.get("split") != "train" or scenario.get("split") != "train":
        raise ValueError("GAIN_VARIANCE_NON_TRAIN_INPUT")
    for key in ("state_id", "scenario_id", "split"):
        expected = str(row.get(key, row.get("parent_id", "")))
        actual = label.get(key)
        if str(actual) != expected:
            raise ValueError(f"GAIN_VARIANCE_ROW_LABEL_IDENTITY_MISMATCH:{key}")
    if scenario.get("parent_id") != label.get("scenario_id"):
        raise ValueError("GAIN_VARIANCE_SCENARIO_LABEL_IDENTITY_MISMATCH")
    if snapshot.get("state_id") != label.get("state_id") or snapshot.get("scenario_id") != label.get("scenario_id"):
        raise ValueError("GAIN_VARIANCE_SNAPSHOT_LABEL_IDENTITY_MISMATCH")
    if label.get("suffix_count") != BASE_SUFFIX_COUNT or len(label.get("pairs", [])) != BASE_SUFFIX_COUNT:
        raise ValueError("GAIN_VARIANCE_BASE_LABEL_NOT_K2")
    if label.get("valid_mask") is not True or label.get("forced") is not False:
        raise ValueError("GAIN_VARIANCE_BASE_LABEL_NOT_ORDINARY_VALID")
    if label.get("source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise ValueError("GAIN_VARIANCE_LABEL_SNAPSHOT_BINDING_MISMATCH")
    if row.get("snapshot_sha256") not in (None, snapshot.get("snapshot_sha256")):
        raise ValueError("GAIN_VARIANCE_ROW_SNAPSHOT_HASH_MISMATCH")
    if row.get("label_sha256") not in (None, label.get("label_sha256")):
        raise ValueError("GAIN_VARIANCE_ROW_LABEL_HASH_MISMATCH")

    oracle = oracle_module.validate_gain_label(label)
    if oracle.get("valid") is not True or oracle.get("label_valid_mask") is not True:
        raise ValueError("GAIN_VARIANCE_BASE_ORACLE_INVALID")
    seed_rows = row.get("suffix_seed")
    if not isinstance(seed_rows, list) or len(seed_rows) != BASE_SUFFIX_COUNT:
        raise ValueError("GAIN_VARIANCE_BASE_SUFFIX_SEEDS_INVALID")
    for index, (suffix_json, pair, expected_seed) in enumerate(zip(
        bundle["suffix_jsons"], label["pairs"], seed_rows,
    )):
        suffix = pairs_module._snapshot_payload(suffix_json, pairs_module.SUFFIX_SCHEMA)
        if suffix.get("source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
            raise ValueError(f"GAIN_VARIANCE_BASE_SUFFIX_SOURCE_MISMATCH:{index}")
        if type(expected_seed) is not int or suffix.get("suffix_seed") != expected_seed or pair.get("suffix_seed") != expected_seed:
            raise ValueError(f"GAIN_VARIANCE_BASE_SUFFIX_SEED_MISMATCH:{index}")
        if pair.get("suffix_json_sha256") != pairs_module._digest(suffix_json):
            raise ValueError(f"GAIN_VARIANCE_BASE_SUFFIX_HASH_MISMATCH:{index}")
    return scenario, snapshot, oracle


def _continuation_cpu_seconds(pair: Mapping[str, Any], arm: str) -> float:
    branch = pair.get(arm)
    trace = branch.get("trace") if isinstance(branch, Mapping) else None
    if not isinstance(trace, Mapping):
        return 0.0
    value = float(trace.get("continuation_policy_cpu_seconds", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("GAIN_VARIANCE_CONTINUATION_CPU_INVALID")
    return value


def _compare_base_suffix_replays(
    original_label: Mapping[str, Any], k8_label: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comparisons = []
    for index, original in enumerate(original_label["pairs"]):
        rerun = k8_label["pairs"][index]
        if original.get("suffix_seed") != rerun.get("suffix_seed"):
            raise ValueError(f"GAIN_VARIANCE_K2_REPLAY_SUFFIX_MISMATCH:{index}")
        old_gain = original.get("gain")
        new_gain = rerun.get("gain")
        gain_delta = None if old_gain is None or new_gain is None else float(new_gain) - float(old_gain)
        old_keep_trace = original.get("keep", {}).get("trace", {})
        new_keep_trace = rerun.get("keep", {}).get("trace", {})
        old_replan_trace = original.get("replan", {}).get("trace", {})
        new_replan_trace = rerun.get("replan", {}).get("trace", {})
        comparisons.append({
            "suffix_index": index,
            "suffix_seed": original.get("suffix_seed"),
            "saved_k2_gain": old_gain,
            "rerun_within_k8_gain": new_gain,
            "gain_delta": gain_delta,
            "keep_safety_monitor_cpu_seconds_saved_k2": _continuation_cpu_seconds(original, "keep"),
            "keep_safety_monitor_cpu_seconds_rerun_k8": _continuation_cpu_seconds(rerun, "keep"),
            "replan_safety_monitor_cpu_seconds_saved_k2": _continuation_cpu_seconds(original, "replan"),
            "replan_safety_monitor_cpu_seconds_rerun_k8": _continuation_cpu_seconds(rerun, "replan"),
            "keep_event_tape_sha256_matches": (
                old_keep_trace.get("event_tape_sha256") == new_keep_trace.get("event_tape_sha256")
            ),
            "replan_event_tape_sha256_matches": (
                old_replan_trace.get("event_tape_sha256") == new_replan_trace.get("event_tape_sha256")
            ),
        })
    return comparisons


def _planner_action_and_cost(label: Mapping[str, Any], modules: Mapping[str, Any]) -> tuple[dict[str, Any], float, int, Any]:
    try:
        first_pair = label["pairs"][0]
        action = copy.deepcopy(dict(first_pair["replan"]["action"]))
        action_hash = modules["pairs"]._digest(action)
        for pair in label["pairs"]:
            if modules["pairs"]._digest(pair["replan"]["action"]) != action_hash:
                raise ValueError("GAIN_VARIANCE_BASE_ACTION_DIFFERS_BY_SUFFIX")
        trace = first_pair["replan"]["trace"]
        cpu_seconds = float(trace["initial_planner_cpu_seconds"])
        delay = int(first_pair["action_delay_ticks"])
        raw_cost = first_pair["replan"]["cost"]["cost_config"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("GAIN_VARIANCE_"):
            raise
        raise ValueError("GAIN_VARIANCE_BASE_LABEL_REPLAN_FIELDS_MISSING") from exc
    if not math.isfinite(cpu_seconds) or cpu_seconds < 0 or delay < 0 or not isinstance(raw_cost, Mapping):
        raise ValueError("GAIN_VARIANCE_BASE_LABEL_REPLAN_FIELDS_INVALID")
    return action, cpu_seconds, delay, _build_cost_config(raw_cost, modules)


def run_gain_variance_diagnostic(
    training_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    modules: Mapping[str, Any],
    continuation_policy: Callable[[Any], Mapping[str, Any]],
    record_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    additional_suffixes: int = DEFAULT_ADDITIONAL_SUFFIXES,
    selection_seed: int = 8201,
    resource_check: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Re-evaluate 20 gain-blind train states on eight matched suffixes each.

    ``training_records`` may contain fully loaded bundles with ``row``,
    ``scenario``, ``snapshot_json``, ``label``, and ``suffix_jsons`` fields.
    With ``record_loader``, it may instead contain lightweight collection rows;
    the callback loads only each selected row's immutable artifacts.
    """
    if not isinstance(modules, Mapping) or not all(key in modules for key in ("pairs", "oracle", "scenarios", "env_class")):
        raise ValueError("GAIN_VARIANCE_MODULES_INCOMPLETE")
    if not callable(continuation_policy):
        raise ValueError("GAIN_VARIANCE_CONTINUATION_POLICY_INVALID")
    if type(additional_suffixes) is not int or not 1 <= additional_suffixes <= DEFAULT_ADDITIONAL_SUFFIXES:
        raise ValueError("GAIN_VARIANCE_ADDITIONAL_SUFFIXES_INVALID")
    if sample_size != DEFAULT_SAMPLE_SIZE and sample_size < 1:
        raise ValueError("GAIN_VARIANCE_SAMPLE_SIZE_INVALID")
    selected, selection = select_training_records(
        training_records, sample_size=sample_size, selection_seed=selection_seed,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    state_dir = target / "states"
    state_dir.mkdir()
    summary_rows: list[dict[str, Any]] = []
    try:
        for selected_record in selected:
            if resource_check is not None:
                resource_check()
            bundle = _load_bundle(selected_record, record_loader)
            scenario, snapshot, k2_oracle = _validate_source_bundle(bundle, modules)
            row, label = bundle["row"], bundle["label"]
            action, planner_cpu, action_delay, cost_config = _planner_action_and_cost(label, modules)
            state_id = str(label["state_id"])
            scenario_id = str(label["scenario_id"])
            timepoint = int(snapshot["time"])
            horizon = int(label["horizon_requested_minutes"])
            seeds = list(row["suffix_seed"])
            if len(set(seeds)) != BASE_SUFFIX_COUNT:
                raise ValueError("GAIN_VARIANCE_BASE_SUFFIX_SEEDS_NOT_UNIQUE")
            expected_base_seeds = [
                suffix_seed(scenario_id, state_id, timepoint, index)
                for index in range(BASE_SUFFIX_COUNT)
            ]
            if seeds != expected_base_seeds:
                raise ValueError("GAIN_VARIANCE_BASE_SUFFIX_SEED_SEQUENCE_MISMATCH")

            suffix_jsons = list(bundle["suffix_jsons"])
            extra_suffix_receipts = []
            for suffix_index in range(BASE_SUFFIX_COUNT, BASE_SUFFIX_COUNT + additional_suffixes):
                seed = suffix_seed(scenario_id, state_id, timepoint, suffix_index)
                if seed in seeds:
                    raise ValueError("GAIN_VARIANCE_EXTRA_SUFFIX_SEED_COLLISION")
                events = modules["scenarios"].sample_future_event_suffix(
                    copy.deepcopy(scenario), after_time=timepoint,
                    horizon_minutes=horizon, suffix_seed=seed,
                )
                suffix_json = modules["pairs"].prepare_suffix_snapshot(
                    bundle["snapshot_json"], future_events=events, suffix_seed=seed,
                    env_factory=modules["env_class"],
                )
                prepared = modules["pairs"]._snapshot_payload(suffix_json, modules["pairs"].SUFFIX_SCHEMA)
                suffix_jsons.append(suffix_json)
                seeds.append(seed)
                extra_suffix_receipts.append({
                    "suffix_index": suffix_index, "suffix_seed": seed,
                    "future_event_count": len(events),
                    "future_events_sha256": _digest(events),
                    "suffix_snapshot_sha256": prepared["snapshot_sha256"],
                    "event_tape_sha256": prepared["event_tape_sha256"],
                })

            expected_total = BASE_SUFFIX_COUNT + additional_suffixes
            k8_label = modules["pairs"].run_paired_counterfactual(
                bundle["snapshot_json"], replan_action=action, suffixes=suffix_jsons,
                continuation_policy=continuation_policy,
                replanner_version=str(label["replanner_version"]),
                replanner_cpu_seconds=planner_cpu, cost_config=cost_config,
                horizon_minutes=horizon, action_delay_ticks=action_delay,
                env_factory=modules["env_class"],
            )
            if k8_label.get("suffix_count") != expected_total or len(k8_label.get("pairs", [])) != expected_total:
                raise ValueError("GAIN_VARIANCE_K8_LABEL_PAIR_COUNT_MISMATCH")
            k8_oracle = modules["oracle"].validate_gain_label(k8_label)
            if k8_oracle.get("valid") is not True:
                raise ValueError("GAIN_VARIANCE_K8_ORACLE_VALIDATION_FAILED")
            k2_recomputed = modules["oracle"].recompute_gain_label(label)
            k8_recomputed = modules["oracle"].recompute_gain_label(k8_label)
            k2_summary = _gain_summary(k2_recomputed)
            k8_summary = _gain_summary(k8_recomputed)
            k2_replay_comparisons = _compare_base_suffix_replays(label, k8_label)
            raw_payload = {
                "schema": STATE_SCHEMA,
                "state_id": state_id, "scenario_id": scenario_id,
                "source_row_snapshot_sha256": row.get("snapshot_sha256"),
                "source_label_sha256": label["label_sha256"],
                "k2_oracle": k2_oracle,
                "k2_recomputed": k2_recomputed,
                "k8_label": k8_label,
                "k8_oracle": k8_oracle,
                "k8_recomputed": k8_recomputed,
                "k2_suffix_replay_comparison": k2_replay_comparisons,
                "additional_pairs": copy.deepcopy(k8_label["pairs"][BASE_SUFFIX_COUNT:]),
                "additional_suffixes": extra_suffix_receipts,
            }
            artifact_path = state_dir / f"{state_id}.k8.json"
            artifact_sha = _write_json_exclusive(artifact_path, raw_payload)
            mean_delta = (
                None if not k2_summary["valid_mask"] or not k8_summary["valid_mask"]
                else float(k8_summary["gain_mean"] - k2_summary["gain_mean"])
            )
            variance_delta = (
                None if k2_summary["paired_variance"] is None or k8_summary["paired_variance"] is None
                else float(k8_summary["paired_variance"] - k2_summary["paired_variance"])
            )
            summary = {
                "state_id": state_id, "scenario_id": scenario_id,
                "split": "train", "condition": row.get("condition"),
                "state_snapshot_sha256": snapshot["snapshot_sha256"],
                "source_label_sha256": label["label_sha256"],
                "k2": k2_summary, "k8": k8_summary,
                "mean_delta_k8_minus_k2": mean_delta,
                "variance_delta_k8_minus_k2": variance_delta,
                "mean_sign_changed": (
                    None if mean_delta is None else k2_summary["mean_sign"] != k8_summary["mean_sign"]
                ),
                "k2_suffix_replay_comparison": k2_replay_comparisons,
                "additional_suffix_seeds": [item["suffix_seed"] for item in extra_suffix_receipts],
                "additional_suffixes": extra_suffix_receipts,
                "raw_artifact_path": artifact_path.relative_to(target).as_posix(),
                "raw_artifact_sha256": artifact_sha,
            }
            summary_rows.append(summary)
            if resource_check is not None:
                resource_check()
    except Exception as exc:
        failure = {
            "schema": DIAGNOSTIC_SCHEMA,
            "status": "FAILED",
            "completed_state_count": len(summary_rows),
            "selection": selection,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json_exclusive(target / "FAILURE.json", failure)
        raise

    manifest = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "COMPLETE",
        "protocol": {
            "base_suffix_count": BASE_SUFFIX_COUNT,
            "additional_suffix_count": additional_suffixes,
            "total_suffix_count": BASE_SUFFIX_COUNT + additional_suffixes,
            "shared_keep_replan_suffixes": True,
            "continuation_policy": "same_callback_as_K2_training_label",
            "selection_uses_gain_values": False,
            "selection_unit": "state_id_within_train_split",
            "k2_suffix_replay_note": "K8 re-runs the two saved K2 suffixes; measured safety-monitor CPU is sampled again and can shift the gain through C_replan. Per-suffix CPU and gain deltas are retained.",
        },
        "selection": selection,
        "state_count": len(summary_rows),
        "mean_sign_change_count": sum(row["mean_sign_changed"] is True for row in summary_rows),
        "mean_sign_change_unknown_count": sum(row["mean_sign_changed"] is None for row in summary_rows),
        "states": summary_rows,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    _write_json_exclusive(target / "GAIN_VARIANCE_DIAGNOSTIC.json", manifest)
    return manifest
