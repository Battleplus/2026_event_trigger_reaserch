#!/usr/bin/env python3
"""Run keep/replan suffixes in separate Isaac worker processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from run_capture import run


SUFFIX_SEEDS = (730201, 730202)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_state(index_path: Path, state_id: str):
    index_path = index_path.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    matches = [entry for entry in index["states"]
               if Path(entry["private_file"]).stem == state_id
               or str(entry["sim_time_s"]) == state_id]
    if len(matches) != 1:
        raise ValueError(f"state-id {state_id!r} matched {len(matches)} index entries")
    entry = matches[0]
    private_path = (index_path.parent / entry["private_file"]).resolve()
    if not private_path.is_file() or sha(private_path) != entry["private_sha256"]:
        raise ValueError("private snapshot is missing or its indexed SHA256 differs")
    return index_path, index, entry, private_path


def load_suffix_inputs(config_path: Path, source_root: Path, state_index: Path,
                       state_id: str):
    config_path = config_path.resolve()
    source_root = source_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index_path, index, entry, private_path = select_state(state_index, state_id)
    manifest_path = index_path.parent.parent / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_hash = sha(config_path)
    internal_config_hash = (source_manifest.get("internal_run_config_sha256")
                            or source_manifest.get("config_sha256"))
    if config_hash != internal_config_hash:
        raise ValueError("config SHA256 does not match the source sampling manifest")
    manifest_seed = source_manifest.get("seed", source_manifest.get("config", {}).get("seed"))
    if config.get("seed") != manifest_seed:
        raise ValueError("config seed does not match the source sampling manifest")
    if config.get("run_id") != source_manifest.get("run_id"):
        raise ValueError("config run_id does not match the source sampling manifest")
    if entry.get("continuation_phase") != "post_public_updates_pre_candidate_generation_pre_step":
        raise ValueError("snapshot continuation phase is not supported")
    if int(entry["sim_time_s"]) != int(entry["decision_time_s"]):
        raise ValueError("decision and simulation times differ in state index")
    metadata = {
        "state_id": Path(entry["private_file"]).stem,
        "state_index": str(index_path),
        "state_index_sha256": sha(index_path),
        "state_index_metadata": index,
        "state_index_entry": entry,
        "private_snapshot": str(private_path),
        "private_snapshot_sha256": sha(private_path),
        "config_sha256": config_hash,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha(manifest_path),
        "source_config_sha256": internal_config_hash,
        "source_manifest_config_sha256": internal_config_hash,
        "source_manifest_metadata": source_manifest,
        "scenario_id": config.get("scenario_id"),
        "split": config.get("formal", {}).get("split"),
        "sim_time_s": int(entry["sim_time_s"]),
        "decision_time_s": int(entry["decision_time_s"]),
        "backend_time_s": float(entry["backend_time_s"]),
        "continuation_phase": entry["continuation_phase"],
        "source_root": str(source_root),
    }
    return config_path, source_root, private_path, metadata


def run_suffix_worker(config_path: Path, source_root: Path, state_index: Path,
                      state_id: str, output: Path, suffix_seed: int) -> int:
    config_path, source_root, private_path, metadata = load_suffix_inputs(
        config_path, source_root, state_index, state_id)
    if suffix_seed not in SUFFIX_SEEDS:
        raise ValueError(f"unsupported suffix seed {suffix_seed}; expected one of {SUFFIX_SEEDS}")
    pair_output = output.resolve() / f"suffix_{suffix_seed}"
    pair_output.mkdir(parents=True, exist_ok=False)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    with private_path.open("rb") as stream:
        private_state = pickle.load(stream)
    if not (isinstance(private_state, tuple) and len(private_state) == 2):
        raise ValueError("private snapshot has an unexpected structure")
    return run(config_path, source_root, pair_output, "suffix_pair",
               resume_state=private_state, suffix_seed=suffix_seed,
               suffix_meta=metadata)


def worker_environment():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "1"
    return env


def run_pairs(config_path: Path, source_root: Path, state_index: Path,
              state_id: str, output: Path) -> int:
    config_path = config_path.resolve()
    source_root = source_root.resolve()
    state_index = state_index.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"suffix_{seed}").exists() for seed in SUFFIX_SEEDS):
        raise FileExistsError("suffix output already exists; refusing to overwrite or retry")
    results = []
    for position, suffix_seed in enumerate(SUFFIX_SEEDS):
        start = time.perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--config", str(config_path), "--source-root", str(source_root),
                   "--state-index", str(state_index), "--state-id", state_id,
                   "--output", str(output), "--suffix-seed", str(suffix_seed)]
        completed = subprocess.run(command, check=False, env=worker_environment())
        pair_dir = output / f"suffix_{suffix_seed}"
        receipt_path = pair_dir / "run_receipt.json"
        worker_pid = None
        if receipt_path.is_file():
            try:
                worker_pid = json.loads(receipt_path.read_text(encoding="utf-8")).get("suffix_worker_pid")
            except (OSError, json.JSONDecodeError):
                pass
        results.append({"suffix_seed": suffix_seed, "return_code": completed.returncode,
                        "worker_pid": worker_pid, "output": pair_dir.name,
                        "wall_duration_s": time.perf_counter() - start})
        if completed.returncode != 0:
            break
    receipt = {"schema": "isaac_roi_suffix_workers_v1",
               "status": "complete" if all(item["return_code"] == 0 for item in results) else "failed",
               "state_id": state_id, "workers": results,
               "not_started_suffix_seeds": list(SUFFIX_SEEDS[position + 1:])
                   if results[-1]["return_code"] != 0 else [],
               "separate_processes": all(item["worker_pid"] is not None for item in results)
                   and len({item["worker_pid"] for item in results}) == len(results),
               "retry_policy": "none"}
    (output / "suffix_worker_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--state-id", required=True,
                        help="private snapshot filename stem or simulation time")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suffix-seed", type=int,
                        help="run exactly one suffix worker; omit to launch both workers")
    args = parser.parse_args()
    if args.suffix_seed is not None:
        raise SystemExit(run_suffix_worker(args.config, args.source_root, args.state_index,
                                           args.state_id, args.output, args.suffix_seed))
    raise SystemExit(run_pairs(args.config, args.source_root, args.state_index,
                               args.state_id, args.output))


if __name__ == "__main__":
    main()
