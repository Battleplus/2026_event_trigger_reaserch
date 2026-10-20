#!/usr/bin/env python3
"""Resume trusted ordinary Isaac snapshots as two-seed keep/replan suffix pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
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
    return index_path, entry, private_path


def run_pairs(config_path: Path, source_root: Path, state_index: Path,
              state_id: str, output: Path) -> int:
    config_path = config_path.resolve()
    output = output.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    index_path, entry, private_path = select_state(state_index, state_id)
    manifest_path = index_path.parent.parent / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_hash = sha(config_path)
    if config_hash != source_manifest.get("config_sha256"):
        raise ValueError("config SHA256 does not match the source sampling manifest")
    if cfg.get("seed") != source_manifest.get("config", {}).get("seed"):
        raise ValueError("config seed does not match the source sampling manifest")
    if cfg.get("run_id") != source_manifest.get("run_id"):
        raise ValueError("config run_id does not match the source sampling manifest")
    if entry.get("continuation_phase") != "post_public_updates_pre_candidate_generation_pre_step":
        raise ValueError("snapshot continuation phase is not supported")
    if int(entry["sim_time_s"]) != int(entry["decision_time_s"]):
        raise ValueError("decision and simulation times differ in state index")
    # The snapshot contains source-backed objects whose modules must be importable
    # before unpickling. Add the explicitly selected Isaac source tree first.
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    # Pickle is executable input. This runner accepts only the indexed private file
    # from the explicitly selected sampling directory.
    with private_path.open("rb") as stream:
        private_state = pickle.load(stream)
    if not (isinstance(private_state, tuple) and len(private_state) == 2):
        raise ValueError("private snapshot has an unexpected structure")
    output.mkdir(parents=True, exist_ok=True)
    for suffix_seed in SUFFIX_SEEDS:
        pair_output = output / f"suffix_{suffix_seed}"
        pair_output.mkdir(parents=True, exist_ok=False)
        meta = {"state_id": state_id, "state_index": str(index_path),
                "private_snapshot": str(private_path), "private_snapshot_sha256": sha(private_path),
                "config_sha256": config_hash, "source_manifest": str(manifest_path),
                "source_config_sha256": source_manifest["config_sha256"],
                "sim_time_s": int(entry["sim_time_s"]),
                "backend_time_s": float(entry["backend_time_s"]),
                "continuation_phase": entry["continuation_phase"]}
        rc = run(config_path, source_root, pair_output, "suffix_pair",
                 resume_state=private_state, suffix_seed=suffix_seed, suffix_meta=meta)
        if rc != 0:
            return rc
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--state-id", required=True, help="private snapshot filename stem or simulation time")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_pairs(args.config, args.source_root, args.state_index,
                               args.state_id, args.output))


if __name__ == "__main__":
    main()
