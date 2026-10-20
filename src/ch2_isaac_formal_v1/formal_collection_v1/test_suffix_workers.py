from __future__ import annotations

import json
import hashlib
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import run_suffix_pairs


def test_suffix_metadata_binds_source_config_scenario_and_split(tmp_path):
    source = tmp_path / "source"
    sampling = source / "state_sampling"
    sampling.mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config = {"run_id": "formal-run", "scenario_id": "condition-seed1",
              "seed": 1, "formal": {"split": "train"}}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (source / "manifest.json").write_text(json.dumps({
        "run_id": "formal-run", "config": config, "config_sha256": config_hash,
    }), encoding="utf-8")
    snapshot = sampling / "state_00_000240.pkl"
    snapshot.write_bytes(pickle.dumps(({}, ())))
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    state_index = sampling / "state_index.json"
    state_index.write_text(json.dumps({"states": [{
        "private_file": snapshot.name, "private_sha256": snapshot_hash,
        "sim_time_s": 240, "decision_time_s": 240, "backend_time_s": 239.0,
        "continuation_phase": "post_public_updates_pre_candidate_generation_pre_step",
    }]}), encoding="utf-8")

    _, _, _, metadata = run_suffix_pairs.load_suffix_inputs(
        config_path, source, state_index, "state_00_000240")

    assert metadata["source_config_sha256"] == config_hash
    assert metadata["source_manifest_config_sha256"] == config_hash
    assert metadata["scenario_id"] == "condition-seed1"
    assert metadata["split"] == "train"


def test_suffix_seeds_are_dispatched_as_separate_single_seed_processes(tmp_path, monkeypatch):
    output = tmp_path / "state_00_000240"
    commands = []
    real_run = subprocess.run

    def fake_run(command, check, env):
        commands.append(command)
        seed = int(command[command.index("--suffix-seed") + 1])
        pair_dir = output / f"suffix_{seed}"
        pair_dir.mkdir(parents=True)
        probe = real_run([sys.executable, "-c", "import os; print(os.getpid())"],
                         check=True, capture_output=True, text=True)
        (pair_dir / "run_receipt.json").write_text(
            json.dumps({"suffix_worker_pid": int(probe.stdout.strip())}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_suffix_pairs.subprocess, "run", fake_run)
    rc = run_suffix_pairs.run_pairs(Path("config.json"), Path("source"),
                                    Path("state_index.json"), "state_00_000240", output)
    receipt = json.loads((output / "suffix_worker_receipt.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(commands) == 2
    assert [int(command[command.index("--suffix-seed") + 1]) for command in commands] == [730201, 730202]
    assert len({worker["worker_pid"] for worker in receipt["workers"]}) == 2
    assert receipt["separate_processes"] is True


def test_first_suffix_failure_stops_second_worker_without_retry(tmp_path, monkeypatch):
    output = tmp_path / "failed_state"
    commands = []

    def fake_run(command, check, env):
        commands.append(command)
        seed = int(command[command.index("--suffix-seed") + 1])
        pair_dir = output / f"suffix_{seed}"
        pair_dir.mkdir(parents=True)
        (pair_dir / "run_receipt.json").write_text(
            json.dumps({"suffix_worker_pid": 4321}), encoding="utf-8")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(run_suffix_pairs.subprocess, "run", fake_run)
    rc = run_suffix_pairs.run_pairs(Path("config.json"), Path("source"),
                                    Path("state_index.json"), "state_00_000240", output)
    receipt = json.loads((output / "suffix_worker_receipt.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert len(commands) == 1
    assert receipt["not_started_suffix_seeds"] == [730202]
    assert receipt["retry_policy"] == "none"
