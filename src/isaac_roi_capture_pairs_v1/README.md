# Suffix pair runner

`run_suffix_pairs.py` resumes one indexed ordinary Isaac decision snapshot in a fresh process and writes two independent suffix shards (`suffix_730201/` and `suffix_730202/`). Each shard contains `trajectory.jsonl`, `events.jsonl`, `run_receipt.json`, and `manifest.json`; keep and replan restore the same complete backend/common snapshot and are audited with SHA256 restore hashes.

The selected snapshot must come from `state_sampling/state_index.json` and retain continuation phase `post_public_updates_pre_candidate_generation_pre_step`. The config identity is checked against the parent sampling `manifest.json` (`run_id`, seed, and config SHA256). The horizon is `min(1800, 5400-(t-1))`, where `t` is decision time and the backend snapshot is at `t-1`. The first suffix step skips already executed pre-public updates and advances the physical backend once at time `t`; normal pre-public processing resumes at `t+1`.

Only future time-addressed communication dropout draws use the suffix seed. Fault, outage, activation, anomaly, and other disturbance schedules remain frozen by simulation time. Controllers do not receive the seed. Replan performs one measured catalog call and queues its selected action with at least a one-second delay; keep does not call the catalog. No scalar score or pseudo-label is emitted.

Example (requires an Isaac Sim Python environment):

```powershell
python run_suffix_pairs.py --config ..\collection\configs\capture_v1_representative.json `
  --source-root E:\path\to\isaac_source `
  --state-index ..\collection\state_sampling\state_index.json `
  --state-id state_00_001200 --output .\suffix_runs
```

This workspace does not run Isaac during development; no result is claimed until the command completes with `actual_isaac_run: true` in both receipts.

Pure interface checks (no Isaac initialization):

```powershell
python -m pytest -q test_suffix_pairs.py
```

Expected result: `2 passed`. These cover only the pure horizon and time-addressed dropout helpers; they do not validate resume execution or branch behavior.
