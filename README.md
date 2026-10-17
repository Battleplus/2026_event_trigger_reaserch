# Event-Triggered Gain-Gate Study

This repository contains the Chapter 2 simulation implementation, frozen protocol materials, review records, and a bounded R1 collection diagnosis for real-time restructuring of heterogeneous disaster-relief UAV task assignments. The current snapshot is diagnostic evidence, not a validated method result.

## Current Result

The bounded R1 collection completed for 168 scenario instances across seven conditions and train/validation splits. It yielded 243 valid ordinary paired labels: 2 positive and 241 non-positive. In 229 of those states the fixed replanner offered only NOOP. The R1 review records engineering **REPAIR_REQUIRED** because the simulator safety-control-link fallback remains unmet and 221 safety dispositions remain unresolved. There are also 107 ordinary paired failures retained as unknown/masked outcomes. The method remains **INCONCLUSIVE** because the registered light-model entry threshold was not met. No light model, test/OOD run, closed-loop effectiveness comparison, event-aware world model, or PPO transfer was run. See [STATUS.md](STATUS.md), [the review](reviews/P2_R1_REVIEW_20261017.json), and [the diagnostic](reports/P2_R1_COLLECTION_DIAGNOSTIC_20261017.md).

## Runtime

The measured collection runtime was Python 3.14.4 on single-thread CPU with CUDA disabled: NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.7.0, and PyTorch 2.13.0+cpu. Pytest 9.0.3 was used for the package tests. See [requirements.txt](requirements.txt).

From the repository root, install the declared dependencies and run the package tests:

```powershell
python -m pip install -r requirements.txt
$testFiles = Get-ChildItem package/tests/test_*.py | ForEach-Object { $_.FullName }
python -m pytest $testFiles
python -m unittest discover -s staging_tools -p "test_*.py"
```

The runner exposes its supported stages and options with:

```powershell
python -B package/civilian/gain_runner.py --help
```

For a bounded collection reproduction in a fresh checkout, use a new output directory and the reviewed P1 binding-fix record:

```powershell
python -B package/civilian/gain_runner.py --stage collect --config configs/GAIN_GATE_CONFIG.json --output outputs/r1-reproduction-run-001 --p1-review reviews/P1_ARCHIVE_REPRODUCTION_REVIEW.json --limit-instances-per-split 12 --conditions none_or_light task_update_or_arrival platform_failure_or_capability_loss energy_anomaly communication_degradation environment_or_obstacle_change compound --splits train validation
```

This command reproduces the bounded sampling scale, not the original run identity. The runner requires a new, nonexistent output directory and preserves failed run evidence. Do not run `train-light` on the current R1 data: its registered positive-label requirement fails.

## Contents and Provenance

`package/` contains the simulator, event-trigger code, gain-gate implementation, fixed public-C1 catalog replanner, independent consequence oracle, and tests. `configs/` and `protocols/` contain the environment and frozen study contracts. `reviews/` records coordinator decisions; `reports/` contains public-safe R1 summaries. `staging_tools/prepare_public_archive.py` defaults to a read-only preview and has focused tests for its source allowlist and evidence redaction. The archived review and report files are the redacted public projections; their source and projection hashes are recorded in `evidence/public_projection/REDACTION_MAP.json`. Raw run outputs, paired trajectories, checkpoints, dependency caches, and local databases are excluded.

The code derives from the N02 disaster-relief and event-trigger source identified in [BASELINE_PROVENANCE.json](BASELINE_PROVENANCE.json). The Chapter 1 adapter uses public C1 information; it is not the user's complete Chapter 1 solver. `REPO_AUDIT.md`, `REUSE_INVENTORY.json`, `BASELINE_PROVENANCE.json`, `SOURCE_FILE_INVENTORY.json`, `SOURCE_BOUNDARY_AUDIT.json`, and the earlier P1/P2 reviews are historical P0/P1 or preceding-pilot snapshots. In particular, `REPO_AUDIT.md` predates R1 and its no-collection statement is superseded by this README, [STATUS.md](STATUS.md), and the controlling R1 review. These historical records are not a current R1 boundary audit. `PUBLIC_ARCHIVE_MANIFEST.json` and `SHA256SUMS.txt` bind the exact files in this archive.

No license grant has been verified for all inherited source and fixtures. No `LICENSE` file is included; public visibility of this archive is not a permissive license grant. Review component provenance and licensing before reuse or redistribution.
