# Lightweight Capture V2

This copy writes lossless delta trajectories for independent paired/suffix
reduction and a public-only representative stream. It does not modify
`formal_collection_v1` or prior outputs.

Paired/suffix `trajectory.jsonl.gz` files use `lightweight_actor_delta_v1` and
live under `labels/pairs/`. Use
`evaluate_suffix_light.py` for suffix reduction; the original evaluator remains
unchanged and expects ordinary JSONL rows. The manifest records the trajectory
encoding for paired/suffix runs. A representative scene contains only
`config.json`, `records.jsonl`, `manifest.json`, and
`labels/state_index.json`, `labels/snapshots/`, and `labels/pairs/`:
the config stores static configuration once, while records stores five-second
samples, actual pre-decision state samples, immediate events, and the final
summary. Per-second physical rows remain in memory for runtime metrics and
labels; representative runs do not persist a per-second trajectory or separate
event log. `config.json` is experiment metadata and stores fixed schedules once;
the policy receives only observations available at decision time. Public state
records omit future schedules and hidden target truth. The old `static_config.json`/`ticks.jsonl.gz` layout
remains readable by conversion tools and is never modified.

Export a completed public capture to the user-facing compact layout without
rerunning the simulator. New captures are copied without creating another
trajectory; old captures are converted through the compatibility reader:

```powershell
python export_records.py --capture-dir <scene> --output-dir <scene>/converted --receipt <scene>/run_receipt.json
```

This writes the frozen scene layout (`config.json`, `records.jsonl`,
`manifest.json`, and `labels/`); it does not include hidden truth, candidate
catalogs, WM outputs, or PPO internals in the public records.

To migrate an old scene directory in place of a batch shell, use
`migrate_scene.py`. It preserves the source. If no legacy public capture exists,
it writes only a `conversion_unavailable` manifest and does not invent public
records.

For a completed archived scene tree, `migrate_batch.py --source-root <old-root>
--output-root <new-root>` applies the same rule once per scene and writes a
`batch_manifest.json`; it does not retry or overwrite a target tree.

Convert one already complete suffix without modifying its source:

```powershell
python compact_existing_suffix.py --suffix-dir <old-suffix> --output <new-conversion-dir>
```

Audit a converted suffix with the unchanged independent reducer:

```powershell
python evaluate_suffix_light.py --suffix-dir <new-conversion-dir> --source-index <state-index.json> --source-root <isaac-source> --collector-root <formal_collection_v1> --evaluator <formal_evidence_v1/evaluate_suffix.py> --output <gain-report.json>
```

After checking the previous output root, launch only scenes whose old output
directory does not exist. The launcher writes into a new output root, records a
failed scene, and continues with later selected scenes. It never retries a
scene or makes replacement seeds:

```powershell
python run_remaining_batch.py --registry <formal_batch_registry.json> --source-root <isaac-source> --old-output-root <previous-batch-output> --new-output-root <new-run-output>
```

The batch receipt lists the original 168 registered scenes, references prior
scene directories, records only newly acquired scenes in this run, and marks
the old/new union reduction as pending. It does not report a terminal 30/30
gate before that union is independently reduced.

The current local checks exercise codecs, candidate-event removal, public
state reconstruction and remaining-scene selection. Isaac execution and
remote deployment are intentionally separate from these checks.

For a deliberately bounded continuation run, use the explicit scene slice
options. A prior output root is only used to skip directories already recorded
by an earlier run; no scene is retried implicitly:

```powershell
python run_remaining_batch.py --registry <formal_batch_registry.json> --source-root <isaac-source> --old-output-root <old-output> --skip-output-root <prior-new-output> --scene-ids-file <slice.txt> --new-output-root <new-slice-output>
```

Routine records omit repeated waypoint send/deliver/apply messages. Only the
first lifecycle event per actual assignment decision and affected platform is
kept; aggregate message counts remain in the final summary. A logging-only
interruption is archived separately and is never described as a failed task.
