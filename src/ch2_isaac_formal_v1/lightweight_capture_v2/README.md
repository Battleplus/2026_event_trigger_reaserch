# Lightweight Capture V2

This copy writes lossless delta trajectories for independent keep/replan
reduction and a separate public-only representative stream. It does not modify
`formal_collection_v1` or prior outputs.

`trajectory.jsonl.gz` uses `lightweight_actor_delta_v1`. Use
`evaluate_suffix_light.py` for suffix reduction; the original evaluator remains
unchanged and expects ordinary JSONL rows. The manifest records the trajectory
encoding. `public_capture/static_config.json` contains public fixed geometry and
platform parameters, `ticks.jsonl.gz` stores one-second public deltas,
`view_5s.jsonl.gz` stores actual five-second snapshots plus decision timestamps,
and `evaluation_audit.jsonl.gz` stores evaluator-only discoveries. Public files
omit future activation/disturbance schedules and hidden target truth.

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
