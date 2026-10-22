# Light models v1

This directory contains the guarded, CPU-first model interface for the frozen
`formal_collection_v1` Isaac study. It does not collect data and it does not
read gain labels until the batch is terminal.

`train_models.py --batch-output <root> --output-dir <run-dir>` first reads
`<root>/batch_receipt.json`. The receipt must be terminal and must contain 168
unique scene attempt records. Only then does the entry point read the canonical
`<root>/aggregate.json`, and it reads only the aggregate coverage summary at the
gate. A missing or insufficient summary produces a `NOT_RUN` receipt. The
`end_reduction.status` field is not used as the sole gate.

After the gate passes, `dataset.py` uses each aggregate state's
`source_reports` path relative to `<root>/reductions` to bind one scene and local
state to `scenes/<scene>/representative/state_sampling/<local_state>.public.json`.
It checks the state index's public hash and never opens private snapshots or
future/truth fields. Invalid labels and missing or hash-mismatched projections
are excluded and the 30/30 plus split coverage gate is recomputed on the
remaining feature rows.

Features are fixed-width numeric summaries of the public projection: task
queue/activation counts, platform availability/fault/hold status by type and
home zone, capacity-normalized energy, velocity and centroid RMS dispersion,
and public-observation counts, zones, types, and freshness. Aircraft, task, and
target identifiers are used only for binding or counting and are not feature
values. The implementation uses one-pass summaries and no pairwise distance
matrix.

The entry point includes a NumPy constant baseline and ridge model. ExtraTrees
is optional and reports `TREE_NOT_RUN_DEPENDENCY_MISSING` when sklearn is absent;
it is never installed automatically. The 64x64 MLP requires explicit selection
of the frozen physical GPU UUID and uses its remapped `cuda:0` device; it
reports an explicit unavailable status rather than falling back to CPU. Neural
repeats use seeds 8201, 8202, and 8203.

The MLP launcher must expose only the physical study GPU before starting the
process, for example (using the frozen UUID):

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e"
python -B -m light_models_v1.train_models --batch-output <root> --output-dir <run-dir>
```

The process then uses its remapped `cuda:0`; an unset selector, a physical GPU
0 selector, or a multi-device list yields an explicit `NOT_RUN` model status.

Validation scenarios are assigned as whole groups to `validation_selection`
or `validation_calibration`; states and suffix reports from one scenario are
never split across those roles. Epsilon is selected on selection scenarios by
scenario-macro decision regret over the fixed finite grid in `selection.py`.
Equal scores keep the first grid value. Calibration scenarios are used only for
the lower one-sided target-minus-prediction residual order statistic at alpha
0.1. Each scenario contributes its minimum residual, so repeated states do not
inflate the calibration sample count; calibration never alters epsilon. The
finite-sample rank is `floor((n_scenarios + 1) * alpha)`. When this is below 1,
lower-bound calibration is `NOT_RUN` and no lower-bound deployment mode is
emitted. A lower-bound prediction is `mean_prediction + q` when calibration is
available.

Separately, `training_grouped_cv` assigns training scenarios to three fixed
folds and fits the constant and Ridge baselines on the other two folds. It
records fold membership and held-out metrics and never uses validation rows;
these diagnostics are not validation selection or calibration results. MLP
training remains at the configured three seeds without additional fold fits.

The training receipt records per-scenario selection diagnostics and scenario
identities. Constant, Ridge, ExtraTrees, and each configured MLP seed emit model
artifacts with the selected epsilon, calibrated q, feature schema, and dataset
identity. Use `--deployment-mode mean` or `--deployment-mode lower_bound` with
`closedloop_v1.runtime --checkpoint <artifact>` to select the prediction mode;
the runner applies the artifact epsilon as the gain gate threshold. Constant and
Ridge artifacts are JSON, ExtraTrees artifacts are trusted pickle files, and
MLP artifacts are PyTorch checkpoints. This defines deployment plumbing only;
no real collection, holdout, or Isaac performance result is implied.

Only interface tests use synthetic projections and receipts:

```powershell
python -B -m unittest test_light_models -v
```

Those tests are not Isaac runs and do not establish model performance.
