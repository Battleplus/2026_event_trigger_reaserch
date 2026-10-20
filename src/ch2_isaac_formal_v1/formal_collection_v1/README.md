# Formal Collection v1

This directory contains the frozen collection code and the generator for a 168-scene registry. The registry covers seven conditions (six single-mechanism conditions and `composite_all_enabled`), with 12 scenes per condition in each of `train` and `validation`. Each condition/split cell has four scenes per trigger source: `Existing-ET`, `Periodic-4min`, and `Random-Allow-Suppress`. Seeds and output paths are unique. Conditions, target count, timing values, and source allocation are fixed design values; no pilot was run to select them, and they do not represent an empirically optimal configuration.

## Prepare and run

Run the registry generator once after the source files are frozen:

```powershell
python prepare_formal_registry.py
```

It writes `formal_batch_registry.json`, `formal_registry_sha256.json`, and one config per scene under `configs/scenes/`. The batch runner validates the registry, config identities, grid balance, and recorded SHA256 values before starting any Isaac process. Regenerate the SHA manifest after any intentional code or config change.

Run the batch with the Isaac Python runtime that can import the required Isaac Sim packages. `--registry-file` must name the generated registry; `--source-root` must name the trusted source tree used by the backend and snapshot pickle classes; `--output-root` must be a new, empty directory on the collection volume.

```powershell
python run_formal_batch.py `
  --registry-file .\formal_batch_registry.json `
  --source-root E:\path\to\trusted\isaac\source `
  --output-root E:\path\to\collection\formal-v1
```

The registry requests one GPU worker and each child process is limited to one CPU thread. The batch proceeds directly through the 168 registered scenes; there is no separate representative scene or single-state suffix preflight gate. Each formal scene runs a 90-minute state-sampling acquisition episode (the collector's internal phase label is `representative`), then runs up to eight ordinary indexed states with suffix seeds `730201` and `730202` in separate child processes. It invokes the adjacent `formal_evidence_v1/evaluate_suffix.py` evaluator once for each complete suffix and then runs the aggregate reducer.

## Failure and outputs

There is no retry or replacement sampling. A representative failure skips that scene's suffixes. A suffix failure stops the remaining states for that scene. The scene is recorded as failed or partially failed, and the batch proceeds to the next registry scene. A storage-capacity exception is handled as a failure of the current scene; it does not trigger a retry. Confirm sufficient free space before starting. The current remote estimate is about 114 GiB available, which is a planning constraint rather than a guarantee of the final collection size.

Each scene directory contains its representative and suffix artifacts plus `scene_receipt.json` when the output volume permits writing it. `batch_events.jsonl` is appended after each scene and `batch_receipt.json` summarizes all scenes and the final reduction status. Raw trajectories are written directly as `trajectory.jsonl.gz` using gzip compression level 1; no uncompressed trajectory copy is retained. Evaluator outputs are written beneath `reductions/<run_id>/<state_id>/suffix_<seed>/reduction.json`, with the aggregate at `aggregate.json` in the output root.

The registry starts with `execution: NOT_RUN`; this repository state records a plan and does not assert that the formal capture, a pilot, or model training has run. Training is outside this collection runner.
