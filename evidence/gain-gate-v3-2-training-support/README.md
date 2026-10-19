# N02 External Training Support

This directory is an external adapter. It is not imported by, copied into, or included in the source binding for either N02 project. An explicit --legacy-source binding is required; this adapter is pending integration review.

Training reads only committed gzip JSONL shards. Gain rows must be wrapper objects with `training_row`; the adapter uses only `training_row.public_features` as model input, and reads the target from a valid oracle receipt cross-checked against the nested label. Forced, invalid, and non-oracle rows are excluded. The gate uses `v3_shards.summarize_gain_states` to deduplicate ordinary `state_id`s and requires at least 30 positive and 30 non-positive states across train/validation, with both classes represented in each split. A failed gate writes `TRAINING_STATUS.json` with `status: not_run`, then raises. No CPU fallback is attempted for a neural run.

The two gain entry points have separate dependency paths:

```powershell
python training_support.py train-cpu `
  --shard-root E:\path\to\committed-run `
  --output-dir outputs\gain-cpu

python training_support.py train-neural `
  --shard-root E:\path\to\committed-run `
  --output-dir outputs\gain-cuda
```

`train-cpu` fits constant and Ridge baselines, and attempts the existing gradient-boosting baseline only when scikit-learn is installed. `train-neural` imports no scikit-learn; it requires visible CUDA device 0 and runs three 64x64 MLP seeds (8201/8202/8203), Adam at 0.001, batch 32, Huber plus the legacy positive-gain auxiliary loss, at most 100 epochs, and patience 10. Each checkpoint stores model and optimizer state. Each bundle is loaded and predicted in a fresh Python process using the legacy runtime restore check.

`event_samples_from_trajectories` builds action-head samples from adjacent `public_before`/`public_after` states in committed trajectory traces. Its one-tick target contract does not consume 30-minute gain labels or old event-oracle horizon labels. The augmented model shares the legacy graph/GRU encoder: its gain head predicts fixed-replanner oracle gain from public history alone, while action heads use the same encoder and adjacent-tick supervision. `train-event-wm` jointly trains both losses, selects checkpoints on the validation-model split, calibrates gain lower bounds on the held-out validation-calibration split, and verifies both checkpoint and optimizer restoration in a fresh CUDA process. The restore command-line paths are internal probe modes; ordinary invocations still require explicit source and shard paths. A failed row gate writes `not_run` before any optimizer is constructed.

```powershell
python training_support.py train-event-wm `
  --shard-root E:\path\to\committed-run `
  --output-dir outputs\event-wm-cuda
```

The WM entry point first applies the gain-state 30/30 gate, then requires nonempty model-fitting, selection and calibration subsets; the 30/30 requirement applies to unique ordinary states globally, not separately to every subset. The executable adapter remains pending review. Action calibration examples are counted and hashed but are optional because calibration currently applies to the gain lower-bound head. Fresh-process checks compare predictions and verify optimizer tensors were restored onto CUDA.

`train_four_variants` in the frozen `event_policy` has a separate input contract: it requires exactly six parent scenario objects split into four `train` and two `development`, with frozen seed/config expectations. The V3 84/84 training and validation rows are not directly consumable by that API. An integration adapter would need to construct the six scenario objects with the required parent/event-tape fields and explicitly map its splits; the WM training rows alone do not supply that interface. The source was inspected read-only; no PPO training was run.

## PPO and Delay Wiring Notes

For the paired PPO comparison, keep one shared ordered training-scenario list for all variants and seeds, initialize each WM/non-WM actor from the same seed-matched actor state, and use one optimizer per policy so the WM auxiliary critic receives gradients while the loaded WM ensemble remains frozen. Preserve the current `event_policy.train_four_variants` checks for paired actor digests and frozen world-model digests; do not let WM predictions select different rollouts between paired arms. Existing `_update_policy` already optimizes the auxiliary critic when present.

The gain counterfactual code accepts `action_delay_ticks`, defaults it to zero, and can only advance whole environment ticks during the compute-wait branch. It cannot represent measured millisecond execution latency without changing the simulation clock and event scheduling to support fractional-time advancement. No delay approximation is introduced here; the exact delay contract remains an integration prerequisite.

No optimizer was run while producing this adapter. CPU checks are limited to syntax/schema and target-shape inspection.

`ppo_support.py` is a separate external adapter for a future explicitly bound V3 source. It validates exactly 84 train and 84 validation parent scenarios, uses paired `Periodic-G`/`Periodic-G-WM` actors for seeds 8201/8202/8203, shared episode order, CUDA rollout/update tensors, frozen augmented WM members, auxiliary-critic gradient audits, update logs, resource telemetry, and fresh-process CUDA checkpoint restore. Its proposed config is [ppo_config.json](ppo_config.json) and remains disabled until the parent freezes it (`freeze_status=frozen`, `training_enabled=true`, and `--enable-training`). The CLI requires `--source`; it has no default legacy source and does not import `event_runner`. The current environment advances integer ticks, so float execution delay requires the separately bound V3.3 delay-integrated source and is not approximated here.

The joint WM constants are fixed before training at hidden size 64, Adam learning rate 0.001, batch size 8, 40 maximum epochs, patience 10, and seeds 8201/8202/8203. Action slot auditing rejects and reports any adjacent action exceeding the frozen six-dispatch or twelve-operation encoder capacity; no truncation is performed. Resource receipts record synchronized wall time, process CPU time, peak CUDA allocated/reserved bytes, GPU model, and CUDA runtime.
