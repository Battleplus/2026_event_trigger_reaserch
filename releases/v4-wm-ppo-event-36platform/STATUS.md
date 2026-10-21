# V4 Stage Status

| Stage | Status | Evidence |
|---|---|---|
| Registry freeze | COMPLETE | `package/.v4-final-integration-20261021b/FROZEN_REGISTRY.json` |
| Environment fixture | COMPLETE | `package/.v4-final-integration-20261021b/ENVIRONMENT_CHECK.json` |
| Collection | RUNNING (at least 13/168 scenarios committed at archive snapshot) | `runs/v4-formal-20261021-01/COLLECTION_PROGRESS.jsonl`; `reviews/V4_NEW_DATA_CHECK_20261021.md` |
| World-model training | NOT_RUN | `WORLD_MODEL_RECEIPT.json` |
| PPO training | NOT_RUN | `PPO_RECEIPT.json` |
| Frozen-test evaluation | NOT_RUN | `EVALUATION_RECEIPT.json` |
| Analysis | NOT_RUN | `ANALYSIS.json` |

The isolated PPO CUDA witness passed on RTX 2080 Ti: actor and critic each updated once, frozen fixture world models remained unchanged, and a fresh process restored predictions and optimizer state. Evidence: `evidence/V4_POLICY_GPU_WITNESS.json`. These world models are explicitly untrained seeded fixtures; this is not formal training or evidence of method effectiveness. The checkpoint witness and focused test compare restored model predictions and optimizer state values against CPU references using fixtures; this verifies restore behavior only. Formal world-model and PPO training remain NOT_RUN.

The 90-minute compound diagnostic replay verified 17/17 triggered actions passed pre/post-repair legality checks. Across all 90 minutes, 16 delayed communication dispatches were rejected in nine minutes (15 platform_busy, one task_unavailable); these explain the earlier nine-count diagnostic and are not evidence of illegal actions actually executing. Rejections remain recorded separately. The rollout now records per-minute public queue-feasibility execution evidence, confirmed/unconfirmed dispatches and external-event associations. The archived execution check covers public task-queue feasibility only and does not establish full physical or flight safety. Full PPO training remains NOT_RUN.

World-model evaluation covers one-minute entity deltas and all five ten-minute outcome heads, with valid masks and per-head sample counts. Sealed-test collection remains unrun and is scheduled only after trained policies are frozen. Test raw trajectories and model samples will be saved per scenario. The 97/101-entity batch baseline alignment bug was fixed; grouped and single-batch fixture metrics agree within 1e-6. Evidence: `evidence/V4_TEST_HEADS_INTERFACE_CHECK.json` (untrained fixtures, not prediction-quality results).

Each run uses a dedicated output directory. A stage is complete only when its receipt exists and validates against the frozen configuration and source identities.

The single-scenario collector integration is recorded at `package/.v4-final-integration-20261021c/COLLECT_SCENARIO_INTEGRATION.json` with its per-scenario shard. Formal collection is running against the approved bound review in `runs/v4-formal-20261021-01/environment_review.json` (168 train and validation scenarios, PID 18488; logs: `collect.stdout.log`, `collect.stderr.log`). First scenario completed in 9m21s wall time with 318 samples. Its compressed model-sample shard is 11,817,827 bytes; `_read_samples` took 1.79s and batch-8 collation took 0.48s. This is measurement only; the active collector and frozen data path were not changed.
