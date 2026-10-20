# V3.3 execution status

Date:2026-10-20 (Asia/Shanghai). Implementation: GPT-6 Luna. Focused validation and fresh read-only source review found execution/label issues; repairs IN_PROGRESS before a new freeze. V3.3 smoke/pilot/formal collection/training NOT_RUN. Protocol:[V33](protocols/V33_CIVILIAN_SEARCH_RESCUE.md). Separate version; no old labels reused. No scientific improvement claimed.

The formal registry was checked as168 train/validation scenarios,12 per condition/split. All conditions share nine regional search arrivals. Runtime cutoffs are removed; resources are recorded only. The CUDA kernel preflight succeeded in an existing dedicated runtime; no experiment optimizer update has occurred. Local measured dependencies are recorded in evidence/v33/LOCAL_RUNTIME.json; inherited requirements are not a verified fresh installation.

Current repair items: preserve the complete remaining exogenous event tape while scoring only30 minutes; propagate verified live safety resolution to branch/label validity; preserve search-to-rescue discovery when cooperative child tasks complete their logical parent. Failed diagnostic attempts are retained. A completed90-minute NOOP is engineering evidence only, not method support.

## Historical V3.2 final status (preserved)

Date: 2026-10-19 (Asia/Shanghai).

| Item | Result |
|---|---|
| Engineering | STOP: smoke completed 7/18; fixed CPU budget exceeded |
| Formal 30/30 data gate | NOT_EVALUATED: formal168 collection NOT_RUN |
| Method | INCONCLUSIVE |
| CPU baselines / GPU training | NOT_RUN |
| World model / PPO | NOT_RUN |

Single new run: v3-2-smoke-20261019. Worker-tree CPU1800.046875 seconds, wall1967.2744005 seconds, working-set metric1,637,158,912 bytes (sum of per-process peak working sets, not simultaneous RSS). No budget increase or repeat smoke.

Focused regressions:17 passed. One90-minute NOOP completed36 initially queued tasks; this missing acceptance was supplemented during smoke and explicitly documented. Original slow-instance action/physical hashes match the90-tick historical reference; no controlled speedup is claimed because the corrected run executes previously failing branches.

All7 committed raw smoke shards were independently recomputed by the coordinator:52 rows,36 forced excluded,2 masked empty windows,14 valid ordinary states,0 positive/14 nonpositive. They remain smoke diagnostics, not formal training data. The fresh GPT-6 Astra read-only review independently checked full arithmetic for the earlier4-shard snapshot, with final stop receipt and7-marker count verified. Its integrity verdict is WARN; smoke acceptance FAIL.

Open scientific issue: completed deliveries can retain positive remaining-work penalty. Arithmetic agreement does not validate that semantics. The inherited smoke flag also accepts structurally valid masked labels; external raw recomputation is required.

GPU preflight found2 RTX2080Ti, but default PyTorch import fails on CUDA dependency resolution. No training/environment modification occurred.

Exact runnable source: releases/gain-gate-v3-2-performance-light. Final93 source-file hashes and7 shard bindings match executed source. Public source/configuration, hashes, safe summaries, resource and review reports are archived; raw large traces are retained locally and excluded from public push.

See reviews/V32_COORDINATOR_DECISION.md and reviews/V32_EXPERIMENT_AUDIT.md. This branch does not change the unrelated Isaac/full-world-model research. No successful stage tag is created.
