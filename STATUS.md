# V3.3 execution status

Snapshot updated:2026-10-21 (Asia/Shanghai). Implementation: GPT-6 Luna. Protocol: [V33](protocols/V33_CIVILIAN_SEARCH_RESCUE.md). Separate version; no old labels reused. No scientific improvement claimed.

| Stage | Status at archive snapshot |
|---|---|
| Smoke | COMPLETE: 18/18 instances; 278 rows, 45 valid and 233 masked. Engineering evidence only. |
| Formal collection | ACTIVE / INCOMPLETE: 109/168 scenarios committed at the 2026-10-21 snapshot (93 new plus 16 lineage-verified reused); 59 pending, 2 worker failures recorded. Worker PID 53708 was live at snapshot time. |
| Formal data gate | NOT MET in the 66-marker analysis snapshot: 164 evaluable ordinary states, 2 positive and 162 nonpositive; 30/30 positive/nonpositive and split coverage gates are not met. |
| CPU baselines / GPU training | NOT_RUN |
| World model / PPO | NOT_RUN |
| Method conclusion | INCONCLUSIVE; no scientific performance claim. |

The collection snapshot and the conclusion report have different cutoffs: the public report is bound to 66 commit markers (`d91757825888ad6334cf9b7bdbab56f78955066c95da06787f85e2ed1b5329db`); the collection snapshot later advanced to 109/168. Do not read the report counts as a recomputation of the later collection state. The 16 reused shards retain source-run lineage and marker/content hashes. Raw JSONL traces and compressed shard files are not included in this public archive.

Focused exact-once simulation evidence covers split and cooperate fixtures: one report generation and receipt per mode, no repeated-NOOP discovery events, valid paired labels, and inclusion of the revealed rescue obligation. An independent GPT-6 Astra review is same-family and provisional; the source audit verdict is WARN. This evidence does not establish collection acceptance or method performance. Completed-task residual penalty was independently recomputed as zero in the snapshot report; the earlier residual-cost issue is historical V3.2 context only.

Collection snapshot resource receipts: 93 newly committed shards report summed process CPU 53,795.563 seconds and summed per-shard wall time 60,641.739 seconds; maximum receipt-time RSS is 1,252,184,064 bytes. These per-shard sums are not an end-to-end run wall-clock receipt or peak-memory measurement. The 66-marker analysis snapshot reports planner CPU 2,486.875 seconds and committed-shard process CPU 29,163.594 seconds; peak memory is UNKNOWN. These are separate cutoffs and must not be combined as one analysis.

Failure evidence for registry-097 records exit code 1, empty stdout/stderr, both empty-log SHA-256 values, and a redacted job summary. The cause is UNKNOWN; no scientific-failure interpretation is supported by the available evidence.

Registry-106 exited with a `MemoryError` while `json.dumps` serialized a paired counterfactual trace. Its compact redacted traceback excerpt and empty stdout hash are preserved in `evidence/v33/collection-progress/FAILURE_REGISTRY_106.json`; it has not been rerun. These worker failures do not establish experiment-level failure or success.

Analysis preparation artifacts are archived separately from experimental results. Synthetic repair evidence records `experimental_data=false`, `real_shards_read=false`, and `full_analysis_run=false`. The first fresh same-family Astra review was provisional WARN; the later static repair reinspection was provisional PASS limited to the listed repairs. Neither review approves the unfinished 168-scenario experiment. The support trajectory is one selected case frozen from the generation-time committed snapshot; later shard additions and the expanded cross-region definition were not globally rescanned, and the case was not selected by gain. It is not a treatment-effect result.

The formal registry was checked as168 train/validation scenarios,12 per condition/split. All conditions share nine regional search arrivals. Runtime cutoffs are removed; resources are recorded only. The CUDA kernel preflight succeeded in an existing dedicated runtime; no experiment optimizer update has occurred. Local measured dependencies are recorded in evidence/v33/LOCAL_RUNTIME.json; inherited requirements are not a verified fresh installation.

Focused validation and fresh read-only source review found execution/label issues before the collection freeze. A completed 90-minute NOOP is engineering evidence only, not method support.

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
# V3.3 full collection conclusion (2026-10-21)

The frozen 168-scenario collection is complete and independently analyzed. The result is `COMPLETE_168_GATE_NOT_MET`: 2,880 unique states yielded 417 evaluable ordinary states, only 3 positive gains and 414 negative gains, so the 30/30 training gate was not met. Cross-region actions were generated and executed, but the 12 ordinary selected cross-region states had descriptive global `J_total` replan-minus-keep of +0.184677; this is not a causal cross-region estimate and does not support a global effectiveness claim. Gain models, GPU MLP, world model, and PPO were not run. See `reports/v33-conclusion-final-168/FINAL_CONCLUSION.md`.
