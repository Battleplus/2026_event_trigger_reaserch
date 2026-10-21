# V3.3 execution status

## V4 environment/data milestone (2026-10-21)

The separate `archive/v4-wm-ppo-event-36platform` branch contains the frozen V4 environment, data/model/policy runtime source, focused tests, and compact receipts under `releases/v4-wm-ppo-event-36platform/`. Formal collection is in progress with at least 11/168 scenarios committed; raw shards and trajectories are excluded from the archive. The approved review binds the frozen configuration and protocol and covers environment and collection only. Formal world-model training, PPO training, sealed-test evaluation, and analysis remain `NOT_RUN`. `V4_POLICY_GPU_WITNESS.json` is a single update using untrained fixture world models, and `V4_EXECUTION_EVIDENCE_CHECK.json` covers public task-queue feasibility only, not flight safety. V3.3 outcomes below remain historical and unchanged.

Snapshot updated:2026-10-21 (Asia/Shanghai). Implementation: GPT-6 Luna. Protocol: [V33](protocols/V33_CIVILIAN_SEARCH_RESCUE.md). Separate version; no old labels reused. No engineering or safety pass is asserted.

| Stage | Status at archive snapshot |
|---|---|
| Smoke | COMPLETE: 18/18 instances; 278 rows, 45 valid and 233 masked. Engineering evidence only. |
| Formal collection | COMPLETE_168: all 168 frozen scenario identities have committed headers and no scenario is missing. Historical worker failures and retries remain documented in the collection progress records. |
| Formal data gate | NOT MET: 417 evaluable ordinary states (3 positive, 0 zero, 414 negative); all 417 are from 56 `Periodic-4min` scenarios. `Existing-ET` and `Random-Allow-Suppress` each contributed no evaluable ordinary labels. The 30-positive/30-nonpositive threshold and split coverage gate are false. |
| CPU baselines / GPU training | NOT_RUN |
| World model / PPO | NOT_RUN |
| Engineering review | REPAIR_REQUIRED: the 343 unresolved disposition receipts all have `safety_receipt_pre_public_hash_mismatch`; this neither proves violations nor establishes safety. |
| Method interpretation | The data gate does not support fitting or deploying a gain gate. No causal cross-region improvement, method safety, or real-world performance claim is established. |

The 168-marker report is the complete collection snapshot. The earlier 66-marker report (`d91757825888ad6334cf9b7bdbab56f78955066c95da06787f85e2ed1b5329db`) remains a separate historical analysis and its counts are not a recomputation of the full collection. The 16 reused shards retain source-run lineage and marker/content hashes. Full aggregate analysis summaries are public; raw JSONL traces and compressed shard files remain local and are excluded from this archive.

The 343 unresolved disposition receipts are separate from the 2 validation gain labels excluded as unknown: both findings share the `safety_receipt_pre_public_hash_mismatch` reason, but the disposition total is not a count of excluded labels or proven safety violations. Both validation labels are included in the unknown count; none are included among the 417 evaluable ordinary states.

Focused exact-once simulation evidence covers split and cooperate fixtures: one report generation and receipt per mode, no repeated-NOOP discovery events, valid paired labels, and inclusion of the revealed rescue obligation. An independent GPT-6 Astra review is same-family and provisional; the source audit verdict is WARN. This evidence does not establish collection acceptance or method performance. Completed-task residual penalty was independently recomputed as zero in the snapshot report; the earlier residual-cost issue is historical V3.2 context only.

The full report contains per-shard resource receipts and distinguishes them from end-to-end runtime and peak-memory measurements. The 66-marker report's resource figures refer only to that historical snapshot and must not be combined with the full report.

Failure evidence for registry-097 records exit code 1, empty stdout/stderr, both empty-log SHA-256 values, and a redacted job summary. The cause is UNKNOWN; no scientific-failure interpretation is supported by the available evidence. Its scenario was later represented by a committed header in the final collection.

Registry-106 had a `MemoryError` while `json.dumps` serialized a paired counterfactual trace. Its compact redacted traceback excerpt and empty stdout hash are preserved in `evidence/v33/collection-progress/FAILURE_REGISTRY_106.json`; its scenario was later represented by a committed header. These worker-attempt records do not establish experiment-level failure or success.

Analysis preparation artifacts and the final 168-scenario report are distinct evidence. Synthetic repair evidence records `experimental_data=false`, `real_shards_read=false`, and `full_analysis_run=false`; static repair reinspection is limited to the listed repairs. Runtime disposition receipt cross-checks cover 168/168 scenarios, but do not establish all hard constraints or method safety. The support trajectory is one selected case frozen from its generation-time snapshot; later shard additions and the expanded cross-region definition were not globally rescanned, and the case was not selected by gain. It is not a treatment-effect result.

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
