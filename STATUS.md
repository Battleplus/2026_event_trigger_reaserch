# V3.2 Full World-Model Study Status

Date: 2026-10-19 (Asia/Shanghai).

- Protocol: `protocols/FULL_WORLD_MODEL_V3_2.md`. Fixed CPU/GPU time caps removed by user instruction; memory and actual fault handling remain.
- New source snapshot: `releases/gain-gate-v3-2-full-world-model/`, with per-file hashes. This snapshot is not an acceptance or a training result.
- V3.2 smoke: STOPPED FOR REPAIR after 4/18 committed instances. Independent replay showed the periodic/local scope filtered every non-NOOP alternative although the same public state had 15 legal global alternatives. This is an implementation defect; these four shards remain diagnostic only and do not count toward smoke acceptance.
- Two pairing defects repaired: stale derived scenario hash after future-suffix substitution; ambiguous continuation receipt/action envelope.
- Necessary local checks: 20 passed, 6 temporary-directory fixture cases deselected due to host permissions; separate real shard commit/readback succeeded.
- Server GPU1: real small CUDA forward/backward probe passed. No new model training or checkpoint exists yet.
- Formal 168-instance collection: NOT RUN. Data gate: NOT EVALUATED. Models/PPO: NOT RUN. Scientific conclusions: INCONCLUSIVE.
- Required repair before a new smoke: local-to-global candidate fallback, canonical fractional command-delay integration, and a new source binding. V3.2 is not eligible for formal collection.
- Execution-delay semantics are being developed separately; current minute-clock smoke is engineering evidence only and cannot establish subminute decision latency benefits.
- Archive push and remote ref readback succeeded: f3ac0913d95fa338a82d858c9498c4c5143b99c8 on archive/gain-gate-v3-2-full-world-model. These artifacts document a repair, not study acceptance.

Historical status below is preserved verbatim and is not the current execution status.

---

# Historical V3 Archive Status

Date: 2026-10-18 (Asia/Shanghai).

- Milestone: protocol baseline, implementation in progress.
- V3 engineering smoke: NOT PASSED / NOT REVIEWED. First real execution attempt failed before reset at legacy validator dispatch; repair is in progress.
- V3 formal data: NOT RUN.
- V3 30/30 training gate: NOT EVALUATED.
- Light model / world model / PPO: NOT RUN.
- Method: INCONCLUSIVE.

Only `protocols/v3-baseline/` and these status documents describe V3 in this commit. Other existing tracked source and evidence are historical V2 artifacts, retained without a V3 success claim. Event-generator details remain provisional until sampler implementation and config-hash review.

The V2 interrupted CPU total remains unknown; 1890.72 seconds was a lower bound. No complete historical resource settlement is fabricated. V3 will use separate measured receipts and cannot claim that the original project budget has been reset.

Coordinator probes are archived in `reviews/v3-in-progress/`. Region helper checks passed only in their stated scope. Current baseline regression rerun: 124 passed / 9 failed (four missing-dependency failures, five legacy review-fixture failures). No complete V3 runtime or scientific outcome is claimed.
