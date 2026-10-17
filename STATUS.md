# Status

Current archive snapshot: `R1-sample-scale-only` for the Chapter 2 gain-gate study.

Engineering disposition: **REPAIR_REQUIRED**. The read-only R1 review found the simulator safety-control-link fallback unmet, 221 unresolved safety dispositions, and 107 ordinary paired failures retained as unknown/masked outcomes. The collection completed within its resource limits, and all 3,100 entries in its evidence manifest matched their recorded hashes; hash verification does not override the engineering finding.

Method disposition: **INCONCLUSIVE**. The R1 collection contained 2 positive and 241 non-positive ordinary paired labels, below the registered 30-positive entry requirement. The light model was not trained. No test, OOD, closed-loop effectiveness, world-model, or PPO-transfer result was produced.

The earlier P1 review is limited to smoke-scale pairing and integrity checks. It does not approve the expanded R1 collection or change the R1 disposition. See [the R1 review](reviews/P2_R1_REVIEW_20261017.json) and [the collection diagnostic](reports/P2_R1_COLLECTION_DIAGNOSTIC_20261017.md).

This status describes the `gain-gate-p2-diagnostic-20261017` archive snapshot and the R1 evidence available at that snapshot cut. It is not a release approval.
