# V3.3 analysis preparation review

Date: 2026-10-21. Reviewer: fresh GPT-6 Astra ultra, read-only. `review_independence=same-family`, `acceptance_status=provisional`. Overall **WARN**. Evaluation type: `simulation_only`. This review does not approve the unfinished 168-scenario experiment.

The reviewer inspected analysis and plotting code, independent oracle/cost/replanner/shard/region source, the 66-r2 partial report, repair evidence, and small plotting artifacts. No large raw shard was opened or analyzed. Pure-memory checks and artifact hashes were checked. No files were changed by the reviewer.

## Findings

- GT provenance PASS at source level: `counterfactual_pairs.py:860,1022` executes paired restored simulations. `gain_oracle.py:520,700,869` independently rebuilds costs, obligations and keep-minus-replan gain, without importing the production scorer. This does not prove that every collected trajectory has already passed final recomputation.
- Future-information isolation PASS in inspected paths: `fixed_replanner.py:96,108` ranks current public information; `event_environment.py:396,463` excludes future tape and identity from the public projection. Tape IDs in receipts are audit metadata, not observed ranking inputs.
- Normalization PASS: `cost_accounting.py:64` and oracle use common obligation priority mass, not model-output statistics.
- Statistical unit PASS with a duplicate-record WARN: analyzer uses unique ordinary valid state IDs and scenario means; forced/empty/unknown are excluded, suffixes are not independent samples, n<2 has no CI. Selected-cross intervals are explicitly descriptive.
- Cost accounting PASS in inspected source: switch/scale deduplication, planner/safety CPU separation, and NOOP initial invocation cost are included. NOOP/non-NOOP output is separate.
- Registry/failure preparation PASS: frozen 168 identities are compared by scenario and header; historical failure records parse scenario IDs and distinguish subsequently committed results. The repair fixture is synthetic functionality evidence, not experiment data.
- Current partial claims PASS internally: 66-r2 explicitly reports 66/168, 1007 unique states, 164 ordinary valid, 2 positive/162 negative, 839 forced and 4 empty. It is an old snapshot, not evidence that the newly repaired analyzer was run.
- Plot artifact binding PASS: PNG/SVG hashes match JSON; marker hash/compressed hash/scenario hash match source references. Thirty public points t9–38 agree with the image. Raw-shard pointwise agreement and global first-case selection were not independently rerun.

## Required small repairs

1. Region naming: object IDs use ordinal 00..08, while the region contract uses r00/r01/r02/r10/r11/r12/r20/r21/r22 (`v33_scenarios.py:68,76,92`; `region_contract.py:25`). Geometry and cross-region counts are unchanged by this bijection; labels must disclose it.
2. Plot generator: `trace_nodes` includes truth endpoints but `choose` counts all nodes for its two-public-point condition. Pure-memory test showed zero public points plus two truth endpoints passes. Count public-observation nodes only. `is_cross` also omits a donor-home transfer into the task's home region; align definitions or narrow the stated selection population. The present insert/30-public-point case is unaffected by these boundary issues.
3. Analyzer duplicate rows: only scenario/gain/valid/reason were compared and counters continued per row (`tools/v33_conclusion_analysis.py:389`). For identical complete evidence, skip the repeated row; reject conflicting state evidence. No existing duplicate inflation was established: old snapshot ranking checks and unique states both equal 1007.
4. Bind final shards to frozen source/config: `_read_shard` must pass the existing expected binding to `verify_scenario_shard` (or reuse bound `verify_run`). Internal container hashes/registry alone do not exclude another internally consistent source/config. No actual contamination was found.
5. Correct the narrative JSON path to `outcome_decomposition.global_cost_component_means`; the values exist and are not phantom results.

Do not add a generic defensive framework or rerun large analyses during collection. Repair these specific reporting/statistical issues, then run final recomputation on the completed, bound registry and perform a final audit.

## Key audited hashes (SHA-256)

```text
tools/v33_conclusion_analysis.py CEA1FEF5EA6CDA1873C30C1E9E806D45B9B13D6FDEF98209AA66933A6107E59B
tools/v33_support_plot.py C2473DA52E00D970080974CB1D04E645B8FDC661BEDEA00E8265801F8D69D2B4
package/civilian/gain_oracle.py 604BC620685A971D71BF9E922B805E7CD48B820196E90AEABFF7554AF59A284B
package/civilian/cost_accounting.py D38432191FCF09DFC367FF7DA3A8664D9EB682AD359D89CFD8C50B5FF5FE5E13
package/civilian/fixed_replanner.py 585868146DAF30BFFD62BBEADCC6DC9DCB1AA537B3E7F62BBE62EFE3F8B04735
package/civilian/v3_shards.py 2DDDCAFDB79470251C82E9804F50E4DBA46E34AB67C7D79FCB103FB9014D33C5
package/civilian/region_contract.py 6E0160E17F91AF334EDCAF1963CA3A5E8B4469CBC6CA88DF9F9690E7C8D9BB63
reports/v33-analysis-preparation/REPAIR_EVIDENCE.json B6600676F758BC3DFC051A5A77DF8CF46C0CC2B24D7C6DEF5EDD32B070045B31
reports/v33-support-trajectory/data.json 005EF4D4E0222767C079EAD5EFD870974E5753B5E683271319028B9786D41692
reports/v33-support-trajectory/trajectory.png A2EC2051208EACB9B6EC5A62426A8E3CF1E937CBBD2357815555D6BF4780B3F9
reports/v33-support-trajectory/trajectory.svg 11310DA00BF80BC5B0385BB4E2FE46FE5599E62AB3967D5E969B3487B1957E21
reports/v33-conclusion-snapshot-66-r2/conclusion.md 8041B0D95E71E17E64901B3C7B6E1B1F76ED4C28FEB31E64C91D49A3EFCB6145
reports/v33-conclusion-snapshot-66-r2/summary.json 42E86D92FF7F60A44D5B5E797752E6898A6E9D62D080B26DB1299A3BC4B807BB
configs/GAIN_GATE_CONFIG.json 08C9C523F3301C1235CC35C7D87BCFB04B9AFDDA4D0578A6A06061960ED85859
```

Reviewer task: `/root/v33_analysis_artifact_review`. This document faithfully records the returned findings; the original tool response remains in the task conversation. Future repairs require new hashes and reinspection, not rewriting this historical verdict.
