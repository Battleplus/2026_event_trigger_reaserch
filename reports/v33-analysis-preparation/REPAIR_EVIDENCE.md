# V33 Analysis Repair Evidence

The analyzer changes are prepared for independent review. The script SHA-256 is recorded in `REPAIR_EVIDENCE.json`.

Validation used the frozen registry API and a single synthetic committed header to exercise registry expectations, identity checking, historical versus unresolved failure handling, and acceptance/rejection of matching versus mismatching frozen bindings. Pure in-memory synthetic rows checked full-row evidence digest behavior; a synthetic route checked recipient, platform-home, donor, other-region, and overlapping-role aggregation. These are functional checks, not experimental data. No real shard was read and no full analysis was run. Python syntax compilation succeeded.

The current script SHA-256 is recorded in the JSON evidence. A tiny synthetic analyzer fixture accepted a matching binding and rejected a mixed-binding batch. The report now states the row-major ordinal-to-contract-region mapping, the nested global-cost JSON path, `source_run_id` lineage, and reused plus newly committed progress counts.

Two temporary-directory fixture attempts hit Windows `PermissionError` during creation and automatic cleanup. The resulting empty directories remain at the paths recorded in the JSON evidence. The later in-workspace synthetic fixture completed and was removed. This evidence does not claim a stage pass; full data and independent review remain outstanding.
