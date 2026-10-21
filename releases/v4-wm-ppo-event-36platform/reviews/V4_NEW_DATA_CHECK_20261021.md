# V4 new-data check

Reviewer: Codex primary coordinator. Read-only inspection and recomputation; no production source was modified during this check.

At the check, PID 18488 was live and 11 scenarios were committed. The latest completed training scenario was `v33-train-none_or_light-search_rescue-n3-010-00008211`.

This scenario contains three 90-minute trajectory sources and 24 sampled windows. All 57 branches share their respective window's starting truth and public history, retain ten-minute horizons, and use NOOP continuation after the first action. Branch actions are unique within each window, including NOOP. Public histories end no later than the decision minute.

All 840 stored trajectory/branch step rewards were independently recomputed from raw events with the existing event oracle. All 57 branch cost records were recomputed from raw traces and endpoints. These values matched the stored results. Corresponding action legality and live safety-disposition receipts were valid.

Result: the inspected scenario passed these data checks. This does not accept the remaining dataset or establish prediction, policy, safety across scenarios, or method effectiveness. No sealed-test data was read and no full source review was repeated.
