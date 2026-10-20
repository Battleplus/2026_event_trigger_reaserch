# Pilot readiness review — 2026-10-20

Status: NOT_RUN / not yet accepted for execution. attempt008 representative and pair validation is a prerequisite. Its upload permission is pending; this document does not authorize transfer or claim a live remote job.

## Sampling order

The current objective requires the ordinary-state 30-positive/30-nonpositive gate BEFORE the 168-scene formal collection. Therefore the 36-scene pilot must supply the initial gain diagnosis. One pair per pilot scene could supply at most 36 unique states and cannot meet 60 states. Pilot sampling must instead select at most eight ordinary candidate states per trajectory, without reading future gain, and evaluate two paired suffixes per selected state. These suffixes are repetitions of a state, not new independent states.

Freeze the six seed families across all mode/load variants. Assign the first three seeds to exploratory training and the last three to exploratory validation; all six variants of a seed stay in its group. Pilot data remains development evidence, never independent test evidence. Reusing seed families across load and mode requires grouped statistics; 36 configurations are not 36 independent seed families.

## Necessary work before pilot execution

1. attempt008 actual representative and pair audits, including delayed application and full restore.
2. Generate the fixed 36-row mode/load/seed/condition matrix, without running it.
3. Verify disturbance toggles are actually consumed. The representative runner has hard-coded fault/outage targets and times: changing a metadata condition name alone cannot disable them. Single-condition configs must schedule unwanted events outside the full run window and remove unwanted energy, activation and speed mechanisms.
4. Implement selected-state persistence and paired suffix execution. The current A1 pair captures only the configured snapshot time; it is not yet an eight-state collector. A future suffix must retain the observed past and use time-addressed exogenous randomness shared across keep/replan.
5. Freeze the scalar formula and its independent trajectory reducer before running the pilot. The existing scalar draft is not an implemented/frozen label. Original physical components and alternative cost weights must be preserved; weights cannot be fitted to pilot gains.
6. Record empty windows, unknown consequences and actual failures separately. Do not count forced events, suffix repetitions, or mode/load duplicates of an identical state to reach 30/30.

## Current implementation limits

The ordinary event gate is a count/cooldown trigger, not a learned gain gate. The catalog currently offers unstarted UAV transfers and NOOP, not a complete task-swap/cooperative search implementation. Abstract service uses the existing post-observation route logic and still requires actual pilot evidence. CPU planning timings exist, but a complete measured detection/feature/inference/repair/transfer accounting pipeline is not yet implemented. No training, world model, PPO, independent test or OOD result is established.

Scope: primary coordinator read-only review of current source interfaces. No external independent reviewer/model claim. This is a list of prerequisites derived from the requested experiment, not a new defensive-check framework.
