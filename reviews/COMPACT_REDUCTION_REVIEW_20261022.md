# Compact reduction source review

Decision: ACCEPT_PARTIAL_REDUCTION_SOURCE. This is not an Isaac execution, full safety PASS, compact-only deployment approval, or measured storage saving.

Implementation: GPT-6 Luna. Review: main Codex coordinator, read-only inspection of implementation and frozen SHA-256. No claim of GPT-6.1 Sol High or cross-family independence.

## Frozen files

| File | SHA-256 |
| --- | --- |
| closedloop_v1/runtime.py | 4c9ee82442f8089b22f045820a4e6de573effa9f663f428f84bab64be128dbc9 |
| closedloop_v1/compact_capture.py | cb5517c2fb71cf797480fbbab280e02bd34adfacd0b44252f146aa26749679d0 |
| closedloop_v1/evaluate_episode.py | 53c0b0432d3d22a6904c591557d3efb9f28bdd2477a57a02563fe277d95efa66 |
| closedloop_v1/test_compact_reduce.py | 7bada8e0377d76a414c8b41c25586c2b3844b594f774e87c5945adf7e24b136b |

The coordinator independently read these hashes and inspected the capture boundary, reconstructed reducer inputs, evaluator-only separation and synthetic assertions. Luna reported five focused tests and relevant compilation passing; the coordinator did not repeat the same suite.

## Evidence and accepted scope

`evaluate_episode.py --compact-dir` rebuilds standard trajectory and event inputs, then invokes the existing independent `reduce_episode`; it does not use production gain labels as reference results. Public ticks retain actual positions, transition energy/command quantities and public sweep geometry after activation. The complete online timing event is appended after it exists. Evaluator truth, raw discovery labels and complete private fault-release/event details are stored separately from the public reader.

The single synthetic case includes a geometrically discoverable target, task completion and two ticks of phase timing. It asserts equal raw/compact energy, discovery IDs, waypoint/completion results, communication counts and phase totals. It also asserts that the public reader does not expose the evaluator's raw target-discovered event. Both reductions remain UNKNOWN because full forced-safety identity evidence is absent; equal partial results do not prove complete-run validity.

## Remaining limits

- Original full logs are still written. Total storage reduction has not been measured; no duplicate raw evidence may be removed based on this change.
- The compact capture is currently restricted to the representative branch. Full private paired-state recovery and both branch captures have not been demonstrated through this path.
- No Isaac run or new forced-fault case was executed for this change. Current formal collection remains unchanged.
- Compact I/O is outside the four method phases; its own storage and resource overhead is not established by the synthetic equality check.
- Missing evidence must remain UNKNOWN. The synthetic result does not establish a full end-to-end cost or safety PASS.

Latest live observation: 2026-10-22 03:08:43 UTC, 77 scene receipts COMPLETE, collector PIDs 2570396/2570401 and Isaac child 3553105 present, 36,681,027,584 bytes free. No terminal batch receipt or aggregate yet. This is a verified wait, not the final data gate.
