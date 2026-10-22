# One-step supervision scope review — 2026-10-22

Coordinator read-only source review of raw_public_windows.py, public_dynamics.py, features.py, contracts.py and formal_collection_v1/run_capture.py. No new test suite or real-data training was run.

## What the current source can supervise

- Task progress is the count of public search_cell_complete events in the next second. It is not continuous sweep coverage or target-discovery truth.
- Energy is the mean remaining-energy delta over matching platform IDs. Missing energy is masked; this aggregate does not establish per-platform energy prediction.
- Delay is sent-to-applied elapsed time for an unambiguous public aircraft/decision join whose application falls in the next interval. Unapplied/right-censored commands are masked, not assigned zero delay.
- Risk is an indicator of public fault/hold/violation events. It includes exogenous faults and holds; it is not a demonstrated predictor of controllable hard-constraint violations alone.

The source uses adjacent same-branch integer-second rows and clears history on gaps/incomplete actor ticks. Post-t events cannot enter the input event tensor. Paired decision gains use a separate pre-candidate projection path; a mixed pre/post-step history is not accepted as physical one-step supervision.

## Limits to preserve in training and reports

Raw capture does not provide full applied action vectors, future task metadata or complete public queues per tick. Missing action remains unknown, not a measured NOOP. Responsibility-zone edges are not measured communication connectivity. Sparse paired projections do not supply a continuous action-conditioned transition dataset.

Event-derived zero targets rely on a complete event log: the decoder combines parser status with receipt events_complete (default true). The inspected collector writes/flushes events and records completion on normal execution, but no real-file completeness proof is established by this source review alone. Terminal-file integrity remains part of the required data check.

Current formal capture uses a one-second task clock and ceil-to-second planning delay with at least one second in the paired replan path. Reports must identify that execution-delay model and separate measured wall/CPU time. It cannot substantiate millisecond-level real-time advantages.

These boundaries must be reported alongside actual target coverage counts. A successful loss update alone cannot establish useful dynamics, causal action simulation, or world-model added value. Formal training and scientific conclusions remain NOT_RUN pending the terminal dataset gate.
