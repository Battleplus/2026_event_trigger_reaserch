# Formal Isaac collection v1: coordinator read-only source review

Date: 2026-10-21. Implementation: delegated GPT-6 Luna workers. Review: primary
coordinator; no claim of GPT-6.1 Sol High review. This is source/interface review,
not evidence that the new Isaac experiments have completed.

The user's latest execution instruction supersedes the earlier representative
and single-state prerequisites. The batch directly acquires 168 independent
scenes. Its internal `representative` phase means the 90-minute acquisition
trajectory of each formal scene, not a separate acceptance gate. No 36-scene
pilot is required or claimed.

Reviewed collector/evaluator interfaces and registry generation:

- Seven conditions, 84 train and 84 validation scenes; unique scenario seeds;
  four scenes from each of three trigger sources per condition/split.
- Search mode and three hidden targets per region are fixed design choices,
  not settings selected for observed superiority. Test/OOD are not collected.
- At most eight ordinary snapshots per scene; synthetic periodic ticks alone
  do not qualify. Hash-priority sampling does not use future gain.
- Each state has two suffix seeds in separate Isaac worker processes.
  Keep/replan restore the same 17-field common state and backend snapshot.
- Evaluator loads the original indexed snapshot with both collector and
  original scene import roots. Reductions are separated by scene/state/seed;
  original trajectories and receipts are not rewritten to manufacture prefixes.
- Failures stop the affected scene; other independent scenes continue. No
  replacement scenes, automatic retries or repeated preliminary acceptance runs.
- Gzip stores complete raw trajectory fields. The aggregate is generated once
  at the end and excludes forced, empty, unknown and synthetic fixture labels.

Scientific limits retained: source kinematic aircraft with USD position reads,
analytical energy and communication, 1-second task clock and at least 1-second
planning delay. Suffix seeds resample time-addressed communication dropout only;
other scheduled disturbances remain frozen. This does not establish all-future
uncertainty coverage, physical flight validity or millisecond real-time operation.

The mistaken legacy analytical collector is excluded from this Isaac study;
its data must not enter this batch or the 30/30 count. See
`FORMAL_EXECUTION_CORRECTION.md`.

No engineering or scientific PASS is assigned before actual execution evidence.
End-of-collection 30/30 and split coverage decide whether gain training can start.
