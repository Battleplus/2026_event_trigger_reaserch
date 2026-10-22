# WM gain adapter: source acceptance — 2026-10-22

Decision: ACCEPT_SOURCE_INTERFACE, NOT_INTEGRATED. Implementer: GPT-6 Luna. Reviewer: coordinating Codex agent (actual model identifier unavailable). Coordinator review was read-only; no repeated test suite or GPU execution.

Inspected gain_deployment.py, trainer changes, synthetic fixtures and scope documentation. The trainer emits validation gain predictions and a per-seed deployment sidecar bound to checkpoint/data/schema identities. Threshold selection reuses the existing scene-grouped validation selection split; residual calibration uses disjoint validation scenarios. A changed calibration target cannot reselect epsilon. Insufficient calibration scenarios leaves lower_bound unavailable, while mean remains available. Test/OOD rows are rejected by the calibration interface.

Inference takes explicit public history, requires the declared mixed phase (physical through t-1 plus pre-candidate projection t), and accepts 2–8 contiguous integer-second frames. Action and legacy event payloads are stripped; current public_events remain filtered by the numeric feature mapper. No candidate catalogue or paired consequences are requested. The GPU path requires the authorized GPU1 UUID and logical cuda:0 without CPU fallback. A CPU option exists only as an explicitly marked synthetic fixture interface.

Timing separates graph construction, host/device transfer, synchronized forward and return transfer, plus calling-thread CPU and full prediction wall time. The full total and components overlap and must not be summed again in a future runtime ledger. Upstream history acquisition, model loading and complete runtime phases remain outside this adapter measurement.

Luna reports 15 focused synthetic/public-dynamics/event-graph checks and compilation passed. The inspected new fixtures cover selection/calibration separation, calibration-target mutation, unavailable small-sample lower bound, nonvalidation exclusion, action invariance, wrong/empty/gapped history rejection and timing fields. No real GPU model loading or inference test occurred.

No fitting, real-data calibration, runtime deployment or method E evaluation was run. The caller is still responsible for obtaining genuine current pre-candidate public frames; a phase argument alone does not prove physical provenance. An online history accumulator and closed-loop integration remain to implement. G/G-WM policies and their own paired labels remain outstanding.

Frozen SHA-256:

- gain_deployment.py: 9b5efe23af0980babe5172e97973745d89abce4aaf06b8b0168de20315914fe9
- trainer.py: cd1e19df679f1b79560eda7c503ecbc2ea5e7ae6f862e660f8abc857a6102493
- __init__.py: 5f7b729db14a621518dccb34cb9b5a3b968fe1bd419394132e202bf92a9aeb93
- tests/test_synthetic.py: be6e58b8923de12c625ff63c4def5f3a50f94fa6d20d42163730f30819e53f4a
- README.md: 70bfd59c6602ba7ddece648970c7a16c8776ac8b714c18ad81e6fdf2d3e6a296
- PROTOCOL.md: 6a3ea71553f09ec66d02103aca35d202ffc93dbb9636f1526df3371071b14d52
