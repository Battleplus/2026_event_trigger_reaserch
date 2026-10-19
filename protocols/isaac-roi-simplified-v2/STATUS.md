# Protocol Status

As of 2026-10-19, this directory contains the v2 execution specification, experiment matrix, migration contract, and this status record. A1 prototype execution has occurred, but the attempts remain under repair and do not establish scientific acceptance. A2 pilot, A3 formal trajectories, and A4 model training/evaluation are not established as run.

## Verified prior evidence

The v1 Isaac loading-validation report and coordinator review record a successful 2-platform and 30-platform diagnostic, each with 300 Kit/application updates, using 30 quadrotor entities for the larger stage. The accepted short run used Isaac Sim 4.5.0 and reported movement plus USD pose agreement. It is evidence that this narrow scene load completed; it is not a 90-minute mission, the required 18-UAV/18-helicopter composition, an ROI task episode, or a method result. Timing reported for the 30-quadrotor stage was slower than real time.

The v1 pilot implementation report records an earlier two-platform civil pilot that was not run and an earlier rejected staging/upload attempt. That deployment issue was subsequently resolved with explicit authorization: the required private source and Isaac runtime have since been deployed, and the current coordinator reports SSH access is available for the authorized A1 work. The older `runs/pilot_status.json` remains a historical snapshot, not current deployment status. The pilot source/config syntax checks do not count as simulator evidence. This v2 status omits host, account, directory, token, and private source paths.

The latest coordinator-provided runtime snapshot is Python 3.10.12, Isaac Sim 4.5.0.0, Torch 2.5.1+cu121, and NumPy 1.26.4. GPU 1 is an RTX 2080 Ti with 11 GiB and was observed at 39 MiB used; GPU 0 was observed at 6 MiB used. Available disk was reported as 131 GiB and available RAM as approximately 28 GiB. These are a point-in-time preflight snapshot, not proof of A1 execution or training. This runtime differs from the older N02 Python 3.11/Torch 2.7 environment described in v1 artifacts; do not conflate their results.

## v2 execution state

- A1: actual Isaac representative and paired prototype phases are recorded in the repository root status and attempt-specific evidence. Known limits include public greedy C1 proxy, kinematic motion without turn constraints, analytic 2D discovery at 20 km, and no hard deadline. Candidate cost/gain and ordinary-trigger implementations remain incomplete; current attempts are not accepted study results.
- A2: planned; zero of 36 pilot episodes evidenced.
- A3: planned; zero v2 trajectories evidenced, no 30 positive plus 30 non-positive unique training state IDs established.
- A4: planned; no fitted baselines or GPU model checkpoints, no frozen `G`/`GWM`, and zero of 42 independent test scenes evidenced.
- Stage B: blocked on identifying and supplying the formal platform/model interface and artifacts. No compatibility or migration result is claimed.

## Outstanding blockers and evidence needed

1. Actual 36-platform Isaac task adapter and scenario/task implementation, including randomized ROI, nine-region movement rules, typed tasks, abstract-strike semantics, and full-ROI logging.
2. Full-state/exogenous-stream pairing and event provenance implementation, with verified restoration and per-step receipts.
3. Frozen episode-level splits, label definitions, end-to-end cost instrumentation, model training/evaluation pipeline, and evidence meeting A3 positive/non-positive minimums.
4. Formal platform/model identity and interface to resolve Stage B mapping.

Do not use `PASS` for any v2 stage until its listed artifacts exist and are reviewed. Current stage status is `planned` (Stage B dependency `blocked`), not passed.
