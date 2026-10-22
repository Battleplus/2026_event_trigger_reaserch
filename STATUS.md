# Actual lightweight collection: 2026-10-22

The Isaac acquisition has switched to the reviewed lightweight V2 source.
92 old scenes are preserved; the remaining 76 registered scenes are being
acquired with public 1-second deltas, static data once, immediate events and
lossless private paired evidence. GPU1 only. Seven focused tests and one real
suffix format/independent-scalar comparison passed. The first new 90-minute representative episode has completed, with
5400 one-second records independently read back; its paired suffixes are running.
Union reduction, 30/30,
training and scientific conclusions are pending. See
[evidence and limited review](evidence/isaac-lightweight-switch-20261022/REVIEW.md).

Historical checkpoints below describe their observation times.

---

# Current Chapter 2 status: 2026-10-22

The authoritative revised objective is [CURRENT_OBJECTIVE](protocols/CURRENT_OBJECTIVE.md).
The latest verified server observation was 2026-10-22 08:38:48 UTC: 89 of the
168 scene receipts were COMPLETE; collector and Isaac child processes were
alive, with no terminal batch receipt or aggregate. This is acquisition progress,
not scientific acceptance. The 30/30 gate, model training and closed-loop method
evaluation have not run. Test/OOD remain sealed.

Future collection is narrowed to assignment, gain and public one-step learning
information. Existing raw data and the live collector are preserved. Compact
partial reduction and repaired event-graph source are archived with scoped
reviews; actual storage reduction, GPU training, world-model added value and
PPO results remain unestablished. A streaming raw-trajectory adapter is being
implemented and source-reviewed locally. The WM gain-history runtime bridge,
masked PPO update core and public candidate encoder have scoped source acceptance;
real GPU inference, real Isaac PPO rollouts and action-conditioned WM supervision
remain unverified. The candidate policy callback is being integrated and reviewed.
No cache cleanup, scene retry or favorable seed replacement
has been performed.

The older dated entries below are preserved historical evidence, not the
current execution status.

---

# Collection source preparation: 2026-10-20

A separate local capture runner now selects up to eight ordinary states before candidate generation and separates trusted private snapshots from whitelisted public input. Three sampler tests and two service-safety tests passed; source review is provisional for integration. Isaac capture is NOT_RUN; two-suffix rollout and independent service trajectory reduction are NOT_IMPLEMENTED. attempt008 is unchanged and its remote deployment authorization remains pending. See [capture source](src/isaac_roi_capture_local_v1/) and [review](evidence/isaac-roi-capture-local-v1/REVIEW.md).

---

# attempt008 local repair review: 2026-10-20

Local repair accepted for Isaac validation; three targeted tests passed. Remote upload/run is NOT_RUN pending specific attempt008 authorization. The fixed pilot matrix is being prepared locally. Pilot requires multi-state/two-suffix collection before its 30/30 gate; the A1 single-snapshot runner does not yet implement that workflow. See [local review](evidence/isaac-roi-v3-attempt008-local/REVIEW.md) and [pilot readiness](evidence/isaac-roi-v3-attempt008-local/PILOT_READINESS_REVIEW.md).

---

# V3 Isaac ROI current review: 2026-10-20

attempt007 representative and pair completed with raw trajectory audits PASS. Protocol acceptance remains REPAIR_REQUIRED: paired planner-delay handling and mandatory NOOP retention need repair. Measured local slowdown and energy reserve pause are established; no scalar gain, trained model, pilot or formal data result is claimed. See [attempt007 review](evidence/isaac-roi-v3-attempt007/REVIEW.md).

---

# V3 Isaac ROI: 2026-10-20 update

attempt006: REPAIR_REQUIRED; representative audit PASS, pair original audit FAIL due to command-stage accounting. Independent sends+drops recount confirms production attempt delta -25. Zero measured speed-block exposure, incomplete disturbance coverage and undercounted planner calls still block pilot acceptance. Formal collection, training, world model and PPO: NOT_RUN. See [attempt006 review](evidence/isaac-roi-v3-attempt006/REVIEW.md). Historical status follows.

---

# Isaac ROI Simplified v2: Current Status

Date: 2026-10-19. Implementation is in progress. Attempt001's 90-minute representative run completed in actual Isaac: 5400 simulation steps, 194400 trajectory rows, exit 0, 101.61 seconds wall time, and 4391696 KiB maximum RSS. All six source dependency hashes and the downloaded trajectory hash matched. Its independent audit failed with 506037 findings, including actor type/kind/speed schema mismatch and non-finite energy values. Attempt001 pair: NOT_RUN. Attempt002's representative and keep/replan execution completed, but its acceptance remains `REPAIR_REQUIRED` (details below). Pilot: NOT_RUN. Formal 168-episode collection: NOT_RUN. Training: NOT_RUN. See [the v2 protocol](protocols/isaac-roi-simplified-v2/EXECUTION_PROTOCOL.md), [attempt001 summary](evidence/isaac-roi-v2-a1-attempt001/A1_ATTEMPT_SUMMARY.md), and [attempt002 summary](evidence/isaac-roi-v2-a1-attempt002/A1_ATTEMPT_SUMMARY.md).

Attempt002 remains a preserved diagnostic failure. Attempt003 repaired those accounting and safety issues and completed a new actual representative plus fair pair. The independent geometric recheck passed for sweep completion, evaluator-only discovery, helicopter zone bounds, displacement speed/energy, and communication hold/resume. Full state restoration matched with zero position error. The limited engineering decision is `A1_SIMPLIFIED_ENGINEERING_PASS`; this is not a method result. The pair ended with 79/81 public cells and 27 discovered targets in both branches; replan used 6.803 kWh more in the window and one additional switch. Scalar net gain is not computed. Pilot, formal collection, and training remain NOT_RUN.

A1 prototype scope: initial assignment is a public greedy C1 proxy; motion is source kinematic without turn-rate/turn-radius constraints; discovery is analytic 2D at commanded/display altitude 20 km; there is no hard deadline. These limits apply to the A1 prototype only. The candidate cost formula and genuine ordinary-decision trigger remain under implementation and require pilot validation; they are not established capabilities.

---

# Isaac 20w world-model study: scope update

Date: 2026-10-19 (Asia/Shanghai).

- User requires the existing 20w Isaac Sim scene and actual Isaac-origin world-model data; formation geometry is not fixed.
- Protocol amendment: protocols/ISAAC_20W_SCOPE_AMENDMENT.md.
- Scene inventory and coordinator source review are archived in evidence/isaac20w-inventory-v1. The eight-platform kinematic scene needs a new civil task/waypoint adapter; this is not 30-platform acceptance.
- Candidate Isaac runtime installation was still in progress during inspection. A short actual-frame/waypoint pilot adapter is now being implemented; no runtime success is assumed.
- Pure-Python N02 V3.3 collection is paused. Prior shards are diagnostic history, not Isaac data.
- New Isaac episodes, training and six-arm results: NOT RUN. Scientific conclusions: INCONCLUSIVE.
- This milestone archives the revised protocol only; no scene acceptance or trained model is claimed.

---

Historical status follows.

# V3.2 Full World-Model Study Status

Date: 2026-10-19 (Asia/Shanghai).

- Protocol: `protocols/FULL_WORLD_MODEL_V3_2.md`. Fixed CPU/GPU time caps removed by user instruction; memory and actual fault handling remain.
- New source snapshot: `releases/gain-gate-v3-2-full-world-model/`, with per-file hashes. This snapshot is not an acceptance or a training result.
- V3.2 smoke: STOPPED FOR REPAIR after 4/18 committed instances. Independent replay showed the periodic/local scope filtered every non-NOOP alternative although the same public state had 15 legal global alternatives. This is an implementation defect; these four shards remain diagnostic only and do not count toward smoke acceptance.
- Two pairing defects repaired: stale derived scenario hash after future-suffix substitution; ambiguous continuation receipt/action envelope.
- Necessary local checks: 20 passed, 6 temporary-directory fixture cases deselected due to host permissions; separate real shard commit/readback succeeded.
- Server GPU1: real small CUDA forward/backward probe passed. No new model training or checkpoint exists yet.
- Formal 168-instance collection: NOT RUN. Data gate: NOT EVALUATED. Models/PPO: NOT RUN. Scientific conclusions: INCONCLUSIVE.
- Required repair before a new smoke: local-to-global candidate fallback, canonical fractional command-delay integration, and a new source binding. V3.2 is not eligible for formal collection.
- Execution-delay semantics are being developed separately; current minute-clock smoke is engineering evidence only and cannot establish subminute decision latency benefits.
- Archive push and remote ref readback succeeded: f3ac0913d95fa338a82d858c9498c4c5143b99c8 on archive/gain-gate-v3-2-full-world-model. These artifacts document a repair, not study acceptance.

Historical status below is preserved verbatim and is not the current execution status.

---

# Historical V3 Archive Status

Date: 2026-10-18 (Asia/Shanghai).

- Milestone: protocol baseline, implementation in progress.
- V3 engineering smoke: NOT PASSED / NOT REVIEWED. First real execution attempt failed before reset at legacy validator dispatch; repair is in progress.
- V3 formal data: NOT RUN.
- V3 30/30 training gate: NOT EVALUATED.
- Light model / world model / PPO: NOT RUN.
- Method: INCONCLUSIVE.

Only `protocols/v3-baseline/` and these status documents describe V3 in this commit. Other existing tracked source and evidence are historical V2 artifacts, retained without a V3 success claim. Event-generator details remain provisional until sampler implementation and config-hash review.

The V2 interrupted CPU total remains unknown; 1890.72 seconds was a lower bound. No complete historical resource settlement is fabricated. V3 will use separate measured receipts and cannot claim that the original project budget has been reset.

Coordinator probes are archived in `reviews/v3-in-progress/`. Region helper checks passed only in their stated scope. Current baseline regression rerun: 124 passed / 9 failed (four missing-dependency failures, five legacy review-fixture failures). No complete V3 runtime or scientific outcome is claimed.


## Isaac pilot implementation review (2026-10-19)

Two-platform kinematic collection adapter and config frozen; syntax/config checks passed. Coordinator decision: READY_FOR_RUNTIME_VERIFICATION, not experiment acceptance. Isaac Sim 4.5 package metadata verified; no scene started or actual data collected. Specific server transfer awaits explicit approval after automatic review rejection. Full 30-platform mission, fair mission pairing, training and conclusions remain pending.


## Actual Isaac loading verification (2026-10-19)

Native20w terrain and140trees loaded. Two and30 source-backed quadrotor entities each completed300Kit/app updates with positive USD displacement; exit0. Coordinator independently checked final poses, entity uniqueness, counts and deployed hashes. GPU1 sampled peak1839MiB;GPU0 had3MiB enumeration overhead,0% sampled utilization.30-platform mean scene+app time93.89ms: load capacity passed, real-time performance not established. No90-minute mission,54-task execution, training or scientific effectiveness claim. See evidence/isaac20w-loading-capacity-v1.
