# V3.3 civilian multi-region search and rescue

Approved execution: 2026-10-20 (Asia/Shanghai). Implementation: GPT-6 Luna. Parent coordinator reviews executable changes read-only; no claim of Sol High review.

## Frozen design

- Civilian search and rescue/supply delivery only; no strike or weapon dispatch.
- Map100x100, nine regions, two UAVs and two helicopters per region (36 platforms).
- Ninety simulated minutes, <=16 candidate actions including NOOP, paired horizon30min truncated at episode end, two matched suffixes.
- Default: search then rescue delivery, three target sites/region. Sites and generated dependent tasks are separate counts. Unobserved sites/demands are not public policy inputs.
- Responsibility regions permit assistance. Selection accounts for capability, travel, energy, communication and donor-region outstanding commitments.
- No custom CPU/wall/memory guardian termination. Record actual usage; preserve genuine errors. No fabricated completion receipts.

## Ordered gates

1. Correct completed-task residual scoring and trainable-mask counting; adapt task semantics and36-platform scene. Preserve V3.2 failures; exclude old labels.
2. Focused checks: completed/incomplete scoring; reveal causality; real non-NOOP cross-region pair, nontarget effects and independent raw recomputation; affected region/safety regressions. Coordinator reviews before launch.
3. Complete all18 default smoke instances (six inherited conditions x three). No partial old/new concatenation.
4. Configuration pilot: search-only/search-and-rescue x1/3/6 sites per region x three shared seeds (18 instances). Default changes require a documented feasibility/load rationale, never favorable method-effect selection.
5. After engineering review, freeze source/config and collect seven conditions x(12train+12validation)=168 scenarios, <=8ordinary states/trajectory, two suffixes. Existing-ET/Periodic-4min/Random-Allow-Suppress proportions remain frozen.
6. Require >=30positive and >=30nonpositive unique valid ordinary states, classes covered across splits/groups. Exclude forced, unknown and empty windows. Suffixes are not independent states. No extra seeds to chase this gate.
7. Only after gate: CPU constant/Ridge/tree and CUDA MLP(64,64; batch32; lr0.001; <=100epochs; patience10; seeds8201/8202/8203), real update and fresh-process restore. Repair existing CUDA issue; no silent CPU fallback.
8. Closed-loop methods share frozen replanner/safety. Measure detection, features, transfer/sync, inference, search, repair, communication and switching. Scene-level paired CIs. Independent test7x20 only after development criteria. WM/PPO conditional on stable light-model closed-loop benefit.

## Scientific checks

Keep/replan share snapshot, exogenous suffix and continuation safety policy. Labels derive from trajectories, not predictions. Completed tasks have zero residual-work penalty; unknowns remain masked. Count donor losses and receiver benefits. No hidden schedule, private identity or unobserved demand enters features.

Retained criteria:100%forced safety handling; no added controllable hard violations; weighted completion paired95%CI lower bound above-2percentage points; total-cost improvement supported by paired95%CI. Engineering completion is separate from method support.

Archive source, safe summaries, hashes and failures after milestones/conditions. Large raw traces stay local as on the previously approved archive route. No successful tags for unexecuted phases.

## Initial state

Implementation in progress. V3.3 smoke, pilot, collection and training NOT_RUN at this protocol commit. Prior V3.2 STOP is historical, not the V3.3 outcome.
