# Full world-model study V3.2

## Authorized protocol revision

The user authorized this revision on 2026-10-19. This replaces the fixed 1800-second smoke cap and cumulative CPU/GPU budgets with completion of the frozen experiment matrix. Time and resource consumption remain measured. An 8 GiB host memory limit remains. Old V3/V3.1 failures are retained and do not count toward the new smoke.

GPT-6 Luna implements; the coordinator independently reviews frozen artifacts. Engineering acceptance, data eligibility, and scientific support are separate decisions.

## Matrix and ordering

- 100x100 map, nine geographic regions, 24 UAVs and six helicopters, 54 logical tasks, 90-minute episodes, at most 16 physically distinct candidates.
- Eighteen engineering smoke instances, followed by 168 collection instances: seven conditions x (12 training + 12 validation). Test and OOD remain sealed during collection and fitting.
- Train only if independently recomputed ordinary states include at least 30 positive and 30 nonpositive gains with train/validation scenario coverage. Forced, empty, unknown and repeated state IDs are excluded.
- CPU: simulation, branching, constant/Ridge/tree. CUDA: MLP, graph/GRU world model, G/G-WM PPO. No silent CPU fallback; no paid compute.
- Three neural training seeds: 8201, 8202, 8203. No old model weights or old training labels.
- Train the initial model from real simulation transitions, including one-step targets. Train paired G/G-WM policies, freeze each replanner, recollect its own gain labels, then fit matching gates. A weak light-model result alone does not cancel the world-model comparison.
- After validation configuration and thresholds freeze, evaluate 42 independent test instances: six per condition. Do not tune on test results.

## Six comparisons

| Group | Trigger/gate | Frozen replanner |
|---|---|---|
| A | Existing event trigger | G |
| B | Light gain gate | G |
| C | Existing event trigger | G-WM |
| D | Light gain gate | G-WM |
| E | Event-aware world-model gain gate | G-WM |
| F | Four-minute periodic | G-WM |

All groups share candidate rules, repair and mandatory safety handling. Compare A/B and C/D for gating, A/C for WM-assisted policy training, D/E for WM-based gain prediction, and C/F for event versus periodic triggering. The light predictor is selected from constant, Ridge, tree and MLP using validation only.

## Required evidence

Keep/replan branches share the snapshot, exogenous suffix and safety continuation; model input is the current public projection/history only. Labels are independently recomputed. Cooperative actions commit atomically, safety pauses change actual execution, and logical completion reward is counted once.

Record CPU/GPU compute, inference, transfers, communication and switches in end-to-end cost. Delayed commands become effective after computation and are revalidated before execution; reporting time alone does not model delay. GPU optimization and fresh-process checkpoint restoration must actually occur. World-model parameters remain frozen during G-WM PPO and the auxiliary critic must receive nonzero gradients.

## Scientific decisions

Use independent scenario instances as the statistical unit, with paired effects and 95% confidence intervals. Training seeds are repeated fits, not independent environments. Method support requires mandatory safety handling for every applicable event, no new controllable hard-constraint violations, a weighted completion-rate difference lower confidence bound above -2 percentage points relative to the existing event trigger, and a confidence interval supporting lower total cost.

Report engineering PASS/REPAIR/STOP, data gate MET/NOT_MET/NOT_EVALUATED, method SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE and WM added value separately. Preserve negative outcomes, failures, hashes, resource receipts and actual run scope. No favorable seed selection or post-result scenario changes.

## Current execution status

This file records a protocol, not completed results. V3.2 preparation identified two old masked pairing faults (derived suffix scenario hash and continuation receipt/action unpacking); fixes and real-pair evidence are required before the new smoke is accepted.
