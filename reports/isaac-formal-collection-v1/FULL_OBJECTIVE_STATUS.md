# Full Chapter 2 objective: evidence and remaining work

Updated 2026-10-21. The objective remains active. The user's newer direct-formal
instruction removed separate representative, single-state and 36-scene pilot
prerequisites; it did not establish any scientific result or waive fair pairing,
hidden-information isolation, safety, costs or the end-of-collection 30/30 gate.

## Current authoritative evidence

- Formal registry: 168 independent scene seeds, seven conditions, 84 training
  and 84 validation scenes; registry SHA256
  `d09eb7165511c34614219aa90c03cc9b0c14b13a766409d60bb7a82c76aabf70`.
- Isaac 4.5 formal batch launched on physical GPU1 at 2026-10-21 00:07:55
  Asia/Shanghai. Process PID 2570396 was re-observed live this continuation.
- First acquisition episode completed with 36 platforms and 81 search tasks,
  76,881,535 compressed trajectory bytes. Its first state's two suffix receipts
  reached complete; the next state was running. This is partial runtime evidence,
  not completion of a scene, all disturbances or the whole experiment matrix.
- First two suffix worker receipts recorded about 111.9/133.1 wall seconds and
  351.2/388.7 process CPU seconds. The one-thread numerical-library environment
  does not imply the Isaac engine used only one CPU core; report actual resources.
- Frozen collector and reducer are publicly archived on the independent branch
  `archive/ch2-isaac-roi-simplified-v2`, commit `39b4580` (remote readback confirmed).
- The mistaken analytical collector remains excluded from Isaac evidence.

## Requirements still unproved

| Requirement | Current evidence/status |
| --- | --- |
| All 168 scenes attempted with per-scene failures retained | RUNNING |
| All six single disturbances and composite produce physical consequences | Not yet established across the matrix |
| Full pairing, geometry, safety and costs independently recomputed | Reducer implemented; end-of-batch execution pending |
| At least 30 positive and 30 nonpositive valid ordinary states, split coverage | Not evaluated; evaluate once at batch end |
| Ordinary-state signal and grouped baseline comparisons | NOT_RUN |
| Constant, Ridge, tree and three-seed GPU MLP training | NOT_RUN; implementation may be prepared without fitting |
| Frozen threshold and independent residual calibration | NOT_RUN |
| Closed-loop event/periodic/risk/heuristic/learned comparison | NOT_RUN |
| Actual full online costs and action-effective waiting | Data fields implemented; full deployed model comparison pending |
| Genuine one-step event-aware world model and its added decision value | NOT_RUN; depends on usable data |
| G/G-WM real PPO updates, frozen WM, corresponding new gain labels | NOT_RUN |
| Independent test and OOD confirmation, scenario-level confidence intervals | NOT_RUN; no scientific support claim |
| Complete raw data/checkpoint/resource/failed-case archive and final review | Pending actual completion |

The formal collection currently uses search mode and three targets per region as
fixed design settings. The code's support for one/five targets and abstract service
does not establish performance for those settings. A simplified source-kinematic
Isaac experiment does not establish results for later formal aircraft models.

Collection alone cannot support the method. Until closed-loop independent test
and OOD evidence meet the original safety, completion noninferiority and total
cost criteria, the method remains INCONCLUSIVE and world-model added value is
unestablished. No successful model or experiment milestone tag is warranted.
