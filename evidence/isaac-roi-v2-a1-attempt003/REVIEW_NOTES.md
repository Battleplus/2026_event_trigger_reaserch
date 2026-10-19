# A1 attempt003: coordinator observations pending final recheck

The representative episode and paired replay completed in actual Isaac Sim. Their frozen source remained unchanged after execution. Receipt archive SHA-256: `177e318df60b00fce749ff246baac49c6eaab64ee53fffa7d71d25c192386e45`.

Representative: 5,400 simulated seconds; 194,400 actor-second records; 36 platforms; 81/81 public search cells complete and 27 targets discovered. External geometric audit reports no failures, including ordered sweep replay, first same-zone/radius discovery from evaluator-only truth, actual helicopter home-zone positions, displacement-based speed/energy, and communication hold/resume.

Pair: 1,200-second prefix, then 1,800 seconds per branch. Restored full-state digest equals the original and maximum restored position error is 0 m. The selected UAV transfer moves `z0-cell7` from `uav-0-1` to `uav-1-0`. Both branches finish with 79/81 public cells and 27 discovered targets. Window analytical energy is 299.239477307883 kWh for keep and 306.042885169803 kWh for replan; controller switches are 18 and 19. Thus this single sample does not show a completion advantage, and replan consumes approximately 6.8034 kWh more. A scalar net gain has not been computed.

The first pair audit fails only on the communication attempt summary: production records 2,393 versus 2,426 issuance attempts, while the original auditor counts 2,396 versus 2,430 delivery/drop events. Commands issued before the evaluation window may arrive inside it, so delivery is not issuance. Keep this failed report; correct only the auditor's metric definition and write a separate recheck against the original trajectory. No simulator rerun is needed for this reporting error.

Run wall times are 105.89 seconds for the representative episode and 100.48 seconds for the paired replay. CPU utilization is 340% and 353%, respectively: thread-limited BLAS does not make the Isaac renderer a one-core process. GPU1 was selected; after completion GPU0/GPU1 usage returned to the pre-run 6/39 MiB and 0% utilization.

Scope remains simplified kinematic platforms at 20 km display altitude, analytical 2D sensing/energy/communication, and a public greedy initial-allocation proxy. Hard task deadlines are not implemented. This run does not validate formal aircraft models, a trained policy, or net-benefit prediction. A2-A4 remain unexecuted.
