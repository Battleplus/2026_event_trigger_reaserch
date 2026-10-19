# V3.2 performance and light-gain experiment

Date: 2026-10-19 (Asia/Shanghai).

This branch implements the user-approved V3.2 performance-first plan. It starts from the V3.1 archive 0c69f60 and is independent of the Isaac/full-world-model branches.

Implementation: GPT-6 Luna. Read-only technical review: coordinating model, recorded without claiming a different reviewer identity.

1. Profile the original slow task-change scenario, preserving seed and configuration. Optimize measured repeated work only. Compare actions, physical trajectories and execution-cost components; measured CPU cost may change.
2. Necessary checks: slow-scenario equivalence with independent label recomputation, one 90-minute NOOP, existing regional/safety checks. Repair actual worker RSS accounting if the launcher measurement is confirmed incomplete.
3. Freeze reviewed source and run a new 18-instance smoke from scratch, with 1800 CPU seconds, 8 GiB, one CPU thread, 90-minute episodes. Another budget stop ends this version; no automatic retry or budget increase.
4. Only after smoke passes: collect the frozen 168 scenario instances (7 conditions x 12 train + 12 validation), at most 8 ordinary states per trajectory and 2 matched suffixes per ordinary state. Do not change the fixed replanner, scoring weights or sampling to obtain positive labels.
5. Count unique state_id, excluding forced, unknown and empty windows. Require at least 30 positive and 30 nonpositive states, with both splits and multiple scenario groups represented. Otherwise stop training and report INCONCLUSIVE.
6. Only after the data gate: CPU constant/Ridge/tree and GPU 64x64 MLP, seeds 8201/8202/8203, batch 32, lr .001, at most 100 epochs, patience 10. Fit normalization on training only. Model selection and calibration use separate halves of the validation scenarios. No silent CPU fallback for CUDA training.
7. Compare Existing-ET, periodic 4-minute replanning, risk, heuristic gain, Gain-Tree and Gain-MLP with the same replanner and safety rules. Count end-to-end costs. Independent test and WM/PPO are conditional on the registered safety, completion and cost criteria and available resources.

Keep old failures. Record measured CPU/GPU resource use, configuration and result hashes. Do not introduce another authorization, identity or general validation framework. Formal collection checks new data and essential invariants; the complete hash/review is repeated at final archive.
