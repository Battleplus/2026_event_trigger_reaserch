# WM runtime history integration — read-only review

Reviewer: primary coordinator, current session model identity; implementation assigned to GPT-6 Luna. This is role separation, not a claim of cross-family independent review.

Status: SOURCE_ACCEPTED_WITH_RUNTIME_NOT_RUN after the two repairs below. Source inspection only; no Isaac execution, GPU execution, training or efficacy established.

The submitted implementation bounds physical history to seven preceding one-second frames and appends the public pre-candidate decision frame. It uses the raw decoder with an empty task manifest, matching the existing decision-training path. The ordinary missing-history path suppresses planning explicitly and consumes normal trigger/cooldown state; forced safety bypasses the predictor. Physical history update time is inside the post-step online-state phase.

Two repairs were returned to Luna:

1. Runtime currently immediately reduces the prediction to `.score`, losing the mean, lower bound, threshold, model version, device and inference timing breakdown from the actual decision event. Preserve these for that decision, without charging nested timing twice or reusing a previous prediction after missing history.
2. The submitted fixture checks history length, hidden-field removal and an unavailable gap, but not successful training/runtime window equivalence. Add one focused synthetic comparison against the raw-decoder plus public-projection training path, with an effective seven-plus-one-frame window.

Luna reports 11 focused checks, syntax compilation and CLI help passing. A separate compact-capture test encountered Windows temporary-directory access denial; it is not recorded as passing. Those results establish limited source behavior only. No claim that the full method E or G-WM was executed is permitted.

No live collector source was modified during this review.

## Repair review

The coordinator inspected the repaired decision/runtime/compact event paths and the successful seven-plus-one-frame fixture, and independently recomputed all six submitted source hashes. Prediction objects now retain their score, mean, lower bound, threshold, mode, version, device and timing breakdown; runtime emits this metadata on the matching decision event and compact capture retains it. Missing-history results are newly constructed with no prediction metadata. Nested prediction timings remain descriptive rather than additional cost charges.

The new fixture compares the live window to seven raw-decoder frames with an empty task manifest plus the decision-training projection frame. This is synthetic parity for the exercised fields, not full real-trajectory parity. Luna reports 13 focused checks and syntax compilation passing after repair; no duplicate historical suite was requested.

Frozen SHA-256:

- decision.py: `966e7d473afd5ebbb612933f204e7e32341a04ee4edb5e79f42a471ce928c55a`
- runtime.py: `bc9603e60d12bf4ca58d472d6e6a9072b2056907631a6903e69484471b0db94b`
- compact_capture.py: `93d23fc730dcac3304894fe0df45db4f0449413d902e0f626949bdac61b884db`
- test_wm_history.py: `44afdf5a50c6da0b6ddcc49839ad9968ddccd7c653bfe83b38410439270322f6`
- closedloop_v1/README.md: `88df1bfa1bf55a4ef351542b7f630b1afd1ce0b4801a94a95abc2cd112b27787`
- wm_ppo_v1/closedloop_history.py: `fa1a3fcc96ab735c07fe55b678809de4d5efd95670803502267da7a0bee5bb2f`

The method is catalogue reconstruction with a WM gain gate. It must not be labelled the full G-WM method E. Real fitting, device execution and scientific results remain NOT_RUN.
