# WM training path: partial source acceptance — 2026-10-22

Decision: ACCEPT_PARTIAL_SOURCE. Implementer: GPT-6 Luna. Reviewer: coordinating Codex agent; actual model identifier unavailable. Source review was read-only. This does not establish GPU training, scientific validity, deployment readiness or PPO completion.

## Evidence inspected

The coordinator inspected wm_entry.py, trainer.py, checkpoint.py, restore_worker.py, the modified synthetic fixtures, README and PROTOCOL and matched their file hashes to Luna's frozen delivery. The world-model stage now calls a real gated trainer; G/G-WM/PPO remain explicit NOT_RUN. Terminal formal reduction, feature coverage and decision-window coverage precede fitting. Training and validation sources are isolated by scenario. Normalization uses train histories only; physical adjacent-state and decision-gain phases are separate. The entry writes the requested receipt path.

Luna reported 13 focused synthetic/public-dynamics/event-graph tests and compilation passed, plus a G NOT_RUN command check. The coordinator did not repeat these tests. A fresh CPU process compared fixed predictions and synthetic nonempty Adam state; the fixture injected state tensors without an optimizer update. Two existing decision/raw-window tests failed during broad discovery because their temporary directory creation was denied. They are not claimed as passing in this run. Their prior focused evidence is historical, not a new full-suite pass.

## Repairs verified in source

- Standard top-level package imports work via the existing import convention.
- GPU1 UUID selection is checked before full dynamic normalization scans, after the necessary data/split gates.
- Checkpoint payload loads on CPU so CPU RNG tensors remain CPU tensors; model/optimizer/probe are restored for the requested device.
- The formal trainer requests a fresh-process probe on the same CUDA device, not an unconditional CPU probe.
- Checkpoints contain model, optimizer, normalization, data/schema identity, architecture, device and RNG state; prediction tolerance is explicit (1e-5 absolute/relative).

## Remaining evidence and implementation gaps

No actual formal file extraction, optimizer update, GPU memory/timing, CUDA restore, training convergence, gain calibration, deployment integration or closed-loop WM result was run. Data gate is still unavailable. Normalization and dynamic loading stream scene batches, but repeated gzip scans may be expensive and need actual resource receipts. Actual per-target supervision counts are emitted by training/evaluation; they have not been measured yet.

The code restores RNG state; the current fixture does not independently compare resumed random draws or a resumed optimizer update. Reports must not promote this to demonstrated training continuation. Prediction/optimizer consistency alone is narrower evidence.

There is no action-complete causal simulator here: raw action/queue/availability fields have known gaps, aggregate dynamic targets retain the limitations in WM_SUPERVISION_SCOPE_REVIEW_20261022.md. Validation gain calibration and an online WM adapter still need implementation before method E can be evaluated. G/G-WM/PPO remain outstanding.

## Frozen SHA-256

- wm_entry.py: fd2d13fe4076d4a99bbbaf432f3c19f25d802460116d8ac0ee4fb8760366cf34
- trainer.py: 9cab350a61b934ec6e19910c14764eb0ec211986c6efdf592e8133f61cd17251
- checkpoint.py: 6fa814e7e784a6f354a05610fae0cd4c69118225aa99548436da5fc37c7710ee
- restore_worker.py: d873391fa99ad6e8d4e533ea3659452e0a99417055c54e67980915e2cb6721dd
- tests/test_synthetic.py: 652ce0d8ca3fdbab7e33a2247d681155332457a92e792bd651229174b0a93045
- README.md: 104343b0743cf443cf5a238d6941e4593728ceef272c149a3d2565c316ed8e14
- PROTOCOL.md: 9480638a792ea3096fdd8a115493598a1b6407a01c0cdb7cdd7debedaf5542a9
