# PPO core interim source review

Status: SOURCE_ACCEPTED_WITH_ISAAC_GPU_NOT_RUN after follow-up repairs. Coordinator read-only review of Luna's `ppo_core.py`; no formal fitting or remote source change.

Initial source includes disjoint graph actor/critic parameters, masked categorical candidate selection, elapsed-time GAE with terminal/truncation handling, clipped policy/value losses and a frozen-WM auxiliary target path. Final source hashes and focused test results are still pending.

Requested repairs before source freeze:

1. Active graph edge endpoints must not be silently clamped into different valid nodes. Reject invalid active indices; ignore padding through its explicit mask.
2. Separate the actor's public candidate description from the five-dimensional WM action encoding. Provide candidate WM action vectors separately, index them by the actual executed candidate, and verify their binding. Actor candidate descriptions may require more dimensions to distinguish public task-transfer alternatives.
3. Document that dynamic target scales must come from training-only dynamic outcomes. Current WM training stores node-feature normalisation, which does not establish dynamic target scales or known-action joint support. Synthetic fixtures may provide explicit scales/masks, but this does not establish formal-data availability.

Live batch observation at 2026-10-22 08:13:45 UTC: 88/168 COMPLETE scene receipts; collector and capture subprocess alive; no terminal batch receipt or aggregate; free bytes 22,629,199,872. The batch was not restarted. No 30/30 gate or model fitting was performed.

## Follow-up source inspection

The candidate WM action tensor is now separate from the actor candidate descriptors; the synthetic update uses seven descriptor dimensions and five WM action dimensions. Active edge endpoints are checked before padding-safe indexing. The auxiliary path requires explicit training-only dynamic scale evidence rather than reusing node normalisation. The reviewed test source contains a hand-computed elapsed-time GAE example, illegal-candidate zero probability and over-limit rejection, plus a real synthetic CPU optimiser step checking actor/critic parameter changes, frozen WM parameters and nonzero critic auxiliary gradients. Final execution results and frozen hashes remain pending, so this follow-up does not accept or certify the running implementation.

Latest read-only batch observation at 2026-10-22 08:16:44 UTC again confirmed both collector and suffix subprocess live, 88/168 completed, no terminal receipt or aggregate, and 22,515,077,120 free bytes.

## Frozen source acceptance

Luna reports the three focused synthetic CPU tests passing and compilation of the two changed Python files passing. The coordinator inspected the exercised formulas and gradient assertions and recomputed all four submitted hashes. This accepts the source core for later adapter integration; it does not establish a complete PPO trainer, checkpoint recovery, real Isaac rollout, GPU updates or method efficacy. With no valid auxiliary elements the update is explicitly `UNAVAILABLE_G_EQUIVALENT`, not an executed G-WM auxiliary experiment.

- ppo_core.py: `2bf0c81d808840d8d193dc43c0ba10b53d89ce1d0c89d9965c7c6e9f9693e637`
- tests/test_ppo_core_synthetic.py: `8a9f0e6e344ea6b8f3cce6ecbc93a17f3383994079b02eee436fbe15894e7437`
- PROTOCOL.md: `ef9230b614da07abb141f548ffa485748815467a2c734887f067a59d8d6eed81`
- README.md: `bb27e296a556e7f375061b68eaa72a348791cdaeeb9c2abf94448ab27c8efa90`
