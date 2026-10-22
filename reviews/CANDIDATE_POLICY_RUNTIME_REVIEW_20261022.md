# Candidate policy runtime source review

Status: SOURCE_ACCEPTED_WITH_ISAAC_GPU_NOT_RUN. Implementation by GPT-6 Luna; coordinator read-only source, focused-fixture and hash review. The current formal collector was not changed.

The optional runtime callback acts on an already generated catalogue after an ordinary gate opens. Forced local safety and suppressed ordinary gates skip the policy. A configured callback enables physical history updates every second; the API requires a contiguous public history ending at the current projection, explicit train-only node/candidate normalisation and matching feature schemas. Evaluation resolves equal probabilities by stable action key. Sampling returns the original catalogue action with its index/key, log probability and value.

Known history or device prerequisite failures are recorded as unavailable without falling back to the sorted public choice. Action mapping, model and numeric errors propagate to the existing run failure receipt. The broad intermediate exception handlers were removed during review. History assembly and selection are timed together; catalogue plus selection timing is a nested diagnostic, while the existing exclusive online phase remains the cost-total source. Selected actions use existing delay/stale-action execution. Compact logs retain the policy-selection event and its mapping/timing metadata.

Luna reports 10 focused synthetic checks passing and compilation passing. The new compact policy-event fixture passed separately; an existing temporary-directory compact test failed with a Windows PermissionError and is not claimed as passing. The coordinator inspected the six policy-specific tests and the modified call sites, then recomputed the six frozen file hashes. This is source acceptance only; no full runtime simulation, GPU inference, trained policy artifact, PPO checkpoint recovery or efficacy is established.

Frozen SHA-256:

- closedloop_v1/runtime.py: `cb57fcab6eac2c4480f08278d39575bb4414f533b12a369273f526e74e5bda1e`
- closedloop_v1/compact_capture.py: `fbfc73b02dd1ac3cb3eaf0d94ce9add23db2d1b4179a2e4eb17587e71e60f3da`
- closedloop_v1/test_compact_capture.py: `d42259bbc95666807359ea3d43d3d1fa7cc10b2519036e340f8d276394143997`
- wm_ppo_v1/policy_choice.py: `37834d29fc23164791c62feea812dd83857527c4cafc799908ee798a87dc72ff`
- wm_ppo_v1/tests/test_policy_choice_synthetic.py: `0c5b6ddf7a18be16d26db17a318dd9656222bb724b02c5b3ec0a680f3bd8b8e9`
- wm_ppo_v1/CANDIDATE_POLICY_INFERENCE.md: `a521450d3eb30f65dfe5d843d774342e5aadebd107a71266696b3af5c98e21d3`

Model checkpoint loading, fitted train-only normalisation provenance, actual Isaac rollout collection and GPU PPO training remain outstanding. Current raw data still do not supply action-conditioned WM supervision; this callback does not repair that limitation.
