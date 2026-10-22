# Scene statistics implementation review: repair round

Reviewer: coordinating assistant, read-only source review. The runtime does not
expose a verified reviewer model identifier; no GPT-6.1 Sol High or cross-family
independence claim is made. Implementation is assigned to GPT-6 Luna.

Scope: `closedloop_v1/compare_methods.py` and its focused arithmetic test.
This is an in-progress repair review, not acceptance of a frozen revision and
not evidence from experimental episodes. The live collector is unchanged.

## Required corrections returned to Luna

1. Never infer method SUPPORTED from one split's numerical thresholds. Full
   safety, independent test/OOD and model-value evidence remain separate.
2. Use independent scenario identity, not a shared parent-family template, as
   the inference unit. Preserve missing or malformed registration exclusions.
3. Average registered training seeds once within scenario/method. Duplicate
   seeds, missing preregistered seeds and repeated seedless baselines must not
   receive extra weight or be silently treated as complete crossed repeats.
4. Report scenario-level sample standard deviation and a single declared Holm
   family alongside paired bootstrap intervals and sign-permutation results.
5. When incomplete groups are excluded, downstream metric and violation loops
   must use available pairs rather than indexing excluded scenario aggregates.
6. A stationary-hold check does not establish 100% forced-event handling. Label
   the available safety evidence with its actual scope and leave the broader
   requirement unproved.

Status: REPAIR_REQUIRED pending Luna's corrected frozen source and focused
verification. No simulations, training, partial-data gain analysis or historical
test-suite reruns were requested by this review.
