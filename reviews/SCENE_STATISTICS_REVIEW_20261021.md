# Scene statistics source review

Reviewer: coordinating assistant, read-only review of the implementation by
GPT-6 Luna. A verified runtime model identifier is not exposed; no GPT-6.1 Sol
High review or cross-model-family independence is claimed.

Decision: ACCEPTED_FOR_NUMERICAL_SUMMARIES. This supersedes the repair-round
findings only for the frozen revision below. No scientific result is accepted.

## Frozen revision

| File | SHA-256 |
| --- | --- |
| compare_methods.py | 1b98ffed44b9612956a2ffb57935b1db837f6e059c4f469a4e182424d9354c47 |
| test_compare_methods.py | 5559d908d637999d3d8a481785a7b59df676b55bbfaa33d636c9f39f5d213edd |
| README.md | 136d70af2ff157192b69617f6f8cdd80677bd83e7c499b45d1eb55422c3097d6 |

Hashes were independently reread from the saved files. Review covered identity
grouping, repeated training seeds, missing reductions, available-pair iteration,
metric direction, sample SD, 10,000 scene bootstrap resamples, paired random-sign
permutation and a single declared within-split Holm family. Luna reported three
focused synthetic tests and syntax compilation passing; these are arithmetic
and implementation checks, not actual experiment measurements. The coordinator
did not repeat the tests or change implementation files during review.

Incomplete registered seed groups and duplicate baseline rows are excluded with
reasons. Parent-family templates do not replace independent scenario IDs.
Completion improvement is method minus baseline; cost improvement is baseline
minus method. Confidence intervals explicitly remain unadjusted. Numerical
threshold flags do not produce SUPPORTED.

## Remaining evidence limits

- No actual closed-loop evaluation, test/OOD confirmation or gain training has
  run through this comparator.
- Forced-safety completeness requires explicit independent episode-reducer
  evidence. The current static reducer contract does not yet supply the full
  `all_forced_safety_actions_applied` evidence; its absence remains unknown.
- The final evaluation registry must name the complete expected neural seed set
  for every neural method. Otherwise missing unregistered fits cannot be inferred
  from a list of observed runs.
- A complete registered comparison family, full safety evidence, independent
  test/OOD and the original method criteria still govern final conclusions.
- Metric summaries may retain available local values from imperfect inputs;
  they must not be interpreted as a complete experiment or a support decision.

The collector, frozen labels and ongoing formal batch were not modified.
