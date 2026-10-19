# Protocol Amendment: Single Terminal Remaining-Work Penalty

Amendment ID: `gain-gate-cost-amendment-20261017T045635Z`  
Applies to: `gain-gate-v1-100x100`  
Issued: `2026-10-17T04:56:35Z`  
Status: pre-collection protocol correction; awaiting root read-only review.

This amendment supersedes only the terminal completion-cost paragraph in
`CH2_GAIN_GATE_PLAN_20261017T040502Z.md`. It does not alter the paired-state,
scenario, split, model, safety, or statistical protocol. The earlier timestamped
plan remains preserved as the record of the prior design.

## Frozen terminal completion accounting

For each logical task in the paired window task set, let `p_i` be its normalized
priority weight, and let `r_i` be its remaining-work fraction at the window
endpoint, clipped to `[0, 1]`. Charge terminal remaining work once:

```text
completion_cost = completion_weight * sum_i(p_i * r_i)
completion_weight = 1.0
```

Do not add a separate `1 - weighted_completion` or other binary non-completion
term to `J`. Report weighted full-completion rate separately as a quality
outcome:

```text
weighted_full_completion = sum_i(p_i * 1[task i is fully complete at endpoint])
```

The quality metric is not added to the cost a second time. Remaining work is an
endpoint quantity, not a per-minute cumulative charge. Existing rules continue
to use the same task set and priority weights for both paired branches; an
exogenous cancellation before completion is removed from both denominators.
The delay, energy, and risk terms remain separate components of `J`, with their
existing design weights and raw physical quantities retained.

## Data identity and evidence handling

The cost implementation and its output schema must carry a new version/hash
identity before any training-pair collection. Labels produced under the prior
binary-plus-residual formula are incompatible with this amendment and must not
be mixed into the new dataset. Any smoke output generated before the updated
implementation is re-run remains immutable historical evidence; it must not be
overwritten or silently relabeled.

The first collection run must record the amendment ID, the updated cost/schema
version, and hashes of the exact production cost code and independent oracle
used for the run. The independent oracle must independently reproduce the
single terminal charge from raw traces.
