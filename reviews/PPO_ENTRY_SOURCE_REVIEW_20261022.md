# PPO real-environment entry: scoped source review

Reviewer: primary Codex coordinator (GPT-6 family; exact service model not independently attested).
Implementer: GPT-6 Luna. Review is read-only with repairs returned to Luna.

Decision: SOURCE_ACCEPTED_WITH_ISAAC_GPU_NOT_RUN.

The three interim defects are corrected in the inspected source: next-value
bootstrap shifts recorded decision values; training seeds initialize separate
models/optimizers/checkpoints; scenario seeds remain bound to the 84 frozen
registry training configurations, with identical scene order across seeds.
Each seed reports its actual completed/failed episodes and updates separately.

Four focused synthetic tests are reported passing by the implementer. The
review inspected their coverage: gate refusal before setup, hand-calculated
nonzero-value two-decision GAE, reward-sum agreement with the independent
geometry/cost reducer, and two-candidate CPU actor/value updates plus state
checkpoint restoration. These are CPU fixture checks, not GPU/Isaac results.
No duplicate history suite was requested.

The public runtime callback captures histories, legal catalogue, selected
candidate, log probability, value and time. Learning rewards use evaluated
post-action outcomes; they are not actor features. Reward intervals are
[t,next_t), ending D+1; the prefix is separate. Normal complete episodes alone
receive terminal -1. The registered planner-only objective used for training
is distinct from full online CPU costs used for final method evaluation.

Remaining dependencies: terminal old/raw + new/delta union reduction and
public feature binding; post-extraction 30/30; genuine train-only fitted policy
normalization; real GPU1 rollouts/updates; new-process prediction recovery;
policy-specific gain relabeling after policy freeze; independent closed-loop
results. Existing logs do not yet support genuine action-conditioned G-WM
auxiliary supervision, so that path remains UNAVAILABLE_G_EQUIVALENT.

This review accepts source scope only. It does not establish formal three-seed
training, world-model added value, or method support. The running collector
was not modified by this source review.

## Frozen reviewed files

- `wm_ppo_v1/isaac_policy_training.py`: `f35a753d76866425dfea92d84e6794f70bdfcb77758467f4f22827d61629f4a1`
- `wm_ppo_v1/tests/test_isaac_policy_training_synthetic.py`: `9e035cb228d2f4b46509eb58b4cec1b3e51780e2a2ef87c31617c270447508cc`
- `wm_ppo_v1/policy_choice.py`: `133197650ad772e3f30edf143e280e325d259ed2f802facf8b703251d7960e57`
- `closedloop_v1/runtime.py`: `399e9242b40bc03dcd50bda50acf6fecda5a60af0fe541e0442b83e4baf5e1ff`
