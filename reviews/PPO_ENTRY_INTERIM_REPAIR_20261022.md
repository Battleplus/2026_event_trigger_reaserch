# PPO Isaac training entry: interim read-only review

Reviewer: primary Codex coordinator (GPT-6 family; exact service model not
independently attested). Implementer: delegated GPT-6 Luna.

Scope: interim source inspection of `wm_ppo_v1/isaac_policy_training.py`,
compared with `ppo_core.py` and the independent episode reducer. The file is
still under implementation. No real Isaac rollout, GPU optimizer update, or
trained-checkpoint recovery is asserted by this review.

Decision: REPAIR_REQUIRED. Findings were returned to Luna before fitting.

1. `_update_from_episode` passes zero next-values for all transitions. For
   nonterminal transitions, use the next recorded public decision value;
   only a genuine completed episode's final transition bootstraps zero.
   With gamma=lambda=1, zeroing every bootstrap still produces incorrect GAE
   by repeatedly subtracting later value estimates. Verify against a small
   hand-computed multi-decision example with nonzero values.
2. `run_g_training` constructs one model and optimizer outside its seed loop.
   This turns seeds 8201/8202/8203 into episodes in a single training run.
   Each training seed requires independent initialization, optimizer state,
   output identity and checkpoint, with the same frozen training-scene order
   and sampling budget across repetitions.
3. Assigning the training seed to `config['seed']` replaces the frozen scene
   identity with a newly generated scene. Use only registry-bound training
   scene configurations and preserve their scene seeds. Training seeds govern
   model initialization and policy sampling, not split membership.

Follow-up scope clarification: the default formal training sequence must use
the frozen registry's 84 training configurations in the same deterministic
order for all three independently initialized training seeds. Selecting only
the first scene in each condition/source group (21 scenes) was not accepted as
the default formal experiment. Such a subset, if ever used for debugging,
must have a separate explicit identity and cannot establish matrix completion.
Receipts must report completion and updates per seed; one successful seed
cannot establish successful completion of the three-seed experiment. Failed
episodes retain their evidence and cannot receive a fabricated terminal reward.

The inspected reward expression correctly includes sent plus dropped messages
and both donor and recipient in the episode-wide unique changed-actor set.
The planner-only training subtotal must remain separate from the alternative
full online CPU evaluation subtotal. Rewards remain attributed to [t,next_t),
with final interval ending D+1 and pre-action reward retained separately.

No change to the running formal collector or existing evidence was requested.
