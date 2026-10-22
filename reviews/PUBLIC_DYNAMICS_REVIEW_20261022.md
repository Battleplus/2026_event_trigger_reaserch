# Public one-step interface source review

Coordinator read-only review of GPT-6 Luna implementation. No verified reviewer
model identifier is exposed; no GPT-6.1 Sol High or cross-family audit is claimed.

Decision: ACCEPTED_AS_PUBLIC_ONE_STEP_INTERFACE, not a trained world model or a
validated real dataset. The ongoing formal collector was not changed.

## Frozen SHA-256 independently read from saved files

| File under wm_ppo_v1 | SHA-256 |
| --- | --- |
| public_dynamics.py | ceec6cdf8d4bd1134abe24e958e52b44120c6bce15fd903e704f025e5bbfc396 |
| contracts.py | a6416497d3bc4720b4f61599ca016d41dbf000ea7b097d039d1f23af3a009ad7 |
| features.py | 74908a1d3692779cc8174334bf5c4fc435264caf8288a7965b3ce728dfc1352b |
| tests/test_public_dynamics.py | f19e6c8f1e75a6fccb8b4c0f9fe32b4f49ddbe5f2bf98fb9f25ea057b41a4ca0 |
| README.md | 9c852b1fd1cc3cb2eaba06a1fd305a5cbcb01922ddc4ec63e6a251bc67ff087d |

Review covered the explicit public projection whitelist, observation availability
filter, adjacent one-second history, held-out final snapshot, separate gain
supervision, target masks and anonymous numeric features. The repaired zone
normalization preserves zones 4 and 8; ambiguous repeated command transmissions
cannot create delay targets. Unknown queue/energy-fraction/priority information
has explicit known indicators rather than a claim of complete observations.

Luna reported five focused synthetic checks and syntax compilation passing.
The coordinator did not duplicate the tests or access partial formal gain data.

## Evidence limits and remaining requirements

- The formal sampler retains at most eight sparse ordinary decision projections;
  these alone are not adjacent one-second histories. No actual one-step dataset
  or GPU training is established by this interface.
- Task progress supervision here is public cell-completion event count, not
  per-task geometric progress. Delays are identifiable applied-command elapsed
  times, not all pending/right-censored command delays.
- Queue membership is represented by counts and position, without task-platform
  graph edges. Task position, priority, deadline and capability fields absent
  from the public projection remain unsupported.
- Current world_model.py uses independent node encoding, masked pooling and a
  GRU, not graph message passing. Frame event inputs remain empty; event labels
  provide supervision only. This does not finish the planned event-aware graph
  world model, action conditioning, real training or PPO adapters.
- Completed-data 30/30 and feature-availability gates remain mandatory before
  gain training. Actual public histories, masking coverage, checkpoint restore,
  closed-loop independent test/OOD and added-value evidence remain NOT_RUN.

Future acquisition follows protocols/COLLECTION_SCOPE_AMENDMENT_20261021.md:
retain necessary public changes and assignment queues without repeated static
scene or observation-list storage, and keep evaluator truth separate.
