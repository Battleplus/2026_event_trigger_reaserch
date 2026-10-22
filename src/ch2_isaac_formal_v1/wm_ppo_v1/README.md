# Public one-step data interface only

This scoped archive contains public-projection conversion, masked one-second
targets, variable-size node tensor collation and their supporting contracts.
It does not contain a completed world model or PPO implementation. No real
one-step dataset, GPU update, checkpoint or scientific benefit is established.

See reviews/PUBLIC_DYNAMICS_REVIEW_20261022.md for frozen source hashes,
independently reviewed scope, synthetic-check evidence and remaining work.

From the parent directory, with NumPy installed:

    python -B -m unittest discover -s wm_ppo_v1/tests -p test_public_dynamics.py -v

These are synthetic interface checks, not Isaac experiment results. Sparse
decision snapshots cannot be relabelled as adjacent one-second histories.
