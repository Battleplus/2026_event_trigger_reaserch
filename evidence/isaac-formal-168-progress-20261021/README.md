# Formal Isaac collection: partial progress, not a result

At 2026-10-21 09:08:37 UTC, the original formal-168-v1 batch was live with
37/168 scene execution receipts marked COMPLETE and zero failed receipts.
These receipts establish execution progress only; the independent final
reduction, 30/30 gate, training and scientific comparisons remain pending.

Completed condition counts: light delay 12, task activation 12, platform fault
12, energy anomaly 1. Training/validation collection is still in progress;
do not infer label signs or method performance from these counts.

The batch proceeds directly through the registered scenes. The user removed
separate representative, single-state and pilot prerequisites. There is no
retry, seed replacement or partial-data training. The existing runner performs
the independent suffix reductions and aggregate once acquisition ends.

Capacity observation: the batch occupied approximately 36GiB at 37 completed
scenes; server free space was 80,818,614,272 bytes at the observation above.
Linear extrapolation suggests around 163GiB for the full batch, so existing
capacity may be insufficient. This is an estimate, not an observed failure.
No evidence was deleted and the active run was not restarted or modified.
An additional existing storage location was requested from the user.

Method: INCONCLUSIVE. World model/PPO: NOT_RUN. GPU model training: NOT_RUN.
No successful experiment milestone tag is created by this progress archive.
The accompanying SHA256SUMS covers the sanitized progress observation.
