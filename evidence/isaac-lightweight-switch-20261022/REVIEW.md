# Lightweight collection switch: limited review

Implementation: GPT-6 Luna. Review: primary coordinator in this conversation.
The coordinator reviewed source and evidence without repairing implementation.
No other reviewer model identity is claimed.

The actual Isaac collector now writes public one-second deltas, five-second
views with decision timestamps, static geometry once, and separate evaluator
truth. Private keep/replan trajectories use a lossless gzip field-delta codec.
Post-step views are labelled as post-step; the existing sampled pre-candidate
public snapshots remain the gain-model decision inputs.

Seven focused tests passed. A complete existing suffix of 129600 rows was
encoded from 30231759 to 3077686 bytes, a reduction of 89.82%. Reconstructed
canonical hashes, independent branch costs and G matched exactly. The observed
G of -0.0045143752 is a single diagnostic label, not a method effect.

The initial review found the old independent evaluator incorrectly expected
motion_model_state.position. Actual Isaac snapshots store MotionState under
motion_model_state.state. A separate I/O adapter exposes that existing position
only after the original snapshot digest is checked. Original snapshots,
restoration hashes, geometry calculations and scalar cost formulas are unchanged.
The initial UNKNOWN outcome was retained; after the adapter both raw and delta
inputs received PASS for this suffix. This does not audit every old suffix.

Old acquisition stopped at a completed-scene boundary: 92 complete scene
receipts, no old worker child. Its progress log contains 91 entries because the
parent paused immediately before appending the 92nd; the scene receipt is kept.
The new batch references those 92 directories and selects the other 76 original
registered scenes. It retains the same configurations, seeds and suffixes, with
no recollection, replacement seeds or automatic retries.

Remote readback confirms actual Isaac execution, 36 actors, 81 public tasks,
nine responsibility zones and first public seconds reconstructed as 1, 2, 3.
At the recorded observation the new scene reached simulation second 2940.
GPU1 was active; GPU0 remained at 9 MiB. Source/config hashes were checked before
launch. This is limited format and startup acceptance, not full-scene or method
acceptance. Full old/new union reduction and the 30/30 gate remain pending;
models, world model and PPO remain NOT_RUN. No success milestone tag is created.

Later direct readback at 2026-10-22 10:12:03 UTC: the first new 90-minute
representative episode completed with 5400 one-second public records and
194400 actor rows represented in the lossless trace. Full public reconstruction
found zero prematurely available observations and retained five suppressed
ordinary trigger decisions. Its eight sampled states are entering paired suffix
acquisition. This does not yet establish a complete paired scene or a 30/30 gate.
