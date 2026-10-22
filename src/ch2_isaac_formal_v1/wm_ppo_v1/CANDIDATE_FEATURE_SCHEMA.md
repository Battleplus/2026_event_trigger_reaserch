# Candidate Feature Schema v1

`candidate_features.py` builds actor-choice descriptors from one current public
projection and a catalog that an upstream caller has already declared legal.
It consumes the catalog in its existing order and does not enumerate actions,
run the planner, or inspect labels, outcomes, hidden truth, or future state.

The supported action keys are `("noop",)` and
`("uav_transfer", donor, recipient, task_id)`. Those identifiers are used only
for public joins and exact action mapping; they are never converted to feature
numbers or hashes. Duplicate keys, unsupported candidate kinds, catalogs
without exactly one NOOP, and catalogs above 16 entries raise `ValueError`.
The candidate mask marks every supplied catalog entry valid. `noop_mask` marks
the sole NOOP entry. `actions[i]` preserves the supplied action mapping at
`action_keys[i]`.

The actor feature order is given by `FEATURE_NAMES`: action-kind one-hot fields,
donor and recipient public aircraft-kind indicators, raw-unit numeric fields,
then a known bit for each numeric field. Numeric fields use only fields present
in the input: aircraft position in meters, home zone, available energy in Wh,
queue length and public availability/fault status; home zones are divided by
eight; the donor task's position in
the donor's public queue; and catalog-provided task ETA, route distance, energy,
gain, and switch cost. Candidate metric units are seconds, kilometers, and Wh
as named. Missing or malformed values are zero-filled with known bit zero.
No public task coordinate/type metadata is currently emitted by the projection,
so this encoder does not invent it. Symmetric public states may produce equal
feature rows; stable action keys still preserve their action mapping.

This schema is actor-selection input only. `wm_action_summary` remains `None`
and `wm_action_known` remains false because a high-level NOOP/transfer choice
does not establish the five-value per-tick WM action vector. Do not use these
candidate features as action-supervised WM input or executed-action labels.
