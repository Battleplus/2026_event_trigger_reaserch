# compact_collection_v1 runner integration proposal

This package is a local, synthetic-tested storage interface. It is not wired
into the current runner and has not been used for an Isaac job. The writer
requires an already-public projection as `public_state`; known private field
names are filtered as defense in depth, but arbitrary private fields cannot be
identified automatically. The runner must keep public and evaluation-truth
inputs separate.

At the runner's existing public projection boundary, construct one
`CompactCollectionWriter` in a new per-run directory, passing the immutable
shared public configuration once. On every existing one-second sampling
callback, call `append_tick(public_state, events=..., decisions=..., costs=...,
event_log_complete=...)`. Keep the runner's current sampling cadence; this
writer rejects missing seconds instead of silently resampling. Send hidden
evaluation truth only through the separate `hidden_truth` argument. Record
paired keep/replan public snapshots through `append_pair_snapshot`, including
the external pair engine's full-state snapshot reference and digest. These are
complete public projection snapshots only. The pair engine remains responsible
for full keep/replan recovery state, exogenous suffixes, and immutable
references. The public reader does not read paired snapshots or the
evaluation-truth file.

The public runtime projection currently has aircraft state, aircraft task ID
assignments/queues, and public observations. It does not currently supply task
position, allowed vehicle types, priority, deadline, or geometric progress.
The writer stores fields only when the runner actually provides them; it does
not add defaults or infer task geometry. The present formal artifacts contain
at most eight sparse projections, not a continuous one-second sequence, and
cannot be represented as satisfying this stream contract without new capture.

After capture, `CompactPublicReader.iter_public_states()` reconstructs
one-second public projections. For any tick, `public_observations` includes
only observations with a finite `available_at_sim_time_s <= sim_time_s`, even
if the source payload already contains future observations. Thus equality is
with the sanitized source state after availability filtering, not with raw
payloads that included not-yet-available observations. The
`iter_public_one_step_examples` adapter feeds contiguous reconstructed states
to `wm_ppo_v1.public_dynamics.build_public_one_step_example`.

The runner integration should be reviewed as a separate change before any
future Isaac run. This proposal does not change the active batch, collectors,
or existing data.
