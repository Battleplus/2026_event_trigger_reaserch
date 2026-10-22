# WM Action Recovery Scope

## Finding

The saved public artifacts cannot unambiguously reconstruct the WM five-value
action vector for a raw physical tick. In particular, they do not record the
per-actor movement command vector, normalized magnitude, normalized action
duration, or an explicit WM action record. These values must not be inferred
from the next tick's position/energy delta or from later branch outcomes.

## What Is Observable

The event stream distinguishes parts of the simulated command lifecycle:

* `transmission_sent` is emitted when a desired movement command passes the
  dropout check and enters the pending queue. It records actor, optional
  `decision_id`, send/effect times, but not the command vector or a unique
  command-instance ID (`formal_collection_v1/run_capture.py:545-555`).
* `command_dropped` is emitted on a failed attempt with forced-outage detail,
  but has no `decision_id` or command ID (`run_capture.py:545-556`).
* `command_delivered` and `command_applied` are emitted when a due queue item is
  popped and assigned to the executor's active target. Their records carry an
  actor, optional `decision_id` and timestamp, but no target/action values
  (`run_capture.py:525-536`). `command_applied` therefore identifies an
  executor state transition; it does not expose a serialized movement vector.
* The trajectory row records `command_attempted`, accepted/drop status,
  `command_delivered`, communication safety hold, and configured communication
  delay (`run_capture.py:624-628`). `command_attempted` must be checked because
  the no-attempt default is `(accepted=True, forced=False)` (`run_capture.py:600`).
* Known safety-state evidence is available separately. Communication outage
  records a hold event with the held position and pending-queue clear
  (`run_capture.py:509-519`); energy reserve creates a local hold
  (`run_capture.py:501-506`); physical rows retain fault and safety state
  (`run_capture.py:607-628`). This can identify some hold intervals, but does
  not define or populate the WM action field `held`.

## Five-Value WM Encoding

The current feature encoder maps an explicitly supplied `frame.action` to
`[known, applied, normalized magnitude, normalized duration, held]`. A mapping
sets `known=1`; absent action data produces an all-zero action vector, whose
first component means unknown and must not be interpreted as NOOP
(`wm_ppo_v1/features.py:180-186`). The public projection converter only reads
`projection.applied_action` (`wm_ppo_v1/public_dynamics.py:180-185`), but
`formal_collection_v1/state_sampling.py:19-40` emits only time, aircraft and
public observations. It has no `applied_action` field. The raw-tick adapter
also explicitly creates every physical history frame with `action=None`
(`wm_ppo_v1/raw_public_windows.py:206-210`).

Thus the components are only partially recoverable:

* **Known/applied:** event and trajectory fields reveal attempted, dropped,
  delivered and executor-applied transitions, but do not provide a single
  action-vector record or command-instance join. An absent attempt is not an
  applied NOOP.
* **Held:** safety hold state is observable for communication and energy
  holds. It is not recorded as the `held` component of an action vector, and
  it does not cover the full action semantics.
* **Magnitude:** not recorded. `transmission_sent` omits the `desired` target
  list that was queued (`run_capture.py:538-555`).
* **Duration:** not recorded. Trajectory `dt_s=1.0` is the physical step
  duration; communication `delay_s` and `command_effect_sim_time_s` describe
  delivery latency, not action duration (`run_capture.py:552-555,607-628`).

There is also a separate semantic gap: the source and WM feature schema do not
define how a waypoint/hold command's `magnitude_norm` or `duration_norm` should
be computed. The feature encoder only clips an absolute magnitude and a supplied
duration to `[0,1]`; it does not define their units, reference scales, or the
control interval over which they apply (`wm_ppo_v1/features.py:180-185`). A
future schema may define a fixed one-second action horizon, but that choice
requires explicit protocol versioning and model retraining. Neither missing
fields nor undefined semantics can be repaired by backfilling old all-zero
unknown-action vectors as known NOOP actions.

## Candidate Action Versus Movement Command

The catalog action is a high-level choice: NOOP or a task transfer. A selected
replan is queued with a planning/effect time (`run_capture.py:96-100,487-494`),
then validated and applied to queues/assignment at that later time, or rejected
as stale (`run_capture.py:342-390`). Successful transfer logs a
`donor_support_transfer` linked to the decision ID. This changes mission
allocation; it is not itself a per-second movement command.

The movement command is a separate per-actor waypoint target, derived each
tick from the active route or service target, enqueued subject to communication
delay/drop, and passed to `backend.step(commands, 1.0)` (`run_capture.py:537-558`).
On a successful transfer, the decision ID may be assigned to the recipient and
later copied to its movement transmission; this is only a partial causal link
and is absent for drops (`run_capture.py:375-386,530-555`). A transfer may also
cause no new movement transmission on a given tick. Consequently, candidate
choice and movement-command lifecycle cannot be collapsed into one recovered
five-value action.

Suffix workers call the same `run_capture.run` implementation with a restored
state and branch-specific output (`formal_collection_v1/run_suffix_pairs.py:84-100`),
so their public events/trajectory have the same action-field limitation. Raw
dynamic windows group rows by `(branch, sim_time_s)`, require adjacent ticks,
yield the prior physical history, and construct frames with `action=None`
(`wm_ppo_v1/raw_public_windows.py:288-311,410-430,206-210`). Their one-step
targets use adjacent state differences and public event joins, not action
reconstruction (`raw_public_windows.py:213-270`).

## Minimal Future Recording

For a future source version, use one action-recording hook where the final
per-actor `commands` mapping is frozen immediately before
`backend.step(commands, 1.0)` (`formal_collection_v1/run_capture.py:557-558`).
Carry that record into the corresponding actor trajectory row. The versioned
record should contain:

* `action_schema` and actor/tick identity;
* explicit `known`, `applied`, and `held` booleans;
* `magnitude_norm` and `duration_norm` using a frozen, documented scale;
* a unique `command_id` and optional causal `decision_id`;
* an optional high-level `candidate_action_id` kept distinct from the movement
  command identity.

When action values or their scale are unavailable, record `known=false` and
leave the action unavailable; do not fill zeros and label it NOOP. Keep the
existing lifecycle events, adding the same `command_id` to sent/dropped,
delivered and applied events so attempts and effects join without assuming
one command per decision. The high-level candidate log remains the source for
NOOP/transfer semantics; the new per-tick record describes the movement action
that the executor actually presents to the simulator. A later parser version
can map this record into `PublicHistoryFrame.action` without deriving it from
future physical outcomes.

This is a source-level field audit only. No formal trajectories, private
snapshots, hidden truth, live processes or remote data were read.
