# World model / G / G-WM / PPO protocol v1

## Scope

This is an interface and receipt contract only. The formal Isaac batch remains
the source of all data. This package must be copied into a new run identity
before any future training. No result in this directory is a formal result.

## Public graph and one-step supervision

At each time `t`, the controller projects currently available public state into
dynamic nodes for platforms, active/public tasks, and available observations.
The projection excludes hidden target coordinates before discovery, future event
tapes, private snapshots, scenario identity hashes, and branch outcomes. Node
counts may vary by frame; padding is accompanied by node and sequence masks.

The world model predicts at `t+1`:

* task progress;
* energy delta;
* delay;
* public safety/fault risk events, including exogenous faults and safety holds.

These are true one-step targets collected from adjacent physical simulator
states. The gain head predicts the paired `G` label only when an independent
keep/replan reducer marks it valid. The 30-minute scalar in the frozen scalar
protocol is not a substitute for the one-step dynamic targets.

## Fixed methods

| ID | Trigger | Gate | Replanner |
|---|---|---|---|
| A | original event | none | G |
| B | original event | light gain | G |
| C | original event | none | G-WM |
| D | original event | light gain | G-WM |
| E | original event | WM gain | G-WM |
| F | four-minute periodic | none | G-WM |

All six methods share safety forcing, platform/task legality, candidate rules,
and the same cost ledger. The implementation records CPU/GPU inference,
transfer, communication, command waiting, switching, energy and risk fields.
The frozen incremental cost is computed from signed paired differences:

```text
0.01 * messages_increment / 1000
+ 0.05 * switches_increment / 100
+ 0.05 * planner_calls_increment / 100
+ 0.01 * changed_actors / 36
+ 0.01 * planning_cpu_seconds_increment
```

## Freeze and restore evidence

Before any G/G-WM training, record the formal batch receipt, source/config
hashes, split and feature schema, world-model architecture, seed, device, and
the frozen replanner identity. The checkpoint receipt must include model,
optimizer, normalisation and data hashes. A fresh process must load the
checkpoint and produce identical predictions on a fixed public input; the
comparison receipt is `PASS` only when this is true.

The world model is frozen while G-WM is trained. The auxiliary critic must show
non-zero gradients from its declared one-step targets; actor and critic updates
must come from real Isaac episodes. The PPO entry remains `NOT_RUN` until these
dependencies exist.

## PPO core and G-WM auxiliary target

`ppo_core.py` is an optimizer component for a later caller-collected Isaac
rollout; it does not create an environment or make this protocol's PPO entry
run. The graph actor and critic have separate parameters and separate public
graph encoders. Candidate logits are masked before the categorical
distribution, every state must contain exactly one legal NOOP, and catalogs
above 16 candidates are rejected rather than truncated. Candidate feature
ordering is caller-owned and must remain stable within a rollout; public
descriptors may include donor/recipient/task and queue/energy effects. They are
separate from each candidate's five-value WM action summary in the existing
`[known, applied, normalized magnitude, normalized duration, held]` order. For
a known applied effect, the executed five-value summary must match the selected
candidate's WM summary. The caller supplies train-normalized public graph
tensors using the existing dynamic node and edge schema.

For a transition with actual decision interval `dt_s`, the discount is
`gamma_t = gamma ** dt_s` and the trace factor is
`lambda_t = (gamma * gae_lambda) ** dt_s`. A true terminal transition has no
bootstrap. A time-limit truncation bootstraps its supplied next-state value,
then stops the advantage recursion at the boundary. The policy objective uses
the clipped PPO ratio, the value objective uses clipped value error, and
entropy is subtracted from the total loss.

The G-WM auxiliary loss distills the frozen model's one-step dynamic prediction
for the actually executed candidate only. The executed action vector uses the
existing five-value WM action order `[known, applied, normalized magnitude,
normalized duration, held]`; it is passed as the final-frame action input to
the frozen WM under `no_grad`. The critic's auxiliary head predicts the four
dynamic outputs from the critic's public graph representation and executed
candidate features. The critic value and auxiliary heads share the critic
encoder; actor and critic share no parameters. If `frozen_wm(s, a)` is the WM
prediction, `z_critic(s, c)` is the auxiliary output, and `m_{i,k}` is the
explicit joint support mask, then:

```text
L_aux = 0.1 * sum_{i,k} m_{i,k} * ((z_critic(s_i,c_i)_k - frozen_wm(s_i,a_i)_k) / scale_k)^2
        / max(1, sum_{i,k} m_{i,k})
```

`scale_k` is a positive train-split dynamic-target scale. The effective mask
intersects the caller's checkpoint joint action/dimension support, the
checkpoint's supervised dynamic dimensions, a known applied action, and the
legality of the executed candidate. This is prediction distillation, not a
real dynamic label or imagined actor rollout. The existing WM checkpoint does
not record per-dimension joint action support, and the current raw trajectory
usually has no applied action vector. Until a future collector/checkpoint
provides those fields, the G-WM auxiliary mask can be empty and no auxiliary
learning is claimed. An update with no effective auxiliary mask reports
`UNAVAILABLE_G_EQUIVALENT`; it is a G-equivalent PPO update and cannot count as
a completed G-WM update. The caller must also provide positive per-target standard
deviations and evidence binding them to the train split: `scope`, target names,
estimator, exact scale values, dataset SHA-256, and valid target counts. The
current WM trainer writes no such dynamic-target scale evidence; it must not be
inferred from node normalization or validation/test targets. A synthetic CPU
fixture supplies explicit scales and support only to test the gradient path;
it is not formal data or GPU training. The G baseline uses the same critic
capacity with auxiliary coefficient zero.

## PPO gate

`wm_entry.py --stage world-model` is a conditional entry point: it requires the
terminal 168-scene batch gate, post-extraction 30/30 feature coverage,
independently labeled decision-gain coverage, non-overlapping scenario splits
and the explicitly selected GPU1 UUID. It trains the dynamic head from
adjacent raw post-step frames and the gain head from the separate pre-candidate
decision projection; these losses never share a label or phase. Normalization
uses train scenes only. The three fixed seeds are 8201, 8202 and 8203. Each
checkpoint carries model/optimizer state, train-only normalization, data and
schema identity, RNG state, architecture, device mapping and saved probe
predictions. Recovery is checked in a new process on the training GPU; there
is no CPU fallback for a training run.

No formal training was run for this implementation. G, G-WM and PPO remain
`NOT_RUN`. Synthetic tests cover interface behavior, gate refusal,
shape/masking, cost arithmetic and a CPU fresh-process checkpoint fixture;
they do not establish a scientific result or verify CUDA recovery.

The WM gain adapter is wired into the local `closedloop_v1` runner's existing
catalogue gate through the public-history bridge. The artifact's own
`integration_status` still identifies the adapter as not independently
integrated; the runner receipt records the actual integration scope. No live
Isaac/PPO run or method-E result was produced. Validation scenarios are split
with `light_models_v1.train_models._scenario_validation_split`;
only validation-selection scenarios choose epsilon, and only disjoint
validation-calibration scenarios fit the one-sided lower residual quantile.
Test and OOD rows are rejected. With alpha 0.1, fewer than nine calibration
scenarios leave the lower-bound mode unavailable; the mean mode remains
available. Inference requires an explicit non-empty `PublicHistoryFrame`
history and the phase `post_step_through_t_minus_1_plus_pre_candidate_projection_t`.
It enforces a maximum of eight contiguous one-second frames and strips actions.
It computes feature, CPU-to-device, synchronized forward, device-to-CPU,
calling-thread CPU time and end-to-end wall time, verifies the checkpoint and input schema identity, and reports
the mean, optional lower bound, threshold, version, device and
`NOT_INTEGRATED` artifact status. Its local runner integration is limited to
the gain gate before the existing candidate catalogue and does not establish
method-E results.
