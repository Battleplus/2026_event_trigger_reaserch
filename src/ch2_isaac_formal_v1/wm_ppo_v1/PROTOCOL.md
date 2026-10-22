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
