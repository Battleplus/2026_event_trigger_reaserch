# V3 Status

## Current gate

Status: IMPLEMENTATION IN PROGRESS. No V3 smoke result exists yet. This status is not a pass.

## Frozen target

- Global map: 100 x 100, coordinates [0, 99].
- Regions: 3 x 3; boundaries [0, 33, 66, 100), with boundary points assigned to the higher-index cell and coordinate 99 assigned to cell 2.
- Fleet: 24 UAV + 6 helicopters; UAV home distribution 3/3/3/2/3/3/2/3/2, rotated by a deterministic seed-derived region permutation.
- Tasks: six logical tasks per region, three delivery and three inspection; four per region public at initialization, two revealed by arrivals; 54 total.
- Episode: 90 minutes; counterfactual window 30 minutes; up to 16 physically unique candidates including NOOP.
- Local disturbance patch: 2 x 2 and independent of responsibility regions.
- Runtime: one CPU thread, CUDA disabled; per-job wall limit 2 h and RSS limit 8 GiB.

## Provenance and resource boundary

V3 source is copied from `development/ch2_disaster_relief_n02_gain_gate_v2_100x100` with source files identified by relative path and SHA-256 in `BASELINE_PROVENANCE.json`. Prior V2 data, outputs, checkpoints, and resource charges are not reused. V2's interrupted collection CPU total is unknown; the reported 1890.72 CPU seconds is only a lower bound. V3 will report its own measured CPU ledger and keep the historical unknown boundary explicit. No paid compute is authorized.

## Gate decisions

1. Build and run deterministic construction cases plus 18 engineering smoke instances.
2. Freeze source, config, and evidence hashes for read-only review.
3. Do not start formal collection until the coordinator records approval.
4. Formal collection is capped at the frozen 168 train/validation instances; do not add instances to force a 30/30 label count.
5. If valid ordinary labels do not include at least 30 positive and 30 non-positive independent states, stop before model fitting.
6. World-model and PPO stages are outside this V3 milestone.
