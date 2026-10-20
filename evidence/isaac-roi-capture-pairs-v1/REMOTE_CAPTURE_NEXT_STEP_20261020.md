# Next remote capture step

Status: prepared locally, not uploaded and not executed.

## Package

Use the files in `src/isaac_roi_capture_pairs_v1/` together with the frozen configuration and the independent reducer in `src/isaac_roi_gain_reduction_v1/`. The package must be copied to a new run directory; it must not append to an earlier attempt.

## Required sequence

1. Copy the package to the explicitly authorized remote directory.
2. Run one representative capture on GPU1 only.
3. Audit the representative trajectory and hash inputs.
4. Select one ordinary state from the produced index.
5. Run suffix seeds `730201` and `730202` from the same private snapshot.
6. Run independent audits and the reducer for both suffixes.
7. Aggregate the two suffixes by `state_id`.

No 36-scene pilot, 168-scene collection, model training, world-model training, or PPO run is permitted until this sequence passes.

## Current evidence

- Local focused tests: 9 passed, 5 subtests passed.
- Local commit: `a3189fd`.
- No new remote trace has been produced by this package.
- A successful short run would establish data-pipeline readiness only; it would not establish a research result.
