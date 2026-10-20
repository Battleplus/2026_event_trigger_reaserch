# Formal evidence v1

`evaluate_suffix.py` independently validates and reduces one mixed keep/replan suffix. It binds the suffix to its indexed source snapshot, public projection, frozen config and source manifest, checks the snapshot plus three restore hashes, then recomputes the scalar from trajectory and event logs. It accepts `trajectory.jsonl` or `trajectory.jsonl.gz`.

The evaluator needs two import roots when reading a private pickle. `--source-root` is the trusted simulator/source tree used by the snapshot; `--collector-root` is the local `formal_collection_v1` directory containing `mission.py` and `EventTriggerGate`. Both are required, including when calling the Python API.

For each completed suffix, write one reduction to its own state/seed path. The formal batch runner uses this layout:

```text
<batch-output>/reductions/<scene>/<state-id>/suffix_<seed>/reduction.json
```

Equivalent direct invocation:

```powershell
python .\evaluate_suffix.py `
  --suffix-dir E:\batch\scenes\scene-a\suffixes\state_00_001200\suffix_730201 `
  --source-index E:\batch\scenes\scene-a\representative\state_sampling\state_index.json `
  --source-root E:\isaac_source `
  --collector-root ..\formal_collection_v1 `
  --output E:\batch\reductions\scene-a\state_00_001200\suffix_730201\reduction.json
```

After all suffix evaluations finish, aggregate that reductions directory once:

```powershell
python .\aggregate.py `
  --root E:\batch\reductions `
  --output E:\batch\aggregate.json
```

The aggregate admits exactly two distinct suffix seeds per state with matching snapshot and config hashes. Non-PASS, fixture-only, forced, empty-window, duplicate-seed and cross-split rows do not form a valid state label. `PASS` from the aggregate means the frozen 30-positive/30-nonpositive and train/validation coverage gates are met.

Run the synthetic interface checks without Isaac initialization:

```powershell
python -B -m unittest test_formal_evidence -v
```

These tests exercise published CLI loading in a fresh Python process, including a fixture pickle containing `mission.EventTriggerGate`; they do not constitute a real Isaac suffix result.
