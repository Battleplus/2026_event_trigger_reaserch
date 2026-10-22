# Raw public one-step window reader review

Decision: ACCEPT_SOURCE_ADAPTER. This is a data-interface acceptance only. It is not a 30/30 result, model training, GPU result, Isaac replay, or method conclusion.

Implementation: GPT-6 Luna. Review: main Codex coordinator, read-only source and focused-test inspection. No claim of GPT-6.1 Sol High or cross-family independence.

## Frozen hashes

| File | SHA-256 |
| --- | --- |
| wm_ppo_v1/raw_public_windows.py | f703ce8850e827f7345edfb2a3c64fe54175a41be896b87298a93465a40dd29e |
| wm_ppo_v1/features.py | 24b3e39decb4141f1d5159b85405353810b17cf66aecf710b1e89649917bb005 |
| wm_ppo_v1/tests/test_raw_public_windows.py | 805bb166147886b05b0bd83856c7f49222e7b31205dee82691038f5245996555 |
| wm_ppo_v1/RAW_PUBLIC_WINDOWS_SCOPE.md | d498f4c266c088e851e66d9841281c2b95057e8ff07a719c18d056373c072198 |

The focused raw gzip test and affected graph test passed (`Ran 2, OK`). The coordinator inspected the final source and hashes; the same tests were not rerun.

## Accepted behavior

The reader consumes gzip trajectory rows once, keeps only a bounded rolling history, and emits windows only for adjacent seconds within one branch. It projects actual position, velocity, remaining energy, platform kind/zone, fault, communication hold, safety pause, publicly available observations, and public task activation state. Hidden truth, target discovery IDs, scenario hashes and command payloads are excluded. Unknown availability and missing queues remain masked. Future or delayed events are excluded from the current input; one-step labels use only the following interval.

The shared `pair_prefix` contributes only public task activation/completion metadata to the first keep/replan frame at the shared snapshot time. It does not contribute prefix trajectory history, private state or hidden truth. This preserves a common public snapshot while keeping branch trajectories separate. Explicit suffix receipt task IDs are preferred when present. The reader does not infer an applied action from command delivery.

## Limits

- The adapter has not been run on a completed formal Isaac scene; the focused fixture is synthetic.
- It does not copy or rewrite the large raw trajectory; actual resource and window yield remain to be measured after terminal batch reduction.
- The raw capture does not provide an applied command vector, queue snapshot or explicit platform availability on every row; these inputs remain unknown/masked.
- Gain labels are accepted only with exact state/run/phase/branch/time identity and an independent validation flag. No gain is fabricated from branch summaries.
- Formal collection, terminal reduction, 30/30 gate, model fitting and closed-loop evaluation remain pending.
