# Event graph repair: source review

Decision: ACCEPT_GRAPH_INTERFACE_SOURCE. This supersedes the first implementation's REPAIR only for the defects inspected here. It does not establish a trained world model, GPU update, physical data coverage or added value.

Implementation: GPT-6 Luna. Reviewer: main Codex coordinator, read-only source and hash inspection. No claim of GPT-6.1 Sol High or cross-family independence.

## Frozen files and SHA-256

| File under wm_ppo_v1 | SHA-256 |
| --- | --- |
| contracts.py | 8c84f9528a3a8c6f07f0409579656fb7a92d9ffef0a60406e605a3f312363c23 |
| features.py | 85952777709ec8ec20387a4d05d5f05e2c88e5b89d2308269b1cf9d5e3602fc0 |
| public_dynamics.py | de5a5c0f0ef5902e34b4c6da782b919328d90b9b2f524541a8008c1ddea9608b |
| world_model.py | 08811970101f80ee84d862a47aea70df96d17a3d2b8daa462517d49f32763732 |
| tests/test_event_graph_structure.py | 55013cc71ed14113c2924a05e7b56487cb1c091f1cec442ffd34017f3f215988 |
| SCOPE_REPORT.md | a66a7c1fdb0d0009ff6961c18f92f6801f241b5e836796cc54f537c57db6af9e |

The coordinator independently read these hashes and inspected the adapter, edge builder, message-mask application and focused assertions. Luna reported the one focused structure test and five public-dynamics tests passed. The coordinator did not repeat the same tests.

## Verified repairs

The adapter now reads the actual task-ID-keyed public task mapping, including activated unassigned and completed tasks, and joins ownership/queue information without putting IDs into numeric features. Legacy sparse queue nodes mark task metadata unknown. Public region IDs, rather than default-zero coordinates, determine same-region and four-neighbor edges. These are responsibility-region relations, not measured current communication links or exact current-position proximity.

Event input respects generation and explicit availability time. Padding nodes cannot supply or receive graph messages. The dynamic head receives optional action features while gain uses the recurrent representation without that action input. The focused test uses a compact-shaped projection and checks hidden-value invariance, future/delayed event exclusion, actual unassigned task retention, missing-region edge exclusion, padding isolation and dynamic response with gain invariance to action changes.

## Limits

- No real dataset fitting, GPU update, new-process checkpoint restore, G/G-WM training or closed-loop evaluation has run.
- Applied actions are still an optional aggregate contract; actual formal data coverage and action semantics must be established by a reader. This does not prove candidate-action-conditioned rollout capability.
- The representation is a pooled graph/GRU with aggregate dynamic targets. It does not yet prove full per-task capability, priority/deadline or fine progress coverage.
- New model state dictionaries are incompatible with the earlier encoder-only model. No existing checkpoint is silently migrated.
- The existing checkpoint helper's same-process double-load comparison does not prove the required trained-process/new-process restore.
- Current formal data contains one-second raw rows, but continuous public windows have not yet been constructed through a reviewed raw reader. Sparse sampled states alone are not used as a substitute.

Latest live observation: 2026-10-22 04:26:35 UTC, 79 COMPLETE scene receipts, collector PIDs 2570396/2570401 and active child 3589425 present; 32,177,815,552 bytes free; no terminal batch receipt or aggregate. Formal data and its eventual 30/30 gate remain separate.
