# Compact collection interface review — 2026-10-22

Decision: ACCEPT_INTERFACE_ONLY. This is not an Isaac collection PASS or a scientific result.

Implementation: delegated GPT-6 Luna. Reviewer: main Codex coordinator in this session; no claim of GPT-6.1 Sol High or cross-family independent review. Review was read-only for implementation files.

## Evidence inspected

Inspected writer, public reader, one-step adapter, synthetic tests and runner integration proposal. Compared delivered SHA-256 identities to local files:

| File | SHA-256 |
| --- | --- |
| compact_collection.py | c23df870a5511d2865fd1ac9a09ed8774acf7fbacbdde04ccbe43bb7d5315d88 |
| __init__.py | 912ac6f60f30fd9aeb8f28fb71844b6132a8fe3c76a9a3e1b57931f2ba214777 |
| tests/test_compact_collection.py | ef44f9ee7c6a25346ee4629a33c1bfa1c1aaed6723e9eb115acdfb17d38b8fca |
| RUNNER_INTEGRATION_PROPOSAL.md | 7e24d725dd37af4ac3956a19d4f8674b9c1beddda3934914f89adb47e04a1737 |

Luna reports three focused synthetic tests and compilation passed. The coordinator previously ran the initial three tests outside the sandbox (3 passed); the final extended test was inspected after Luna's run rather than redundantly rerun. The initial sandbox permission failure was an environment failure, not evidence of a code assertion failure.

## Accepted scope

Shared public configuration is stored once with a digest. Public entity/observation changes use upsert/delete records and retain each one-second tick, events, decisions and costs. The public reader reconstructs states and filters observations by their availability time. The one-step adapter calls the existing public dynamics target builder. Synthetic coverage includes queue reordering, task insertion/deletion, stationary pause, delayed observation visibility, truth isolation and one-step energy/delay targets.

Public paired snapshots carry a reference and digest for the external pair engine's full recovery snapshot. They are not themselves private simulator recovery snapshots. Hidden truth is written separately and is not opened by the public reader.

## Limits and next integration

No running collector was changed; no current data was migrated, deleted or read for learning. No Isaac job, actual compression measurement, GPU training or restore experiment is proved here. Input must already be an explicit public projection: the known-name filter cannot establish absence of arbitrary private fields. The caller must not pass a private snapshot as a public state.

Current projections omit task geometry, priority, deadlines and detailed capability; these remain unavailable until a future runner supplies them from public sources. The current sparse formal snapshots do not become a continuous one-second WM dataset through this interface. Full private restoration, suffix fairness and engine-state restoration remain responsibilities of the existing pair engine, with separate evidence required.

Integration into a future version must capture the actual public one-second boundary, provide genuine task state fields, preserve the full private recovery artifact and record the new version identity. This acceptance does not authorize replacing the current live batch.
