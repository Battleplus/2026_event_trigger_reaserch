# Compact public capture integration: source review

Decision: PUBLIC_CAPTURE_INTEGRATION_SOURCE_ONLY. No deployed collector, physical run, storage saving, training or method result is established.

Implementation: GPT-6 Luna. Reviewer: main Codex coordinator, read-only source review with independently inspected hashes. No claim of GPT-6.1 Sol High or cross-family independence.

Frozen SHA-256:

- runtime.py: f73763d9eedae05df5afe8fe4383acba8ac108d890c26e3fdc863f2073b4697b
- compact_capture.py: 640f41e70b3dfa78b8cae356c5c20afe55dcf15e9a7edb7fd0b4c5986bf9eef5
- test_compact_capture.py: 77a7093a999be370aaee1eca8ddda3cdbec2a6ae5512ff8dd6d2b94bf926f29e

Luna reports four focused tests (capture adapter plus compact writer) and relevant compilation passed. The coordinator inspected the actual opt-in call site, static configuration, adapter and synthetic test. Tests were not redundantly rerun.

## Accepted scope

The opt-in `--compact-capture` writes an additional one-second public stream from the runtime's actual readback/queue boundary. Availability comes from execution state, queues retain active task IDs, and already-activated unassigned tasks remain visible with null assignment. Task coordinates and permitted type are copied from the public task model; missing priority/deadline stay null. No future task catalog or event schedule is copied into shared public configuration. Truth is written separately. Deferred public observations remain filtered by availability time. Event omission marks event completeness false rather than asserting complete dynamic supervision.

## Limits requiring later work

Original trajectory/events files are still written. This increases output rather than proving a reduction in total bytes. The compact adapter retains only part of timing/cost evidence; the full four-phase online timing event is emitted later in the runtime loop and is not yet available in compact-only output. Do not use the compact subtotal as full end-to-end cost or delete original evidence based on this implementation.

No actual Isaac capture or paired recovery was run. Compact public snapshots do not replace the pair engine's private recovery snapshots. Existing sparse formal states cannot be converted into a proven continuous one-second history without source evidence. Full compressed-only acquisition requires independent raw-consequence reduction and complete timing evidence before old duplicate output can be disabled in a future version.

The currently running formal-168-v1 batch is unchanged. Its labels, final data gate, test/OOD seal and existing failures remain independent of this source-only acceptance.
