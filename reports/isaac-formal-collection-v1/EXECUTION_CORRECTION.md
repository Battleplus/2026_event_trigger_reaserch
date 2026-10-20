# Isaac formal collection execution correction

Date: 2026-10-20. Status: PREPARING, NOT LAUNCHED.

The current user instruction removes the representative-scene and single-state/two-suffix prerequisites. Formal collection has 168 independent scene instances (seven conditions, 12 train and 12 validation each). Each scene retains its own episode, sampled states and two suffixes. A failed scene is recorded and the next independent scene proceeds. The ordinary-state 30/30 gate is evaluated once after collection and controls training, not permission to collect. Test/OOD remain sealed.

## Excluded execution

The coordinator accidentally started `v33-collect-resume-20261020T152712Z-caee9a48a4c8` in the legacy analytic V3.3 project. That project uses a 100x100 map and water-delivery/search tasks; it does not use Isaac, the required background or ROI. Its progress and gain counts must not be presented as Isaac results. The coordinator stopped its launcher and children on this date and added a separate exclusion note without changing its data or receipts. The separately started redundant legacy verifier was also stopped. Earlier legacy runs remain untouched.

## Raw evidence and failed proposal

Previous representative and suffix files remain historical diagnostics, not formal training data. The proposal to relabel representative rows as pair-prefix rows and change a derived receipt to complete was rejected by automatic approval review and was not executed. No equivalent workaround will be used. New evidence checking must consume the original indexed snapshot and original suffix rows directly, preserving their provenance and failure masks.

## Current implementation work

GPT-6 Luna owns the new `formal_collection_v1/` and `formal_evidence_v1/` implementations. The primary coordinator reviews their source and evidence. The batch collector will use one GPU1 process at a time and a distinct process for each suffix, because closing Isaac may exit its Python interpreter. Streaming gzip stores the original full row fields; it does not subsample the trace. The old audit assumed a run starting at zero and the old reducer expected separate branch directories; neither contract matches restored mixed-branch suffix files. The new evaluator must use the saved baseline, not fabricate an absent prefix or accept those mismatches as scientific failure/success.

No formal Isaac collection, model training, world model or PPO result is claimed by this document. No engineering success tag is created.
