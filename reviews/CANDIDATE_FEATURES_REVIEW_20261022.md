# Public candidate encoder source review

Status: SOURCE_ACCEPTED_WITH_RUNTIME_NOT_RUN. Implementation: GPT-6 Luna; coordinator read-only review of source, schema, three synthetic tests and recomputed hashes. No real Isaac rollout, policy training or new formal-data read.

The encoder consumes the existing legal catalogue without invoking enumeration or planning. It supports its actual NOOP and UAV-transfer keys and retains exact action mapping. Numeric inputs come from current public platform fields and the catalogue's existing public ETA/distance/energy/gain/switch estimates. IDs are used only as joins and action keys. Hidden/future fields are not encoded. Missing fields retain explicit known bits, including boolean availability/fault status and responsibility-zone information. Sixteen candidates are a hard input limit, not silent truncation. Supplying a catalogue does not independently prove its legality.

The focused tests reported by Luna pass 3/3: actual catalogue structure and stable mapping, invariance to hidden projection fields, and missing-field masks/over-limit rejection. No repeated historical suite was requested. The coordinator inspected the fixtures and the source fields against the real catalogue builder; this is source acceptance only.

The module leaves the separate WM action summary unknown. It cannot solve the missing executed-control supervision or establish a trained G/G-WM replanner. Train-only scaling and real runtime integration remain later work; raw metre/Wh/second fields must not be confused with already-normalised model inputs.

Frozen SHA-256:

- candidate_features.py: `cd89c65e5faf75dcb4029cd1bb5cb7ca2e9768a15ca677c0e2c29fd81b9576b0`
- tests/test_candidate_features.py: `52d93e6d299871d3ba339601b20abc0828f23ddd9f460fe8a8eae3fe48f991ef`
- CANDIDATE_FEATURE_SCHEMA.md: `f652c3bf1d0cbed08072c5d6efdfd23b406056de2730bf2db6d012ec04048522`
