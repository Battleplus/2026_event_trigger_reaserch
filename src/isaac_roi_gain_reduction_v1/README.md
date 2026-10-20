# Offline Gain Reduction

`reduce_gain.py` reads one suffix directory containing `keep/` and `replan/` trajectory and event streams plus `manifest.json`, `run_receipt.json`, and `evaluator_truth.json`. It implements the frozen scalar protocol in `protocols/FROZEN_SCALAR_PROTOCOL_V1.md` without importing production mission or cost functions.

The reducer independently computes energy from USD displacement and each row's `effective_wh_per_km`, applies the protocol's Q/L/J terms, and subtracts signed incremental messages (`transmission_sent` + `command_dropped`), controller switches, candidate-generation calls, planning CPU seconds, and initial transfer actors. Discovery is valid only with a PASS audit report whose `input_sha256` binds trajectory, events, and evaluator truth. Search+service additionally requires the same binding from a PASS service audit. Missing snapshot baseline IDs, complete actual-Isaac receipt, restore hash match, row coverage, or audit binding yields `UNKNOWN`.

`fixture_only=true` is preserved in synthetic test outputs and never represents an Isaac result. The current attempt traces are not reduced here because their suffix baseline IDs and/or trajectory evidence are incomplete.

Verification: `python -B -m unittest test_reduce_gain` passed (2 tests, exit 0), and `python -B -m py_compile reduce_gain.py test_reduce_gain.py` passed. Probing `attempt007/evidence/pair` returns `UNKNOWN`: the suffix trajectory streams and baseline snapshot IDs are absent, and the existing audit report does not bind all required input hashes. No Isaac run or remote action is performed by this module.
