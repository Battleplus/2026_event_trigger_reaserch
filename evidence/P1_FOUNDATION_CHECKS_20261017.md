# P1 Foundation Check Receipt

Run on 2026-10-17 before the 18-instance pilot. This receipt records engineering
checks only and does not grant P1 approval.

From the gain-gate project root, the focused tests were run with:

```powershell
python -B -m pytest -q -p no:cacheprovider package/tests/test_gain_safety.py package/tests/test_100x100_contract.py
python -B -m pytest -q -p no:cacheprovider package/tests/test_event_environment.py -k cooperative
python -B -m pytest -q -p no:cacheprovider package/tests/test_100x100_contract.py::test_registered_inputs_and_fixtures_use_the_100x100_contract
python -B evidence/p1_foundation_probes.py
```

Recorded outcomes:

- Gain safety and 100x100 contract tests: `17 passed`.
- Cooperative dispatch regression test: `1 passed, 9 deselected`.
- Registered input contract test: `1 passed`.
- `p1_foundation_probes.py`: archived pre-fix source raises the historical
  `ValueError: cooperative subworkloads must dispatch as a pair` when a single
  cooperative child is injected into the operational dispatch result; current
  source returns with no operational dispatch and records `cooperative_pair_wait`.
  This is a controlled injection reproducing the recorded failure condition; it
  is not a replay of the complete original M2 branch trajectory.
- The same probe runs `validate_registry()` while the stricter execution/layout
  resolver is patched to fail. Registry validation succeeds and verifies all 3
  registered scenarios, confirming the read-only data-validation path does not
  require the execution/layout gate.

The probe reads the old archive in memory, verifies that its event config
matches the read-only source tree, and performs no writes to that source tree or
to prior failed-smoke artifacts. Formal gain data collection has not started.
