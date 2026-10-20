# Service Safety Repair

Strike service now requires an aircraft to remain available, unpaused, and within the configured service radius for the full dwell duration. Any unavailable or paused step resets the service start time; leaving the radius also resets it. Resuming service starts a new continuous dwell interval. Existing `abstract_strike` callers retain the prior normal-operation behavior through default `available=True` and `paused=False` parameters.

The runner passes the aircraft's live availability and fault, energy-hold, and communication-hold state to the helper. It only calls the helper for targets in the public-observation map, preserving the public-observation queue boundary. Type matching and helicopter home-zone restrictions remain enforced by the helper.

Targeted verification command:

```text
python -B -m unittest test_service_safety
```

Captured output and exit code:

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK
```

Exit code: `0`. AST parsing of `mission.py`, `run_a1.py`, and `test_service_safety.py` also passed. No Isaac run was performed.
