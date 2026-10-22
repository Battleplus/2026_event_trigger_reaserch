# Decision input source review — 2026-10-22

Decision: ACCEPT_SOURCE_INTERFACE. Implementation: GPT-6 Luna; reviewer: coordinating Codex agent (actual model identifier unavailable). Coordinator review was read-only. This is source evidence, not a trained-model or scientific acceptance.

Reviewed decision_windows.py, features.py, world_model.py, the focused fixture and scope note. Gain histories end at raw t-1 and append only the hashed pre-candidate public projection at t. The adapter enforces backend_time_s=t-1 and the recorded continuation phase. Paired consequences and hidden truth are not input features; adjacent physical dynamics remain a separate interface. Terminal batch and dataset gates delegate to the existing light-model loader.

Luna reported two focused tests passed: `python -m unittest wm_ppo_v1.tests.test_decision_windows wm_ppo_v1.tests.test_event_graph_structure`, plus compilation. The coordinator inspected the repaired source and test coverage without repeating the suite. The fixture includes a missing raw t row, post-decision row mutation, projection/phase/backend rejection and nonzero ordinary-event channels.

Event schema is public_events_v2, dimension 12. Previous event-encoder weights are incompatible. Public raw histories still have unknown queue/availability fields where absent; the interface does not establish complete physical state observability or measured action vectors.

Frozen SHA-256:

- decision_windows.py: 07df2864c47dac93a54498c31284b9ff7835c6aad667cceec20c11a4212be633
- features.py: e1e01a7e6e36f7eed5da7ef745993699e616caaf96ca6aed7b91f2aa5de1d1e0
- world_model.py: 88e48dff9dc420876e7900448339056922736e3776945327a0ad00fb8b9ee0ab
- tests/test_decision_windows.py: 7c39430cbe038e149c1c08ffaec9c93b8364091f7b8fff7c2780df0c9fd55950
- DECISION_WINDOWS_SCOPE.md: 52638c307bb6b5969ab21fadd43234d1b2aa02c7d536771aaf53ef0c158ce9b6

No live collector changes, real-data fitting, GPU training or deployment occurred. At 06:55 UTC the collection had 85 completed scenes, no terminal receipt/aggregate, and its parent and child processes were alive. Training remains NOT_RUN pending terminal reduction and 30/30. No storage cleanup was performed.
