# Isaac ROI simplified v2 A1 attempt003 release

This archive records `A1_SIMPLIFIED_ENGINEERING_PASS` for one 90-minute representative search and one 30-minute keep/replan pair in actual Isaac Sim using the simplified kinematic prototype.

Scope is limited to a public greedy C1 proxy, source kinematic motion, analytic two-dimensional discovery at commanded/display altitude 20 km, and no hard task deadline. It excludes A2 pilot implementation, A3 formal trajectories, A4 training/evaluation, historical Chapter 1 solver claims, physical sensing, turn-rate fidelity, and a frozen production scalar.

The seven frozen source files are under `source/`; their SHA-256 values are listed in `SOURCE_MANIFEST.json`. The independent trajectory audit source is retained separately in evidence because it is an evaluator, not runtime source.
