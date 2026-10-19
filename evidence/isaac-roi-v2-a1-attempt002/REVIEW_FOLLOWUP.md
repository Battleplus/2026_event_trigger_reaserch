# Coordinator follow-up: attempt002

Decision: REPAIR_REQUIRED. This is a read-only review of the frozen execution, not a scientific acceptance.

The representative process completed 5,400 simulated seconds and 194,400 actor-second records. The paired process completed a 1,200-second prefix and two 1,800-second branches, producing 172,800 records. Both external consistency audits reported zero failures. The restored full-state digest matched and the maximum restored position error was zero. NOOP ranked first, so the branches had identical execution outcomes; the replan branch additionally recorded one planner invocation. This provides no positive-gain evidence.

The existing audit independently recomputes displacement-based analytical energy and checks fault holds, key coverage, type compatibility, and event consistency. It does not yet establish search completion from waypoint geometry, discovery from an evaluator-only target catalog, or helicopter confinement from actual coordinates. A field/event audit `pass` is therefore insufficient for A1 mission acceptance.

Two additional implementation errors were identified by reading the frozen source:

1. Candidate energy uses task seconds / 3600 multiplied by a value configured in Wh/km. This treats a distance rate as power and underestimates energy. Candidate ETA also omits approach and queue-connection distances. Legal-candidate claims need correction using current public positions, configured speed, distance-based energy and reserve.
2. Pair branch completion counts represent window additions but use the full 81-task denominator for remaining work. In this run, 36 prefix completions plus 43 branch additions leave 2 terminal unfinished tasks, not 38. Preserve both incremental and cumulative metrics before defining terminal penalties.

The communication outage currently blocks new commands while allowing an active command to continue. It is not evidence of a loss-of-link safety pause. The next version must make its safety rule explicit and verify its actual motion consequence.

The scene uses source kinematic motion, 20 km display altitude, two-dimensional analytical discovery, analytical energy, and synthetic command delay/drop. It is neither a physical sensor experiment nor a formal aircraft dynamics validation. The initial assignment is a newly implemented public greedy C1 proxy, not the historical Chapter 1 solver.

The next run must use a new frozen identity and independent geometric replay. No scalar gain, 30/30 gate, pilot acceptance, model training, or method advantage is established by attempt002. Preserve the frozen artifacts unchanged.
