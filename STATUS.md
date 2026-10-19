# V3.2 performance/light experiment status

Date: 2026-10-19 (Asia/Shanghai).

- Current stage: necessary checks before the single new 18-instance smoke.
- Production changes: snapshot duplication/key serialization, two proven label-interface repairs, actual worker-tree RSS measurement. No scenario, candidate, safety or scoring-weight change.
- Slow-instance regression: 90 ticks; action and physical hashes match the independent V3.1 reference; 8/8 ordinary labels valid, 4 forced states excluded. CPU 643.96875 seconds. Not a comparable speedup against previously failing branches.
- Snapshot/source: releases/gain-gate-v3-2-performance-light; 93 files, one manifest/hash set. Acceptance pending.
- New smoke: NOT_RUN. Formal 168-instance collection: NOT_RUN. Data gate: NOT_EVALUATED.
- CPU/GPU training, world model and PPO: NOT_RUN. Method: INCONCLUSIVE.
- GPU preflight: 2 RTX 2080 Ti reachable; default PyTorch import fails on CUDA library dependency resolution. No environment changes.
- This branch follows protocols/V32_PERFORMANCE_LIGHT.md and does not change the separate Isaac/full-world-model study.
