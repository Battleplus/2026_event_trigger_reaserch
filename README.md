# V3.2: nine-region performance and gain-gate study

**Final outcome: STOP.** The single18-instance smoke completed7 scenarios before the fixed1800 CPU-second limit. Formal168-scenario collection, CPU/GPU training, world model and PPO were not started. Method evidence remains inconclusive.

The executed scene is100×100,9 regions,24 UAVs+6 helicopters,54 logical tasks,90 minutes, up to16 candidates and30-minute paired labels with2 suffixes. Implementation: GPT-6 Luna; coordinator technical checks plus a fresh same-family GPT-6 Astra read-only audit. See [final decision](reviews/V32_COORDINATOR_DECISION.md), [fresh audit](reviews/V32_EXPERIMENT_AUDIT.md), [protocol](protocols/V32_PERFORMANCE_LIGHT.md) and [status](STATUS.md).

Runnable source is in [the isolated release directory](releases/gain-gate-v3-2-performance-light/). Root package/history files belong to earlier versions. The release contains93 exact files, source/configuration bindings, one manifest/hash set and focused tests. Actual measured dependencies are in the resource receipts; a fresh environment install has not been independently verified.

Performance diagnosis identified snapshot copying/encoding, not candidate enumeration, as the dominant hotspot. The corrected slow-instance action/physical hashes match the original90-step reference, but its previously failing branches now run, so runtime ratios do not measure a controlled speedup. [Diagnostics](reports/V32_PERFORMANCE_DIAGNOSIS.md).

All7 committed smoke shards passed coordinator raw recomputation:14 valid ordinary states, all nonpositive;2 empty windows masked;36 forced states excluded. These are smoke diagnostics, not formal30/30 evidence. A completed-delivery residual-cost defect remains and must be corrected in a future version before scientific use. No positive result is asserted.

[Final smoke receipts and hashes](evidence/v32-smoke/) and [safe recomputation summaries](evidence/v32-performance/smoke-final-recomputation.json) are public. Raw large trajectories remain local and are excluded from Git after automatic approval rejected their public upload. Reproduction scripts can recompute user-held raw shards with the independent oracle.

For reproducibility only, from the release directory:

```powershell
python -B package/civilian/v3_runner.py smoke --output-dir <output> --run-id <new-id>
```

The approved V3.2 run is already stopped; no automatic retry is scheduled and no successful milestone tag is created. The fixed command retains18 instances and1800 CPU seconds. Historical failures stay intact. GPU hardware was reachable, but its default PyTorch environment failed a read-only CUDA import check; GPU training remains NOT_RUN.
