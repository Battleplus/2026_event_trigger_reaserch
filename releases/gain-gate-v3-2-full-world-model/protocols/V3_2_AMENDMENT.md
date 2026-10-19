# V3.2 Full World Model Execution Amendment

This amendment applies to `gain-gate-v3-2-full-world-model`. It preserves the V3 task, perturbation, cost, and seed mechanisms while correcting two collection adapters: sampled future suffixes now refresh their scenario content identity, and paired continuations return the action in the structured receipt field expected by the counterfactual collector.

Smoke and collection runs have no CPU-time or wall-time stop. Those quantities remain measured and reported. Simulation uses one CPU thread with CUDA disabled. GPU use is reserved for later model training. The runner guardian enforces an 8 GiB resident working-set cap over the worker process tree. Its receipt reports the peak sampled sum of process RSS values; brief peaks between samples may be missed.

Every run and shard binds source and configuration hashes at start and end. Training rows retain numeric public features and the last four public observations without provenance. Paired labels use only the decision snapshot and its sampled future suffix. Unknown labels remain masked with their reason.

The smoke batch contains 18 independent scenarios across the six frozen perturbation families. Each committed shard prints its completion index and measured wall time, process CPU time, and peak RSS. Smoke completion is evidence for coordinator review only; it does not authorize formal collection by itself.
