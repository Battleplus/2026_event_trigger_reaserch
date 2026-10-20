# V33 Smoke Audit

- Run: `v33-smoke-20261020T120749Z-821648377615`
- Scope: read-only check of the manifest, summary, run receipt, and all 18 committed shards.
- Result: `PASS` for the smoke engineering checks.

The audit found 18/18 committed instances, matching marker and compressed-shard hashes, 90-step public trajectories, 36 recorded platforms per instance, 54 logical tasks, 9 regions, and a public C1 assignment receipt in every scenario. Live unresolved safety disposition count is zero. No illegal or duplicate action, cooperation half-action, or label/oracle mask mismatch was found. Each instance has a complete resource receipt.

There are 278 training rows. Of these, 230 are forced states and 48 are ordinary states. There are 45 valid ordinary labels and 233 masked rows; all 45 valid smoke gains are non-positive. This is a smoke engineering result, not the formal 30/30 training gate, and no model training was run.

Resource receipt: 5,044.859375 process CPU seconds, 5,348.8150821 wall seconds, and 586,960,896 bytes current RSS. Simulation used single-thread CPU with CUDA disabled.

The smoke directory remains separate from formal train/validation collection.
