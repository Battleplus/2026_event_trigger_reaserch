# V3.2 measured performance diagnosis

Implementation/profile: GPT-6 Luna. Coordinator read the emitted profile and compared the action and physical hashes with the independent historical reference. These are simulation diagnostics, not trained-model results.

The original task-change instance (same seed and Periodic-4min policy) completed all 90 ticks under cProfile: 1028.84375 process CPU seconds, 1171.3803023 wall seconds. Its action and physical-trajectory hashes exactly match the independently extracted old shard reference. The historical unprofiled CPU time was 412.6875 seconds; profiling overhead prevents interpreting their ratio as a speedup or slowdown.

| Nested stage | Calls | CPU seconds |
|---|---:|---:|
| Scenario construction | 1 | 0.046875 |
| Paired label collection | 12 | 708.4375 |
| Replanner (also called inside collection) | 25 | 41.359375 |
| Environment steps (also called inside branches) | 114 | 27.953125 |
| Suffix sampling | 16 | 0.65625 |
| Independent label validation | 12 | 0.140625 |

These timers overlap and must not be summed. cProfile identifies snapshot restore, deepcopy, recursive encoding and JSON key canonicalization as the main hotspots. It recorded about 50.5 million JSON dumps. Candidate enumeration is not the dominant measured cost. Optimization will preserve snapshot fields, physical semantics and action ordering.

None of the 12 selected labels is trainable: 4 are intentionally forced, 6 have paired_collection_failed and 2 suffix_preparation_failed. A true oracle validity flag accepts an honestly masked unknown label; it does not mean the branch executed. The two independently diagnosed interface failures are repaired separately from performance changes, followed by fresh real branch execution and cost recomputation.

Resource accounting: the completed profile is measured. A preceding failed import/restore run has an observed lower bound of about 208 CPU seconds and no complete final receipt; it remains unknown, not zero. The old V3.1 smoke used 1800.015625 process-tree CPU seconds and is historical, not a new smoke attempt. Later stage receipts will be added without double counting shard subtotals.

New 18-instance smoke, formal collection, GPU models, world model and PPO are NOT_RUN at this checkpoint. No improvement or scientific benefit is claimed yet.
