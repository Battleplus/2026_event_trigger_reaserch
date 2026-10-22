# World-model training dependencies: actual source status

The latest collection objective and direct-formal user instruction apply. No representative or pilot prerequisite is reintroduced by this note. Formal fitting still waits for the terminal formal data receipt, independent reduction and 30/30 gate.

## Outstanding implementation requirements

1. Repair the event graph adapter against actual public compact task mappings and public event availability. The first submitted graph implementation is under REPAIR, not an accepted milestone.
2. Supply actual adjacent one-second public history. Sparse sampled formal states do not prove a continuous history; missing transitions cannot be fabricated. The compact interface is source-only and has not been deployed.
3. Replace the WM entry's unconditional NOT_RUN receipt with a real, explicitly gated training path after the required data interface exists. Retain target masks, train-only normalization, scenario grouping and the separation between dynamic action conditioning and pre-gate gain prediction.
4. Implement training checkpoints with model, optimizer, normalization, data/schema hashes, architecture, RNG/seed and CUDA device identity. The existing save_checkpoint helper saves model plus caller metadata only; it is not proof these fields were supplied.
5. Implement a real new-process restore comparison against predictions saved by the trained process. Existing restore_and_compare loads the same saved state into two fresh models in one process; this does not establish before-save/new-process equivalence or optimizer recovery.
6. G/G-WM entry points remain stubs. Real Isaac rollout, PPO/GAE updates, frozen-WM evidence, critic auxiliary gradients and per-frozen-replanner new gain labels remain unimplemented or unexecuted.

The helper limitations are observed source facts, not failures of an actual training run. No training, restore success, world-model added value or PPO efficacy is claimed. Only GPU1 is authorized for neural training, with no silent CPU fallback.
