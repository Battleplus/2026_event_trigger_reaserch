# V4 environment and collection review

Decision: PASS for starting the frozen 84 train + 84 validation collection. This is engineering approval only.

Reviewer: Codex primary coordinator in this session. The runtime does not expose a verifiable exact model identifier; no named-model review is claimed. Implementation and execution are assigned to GPT-6 Luna. The review did not modify production or test source.

The reviewed environment/data/oracle/split source identities match the completed environment fixture. The live 36-platform, 54-task compound fixture ran 90 minutes and recorded failure and communication events, actual task pause and recovery. Regional takeover and cooperation tests use real simulator transitions; the cooperative logical task completion reward matches a single completion, not the number of child operations. The regional constructed test uses the existing 30-platform fixture, while the full V4 fixture covers the 36-platform scene.

The collector integration ran three full trajectories, sampled eight fixed-seed minutes per source, and produced 318 samples including 270 one-step samples. Windows without alternatives retain only NOOP. Branches clone the same live snapshot and event tape, use the same safety continuation, and independently recompute raw-trace rewards and execution costs. Public sampling does not filter by future gain. Hidden-future perturbation changed suffix hashes while current public history, fixed decision and encoded input tensors remained equal.

Source review covered candidate legality/dedup/cap, provenance-free safety state hashing, collection history and branch sampling, target masks and raw-trace oracle calls. Previously completed focused safety/regional tests and the latest collector integration were reused; no full historical suite was repeated.

This review does not establish prediction accuracy, PPO quality, safe closed-loop results or scientific benefit. Those require the planned full runs and later reviews. Chapter 1 remains a public C1 proxy, not authoritative integration.

Configuration SHA-256: b640c66059ee815c729ae6089b81baac686ed0905ae60176da0934994712db3c

Protocol SHA-256: dd20219a9f76a8b219dd8b80bff0e9e40e362ba67d7312a688c8b15a944ca555
