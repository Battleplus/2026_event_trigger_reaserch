# Candidate Policy Inference v1

`PublicCandidatePolicy` is an optional callback for `closedloop_v1.runtime.run`.
It receives the current public projection, a contiguous public history ending
at that projection, and the already-built legal catalog. The runtime invokes it
only after the existing ordinary gate opens. Suppressed gates and forced local
safety do not invoke the ordinary actor. Every physical second is retained in
the bounded live history when this callback is configured.

The callback converts public frames and catalog entries with the existing graph
and candidate encoders, then applies explicit node and candidate centers/scales
whose evidence must declare `scope=train_split_only` and match both feature-name
lists. Histories require at least two contiguous one-second frames and are
bounded to eight. A known missing history or GPU deployment prerequisite is
recorded as policy unavailable. The runtime does not fall back to the sorted
catalog choice in that case; it leaves allocation unchanged and records no
replan decision. Other callback, model, numeric, or action-mapping errors fail
the run and use the existing failure receipt.

Sampling mode uses a masked categorical distribution and returns the sampled
index, key, original action, old log probability, value, legal mask, model
version, and measured inference cost. Evaluation mode uses deterministic
argmax with stable-key tie-breaking. Real neural inference is pinned to the
configured GPU1 UUID (`cuda:0` after explicit UUID selection). CPU execution is
available only when both `synthetic_cpu_fixture=True` and the model version
starts with `synthetic:`.

The runtime adds catalog-generation and policy-selection timing once in the
planning diagnostic. Its existing projection/policy/planning phase remains the
single source for online phase totals; nested decision timing is diagnostic
and is not added again. The choice is queued through the existing command delay
and stale-transfer checks. Forced safety remains immediate local behavior and
does not wait for candidate selection.

No candidate-policy checkpoint, training, or real GPU inference is supplied by
this interface. Production candidate policy remains `NOT_RUN` until a trained
actor checkpoint and matching train-only normalization artifact are provided.
