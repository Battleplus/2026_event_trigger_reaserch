# Pre-collection objective clarification

Reviewer: Codex GPT-6 (exact deployment alias unavailable).
Decision time: 2026-10-17, before any protected-stage research collection or training.

The draft production scorer added a binary unfinished-completion loss to the weighted terminal residual. This makes the treatment of unfinished work ambiguous against the approved requirement that the terminal residual be penalized once.

Freeze one terminal term instead: `completion_cost = completion_weight * sum(priority_weight * (1 - terminal_progress))`. Fully completed tasks have terminal progress 1. Report weighted fully-completed task fraction separately as a quality endpoint; do not add a second binary unfinished loss to the objective. Delay, energy, risk, communication, switching, planning calls, reconstruction size and measured CPU terms retain their registered definitions and weights.

Luna must change producer, independent oracle, evaluator, focused tests, protocol and schema together before collecting data. Existing failed smoke evidence remains unchanged. This is a specification clarification before outcomes are observed, not a response to measured method performance. No model or scientific result exists for this new score yet.
