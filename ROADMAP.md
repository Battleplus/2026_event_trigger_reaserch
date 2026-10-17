# Roadmap

## Current Snapshot

V2 safety and candidate repairs passed at pilot scale. Formal collection was started twice and interrupted; the final run is retained with a read-only integrity audit. The current disposition is **INCOMPLETE_COLLECTION; STOP** and method support is **INCONCLUSIVE**.

## Next Steps

1. Preserve the final smoke evidence and source/config identities.
2. Preserve both interrupted collection directories and the read-only audit; do not repair aggregate gzip files in place.
3. Stop formal analysis because the final run has 0 ordinary positive labels and does not meet the 30-positive/30-non-positive gate.
4. Train light models only after a separately reviewed collection reaches both registered sign counts.
5. Preserve all failures and masked/forced/empty windows; do not manufacture a learnable signal.

Do not relax sign-count or validation-group guards, search seeds for positive labels, or change cost weights to manufacture a learnable signal.

## Not Run

Formal test/OOD collection, light-model training, closed-loop method evaluation, event-aware world-model training, PPO transfer, and release tagging remain unrun. The interrupted final run is not a 168-instance result and has no final resource accounting.
