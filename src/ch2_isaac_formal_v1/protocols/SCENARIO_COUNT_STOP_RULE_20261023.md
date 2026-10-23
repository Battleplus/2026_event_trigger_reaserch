# Scenario-count stopping rule amendment

Effective: 2026-10-23 (Asia/Shanghai).

The earlier 168-scene matrix remains the registered design and its provenance
is preserved. It is no longer a mandatory completion count for the Isaac/world
model collection. The count was chosen to provide independent seeds, ROI,
target and event-object variation across seven conditions; it was not a
statistical theorem or a requirement of Isaac or the world model.

Collection is stopped for the current run. Existing old and lightweight data
remain read-only evidence. No scene is restarted, replaced, or selected based
on favorable outcome.

Before any decision to collect more data, produce one old/new union inventory
with, for each registered scene: condition, split, scenario identity, execution
status, source format, receipt/config hashes, valid ordinary state count,
positive/nonpositive/unknown counts, and evidence that the configured
disturbance had an observed consequence. The inventory must distinguish
independent scenes from multiple states within one scene.

The next decision is conditional:

1. If existing data cover the required conditions and train/validation split,
   and the independent union reduction reaches at least 30 unique positive and
   30 unique nonpositive ordinary states with both signs represented in both
   splits, proceed to light-model training and validation. Do not add scenes
   merely to reach 168.
2. If the 30/30 gate is met but a condition or split lacks meaningful coverage,
   freeze a targeted supplementation list before any collection. Supplement
   only the missing condition/split cells with new registered seeds; never
   choose scenes by observed gain.
3. If 30/30 or independent consequence evidence is not met, stop model fitting
   and report the signal/coverage deficiency. Do not manufacture samples from
   correlated time points or suffixes.

The final method claim still requires independent test/OOD evidence, safety,
task quality and total-cost criteria. A smaller sufficient dataset does not
weaken those criteria. The change concerns the stopping count only.
