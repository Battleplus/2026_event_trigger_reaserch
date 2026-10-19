# V3.2 fresh read-only experiment audit

Date: 2026-10-19. Reviewer: GPT-6 Astra, ultra, fresh agent `/root/v32_integrity_review`, via experiment-audit skill. Review independence: same-family. Acceptance status: provisional. The reviewer changed no files and started no simulations.

Overall integrity: **WARN**. Smoke acceptance: **FAIL**. Required decision: **STOP**, formal data **NOT_EVALUATED**, training **NOT_RUN**, method **INCONCLUSIVE**.

Paths beginning `S/` refer to `releases/gain-gate-v3-2-performance-light/package/civilian/`. Raw references refer to the local stopped run `v3-2-smoke-20261019`; hashes appear in `evidence/v32-smoke/shard-index.json`. Raw large logs are not publicly uploaded.

## A. Ground-truth provenance: PASS with metric-semantics WARN

Labels derive from actual simulator branches, raw events and endpoint truth, not model predictions. `S/counterfactual_pairs.py:581` records `env.step`; line 653 persists endpoint truth; line 970 runs both arms. The separate oracle imports neither producer nor scorer (`S/gain_oracle.py:3`, 451, 655).

Inherited scoring issue: a completed delivery can retain positive remaining-progress cost. In task-update instance 001, decompressed row 3, `label.pairs[0].keep.trace.raw_truth.tasks[id="v3-task-16"]` records completed status, delivered=demand=2.35, progress=0.5, service_progress=1.0, service_total=2.0 and priority weight 0.10698416315369717. It contributes 0.053492081576848585 residual cost despite completion. There are 74 arm/task occurrences of completed tasks with progress below one in the 32 audited arms.

`S/event_environment.py:2339` divides the current service counter by load-plus-unload total; `S/cost_accounting.py:98`–104 uses `1-progress` even for completed tasks. The oracle reconstructs the same convention (`S/gain_oracle.py:362`). Arithmetic agreement therefore does not establish correct remaining-work semantics. These contributions cancel in the audited pairs with matching physical outcomes; effects on genuinely different outcomes remain unverified.

## B. Score normalization: PASS for integrity, WARN for interpretation

Weights normalize by total task priority, not model-output statistics (`S/cost_accounting.py:64`, `S/counterfactual_pairs.py:784`). Both arms share the obligation set; physical and computational components are retained (`S/cost_accounting.py:108`, 127, 189).

Fresh independent stdlib arithmetic over 16 pairs / 32 arms matched recorded gains, means and sample variances. Maximum J_total difference: 2.22e-16. This validates arithmetic against saved evidence subject to the residual-work issue.

## C. Files and claims: PASS for the frozen partial snapshot; final smoke FAIL

The independently audited four-shard snapshot contains 27 rows: 19 forced excluded and 8 valid ordinary, all nonpositive. Compressed/decompressed hashes, marker hashes, counts, all 27 label hashes, 16 pair hashes and 32 trace hashes were verified. Each trajectory has 90 ticks.

Final FAILURE.json supersedes live observations: worker-tree CPU 1800.046875 seconds against cap 1800; killed; per-process peak-working-set sum 1,637,158,912 bytes; wall 1967.2744004999986 seconds. Exactly 7 committed markers exist. The three newer shards received hash/row checks in this fresh review; its full branch-arithmetic scope is four shards. The coordinator separately recomputed all seven using the independent oracle; do not conflate those review scopes.

No 18/18 completion, formal gate, or training claim is supported. Historical failed branches, profiled execution and corrected full branches are not controlled speedup comparators. The optimized slow-instance diagnostic lacks raw labels; its summary alone is not reproducible branch evidence.

## D. Reachability: PASS; smoke flag WARN

Runner collection and validation are reachable (`S/v3_runner.py:406`, 416), branches call scoring (`S/counterfactual_pairs.py:987`) and oracle recomputes nested costs (`S/gain_oracle.py:727`).

The inherited smoke check at `S/v3_runner.py:814`–818 tests only that oracle.valid is boolean. Honest masked labels return valid=true but label_valid_mask=false (`S/gain_oracle.py:742`–751). That flag is insufficient for accepting training labels; the external raw recomputation gate is necessary.

Diff scope is bounded scalar sort-key caching, redundant deepcopy removal, conditional scenario hash refresh, receipt-schema repair and RSS accounting. `_encode` creates fresh containers on the synchronous supported-state path (`S/counterfactual_pairs.py:56`–99). Nine reviewed source files match MANIFEST entries. This supports limited implementation confidence, not an across-scenario equivalence proof.

## E. Scope and fairness: WARN

All 16 audited replan actions are NOOP, with matching keep/replan endpoint truth. The eight labels test added planner-call cost; they do not prove beneficial reconstruction, positive coverage or method benefit.

Both arms restore the same suffix (`S/counterfactual_pairs.py:970`, 976); replacement checks public observation/history at line 419; oracle checks truth/tape bindings (`S/gain_oracle.py:602`, 675). Suffix sampling conditions on the prefix and frozen event quota (`S/v3_scenarios.py:327`–384).

No demonstrated hidden-future effect was found in inspected ranking paths. Public observations omit tape identities (`S/event_environment.py:457`); candidates use reported tasks (854, 2969); ranking filters known tasks (`S/fixed_replanner.py:108`). Private task specifications are merged in `_planned_queue_state` (`S/event_environment.py:1383`), but inspected ranking/scheduling paths use queued or public tasks. This is static evidence, not a complete adversarial information-flow test.

Forced labels are honestly masked (`S/counterfactual_pairs.py:919`); failures preserve null gains and false masks (1014). All 608 audited continuation safety receipts are not_required/resolved=true; unresolved receipt handling is not exercised by these valid branches.

The NOOP helper checks ticks/time/nonempty completion (`reviews/check_noop_v32.py:54`). This fresh review did not independently audit its raw trajectory. Its receipt explicitly records that the check occurred during, rather than before, smoke.

## F. Evaluation classification: PASS, simulation_only

Scenario generation declares synthetic simulation and energy units (`S/v3_scenarios.py:130`–138). No field trial or real-world performance conclusion is justified.

Required actions: preserve stopped smoke; update final status; retain scoring-semantic and smoke-flag findings; do not train or claim method benefit. Any future scoring repair requires a new version and labels, not overwritten evidence.
