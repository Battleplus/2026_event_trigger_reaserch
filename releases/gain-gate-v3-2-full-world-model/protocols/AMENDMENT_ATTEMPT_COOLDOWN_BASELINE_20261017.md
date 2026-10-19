# Amendment: Attempt-Cooldown ET Baseline

Status: proposed before collection; pending read-only coordinator review. No labels, models, test results, or OOD results have been collected under this amendment.

The original `ExistingET` method remains unchanged and is retained as a historical comparator. A second baseline, `ExistingETAttemptCooldown`, is added to address repeated ordinary-event replanner calls when the selected action causes no allocation change.

`ExistingETAttemptCooldown` applies the same event detector, safety rules, fixed replanner, action validation, and cost accounting as `ExistingET`. Every successful formal replanner invocation for an ordinary event starts a four-minute cooldown measured from command issue time, whether or not the selected action changes the allocation. Forced safety handling bypasses this cooldown. The cooldown length is frozen as `evaluation.existing_et_attempt_cooldown_minutes = 4` in the run configuration.

The test-set primary comparison family contains every non-baseline method compared with both `ExistingET` and `ExistingETAttemptCooldown`, for `J_total` and `weighted_completion_rate`, plus `ExistingETAttemptCooldown` compared with `ExistingET` for both endpoints. With seven non-baseline methods, Holm correction covers 30 two-sided paired tests. OOD results are reported separately and do not determine the primary support status.

A non-baseline method is `SUPPORTED` only when, against each ET baseline, its paired scenario-instance `J_total` marginal bootstrap 95% CI has an upper bound below zero, its two-sided permutation p-value has Holm-adjusted p below 0.05, and its weighted-completion marginal bootstrap 95% CI has a lower bound above -0.02. The completion noninferiority CI is marginal and is not itself multiplicity-adjusted. An unresolved or unhandled safety disposition prevents a support claim while retaining the complete episode score. Confidence intervals and adjusted p-values use independent scenario instances as the inference units.

The three prespecified GainMLP training seeds remain visible as separate rows. A scenario-wise mean across all three seeds is included as a supplementary cross-seed summary, not as a deployed ensemble and not as permission to select the best seed.

This amendment only governs future collection and evaluation. It does not replace or relabel prior ExistingET results.
