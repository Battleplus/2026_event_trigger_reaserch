# Selection and calibration source review

Main coordinator read-only review of selection.py, artifact loader, and training
integration. Implementer: GPT-6 Luna. No real fitting or simulator run occurred.

Disposition: REPAIR.

- Lower target-minus-prediction residual quantile uses ceil((n+1)*alpha).
  For a distribution-free one-sided lower bound this is optimistic: n=10 and
  alpha=0.1 select rank 2, yielding only 9/11 nominal marginal coverage under
  exchangeability. Use floor with explicit insufficient-sample handling when
  the rank is below 1; do not clamp and claim 90 percent coverage. Scenario
  minimum residuals preserve the scenario as the calibration score unit.
- Verify disjoint selection/calibration scenario sets once at metadata creation.
  Calibration must not tune epsilon or share its scenario groups.
- Grouped validation threshold selection is implemented, but training grouped
  folds remain absent. Add fixed training-only scenario folds for inexpensive
  constant/Ridge signal diagnostics and preserve fold memberships. Do not call
  validation diagnostics training cross-validation or expand GPU fitting budget.

Returned to Luna for focused fixes and only the relevant arithmetic/grouping
checks. The live acquisition, frozen labels, test/OOD and scientific status
remain unchanged.

## Repair readback

Verified selection.py c6bb84c86337ac8466d490a8204040d632111f18c00ccc987385a10acf7af7b9
and train_models.py 173a15a3b717c546a25064cc255cd12c34243e889790d7efb8d0d8cc180fb4f7.
Lower residual rank now uses floor; insufficient groups return q=null and do
not expose a lower-bound mode. Selection/calibration overlap is rejected.
Training-only three-fold constant/Ridge diagnostics record scenario membership
and fit normalization only within the fitting fold. No extra GPU fitting is
introduced. These are static implementation observations, not real model results.

Returned two deployment repairs: runtime infers the artifact model kind but
passes the CLI default kind into the receipt instead; and a missing deployment
metadata error silently falls back to a legacy loader with epsilon=0, even for
a requested lower-bound mode. Bind the actual artifact kind and remove this
automatic fallback. No historical trained artifacts require that compatibility.
Disposition: core selection repair accepted statically, deployment REPAIR.

## Deployment readback

Verified runtime.py
70d9c013c6f9804d71eaff6507498296e7eb02c95344ab91604a56aecca0c219.
The CLI now passes the actual artifact model type to run/receipt and treats an
explicit --model only as a type assertion. Legacy metadata-free fallback is
removed; unavailable lower-bound mode cannot silently run at epsilon zero.

Disposition: STATIC_SELECTION_DEPLOYMENT_ACCEPTED for reviewed code only.
Luna reports five focused quantile/grouping checks and one CLI binding check;
no historical suite was repeated by the coordinator. No actual model training,
GPU update, trained checkpoint restoration, Isaac episode, calibration coverage
measurement, test/OOD or scientific support is established by these checks.
