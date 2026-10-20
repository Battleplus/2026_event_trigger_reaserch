# Pair and scalar protocol review — 2026-10-20

The independent collection-pairs source resumes a trusted indexed state with backend time `t-1` at decision time `t`, then runs two suffix seeds. Each suffix restores the exact state separately for keep and replan. Keep makes no catalog call; replan makes one catalog call and applies its selected action after a measured minimum-one-second delay. The suffix seed affects only future time-addressed communication dropout; branch pairings share that seed.

Three meaningful helper tests cover horizon, the physical loop step-time function and addressable dropout; earlier constant-contract assertions were removed because they did not verify execution. Collection dependencies passed five focused tests plus syntax compilation. Source review found and Luna repaired a one-second suffix-clock offset and mismatched decision ID. No complete resume-path test or actual Isaac suffix execution has passed. Isaac execution, raw suffix trajectories, independent label reduction and 30/30 counts remain NOT_RUN.

The scalar protocol is now frozen before any gain labels are produced. It defines the 1800-step window, search/search-service unresolved-work score, incremental costs and unknown masks. No scalar has been computed from observed attempt007/008 outcomes.

Service audit is independent of production completion code. Existing attempt007 is search mode and lacks raw trajectory, so service status remains NOT_RUN. Fixture checks cover legal continuous dwell and a pause failure; they do not establish Isaac service behavior.

Decision: SOURCE_REVIEW_ACCEPTED_FOR_ISAAC_VALIDATION; no runtime or research acceptance. Service completeness still requires producer start/reset evidence. The auditor rejects branch-local duplicate completions and marks missing fields NOT_VERIFIED instead of defaulting safety fields to false. No method, model, or world-model conclusion is claimed. Upload and remote execution require explicit authorization for the new pair package/directory.
