# V3.3 repair reinspection

2026-10-21. Fresh GPT-6 Astra ultra. `review_independence=same-family`; `acceptance_status=provisional`. Verdict: **PASS limited to static reinspection of the previous repairs**. The prior WARN report remains historical evidence. This does not approve the unfinished 168-scenario experiment.

The reviewer returned the following findings, without modifying files, executing tests/analysis, or reading large raw shards:

| Previous finding | Reinspection |
|---|---|
| Object ordinal versus contract region | Fixed: analyzer lines 27,622,698 give the mapping; plot lines 222,246 and README/data/image disclose object03→r10. |
| Truth endpoints counted as public points | Fixed: plot line170 counts only public_observation. Current image has30 public points and no truth endpoint. |
| Donor transfer into task home region omitted | Fixed: plot lines37–40 use cross_home or cross_transfer, consistent with analyzer. |
| Duplicate-state evidence/counters | Fixed: analyzer line56 hashes the complete training row; lines378–383 reject conflicting evidence and skip identical rows before subsequent counting. |
| Frozen source/config binding omitted | Fixed: analyzer line292 obtains binding,297–308 compare progress and all markers,44 passes expected binding to verification. |
| Wrong cost JSON path | Current generator line735 uses outcome_decomposition.global_cost_component_means. Old66-r2 stays an historical snapshot. |

Repair evidence is explicitly synthetic functionality only: experimental_data=false, real_shards_read=false, full_analysis_run=false. Its analyzer hash matches. PNG/SVG hashes match data, and region labels were visually inspected.

Limits retained: this reinspection did not prove that the illustrated state remains the first eligible state after expanding the cross-region definition, and did not independently compare each point with the source shard. The representative case is not a treatment-effect claim. Final collection/recomputation/audit remain outstanding.

```json
{
  "reviewer_task": "/root/v33_analysis_artifact_review",
  "reviewer_model": "gpt-6-astra",
  "reviewer_reasoning": "ultra",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "scope": "previous_repairs_static_reinspection_only",
  "verdict": "PASS",
  "audited_input_hashes": {
    "tools/v33_conclusion_analysis.py": "7D9D3182D2E2DF704F651FBE194B4F40D9A86585B47D736B980053DC406FA7F3",
    "tools/v33_support_plot.py": "F962CD7619B73E2F80721628F80D3091820F58E4168ACED51CA696DA4F54842A",
    "reports/v33-analysis-preparation/REPAIR_EVIDENCE.json": "15D5EE4C3B34A57E67EBE3E9DB2A2B01917F1B79F4287DD8B75AA2ADBA686E43",
    "reports/v33-analysis-preparation/REPAIR_EVIDENCE.md": "763F2A1AC60AE106049A4C4315329071562D8E2E3728DB8F7D1B42094BA203BD",
    "reports/v33-support-trajectory/data.json": "A8020E83504A4B4760AA81C1874CBC0C09CAB24D4EB2162A0BA1D13793FD3132",
    "reports/v33-support-trajectory/README.md": "319A7AE222E0C0F35D00253AD3BB0E7676960F5F7C39E00E2EF58FFB667E03F1",
    "reports/v33-support-trajectory/trajectory.png": "C1B23D02E3F3E5819AF82E88880CFA91ECC5363080BEE4DA7A04387335FD5A34",
    "reports/v33-support-trajectory/trajectory.svg": "3189B584972B77E58363D0234AD26DF9E1FD2D64E5F88C5BE05E9EB5819A45A3"
  }
}
```
