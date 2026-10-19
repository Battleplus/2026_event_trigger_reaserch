# A1 attempt003 evidence summary

Coordinator decision: `A1_SIMPLIFIED_ENGINEERING_PASS`.

Representative audit passed for 194,400 trajectory rows and 36 actors. Pair audit was first recorded as failed because the auditor counted delivered commands as issuance attempts; the preserved independent recheck passed after the command-attempt aggregation was corrected. The pair recheck covered 172,800 rows and verified geometry, speed, discovery, zone, hold/resume, and production pair metrics.

Representative: 81/81 public tasks completed, 27 targets discovered, analytical energy 1759.077 kWh. Pair keep/replan: both 79/81 completed and 27 targets discovered; window analytical energy was 299.239 kWh and 306.043 kWh respectively. These are analytical displacement estimates, not physical energy measurements. No genuine positive scalar gain is claimed; production scalar is `NOT_COMPUTED`/`NOT_VERIFIED`.

The release retains both original audit reports, the pair recheck report and its independent recheck source, plus the coordinator final review. Raw receipts and evaluator truth are omitted from the public archive; their output hash record is retained.
