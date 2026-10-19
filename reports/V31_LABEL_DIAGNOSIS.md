# Historical V3.1 label diagnosis

Read-only coordinator check, 2026-10-19. These are old diagnostic shards, not V3.2 training data.

- 13 committed scenario shards contain 88 selected rows.
- 56 rows are forced safety states and are intentionally excluded from gain training.
- All 32 ordinary-state labels are invalid: 19 suffix_preparation_failed, 13 paired_collection_failed.
- A representative suffix failure reports V3_SCENARIO_CONTENT_HASH after future tape replacement.
- A representative branch failure reports illegal continuation action: invalid_scope; the action is a continuation receipt envelope rather than the enclosed environment action.

The current smoke code checks that oracle.valid is a boolean, not that ordinary branches actually executed and costs were independently recomputed. File completion alone will not be accepted as fair-label acceptance. Unknown labels remain masked. The two proven interface defects require minimal repair and fresh branch recomputation before formal collection.

The old 413-second runtime includes failed branch collection and is not an estimate for corrected, full-length branches.
