# Event graph first implementation: read-only review

Decision: REPAIR. No accepted WM milestone or training result.

Implementation: GPT-6 Luna. Reviewer: main Codex coordinator. Review scope: actual public projection adapter, feature/edge builder, graph model and structural test. No claim of GPT-6.1 Sol High or cross-family independence.

The first implementation adds typed-edge messages, event features and a dynamic-head-only action path. Its reported 11 tests do not cover the following actual data-contract defects:

1. The compact runtime supplies `public_tasks` as a task-ID-keyed mapping with `public_xy_km`, zone, assignment and completion fields. The adapter accepts only a sequence and normalized synthetic geometry keys, so it ignores real task geometry and omits activated unassigned tasks.
2. The coordinate helper substitutes zero for absent values; the later None check cannot prevent false geographic edges at the origin. A normalized distance threshold of 0.2 also corresponds to 60 km on the 300 km background, covering almost the whole 50 km ROI.
3. Event filtering checks generation time but not a later explicit availability time. This does not prove delayed observations remain unavailable to the model.
4. The structural test supplies synthetic frames directly rather than the actual compact mapping. It asserts gain invariance to an action change, but does not assert dynamic predictions respond to the action or verify the hidden/availability perturbations requested.

Repair requested within the same narrow files: consume the actual public task mapping, retain activated unassigned tasks, join queue ownership without IDs as values, use known same/four-neighbor public region edges, filter generation and availability times, respect node masks, and extend the single focused fixture to cover these real risks. Missing data must stay unknown/masked, with no fabricated geometry or communication links.

No live collector, data gate, training device or experiment matrix was changed. The existing formal batch was verified live at 2026-10-22 03:16:57 UTC with 77 COMPLETE scene receipts and no terminal aggregate/receipt.
