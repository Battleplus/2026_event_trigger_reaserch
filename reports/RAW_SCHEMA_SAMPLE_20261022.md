# Actual raw schema sample — 2026-10-22

Coordinator read-only SSH inspection of the first completed training scene (light_delay-seed202610210001), limited to the first 72 representative trajectory rows. No private recovery snapshot or hidden truth values were read into model features, and no fitting or gain evaluation was performed.

- Run receipt says complete and actual_isaac_run=true.
- Actor manifest: 36; public task manifest: 81.
- Sample t=1 and t=2 each contain 36 distinct aircraft IDs, branch representative.
- Actual position, previous position, velocity, aircraft type, zone, task ID and fault fields are present.
- Energy object includes remaining_wh, incremental_wh, effective_wh_per_km, anomaly_active and safety_pause.
- Communication object includes attempted/accepted/delivered/dropped fields, delay_s, forced_outage and safety_hold.
- public_target_observations is present; available is absent. Availability must remain unknown in the raw adapter, rather than assumed true.
- events_complete is absent from this run receipt. Decoder defaults do not constitute a separate completeness measurement.

This supports the adapter's field mapping for two ticks only. It does not prove full-file integrity, all-scene compatibility, complete event logging, valid gain labels, one-step coverage throughout episodes, 30/30 or scientific efficacy. Existing identifier/hidden-truth columns remain excluded from numeric features. Static terrain/trees/config duplication in the old capture is not removed by this sample or by source-only compact adapters.

## Related source parity finding

The current formal state_sampling.public_projection returns sim_time_s, aircraft and public_observations; it does not return public_tasks. Its queue IDs permit legacy task joins but do not supply task location, priority or deadlines. The newer graph decoder's support for a public_tasks mapping therefore is an optional supported input, not proof those fields exist in this formal collection. Decision gain histories currently call the raw frame decoder with an empty task manifest; physical dynamic histories can use the receipt task manifest. Online gain history must match its actual training path, including missing-data masks, rather than silently enrich historical task information.
