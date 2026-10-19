# Isaac ROI simplified v2 A1 attempt 001

This frozen package ran one 90-minute representative episode in Isaac with 36 platforms. Motion uses the source scene's simplified kinematic model; it has no turn-rate/turn-radius constraint and is not a physical flight-dynamics result. The actual simulator data are real Isaac output, but the independent audit found actor type/kind/speed and task-schema mismatches plus non-finite energy values. The coordinator decision is `REPAIR_REQUIRED`; this is not an accepted method result. The paired A1 comparison, pilot, formal collection, and training were not run.

See `../../evidence/isaac-roi-v2-a1-attempt001/` for the sanitized receipt, run manifest, coordinator review, and hash summary. Large raw trajectory, event, and audit files are retained outside this public archive; their source hashes are recorded in the evidence summary.
