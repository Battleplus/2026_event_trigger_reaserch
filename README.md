# V3: Nine Regions, Thirty Platforms, Fifty-Four Tasks

Status: **PROTOCOL BASELINE / IMPLEMENTATION IN PROGRESS**. No V3 smoke or data result is claimed.

The target is a 100x100 map, nine geographical responsibility regions, 24 UAVs plus six helicopters, 54 tasks, and a 90-minute episode. See the [V3 protocol baseline](protocols/v3-baseline/protocols__V3_PROTOCOL.md) and [configuration snapshot](protocols/v3-baseline/configs__GAIN_GATE_CONFIG.json).

The existing `package/`, old configurations, reports and reviews in this protocol-only commit remain historical V2 source/evidence. They are **not the implemented V3 package**. V3 source will be published in a later reviewed milestone. This branch preserves the V2 interrupted-collection history; no old collection or model is reused.

Formal collection requires a new V3 smoke review. Light-model training requires at least 30 positive and 30 non-positive valid ordinary states. World models and PPO are not started.

Historical V2 collection: incomplete; 0 positive and 78 non-positive ordinary labels in the parseable partial evidence. This is not a conclusion that the proposed method is ineffective. See the [interrupted collection audit](reports/V2_INTERRUPTED_COLLECTION_AUDIT_20261017.json).
