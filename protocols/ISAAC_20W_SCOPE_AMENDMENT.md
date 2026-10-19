# Isaac 20w environment amendment

Date: 2026-10-19 (Asia/Shanghai).

The user requires reuse of the existing 20w Isaac Sim scene in the supplied 54_20-master.zip. World-model training data must originate from actual Isaac execution. Formation geometry is not fixed.

The pure-Python N02 V3.3 development and new smoke/collection are paused. Prior N02 shards remain historical diagnostics and cannot count as Isaac training, acceptance or method evidence.

Verified archive: SHA256 99ec901f5ef9ff9ceaccc84c932507f207ae5b06af824da3fc04724a60d0eca5. The README identifies test_air_combat_scene.py and scenes/air_combat_scene.py as the 20w entry/scene. The scene defaults include 100 metres per USD unit and a time scale of 45; effective entry-point overrides still require runtime verification. The mountain_forest_aircraft.yaml configuration is a different scene configuration and must not be substituted silently.

Implementation sequence:
1. Bind the exact scene, effective units/time scale, motion backend and Isaac runtime.
2. Connect civilian task commands and public telemetry; retain 9 regions, 30 platforms and 54 tasks as the task-layer target, with an explicit native-coordinate mapping.
3. Verify two different legal commands produce different actual Isaac trajectories/outcomes.
4. Implement complete scene/mission/random-state restore, or same-seed reconstruction with prefix replay, before collecting fair keep/replan pairs. The existing IsaacAirCombatEnvironment.reset only resets counters; it is not a complete simulator snapshot restore.
5. Complete the 18-instance Isaac engineering validation, then collect the frozen 168 train/validation instances. No old Python labels or checkpoints enter this dataset.
6. Apply the unique ordinary-state 30/30 gate, then train light and event-aware world models on GPU, paired G/G-WM, policy-specific labels/gates, and the frozen six-arm evaluation with 42 independent tests and seeds 8201/8202/8203.

Only actual Isaac frame/state/command records count as simulator data. Offline mock controllers and old result exports do not. Record any analytic energy or communications layer explicitly; running inside Isaac does not establish measured motor/battery power or physical radio fidelity. Preserve native time and effective command timing, including prediction/planning waits.

This is an authorized scope amendment, not an acceptance, training result or scientific conclusion. Runtime availability and interface gaps are under Luna inventory and coordinator read-only review.
