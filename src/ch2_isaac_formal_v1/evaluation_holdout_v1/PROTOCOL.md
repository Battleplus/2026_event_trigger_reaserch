# Evaluation Holdout v1

This directory contains sealed configuration files for the independent evaluation holdout. The files are generated from the frozen `formal_collection_v1.prepare_formal_registry.make_config` function and are configuration artifacts only. No Isaac scene, suffix worker, evaluator, aggregate reducer, model fit, threshold fit, or result read is performed by the generator.

The ID test contains 42 scenes: seven formal conditions, six `Existing-ET` scenes per condition, `split=test`, and seeds beginning at `202610230001` with a 100-seed condition offset. The OOD cohort contains 20 `composite_all_enabled` scenes, `split=ood`, `Existing-ET` as the source placeholder, and seeds beginning at `202610240001`.

For OOD, the only intentional configuration change is `communication.dropout_probability: 0.03 -> 0.15`. Mechanism switches, schedules, task geometry, timing, target count, and other search3 settings come from the current formal `make_config` output. The collector already consumes this field in its stable command-drop model; no incompatible setting was detected.

Every scene is marked `forbidden_for_training: true`. The manifest records config SHA256 values, the frozen formal registry SHA256, the base config SHA256, and the exact cohort protocol. These configurations remain sealed and must not be used for model fitting, calibration, threshold selection, or hyperparameter selection.

The `test` and `ood` split labels are evaluation-only metadata. They are intentionally outside the frozen formal batch runner's `train`/`validation` registry validator and are not a request to alter that runner.
