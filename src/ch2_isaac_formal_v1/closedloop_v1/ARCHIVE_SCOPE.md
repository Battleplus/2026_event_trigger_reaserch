# Static episode and model-deployment source archive

checkpoint.py is the statically reviewed public-feature predictor loader.
The corresponding ExtraTrees export is in ../light_models_v1/train_models.py.
The archive now includes the frozen episode runtime and independent reducer.
It is not a real-simulator validated release, a real checkpoint, or evidence
of efficacy. Static runtime/reducer review scope is documented separately in
reviews/EPISODE_REDUCER_REVIEW_20261021.md.

The later scoped safety extension is reviewed in
reviews/FORCED_SAFETY_SCOPE_REVIEW_20261021.md. It verifies observable actor
pause/resume and motion effects, not independent fault-task queue release.
The comprehensive all_forced_safety_actions_applied evidence remains unknown.
The scenario comparator is reviewed in reviews/SCENE_STATISTICS_REVIEW_20261021.md;
its numerical thresholds never automatically establish method support.

No model training was performed: the formal batch is still running. The local
implementation check used a test double because scikit-learn was unavailable.
Real artifact acceptance remains pending the completed-data gate and training.
See reviews/TREE_ARTIFACT_REVIEW_20261021.md for evidence scope.

SHA-256:

- checkpoint.py: 3e1860ce0f918bb023da4b140c2de2439b958146540f5058a4426b720bba5275
- ../light_models_v1/train_models.py: 173a15a3b717c546a25064cc255cd12c34243e889790d7efb8d0d8cc180fb4f7
- ../light_models_v1/selection.py: c6bb84c86337ac8466d490a8204040d632111f18c00ccc987385a10acf7af7b9
- runtime.py: 70d9c013c6f9804d71eaff6507498296e7eb02c95344ab91604a56aecca0c219
- evaluate_episode.py: bc96d4c205f2d1df7db3f40c98ab8ddb75c0377aa136caad95bdf6ae107f40b4
- compare_methods.py: 1b98ffed44b9612956a2ffb57935b1db837f6e059c4f469a4e182424d9354c47
