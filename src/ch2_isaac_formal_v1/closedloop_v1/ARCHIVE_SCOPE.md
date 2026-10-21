# Static episode and model-deployment source archive

checkpoint.py is the statically reviewed public-feature predictor loader.
The corresponding ExtraTrees export is in ../light_models_v1/train_models.py.
The archive now includes the frozen episode runtime and independent reducer.
It is not a real-simulator validated release, a real checkpoint, or evidence
of efficacy. Static runtime/reducer review scope is documented separately in
reviews/EPISODE_REDUCER_REVIEW_20261021.md.

No model training was performed: the formal batch is still running. The local
implementation check used a test double because scikit-learn was unavailable.
Real artifact acceptance remains pending the completed-data gate and training.
See reviews/TREE_ARTIFACT_REVIEW_20261021.md for evidence scope.

SHA-256:

- checkpoint.py: 3e1860ce0f918bb023da4b140c2de2439b958146540f5058a4426b720bba5275
- ../light_models_v1/train_models.py: 173a15a3b717c546a25064cc255cd12c34243e889790d7efb8d0d8cc180fb4f7
- ../light_models_v1/selection.py: c6bb84c86337ac8466d490a8204040d632111f18c00ccc987385a10acf7af7b9
- runtime.py: 70d9c013c6f9804d71eaff6507498296e7eb02c95344ab91604a56aecca0c219
- evaluate_episode.py: 44e319a56c650843c60670da2465fe9283994371e60fbc92bdf4624cd8c9f95f
