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

- checkpoint.py: 48917e26aa3471a9d0602f7122d03c0c25a8ef6ee9dde50a294e4765e3811af3
- ../light_models_v1/train_models.py: ab1137aa7dcdb3b1633e824d38947afb7b076849ca4ba663cc7703b3fea4f81b
- runtime.py: ef4a939e324351ed8e11ffba15ce9217b9647e4b02288c49e5abd029f78c63cf
- evaluate_episode.py: 44e319a56c650843c60670da2465fe9283994371e60fbc92bdf4624cd8c9f95f
