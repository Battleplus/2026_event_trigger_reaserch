# Partial model-deployment source archive

checkpoint.py is the statically reviewed public-feature predictor loader.
The corresponding ExtraTrees export is in ../light_models_v1/train_models.py.
This increment does not contain a complete frozen episode runtime. It is not
a runnable closed-loop release, a real checkpoint, or evidence of efficacy.

No model training was performed: the formal batch is still running. The local
implementation check used a test double because scikit-learn was unavailable.
Real artifact acceptance remains pending the completed-data gate and training.
See reviews/TREE_ARTIFACT_REVIEW_20261021.md for evidence scope.

SHA-256:

- checkpoint.py: 48917e26aa3471a9d0602f7122d03c0c25a8ef6ee9dde50a294e4765e3811af3
- ../light_models_v1/train_models.py: ab1137aa7dcdb3b1633e824d38947afb7b076849ca4ba663cc7703b3fea4f81b
