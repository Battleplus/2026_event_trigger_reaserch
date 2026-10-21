# Tree deployment source review

Reviewer: main coordinator, read-only source review; actual runtime model
identifier is not exposed. Implementer: GPT-6 Luna. This is role separation.

Inspected training export, model loader, runtime CLI and unchanged public
feature ordering. Tree fitting and deployment both use the same ordered raw
features, without the MLP normalization. The export records estimator/schema,
seed, probe, artifact hash and a disk round-trip prediction check. Runtime
explicitly selects tree or MLP; the MLP checkpoint contract is preserved.

Reviewed SHA-256:

- train_models.py: ab1137aa7dcdb3b1633e824d38947afb7b076849ca4ba663cc7703b3fea4f81b
- checkpoint.py: 48917e26aa3471a9d0602f7122d03c0c25a8ef6ee9dde50a294e4765e3811af3
- runtime.py: b3ca94d76dfd940d43cc39651aa5481fac6dc2ed6ab738305c30c27a27116103

Disposition: STATIC_INTERFACE_ACCEPTED. Luna reports syntax compilation and
an in-memory pickle/predictor check using a test double, because scikit-learn
is absent locally. This does not establish actual ExtraTrees serialization,
fresh-process deployment, GPU training, real Isaac execution or efficacy.
Acceptance of a real artifact remains pending until the collection gate passes
and the relevant model is actually trained. The frozen collection is unchanged.
