=========
CHANGELOG
=========

1.0.4
=====

* feature: Estimators: add support for Amazon Neural Topic Model(NTM) algorithm
* feature: Documentation: Fix description of an argument of sagemaker.session.train
* feature: Documentation: Add FM and LDA to the documentation
* feature: Estimators: add support for async fit
* bug-fix: Estimators: fix estimator role expansion

1.0.3
=====

* feature: Estimators: add support for Amazon LDA algorithm
* feature: Hyperparameters: Add data_type to hyperparameters
* feature: Documentation: Update TensorFlow examples following API change
* feature: Session: Support multi-part uploads


1.0.2
=====

* feature: Estimators: add support for Amazon FactorizationMachines algorithm
* feature: Session: Correctly handle TooManyBuckets error_code in default_bucket method
* feature: Tests: add training failure tests for TF and MXNet
* feature: Documentation: show how to make predictions against existing endpoint
* feature: Estimators: implement write_spmatrix_to_sparse_tensor to support any scipy.sparse matrix


1.0.1
=====

* api-change: Model: Remove support for 'supplemental_containers' when creating Model
* feature: Documentation: multiple updates
* feature: Tests: ignore tests data in tox.ini, increase timeout for endpoint creation, capture exceptions during endpoint deletion, tests for input-output functions
* feature: Logging: change to describe job every 30s when showing logs
* feature: Session: use custom user agent at all times
* feature: Setup: add travis file


1.0.0
=====

* Initial commit

