=========
CHANGELOG
=========

1.2.4
=====

* feature: Estimators: add support for Amazon Random Cut Forest algorithm

1.2.3
=========
* bug-fix: Fix local mode not using the right s3 bucket

1.2.2
=====

* bug-fix: Estimators: fix valid range of hyper-parameter 'loss' in linear learner

1.2.1
=====

* bug-fix: Change Local Mode to use a sagemaker-local docker network

1.2.0
=====

* feature: Add Support for Local Mode
* feature: Estimators: add support for TensorFlow 1.6.0
* feature: Estimators: add support for MXNet 1.1.0
* feature: Frameworks: Use more idiomatic ECR repository naming scheme

1.1.3
=====

* bug-fix: TensorFlow: Display updated data correctly for TensorBoard launched from ``run_tensorboard_locally=True``
* feature: Tests: create configurable ``sagemaker_session`` pytest fixture for all integration tests
* bug-fix: Estimators: fix inaccurate hyper-parameters in kmeans, pca and linear learner
* feature: Estimators: Add new hyperparameters for linear learner.

1.1.2
=====

* bug-fix: Estimators: do not call create bucket if data location is provided

1.1.1
=====

* feature: Estimators: add ``requirements.txt`` support for TensorFlow


1.1.0
=====

* feature: Estimators: add support for TensorFlow-1.5.0
* feature: Estimators: add support for MXNet-1.0.0
* feature: Tests: use ``sagemaker_timestamp`` when creating endpoint names in integration tests
* feature: Session: print out billable seconds after training completes
* bug-fix: Estimators: fix LinearLearner and add unit tests
* bug-fix: Tests: fix timeouts for PCA async integration test
* feature: Predictors: allow ``predictor.predict()`` in the JSON serializer to accept dictionaries

1.0.4
=====

* feature: Estimators: add support for Amazon Neural Topic Model(NTM) algorithm
* feature: Documentation: fix description of an argument of sagemaker.session.train
* feature: Documentation: add FM and LDA to the documentation
* feature: Estimators: add support for async fit
* bug-fix: Estimators: fix estimator role expansion

1.0.3
=====

* feature: Estimators: add support for Amazon LDA algorithm
* feature: Hyperparameters: add data_type to hyperparameters
* feature: Documentation: update TensorFlow examples following API change
* feature: Session: support multi-part uploads
* feature: add new SageMaker CLI


1.0.2
=====

* feature: Estimators: add support for Amazon FactorizationMachines algorithm
* feature: Session: correctly handle TooManyBuckets error_code in default_bucket method
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
