=========
CHANGELOG
=========

1.14.3-dev
======

* bug-fix: Changes to use correct S3 bucket and time range for dataframes in TrainingJobAnalytics.
* bug-fix: Local Mode: correctly handle the case where the model output folder doesn't exist yet
* feature: Add APIs to export Airflow training and tuning config
* doc-fix: Fix typos in tensorflow serving documentation
* doc-fix: Add estimator base classes to API docs
* feature: HyperparameterTuner: add support for Automatic Model Tuning's Warm Start Jobs
* feature: HyperparameterTuner: Make input channels optional
* feature: Add support for Chainer 5.0
* feature: Estimator: add support for MetricDefinitions

1.14.2
======

* bug-fix: support ``CustomAttributes`` argument in local mode ``invoke_endpoint`` requests
* enhancement: add ``content_type`` parameter to ``sagemaker.tensorflow.serving.Predictor``
* doc-fix: add TensorFlow Serving Container docs
* doc-fix: fix rendering error in README.rst
* enhancement: Local Mode: support optional input channels
* build: added pylint
* build: upgrade docker-compose to 1.23
* enhancement: Frameworks: update warning for not setting framework_version as we aren't planning a breaking change anymore
* feature: Estimator: add script mode and Python 3 support for TensorFlow
* enhancement: Session: remove hardcoded 'training' from job status error message
* bug-fix: Updated Cloudwatch namespace for metrics in TrainingJobsAnalytics
* bug-fix: Changes to use correct s3 bucket and time range for dataframes in TrainingJobAnalytics.
* enhancement: Remove MetricDefinition lookup via tuning job in TrainingJobAnalytics

1.14.1
======

* feature: Estimators: add support for Amazon Object2Vec algorithm

1.14.0
======

* feature: add support for sagemaker-tensorflow-serving container
* feature: Estimator: make input channels optional

1.13.0
======

* feature: Estimator: add input mode to training channels
* feature: Estimator: add model_uri and model_channel_name parameters
* enhancement: Local Mode: support output_path. Can be either file:// or s3://
* enhancement: Added image uris for SageMaker built-in algorithms for SIN/LHR/BOM/SFO/YUL
* feature: Estimators: add support for MXNet 1.3.0, which introduces a new training script format
* feature: Documentation: add explanation for the new training script format used with MXNet
* feature: Estimators: add ``distributions`` for customizing distributed training with the new training script format

1.12.0
======

* feature: add support for TensorFlow 1.11.0

1.11.3
======

* feature: Local Mode: Add support for Batch Inference
* feature: Add timestamp to secondary status in training job output
* bug-fix: Local Mode: Set correct default values for additional_volumes and additional_env_vars
* enhancement: Local Mode: support nvidia-docker2 natively
* warning: Frameworks: add warning for upcoming breaking change that makes framework_version required

1.11.2
======

* enhancement: Enable setting VPC config when creating/deploying models
* enhancement: Local Mode: accept short lived credentials with a warning message
* bug-fix: Local Mode: pass in job name as parameter for training environment variable

1.11.1
======

* enhancement: Local Mode: add training environment variables for AWS region and job name
* doc-fix: Instruction on how to use preview version of PyTorch - 1.0.0.dev.
* doc-fix: add role to MXNet estimator example in readme
* bug-fix: default TensorFlow json serializer accepts dict of numpy arrays

1.11.0
======

* bug-fix: setting health check timeout limit on local mode to 30s
* bug-fix: make Hyperparameters in local mode optional.
* enhancement: add support for volume KMS key to Transformer
* feature: add support for GovCloud

1.10.1
======

* feature: add train_volume_kms_key parameter to Estimator classes
* doc-fix: add deprecation warning for current MXNet training script format
* doc-fix: add docs on deploying TensorFlow model directly from existing model
* doc-fix: fix code example for using Gzip compression for TensorFlow training data

1.10.0
======

* feature: add support for TensorFlow 1.10.0

1.9.3.1
=======

* doc-fix: fix rst warnings in README.rst

1.9.3
=====

* bug-fix: Local Mode: Create output/data directory expected by SageMaker Container.
* bug-fix: Estimator accepts the vpc configs made capable by 1.9.1

1.9.2
=====

* feature: add support for TensorFlow 1.9

1.9.1
=====

* bug-fix: Estimators: Fix serialization of single records
* bug-fix: deprecate enable_cloudwatch_metrics from Framework Estimators.
* enhancement: Enable VPC config in training job creation

1.9.0
=====

* feature: Estimators: add support for MXNet 1.2.1

1.8.0
=====

* bug-fix: removing PCA from tuner
* feature: Estimators: add support for Amazon k-nearest neighbors(KNN) algorithm

1.7.2
=====

* bug-fix: Prediction output for the TF_JSON_SERIALIZER
* enhancement: Add better training job status report

1.7.1
=====

* bug-fix: get_execution_role no longer fails if user can't call get_role
* bug-fix: Session: use existing model instead of failing during ``create_model()``
* enhancement: Estimator: allow for different role from the Estimator's when creating a Model or Transformer

1.7.0
=====

* feature: Transformer: add support for batch transform jobs
* feature: Documentation: add instructions for using Pipe Mode with TensorFlow

1.6.1
=====

* feature: Added multiclass classification support for linear learner algorithm.

1.6.0
=====

* feature: Add Chainer 4.1.0 support

1.5.4
=====

* feature: Added Docker Registry for all 1p algorithms in amazon_estimator.py
* feature: Added get_image_uri method for 1p algorithms in amazon_estimator.py
* Support SageMaker algorithms in FRA and SYD regions

1.5.3
=====

* bug-fix: Can create TrainingJobAnalytics object without specifying metric_names.
* bug-fix: Session: include role path in ``get_execution_role()`` result
* bug-fix: Local Mode: fix RuntimeError handling

1.5.2
=====

* Support SageMaker algorithms in ICN region

1.5.1
=====

* enhancement: Let Framework models reuse code uploaded by Framework estimators
* enhancement: Unify generation of model uploaded code location
* feature: Change minimum required scipy from 1.0.0 to 0.19.0
* feature: Allow all Framework Estimators to use a custom docker image.
* feature: Option to add Tags on SageMaker Endpoints

1.5.0
=====

* feature: Add Support for PyTorch Framework
* feature: Estimators: add support for TensorFlow 1.7.0
* feature: Estimators: add support for TensorFlow 1.8.0
* feature: Allow Local Serving of Models in S3
* enhancement: Allow option for ``HyperparameterTuner`` to not include estimator metadata in job
* bug-fix: Estimators: Join tensorboard thread after fitting

1.4.2
=====

* bug-fix: Estimators: Fix attach for LDA
* bug-fix: Estimators: allow code_location to have no key prefix
* bug-fix: Local Mode: Fix s3 training data download when there is a trailing slash

1.4.1
=====

* bug-fix: Local Mode: Fix for non Framework containers

1.4.0
=====

* bug-fix: Remove __all__ and add noqa in __init__
* bug-fix: Estimators: Change max_iterations hyperparameter key for KMeans
* bug-fix: Estimators: Remove unused argument job_details for ``EstimatorBase.attach()``
* bug-fix: Local Mode: Show logs in Jupyter notebooks
* feature: HyperparameterTuner: Add support for hyperparameter tuning jobs
* feature: Analytics: Add functions for metrics in Training and Hyperparameter Tuning jobs
* feature: Estimators: add support for tagging training jobs


1.3.0
=====

* feature: Add chainer

1.2.5
=====

* bug-fix: Change module names to string type in __all__
* feature: Save training output files in local mode
* bug-fix: tensorflow-serving-api: SageMaker does not conflict with tensorflow-serving-api module version
* feature: Local Mode: add support for local training data using file://
* feature: Updated TensorFlow Serving api protobuf files
* bug-fix: No longer poll for logs from stopped training jobs

1.2.4
=====

* feature: Estimators: add support for Amazon Random Cut Forest algorithm

1.2.3
=====

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
