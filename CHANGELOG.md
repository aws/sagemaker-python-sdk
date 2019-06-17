# Changelog

## v1.28.0 (2019-06-17)

### Features

 * Add DataProcessing Fields for Batch Transform

## v1.27.0 (2019-06-11)

### Features

 * add wait argument to estimator deploy

### Bug fixes and other changes

 * fix logger creation in Chainer integ test script

## v1.26.0 (2019-06-10)

### Features

 * emit estimator transformer tags to model
 * Add extra_args to enable encrypted objects upload

### Bug fixes and other changes

 * downgrade c5 in integ tests and test all TF Script Mode images

### Documentation changes

 * include FrameworkModel and ModelPackage in API docs

## v1.25.1 (2019-06-06)

### Bug fixes and other changes

 * use unique job name in hyperparameter tuning test

## v1.25.0 (2019-06-03)

### Features

 * repack_model support dependencies and code location

### Bug fixes and other changes

 * skip p2 tests in ap-south-east
 * add better default transform job name handling within Transformer

### Documentation changes

 * TFS support for pre/processing functions

## v1.24.0 (2019-05-29)

### Features

 * add region check for Neo service

## v1.23.0 (2019-05-27)

### Features

 * support MXNet 1.4 with MMS

### Documentation changes

 * update using_sklearn.rst parameter name

## v1.22.0 (2019-05-23)

### Features

 * add encryption option to "record_set"

### Bug fixes and other changes

 * honor source_dir from S3

## v1.21.2 (2019-05-22)

### Bug fixes and other changes

 * set _current_job_name in attach()
 * emit training jobs tags to estimator

## v1.21.1 (2019-05-21)

### Bug fixes and other changes

 * repack model function works without source directory

## v1.21.0 (2019-05-20)

### Features

 * Support for TFS preprocessing

## v1.20.3 (2019-05-15)

### Bug fixes and other changes

 * run tests if buildspec.yml has been modified
 * skip local file check for TF requirements file when source_dir is an S3 URI

### Documentation changes

 * fix docs in regards to transform_fn for mxnet

## v1.20.2 (2019-05-13)

### Bug fixes and other changes

 * pin pytest version to 4.4.1 to avoid pluggy version conflict

## v1.20.1 (2019-05-09)

### Bug fixes and other changes

 * update TrainingInputMode with s3_input InputMode

## v1.20.0 (2019-05-08)

### Features

 * add RL Ray 0.6.5 support

### Bug fixes and other changes

 * prevent false positive PR test results
 * adjust Ray test script for Ray 0.6.5

## v1.19.1 (2019-05-06)

### Bug fixes and other changes

 * add py2 deprecation message for the deep learning framework images

## v1.19.0 (2019-04-30)

### Features

 * add document embedding support to Object2Vec algorithm

## v1.18.19 (2019-04-30)

### Bug fixes and other changes

 * skip p2/p3 tests in eu-central-1

## v1.18.18 (2019-04-29)

### Bug fixes and other changes

 * add automatic model tuning integ test for TF script mode

## v1.18.17 (2019-04-25)

### Bug fixes and other changes

 * use unique names for test training jobs

## v1.18.16 (2019-04-24)

### Bug fixes and other changes

 * add KMS key option for Endpoint Configs
 * skip p2 test in regions without p2s, freeze urllib3, and specify allow_pickle=True for numpy
 * use correct TF version in empty framework_version warning
 * remove logging level overrides

### Documentation changes

 * add environment setup instructions to CONTRIBUTING.md
 * add clarification around framework version constants
 * remove duplicate content from workflow readme
 * remove duplicate content from RL readme

## v1.18.15 (2019-04-18)

### Bug fixes and other changes

 * fix propagation of tags to SageMaker endpoint

## v1.18.14.post1 (2019-04-17)

### Documentation changes

 * remove duplicate content from Chainer readme

## v1.18.14.post0 (2019-04-15)

### Documentation changes

 * remove duplicate content from PyTorch readme and fix internal links

## v1.18.14 (2019-04-11)

### Bug fixes and other changes

 * make Local Mode export artifacts even after failure

## v1.18.13 (2019-04-10)

### Bug fixes and other changes

 * skip horovod p3 test in region with no p3
 * use unique training job names in TensorFlow script mode integ tests

## v1.18.12 (2019-04-08)

### Bug fixes and other changes

 * add integ test for tagging
 * use unique names for test training jobs
 * Wrap horovod code inside main function
 * add csv deserializer
 * restore notebook test

## v1.18.11 (2019-04-04)

### Bug fixes and other changes

 * local data source relative path includes the first directory
 * upgrade pylint and fix tagging with SageMaker models

### Documentation changes

 * add info about unique job names

## v1.18.10 (2019-04-03)

### Bug fixes and other changes

 * make start time, end time and period configurable in sagemaker.analytics.TrainingJobAnalytics

### Documentation changes

 * fix typo of argument spelling in linear learner docstrings

## v1.18.9.post1 (2019-04-02)

### Documentation changes

 * spelling error correction

## v1.18.9.post0 (2019-04-01)

### Documentation changes

 * move RL readme content into sphinx project

## v1.18.9 (2019-03-28)

### Bug fixes

 * hyperparameter query failure on script mode estimator attached to complete job

### Other changes

 * add EI support for TFS framework

### Documentation changes

 * add third-party libraries sections to using_chainer and using_pytorch topics

## v1.18.8 (2019-03-26)

### Bug fixes

 * fix ECR URI validation
 * remove unrestrictive principal * from KMS policy tests.

### Documentation changes

 * edit description of local mode in overview.rst
 * add table of contents to using_chainer topic
 * fix formatting for HyperparameterTuner.attach()

## v1.18.7 (2019-03-21)

### Other changes

 * add pytest marks for integ tests using local mode
 * add account number and unit tests for govcloud

### Documentation changes

 * move chainer readme content into sphinx and fix broken link in using_mxnet

## v1.18.6.post0 (2019-03-20)

### Documentation changes

 * add mandatory sagemaker_role argument to Local mode example.

## v1.18.6 (2019-03-20)

### Changes

 * enable new release process
 * Update inference pipelines documentation
 * Migrate content from workflow and pytorch readmes into sphinx project
 * Propagate Tags from estimator to model, endpoint, and endpoint config

## 1.18.5

* bug-fix: pass kms id as parameter for uploading code with Server side encryption
* feature: ``PipelineModel``: Create a Transformer from a PipelineModel
* bug-fix: ``AlgorithmEstimator``: Make SupportedHyperParameters optional
* feature: ``Hyperparameter``: Support scaling hyperparameters
* doc-fix: Remove duplicate content from main README.rst, /tensorflow/README.rst, and /sklearn/README.rst and add links to readthedocs content

## 1.18.4

* doc-fix: Remove incorrect parameter for EI TFS Python README
* feature: ``Predictor``: delete SageMaker model
* feature: ``PipelineModel``: delete SageMaker model
* bug-fix: Estimator.attach works with training jobs without hyperparameters
* doc-fix: remove duplicate content from mxnet/README.rst
* doc-fix: move overview content in main README into sphynx project
* bug-fix: pass accelerator_type in ``deploy`` for REST API TFS ``Model``
* doc-fix: move content from tf/README.rst into sphynx project
* doc-fix: move content from sklearn/README.rst into sphynx project
* doc-fix: Improve new developer experience in README
* feature: Add support for Coach 0.11.1 for Tensorflow

## 1.18.3.post1

* doc-fix: fix README for PyPI

## 1.18.3

* doc-fix: update information about saving models in the MXNet README
* doc-fix: change ReadTheDocs links from latest to stable
* doc-fix: add ``transform_fn`` information and fix ``input_fn`` signature in the MXNet README
* feature: add support for ``Predictor`` to delete endpoint configuration by default when calling ``delete_endpoint()``
* feature: add support for ``Model`` to delete SageMaker model
* feature: add support for ``Transformer`` to delete SageMaker model
* bug-fix: fix default account for SKLearnModel

## 1.18.2

* enhancement: Include SageMaker Notebook Instance version number in boto3 user agent, if available.
* feature: Support for updating existing endpoint

## 1.18.1

* enhancement: Add ``tuner`` to imports in ``sagemaker/__init__.py``

## 1.18.0

* bug-fix: Handle StopIteration in CloudWatch Logs retrieval
* feature: Update EI TensorFlow latest version to 1.12
* feature: Support for Horovod

## 1.17.2

* feature: HyperparameterTuner: support VPC config

## 1.17.1

* enhancement: Workflow: Specify tasks from which training/tuning operator to transform/deploy in related operators
* feature: Supporting inter-container traffic encryption flag

## 1.17.0

* bug-fix: Workflow: Revert appending Airflow retry id to default job name
* feature: support for Tensorflow 1.12
* feature: support for Tensorflow Serving 1.12
* bug-fix: Revert appending Airflow retry id to default job name
* bug-fix: Session: don't allow get_execution_role() to return an ARN that's not a role but has "role" in the name
* bug-fix: Remove ``__all__`` from ``__init__.py`` files
* doc-fix: Add TFRecord split type to docs
* doc-fix: Mention ``SM_HPS`` environment variable in MXNet README
* doc-fix: Specify that Local Mode supports only framework and BYO cases
* doc-fix: Add missing classes to API docs
* doc-fix: Add information on necessary AWS permissions
* bug-fix: Remove PyYAML to let docker-compose install the right version
* feature: Update TensorFlow latest version to 1.12
* enhancement: Add Model.transformer()
* bug-fix: HyperparameterTuner: make ``include_cls_metadata`` default to ``False`` for everything except Frameworks

## 1.16.3

* bug-fix: Local Mode: Allow support for SSH in local mode
* bug-fix: Workflow: Append retry id to default Airflow job name to avoid name collisions in retry
* bug-fix: Local Mode: No longer requires s3 permissions to run local entry point file
* feature: Estimators: add support for PyTorch 1.0.0
* bug-fix: Local Mode: Move dependency on sagemaker_s3_output from rl.estimator to model
* doc-fix: Fix quotes in estimator.py and model.py

## 1.16.2

* enhancement: Check for S3 paths being passed as entry point
* feature: Add support for AugmentedManifestFile and ShuffleConfig
* bug-fix: Add version bound for requests module to avoid conflicts with docker-compose and docker-py
* bug-fix: Remove unnecessary dependency tensorflow
* doc-fix: Change ``distribution`` to ``distributions``
* bug-fix: Increase docker-compose http timeout and health check timeout to 120.
* feature: Local Mode: Add support for intermediate output to a local directory.
* bug-fix: Update PyYAML version to avoid conflicts with docker-compose
* doc-fix: Correct the numbered list in the table of contents
* doc-fix: Add Airflow API documentation
* feature: HyperparameterTuner: add Early Stopping support

## 1.16.1.post1

* Documentation: add documentation for Reinforcement Learning Estimator.
* Documentation: update TensorFlow README for Script Mode

## 1.16.1

* feature: update boto3 to version 1.9.55

## 1.16.0

* feature: Add 0.10.1 coach version
* feature: Add support for SageMaker Neo
* feature: Estimators: Add RLEstimator to provide support for Reinforcement Learning
* feature: Add support for Amazon Elastic Inference
* feature: Add support for Algorithm Estimators and ModelPackages: includes support for AWS Marketplace
* feature: Add SKLearn Estimator to provide support for SciKit Learn
* feature: Add Amazon SageMaker Semantic Segmentation algorithm to the registry
* feature: Add support for SageMaker Inference Pipelines
* feature: Add support for SparkML serving container

## 1.15.2

* bug-fix: Fix FileNotFoundError for entry_point without source_dir
* doc-fix: Add missing feature 1.5.0 in change log
* doc-fix: Add README for airflow

## 1.15.1

* enhancement: Local Mode: add explicit pull for serving
* feature: Estimators: dependencies attribute allows export of additional libraries into the container
* feature: Add APIs to export Airflow transform and deploy config
* bug-fix: Allow code_location argument to be S3 URI in training_config API
* enhancement: Local Mode: add explicit pull for serving

## 1.15.0

* feature: Estimator: add script mode and Python 3 support for TensorFlow
* bug-fix: Changes to use correct S3 bucket and time range for dataframes in TrainingJobAnalytics.
* bug-fix: Local Mode: correctly handle the case where the model output folder doesn't exist yet
* feature: Add APIs to export Airflow training, tuning and model config
* doc-fix: Fix typos in tensorflow serving documentation
* doc-fix: Add estimator base classes to API docs
* feature: HyperparameterTuner: add support for Automatic Model Tuning's Warm Start Jobs
* feature: HyperparameterTuner: Make input channels optional
* feature: Add support for Chainer 5.0
* feature: Estimator: add support for MetricDefinitions
* feature: Estimators: add support for Amazon IP Insights algorithm

## 1.14.2

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

## 1.14.1

* feature: Estimators: add support for Amazon Object2Vec algorithm

## 1.14.0

* feature: add support for sagemaker-tensorflow-serving container
* feature: Estimator: make input channels optional

## 1.13.0

* feature: Estimator: add input mode to training channels
* feature: Estimator: add model_uri and model_channel_name parameters
* enhancement: Local Mode: support output_path. Can be either file:// or s3://
* enhancement: Added image uris for SageMaker built-in algorithms for SIN/LHR/BOM/SFO/YUL
* feature: Estimators: add support for MXNet 1.3.0, which introduces a new training script format
* feature: Documentation: add explanation for the new training script format used with MXNet
* feature: Estimators: add ``distributions`` for customizing distributed training with the new training script format

## 1.12.0

* feature: add support for TensorFlow 1.11.0

## 1.11.3

* feature: Local Mode: Add support for Batch Inference
* feature: Add timestamp to secondary status in training job output
* bug-fix: Local Mode: Set correct default values for additional_volumes and additional_env_vars
* enhancement: Local Mode: support nvidia-docker2 natively
* warning: Frameworks: add warning for upcoming breaking change that makes framework_version required

## 1.11.2

* enhancement: Enable setting VPC config when creating/deploying models
* enhancement: Local Mode: accept short lived credentials with a warning message
* bug-fix: Local Mode: pass in job name as parameter for training environment variable

## 1.11.1

* enhancement: Local Mode: add training environment variables for AWS region and job name
* doc-fix: Instruction on how to use preview version of PyTorch - 1.0.0.dev.
* doc-fix: add role to MXNet estimator example in readme
* bug-fix: default TensorFlow json serializer accepts dict of numpy arrays

## 1.11.0

* bug-fix: setting health check timeout limit on local mode to 30s
* bug-fix: make Hyperparameters in local mode optional.
* enhancement: add support for volume KMS key to Transformer
* feature: add support for GovCloud

## 1.10.1

* feature: add train_volume_kms_key parameter to Estimator classes
* doc-fix: add deprecation warning for current MXNet training script format
* doc-fix: add docs on deploying TensorFlow model directly from existing model
* doc-fix: fix code example for using Gzip compression for TensorFlow training data

## 1.10.0

* feature: add support for TensorFlow 1.10.0

## 1.9.3.1

* doc-fix: fix rst warnings in README.rst

## 1.9.3

* bug-fix: Local Mode: Create output/data directory expected by SageMaker Container.
* bug-fix: Estimator accepts the vpc configs made capable by 1.9.1

## 1.9.2

* feature: add support for TensorFlow 1.9

## 1.9.1

* bug-fix: Estimators: Fix serialization of single records
* bug-fix: deprecate enable_cloudwatch_metrics from Framework Estimators.
* enhancement: Enable VPC config in training job creation

## 1.9.0

* feature: Estimators: add support for MXNet 1.2.1

## 1.8.0

* bug-fix: removing PCA from tuner
* feature: Estimators: add support for Amazon k-nearest neighbors(KNN) algorithm

## 1.7.2

* bug-fix: Prediction output for the TF_JSON_SERIALIZER
* enhancement: Add better training job status report

## 1.7.1

* bug-fix: get_execution_role no longer fails if user can't call get_role
* bug-fix: Session: use existing model instead of failing during ``create_model()``
* enhancement: Estimator: allow for different role from the Estimator's when creating a Model or Transformer

## 1.7.0

* feature: Transformer: add support for batch transform jobs
* feature: Documentation: add instructions for using Pipe Mode with TensorFlow

## 1.6.1

* feature: Added multiclass classification support for linear learner algorithm.

## 1.6.0

* feature: Add Chainer 4.1.0 support

## 1.5.4

* feature: Added Docker Registry for all 1p algorithms in amazon_estimator.py
* feature: Added get_image_uri method for 1p algorithms in amazon_estimator.py
* Support SageMaker algorithms in FRA and SYD regions

## 1.5.3

* bug-fix: Can create TrainingJobAnalytics object without specifying metric_names.
* bug-fix: Session: include role path in ``get_execution_role()`` result
* bug-fix: Local Mode: fix RuntimeError handling

## 1.5.2

* Support SageMaker algorithms in ICN region

## 1.5.1

* enhancement: Let Framework models reuse code uploaded by Framework estimators
* enhancement: Unify generation of model uploaded code location
* feature: Change minimum required scipy from 1.0.0 to 0.19.0
* feature: Allow all Framework Estimators to use a custom docker image.
* feature: Option to add Tags on SageMaker Endpoints

## 1.5.0

* feature: Add Support for PyTorch Framework
* feature: Estimators: add support for TensorFlow 1.7.0
* feature: Estimators: add support for TensorFlow 1.8.0
* feature: Allow Local Serving of Models in S3
* enhancement: Allow option for ``HyperparameterTuner`` to not include estimator metadata in job
* bug-fix: Estimators: Join tensorboard thread after fitting

## 1.4.2

* bug-fix: Estimators: Fix attach for LDA
* bug-fix: Estimators: allow code_location to have no key prefix
* bug-fix: Local Mode: Fix s3 training data download when there is a trailing slash

## 1.4.1

* bug-fix: Local Mode: Fix for non Framework containers

## 1.4.0

* bug-fix: Remove __all__ and add noqa in __init__
* bug-fix: Estimators: Change max_iterations hyperparameter key for KMeans
* bug-fix: Estimators: Remove unused argument job_details for ``EstimatorBase.attach()``
* bug-fix: Local Mode: Show logs in Jupyter notebooks
* feature: HyperparameterTuner: Add support for hyperparameter tuning jobs
* feature: Analytics: Add functions for metrics in Training and Hyperparameter Tuning jobs
* feature: Estimators: add support for tagging training jobs

## 1.3.0

* feature: Add chainer

## 1.2.5

* bug-fix: Change module names to string type in __all__
* feature: Save training output files in local mode
* bug-fix: tensorflow-serving-api: SageMaker does not conflict with tensorflow-serving-api module version
* feature: Local Mode: add support for local training data using file://
* feature: Updated TensorFlow Serving api protobuf files
* bug-fix: No longer poll for logs from stopped training jobs

## 1.2.4

* feature: Estimators: add support for Amazon Random Cut Forest algorithm

## 1.2.3

* bug-fix: Fix local mode not using the right s3 bucket

## 1.2.2

* bug-fix: Estimators: fix valid range of hyper-parameter 'loss' in linear learner

## 1.2.1

* bug-fix: Change Local Mode to use a sagemaker-local docker network

## 1.2.0

* feature: Add Support for Local Mode
* feature: Estimators: add support for TensorFlow 1.6.0
* feature: Estimators: add support for MXNet 1.1.0
* feature: Frameworks: Use more idiomatic ECR repository naming scheme

## 1.1.3

* bug-fix: TensorFlow: Display updated data correctly for TensorBoard launched from ``run_tensorboard_locally=True``
* feature: Tests: create configurable ``sagemaker_session`` pytest fixture for all integration tests
* bug-fix: Estimators: fix inaccurate hyper-parameters in kmeans, pca and linear learner
* feature: Estimators: Add new hyperparameters for linear learner.

## 1.1.2

* bug-fix: Estimators: do not call create bucket if data location is provided

## 1.1.1

* feature: Estimators: add ``requirements.txt`` support for TensorFlow

## 1.1.0

* feature: Estimators: add support for TensorFlow-1.5.0
* feature: Estimators: add support for MXNet-1.0.0
* feature: Tests: use ``sagemaker_timestamp`` when creating endpoint names in integration tests
* feature: Session: print out billable seconds after training completes
* bug-fix: Estimators: fix LinearLearner and add unit tests
* bug-fix: Tests: fix timeouts for PCA async integration test
* feature: Predictors: allow ``predictor.predict()`` in the JSON serializer to accept dictionaries

## 1.0.4

* feature: Estimators: add support for Amazon Neural Topic Model(NTM) algorithm
* feature: Documentation: fix description of an argument of sagemaker.session.train
* feature: Documentation: add FM and LDA to the documentation
* feature: Estimators: add support for async fit
* bug-fix: Estimators: fix estimator role expansion

## 1.0.3

* feature: Estimators: add support for Amazon LDA algorithm
* feature: Hyperparameters: add data_type to hyperparameters
* feature: Documentation: update TensorFlow examples following API change
* feature: Session: support multi-part uploads
* feature: add new SageMaker CLI

## 1.0.2

* feature: Estimators: add support for Amazon FactorizationMachines algorithm
* feature: Session: correctly handle TooManyBuckets error_code in default_bucket method
* feature: Tests: add training failure tests for TF and MXNet
* feature: Documentation: show how to make predictions against existing endpoint
* feature: Estimators: implement write_spmatrix_to_sparse_tensor to support any scipy.sparse matrix

## 1.0.1

* api-change: Model: Remove support for 'supplemental_containers' when creating Model
* feature: Documentation: multiple updates
* feature: Tests: ignore tests data in tox.ini, increase timeout for endpoint creation, capture exceptions during endpoint deletion, tests for input-output functions
* feature: Logging: change to describe job every 30s when showing logs
* feature: Session: use custom user agent at all times
* feature: Setup: add travis file

## 1.0.0

* Initial commit
