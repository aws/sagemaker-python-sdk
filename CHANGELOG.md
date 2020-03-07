# Changelog

## v1.50.18.post0 (2020-03-05)

### Documentation Changes

 * correct Estimator code_location default S3 path

## v1.50.18 (2020-03-04)

### Bug Fixes and Other Changes

 * change default compile model max run to 15 mins

## v1.50.17.post0 (2020-03-03)

### Testing and Release Infrastructure

 * fix PR builds to run on changes to their own buildspecs
 * programmatically determine partition based on region

## v1.50.17 (2020-02-27)

### Bug Fixes and Other Changes

 * upgrade framework versions

## v1.50.16 (2020-02-26)

### Bug Fixes and Other Changes

 * use sagemaker_session when initializing Constraints and Statistics
 * add sagemaker_session parameter to DataCaptureConfig
 * make AutoML.deploy use self.sagemaker_session by default

### Testing and Release Infrastructure

 * unset region during integ tests
 * use sagemaker_session fixture in all Airflow tests
 * remove remaining TF legacy mode integ tests

## v1.50.15 (2020-02-25)

### Bug Fixes and Other Changes

 * enable Neo integ tests

## v1.50.14.post0 (2020-02-24)

### Testing and Release Infrastructure

 * remove TF framework mode notebooks from PR build
 * don't create docker network for all integ tests

## v1.50.14 (2020-02-20)

### Bug Fixes and Other Changes

 * don't use os.path.join for S3 path when repacking TFS model
 * dynamically determine AWS domain based on region

## v1.50.13 (2020-02-19)

### Bug Fixes and Other Changes

 * allow download_folder to download file even if bucket is more restricted

### Testing and Release Infrastructure

 * configure pylint to recognize boto3 and botocore as third-party imports
 * add multiple notebooks to notebook PR build

## v1.50.12 (2020-02-17)

### Bug Fixes and Other Changes

 * enable network isolation for amazon estimators

### Documentation Changes

 * clarify channel environment variables in PyTorch documentation

## v1.50.11 (2020-02-13)

### Bug Fixes and Other Changes

 * fix HyperparameterTuner.attach for Marketplace algorithms
 * move requests library from required packages to test dependencies
 * create Session or LocalSession if not specified in Model

### Documentation Changes

 * remove hardcoded list of target devices in compile()
 * Fix typo with SM_MODEL_DIR, missing quotes

## v1.50.10.post0 (2020-02-12)

### Documentation Changes

 * add documentation guidelines to CONTRIBUTING.md
 * Removed section numbering

## v1.50.10 (2020-02-11)

### Bug Fixes and Other Changes

 * remove NEO_ALLOWED_TARGET_INSTANCE_FAMILY

## v1.50.9.post0 (2020-02-06)

### Documentation Changes

 * remove labels from issue templates

## v1.50.9 (2020-02-04)

### Bug Fixes and Other Changes

 * account for EI and version-based ECR repo naming in serving_image_uri()

### Documentation Changes

 * correct broken AutoML API documentation link
 * fix MXNet version lists

## v1.50.8 (2020-01-30)

### Bug Fixes and Other Changes

 * disable Debugger defaults in unsupported regions
 * modify session and kms_utils to check for S3 bucket before creation
 * update docker-compose and PyYAML dependencies
 * enable smdebug for Horovod (MPI) training setup
 * create lib dir for dependencies safely (only if it doesn't exist yet).
 * create the correct session for MultiDataModel

### Documentation Changes

 * update links to the local mode notebooks examples.
 * Remove outdated badges from README
 * update links to TF notebook examples to link to script mode examples.
 * clean up headings, verb tenses, names, etc. in MXNet overview
 * Update SageMaker operator Helm chart installation guide

### Testing and Release Infrastructure

 * choose faster notebook for notebook PR build
 * properly fail PR build if has-matching-changes fails
 * properly fail PR build if has-matching-changes fails

## v1.50.7 (2020-01-20)

### Bug fixes and other changes

 * do not use script for TFS when entry_point is not provided
 * remove usage of pkg_resources
 * update py2 warning message since python 2 is deprecated
 * cleanup experiments, trials, and trial components in integ tests

## v1.50.6.post0 (2020-01-20)

### Documentation changes

 * add additional information to Transformer class transform function doc string

## v1.50.6 (2020-01-18)

### Bug fixes and other changes

 * Append serving to model framework name for PyTorch, MXNet, and TensorFlow

## v1.50.5 (2020-01-17)

### Bug fixes and other changes

 * Use serving_image_uri for Airflow

### Documentation changes

 * revise Processing docstrings for formatting and class links
 * Add processing readthedocs

## v1.50.4 (2020-01-16)

### Bug fixes and other changes

 * Remove version number from default version comment
 * remove remaining instances of python-dateutil pin
 * upgrade boto3 and remove python-dateutil pin

### Documentation changes

 * Add issue templates and configure issue template chooser
 * Update error type in delete_endpoint docstring
 * add version requirement for using "requirements.txt" when serving an MXNet model
 * update container dependency versions for MXNet and PyTorch
 * Update supported versions of PyTorch

## v1.50.3 (2020-01-15)

### Bug fixes and other changes

 * ignore private Automatic Model Tuning hyperparameter when attaching AlgorithmEstimator

### Documentation changes

 * add Debugger API docs

## v1.50.2 (2020-01-14)

### Bug fixes and other changes

 * add tests to quick canary
 * honor 'wait' flag when updating endpoint
 * add default framework version warning message in Model classes
 * Adding role arn explanation for sagemaker role
 * allow predictor to be returned from AutoML.deploy()
 * add PR checklist item about unique_name_from_base()
 * use unique_name_from_base for multi-algo tuning test
 * update copyright year in license header

### Documentation changes

 * add version requirement for using "requirement.txt" when serving a PyTorch model
 * add SageMaker Debugger overview
 * clarify requirements.txt usage for Chainer, MXNet, and Scikit-learn
 * change "associate" to "create" for OpenID connector
 * fix typo and improve clarity on installing packages via "requirements.txt"

## v1.50.1 (2020-01-07)

### Bug fixes and other changes

 * fix PyTorchModel deployment crash on Windows
 * make PyTorch empty framework_version warning include the latest PyTorch version

## v1.50.0 (2020-01-06)

### Features

 * allow disabling debugger_hook_config

### Bug fixes and other changes

 * relax urllib3 and requests restrictions.
 * Add uri as return statement for upload_string_as_file_body
 * refactor logic in fw_utils and fill in docstrings
 * increase poll from 5 to 30 for DescribeEndpoint lambda.
 * fix test_auto_ml tests for regions without ml.c4.xlarge hosts.
 * fix test_processing for regions without m4.xlarge instances.
 * reduce test's describe frequency to eliminate throttling error.
 * Increase number of retries when describing an endpoint since tf-2.0 has larger images and takes longer to start.

### Documentation changes

 * generalize Model Monitor documentation from SageMaker Studio tutorial

## v1.49.0 (2019-12-23)

### Features

 * Add support for TF-2.0.0.
 * create ProcessingJob from ARN and from name

### Bug fixes and other changes

 * Make tf tests tf-1.15 and tf-2.0 compatible.

### Documentation changes

 * add Model Monitor documentation
 * add link to Amazon algorithm estimator parent class to clarify **kwargs

## v1.48.1 (2019-12-18)

### Bug fixes and other changes

 * use name_from_base in auto_ml.py but unique_name_from_base in tests.
 * make test's custom bucket include region and account name.
 * add Keras to the list of Neo-supported frameworks

### Documentation changes

 * add link to parent classes to clarify **kwargs
 * add link to framework-related parent classes to clarify **kwargs

## v1.48.0 (2019-12-17)

### Features

 * allow setting the default bucket in Session

### Bug fixes and other changes

 * set integration test parallelization to 512
 * shorten base job name to avoid collision
 * multi model integration test to create ECR repo with unique names to allow independent parallel executions

## v1.47.1 (2019-12-16)

### Bug fixes and other changes

 * Revert "feature: allow setting the default bucket in Session (#1168)"

### Documentation changes

 * add AutoML README
 * add missing classes to API docs

## v1.47.0 (2019-12-13)

### Features

 * allow setting the default bucket in Session

### Bug fixes and other changes

 * allow processing users to run code in s3

## v1.46.0 (2019-12-12)

### Features

 * support Multi-Model endpoints

### Bug fixes and other changes

 * update PR template with items about tests, regional endpoints, and API docs

## v1.45.2 (2019-12-10)

### Bug fixes and other changes

 * modify schedule cleanup to abide by latest validations
 * lower log level when getting execution role from a SageMaker Notebook
 * Fix "ValueError: too many values to unpack (expected 2)" is occurred in windows local mode
 * allow ModelMonitor and Processor to take IAM role names (in addition to ARNs)

### Documentation changes

 * mention that the entry_point needs to be named inference.py for tfs

## v1.45.1 (2019-12-06)

### Bug fixes and other changes

 * create auto ml job for tests that based on existing job
 * fixing py2 support for latest TF version
 * fix tags in deploy call for generic estimators
 * make multi algo integration test assertion less specific

## v1.45.0 (2019-12-04)

### Features

 * add support for TF 1.15.0, PyTorch 1.3.1 and MXNet 1.6rc0.
 * add S3Downloader.list(s3_uri) functionality
 * introduce SageMaker AutoML
 * wrap up Processing feature
 * add a few minor features to Model Monitoring
 * add enable_sagemaker_metrics flag
 * Amazon SageMaker Model Monitoring
 * add utils.generate_tensorboard_url function
 * Add jobs list to Estimator

### Bug fixes and other changes

 * remove unnecessary boto model files
 * update boto version to >=1.10.32
 * correct Debugger tests
 * fix bug in monitor.attach() for empty network_config
 * Import smdebug_rulesconfig from PyPI
 * bump the version to 1.45.0 (publishes 1.46.0) for re:Invent-2019
 * correct AutoML imports and expose current_job_name
 * correct Model Monitor eu-west-3 image name.
 * use DLC prod images
 * remove unused env variable for Model Monitoring
 * aws model update
 * rename get_debugger_artifacts to latest_job_debugger_artifacts
 * remove retain flag from update_endpoint
 * correct S3Downloader behavior
 * consume smdebug_ruleconfig .whl for ITs
 * disable DebuggerHook and Rules for TF distributions
 * incorporate smdebug_ruleconfigs pkg until availability in PyPI
 * remove pre/post scripts per latest validations
 * update rules_config .whl
 * remove py_version from SKLearnProcessor
 * AutoML improvements
 * stop overwriting custom rules volume and type
 * fix tests due to latest server-side validations
 * Minor processing changes
 * minor processing changes (instance_count + docs)
 * update api to latest
 * Eureka master
 * Add support for xgboost version 0.90-2
 * SageMaker Debugger revision
 * Add support for SageMaker Debugger [WIP]
 * Fix linear learner crash when num_class is string and predict type is `multiclass_classifier`
 * Additional Processing Jobs integration tests
 * Migrate to updated Processing Jobs API
 * Processing Jobs revision round 2
 * Processing Jobs revision
 * remove instance_pools parameter from tuner
 * Multi-Algorithm Hyperparameter Tuning Support
 * Import Processors in init files
 * Remove SparkML Processors and corresponding unit tests
 * Processing Jobs Python SDK support

## v1.44.4 (2019-12-02)

### Bug fixes and other changes

 * Documentation for Amazon Sagemaker Operators

## v1.44.3 (2019-11-26)

### Bug fixes and other changes

 * move sagemaker config loading to LocalSession since it is only used for local code support.

### Documentation changes

 * fix docstring wording.

## v1.44.2 (2019-11-25)

### Bug fixes and other changes

 * add pyyaml dependencies to the required list.

### Documentation changes

 * Correct info on code_location parameter

## v1.44.1 (2019-11-21)

### Bug fixes and other changes

 * Remove local mode dependencies from required.

## v1.44.0 (2019-11-21)

### Features

 * separating sagemaker dependencies into more use case specific installable components.

### Bug fixes and other changes

 * remove docker-compose as a required dependency.

## v1.43.5 (2019-11-18)

### Bug fixes and other changes

 * remove red from possible colors when streaming logs

## v1.43.4.post1 (2019-10-29)

### Documentation changes

 * clarify that source_dir can be an S3 URI

## v1.43.4.post0 (2019-10-28)

### Documentation changes

 * clarify how to use parameter servers with distributed MXNet training

## v1.43.4 (2019-10-24)

### Bug fixes and other changes

 * use regional endpoint for STS in builds and tests

### Documentation changes

 * update link to point to ReadTheDocs

## v1.43.3 (2019-10-23)

### Bug fixes and other changes

 * exclude regions for P2 tests

## v1.43.2 (2019-10-21)

### Bug fixes and other changes

 * add support for me-south-1 region

## v1.43.1 (2019-10-17)

### Bug fixes and other changes

 * validation args now use default framework_version for TensorFlow

## v1.43.0 (2019-10-16)

### Features

 * Add support for PyTorch 1.2.0

## v1.42.9 (2019-10-14)

### Bug fixes and other changes

 * use default bucket for checkpoint_s3_uri integ test
 * use sts regional endpoint when creating default bucket
 * use us-west-2 endpoint for sts in buildspec
 * take checkpoint_s3_uri and checkpoint_local_path in Framework class

## v1.42.8 (2019-10-10)

### Bug fixes and other changes

 * add kwargs to create_model for 1p to work with kms

## v1.42.7 (2019-10-09)

### Bug fixes and other changes

 * paginating describe log streams

## v1.42.6.post0 (2019-10-07)

### Documentation changes

 * model local mode

## v1.42.6 (2019-10-03)

### Bug fixes and other changes

 * update tfs documentation for requirements.txt
 * support content_type in FileSystemInput
 * allowing account overrides in special regions

## v1.42.5 (2019-10-02)

### Bug fixes and other changes

 * update using_mxnet.rst

## v1.42.4 (2019-10-01)

### Bug fixes and other changes

 * Revert "fix issue-987 error by adding instance_type in endpoint_name (#1058)"
 * fix issue-987 error by adding instance_type in endpoint_name

## v1.42.3 (2019-09-26)

### Bug fixes and other changes

 * preserve EnableNetworkIsolation setting in attach
 * enable kms support for repack_model
 * support binary by NoneSplitter.
 * stop CI unit test code checks from running in parallel

## v1.42.2 (2019-09-25)

### Bug fixes and other changes

 * re-enable airflow_config tests

## v1.42.1 (2019-09-24)

### Bug fixes and other changes

 * lazy import of tensorflow module
 * skip airflow_config tests as they're blocking the release build
 * skip lda tests in regions that does not support it.
 * add airflow_config tests to canaries
 * use correct STS endpoint for us-iso-east-1

## v1.42.0 (2019-09-20)

### Features

 * add estimator preparation to airflow configuration

### Bug fixes and other changes

 * correct airflow workflow for BYO estimators.

## v1.41.0 (2019-09-20)

### Features

 * enable sklearn for network isolation mode

## v1.40.2 (2019-09-19)

### Bug fixes and other changes

 * use new ECR images in us-iso-east-1 for TF and MXNet

## v1.40.1 (2019-09-18)

### Bug fixes and other changes

 * expose kms_key parameter for deploying from training and hyperparameter tuning jobs

### Documentation changes

 * Update sklearn default predict_fn

## v1.40.0 (2019-09-17)

### Features

 * add support to TF 1.14 serving with elastic accelerator.

## v1.39.4 (2019-09-17)

### Bug fixes and other changes

 * pass enable_network_isolation when creating TF and SKLearn models

## v1.39.3 (2019-09-16)

### Bug fixes and other changes

 * expose vpc_config_override in transformer() methods
 * use Estimator.create_model in Estimator.transformer

## v1.39.2 (2019-09-11)

### Bug fixes and other changes

 * pass enable_network_isolation in Estimator.create_model
 * use p2 instead of p3 for the Horovod test

## v1.39.1 (2019-09-10)

### Bug fixes and other changes

 * copy dependencies into new folder when repacking model
 * make get_caller_identity_arn get role from DescribeNotebookInstance
 * add https to regional STS endpoint
 * clean up git support integ tests

## v1.39.0 (2019-09-09)

### Features

 * Estimator.fit like logs for transformer
 * handler for stopping transform job

### Bug fixes and other changes

 * remove hardcoded creds from integ test
 * remove hardcoded creds from integ test
 * Fix get_image_uri warning log for default xgboost version.
 * add enable_network_isolation to generic Estimator class
 * use regional endpoint when creating AWS STS client
 * update Sagemaker Neo regions
 * use cpu_instance_type fixture for stop_transform_job test
 * hyperparameter tuning with spot instances and checkpoints
 * skip efs and fsx integ tests in all regions

### Documentation changes

 * clarify some Local Mode limitations

## v1.38.6 (2019-09-04)

### Bug fixes and other changes

 * update: disable efs fsx integ tests in non-pdx regions
 * fix canary test failure issues
 * use us-east-1 for PR test runs

### Documentation changes

 * updated description for "accept" parameter in batch transform

## v1.38.5 (2019-09-02)

### Bug fixes and other changes

 * clean up resources created by file system set up when setup fails

## v1.38.4 (2019-08-29)

### Bug fixes and other changes

 * skip EFS tests until they are confirmed fixed.

### Documentation changes

 * add note to CONTRIBUTING to clarify automated formatting
 * add checkpoint section to using_mxnet topic

## v1.38.3 (2019-08-28)

### Bug fixes and other changes

 * change AMI ids in tests to be dynamic based on regions

## v1.38.2 (2019-08-27)

### Bug fixes and other changes

 * skip efs tests in non us-west-2 regions
 * refactor tests to use common retry method

## v1.38.1 (2019-08-26)

### Bug fixes and other changes

 * update py2 warning message
 * add logic to use asimov image for TF 1.14 py2

### Documentation changes

 * changed EFS directory path instructions in documentation and Docstrings

## v1.38.0 (2019-08-23)

### Features

 * support training inputs from EFS and FSx

## v1.37.2 (2019-08-20)

### Bug fixes and other changes

 * Add support for Managed Spot Training and Checkpoint support
 * Integration Tests now dynamically checks AZs

## v1.37.1 (2019-08-19)

### Bug fixes and other changes

 * eliminate dependency on mnist dataset website

### Documentation changes

 * refactor using_sklearn and fix minor errors in using_pytorch and using_chainer

## v1.37.0 (2019-08-15)

### Features

 * add XGBoost Estimator as new framework

### Bug fixes and other changes

 * fix tests for new regions
 * add update_endpoint for PipelineModel

### Documentation changes

 * refactor the using Chainer topic

## v1.36.4 (2019-08-13)

### Bug fixes and other changes

 * region build from staging pr

### Documentation changes

 * Refactor Using PyTorch topic for consistency

## v1.36.3 (2019-08-13)

### Bug fixes and other changes

 * fix integration test failures masked by timeout bug
 * prevent multiple values error in sklearn.transformer()
 * model.transformer() passes tags to create_model()

## v1.36.2 (2019-08-12)

### Bug fixes and other changes

 * rework CONTRIBUTING.md to include a development workflow

## v1.36.1 (2019-08-08)

### Bug fixes and other changes

 * prevent integration test's timeout functions from hiding failures

### Documentation changes

 * correct typo in using_sklearn.rst

## v1.36.0 (2019-08-07)

### Features

 * support for TensorFlow 1.14

### Bug fixes and other changes

 * ignore FI18 flake8 rule
 * allow Airflow enabled estimators to use absolute path entry_point

## v1.35.1 (2019-08-01)

### Bug fixes and other changes

 * update sklearn document to include 3p dependency installation

### Documentation changes

 * refactor and edit using_mxnet topic

## v1.35.0 (2019-07-31)

### Features

 * allow serving image to be specified when calling MXNet.deploy

## v1.34.3 (2019-07-30)

### Bug fixes and other changes

 * waiting for training tags to propagate in the test

## v1.34.2 (2019-07-29)

### Bug fixes and other changes

 * removing unnecessary tests cases
 * Replaced generic ValueError with custom subclass when reporting unexpected resource status

### Documentation changes

 * correct wording for Cloud9 environment setup instructions

## v1.34.1 (2019-07-23)

### Bug fixes and other changes

 * enable line-too-long Pylint check
 * improving Chainer integ tests
 * update TensorFlow script mode dependency list
 * improve documentation of some functions
 * update PyTorch version
 * allow serving script to be defined for deploy() and transformer() with frameworks
 * format and add missing docstring placeholders
 * add MXNet 1.4.1 support

### Documentation changes

 * add instructions for setting up Cloud9 environment.
 * update using_tensorflow topic

## v1.34.0 (2019-07-18)

### Features

 * Git integration for CodeCommit
 * deal with credentials for Git support for GitHub

### Bug fixes and other changes

 * modify TODO on disabled Pylint check
 * enable consider-using-ternary Pylint check
 * enable chained-comparison Pylint check
 * enable too-many-public-methods Pylint check
 * enable consider-using-in Pylint check
 * set num_processes_per_host only if provided by user
 * fix attach for 1P algorithm estimators
 * enable ungrouped-imports Pylint check
 * enable wrong-import-order Pylint check
 * enable attribute-defined-outside-init Pylint check
 * enable consider-merging-isinstance Pylint check
 * enable inconsistent-return-statements Pylint check
 * enable simplifiable-if-expression pylint checks
 * fix list serialization for 1P algos
 * enable no-else-return and no-else-raise pylint checks
 * enable unidiomatic-typecheck pylint check

## v1.33.0 (2019-07-10)

### Features

 * git support for hosting models
 * allow custom model name during deploy

### Bug fixes and other changes

 * remove TODO comment on import-error Pylint check
 * enable wrong-import-position pylint check
 * Revert "change: enable wrong-import-position pylint check (#907)"
 * enable signature-differs pylint check
 * enable wrong-import-position pylint check
 * enable logging-not-lazy pylint check
 * reset default output path in Transformer.transform
 * Add ap-northeast-1 to Neo algorithms region map

## v1.32.2 (2019-07-08)

### Bug fixes and other changes

 * enable logging-format-interpolation pylint check
 * remove superfluous parens per Pylint rule

### Documentation changes

 * add pypi, rtd, black badges to readme

## v1.32.1 (2019-07-04)

### Bug fixes and other changes

 * correct code per len-as-condition Pylint check
 * tighten pylint config and expand C and R exceptions
 * Update displaytime.sh
 * fix notebook tests
 * separate unit, local mode, and notebook tests in different buildspecs

### Documentation changes

 * refactor the overview topic in the sphinx project

## v1.32.0 (2019-07-02)

### Features

 * support Endpoint_type for TF transform

### Bug fixes and other changes

 * fix git test in test_estimator.py
 * Add ap-northeast-1 to Neo algorithms region map

## v1.31.1 (2019-07-01)

### Bug fixes and other changes

 * print build execution time
 * remove unnecessary failure case tests
 * build spec improvements.

## v1.31.0 (2019-06-27)

### Features

 * use deep learning images

### Bug fixes and other changes

 * Update buildspec.yml
 * allow only one integration test run per time
 * remove unnecessary P3 tests from TFS integration tests
 * add pytest.mark.local_mode annotation to broken tests

## v1.30.0 (2019-06-25)

### Features

 * add TensorFlow 1.13 support
 * add git_config and git_clone, validate method

### Bug fixes and other changes

 * add pytest.mark.local_mode annotation to broken tests

## v1.29.0 (2019-06-24)

### Features

 * network isolation mode in training

### Bug fixes and other changes

 * Integrate black into development process
 * moving not canary TFS tests to local mode

## v1.28.3 (2019-06-20)

### Bug fixes and other changes

 * update Sagemaker Neo regions and instance families

### Documentation changes

 * fix punctuation in MXNet version list
 * clean up MXNet and TF documentation

## v1.28.2 (2019-06-19)

### Bug fixes and other changes

 * prevent race condition in vpc tests

## v1.28.1 (2019-06-17)

### Bug fixes and other changes

 * Update setup.py

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
