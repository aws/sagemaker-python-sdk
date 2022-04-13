# Changelog

## v2.77.1 (2022-02-25)

### Bug Fixes and Other Changes

 * jumpstart model table

## v2.77.0 (2022-02-22)

### Features

 * override jumpstart content bucket
 * jumpstart model id suggestions
 * adding customer metadata support to registermodel step

### Bug Fixes and Other Changes

 * Improve Pipeline workflow unit test branch coverage
 * update lineage_trial_compoment get pipeline execution arn
 * Add lineage doc
 * Support primitive types for left value of ConditionSteps

## v2.76.0 (2022-02-17)

### Features

 * Add FailStep Support for Sagemaker Pipeline

### Bug Fixes and Other Changes

 * use recommended inference image uri from Neo API
 * pin test dependencies
 * Add exception in test_action
 * Update Static Endpoint
 * Add CMH to the non-P3 list

### Documentation Changes

 * Support for generation of Jumpstart model table on build

## v2.75.1 (2022-02-08)

### Bug Fixes and Other Changes

 * Add CMH to the non-P3 list

## v2.75.0 (2022-02-05)

### Features

 * JumpStart Integration
 * Adds support for async inference
 * Update instance types for integ test

### Bug Fixes and Other Changes

 * Revert "feature: CompilationStep support for Sagemaker Pipelines
 * gpu use p3/p2 per avail for region
 * jumpstart typo
 * pin pytest-xdist to avoid release failures
 * set sagemaker_connection and image_uri in register method
 * update to incorporate black v22, pin tox versions
 * Add deprecation warning in Clarify DataConfig

### Documentation Changes

 * Jumpstart doc strings and added new sections
 * Add Jumpstart support documentation

## v2.74.0 (2022-01-26)

### Features

 * Add support for SageMaker lineage queries context

### Bug Fixes and Other Changes

 * support specifying a facet by its column index

### Documentation Changes

 * more documentation for serverless inference

## v2.73.0 (2022-01-19)

### Features

 * Add EMRStep support in Sagemaker pipeline
 * Adds Lineage queries in artifact, context and trial components
 * Add support for SageMaker lineage queries in action
 * Adds support for Serverless inference
 * support checkpoint to be passed from estimator
 * support JsonGet/Join parameterization in tuning step Hyperparameters
 * Support model pipelines in CreateModelStep
 * enable python 3.9
 * Add models_v2 under lineage context

### Bug Fixes and Other Changes

 * allow kms_key to be passed for processing step
 * Remove duplicate vertex/edge in query lineage
 * update pricing link
 * Update CHANGELOG.md
 * fixes unnecessary session call while generating pipeline definition for lambda step

### Documentation Changes

 * Enhance smddp 1.2.2 doc
 * Document the available ExecutionVariables

## v2.72.3 (2022-01-10)

### Features

 * default repack encryption
 * support large pipeline
 * add support for pytorch 1.10.0
 
### Documentation Changes

 * SageMaker model parallel library 1.6.0 API doc

### Bug Fixes and Other Changes

 * Model Registration with BYO scripts
 * Add ContentType in test_auto_ml_describe
 * Re-deploy static integ test endpoint if it is not found
 * fix kmeans test deletion sequence, increment lineage statics
 * Increment static lineage pipeline
 * Fix lineage query integ tests
 * Add label_headers option for Clarify ModelExplainabilityMonitor
 * Add action type to lineage object
 * Collapse cross-account artifacts in query lineage response
 * Update CHANGELOG.md to remove defaulting dot characters

## v2.72.2 (2022-01-06)

### Bug Fixes and Other Changes

 * Update CHANGELOG.md
 * Increment static lineage pipeline
 * fix kmeans test deletion sequence, increment lineage statics
 * Re-deploy static integ test endpoint if it is not found
 * Add ContentType in test_auto_ml_describe
 * Model Registration with BYO scripts

### Documentation Changes

 * SageMaker model parallel library 1.6.0 API doc

## v2.72.1 (2021-12-20)

### Bug Fixes and Other Changes

 * typos and broken link
 * S3Input - add support for instance attributes
 * Prevent repack_model script from referencing nonexistent directories
 * Set ProcessingStep upload locations deterministically to avoid cache

## v2.72.0 (2021-12-13)

### Features

 * allow conditional parellel builds

### Bug Fixes and Other Changes

 * local mode - support relative file structure
 * fix endpoint bug

## v2.71.0 (2021-12-06)

### Features

 * Add support for TF 2.6
 * Adding PT 17/18 Repo
 * Add profile_name support for Feature Store ingestion

### Bug Fixes and Other Changes

 * Fix non-existent variable name
 * Add TF 2.6.2 on training
 * Recreate static lineage test data

## v2.70.0 (2021-12-02)

### Features

 * update boto3 minor version >= 1.20.18
 * Add support for SageMaker lineage queries
 * add CV shap explainability for SageMaker Clarify
 * add NLP support for SageMaker Clarify
 * Add support for ModelMonitor/Clarify integration in model building pipelines
 * adding support for transformers 4.11 for SM Training Compiler
 * SM Training Compiler with an UI to enable/disable compilation for HuggingFace DLCs to speedup training

### Bug Fixes and Other Changes

 * pin coveragepy
 * Add support for PyTorch 1.9.1
 * Update s3 path of scheduling analysis config on ClarifyCheckStep
 * documentation/logging to indicate correct place for DEBUG artifacts from SM trcomp
 * validate requested transformers version and use the best available version
 * Install custom pkgs

## v2.69.0 (2021-11-12)

### Features

 * Hugging Face Transformers 4.12 for Pt1.9/TF2.5

## v2.68.0 (2021-11-02)

### Features

 * CompilationStep support for Sagemaker Pipelines

## v2.67.0 (2021-11-01)

### Deprecations and Removals

 * deprecate Serverless Lambda model-predictor

### Features

 * add joinsource to DataConfig
 * Add support for Partial Dependence Plots(PDP) in SageMaker Clarify

### Bug Fixes and Other Changes

 * localmode subprocess parent process not sending SIGTERM to child
 * remove buildspec from repo

## v2.66.2.post0 (2021-10-28)

### Documentation Changes

 * Update estimator docstrings to add Fast File Mode

## v2.66.2 (2021-10-27)

### Bug Fixes and Other Changes

 * expose num_clusters parameter for clarify shap in shapconfig
 * Update cron job to run hourly

## v2.66.1 (2021-10-26)

### Bug Fixes and Other Changes

 * HuggingFace image_uri generation for inference
 * Update '_' and '/' with '-' in filename creation

## v2.66.0 (2021-10-25)

### Features

 * Add image_uris.retrieve() support for AutoGluon

### Documentation Changes

 * fix documentation for input types in estimator.fit
 * Add JsonGet v2 deprecation

## v2.65.0 (2021-10-21)

### Features

 * modify RLEstimator to use newly generated Ray image (1.6.0)
 * network isolation mode for xgboost
 * update clarify imageURI for PDT

### Bug Fixes and Other Changes

 * retry downstream_trials test
 * Add retries to pipeline execution

## v2.64.0 (2021-10-20)

### Deprecations and Removals

 * warn for deprecation - Lambda model-predictor

### Features

 * Add support for TF 2.5
 * Add a pre-push git hook

### Bug Fixes and Other Changes

 * add s3_analysis_config_output_path field in DataConfig constructor
 * make marketplace jobnames random

## v2.63.2 (2021-10-18)

### Bug Fixes and Other Changes

 * Update timeouts for integ tests from 20 to 40

## v2.63.1 (2021-10-14)

### Bug Fixes and Other Changes

 * HF estimator attach modified to work with py38

## v2.63.0 (2021-10-13)

### Features

 * support configurable retry for pipeline steps

## v2.62.0 (2021-10-12)

### Features

 * Hugging Face Transformers 4.10 for Pt1.8/TF2.4 & Transformers 4.11 for PT1.9&TF2.5

### Bug Fixes and Other Changes

 * repack_model script used in pipelines to support source_dir and dependencies

## v2.61.0 (2021-10-11)

### Features

 * add support for PyTorch 1.9.0

### Bug Fixes and Other Changes

 * Update TRAINING_DEFAULT_TIMEOUT_MINUTES to 40 min
 * notebook test for parallel PRs

## v2.60.0 (2021-10-08)

### Features

 * Add support for Hugging Face 4.10.2

## v2.59.8 (2021-10-07)

### Bug Fixes and Other Changes

 * fix feature store ingestion via data wrangler test

## v2.59.7 (2021-10-04)

### Bug Fixes and Other Changes

 * update feature request label
 * update bug template

## v2.59.6 (2021-09-30)

### Bug Fixes and Other Changes

 * ParamValidationError when scheduling a Clarify model monitor

## v2.59.5 (2021-09-29)

### Bug Fixes and Other Changes

 * support maps in step parameters

## v2.59.4 (2021-09-27)

### Bug Fixes and Other Changes

 * add checks for ExecutionRole in UserSettings, adds more unit tests
 * add pytorch 1.8.1 for huggingface

## v2.59.3.post0 (2021-09-22)

### Documentation Changes

 * Info about offline s3 bucket key when creating feature group

## v2.59.3 (2021-09-20)

## v2.59.2 (2021-09-15)

### Bug Fixes and Other Changes

 * unit tests for KIX and remove regional calls to boto

### Documentation Changes

 * Remove Shortbread

## v2.59.1.post0 (2021-09-13)

### Documentation Changes

 * update experiment config doc on fit method

## v2.59.1 (2021-09-02)

### Bug Fixes and Other Changes

 * pin docker to 5.0.0

## v2.59.0 (2021-09-01)

### Features

 * Add KIX account for SM XGBoost 1.2-2 and 1.3-1

### Bug Fixes and Other Changes

 * revert #2572 and address #2611

## v2.58.0 (2021-08-31)

### Features

 * update debugger for KIX
 * support displayName and description for pipeline steps

### Bug Fixes and Other Changes

 * localmode subprocess parent process not sending SIGTERM to child

## v2.57.0 (2021-08-30)

### Deprecations and Removals

 * Remove stale S3DownloadMode from test_session.py

### Features

 * update clarify imageURI for KIX

### Bug Fixes and Other Changes

 * propagate KMS key to model.deploy
 * Propagate tags and VPC configs to repack model steps

## v2.56.0 (2021-08-26)

### Features

 * Add NEO KIX Configuration
 * Algorithms region launch on KIX

### Bug Fixes and Other Changes

 * remove dots from CHANGELOG

## v2.55.0 (2021-08-25)

### Features

 * Add information of Amazon-provided analysis image used by Model Monitor

### Bug Fixes and Other Changes

 * Update Changelog to fix release
 * Fixing the order of populating container list
 * pass network isolation config to pipelineModel
 * Deference symbolic link when create tar file
 * multiprocess issue in feature_group.py
 * deprecate tag logic on Association

### Documentation Changes

 * add dataset_definition to processing page

## v2.54.0 (2021-08-16)

### Features

 * add pytorch 1.5.1 eia configuration

### Bug Fixes and Other Changes

 * issue #2253 where Processing job in Local mode would call Describe API

## v2.53.0 (2021-08-12)

### Features

 * support tuning step parameter range parameterization + support retry strategy in tuner

## v2.52.2.post0 (2021-08-11)

### Documentation Changes

 * clarify that default_bucket creates a bucket
 * Minor updates to Clarify API documentation

## v2.52.2 (2021-08-10)

### Bug Fixes and Other Changes

 * sklearn integ tests, remove swallowing exception on feature group delete attempt
 * sklearn integ test for custom bucket

### Documentation Changes

 * Fix dataset_definition links
 * Document LambdaModel and LambdaPredictor classes

## v2.52.1 (2021-08-06)

### Bug Fixes and Other Changes

 * revert #2251 changes for sklearn processor

## v2.52.0 (2021-08-05)

### Features

 * processors that support multiple Python files, requirements.txt, and dependencies.
 * support step object in step depends on list

### Bug Fixes and Other Changes

 * enable isolation while creating model from job
 * update `sagemaker.serverless` integration test
 * Use correct boto model name for RegisterModelStep properties

## v2.51.0 (2021-08-03)

### Features

 * add LambdaStep support for SageMaker Pipelines
 * support JsonGet for all step types

## v2.50.1 (2021-08-02)

### Bug Fixes and Other Changes

 * null checks for uploaded_code and entry_point

### Documentation Changes

 * update sagemaker.estimator.EstimatorBase
 * Mark baseline as optional in KernelSHAP.

## v2.50.0 (2021-07-28)

### Features

 * add KIX region to image_uris

### Bug Fixes and Other Changes

 * Rename `PredictorBase.delete_endpoint` as `PredictorBase.delete_predictor`
 * incorrect default argument for callback output parameter

### Documentation Changes

 * Remove years from copyright boilerplate
 * Fix documentation formatting for PySpark and SparkJar processors

### Testing and Release Infrastructure

 * enable py38 tox env

## v2.49.2 (2021-07-21)

### Bug Fixes and Other Changes

 * order of populating container list
 * upgrade Adobe Analytics cookie to 3.0

## v2.49.1 (2021-07-19)

### Bug Fixes and Other Changes

 * Set flag when debugger is disabled
 * KMS Key fix for kwargs
 * Update BiasConfig to accept multiple facet params

### Documentation Changes

 * Update huggingface estimator documentation

## v2.49.0 (2021-07-15)

### Features

 * Adding serial inference pipeline support to RegisterModel Step

### Documentation Changes

 * add tuning step get_top_model_s3_uri and callback step to doc
 * links for HF in sdk
 * Add Clarify module to Model Monitoring API docs

## v2.48.2 (2021-07-12)

### Bug Fixes and Other Changes

 * default time for compilation jobs
 * skip hf inference test

## v2.48.1 (2021-07-08)

### Bug Fixes and Other Changes

 * skip HF inference test
 * remove upsert from test_workflow

### Documentation Changes

 * Add Hugging Face docs
 * add tuning step to doc

## v2.48.0 (2021-07-07)

### Features

 * HuggingFace Inference

### Bug Fixes and Other Changes

 * add support for SageMaker workflow tuning step

## v2.47.2.post0 (2021-07-01)

### Documentation Changes

 * smddp 1.2.1 release note / convert md to rst
 * add smd model parallel 1.4.0 release note / restructure doc files

## v2.47.2 (2021-06-30)

### Bug Fixes and Other Changes

 * handle tags when upsert pipeine

## v2.47.1 (2021-06-27)

### Bug Fixes and Other Changes

 * revert "fix: jsonGet interpolation issue 2426 + allow step depends on pass in step instance (#2477)"

## v2.47.0 (2021-06-25)

### Features

 * support job_name_prefix for Clarify

### Bug Fixes and Other Changes

 * Add configuration option with headers for Clarify Explainability
 * jsonGet interpolation issue 2426 + allow step depends on pass in step instance
 * add default retries to feature group ingestion.
 * Update using_pytorch.rst
 * kms key does not propapate in register model step
 * Correctly interpolate Callback output parameters

## v2.46.1 (2021-06-22)

### Bug Fixes and Other Changes

 * Register model step tags

### Documentation Changes

 * update to include new batch_get_record api call
 * Correct type annotation for TrainingStep inputs
 * introduce input mode FastFile
 * update hf transformer version

## v2.46.0 (2021-06-15)

### Features

 * Add HF transformer version 4.6.1

### Bug Fixes and Other Changes

 * encode localmode payload to UTF-8
 * call DescribeDomain as fallback in get_execution_role
 * parameterize PT and TF version for HuggingFace tests

### Documentation Changes

 * Add import statement in Batch Transform Overview doc

## v2.45.0 (2021-06-07)

### Features

 * Add support for Callback steps in model building pipelines

## v2.44.0 (2021-06-01)

### Features

 * support endpoint_name_prefix, seed and version for Clarify

## v2.43.0 (2021-05-31)

### Features

 * add xgboost framework version 1.3-1

### Bug Fixes and Other Changes

 * remove duplicated tags in _append_project_tags

## v2.42.1 (2021-05-27)

### Bug Fixes and Other Changes

 * default value removed if zero for integer param

## v2.42.0 (2021-05-24)

### Features

 * support for custom pipeline execution name
 * Add data ingestion only data-wrangler flow recipe generation helper function

### Bug Fixes and Other Changes

 * add kms key for processing job code upload
 * remove failing notebooks from notebook pr test
 * fix in and not in condition bug
 * Update overview.rst

### Documentation Changes

 * Update "Ask a question" contact link
 * Update smdp docs with sparse_as_dense support

## v2.41.0 (2021-05-17)

### Features

 * add pipeline experiment config
 * add data wrangler processor
 * support RetryStrategy for training jobs

### Bug Fixes and Other Changes

 * fix repack pipeline step by putting inference.py in "code" sub dir
 * add data wrangler image uri
 * fix black-check errors

## v2.40.0 (2021-05-11)

### Features

 * add xgboost framework version 1.2-2

### Bug Fixes and Other Changes

 * fix get_execution_role on Studio
 * [fix] Check py_version existence in RegisterModel step

### Documentation Changes

 * SM Distributed EFA Launch

## v2.39.1 (2021-05-05)

### Bug Fixes and Other Changes

 * RegisterModel step and custom dependency support

### Documentation Changes

 * reverting SageMaker distributed data parallel EFA doc updates
 * adding new version, SM dist. data parallel 1.2.0.
 * add current Hugging Face supported versions
 * SMDDP 1.2.0 release notes

## v2.39.0.post0 (2021-05-04)

### Testing and Release Infrastructure

 * disable smdataparallel tests

## v2.39.0 (2021-04-28)

### Features

 * Add HF transformer version 4.5.0

### Bug Fixes and Other Changes

 * Allow hyperparameters in Tensorflow estimator to be parameterized

### Testing and Release Infrastructure

 * black format unit tests

## v2.38.0 (2021-04-21)

### Features

 * support multiprocess feature group ingest (#2111)

## v2.37.0 (2021-04-20)

### Features

 * add experiment_config for clarify processing job

### Documentation Changes

 * release notes for smdistributed.dataparallel v1.1.2

## v2.36.0 (2021-04-19)

### Features

 * enable smdataparallel custom mpi options support

## v2.35.0 (2021-04-14)

### Features

 * add support for PyTorch 1.8.1

### Bug Fixes and Other Changes

 * boto3 client param updated for feature store
 * Updated release notes and API doc for smd model parallel 1.3.1

## v2.34.0 (2021-04-12)

### Features

 * Add support for accelerator in Clarify

### Bug Fixes and Other Changes

 * add Documentation for how to use
 * enable local mode tests that were skipped
 * add integ test for HuggingFace with TensorFlow

### Documentation Changes

 * release notes for smdistributed.dataparallel v1.1.1
 * fixing the SageMaker distributed version references

### Testing and Release Infrastructure

 * pin version for ducutils

## v2.33.0 (2021-04-05)

### Features

 * Add environment variable support for SageMaker training job

### Bug Fixes and Other Changes

 * add version length mismatch validation for HuggingFace
 * Disable debugger when checkpointing is enabled with distributed training
 * map user context is list associations response

### Testing and Release Infrastructure

 * disable_profiler on mx-horovod test

## v2.32.1 (2021-04-01)

### Bug Fixes and Other Changes

 * disable profiler in some release tests
 * remove outdated notebook from test
 * add compilation option for ml_eia2
 * add short version to smdataparallel supported list

### Documentation Changes

 * creating a "latest" version sm distributed docs
 * add docs for Sagemaker Model Parallel 1.3, released with PT 1.8
 * update PyTorch version in doc

## v2.32.0 (2021-03-26)

### Features

 * upgrade neo mxnet to 1.8
 * Enable Profiler in China Regions

### Bug Fixes and Other Changes

 * use workflow parameters in training hyperparameters (#2114) (#2115)
 * skip HuggingFace tests in regions without p2 instances

### Documentation Changes

 * add Feature Store methods docs

## v2.31.1 (2021-03-23)

### Bug Fixes and Other Changes

 * added documentation for Hugging Face Estimator
 * mark HuggingFace tests as release tests

### Documentation Changes

 * adding version 1.1.0 docs for smdistributed.dataparallel

## v2.31.0 (2021-03-23)

### Features

 * add HuggingFace framework estimator
 * update TF framework version support
 * Support all processor types in ProcessingStep

### Bug Fixes and Other Changes

 * Add pipelines functions.

## v2.30.0 (2021-03-17)

### Features

 * add support for PyTorch 1.8.0
 * Allow users to send custom attributes to the model endpoint

### Bug Fixes and Other Changes

 * use ResolvedOutputS3Uir for Hive DDL LOCATION
 * Do lazy initialization in predictor

## v2.29.2 (2021-03-11)

### Bug Fixes and Other Changes

 * move pandas to required dependency from specific use cases

## v2.29.1 (2021-03-09)

### Bug Fixes and Other Changes

 * return all failed row indices in feature_group.ingest
 * move service-role path parsing for AmazonSageMaker-ExecutionRole for get_execution_role() into except block of IAM get_role() call and add warning message
 * add description parameter for RegisterModelStep
 * add type annotations for Lineage

### Documentation Changes

 * remove ellipsis from CHANGELOG.md

## v2.29.0 (2021-03-04)

### Features

 * add support for TensorFlow 2.4.1 for training, inference and data parallel
 * Support profiler config in the pipeline training job step
 * support PyTorch 1.7.1 training, inference and data parallel

## v2.28.0 (2021-03-03)

### Features

 * support creating endpoints with model images from private registries

## v2.27.1 (2021-03-03)

### Bug Fixes and Other Changes

 * Change Estimator.logs() to use latest_training_job.name
 * mask creds from docker commands in local mode. Closes #2118

### Documentation Changes

 * fix pipelines processing step typo
 * remove double 'enable-network-isolation' description

## v2.27.0 (2021-03-01)

### Features

 * add inference_id to predict

### Bug Fixes and Other Changes

 * disable profiler by default for regions not support it

### Documentation Changes

 * add TF 2.4.1 support to sm distributed data parallel docs and other updates

## v2.26.0 (2021-02-26)

### Features

 * Add Framework Version support for PyTorch compilation (Neo)

### Bug Fixes and Other Changes

 * add mxnet 1.7.0 eia configuration
 * update source constructor for lineage action and artifact

### Documentation Changes

 * fix typo in create_monitoring_schedule method

## v2.25.2 (2021-02-25)

### Bug Fixes and Other Changes

 * Use the output path to store the Clarify config file
 * feature group should ignore nan values
 * ignore failing smdataparallel test
 * Add tests for Training job & Transform job in visualizer
 * visualizer for pipeline processing job steps

### Documentation Changes

 * update doc for Elastic Inference MXNet 1.7.0

## v2.25.1 (2021-02-20)

### Bug Fixes and Other Changes

 * Add tests for visualizer to improve test coverage

### Documentation Changes

 * specify correct return type

### Testing and Release Infrastructure

 * rename canary_quick pytest mark to release

## v2.25.0 (2021-02-19)

### Features

 * Enable step caching
 * Add other Neo supported regions for Inferentia inference images

### Bug Fixes and Other Changes

 * remove FailStep from pipelines
 * use sagemaker_session in workflow tests
 * use ECR public for multidatamodel tests
 * add the mapping from py3 to cuda11 images
 * Add 30s cap time for tag tests
 * add build spec for slow tests
 * mark top 10 slow tests
 * remove slow test_run_xxx_monitor_baseline tests
 * pin astroid to 2.4.2

### Testing and Release Infrastructure

 * unmark more flaky integ tests
 * remove canary_quick pytest mark from flaky/unnecessary tests
 * remove python3.8 from buildspec
 * remove py38 tox env
 * fix release buildspec typo
 * unblock regional release builds
 * lower test TPS for experiment analytics
 * move package preparation and publishing to the deploy step

## v2.24.5 (2021-02-12)

### Bug Fixes and Other Changes

 * test_tag/test_tags method assert fix in association tests

### Documentation Changes

 * removing mention of TF 2.4 from SM distributed model parallel docs
 * adding details about mpi options, other small updates

## v2.24.4 (2021-02-09)

### Bug Fixes and Other Changes

 * add integration test for listing artifacts by type
 * List Associations integ tests

## v2.24.3 (2021-02-04)

### Bug Fixes and Other Changes

 * Remove pytest fixture and fix test_tag/s method

## v2.24.2 (2021-02-03)

### Bug Fixes and Other Changes

 * use 3.5 version of get-pip.py
 * SM DDP release notes/changelog files

### Documentation Changes

 * adding versioning to sm distributed data parallel docs

## v2.24.1 (2021-01-28)

### Bug Fixes and Other Changes

 * fix collect-tests tox env
 * create profiler specific unsupported regions
 * Update smd_model_parallel_pytorch.rst

## v2.24.0 (2021-01-22)

### Features

 * add support for Std:Join for pipelines
 * Map image name to image uri
 * friendly names for short URIs

### Bug Fixes and Other Changes

 * increase allowed time for search to get updated
 * refactor distribution config construction

### Documentation Changes

 * Add SMP 1.2.0 API docs

## v2.23.6 (2021-01-20)

### Bug Fixes and Other Changes

 * add artifact, action, context to virsualizer

## v2.23.5 (2021-01-18)

### Bug Fixes and Other Changes

 * increase time allowed for trial components to index

## v2.23.4.post0 (2021-01-14)

### Documentation Changes

 * update predict_fn implementation for PyTorch EIA 1.5.1

## v2.23.4 (2021-01-13)

### Bug Fixes and Other Changes

 * remove captureWarninig setting

## v2.23.3 (2021-01-12)

### Bug Fixes and Other Changes

 * improve optional dependency error message
 * add debugger rule container account in PDT
 * assert step execution first in pipeline test
 * add service inserted fields to generated Hive DDL

### Documentation Changes

 * fix description for max_wait
 * use correct classpath in V2 alias documentation.
 * Bad arg name in feat-store ingestion manager

## v2.23.2 (2021-01-06)

### Bug Fixes and Other Changes

 * remove shell=True in subprocess.check_output
 * use SecurityConfig dict key

### Documentation Changes

 * remove D212 from ignore to comply with PEP257 standards

## v2.23.1 (2020-12-29)

### Bug Fixes and Other Changes

 * update git utils temp file
 * Allow online store only FeatureGroups

### Documentation Changes

 * inform contributors when not to mark integration tests as canaries
 * adding change log for smd model parallel

## v2.23.0 (2020-12-23)

### Features

 * Add support for actions in debugger rules.

### Bug Fixes and Other Changes

 * include sparkml 2.4 in image uri config properly
 * Mount metadata dir only if it exists
 * allow urllib3 1.26

## v2.22.0 (2020-12-22)

### Features

 * Support local mode for Amazon SageMaker Processing jobs

### Bug Fixes and Other Changes

 * Add API enhancements for SMP
 * adjust naming convention; fix links
 * lower value used in featurestore test

### Documentation Changes

 * Update GTDD instructions

## v2.21.0 (2020-12-21)

### Features

 * remove D205 to enable PEP257 Docstring Conventions

### Bug Fixes and Other Changes

 * Pin smdebug-rulesconfig to 1.0.0
 * use itertuples to ingest pandas dataframe to FeatureStore

## v2.20.0 (2020-12-16)

### Features

 * add dataset definition support for processing jobs

### Bug Fixes and Other Changes

 * include workflow integ tests with clarify and debugger enabled
 * only run DataParallel and EdgePackaging tests in supported regions

### Documentation Changes

 * fix smp code example, add note for CUDA 11 to sdp
 * adding note about CUDA 11 to SMP. Small title update PyTorch

## v2.19.0 (2020-12-08)

### Features

 * add tensorflow 1.15.4 and 2.3.1 as valid versions
 * add py36 as valid python version for pytorch 1.6.0
 * auto-select container version for p4d and smdistributed
 * add edge packaging job support
 * Add Clarify Processor, Model Bias, Explainability, and Quality Monitors support. (#494)
 * add model parallelism support
 * add data parallelism support (#454) (#511)
 * support creating and updating profiler in training job (#444) (#526)

### Bug Fixes and Other Changes

 * bump boto3 and smdebug_rulesconfig versions for reinvent and enable data parallel integ tests
 * run UpdateTrainingJob tests only during allowed secondary status
 * Remove workarounds and apply fixes to Clarify and MM integ tests
 * add p4d to smdataparallel supported instances
 * Mount metadata directory when starting local mode docker container
 * add integ test for profiler
 * Re-enable model monitor integration tests.

### Documentation Changes

 * add SageMaker distributed libraries documentation
 * update documentation for the new SageMaker Debugger APIs
 * minor updates to doc strings

## v2.18.0 (2020-12-03)

### Features

 * all de/serializers support content type
 * warn on 'Stopped' (non-Completed) jobs
 * all predictors support serializer/deserializer overrides

### Bug Fixes and Other Changes

 * v2 upgrade tool should ignore cell starting with '%'
 * use iterrows to iterate pandas dataframe
 * check for distributions in TF estimator

### Documentation Changes

 * Update link to Sagemaker PyTorch Docker Containers
 * create artifact restricted to SM context note

### Testing and Release Infrastructure

 * remove flaky assertion in test_integ_history_server
 * adjust assertion of TensorFlow MNIST test

## v2.17.0 (2020-12-02)

### Features

 * bump minor version for re:Invent 2020 features

## v2.16.4 (2020-12-01)

### Features

 * Add re:Invent 2020 features

### Bug Fixes and Other Changes

 * use eia python version fixture in integration tests
 * bump version to 2.17.0 for re:Invent-2020

### Documentation Changes

 * add feature store documentation

## v2.16.3.post0 (2020-11-17)

### Testing and Release Infrastructure

 * use ECR-hosted image for ubuntu:16.04

## v2.16.3 (2020-11-11)

### Bug Fixes and Other Changes

 * fix failures for multiple spark run() invocations

## v2.16.2 (2020-11-09)

### Bug Fixes and Other Changes

 * create default bucket only if needed

## v2.16.1 (2020-10-28)

### Bug Fixes and Other Changes

 * ensure 1p algos are compatible with forward-port

## v2.16.0.post0 (2020-10-28)

### Documentation Changes

 * clarify non-breaking changes after v1 forward port

## v2.16.0 (2020-10-27)

### Features

 * update image uri for neo tensorflow

## v2.15.4 (2020-10-26)

### Bug Fixes and Other Changes

 * add kms_key optional arg to Pipeline.deploy()

### Documentation Changes

 * Debugger API - improve docstrings and add examples

## v2.15.3 (2020-10-20)

### Bug Fixes and Other Changes

 * refactor _create_model_request

## v2.15.2 (2020-10-19)

### Bug Fixes and Other Changes

 * preserve model_dir bool value
 * refactor out batch transform job input generation

## v2.15.1 (2020-10-15)

### Bug Fixes and Other Changes

 * include more notebook tests, logger to warn
 * include managed spot training notebook test
 * add missing account IDs for af-south-1 and eu-south-1

## v2.15.0 (2020-10-07)

### Features

 * add network isolation support for PipelineModel
 * forward-port v1 names as deprecated aliases

### Bug Fixes and Other Changes

 * include additional docstyle improvements
 * check optional keyword before accessing
 * use local updated args; use train_max_wait
 * cross-platform file URI for Processing
 * update kwargs target attribute

### Documentation Changes

 * fix Spark class links
 * kwargs descriptions include clickable links
 * fix broken link to moved notebook

## v2.14.0 (2020-10-05)

### Features

 * upgrade Neo MxNet to 1.7

### Bug Fixes and Other Changes

 * add a condition to retrieve correct image URI for xgboost

## v2.13.0 (2020-09-30)

### Features

 * add xgboost framework version 1.2-1

### Bug Fixes and Other Changes

 * revert "feature: upgrade Neo MxNet to 1.7 (#1928)"

## v2.12.0 (2020-09-29)

### Features

 * upgrade Neo MxNet to 1.7

## v2.11.0 (2020-09-28)

### Features

 * Add SDK support for SparkML Serving Container version 2.4

### Bug Fixes and Other Changes

 * pin pytest version <6.1.0 to avoid pytest-rerunfailures breaking changes
 * temporarily skip the MxNet Neo test until we fix them

### Documentation Changes

 * fix conda setup for docs

## v2.10.0 (2020-09-23)

### Features

 * add inferentia pytorch inference container config

## v2.9.2 (2020-09-21)

### Bug Fixes and Other Changes

 * allow kms encryption upload for processing

## v2.9.1 (2020-09-17)

### Bug Fixes and Other Changes

 * update spark image_uri config with eu-north-1 account

## v2.9.0 (2020-09-17)

### Features

 * add MXNet 1.7.0 images

### Documentation Changes

 * removed Kubernetes workflow content

## v2.8.0 (2020-09-16)

### Features

 * add spark processing support to processing jobs

### Bug Fixes and Other Changes

 * remove DataFrame assert from unrelated test

## v2.7.0 (2020-09-15)

### Features

 * reshape Parents into experiment analytics dataframe

## v2.6.0 (2020-09-14)

### Features

 * add model monitor image accounts for af-south-1 and eu-south-1

### Bug Fixes and Other Changes

 * enforce some docstyle conventions

### Documentation Changes

 * fix CSVSerializer typo in v2.rst

## v2.5.5 (2020-09-10)

### Bug Fixes and Other Changes

 * update PyTorch 1.6.0 inference image uri config
 * set use_spot_instances and max_wait as init params from job description
 * run integ tests when image_uri_config jsons are changed
 * Revert "fix: update pytorch inference 1.6 image uri config (#1873)"
 * update pytorch inference 1.6 image uri config

### Documentation Changes

 * fix typo in v2.rst

### Testing and Release Infrastructure

 * fix PyTorch inference packed model integ test

## v2.5.4 (2020-09-08)

### Bug Fixes and Other Changes

 * update max_run_wait to max_wait in v2.rst for estimator parameters
 * Updating regional account ids for af-south-1 and eu-south-1
 * add account ids for af-south-1 and eu-south-1 for debugger rules

## v2.5.3 (2020-09-02)

### Bug Fixes and Other Changes

 * Revert "change: update image uri config for pytorch 1.6.0 inference (#1864)"
 * update image uri config for pytorch 1.6.0 inference
 * add missing framework version image uri config

## v2.5.2 (2020-08-31)

### Bug Fixes and Other Changes

 * refactor normalization of args for processing
 * set TF 2.1.1 as highest py2 version for TF
 * decrease integ test concurrency and increase delay between retries

## v2.5.1 (2020-08-27)

### Bug Fixes and Other Changes

 * formatting changes from updates to black

## v2.5.0 (2020-08-25)

### Features

 * add mypy tox target

### Bug Fixes and Other Changes

 * break out methods to get processing arguments
 * break out methods to get train arguments

## v2.4.2 (2020-08-24)

### Bug Fixes and Other Changes

 * check ast node on later renamers for cli v2 updater

### Documentation Changes

 * Clarify removals in v2

## v2.4.1 (2020-08-19)

### Bug Fixes and Other Changes

 * update rulesconfig to 0.1.5

## v2.4.0 (2020-08-17)

### Features

 * Neo algorithm accounts for af-south-1 and eu-south-1

### Bug Fixes and Other Changes

 * upgrade pytest and other deps, tox clean-up
 * upgrade airflow to 1.10.11
 * update exception assertion with new api change
 * docs: Add SerDe documentation

## v2.3.0 (2020-08-11)

### Features

 * support TF training 2.3

### Documentation Changes

 * update 1p estimators class description

## v2.2.0 (2020-08-10)

### Features

 * new 1P algorithm accounts for af-south-1 and eu-south-1

### Bug Fixes and Other Changes

 * update debugger us-east-1 account
 * docs: Add information on Amazon SageMaker Operators usage in China

## v2.1.0 (2020-08-06)

### Features

 * add DLC account numbers for af-south-1 and eu-south-1

## v2.0.1 (2020-08-05)

### Bug Fixes and Other Changes

 * use pathlib.PurePosixPath for S3 URLs and Unix paths
 * fix regions for updated RL images

### Documentation Changes

 * update CHANGELOG to reflect v2.0.0 changes

### Testing and Release Infrastructure

 * remove v2-incompatible notebooks from notebook build

## v2.0.0 (2020-08-04)

### Breaking Changes

 * rename s3_input to TrainingInput
 * Move _NumpyDeserializer to sagemaker.deserializers.NumpyDeserializer
 * rename numpy_to_record_serializer to RecordSerializer
 * Move _CsvDeserializer to sagemaker.deserializers and rename to CSVDeserializer
 * Move _JsonSerializer to sagemaker.serializers.JSONSerializer
 * Move _NPYSerializer to sagemaker.serializers and rename to NumpySerializer
 * Move _JsonDeserializer to sagemaker.deserializers.JSONDeserializer
 * Move _CsvSerializer to sagemaker.serializers.CSVSerializer
 * preserve script path when S3 source_dir is provided
 * use image_uris.retrieve() for XGBoost URIs
 * deprecate sagemaker.amazon.amazon_estimator.get_image_uri()
 * deprecate fw_registry module and use image_uris.retrieve() for SparkML
 * deprecate Python SDK CLI
 * Remove the content_types module
 * deprecate unused parameters
 * deprecate fw_utils.create_image_uri()
 * use images_uris.retrieve() for Debugger
 * deprecate fw_utils.parse_s3_url in favor of s3.parse_s3_url
 * deprecate unused functions from utils and fw_utils
 * Remove content_type and accept parameters from Predictor
 * Add parameters to deploy and remove parameters from create_model
 * Add LibSVM serializer for XGBoost predictor
 * move ShuffleConfig from sagemaker.session to sagemaker.inputs
 * deprecate get_ecr_image_uri_prefix
 * rename estimator.train_image() to estimator.training_image_uri()
 * deprecate is_version_equal_or_higher and is_version_equal_or_lower
 * default wait=True for HyperparameterTuner.fit() and Transformer.transform()
 * remove unused bin/sagemaker-submit file

### Features

 * start new module for retrieving prebuilt SageMaker image URIs
 * handle separate training/inference images and EI in image_uris.retrieve
 * add support for Amazon algorithms in image_uris.retrieve()
 * Add pandas deserializer
 * Remove LegacySerializer and LegacyDeserializer
 * Add sparse matrix serializer
 * Add v2 SerDe compatability
 * Add JSON Lines serializer
 * add framework upgrade tool
 * add 1p algorithm image_uris migration tool
 * Update migration tool to support breaking changes to create_model
 * support PyTorch 1.6 training

### Bug Fixes and Other Changes

 * handle named variables in v2 migration tool
 * add modifier for s3_input class
 * add XGBoost support to image_uris.retrieve()
 * add MXNet configuration to image_uris.retrieve()
 * add remaining Amazon algorithms for image_uris.retrieve()
 * add PyTorch configuration for image_uris.retrieve()
 * make image_scope optional for some images in image_uris.retrieve()
 * separate logs() from attach()
 * use image_uris.retrieve instead of fw_utils.create_image_uri for DLC frameworks
 * use images_uris.retrieve() for scikit-learn classes
 * use image_uris.retrieve() for RL images
 * Rename BaseDeserializer.deserialize data parameter
 * Add allow_pickle parameter to NumpyDeserializer
 * Fix scipy.sparse imports
 * Improve code style of SerDe compatibility
 * use image_uris.retrieve for Neo and Inferentia images
 * use generated RL version fixtures and update Ray version
 * use image_uris.retrieve() for ModelMonitor default image
 * use _framework_name for 'protected' attribute
 * Fix JSONLinesDeserializer
 * upgrade TFS version and fix py_versions KeyError
 * Fix PandasDeserializer tests to more accurately mock response
 * don't require instance_type for image_uris.retrieve() if only one option
 * ignore code cells with shell commands in v2 migration tool
 * Support multiple Accept types

### Documentation Changes

 * fix pip install command
 * document name changes for TFS classes
 * document v2.0.0 changes
 * update KFP full pipeline

### Testing and Release Infrastructure

 * generate Chainer latest version fixtures from config
 * use generated TensorFlow version fixtures
 * use generated MXNet version fixtures

## v1.72.0 (2020-07-29)

### Features

 * Neo: Add Granular Target Description support for compilation

### Documentation Changes

 * Add xgboost doc on bring your own model
 * fix typos on processing docs

## v1.71.1 (2020-07-27)

### Bug Fixes and Other Changes

 * remove redundant information from the user_agent string.

### Testing and Release Infrastructure

 * use unique model name in TFS integ tests
 * use pytest-cov instead of coverage

## v1.71.0 (2020-07-23)

### Features

 * Add mpi support for mxnet estimator api

### Bug Fixes and Other Changes

 * use 'sagemaker' logger instead of root logger
 * account for "py36" and "py37" in image tag parsing

## v1.70.2 (2020-07-22)

### Bug Fixes and Other Changes

 * convert network_config in processing_config to dict

### Documentation Changes

 * Add ECR URI Estimator example

## v1.70.1 (2020-07-21)

### Bug Fixes and Other Changes

 * Nullable fields in processing_config

## v1.70.0 (2020-07-20)

### Features

 * Add model monitor support for us-gov-west-1
 * support TFS 2.2

### Bug Fixes and Other Changes

 * reshape Artifacts into data frame in ExperimentsAnalytics

### Documentation Changes

 * fix MXNet version info for requirements.txt support

## v1.69.0 (2020-07-09)

### Features

 * Add ModelClientConfig Fields for Batch Transform

### Documentation Changes

 * add KFP Processing component

## v2.0.0.rc1 (2020-07-08)

### Breaking Changes

 * Move StreamDeserializer to sagemaker.deserializers
 * Move StringDeserializer to sagemaker.deserializers
 * rename record_deserializer to RecordDeserializer
 * remove "train_" where redundant in parameter/variable names
 * Add BytesDeserializer
 * rename image to image_uri
 * rename image_name to image_uri
 * create new inference resources during model.deploy() and model.transformer()
 * rename session parameter to sagemaker_session in S3 utility classes
 * rename distributions to distribution in TF/MXNet estimators
 * deprecate update_endpoint arg in deploy()
 * create new inference resources during estimator.deploy() or estimator.transformer()
 * deprecate delete_endpoint() for estimators and HyperparameterTuner
 * refactor Predictor attribute endpoint to endpoint_name
 * make instance_type optional for Airflow model configs
 * refactor name of RealTimePredictor to Predictor
 * remove check for Python 2 string in sagemaker.predictor._is_sequence_like()
 * deprecate sagemaker.utils.to_str()
 * drop Python 2 support

### Features

 * add BaseSerializer and BaseDeserializer
 * add Predictor.update_endpoint()

### Bug Fixes and Other Changes

 * handle "train_*" renames in v2 migration tool
 * handle image_uri rename for Session methods in v2 migration tool
 * Update BytesDeserializer accept header
 * handle image_uri rename for estimators and models in v2 migration tool
 * handle image_uri rename in Airflow model config functions in v2 migration tool
 * update migration tool for S3 utility functions
 * set _current_job_name and base_tuning_job_name in HyperparameterTuner.attach()
 * infer base name from job name in estimator.attach()
 * ensure generated names are < 63 characters when deploying compiled models
 * add TF migration documentation to error message

### Documentation Changes

 * update documentation with v2.0.0.rc1 changes
 * remove 'train_*' prefix from estimator parameters
 * update documentation for image_name/image --> image_uri

### Testing and Release Infrastructure

 * refactor matching logic in v2 migration tool
 * add cli modifier for RealTimePredictor and derived classes
 * change coverage settings to reduce intermittent errors
 * clean up pickle.load logic in integ tests
 * use fixture for Python version in framework integ tests
 * remove assumption of Python 2 unit test runs

## v1.68.0 (2020-07-07)

### Features

 * add spot instance support for AlgorithmEstimator

### Documentation Changes

 * add xgboost documentation for inference

## v1.67.1.post0 (2020-07-01)

### Documentation Changes

 * add Step Functions SDK info

## v1.67.1 (2020-06-30)

### Bug Fixes and Other Changes

 * add deprecation warnings for estimator.delete_endpoint() and tuner.delete_endpoint()

## v1.67.0 (2020-06-29)

### Features

 * Apache Airflow integration for SageMaker Processing Jobs

### Bug Fixes and Other Changes

 * fix punctuation in warning message

### Testing and Release Infrastructure

 * address warnings about pytest custom marks, error message checking, and yaml loading
 * mark long-running cron tests
 * fix tox test dependencies and bump coverage threshold to 86%

## v1.66.0 (2020-06-25)

### Features

 * add 3.8 as supported python version

### Testing and Release Infrastructure

 * upgrade airflow to latest stable version
 * update feature request issue template

## v1.65.1.post1 (2020-06-24)

### Testing and Release Infrastructure

 * add py38 to buildspecs

## v1.65.1.post0 (2020-06-22)

### Documentation Changes

 * document that Local Mode + local code doesn't support dependencies arg

### Testing and Release Infrastructure

 * upgrade Sphinx to 3.1.1

## v1.65.1 (2020-06-18)

### Bug Fixes and Other Changes

 * remove include_package_data=True from setup.py

### Documentation Changes

 * add some clarification to Processing docs

### Testing and Release Infrastructure

 * specify what kinds of clients in PR template

## v1.65.0 (2020-06-17)

### Features

 * support for describing hyperparameter tuning job

### Bug Fixes and Other Changes

 * update distributed GPU utilization warning message
 * set logs to False if wait is False in AutoML
 * workflow passing spot training param to training job

## v2.0.0.rc0 (2020-06-17)

### Breaking Changes

 * remove estimator parameters for TF legacy mode
 * remove legacy `TensorFlowModel` and `TensorFlowPredictor` classes
 * force image URI to be passed for legacy TF images
 * rename `sagemaker.tensorflow.serving` to `sagemaker.tensorflow.model`
 * require `framework_version` and `py_version` for framework estimator and model classes
 * change `Model` parameter order to make `model_data` optional

### Bug Fixes and Other Changes

 * add v2 migration tool

### Documentation Changes

 * update TF documentation to reflect breaking changes and how to upgrade
 * start v2 usage and migration documentation

### Testing and Release Infrastructure

 * remove scipy from dependencies
 * remove TF from optional dependencies

## v1.64.1 (2020-06-16)

### Bug Fixes and Other Changes

 * include py38 tox env and some dependency upgrades

## v1.64.0 (2020-06-15)

### Features

 * add support for SKLearn 0.23

## v1.63.0 (2020-06-12)

### Features

 * Allow selecting inference response content for automl generated models
 * Support for multi variant endpoint invocation with target variant param

### Documentation Changes

 * improve docstring and remove unavailable links

## v1.62.0 (2020-06-11)

### Features

 * Support for multi variant endpoint invocation with target variant param

### Bug Fixes and Other Changes

 * Revert "feature: Support for multi variant endpoint invocation with target variant param (#1571)"
 * make instance_type optional for prepare_container_def
 * docs: workflows navigation

### Documentation Changes

 * fix typo in MXNet documentation

## v1.61.0 (2020-06-09)

### Features

 * Use boto3 DEFAULT_SESSION when no boto3 session specified.

### Bug Fixes and Other Changes

 * remove v2 Session warnings
 * upgrade smdebug-rulesconfig to 0.1.4
 * explicitly handle arguments in create_model for sklearn and xgboost

## v1.60.2 (2020-05-29)

### Bug Fixes and Other Changes

 * [doc] Added Amazon Components for Kubeflow Pipelines

## v1.60.1.post0 (2020-05-28)

### Documentation Changes

 * clarify that entry_point must be in the root of source_dir (if applicable)

## v1.60.1 (2020-05-27)

### Bug Fixes and Other Changes

 * refactor the navigation

### Documentation Changes

 * fix undoc directive; removes extra tabs

## v1.60.0.post0 (2020-05-26)

### Documentation Changes

 * remove some duplicated documentation from main README
 * fix TF requirements.txt documentation

## v1.60.0 (2020-05-25)

### Features

 * support TensorFlow training 2.2

### Bug Fixes and Other Changes

 * blacklist unknown xgboost image versions
 * use format strings instead of os.path.join for S3 URI in S3Downloader

### Documentation Changes

 * consolidate framework version and image information

## v1.59.0 (2020-05-21)

### Features

 * MXNet elastic inference support

### Bug Fixes and Other Changes

 * add Batch Transform data processing options to Airflow config
 * add v2 warning messages
 * don't try to use local output path for KMS key in Local Mode

### Documentation Changes

 * add instructions for how to enable 'local code' for Local Mode

## v1.58.4 (2020-05-20)

### Bug Fixes and Other Changes

 * update AutoML default max_candidate value to use the service default
 * add describe_transform_job in session class

### Documentation Changes

 * clarify support for requirements.txt in Tensorflow docs

### Testing and Release Infrastructure

 * wait for DisassociateTrialComponent to take effect in experiment integ test cleanup

## v1.58.3 (2020-05-19)

### Bug Fixes and Other Changes

 * update DatasetFormat key name for sagemakerCaptureJson

### Documentation Changes

 * update Processing job max_runtime_in_seconds docstring

## v1.58.2.post0 (2020-05-18)

### Documentation Changes

 * specify S3 source_dir needs to point to a tar file
 * update PyTorch BYOM topic

## v1.58.2 (2020-05-13)

### Bug Fixes and Other Changes

 * address flake8 error

## v1.58.1 (2020-05-11)

### Bug Fixes and Other Changes

 * upgrade boto3 to 1.13.6

## v1.58.0 (2020-05-08)

### Features

 * support inter container traffic encryption for processing jobs

### Documentation Changes

 * add note that v2.0.0 plans have been posted

## v1.57.0 (2020-05-07)

### Features

 * add tensorflow training 1.15.2 py37 support
 * PyTorch 1.5.0 support

## v1.56.3 (2020-05-06)

### Bug Fixes and Other Changes

 * update xgboost latest image version

## v1.56.2 (2020-05-05)

### Bug Fixes and Other Changes

 * training_config returns MetricDefinitions
 * preserve inference script in model repack.

### Testing and Release Infrastructure

 * support Python 3.7

## v1.56.1.post1 (2020-04-29)

### Documentation Changes

 * document model.tar.gz structure for MXNet and PyTorch
 * add documentation for EstimatorBase parameters missing from docstring

## v1.56.1.post0 (2020-04-28)

### Testing and Release Infrastructure

 * add doc8 check for documentation files

## v1.56.1 (2020-04-27)

### Bug Fixes and Other Changes

 * add super() call in Local Mode DataSource subclasses
 * fix xgboost image incorrect latest version warning
 * allow output_path without trailing slash in Local Mode training jobs
 * allow S3 folder input to contain a trailing slash in Local Mode

### Documentation Changes

 * Add namespace-based setup for SageMaker Operators for Kubernetes
 * Add note about file URLs for Estimator methods in Local Mode

## v1.56.0 (2020-04-24)

### Features

 * add EIA support for TFS 1.15.0 and 2.0.0

### Bug Fixes and Other Changes

 * use format strings intead of os.path.join for Unix paths for Processing Jobs

## v1.55.4 (2020-04-17)

### Bug Fixes and Other Changes

 * use valid encryption key arg for S3 downloads
 * update sagemaker pytorch containers to external link
 * allow specifying model name when creating a Transformer from an Estimator
 * allow specifying model name in create_model() for TensorFlow, SKLearn, and XGBoost
 * allow specifying model name in create_model() for Chainer, MXNet, PyTorch, and RL

### Documentation Changes

 * fix wget endpoints
 * add Adobe Analytics; upgrade Sphinx and docs environment
 * Explain why default model_fn loads PyTorch-EI models to CPU by default
 * Set theme in conf.py
 * correct transform()'s wait default value to "False"

### Testing and Release Infrastructure

 * move unit tests for updating an endpoint to test_deploy.py
 * move Neo unit tests to a new file and directly use the Model class
 * move Model.deploy unit tests to separate file
 * add Model unit tests for delete_model and enable_network_isolation
 * skip integ tests in PR build if only unit tests are modified
 * add Model unit tests for prepare_container_def and _create_sagemaker_model
 * use Model class for model deployment unit tests
 * split model unit tests by Model, FrameworkModel, and ModelPackage
 * add Model unit tests for all transformer() params
 * add TF batch transform integ test with KMS and network isolation
 * use pytest fixtures in batch transform integ tests to train and upload to S3 only once
 * improve unit tests for creating Transformers and transform jobs
 * add PyTorch + custom model bucket batch transform integ test

## v1.55.3 (2020-04-08)

### Bug Fixes and Other Changes

 * remove .strip() from batch transform
 * allow model with network isolation when creating a Transformer from an Estimator
 * add enable_network_isolation to EstimatorBase

## v1.55.2 (2020-04-07)

### Bug Fixes and Other Changes

 * use .format instead of os.path.join for Processing S3 paths.

### Testing and Release Infrastructure

 * use m5.xlarge instances for "ap-northeast-1" region integ tests.

## v1.55.1 (2020-04-06)

### Bug Fixes and Other Changes

 * correct local mode behavior for CN regions

## v1.55.0.post0 (2020-04-06)

### Documentation Changes

 * fix documentation to provide working example.
 * add documentation for XGBoost
 * Correct comment in SKLearn Estimator about default Python version
 * document inferentia supported version
 * Merge Amazon Sagemaker Operators for Kubernetes and Kubernetes Jobs pages

### Testing and Release Infrastructure

 * turn on warnings as errors for docs builds

## v1.55.0 (2020-03-31)

### Features

 * support cn-north-1 and cn-northwest-1

## v1.54.0 (2020-03-31)

### Features

 * inferentia support

## v1.53.0 (2020-03-30)

### Features

 * Allow setting S3 endpoint URL for Local Session

### Bug Fixes and Other Changes

 * Pass kwargs from create_model to Model constructors
 * Warn if parameter server is used with multi-GPU instance

## v1.52.1 (2020-03-26)

### Bug Fixes and Other Changes

 * Fix local _SageMakerContainer detached mode (aws#1374)

## v1.52.0.post0 (2020-03-25)

### Documentation Changes

 * Add docs for debugger job support in operator

## v1.52.0 (2020-03-24)

### Features

 * add us-gov-west-1 to neo supported regions

## v1.51.4 (2020-03-23)

### Bug Fixes and Other Changes

 * Check that session is a LocalSession when using local mode
 * add tflite to Neo-supported frameworks
 * ignore tags with 'aws:' prefix when creating an EndpointConfig based on an existing one
 * allow custom image when calling deploy or create_model with various frameworks

### Documentation Changes

 * fix description of default model_dir for TF
 * add more details about PyTorch eia

## v1.51.3 (2020-03-12)

### Bug Fixes and Other Changes

 * make repack_model only removes py file when new entry_point provided

## v1.51.2 (2020-03-11)

### Bug Fixes and Other Changes

 * handle empty inputs/outputs in ProcessingJob.from_processing_name()
 * use DLC images for GovCloud

### Testing and Release Infrastructure

 * generate test job name at test start instead of module start

## v1.51.1 (2020-03-10)

### Bug Fixes and Other Changes

 * skip pytorch ei test in unsupported regions

### Documentation Changes

 * correct MultiString/MULTI_STRING docstring

## v1.51.0 (2020-03-09)

### Features

 * pytorch 1.3.1 eia support

### Documentation Changes

 * Update Kubernetes Operator default tag
 * improve docstring for tuner.best_estimator()

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
