# Changelog
## v1.10.0 (2026-05-01)

### New Features

- Add CodeArtifact support for ModelTrainer and FrameworkProcessor requirements.txt installation

### Bug Fixes

- Fix failing train tests for v3

### Other

- Update service-2.json with latest public botocore service model

## v1.9.0 (2026-04-23)

### New Features

- **Train**: Add `wait_timeout` parameter to `train()` for SFT, DPO, RLAIF, RLVR, and BaseTrainer
- **Evaluate**: Add MLflow experiment link to eval output
- **JumpStart**: Allow `SAGEMAKER_HUB_NAME` environment variable to override the `HUB_NAME` constant

### Bug Fixes

- **HyperparameterTuner**: Pass through full `OutputDataConfig` from `ModelTrainer` so `kms_key_id`, `compression_type`, and other fields are preserved
- **HyperparameterTuner / ModelTrainer**: Propagate environment variables that were previously dropped
- **Evaluate**: Skip `None` hyperparameters in `to_dict` instead of converting them to the string `"None"`
- **Nova**: Add `us-west-2` to Nova supported regions

## v1.8.0 (2026-04-16)

- Update module dependencies

## v1.7.1 (2026-03-31)

### Features

- **Telemetry**: Added telemetry emitter to `ScriptProcessor` and `FrameworkProcessor`, enabling SDK usage tracking for processing jobs via the telemetry attribution module (new `PROCESSING` feature enum added to telemetry constants)

### Bug Fixes

- **ModelBuilder**: Fixed `accept_eula` handling in ModelBuilder's LoRA deployment path — previously hardcoded to `True`, now respects the user-provided value and raises a `ValueError` if not explicitly set to `True`
- **Evaluate**: Fixed Lambda handler name derivation in the Evaluator — hardcoded the handler to `lambda_function.lambda_handler` instead of deriving it from the source filename, which caused invocation failures when the source file had a non-default name

## v1.7.0 (2026-03-25)

### Bug fixes and Other Changes

- **Feature**: Add support for AWS Batch Quota Management job submission and job priority update (#5659)
- **Feature**: Extend list_jobs_by_share for quota_share_name (#5669)
- **Feature**: MLflow metrics visualization, enhanced wait UI, and eval job links (#5662)
- **Feature**: Support IAM role for BaseEvaluator (#5671)
- **Fix**: Remove GPT OSS model evaluation restriction (#5658)
- **Tests**: AWS Batch integ test resources are now uniquely named by test run (#5666)

## v1.6.0 (2026-03-19)

### Bug fixes and Other Changes

- **Fix**: Include sm_drivers channel in HyperparameterTuner jobs (#5634)
- **Fix**: resolve PermissionError during local mode cleanup of root-owned Docker files (#5629)
- **Fix**: Add PipelineVariable support to ModelTrainer fields (#5608)
  
## v1.5.0 (2026-03-02)

### Features

- **Feature**: Add support for listing Batch jobs by share identifier (#5585)
- **Feature**: Add stop condition to model customization trainers (#5579)

### Bug fixes and Other Changes

- **Fix**: Remove default for stopping condition for MC trainer (#5586)
- **Fix**: Skip default instance_type/instance_count when instance_groups is set (#5564)
- **Fix**: Bug fixes for Model Customization (#5558)

## v1.4.1 (2026-02-10)

### Bug fixes and Other Changes

- **Fix**: Correct Tag class usage in pipeline creation (#5526)
- **Fix**: HyperparameterTuner now includes ModelTrainer internal channels (#5516)
- **Fix**: Support PipelineVariables in hyperparameters (#5519)

## v1.4.0 (2026-01-22)

### Bug fixes and Other Changes

* Add Nova training support in Model Trainer

## v1.3.1 (2026-01-12)

### Bug fixes and Other Changes
* Removing experiment_config parameter for aws_batch as it is no longer needed with the removal of Estimator

## v1.3.0 (2025-12-19)

### Features
* AWS_Batch: queueing of training jobs with ModelTrainer

## v1.2.0 (2025-12-18)

### Features
* Evaluator handshake with trainer
* Datasets Format validation


## v1.1.1 (2025-12-10)

### Bug fixes and Other Changes
* Add validation to bedrock reward models
* Hyperparameter issue fixes, Add validation s3 output path
* Fix the recipe selection for multiple recipe scenario
* Train wait() timeout exception handling
* Update example notebooks to reflect recent code changes
* Update `model_package_group_name`  param to `model_package_group`  in finetuning interfaces
* remove `dataset` param for benchmark evaluator

## v1.1.0 (2025-12-03)

### Features

* Fine-tuning SDK: SFT, RLVR, and RLAIF techniques with standardized parameter design
* AIRegistry Integration: Added CRUD operations for datasets and evaluators
* Enhanced Training Experience: Implemented MLFlow metrics tracking and deployment workflows

## v1.0.1 (2025-11-19)

* Update project dependencies to include submodules: sagemaker-core, sagemaker-train, sagemaker-serve, sagemaker-mlops

