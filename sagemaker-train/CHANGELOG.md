# Changelog

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

