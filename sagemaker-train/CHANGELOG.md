# Changelog

## v1.17.0 (2026-07-24)

### Bug Fixes

- fix(train): correct Networking field names in ModelTrainer intelligent defaults (#6064)
- fix: resolve MTRL eval base-model ARN against the configured hub (#6040)
- fix(train): Fall back to public hub when private hub lacks base model (#6092)
- fix: datamixing recipe path fix (#6073)

### Tests

- test: Doc update and added SFT integ test (#6018)
- test: Fix role issue in mtrl integ tests (#6070)
- test: Fix gpu integ test failure due to outdated MPG (#6097)
- test(integ): absorb iam:SimulatePrincipalPolicy throttling across suites (#6081)
- test(integ): let exhausted IAM throttling fail instead of skipping (#6094)

## v1.16.0 (2026-07-15)

### Bug Fixes

- fix: filter full recipe template from serverless train() (#6021)
- Fix sm-train unit tests + use single logger in base trainer (#6030)

## v1.15.1 (2026-07-09)

### New Features

- feat: Add granular telemetry signals decorator params and error classification (#5963)

### Bug Fixes

- fix: always apply evaluator identity keywords and allow explicit domain_id (#5989)
- fix: refresh LLMAsJudgeEvaluator allowed evaluator models (#5987)
- fix: define LAMBDA_ARN_REGEX in finetune_utils to fix RLVRTrainer NameError (#5988)
- fix: RLVR validation bugfix (#6000)
- fix: drop claude-sonnet-4-20250514 from evaluator allowlist (#6009)

### Other

- test: Fix/v3 tests (#5996)
- test: wip nova hyperpod integ tests (#5990)

## v1.15.0 (2026-06-22)

### New Features

- feat(train): Add 3-level recipe override support with `get_resolved_recipe()` (#2034)
- feat: Recipe override handling (#2056)
- feat: Add Nova-specific recipe validations (#2052)
- feat(train): Auto-resolve HyperPod recipe from Hub (#2050)
- feat(train): Add Serverless / SMTJ / HyperPod support to trainers and evaluators (#2045)
- feat: Add infra validation (#2049)
- feat(train): Enable Data Mixing for Nova models (#2054)
- feat: Add `is_multimodal` utils function for multimodal data auto-detection (#2033)
- feat: RLVRTrainer Lambda ARN support (#2025)
- feat: RLVR reward Lambda validation (#2036)
- feat: InspectAI evaluator (#2039)
- feat: Add Nova as a target for LLM-as-a-Judge (LLMAJ) (#2059)
- feat: Support serverful training job checkpoint resolution in InspectAI evaluator (#2066)
- feat: IAM role creation — auto-create least-privilege execution roles, SDK-wide (#2041)
- feat: HyperPod IAM creation (#2057)
- feat(train): Add iterative training with `base_model_name` param (#2085)

### Bug Fixes

- fix: Reject unknown recipe overrides (serverless + serverful) and untrusted IAM roles (#2071)
- fix: Apply recipe overrides to hyperparameters in SMTJ serverful path (#2070)
- fix: Recipe override errors in evaluator (#2067)
- fix: Hub-content IAM perms, recipe dataset paths, and log markup escaping (#2065)
- fix: Harden auto-created IAM roles and protect curated recipe keys (#2058)
- fix: Add HyperPod validation in train and evaluate (#2051)
- fix: Evaluation on HyperPod (#2061)
- fix: Resolve HyperPod training image from EKS payload template (#2069)
- fix: Skip `model_package_group` validation when HyperPod compute is provided in CPTTrainer (#2068)
- fix: Use compute param in get fine-tuning utils (#2060)
- fix: Set Converse as S3DataType for Nova models in SMTJ Serverful for SFT and DPO (#2064)
- fix(train): Use Converse S3DataType for Nova SFT/DPO in serverless flow (#2079)
- fix: MLflow error causing OSS model eval to fail (#2075)
- fix: RLVR setup and reward Lambda handling (#2076)
- fix(iam): Validate roles by default, opt-in creation, and add MLflow perms (#2080)
- fix(train): Validate recipe and instance count override for SMTJ serverful (#2082)
- fix(train): Route `.hyperparameters.*` through recipe resolver (#2078)
- fix: Check S3 permission before saving recipe.yaml (#2083)
- fix: `get_resolved_recipe()` includes all overrides and displays as nested dictionary (#2084)

## v1.14.0 (2026-06-18)

### Bug Fixes

- fix: fall back to SageMakerPublicHub when model not found in private hub (#5959)
- fix: always use SageMakerPublicHub for base model ARN in evaluations (#5959)
- fix: fall back to subscription recipes for models without base recipes (#5946)

### Other

- chore: deprecate Python 3.9 support (#5941)

## v1.13.1 (2026-06-04)

### Bug Fixes

- fix: Address MTRL Eval Hyperparameters issue

## v1.13.0 (2026-06-02)

### New Features

- **feat: Model customization** - Add new finetuning Trainer - MultiTurnRLTrainer(Multi-Turn Reinforcement Learning)
- **feat: Model customization** - Add new evaluator - MultiTurnRLEvaluator

### Bug Fixes

- fix: apply gpu_intensive mark at test-level instead of module-level (#5896)

## v1.12.0 (2026-05-19)

### Other

- Update module dependencies

## v1.11.0 (2026-05-12)

### New Features

- Auto-detect subscription recipe hyperparameters in SFTTrainer for Nova Forge datamix support

## v1.10.1 (2026-05-07)

### Bug Fixes

- Fix base_model_arn construction to use private hub when SAGEMAKER_HUB_NAME is set
- Fix imports for Model Customization interfaces
- Increase default timeout for training jobs

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

