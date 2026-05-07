# Changelog
## v1.10.0 (2026-05-01)

### New Features

- Make _PipelineExecution a public class

### Other

- Update service-2.json with latest public botocore service model

## v1.9.0 (2026-04-23)

- Update module dependencies

## v1.8.0 (2026-04-16)

### New Features

- **Feature Group Manager**: Feature Group Manager support

## v1.7.1 (2026-03-31)

### Features

- **Telemetry**: Added telemetry emitter to `ScriptProcessor` and `FrameworkProcessor`, enabling SDK usage tracking for processing jobs via the telemetry attribution module (new `PROCESSING` feature enum added to telemetry constants)

### Bug Fixes

- **ModelBuilder**: Fixed `accept_eula` handling in ModelBuilder's LoRA deployment path — previously hardcoded to `True`, now respects the user-provided value and raises a `ValueError` if not explicitly set to `True`
- **Evaluate**: Fixed Lambda handler name derivation in the Evaluator — hardcoded the handler to `lambda_function.lambda_handler` instead of deriving it from the source filename, which caused invocation failures when the source file had a non-default name

## v1.7.0 (2026-03-25)

- Update module dependencies

## v1.6.0 (2026-03-19)

### Features

- **Feature Processor**: Port feature processor to v3
  
## v1.5.0 (2026-03-02)

- feat: Add Feature Store Support to V3 (#5539)
- feat: EMRStep smart output with `output_args` (#5535)
- chore: Add license to sagemaker-mlops (#5553)

## v1.4.1 (2026-02-10)

- fix: Don't apply default experiment config for pipelines in non-Eureka GA regions (#5500)
- test: Added integration test for pipeline train registry (#5519)

## v1.4.0 (2026-01-22)

* feat: add emr-serverless step for SageMaker Pipelines

## v1.3.1 (2026-01-12)

* sagemaker-mlops bug fix - Correct source code 'dependencies' parameter to 'requirements'
  
## v1.3.0 (2025-12-19)

* Update module dependencies

## v1.2.0 (2025-12-18)

* Update module dependencies

## v1.1.1 (2025-12-10)

* Update project dependencies to include submodules latest versions: sagemaker-core, sagemaker-train, sagemaker-serve, sagemaker-mlops
  
## v1.1.0 (2025-12-03)

* Update project dependencies to include submodules latest versions: sagemaker-core, sagemaker-train, sagemaker-serve, sagemaker-mlops
  
## v1.0.1 (2025-11-19)

* Update project dependencies to include submodules: sagemaker-core, sagemaker-train, sagemaker-serve, sagemaker-mlops
