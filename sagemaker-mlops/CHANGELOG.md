# Changelog

## v1.17.0 (2026-07-24)

### New Features

- feat: Wire BatchWriteRecord and ListRecords into ingest_dataframe (#6026)

### Tests

- test(feature-processor): Isolate pipeline names to fix flaky integ tests (#6095)
- test(integ): absorb iam:SimulatePrincipalPolicy throttling across suites (#6081)
- test(integ): let exhausted IAM throttling fail instead of skipping (#6094)

## v1.16.0 (2026-07-15)

### Tests

- test(mlops): Skip non-PEP440 version keys in sklearn_latest_version (#6022)

## v1.15.1 (2026-07-09)

### New Features

- feat: Add granular telemetry signals decorator params and error classification (#5963)

## v1.15.0 (2026-06-22)

### New Features

- feat: IAM role creation — auto-create least-privilege execution roles, SDK-wide (#2041)

### Bug Fixes

- fix(iam): Validate roles by default, opt-in creation, and add MLflow perms (#2080)
- fix: Resolve MLflow app discovery issues (#5924)

## v1.14.0 (2026-06-18)

### Other

- Update module dependencies
- chore: deprecate Python 3.9 support (#5941)

## v1.13.1 (2026-06-04)

- Update module dependencies
  
## v1.13.0 (2026-06-02)

- Update module dependencies

## v1.12.0 (2026-05-19)

### New Features

- Add configurable `use_lake_formation_credentials` parameter to the `@feature_processor` decorator for Lake Formation credential vending (#5816)
- Dynamic Spark image resolution supporting Spark 3.5 with Python 3.12 (#5816)
- Export `IcebergProperties` to the `feature_store` public API surface (#5816)

## v1.11.0 (2026-05-12)

### New Features

- Add Feature Store reference to Implement MLOps documentation page

## v1.10.1 (2026-05-07)

### Bug Fixes

- Fix KMS key propagation in QualityCheckStep and ClarifyCheckStep

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
