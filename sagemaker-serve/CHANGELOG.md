# Changelog

## v1.17.0 (2026-07-24)

### New Features

- feat(serve): support fine-tuned models in deployment-config API (#6041)

### Bug Fixes

- fix: Fix private hub (#6036)
- fix(serve): support aliased hub content names in private hub deploys (#6039)
- fix(serve): dedicated INFERENCE_RECOMMENDER telemetry feature + type workload param (#6028)
- fix: Fixing EULA check, relying on HostingEulaUri field (#6077)

### Tests

- test: move two tests in serve to gpu-integ-tests (#6096)
- test(integ): absorb iam:SimulatePrincipalPolicy throttling across suites (#6081)
- test(integ): let exhausted IAM throttling fail instead of skipping (#6094)

## v1.16.0 (2026-07-15)

### New Features

- feat(serve): add SageMaker GenAI inference benchmarking and recommendation (#5874)

## v1.15.1 (2026-07-09)

### New Features

- feat: Add granular telemetry signals decorator params and error classification (#5963)

### Bug Fixes

- fix: ModelBuilder resolves private hub artifacts correctly (#5985)
- fix(serve): Invoke pip without shell in xgboost install_package (#5981)

## v1.15.0 (2026-06-22)

### New Features

- feat: Add Nova SMI config bounds validation to ModelBuilder (#2040)
- feat: IAM role creation — auto-create least-privilege execution roles, SDK-wide (#2041)

## v1.14.0 (2026-06-18)

### Bug Fixes

- fix: repair HuggingFace -> JumpStart redirect in ModelBuilder (#5958)

### Other

- chore: deprecate Python 3.9 support (#5941)

## v1.13.1 (2026-06-04)

### Features

- feat: add import job polling and provisioned throughput for Bedrock OSS deployments

## v1.13.0 (2026-06-02)

### Features

- **feat: Deployment** - Add MTRL support for BedrockModelBuilder and ModelBuilder.

### Bug Fixes

- fix: set sagemaker_config=None on mock session in test_from_jumpstart_config_applies_volume_size (#5882)

## v1.12.0 (2026-05-19)

### Bug Fixes

- Fix `AttributeError` on `vpc_config` in networking and telemetry region fallback for classmethods (#5839)
- Prevent code injection in `capture_dependencies` path interpolation via crafted directory names in `ModelBuilder` (#5792)
- Fix `VolumeSizeInGB` not being passed through when deploying JumpStart models with `inference_volume_size` (#5847)

## v1.11.0 (2026-05-12)

### Other

- Update module dependencies

## v1.10.1 (2026-05-07)

### Bug Fixes

- Fix JumpStart network isolation in ModelBuilder
- Fix handling of unrecognized JumpStart container images in ModelBuilder

## v1.10.0 (2026-05-01)

### Bug Fixes

- Fix potential S3 path traversal

## v1.9.0 (2026-04-23)

### Bug Fixes

- **ModelBuilder**: Stop overwriting user-provided `HF_MODEL_ID` for DJL Serving
- **ModelBuilder**: Keep `/opt/ml/model` writable when using `source_code` with DJL LMI

## v1.8.0 (2026-04-16)

### Bug Fixes

- **HuggingFace**: Improve SDK v3 Hugging Face support

## v1.7.1 (2026-03-31)

### Features

- **Telemetry**: Added telemetry emitter to `ScriptProcessor` and `FrameworkProcessor`, enabling SDK usage tracking for processing jobs via the telemetry attribution module (new `PROCESSING` feature enum added to telemetry constants)

### Bug Fixes

- **ModelBuilder**: Fixed `accept_eula` handling in ModelBuilder's LoRA deployment path — previously hardcoded to `True`, now respects the user-provided value and raises a `ValueError` if not explicitly set to `True`
- **Evaluate**: Fixed Lambda handler name derivation in the Evaluator — hardcoded the handler to `lambda_function.lambda_handler` instead of deriving it from the source filename, which caused invocation failures when the source file had a non-default name

## v1.7.0 (2026-03-25)

### Bug fixes and Other Changes

- **Fix**: Sync Nova hosting configs with AGISageMakerInference (#5664)

## v1.6.0 (2026-03-19)

 - **ModelBuilder**: Fix the bug in deploy from LORA finetuning job
   
## v1.5.0 (2026-03-02)

### Bug fixes and Other Changes

- **Fix**: Resolve alt config resolution for jumpstart models (#5563)
- **Fix**: Bug fixes for Model Customization (#5558)
- **Feature**: Add license to sagemaker-serve (#5553)

## v1.4.1 (2026-02-10)

 - Update Module dependencies

## v1.4.0 (2026-01-22)

* Update Module dependencies

## v1.3.1 (2026-01-12)

* Update Module dependencies

## v1.3.0 (2025-12-19)

### Bug Fixes

* Fixes for model registry with ModelBuilder

## v1.2.0 (2025-12-18)

### Features

* Fix pip installation issues

## v1.1.1 (2025-12-10)

### Features

* Bug fixes in Model Builder

## v1.1.0 (2025-12-03)

### Features

* Fine-tuning support in ModelBuilder and adding Bedrock ModelBuilder

## v1.0.1 (2025-11-19)

* Update project dependencies to include submodules: sagemaker-core, sagemaker-train, sagemaker-serve, sagemaker-mlops


