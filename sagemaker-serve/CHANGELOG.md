# Changelog
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


