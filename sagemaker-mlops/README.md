# SageMaker MLOps Package

The `sagemaker-mlops` package provides high-level orchestration capabilities for Amazon SageMaker workflows, including pipeline definitions, step implementations, and model building utilities.

## Purpose

This package sits at the top of the SageMaker SDK dependency hierarchy and orchestrates components from the Core, Train, and Serve packages. It resolves architectural violations by providing a dedicated home for workflow orchestration logic that needs to import from multiple lower-level packages.

## Architecture

The SageMaker SDK follows a clean 4-package architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   Package Architecture                       │
└─────────────────────────────────────────────────────────────┘

                    MLOps (orchestration)
                   ↙      ↓      ↘
              Train     Core     Serve
                   ↘      ↓      ↙
                       Core

Dependency Rules:
✓ Core → nothing (foundation layer)
✓ Train → Core only
✓ Serve → Core only
✓ MLOps → Train, Serve, Core (orchestration layer)
```

### Package Responsibilities

- **sagemaker-core**: Foundation primitives (entities, parameters, functions, conditions, properties)
- **sagemaker-train**: Training functionality (estimators, processors, tuners)
- **sagemaker-serve**: Serving functionality (models, predictors, endpoints)
- **sagemaker-mlops**: Workflow orchestration (pipelines, steps, model building)

## What's in This Package

### Workflow Orchestration (from Core)

The following files were moved from `sagemaker-core/src/sagemaker/core/workflow/` to establish clean architectural boundaries:

**Core Orchestration (2 files):**
- `pipeline.py` - Pipeline class for workflow definition
- `steps.py` - Base Step class and common step logic

**Step Implementations (13 files):**
- `automl_step.py` - AutoML training steps
- `model_step.py` - Model creation and registration steps
- `callback_step.py` - Callback steps for custom logic
- `clarify_check_step.py` - Model bias and explainability checks
- `condition_step.py` - Conditional execution steps
- `emr_step.py` - EMR cluster steps
- `fail_step.py` - Explicit failure steps
- `function_step.py` - Lambda function steps
- `lambda_step.py` - AWS Lambda invocation steps
- `monitor_batch_transform_step.py` - Batch transform monitoring
- `notebook_job_step.py` - Notebook execution steps
- `quality_check_step.py` - Model quality checks
- `step_collections.py` - Step collection utilities
- `step_outputs.py` - Step output handling

**Utilities (6 files):**
- `_utils.py` - Internal utilities
- `_steps_compiler.py` - Step compilation logic
- `_repack_model.py` - Model repackaging utilities
- `_event_bridge_client_helper.py` - EventBridge integration
- `triggers.py` - Pipeline triggers
- `utilities.py` - Public utility functions

**Configuration (6 files):**
- `check_job_config.py` - Quality check configuration
- `parallelism_config.py` - Parallel execution configuration
- `pipeline_definition_config.py` - Pipeline definition settings
- `pipeline_experiment_config.py` - Experiment tracking configuration
- `retry.py` - Retry policies
- `selective_execution_config.py` - Selective execution settings

### Feature Store

The Feature Store module (`sagemaker.mlops.feature_store`) provides comprehensive support for Amazon SageMaker Feature Store operations. This is the V3 equivalent of the V2 `sagemaker.feature_store` module.

**Key Modules:**

- `__init__.py` - Re-exports all Feature Store components from a single entry point
- `feature_definition.py` - Feature definition helpers (FractionalFeatureDefinition, IntegralFeatureDefinition, etc.)
- `feature_utils.py` - Utility functions (ingest_dataframe, create_athena_query, as_hive_ddl, etc.)
- `ingestion_manager_pandas.py` - Multi-threaded DataFrame ingestion manager
- `athena_query.py` - Athena query execution and result retrieval
- `dataset_builder.py` - Dataset building with point-in-time joins across feature groups
- `inputs.py` - Enums for Feature Store operations (TargetStoreEnum, DeletionModeEnum, etc.)
- `feature_processor/` - Feature processor for PySpark-based transformations

**Quick Start:**

```python
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
    create_athena_query,
)

# Create a feature group
feature_defs = load_feature_definitions_from_dataframe(df)
FeatureGroup.create(
    feature_group_name="my-feature-group",
    feature_definitions=feature_defs,
    record_identifier_feature_name="id",
    event_time_feature_name="timestamp",
    role_arn=role,
    online_store_config=OnlineStoreConfig(enable_online_store=True),
)

# Ingest data
ingest_dataframe(feature_group_name="my-feature-group", data_frame=df, max_workers=4)
```

> **Migrating from V2?** See the detailed [Feature Store Migration Guide](src/sagemaker/mlops/feature_store/MIGRATION_GUIDE.md) for V2-to-V3 migration instructions.

### Model Building

ModelBuilder is now located in the `sagemaker-serve` package but is re-exported from MLOps for convenience.

### What Stayed in Core

The following primitive files remain in `sagemaker-core/src/sagemaker/core/workflow/`:

- `entities.py` - Base Entity and PipelineVariable classes
- `parameters.py` - Parameter type definitions
- `functions.py` - Pipeline functions (Join, JsonGet)
- `execution_variables.py` - ExecutionVariable
- `conditions.py` - Condition primitives
- `properties.py` - Property definitions
- `pipeline_context.py` - PipelineSession (refactored to remove Train/Serve imports)
- `__init__.py` - Package initialization

## Installation

Install the package in editable mode for development:

```bash
pip install -e sagemaker-mlops
```

Or install all SageMaker packages together:

```bash
pip install -e sagemaker-core
pip install -e sagemaker-train
pip install -e sagemaker-serve
pip install -e sagemaker-mlops
```