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

## Usage

### Importing Workflow Primitives (from Core)

```python
# Import primitives from Core
from sagemaker.core.workflow import (
    Parameter,
    ParameterString,
    ParameterInteger,
    Join,
    ExecutionVariables,
    Condition,
    ConditionEquals,
)
```

### Importing Workflow Orchestration (from MLOps)

```python
# Import orchestration from MLOps
from sagemaker.mlops.workflow import (
    Pipeline,
    TrainingStep,
    ProcessingStep,
    ModelStep,
    ConditionStep,
)

# Import model building from Serve (re-exported from MLOps for convenience)
from sagemaker.mlops import ModelBuilder
```

### Creating a Pipeline

```python
from sagemaker.core.workflow import ParameterString
from sagemaker.mlops.workflow import Pipeline, TrainingStep
from sagemaker.train import Estimator

# Define parameters (primitives from Core)
input_data = ParameterString(name="InputData", default_value="s3://bucket/data")

# Create estimator (from Train)
estimator = Estimator(
    image_uri="...",
    role="...",
    instance_count=1,
    instance_type="ml.m5.xlarge",
)

# Create training step (orchestration from MLOps)
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": input_data},
)

# Create pipeline (orchestration from MLOps)
pipeline = Pipeline(
    name="MyPipeline",
    parameters=[input_data],
    steps=[train_step],
)

# Execute pipeline
pipeline.upsert(role_arn="...")
pipeline.start()
```

### Using ModelBuilder

```python
from sagemaker.serve import ModelBuilder
from sagemaker.train import ModelTrainer

# Create model trainer (from Train)
trainer = ModelTrainer(
    model_path="s3://bucket/model.tar.gz",
    role="...",
)

# Build model (orchestration from MLOps)
builder = ModelBuilder(model_trainer=trainer)
model = builder.build()
```

## Dependency Hierarchy

This package depends on:

- **sagemaker-core** (>=2.0.0): Foundation primitives
- **sagemaker-train** (>=0.1.0): Training functionality
- **sagemaker-serve** (>=0.1.0): Serving functionality

Lower-level packages should NOT import from this package to maintain clean architecture.

## Architecture Violations Fixed

Creating this package resolved the following architectural violations:

1. **Workflow violations (7)**: Core was importing from Train/Serve through workflow orchestration files
2. **Serve→Train violations (2)**: ModelBuilder in Serve was accepting ModelTrainer objects from Train

After this change:
- Workflow violations: 0 (down from 7)
- Serve→Train violations: 0 (down from 2)
- Clean architecture established: Core ← Train/Serve ← MLOps

## Development

### Running Tests

```bash
pytest sagemaker-mlops/tests/ -v
```

### Code Style

This package follows the same code style as other SageMaker packages:

```bash
black sagemaker-mlops/src/
flake8 sagemaker-mlops/src/
```

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

This package is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
