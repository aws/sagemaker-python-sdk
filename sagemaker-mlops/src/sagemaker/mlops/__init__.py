"""SageMaker MLOps package for workflow orchestration and model building.

This package provides high-level orchestration capabilities for SageMaker workflows,
including pipeline definitions, step implementations, and model building utilities.

The MLOps package sits at the top of the dependency hierarchy and can import from:
- sagemaker.core (foundation primitives)
- sagemaker.train (training functionality)
- sagemaker.serve (serving functionality)

Key components:
- workflow: Pipeline and step orchestration
- model_builder: Model building and orchestration

Example usage:
    from sagemaker.mlops import ModelBuilder
    from sagemaker.mlops.workflow import Pipeline, TrainingStep
"""
from __future__ import absolute_import

__version__ = "0.1.0"

# Model building
from sagemaker.serve.model_builder import ModelBuilder

# Workflow submodule is available via:
#   from sagemaker.mlops import workflow
#   from sagemaker.mlops.workflow import Pipeline, TrainingStep, etc.

__all__ = [
    "ModelBuilder",
    "workflow",  # Submodule
]
