# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Workflow primitives for SageMaker pipelines.

This module contains the foundation types and primitives used by workflow orchestration.
These primitives can be used by Train, Serve, and MLOps packages without creating
circular dependencies.

For pipeline and step orchestration classes (Pipeline, TrainingStep, etc.),
import from sagemaker.mlops.workflow instead.
"""
from __future__ import absolute_import

from sagemaker.core.helper.pipeline_variable import PipelineVariable


def is_pipeline_variable(var: object) -> bool:
    """Check if the variable is a pipeline variable

    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """
    return isinstance(var, PipelineVariable)


def is_pipeline_parameter_string(var: object) -> bool:
    """Check if the variable is a pipeline parameter string

    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """
    from sagemaker.core.workflow.parameters import ParameterString

    return isinstance(var, ParameterString)


# Entities
from sagemaker.core.workflow.entities import (
    DefaultEnumMeta,
    Entity,
)

# Execution Variables
from sagemaker.core.workflow.execution_variables import (
    ExecutionVariable,
    ExecutionVariables,
)

# Functions
from sagemaker.core.workflow.functions import (
    Join,
    JsonGet,
)

# Parameters
from sagemaker.core.workflow.parameters import (
    Parameter,
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
    ParameterTypeEnum,
)

# Properties
from sagemaker.core.workflow.properties import (
    Properties,
    PropertiesList,
    PropertiesMap,
    PropertyFile,
)

# Step Outputs (primitive - used by properties)
from sagemaker.core.workflow.step_outputs import (
    StepOutput,
    get_step,
)

# Conditions (primitive)
from sagemaker.core.workflow.conditions import (
    Condition,
    ConditionComparison,
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionIn,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
    ConditionNot,
    ConditionOr,
    ConditionTypeEnum,
)

# NOTE: Orchestration classes (Pipeline, TrainingStep, ProcessingStep, etc.) have moved
# to sagemaker.mlops.workflow. Import them from there instead:
#   from sagemaker.mlops.workflow import Pipeline, TrainingStep, ProcessingStep

__all__ = [
    # Helper functions
    "is_pipeline_variable",
    "is_pipeline_parameter_string",
    # Entities (primitives)
    "DefaultEnumMeta",
    "Entity",
    # Execution Variables (primitives)
    "ExecutionVariable",
    "ExecutionVariables",
    # Functions (primitives)
    "Join",
    "JsonGet",
    # Parameters (primitives)
    "Parameter",
    "ParameterBoolean",
    "ParameterFloat",
    "ParameterInteger",
    "ParameterString",
    "ParameterTypeEnum",
    # Properties (primitives)
    "Properties",
    "PropertiesList",
    "PropertiesMap",
    "PropertyFile",
    # Step Outputs (primitives)
    "StepOutput",
    "get_step",
    # Conditions (primitives)
    "Condition",
    "ConditionComparison",
    "ConditionEquals",
    "ConditionGreaterThan",
    "ConditionGreaterThanOrEqualTo",
    "ConditionIn",
    "ConditionLessThan",
    "ConditionLessThanOrEqualTo",
    "ConditionNot",
    "ConditionOr",
    "ConditionTypeEnum",
]
