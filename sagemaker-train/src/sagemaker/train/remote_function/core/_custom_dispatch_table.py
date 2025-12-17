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
"""SageMaker remote function data serializer/deserializer."""
from __future__ import absolute_import

from sagemaker.core.remote_function.errors import SerializationError

from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.core.workflow.execution_variables import ExecutionVariable
from sagemaker.core.workflow.properties import (
    Properties,
    PropertiesMap,
    PropertiesList,
)


# Lazy import to avoid circular dependency
# DelayedReturn is in MLOps package which depends on Core
def _get_delayed_return_class():
    """Lazy import of DelayedReturn to avoid circular dependency."""
    try:
        from sagemaker.mlops.workflow.function_step import DelayedReturn

        return DelayedReturn
    except ImportError:
        # If MLOps is not installed, return None
        return None


def _pipeline_variable_reducer(pipeline_variable):
    """Reducer for pipeline variable."""

    raise SerializationError(
        """Please pass the pipeline variable to the function decorated with @step as an argument.
           Referencing to a pipeline variable from within the function
           or passing a pipeline variable nested in a data structure are not supported."""
    )


# Build dispatch table with lazy loading for DelayedReturn
dispatch_table = {
    ParameterInteger: _pipeline_variable_reducer,
    ParameterFloat: _pipeline_variable_reducer,
    ParameterString: _pipeline_variable_reducer,
    ParameterBoolean: _pipeline_variable_reducer,
    ExecutionVariable: _pipeline_variable_reducer,
    PipelineVariable: _pipeline_variable_reducer,
    Properties: _pipeline_variable_reducer,
    PropertiesMap: _pipeline_variable_reducer,
    PropertiesList: _pipeline_variable_reducer,
}

# Add DelayedReturn to dispatch table if MLOps is available
_delayed_return_class = _get_delayed_return_class()
if _delayed_return_class is not None:
    dispatch_table[_delayed_return_class] = _pipeline_variable_reducer
