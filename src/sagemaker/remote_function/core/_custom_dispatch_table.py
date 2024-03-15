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

from sagemaker.remote_function.errors import SerializationError

from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.function_step import DelayedReturn
from sagemaker.workflow.properties import (
    Properties,
    PropertiesMap,
    PropertiesList,
)


def _pipeline_variable_reducer(pipeline_variable):
    """Reducer for pipeline variable."""

    raise SerializationError(
        """Please pass the pipeline variable to the function decorated with @step as an argument.
           Referencing to a pipeline variable from within the function
           or passing a pipeline variable nested in a data structure are not supported."""
    )


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
    DelayedReturn: _pipeline_variable_reducer,
}
