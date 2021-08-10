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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.properties import Properties, PropertyFile


def test_join_primitives_default_on():
    assert Join(values=[1, "a", False, 1.1]).expr == {
        "Std:Join": {
            "On": "",
            "Values": [1, "a", False, 1.1],
        },
    }


def test_join_primitives():
    assert Join(on=",", values=[1, "a", False, 1.1]).expr == {
        "Std:Join": {
            "On": ",",
            "Values": [1, "a", False, 1.1],
        },
    }


def test_join_expressions():
    assert Join(
        values=[
            "foo",
            ParameterFloat(name="MyFloat"),
            ParameterInteger(name="MyInt"),
            ParameterString(name="MyStr"),
            Properties(path="Steps.foo.OutputPath.S3Uri"),
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            Join(on=",", values=[1, "a", False, 1.1]),
        ]
    ).expr == {
        "Std:Join": {
            "On": "",
            "Values": [
                "foo",
                {"Get": "Parameters.MyFloat"},
                {"Get": "Parameters.MyInt"},
                {"Get": "Parameters.MyStr"},
                {"Get": "Steps.foo.OutputPath.S3Uri"},
                {"Get": "Execution.PipelineExecutionId"},
                {"Std:Join": {"On": ",", "Values": [1, "a", False, 1.1]}},
            ],
        },
    }


def test_json_get_expressions():

    assert JsonGet(
        step_name="my-step",
        property_file="my-property-file",
        json_path="my-json-path",
    ).expr == {
        "Std:JsonGet": {
            "PropertyFile": {"Get": "Steps.my-step.PropertyFiles.my-property-file"},
            "Path": "my-json-path",
        },
    }

    property_file = PropertyFile(
        name="name",
        output_name="result",
        path="output",
    )

    assert JsonGet(
        step_name="my-step",
        property_file=property_file,
        json_path="my-json-path",
    ).expr == {
        "Std:JsonGet": {
            "PropertyFile": {"Get": "Steps.my-step.PropertyFiles.name"},
            "Path": "my-json-path",
        },
    }
