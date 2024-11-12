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

import json

import pytest

from enum import Enum

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
)
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile


from tests.unit.sagemaker.workflow.helpers import CustomStep


class CustomEntity(Entity):
    def __init__(self, foo):
        self.foo = foo

    def to_request(self):
        return {"foo": self.foo}


class CustomEnum(Enum, metaclass=DefaultEnumMeta):
    A = 1
    B = 2


@pytest.fixture
def custom_entity():
    return CustomEntity(1)


@pytest.fixture
def custom_entity_list():
    return [CustomEntity(1), CustomEntity(2)]


def test_entity(custom_entity):
    request_struct = {"foo": 1}
    assert custom_entity.to_request() == request_struct


def test_default_enum_meta():
    assert CustomEnum().value == 1


def test_pipeline_variable_in_pipeline_definition(sagemaker_session):
    param_str = ParameterString(name="MyString", default_value="1")
    param_int = ParameterInteger(name="MyInteger", default_value=3)

    step = CustomStep(name="MyStep")

    property_file = PropertyFile(
        name="name",
        output_name="result",
        path="output",
    )
    json_get_func2 = JsonGet(
        step_name="MyStep",
        property_file=property_file,
        json_path="my-json-path",
    )

    cond = ConditionGreaterThan(left=param_str, right=param_int.to_string())
    step_fail = FailStep(
        name="MyFailStep",
        error_message=Join(
            on=" ",
            values=[
                "Execution failed due to condition check fails, see:",
                json_get_func2.to_string(),
                step.properties.TrainingJobName.to_string(),
                param_int,
            ],
        ),
    )
    step_cond = ConditionStep(
        name="MyCondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[step_fail],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[param_str, param_int],
        steps=[step, step_cond],
        sagemaker_session=sagemaker_session,
    )

    dsl = json.loads(pipeline.definition())
    assert dsl["Parameters"] == [
        {"Name": "MyString", "Type": "String", "DefaultValue": "1"},
        {"Name": "MyInteger", "Type": "Integer", "DefaultValue": 3},
    ]
    assert len(dsl["Steps"]) == 2
    assert dsl["Steps"][1] == {
        "Name": "MyCondStep",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "GreaterThan",
                    "LeftValue": {"Get": "Parameters.MyString"},
                    "RightValue": {
                        "Std:Join": {
                            "On": "",
                            "Values": [{"Get": "Parameters.MyInteger"}],
                        },
                    },
                },
            ],
            "IfSteps": [],
            "ElseSteps": [
                {
                    "Name": "MyFailStep",
                    "Type": "Fail",
                    "Arguments": {
                        "ErrorMessage": {
                            "Std:Join": {
                                "On": " ",
                                "Values": [
                                    "Execution failed due to condition check fails, see:",
                                    {
                                        "Std:Join": {
                                            "On": "",
                                            "Values": [
                                                {
                                                    "Std:JsonGet": {
                                                        "PropertyFile": {
                                                            "Get": "Steps.MyStep.PropertyFiles.name"
                                                        },
                                                        "Path": "my-json-path",
                                                    }
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        "Std:Join": {
                                            "On": "",
                                            "Values": [
                                                {
                                                    "Get": "Steps.MyStep.TrainingJobName",
                                                },
                                            ],
                                        },
                                    },
                                    {"Get": "Parameters.MyInteger"},
                                ],
                            }
                        }
                    },
                }
            ],
        },
    }
