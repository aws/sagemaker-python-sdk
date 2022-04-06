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

from botocore.exceptions import ClientError, ValidationError

from mock import Mock

from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import (
    Step,
    ProcessingStep
)
from sagemaker.workflow.steps import StepTypeEnum
from sagemaker.processing import ProcessingInput
from sagemaker.workflow.pipeline_validation import PipelineValidation


class DummyStep(Step):
    def __init__(self, name, input_data, step_type, display_name=None, description=None, inputs=None):
        self.input_data = input_data
        self.step_type = step_type
        self.inputs = inputs

        super(DummyStep, self).__init__(name, display_name, description, step_type, StepTypeEnum.TRAINING)
        path = f"Steps.{name}"
        prop = Properties(path=path)
        prop.__dict__["S3Uri"] = Properties(f"{path}.S3Uri")
        self._properties = prop

    @property
    def arguments(self):
        return {"input_data": self.input_data}

    @property
    def properties(self):
        return self._properties


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock()
    session_mock.default_bucket = Mock(name="default_bucket", return_value="s3_bucket")
    return session_mock


def test_processing_step_inputs_exceeds_limit(sagemaker_session_mock, role_arn):
    parameter = ParameterString("MyStr")
    inputs = []
    for _ in range(11):
        inputs.append(
            ProcessingInput(
                source="",
                destination="processing_manifest"
            )
        )
    # print(inputs)
    steps = DummyStep("DummyStep1", parameter, StepTypeEnum.PROCESSING, inputs=inputs)
    ppv = PipelineValidation([steps])
    with pytest.raises(ValidationError):
        ppv._validate_processing_steps()


