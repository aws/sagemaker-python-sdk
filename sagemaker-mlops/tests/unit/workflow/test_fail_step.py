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
"""Unit tests for workflow fail_step."""
from __future__ import absolute_import

import pytest

from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.steps import StepTypeEnum


def test_fail_step_init():
    fail_step = FailStep(name="fail-step", error_message="Test error")
    assert fail_step.name == "fail-step"
    assert fail_step.error_message == "Test error"
    assert fail_step.step_type == StepTypeEnum.FAIL


def test_fail_step_default_error_message():
    fail_step = FailStep(name="fail-step")
    assert fail_step.error_message == ""


def test_fail_step_arguments():
    fail_step = FailStep(name="fail-step", error_message="Test error")
    args = fail_step.arguments
    assert args == {"ErrorMessage": "Test error"}


def test_fail_step_properties_raises_error():
    fail_step = FailStep(name="fail-step")
    with pytest.raises(RuntimeError, match="terminal step"):
        _ = fail_step.properties
