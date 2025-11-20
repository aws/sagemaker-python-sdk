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
"""Unit tests for workflow lambda_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.lambda_step import LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.steps import StepTypeEnum


def test_lambda_step_init():
    step = LambdaStep(
        name="lambda-step",
        lambda_func=Mock(),
        inputs={"key": "value"}
    )
    assert step.name == "lambda-step"
    assert step.step_type == StepTypeEnum.LAMBDA


def test_lambda_output_init():
    output = LambdaOutput(output_name="test-output", output_type=str)
    assert output.output_name == "test-output"
    assert output.output_type == str


def test_lambda_output_expr():
    output = LambdaOutput(output_name="test-output", output_type=str)
    expr = output.expr("test-step")
    assert "test-step" in str(expr)
    assert "test-output" in str(expr)
