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
"""Unit tests for workflow callback_step."""
from __future__ import absolute_import

from sagemaker.mlops.workflow.callback_step import CallbackStep, CallbackOutput
from sagemaker.mlops.workflow.steps import StepTypeEnum


def test_callback_step_init():
    step = CallbackStep(
        name="callback-step",
        sqs_queue_url="https://sqs.us-west-2.amazonaws.com/123456789012/test-queue",
        inputs={"key": "value"},
        outputs=[]
    )
    assert step.name == "callback-step"
    assert step.step_type == StepTypeEnum.CALLBACK


def test_callback_output_init():
    output = CallbackOutput(output_name="test-output", output_type=str)
    assert output.output_name == "test-output"
    assert output.output_type == str


def test_callback_output_expr():
    output = CallbackOutput(output_name="test-output", output_type=str)
    expr = output.expr("test-step")
    assert "test-step" in str(expr)
    assert "test-output" in str(expr)
