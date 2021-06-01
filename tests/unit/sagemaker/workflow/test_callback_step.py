# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.callback_step import CallbackStep, CallbackOutput, CallbackOutputTypeEnum


def test_callback_step():
    param = ParameterInteger(name="MyInt")
    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    outputParam2 = CallbackOutput(output_name="output2", output_type=CallbackOutputTypeEnum.Boolean)
    cb_step = CallbackStep(
        name="MyCallbackStep",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1, outputParam2],
    )
    cb_step.add_depends_on(["SecondTestStep"])
    assert cb_step.to_request() == {
        "Name": "MyCallbackStep",
        "Type": "Callback",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "SqsQueueUrl": "https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        "OutputParameters": [
            {"OutputName": "output1", "OutputType": "String"},
            {"OutputName": "output2", "OutputType": "Boolean"},
        ],
        "Arguments": {"arg1": "foo", "arg2": 5, "arg3": param},
    }
