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
from __future__ import absolute_import

import re

import pytest

from sagemaker import get_execution_role, utils
from sagemaker.workflow.callback_step import CallbackOutput, CallbackStep, CallbackOutputTypeEnum
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.pipeline import Pipeline


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-callback")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def test_one_step_callback_pipeline(sagemaker_session, role, pipeline_name, region_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    step_callback = CallbackStep(
        name="callback-step",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_callback],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_callback_pipeline_with_output_reference(
    sagemaker_session, role, pipeline_name, region_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    step_callback1 = CallbackStep(
        name="callback-step1",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    step_callback2 = CallbackStep(
        name="callback-step2",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": outputParam1},
        outputs=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_callback1, step_callback2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
