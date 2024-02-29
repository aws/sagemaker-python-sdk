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
from botocore.exceptions import WaiterError

from sagemaker import Session
from sagemaker.workflow.pipeline import _PipelineExecution
from sagemaker.workflow.pipeline import Pipeline


def wait_pipeline_execution(execution: _PipelineExecution, delay: int = 30, max_attempts: int = 60):
    try:
        execution.wait(delay=delay, max_attempts=max_attempts)
    except WaiterError:
        pass


def create_and_execute_pipeline(
    pipeline: Pipeline,
    pipeline_name,
    region_name,
    role,
    no_of_steps,
    last_step_name_prefix,
    execution_parameters,
    step_status,
    step_result_type=None,
    step_result_value=None,
    wait_duration=400,  # seconds
    selective_execution_config=None,
):
    create_arn = None
    if not selective_execution_config:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

    execution = pipeline.start(
        parameters=execution_parameters, selective_execution_config=selective_execution_config
    )

    if create_arn:
        response = execution.describe()
        assert response["PipelineArn"] == create_arn

    wait_pipeline_execution(execution=execution, delay=20, max_attempts=int(wait_duration / 20))

    execution_steps = execution.list_steps()

    assert (
        len(execution_steps) == no_of_steps
    ), f"Expected {no_of_steps}, instead found {len(execution_steps)}"

    assert last_step_name_prefix in execution_steps[0]["StepName"]
    assert execution_steps[0]["StepStatus"] == step_status
    if step_result_type:
        result = execution.result(execution_steps[0]["StepName"])
        assert (
            type(result) == step_result_type
        ), f"Expected {step_result_type}, instead found {type(result)}"

    if step_result_value:
        result = execution.result(execution_steps[0]["StepName"])
        assert result == step_result_value, f"Expected {step_result_value}, instead found {result}"

    if selective_execution_config:
        for exe_step in execution_steps:
            if exe_step["StepName"] in selective_execution_config.selected_steps:
                continue
            assert (
                exe_step["SelectiveExecutionResult"]["SourcePipelineExecutionArn"]
                == selective_execution_config.source_pipeline_execution_arn
            )

    return execution, execution_steps


def validate_scheduled_pipeline_execution(
    execution_arn: str,
    pipeline_arn: str,
    no_of_steps: int,
    last_step_name: str,
    status: str,
    session: Session,
):
    _pipeline_execution = _PipelineExecution(
        arn=execution_arn,
        sagemaker_session=session,
    )
    response = _pipeline_execution.describe()
    assert response["PipelineArn"] == pipeline_arn

    wait_pipeline_execution(execution=_pipeline_execution, delay=20, max_attempts=20)

    execution_steps = _pipeline_execution.list_steps()

    assert len(execution_steps) == no_of_steps

    assert last_step_name in execution_steps[0]["StepName"]
    assert execution_steps[0]["StepStatus"] == status
