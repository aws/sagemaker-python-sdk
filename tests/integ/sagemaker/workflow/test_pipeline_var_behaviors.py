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

import pytest

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker import utils
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-vars")


def test_ppl_var_to_string_and_add(sagemaker_session_for_pipeline, role, pipeline_name):
    param_str = ParameterString(name="MyString", default_value="1")
    param_int = ParameterInteger(name="MyInteger", default_value=3)

    cond = ConditionGreaterThan(left=param_str, right=param_int.to_string())
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[],
    )
    join_fn1 = Join(
        on=" ",
        values=[
            "condition greater than check return:",
            step_cond.properties.Outcome.to_string(),
            "and left side param str is",
            param_str,
            "and right side param int is",
            param_int,
        ],
    )

    step_fail = FailStep(
        name="FailStep",
        error_message=join_fn1,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[param_str, param_int],
        steps=[step_cond, step_fail],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        pipeline_arn = response["PipelineArn"]
        execution = pipeline.start()
        response = execution.describe()
        assert response["PipelineArn"] == pipeline_arn

        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()

        assert len(execution_steps) == 2
        for execution_step in execution_steps:
            if execution_step["StepName"] == "CondStep":
                assert execution_step["StepStatus"] == "Succeeded"
                continue
            assert execution_step["StepName"] == "FailStep"
            assert execution_step["StepStatus"] == "Failed"
            assert (
                execution_step["FailureReason"] == "condition greater than check return: false "
                "and left side param str is 1 and right side param int is 3"
            )

        # Update int param to update cond step outcome
        execution = pipeline.start(parameters={"MyInteger": 0})
        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()

        assert len(execution_steps) == 2
        for execution_step in execution_steps:
            if execution_step["StepName"] == "CondStep":
                assert execution_step["StepStatus"] == "Succeeded"
                continue
            assert execution_step["StepName"] == "FailStep"
            assert execution_step["StepStatus"] == "Failed"
            assert (
                execution_step["FailureReason"] == "condition greater than check return: true "
                "and left side param str is 1 and right side param int is 0"
            )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
