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
from sagemaker import get_execution_role, utils
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals, ConditionNot
from sagemaker.workflow.fail_step import FailStep

from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-fail-step")


def test_two_step_fail_pipeline_with_str_err_msg(sagemaker_session, role, pipeline_name):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond_equal = ConditionEquals(left=param, right=2)
    cond_not_equal = ConditionNot(cond_equal)
    step_fail = FailStep(
        name="FailStep",
        error_message="Failed due to hitting in else branch",
    )
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond_not_equal],
        if_steps=[],
        else_steps=[step_fail],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )

    try:
        response = pipeline.create(role)
        pipeline_arn = response["PipelineArn"]
        execution = pipeline.start(parameters={})
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
            assert execution_step["FailureReason"] == "Failed due to hitting in else branch"
            metadata = execution_steps[0]["Metadata"]["Fail"]
            assert metadata["ErrorMessage"] == "Failed due to hitting in else branch"

        # Check FailureReason field in ListPipelineExecutions
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline.name
        )["PipelineExecutionSummaries"]

        assert len(executions) == 1
        assert executions[0]["PipelineExecutionStatus"] == "Failed"
        assert (
            "Step failure: One or multiple steps failed"
            in executions[0]["PipelineExecutionFailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_fail_pipeline_with_parameter_err_msg(sagemaker_session, role, pipeline_name):
    cond_param = ParameterInteger(name="MyInt")
    cond = ConditionEquals(left=cond_param, right=1)
    err_msg_param = ParameterString(name="MyString")
    step_fail = FailStep(
        name="FailStep",
        error_message=err_msg_param,
    )
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[step_fail],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
        parameters=[cond_param, err_msg_param],
    )

    try:
        response = pipeline.create(role)
        pipeline_arn = response["PipelineArn"]
        execution = pipeline.start(
            parameters={
                "MyInt": 3,
                "MyString": "Failed due to hitting in else branch",
            }
        )
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
            assert execution_step["FailureReason"] == "Failed due to hitting in else branch"
            metadata = execution_steps[0]["Metadata"]["Fail"]
            assert metadata["ErrorMessage"] == "Failed due to hitting in else branch"

        # Check FailureReason field in ListPipelineExecutions
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline.name
        )["PipelineExecutionSummaries"]

        assert len(executions) == 1
        assert executions[0]["PipelineExecutionStatus"] == "Failed"
        assert (
            "Step failure: One or multiple steps failed"
            in executions[0]["PipelineExecutionFailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_fail_pipeline_with_join_fn(sagemaker_session, role, pipeline_name):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[],
    )
    step_fail = FailStep(
        name="FailStep",
        error_message=Join(
            on=": ", values=["Failed due to xxx == yyy returns", step_cond.properties.Outcome]
        ),
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_cond, step_fail],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )

    try:
        response = pipeline.create(role)
        pipeline_arn = response["PipelineArn"]
        execution = pipeline.start(
            parameters={"MyInt": 3},
        )
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
            assert execution_step["FailureReason"] == "Failed due to xxx == yyy returns: false"
            metadata = execution_steps[0]["Metadata"]["Fail"]
            assert metadata["ErrorMessage"] == "Failed due to xxx == yyy returns: false"

        # Check FailureReason field in ListPipelineExecutions
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline.name
        )["PipelineExecutionSummaries"]

        assert len(executions) == 1
        assert executions[0]["PipelineExecutionStatus"] == "Failed"
        assert (
            "Step failure: One or multiple steps failed"
            in executions[0]["PipelineExecutionFailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_fail_pipeline_with_no_err_msg(sagemaker_session, role, pipeline_name):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    step_fail = FailStep(
        name="FailStep",
    )
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[step_fail],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )

    try:
        response = pipeline.create(role)
        pipeline_arn = response["PipelineArn"]
        execution = pipeline.start(parameters={})
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
            assert execution_step.get("FailureReason", None) is None
            metadata = execution_steps[0]["Metadata"]["Fail"]
            assert metadata["ErrorMessage"] == ""

        # Check FailureReason field in ListPipelineExecutions
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline.name
        )["PipelineExecutionSummaries"]

        assert len(executions) == 1
        assert executions[0]["PipelineExecutionStatus"] == "Failed"
        assert (
            "Step failure: One or multiple steps failed"
            in executions[0]["PipelineExecutionFailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_invalid_pipeline_depended_on_fail_step(sagemaker_session, role, pipeline_name):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    step_fail = FailStep(
        name="FailStep",
        error_message="Failed pipeline execution",
    )
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[],
        depends_on=["FailStep"],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_cond, step_fail],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )

    try:
        with pytest.raises(Exception) as error:
            pipeline.create(role)

        assert "CondStep can not depends on FailStep" in str(error.value)
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
