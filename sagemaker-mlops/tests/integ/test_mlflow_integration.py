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

import json
import time
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.workflow.callback_step import (
    CallbackStep,
    CallbackOutput,
    CallbackOutputTypeEnum,
)
from sagemaker.core.shapes.shapes import MlflowConfig
from sagemaker.mlops.workflow.pipeline import Pipeline


@pytest.fixture
def sagemaker_session():
    """Return a SageMaker session for integration tests."""
    return Session()


@pytest.fixture
def role():
    """Return the execution role ARN."""
    return get_execution_role()


@pytest.fixture
def region_name(sagemaker_session):
    """Return the AWS region name."""
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def pipeline_name():
    return f"mlflow-test-pipeline-{int(time.time() * 10 ** 7)}"


def test_pipeline_definition_with_mlflow_config(
    sagemaker_session, role, pipeline_name, region_name
):
    """Verify MLflow config appears correctly in pipeline definition when pipeline is created."""

    mlflow_config = MlflowConfig(
        mlflow_resource_arn=(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/integ-test-server"
        ),
        mlflow_experiment_name="integ-test-experiment",
    )

    callback_step = CallbackStep(
        name="test-callback-step",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"test_input": "test_value"},
        outputs=[CallbackOutput(output_name="output", output_type=CallbackOutputTypeEnum.String)],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[callback_step],
        mlflow_config=mlflow_config,
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        assert response["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])

        assert "MlflowConfig" in definition
        assert definition["MlflowConfig"] == {
            "MlflowResourceArn": (
                "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/integ-test-server"
            ),
            "MlflowExperimentName": "integ-test-experiment",
        }

        assert definition["Version"] == "2020-12-01"
        assert "Steps" in definition
        assert len(definition["Steps"]) == 1

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_pipeline_start_with_mlflow_experiment_override(
    sagemaker_session, role, pipeline_name, region_name
):
    """Verify pipeline can be started with MLflow experiment name override."""

    original_mlflow_config = MlflowConfig(
        mlflow_resource_arn=(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/original-server"
        ),
        mlflow_experiment_name="original-experiment",
    )

    callback_step = CallbackStep(
        name="test-callback-step",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"test_input": "test_value"},
        outputs=[CallbackOutput(output_name="output", output_type=CallbackOutputTypeEnum.String)],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[callback_step],
        mlflow_config=original_mlflow_config,
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        assert response["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])
        assert definition["MlflowConfig"]["MlflowExperimentName"] == "original-experiment"

        execution = pipeline.start(mlflow_experiment_name="runtime-override-experiment")

        assert execution.arn
        execution_response = execution.describe()
        assert execution_response["PipelineExecutionStatus"] in ["Executing", "Succeeded", "Failed"]

        assert (
            execution_response.get("MLflowConfig", {}).get("MlflowExperimentName")
            == "runtime-override-experiment"
        )
        assert (
            execution_response.get("MLflowConfig", {}).get("MlflowResourceArn")
            == "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/original-server"
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_pipeline_update_with_mlflow_config(sagemaker_session, role, pipeline_name, region_name):
    """Verify pipeline can be updated to add or modify MLflow config."""

    callback_step = CallbackStep(
        name="test-callback-step",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"test_input": "test_value"},
        outputs=[CallbackOutput(output_name="output", output_type=CallbackOutputTypeEnum.String)],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[callback_step],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        assert response["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])
        assert definition["MlflowConfig"] is None

        mlflow_config = MlflowConfig(
            mlflow_resource_arn=(
                "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/update-test-server"
            ),
            mlflow_experiment_name="update-test-experiment",
        )
        pipeline.mlflow_config = mlflow_config

        update_response = pipeline.update(role)
        assert update_response["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])
        assert "MlflowConfig" in definition
        assert definition["MlflowConfig"] == {
            "MlflowResourceArn": (
                "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/update-test-server"
            ),
            "MlflowExperimentName": "update-test-experiment",
        }

        pipeline.mlflow_config.mlflow_experiment_name = "modified-experiment"

        update_response2 = pipeline.update(role)
        assert update_response2["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])
        assert definition["MlflowConfig"]["MlflowExperimentName"] == "modified-experiment"
        assert (
            definition["MlflowConfig"]["MlflowResourceArn"]
            == "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/update-test-server"
        )

        pipeline.mlflow_config = None

        update_response3 = pipeline.update(role)
        assert update_response3["PipelineArn"]

        describe_response = pipeline.describe()
        definition = json.loads(describe_response["PipelineDefinition"])
        assert definition["MlflowConfig"] is None

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
