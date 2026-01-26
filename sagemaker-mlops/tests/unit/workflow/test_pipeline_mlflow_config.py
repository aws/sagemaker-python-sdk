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
import pytest
from unittest.mock import Mock, patch

from sagemaker.core.shapes.shapes import MlflowConfig
from sagemaker.mlops.workflow.pipeline import Pipeline, _convert_mlflow_config_to_request
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session for testing."""
    session = Mock()
    session.boto_session.client.return_value = Mock()
    session.boto_region_name = "us-east-1"
    session.sagemaker_client = Mock()
    session.local_mode = False
    session.sagemaker_config = {}
    return session


def ordered(obj):
    """Recursively sort dict keys for comparison."""
    if isinstance(obj, dict):
        return {k: ordered(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [ordered(x) for x in obj]
    return obj


class CustomStep(Step):
    """Custom step for testing."""

    def __init__(self, name, input_data):
        super(CustomStep, self).__init__(name=name, step_type=StepTypeEnum.TRAINING, depends_on=[])
        self.input_data = input_data

    @property
    def arguments(self):
        return {"input_data": self.input_data}

    @property
    def properties(self):
        return None


def test_pipeline_with_mlflow_config(mock_session):
    mlflow_config = MlflowConfig(
        mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/training-test",
        mlflow_experiment_name="training-test-experiment",
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[CustomStep(name="MyStep", input_data="input")],
        mlflow_config=mlflow_config,
        sagemaker_session=mock_session,
    )

    pipeline_definition = json.loads(pipeline.definition())
    assert ordered(pipeline_definition) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "MlflowConfig": {
                "MlflowResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/training-test",
                "MlflowExperimentName": "training-test-experiment",
            },
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": "input"},
                }
            ],
        }
    )


def test_pipeline_without_mlflow_config(mock_session):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[CustomStep(name="MyStep", input_data="input")],
        mlflow_config=None,
        sagemaker_session=mock_session,
    )

    pipeline_definition = json.loads(pipeline.definition())
    assert pipeline_definition.get("MlflowConfig") is None


def test_pipeline_start_with_mlflow_experiment_name(mock_session):
    mock_session.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=mock_session,
    )

    # Test starting with MLflow experiment name
    pipeline.start(mlflow_experiment_name="my-experiment")
    mock_session.sagemaker_client.start_pipeline_execution.assert_called_with(
        PipelineName="MyPipeline", MlflowExperimentName="my-experiment"
    )

    # Test starting without MLflow experiment name
    pipeline.start()
    mock_session.sagemaker_client.start_pipeline_execution.assert_called_with(
        PipelineName="MyPipeline",
    )


def test_pipeline_update_with_mlflow_config(mock_session):
    """Test that pipeline.update() includes MLflow config in the definition sent to the API."""

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[CustomStep(name="MyStep", input_data="input")],
        sagemaker_session=mock_session,
    )

    initial_definition = json.loads(pipeline.definition())
    assert initial_definition.get("MlflowConfig") is None

    mlflow_config = MlflowConfig(
        mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/update-test",
        mlflow_experiment_name="update-test-experiment",
    )
    pipeline.mlflow_config = mlflow_config

    with patch(
        "sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="dummy-role"
    ):
        pipeline.update("dummy-role")

    mock_session.sagemaker_client.update_pipeline.assert_called_once()
    call_args = mock_session.sagemaker_client.update_pipeline.call_args

    pipeline_definition_arg = call_args[1]["PipelineDefinition"]
    definition = json.loads(pipeline_definition_arg)

    assert definition["MlflowConfig"] == {
        "MlflowResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/update-test",
        "MlflowExperimentName": "update-test-experiment",
    }


def test_convert_mlflow_config_to_request_with_valid_config():
    """Test _convert_mlflow_config_to_request with a valid MlflowConfig."""
    mlflow_config = MlflowConfig(
        mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server",
        mlflow_experiment_name="test-experiment",
    )

    result = _convert_mlflow_config_to_request(mlflow_config)

    expected = {
        "MlflowResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server",
        "MlflowExperimentName": "test-experiment",
    }

    assert result == expected


def test_convert_mlflow_config_to_request_with_none():
    """Test _convert_mlflow_config_to_request with None input."""
    result = _convert_mlflow_config_to_request(None)
    assert result is None


def test_convert_mlflow_config_to_request_with_minimal_config():
    """Test _convert_mlflow_config_to_request with minimal required fields."""
    mlflow_config = MlflowConfig(
        mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/minimal",
    )

    result = _convert_mlflow_config_to_request(mlflow_config)

    expected = {
        "MlflowResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/minimal",
        "MlflowExperimentName": None,
    }

    assert result == expected


def test_convert_mlflow_config_to_request_with_unassigned_values():
    """Test _convert_mlflow_config_to_request handles Unassigned values properly."""
    from sagemaker.core.utils.utils import Unassigned

    mlflow_config = MlflowConfig(
        mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test",
    )

    result = _convert_mlflow_config_to_request(mlflow_config)

    expected = {
        "MlflowResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test",
        "MlflowExperimentName": None,
    }

    assert result == expected
