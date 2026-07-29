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
"""Unit tests for workflow pipeline_experiment_config."""
from __future__ import absolute_import

from unittest.mock import Mock

from sagemaker.mlops.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig, PipelineExperimentConfigProperties
)
from sagemaker.mlops.workflow.pipeline import Pipeline, _DEFAULT_EXPERIMENT_CFG
from sagemaker.core.workflow.execution_variables import ExecutionVariables


def test_pipeline_experiment_config_init():
    config = PipelineExperimentConfig(
        experiment_name="test-experiment",
        trial_name="test-trial"
    )
    assert config.experiment_name == "test-experiment"
    assert config.trial_name == "test-trial"


def test_pipeline_experiment_config_with_execution_variables():
    config = PipelineExperimentConfig(
        experiment_name=ExecutionVariables.PIPELINE_NAME,
        trial_name=ExecutionVariables.PIPELINE_EXECUTION_ID
    )
    request = config.to_request()
    assert "ExperimentName" in request
    assert "TrialName" in request


def test_pipeline_experiment_config_properties():
    assert PipelineExperimentConfigProperties.EXPERIMENT_NAME.name == "ExperimentName"
    assert PipelineExperimentConfigProperties.TRIAL_NAME.name == "TrialName"


def _create_mock_session(region: str) -> Mock:
    """Helper to create a mock SageMaker session with specified region."""
    mock_session = Mock()
    mock_session.boto_region_name = region
    mock_session.boto_session = Mock()
    mock_session.boto_session.client = Mock(return_value=Mock())
    mock_session.local_mode = False
    return mock_session


def test_default_config_applied_in_ga_region():
    """Default config applied when nothing provided in GA region."""
    mock_session = _create_mock_session("us-east-1")
    pipeline = Pipeline(name="test-pipeline", sagemaker_session=mock_session)
    assert pipeline.pipeline_experiment_config == _DEFAULT_EXPERIMENT_CFG


def test_no_default_config_in_non_ga_region():
    """No default config when nothing provided in non-GA region (THE FIX)."""
    mock_session = _create_mock_session("us-gov-west-1")
    pipeline = Pipeline(name="test-pipeline", sagemaker_session=mock_session)
    assert pipeline.pipeline_experiment_config is None


def test_explicit_none_respected_in_ga_region():
    """None gets default config in GA region."""
    mock_session = _create_mock_session("us-east-1")
    pipeline = Pipeline(name="test-pipeline", sagemaker_session=mock_session, pipeline_experiment_config=None)
    assert pipeline.pipeline_experiment_config is None


def test_custom_config_respected():
    """Custom config respected regardless of region."""
    mock_session = _create_mock_session("us-east-1")
    custom_config = PipelineExperimentConfig("my-experiment", "my-trial")
    pipeline = Pipeline(name="test-pipeline", sagemaker_session=mock_session, pipeline_experiment_config=custom_config)
    assert pipeline.pipeline_experiment_config == custom_config