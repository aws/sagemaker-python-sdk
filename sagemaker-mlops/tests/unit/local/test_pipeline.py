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
"""Unit tests for local pipeline executor."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch

from sagemaker.mlops.local.pipeline import LocalPipelineExecutor
from sagemaker.mlops.local.exceptions import StepExecutionException
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.execution_variables import ExecutionVariables


@pytest.fixture
def mock_session():
    session = Mock()
    session.sagemaker_client = Mock()
    session.read_s3_file = Mock()
    return session


@pytest.fixture
def mock_execution():
    execution = Mock()
    execution.pipeline = Mock()
    execution.pipeline.name = "test-pipeline"
    execution.pipeline_parameters = {}
    execution.step_execution = {}
    execution.pipeline_execution_name = "exec-123"
    execution.creation_time = "2024-01-01T00:00:00"
    return execution


def test_local_pipeline_executor_init(mock_execution, mock_session):
    with patch("sagemaker.mlops.local.pipeline.PipelineGraph"):
        executor = LocalPipelineExecutor(mock_execution, mock_session)
        assert executor.sagemaker_session == mock_session
        assert executor.execution == mock_execution


def test_evaluate_parameter(mock_execution, mock_session):
    param = ParameterString(name="test-param", default_value="test-value")
    mock_execution.pipeline_parameters = {"test-param": "test-value"}
    
    with patch("sagemaker.mlops.local.pipeline.PipelineGraph"):
        executor = LocalPipelineExecutor(mock_execution, mock_session)
        result = executor.evaluate_pipeline_variable(param, "test-step")
        assert result == "test-value"


def test_evaluate_execution_variable_pipeline_name(mock_execution, mock_session):
    with patch("sagemaker.mlops.local.pipeline.PipelineGraph"):
        executor = LocalPipelineExecutor(mock_execution, mock_session)
        result = executor._evaluate_execution_variable(ExecutionVariables.PIPELINE_NAME)
        assert result == "test-pipeline"


def test_evaluate_execution_variable_execution_id(mock_execution, mock_session):
    with patch("sagemaker.mlops.local.pipeline.PipelineGraph"):
        executor = LocalPipelineExecutor(mock_execution, mock_session)
        result = executor._evaluate_execution_variable(ExecutionVariables.PIPELINE_EXECUTION_ID)
        assert result == "exec-123"
