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
"""Unit tests for Pipeline class."""
from __future__ import absolute_import

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.mlops.workflow.pipeline import Pipeline, _DEFAULT_EXPERIMENT_CFG, _DEFAULT_DEFINITION_CFG
from sagemaker.mlops.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.core.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.mlops.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.mlops.workflow.selective_execution_config import SelectiveExecutionConfig
from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


@pytest.fixture
def mock_session():
    session = Mock()
    session.sagemaker_client = Mock()
    session.boto_session = Mock()
    session.boto_session.client = Mock(return_value=Mock())
    session.local_mode = False
    session._append_sagemaker_config_tags = Mock(return_value=[])
    return session


@pytest.fixture
def mock_step():
    step = Mock(spec=Step)
    step.name = "test-step"
    step.step_type = StepTypeEnum.TRAINING
    step.depends_on = []
    step.to_request = Mock(return_value={
        "Name": "test-step",
        "Type": "Training",
        "Arguments": {}
    })
    return step


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_init_minimal(self):
        """Test Pipeline initialization with minimal parameters."""
        with patch('sagemaker.mlops.workflow.pipeline.Session') as mock_session_class:
            mock_session_class.return_value = Mock()
            
            pipeline = Pipeline(name="test-pipeline")

            assert pipeline.name == "test-pipeline"
            assert pipeline.parameters == []
            assert pipeline.steps == []
            assert pipeline.pipeline_experiment_config == _DEFAULT_EXPERIMENT_CFG
            assert pipeline.pipeline_definition_config == _DEFAULT_DEFINITION_CFG

    def test_init_with_parameters(self, mock_session):
        """Test Pipeline initialization with parameters."""
        param1 = ParameterString(name="param1", default_value="value1")
        param2 = ParameterInteger(name="param2", default_value=10)

        pipeline = Pipeline(
            name="test-pipeline",
            parameters=[param1, param2],
            sagemaker_session=mock_session
        )

        assert len(pipeline.parameters) == 2
        assert pipeline.parameters[0] == param1
        assert pipeline.parameters[1] == param2

    def test_init_with_steps(self, mock_session, mock_step):
        """Test Pipeline initialization with steps."""
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[mock_step],
            sagemaker_session=mock_session
        )

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == mock_step

    def test_init_with_experiment_config(self, mock_session):
        """Test Pipeline initialization with experiment config."""
        exp_config = PipelineExperimentConfig(
            experiment_name="test-experiment",
            trial_name="test-trial"
        )

        pipeline = Pipeline(
            name="test-pipeline",
            pipeline_experiment_config=exp_config,
            sagemaker_session=mock_session
        )

        assert pipeline.pipeline_experiment_config == exp_config

    def test_init_with_none_experiment_config(self, mock_session):
        """Test Pipeline initialization with None experiment config."""
        pipeline = Pipeline(
            name="test-pipeline",
            pipeline_experiment_config=None,
            sagemaker_session=mock_session
        )

        assert pipeline.pipeline_experiment_config is None

    def test_init_with_definition_config(self, mock_session):
        """Test Pipeline initialization with definition config."""
        def_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

        pipeline = Pipeline(
            name="test-pipeline",
            pipeline_definition_config=def_config,
            sagemaker_session=mock_session
        )

        assert pipeline.pipeline_definition_config == def_config


class TestPipelineCreate:
    """Tests for Pipeline.create method."""

    def test_create_success(self, mock_session):
        """Test create pipeline successfully."""
        mock_session.sagemaker_client.create_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                with patch('sagemaker.mlops.workflow.pipeline._append_project_tags') as mock_append:
                    mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                    mock_format.return_value = []
                    mock_append.return_value = []

                    pipeline = Pipeline(
                        name="test-pipeline",
                        sagemaker_session=mock_session
                    )

                    with patch.object(pipeline, 'definition', return_value='{"Steps": []}'):
                        result = pipeline.create(role_arn="arn:aws:iam::123:role/SageMakerRole")

                    assert "PipelineArn" in result
                    mock_session.sagemaker_client.create_pipeline.assert_called_once()

    def test_create_without_role_raises_error(self, mock_session):
        """Test create without role raises ValueError."""
        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            mock_resolve.return_value = None

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            with pytest.raises(ValueError) as exc_info:
                pipeline.create()

            assert "AWS IAM role is required" in str(exc_info.value)

    def test_create_with_description(self, mock_session):
        """Test create pipeline with description."""
        mock_session.sagemaker_client.create_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                with patch('sagemaker.mlops.workflow.pipeline._append_project_tags') as mock_append:
                    mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                    mock_format.return_value = []
                    mock_append.return_value = []

                    pipeline = Pipeline(
                        name="test-pipeline",
                        sagemaker_session=mock_session
                    )

                    with patch.object(pipeline, 'definition', return_value='{"Steps": []}'):
                        result = pipeline.create(
                            role_arn="arn:aws:iam::123:role/SageMakerRole",
                            description="Test pipeline description"
                        )

                    call_kwargs = mock_session.sagemaker_client.create_pipeline.call_args[1]
                    assert call_kwargs.get("PipelineDescription") == "Test pipeline description"

    def test_create_with_tags(self, mock_session):
        """Test create pipeline with tags."""
        mock_session.sagemaker_client.create_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                with patch('sagemaker.mlops.workflow.pipeline._append_project_tags') as mock_append:
                    mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                    mock_format.return_value = [{"Key": "Environment", "Value": "Test"}]
                    mock_append.return_value = [{"Key": "Environment", "Value": "Test"}]

                    pipeline = Pipeline(
                        name="test-pipeline",
                        sagemaker_session=mock_session
                    )

                    with patch.object(pipeline, 'definition', return_value='{"Steps": []}'):
                        result = pipeline.create(
                            role_arn="arn:aws:iam::123:role/SageMakerRole",
                            tags=[{"Key": "Environment", "Value": "Test"}]
                        )

                    call_kwargs = mock_session.sagemaker_client.create_pipeline.call_args[1]
                    assert "Tags" in call_kwargs

    def test_create_with_parallelism_config(self, mock_session):
        """Test create pipeline with parallelism config."""
        mock_session.sagemaker_client.create_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                with patch('sagemaker.mlops.workflow.pipeline._append_project_tags') as mock_append:
                    mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                    mock_format.return_value = []
                    mock_append.return_value = []

                    pipeline = Pipeline(
                        name="test-pipeline",
                        sagemaker_session=mock_session
                    )

                    parallelism_config = ParallelismConfiguration(max_parallel_execution_steps=5)

                    with patch.object(pipeline, 'definition', return_value='{"Steps": []}'):
                        result = pipeline.create(
                            role_arn="arn:aws:iam::123:role/SageMakerRole",
                            parallelism_config=parallelism_config
                        )

                    call_kwargs = mock_session.sagemaker_client.create_pipeline.call_args[1]
                    assert "ParallelismConfiguration" in call_kwargs

    def test_create_local_mode(self, mock_session):
        """Test create pipeline in local mode."""
        mock_session.local_mode = True
        mock_session.sagemaker_client.create_pipeline = Mock(return_value={
            "PipelineArn": "test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            result = pipeline.create(role_arn="arn:aws:iam::123:role/SageMakerRole")

            assert result["PipelineArn"] == "test-pipeline"
            mock_session.sagemaker_client.create_pipeline.assert_called_once()


class TestPipelineUpdate:
    """Tests for Pipeline.update method."""

    def test_update_success(self, mock_session):
        """Test update pipeline successfully."""
        mock_session.sagemaker_client.update_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            with patch.object(pipeline, 'definition', return_value='{"Steps": []}'):
                result = pipeline.update(role_arn="arn:aws:iam::123:role/SageMakerRole")

            assert "PipelineArn" in result
            mock_session.sagemaker_client.update_pipeline.assert_called_once()

    def test_update_without_role_raises_error(self, mock_session):
        """Test update without role raises ValueError."""
        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            mock_resolve.return_value = None

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            with pytest.raises(ValueError) as exc_info:
                pipeline.update()

            assert "AWS IAM role is required" in str(exc_info.value)

    def test_update_local_mode(self, mock_session):
        """Test update pipeline in local mode."""
        mock_session.local_mode = True
        mock_session.sagemaker_client.update_pipeline = Mock(return_value={
            "PipelineArn": "test-pipeline"
        })

        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            result = pipeline.update(role_arn="arn:aws:iam::123:role/SageMakerRole")

            assert result["PipelineArn"] == "test-pipeline"


class TestPipelineUpsert:
    """Tests for Pipeline.upsert method."""

    def test_upsert_creates_new_pipeline(self, mock_session):
        """Test upsert creates new pipeline when it doesn't exist."""
        with patch.object(Pipeline, 'create') as mock_create:
            with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
                with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                    mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                    mock_format.return_value = []
                    mock_create.return_value = {
                        "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
                    }

                    pipeline = Pipeline(
                        name="test-pipeline",
                        sagemaker_session=mock_session
                    )

                    result = pipeline.upsert(role_arn="arn:aws:iam::123:role/SageMakerRole")

                    assert "PipelineArn" in result
                    mock_create.assert_called_once()

    def test_upsert_updates_existing_pipeline(self, mock_session):
        """Test upsert updates pipeline when it already exists."""
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Pipeline already exists"
            }
        }

        # Mock list_tags to return existing tags
        mock_session.sagemaker_client.list_tags = Mock(return_value={
            "Tags": [{"Key": "OldTag", "Value": "OldValue"}]
        })
        mock_session.sagemaker_client.add_tags = Mock()

        with patch.object(Pipeline, 'create') as mock_create:
            with patch.object(Pipeline, 'update') as mock_update:
                with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
                    with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                        mock_resolve.return_value = "arn:aws:iam::123:role/SageMakerRole"
                        mock_format.return_value = [{"Key": "NewTag", "Value": "NewValue"}]
                        mock_create.side_effect = ClientError(error_response, "create_pipeline")
                        mock_update.return_value = {
                            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
                        }

                        pipeline = Pipeline(
                            name="test-pipeline",
                            sagemaker_session=mock_session
                        )

                        result = pipeline.upsert(
                            role_arn="arn:aws:iam::123:role/SageMakerRole",
                            tags=[{"Key": "NewTag", "Value": "NewValue"}]
                        )

                        assert "PipelineArn" in result
                        mock_update.assert_called_once()
                        # Verify tags were merged and added
                        mock_session.sagemaker_client.add_tags.assert_called_once()

    def test_upsert_without_role_raises_error(self, mock_session):
        """Test upsert without role raises ValueError."""
        with patch('sagemaker.mlops.workflow.pipeline.resolve_value_from_config') as mock_resolve:
            with patch('sagemaker.mlops.workflow.pipeline.format_tags') as mock_format:
                mock_resolve.return_value = None
                mock_format.return_value = []

                pipeline = Pipeline(
                    name="test-pipeline",
                    sagemaker_session=mock_session
                )

                with pytest.raises(ValueError) as exc_info:
                    pipeline.upsert()

                assert "AWS IAM role is required" in str(exc_info.value)


class TestPipelineDelete:
    """Tests for Pipeline.delete method."""

    def test_delete_success(self, mock_session):
        """Test delete pipeline successfully."""
        mock_session.sagemaker_client.delete_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline"
        })

        pipeline = Pipeline(
            name="test-pipeline",
            sagemaker_session=mock_session
        )

        result = pipeline.delete()

        assert "PipelineArn" in result
        mock_session.sagemaker_client.delete_pipeline.assert_called_once_with(
            PipelineName="test-pipeline"
        )


class TestPipelineDescribe:
    """Tests for Pipeline.describe method."""

    def test_describe_success(self, mock_session):
        """Test describe pipeline successfully."""
        mock_session.sagemaker_client.describe_pipeline = Mock(return_value={
            "PipelineArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test-pipeline",
            "PipelineName": "test-pipeline",
            "PipelineStatus": "Active"
        })

        pipeline = Pipeline(
            name="test-pipeline",
            sagemaker_session=mock_session
        )

        result = pipeline.describe()

        assert result["PipelineName"] == "test-pipeline"
        assert result["PipelineStatus"] == "Active"
        mock_session.sagemaker_client.describe_pipeline.assert_called_once_with(
            PipelineName="test-pipeline"
        )


class TestPipelineStart:
    """Tests for Pipeline.start method."""

    def test_start_success(self, mock_session):
        """Test start pipeline execution successfully."""
        mock_session.sagemaker_client.start_pipeline_execution = Mock(return_value={
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/exec-123"
        })

        with patch('sagemaker.mlops.workflow.pipeline.retry_with_backoff') as mock_retry:
            mock_retry.return_value = {
                "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/exec-123"
            }

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            result = pipeline.start()

            assert result is not None
            mock_retry.assert_called_once()

    def test_start_with_parameters(self, mock_session):
        """Test start pipeline with parameters."""
        mock_session.sagemaker_client.start_pipeline_execution = Mock(return_value={
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/exec-123"
        })

        with patch('sagemaker.mlops.workflow.pipeline.retry_with_backoff') as mock_retry:
            with patch('sagemaker.mlops.workflow.pipeline.format_start_parameters') as mock_format:
                mock_retry.return_value = {
                    "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/exec-123"
                }
                mock_format.return_value = [{"Name": "param1", "Value": "value1"}]

                pipeline = Pipeline(
                    name="test-pipeline",
                    sagemaker_session=mock_session
                )

                result = pipeline.start(parameters={"param1": "value1"})

                assert result is not None
                mock_format.assert_called_once_with({"param1": "value1"})

    def test_start_with_execution_display_name(self, mock_session):
        """Test start pipeline with execution display name."""
        with patch('sagemaker.mlops.workflow.pipeline.retry_with_backoff') as mock_retry:
            mock_retry.return_value = {
                "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/exec-123"
            }

            pipeline = Pipeline(
                name="test-pipeline",
                sagemaker_session=mock_session
            )

            result = pipeline.start(execution_display_name="Test Execution")

            assert result is not None

    def test_start_local_mode(self, mock_session):
        """Test start pipeline in local mode."""
        mock_session.local_mode = True
        mock_session.sagemaker_client.start_pipeline_execution = Mock(return_value=Mock())

        pipeline = Pipeline(
            name="test-pipeline",
            sagemaker_session=mock_session
        )

        result = pipeline.start(parameters={"param1": "value1"})

        assert result is not None
        mock_session.sagemaker_client.start_pipeline_execution.assert_called_once()


class TestPipelineDefinition:
    """Tests for Pipeline.definition method."""

    def test_definition_returns_json_string(self, mock_session, mock_step):
        """Test definition returns JSON string."""
        with patch('sagemaker.mlops.workflow.pipeline.StepsCompiler') as mock_compiler:
            mock_compiler_instance = Mock()
            mock_compiler_instance.build = Mock(return_value=[mock_step])
            mock_compiler.return_value = mock_compiler_instance

            pipeline = Pipeline(
                name="test-pipeline",
                steps=[mock_step],
                sagemaker_session=mock_session
            )

            result = pipeline.definition()

            assert isinstance(result, str)
            # Should be valid JSON
            parsed = json.loads(result)
            assert "Version" in parsed
            assert "Steps" in parsed



class TestPipelineExecutionMethods:
    """Test Pipeline execution-related methods."""
    
    def test_list_executions(self, mock_session):
        """Test list_executions method."""
        mock_session.sagemaker_client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [
                {
                    "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1",
                    "StartTime": "2024-01-01T00:00:00Z",
                    "PipelineExecutionStatus": "Succeeded"
                }
            ],
            "NextToken": "token123"
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        result = pipeline.list_executions(
            sort_by="CreationTime",
            sort_order="Descending",
            max_results=10
        )
        
        assert "PipelineExecutionSummaries" in result
        assert "NextToken" in result
        assert len(result["PipelineExecutionSummaries"]) == 1
        assert result["NextToken"] == "token123"
        
        mock_session.sagemaker_client.list_pipeline_executions.assert_called_once_with(
            PipelineName="test-pipeline",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10
        )
    
    def test_list_executions_with_next_token(self, mock_session):
        """Test list_executions with next_token."""
        mock_session.sagemaker_client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [],
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        result = pipeline.list_executions(next_token="token123")
        
        mock_session.sagemaker_client.list_pipeline_executions.assert_called_once_with(
            PipelineName="test-pipeline",
            NextToken="token123"
        )
    
    def test_get_latest_execution_arn(self, mock_session):
        """Test _get_latest_execution_arn method."""
        mock_session.sagemaker_client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [
                {
                    "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1",
                }
            ]
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        arn = pipeline._get_latest_execution_arn()
        
        assert arn == "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        mock_session.sagemaker_client.list_pipeline_executions.assert_called_once_with(
            PipelineName="test-pipeline",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
    
    def test_get_latest_execution_arn_no_executions(self, mock_session):
        """Test _get_latest_execution_arn with no executions."""
        mock_session.sagemaker_client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": []
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        arn = pipeline._get_latest_execution_arn()
        
        assert arn is None
    
    def test_get_parameters_for_execution(self, mock_session):
        """Test _get_parameters_for_execution method."""
        mock_session.sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        }
        
        mock_session.sagemaker_client.list_pipeline_parameters_for_execution.return_value = {
            "PipelineParameters": [
                {"Name": "param1", "Value": "value1"},
                {"Name": "param2", "Value": "value2"}
            ]
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        params = pipeline._get_parameters_for_execution(
            "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        )
        
        assert params == {"param1": "value1", "param2": "value2"}
    
    def test_get_parameters_for_execution_with_pagination(self, mock_session):
        """Test _get_parameters_for_execution with pagination."""
        mock_session.sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        }
        
        # First call returns NextToken
        mock_session.sagemaker_client.list_pipeline_parameters_for_execution.side_effect = [
            {
                "PipelineParameters": [
                    {"Name": "param1", "Value": "value1"}
                ],
                "NextToken": "token123"
            },
            {
                "PipelineParameters": [
                    {"Name": "param2", "Value": "value2"}
                ]
            }
        ]
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        params = pipeline._get_parameters_for_execution(
            "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        )
        
        assert params == {"param1": "value1", "param2": "value2"}
        assert mock_session.sagemaker_client.list_pipeline_parameters_for_execution.call_count == 2
    
    def test_build_parameters_from_execution(self, mock_session):
        """Test build_parameters_from_execution method."""
        mock_session.sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        }
        
        mock_session.sagemaker_client.list_pipeline_parameters_for_execution.return_value = {
            "PipelineParameters": [
                {"Name": "param1", "Value": "value1"},
                {"Name": "param2", "Value": "value2"}
            ]
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        params = pipeline.build_parameters_from_execution(
            "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        )
        
        assert params == {"param1": "value1", "param2": "value2"}
    
    def test_build_parameters_from_execution_with_overrides(self, mock_session):
        """Test build_parameters_from_execution with parameter overrides."""
        mock_session.sagemaker_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1"
        }
        
        mock_session.sagemaker_client.list_pipeline_parameters_for_execution.return_value = {
            "PipelineParameters": [
                {"Name": "param1", "Value": "value1"},
                {"Name": "param2", "Value": "value2"}
            ]
        }
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[],
            sagemaker_session=mock_session
        )
        
        with patch.object(pipeline, '_validate_parameter_overrides'):
            params = pipeline.build_parameters_from_execution(
                "arn:aws:sagemaker:us-west-2:123:execution/test-pipeline/exec-1",
                parameter_value_overrides={"param2": "new_value2", "param3": "value3"}
            )
        
        assert params == {"param1": "value1", "param2": "new_value2", "param3": "value3"}


class TestPipelineDefinitionMethod:
    """Test Pipeline definition method."""
    
    def test_definition_basic(self, mock_session, mock_step):
        """Test definition method returns JSON string."""
        pipeline = Pipeline(
            name="test-pipeline",
            steps=[mock_step],
            sagemaker_session=mock_session
        )
        
        definition = pipeline.definition()
        
        # Should return a JSON string
        assert isinstance(definition, str)
        
        # Should be valid JSON
        parsed = json.loads(definition)
        assert "Version" in parsed
        assert "Steps" in parsed
        assert parsed["Steps"][0]["Name"] == "test-step"



