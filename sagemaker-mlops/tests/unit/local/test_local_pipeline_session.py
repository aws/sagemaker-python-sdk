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
"""Unit tests for LocalPipelineSession."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError
from datetime import datetime

from sagemaker.mlops.local.local_pipeline_session import LocalPipelineSession


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.name = "test-pipeline"
    return pipeline


@pytest.fixture
def local_session():
    def mock_init(self, *args, **kwargs):
        self.sagemaker_client = Mock()
        self.sagemaker_client._pipelines = {}
    
    with patch.object(LocalPipelineSession, '__init__', mock_init):
        session = LocalPipelineSession()
        return session


def test_local_pipeline_session_init():
    """Test LocalPipelineSession initialization."""
    def mock_parent_init(self, *args, **kwargs):
        self.sagemaker_client = Mock(spec=[])  # Empty spec means no attributes initially
    
    with patch('sagemaker.core.local.LocalSession.__init__', mock_parent_init):
        session = LocalPipelineSession()
        
        # Verify _pipelines attribute is created as a dict
        assert hasattr(session.sagemaker_client, '_pipelines')
        assert session.sagemaker_client._pipelines == {}


def test_local_pipeline_session_init_with_existing_pipelines():
    """Test LocalPipelineSession initialization when _pipelines already exists."""
    def mock_parent_init(self, *args, **kwargs):
        self.sagemaker_client = Mock()
        self.sagemaker_client._pipelines = {"existing": "pipeline"}
    
    with patch('sagemaker.core.local.LocalSession.__init__', mock_parent_init):
        session = LocalPipelineSession()
        
        # Should not overwrite existing _pipelines
        assert session.sagemaker_client._pipelines == {"existing": "pipeline"}


def test_create_pipeline(local_session, mock_pipeline):
    """Test create_pipeline creates a local pipeline."""
    with patch('sagemaker.mlops.local.local_pipeline_session._LocalPipeline') as mock_local_pipeline:
        mock_local_pipeline_instance = Mock()
        mock_local_pipeline.return_value = mock_local_pipeline_instance
        
        # Call the real method
        result = LocalPipelineSession.create_pipeline(local_session, mock_pipeline, "Test pipeline description")
        
        assert result == {"PipelineArn": "test-pipeline"}
        assert "test-pipeline" in local_session.sagemaker_client._pipelines
        assert local_session.sagemaker_client._pipelines["test-pipeline"] == mock_local_pipeline_instance
        
        mock_local_pipeline.assert_called_once_with(
            pipeline=mock_pipeline,
            pipeline_description="Test pipeline description",
            local_session=local_session,
        )


def test_create_pipeline_with_kwargs(local_session, mock_pipeline):
    """Test create_pipeline ignores extra kwargs."""
    with patch('sagemaker.mlops.local.local_pipeline_session._LocalPipeline') as mock_local_pipeline:
        mock_local_pipeline_instance = Mock()
        mock_local_pipeline.return_value = mock_local_pipeline_instance
        
        result = LocalPipelineSession.create_pipeline(
            local_session,
            mock_pipeline, 
            "Test description",
            extra_param="ignored"
        )
        
        assert result == {"PipelineArn": "test-pipeline"}


def test_update_pipeline(local_session, mock_pipeline):
    """Test update_pipeline updates an existing pipeline."""
    # Create initial pipeline
    mock_local_pipeline = Mock()
    mock_local_pipeline.pipeline_description = "Old description"
    mock_local_pipeline.pipeline = Mock()
    mock_local_pipeline.last_modified_time = 1000.0
    
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    new_pipeline = Mock()
    new_pipeline.name = "test-pipeline"
    
    result = LocalPipelineSession.update_pipeline(local_session, new_pipeline, "New description")
    
    assert result == {"PipelineArn": "test-pipeline"}
    assert mock_local_pipeline.pipeline_description == "New description"
    assert mock_local_pipeline.pipeline == new_pipeline
    assert mock_local_pipeline.last_modified_time > 1000.0


def test_update_pipeline_not_found(local_session, mock_pipeline):
    """Test update_pipeline raises error when pipeline doesn't exist."""
    with pytest.raises(ClientError) as exc_info:
        LocalPipelineSession.update_pipeline(local_session, mock_pipeline, "Description")
    
    error = exc_info.value
    assert error.response['Error']['Code'] == 'ResourceNotFound'
    assert 'test-pipeline' in error.response['Error']['Message']


def test_update_pipeline_with_kwargs(local_session, mock_pipeline):
    """Test update_pipeline ignores extra kwargs."""
    mock_local_pipeline = Mock()
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.update_pipeline(
        local_session,
        mock_pipeline,
        "Description",
        extra_param="ignored"
    )
    
    assert result == {"PipelineArn": "test-pipeline"}


def test_describe_pipeline(local_session):
    """Test describe_pipeline returns pipeline metadata."""
    mock_local_pipeline = Mock()
    mock_local_pipeline.describe = Mock(return_value={
        "PipelineArn": "test-pipeline",
        "PipelineDefinition": "{}",
        "LastModifiedTime": 1234567890
    })
    
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.describe_pipeline(local_session, "test-pipeline")
    
    assert result["PipelineArn"] == "test-pipeline"
    assert "PipelineDefinition" in result
    mock_local_pipeline.describe.assert_called_once()


def test_describe_pipeline_not_found(local_session):
    """Test describe_pipeline raises error when pipeline doesn't exist."""
    with pytest.raises(ClientError) as exc_info:
        LocalPipelineSession.describe_pipeline(local_session, "nonexistent-pipeline")
    
    error = exc_info.value
    assert error.response['Error']['Code'] == 'ResourceNotFound'
    assert 'nonexistent-pipeline' in error.response['Error']['Message']


def test_delete_pipeline(local_session):
    """Test delete_pipeline removes pipeline."""
    mock_local_pipeline = Mock()
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.delete_pipeline(local_session, "test-pipeline")
    
    assert result == {"PipelineArn": "test-pipeline"}
    assert "test-pipeline" not in local_session.sagemaker_client._pipelines


def test_delete_pipeline_not_found(local_session):
    """Test delete_pipeline returns success even if pipeline doesn't exist."""
    result = LocalPipelineSession.delete_pipeline(local_session, "nonexistent-pipeline")
    
    assert result == {"PipelineArn": "nonexistent-pipeline"}


def test_start_pipeline_execution(local_session):
    """Test start_pipeline_execution starts a pipeline."""
    mock_local_pipeline = Mock()
    mock_execution = Mock()
    mock_local_pipeline.start = Mock(return_value=mock_execution)
    
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.start_pipeline_execution(local_session, "test-pipeline")
    
    assert result == mock_execution
    mock_local_pipeline.start.assert_called_once_with()


def test_start_pipeline_execution_with_kwargs(local_session):
    """Test start_pipeline_execution passes kwargs to start."""
    mock_local_pipeline = Mock()
    mock_execution = Mock()
    mock_local_pipeline.start = Mock(return_value=mock_execution)
    
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.start_pipeline_execution(
        local_session,
        "test-pipeline",
        PipelineExecutionDisplayName="test-execution",
        PipelineParameters=[{"Name": "param1", "Value": "value1"}]
    )
    
    assert result == mock_execution
    mock_local_pipeline.start.assert_called_once_with(
        PipelineExecutionDisplayName="test-execution",
        PipelineParameters=[{"Name": "param1", "Value": "value1"}]
    )


def test_start_pipeline_execution_with_parallelism_config(local_session, caplog):
    """Test start_pipeline_execution warns about parallelism config."""
    mock_local_pipeline = Mock()
    mock_execution = Mock()
    mock_local_pipeline.start = Mock(return_value=mock_execution)
    
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    result = LocalPipelineSession.start_pipeline_execution(
        local_session,
        "test-pipeline",
        ParallelismConfiguration={"MaxParallelExecutionSteps": 5}
    )
    
    assert result == mock_execution
    assert "Parallelism configuration is not supported in local mode" in caplog.text


def test_start_pipeline_execution_with_selective_execution_config(local_session):
    """Test start_pipeline_execution raises error for selective execution config."""
    mock_local_pipeline = Mock()
    local_session.sagemaker_client._pipelines["test-pipeline"] = mock_local_pipeline
    
    with pytest.raises(ValueError) as exc_info:
        LocalPipelineSession.start_pipeline_execution(
            local_session,
            "test-pipeline",
            SelectiveExecutionConfig={"SourcePipelineExecutionArn": "arn"}
        )
    
    assert "SelectiveExecutionConfig is not supported in local mode" in str(exc_info.value)


def test_start_pipeline_execution_not_found(local_session):
    """Test start_pipeline_execution raises error when pipeline doesn't exist."""
    with pytest.raises(ClientError) as exc_info:
        LocalPipelineSession.start_pipeline_execution(local_session, "nonexistent-pipeline")
    
    error = exc_info.value
    assert error.response['Error']['Code'] == 'ResourceNotFound'
    assert 'nonexistent-pipeline' in error.response['Error']['Message']
