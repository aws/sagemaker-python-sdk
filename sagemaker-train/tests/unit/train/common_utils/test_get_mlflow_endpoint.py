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
"""Unit tests for get_mlflow_endpoint module."""

import pytest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from sagemaker.train.common_utils.get_mlflow_endpoint import (
    _get_mlflow_tracking_server_endpoint,
    MLflowEndpointError,
)
from sagemaker.train.common_utils.constants import (
    _ErrorConstants,
    _TrainingJobConstants,
    _ValidationConstants,
)


class TestGetMLflowTrackingServerEndpoint:
    """Test cases for get_mlflow_tracking_server_endpoint function."""

    def test_get_endpoint_success(self):
        """Test successful endpoint retrieval."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            mock_client.describe_mlflow_tracking_server.return_value = {
                'TrackingServerUrl': 'https://example.mlflow.com'
            }
            
            result = _get_mlflow_tracking_server_endpoint('test-server')
            
            assert result == 'https://example.mlflow.com'
            mock_boto_client.assert_called_once_with('sagemaker', region_name=_TrainingJobConstants.DEFAULT_AWS_REGION)
            mock_client.describe_mlflow_tracking_server.assert_called_once_with(
                TrackingServerName='test-server'
            )

    def test_get_endpoint_with_custom_region(self):
        """Test endpoint retrieval with custom region."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            mock_client.describe_mlflow_tracking_server.return_value = {
                'TrackingServerUrl': 'https://example.mlflow.com'
            }
            
            result = _get_mlflow_tracking_server_endpoint('test-server', 'us-east-1')
            
            assert result == 'https://example.mlflow.com'
            mock_boto_client.assert_called_once_with('sagemaker', region_name='us-east-1')

    def test_empty_tracking_server_name_raises_error(self):
        """Test that empty tracking server name raises ValueError."""
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_TRACKING_SERVER_NAME_MSG):
            _get_mlflow_tracking_server_endpoint('')
        
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_TRACKING_SERVER_NAME_MSG):
            _get_mlflow_tracking_server_endpoint(None)

    def test_empty_region_raises_error(self):
        """Test that empty region raises ValueError."""
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_REGION_MSG):
            _get_mlflow_tracking_server_endpoint('test-server', '')

    def test_no_tracking_url_in_response(self):
        """Test error when no TrackingServerUrl in response."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            mock_client.describe_mlflow_tracking_server.return_value = {}
            
            with pytest.raises(MLflowEndpointError) as exc_info:
                _get_mlflow_tracking_server_endpoint('test-server')
            
            assert _ErrorConstants.NO_TRACKING_URL.format('test-server') in str(exc_info.value)

    def test_resource_not_found_error(self):
        """Test ResourceNotFound error handling."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            
            client_error = ClientError(
                {'Error': {'Code': 'ResourceNotFound', 'Message': 'Server not found'}}, 
                'describe_mlflow_tracking_server'
            )
            mock_client.describe_mlflow_tracking_server.side_effect = client_error
            
            with pytest.raises(MLflowEndpointError) as exc_info:
                _get_mlflow_tracking_server_endpoint('test-server', 'us-west-2')
            
            expected_error = _ErrorConstants.RESOURCE_NOT_FOUND_ERROR.format('test-server', 'us-west-2')
            assert expected_error in str(exc_info.value)

    def test_generic_client_error(self):
        """Test generic ClientError handling."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            
            client_error = ClientError(
                {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}, 
                'describe_mlflow_tracking_server'
            )
            mock_client.describe_mlflow_tracking_server.side_effect = client_error
            
            with pytest.raises(MLflowEndpointError) as exc_info:
                _get_mlflow_tracking_server_endpoint('test-server')
            
            expected_error = _ErrorConstants.ENDPOINT_RETRIEVAL_ERROR.format('Access denied')
            assert expected_error in str(exc_info.value)

    def test_strips_whitespace_from_inputs(self):
        """Test that whitespace is stripped from inputs."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            mock_client.describe_mlflow_tracking_server.return_value = {
                'TrackingServerUrl': 'https://example.mlflow.com'
            }
            
            result = _get_mlflow_tracking_server_endpoint('  test-server  ', '  us-east-1  ')
            
            assert result == 'https://example.mlflow.com'
            mock_boto_client.assert_called_once_with('sagemaker', region_name='us-east-1')
            mock_client.describe_mlflow_tracking_server.assert_called_once_with(
                TrackingServerName='test-server'
            )

    def test_mlflow_endpoint_error_creation(self):
        """Test MLflowEndpointError exception class."""
        error_msg = "Test error message"
        error = MLflowEndpointError(error_msg)
        assert str(error) == error_msg
        assert isinstance(error, Exception)
