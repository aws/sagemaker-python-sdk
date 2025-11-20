"""Additional unit tests for deployment_progress.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError


class TestDeployDoneWithProgress(unittest.TestCase):
    """Test _deploy_done_with_progress function."""

    @patch('sagemaker.serve.deployment_progress.print')
    def test_deploy_done_with_progress_creating_no_tracker(self, mock_print):
        """Test with Creating status and no progress tracker."""
        from sagemaker.serve.deployment_progress import _deploy_done_with_progress
        
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        
        result = _deploy_done_with_progress(mock_client, "test-endpoint", None)
        
        self.assertIsNone(result)
        mock_print.assert_called()

    @patch('sagemaker.serve.deployment_progress.print')
    def test_deploy_done_with_progress_inservice_no_tracker(self, mock_print):
        """Test with InService status and no progress tracker."""
        from sagemaker.serve.deployment_progress import _deploy_done_with_progress
        
        mock_client = Mock()
        desc = {"EndpointStatus": "InService"}
        mock_client.describe_endpoint.return_value = desc
        
        result = _deploy_done_with_progress(mock_client, "test-endpoint", None)
        
        self.assertEqual(result, desc)


class TestLiveLoggingDeployDoneWithProgress(unittest.TestCase):
    """Test _live_logging_deploy_done_with_progress function."""

    def test_live_logging_validation_exception(self):
        """Test with ValidationException."""
        from sagemaker.serve.deployment_progress import _live_logging_deploy_done_with_progress
        
        mock_client = Mock()
        error_response = {'Error': {'Code': 'ValidationException'}}
        mock_client.describe_endpoint.side_effect = ClientError(error_response, 'DescribeEndpoint')
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", Mock(), {}, 1, None
        )
        
        self.assertIsNone(result)

    @patch('time.sleep')
    def test_live_logging_inservice_with_tracker(self, mock_sleep):
        """Test with InService status and progress tracker."""
        from sagemaker.serve.deployment_progress import _live_logging_deploy_done_with_progress
        
        mock_client = Mock()
        desc = {"EndpointStatus": "InService"}
        mock_client.describe_endpoint.return_value = desc
        
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = []
        
        mock_tracker = Mock()
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 1, mock_tracker
        )
        
        self.assertEqual(result, desc)
        mock_tracker.log.assert_called()

    def test_live_logging_resource_not_found(self):
        """Test with ResourceNotFoundException."""
        from sagemaker.serve.deployment_progress import _live_logging_deploy_done_with_progress
        
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        
        mock_paginator = Mock()
        error_response = {'Error': {'Code': 'ResourceNotFoundException'}}
        mock_paginator.paginate.side_effect = ClientError(error_response, 'FilterLogEvents')
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 1, None
        )
        
        self.assertIsNone(result)

    def test_live_logging_with_log_events(self):
        """Test with log events."""
        from sagemaker.serve.deployment_progress import _live_logging_deploy_done_with_progress
        
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = [
            {
                "nextToken": "token123",
                "events": [
                    {"message": "Log line 1"},
                    {"message": "Log line 2"}
                ]
            }
        ]
        
        mock_tracker = Mock()
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 1, mock_tracker
        )
        
        self.assertIsNone(result)
        self.assertEqual(mock_tracker.log.call_count, 2)


if __name__ == "__main__":
    unittest.main()
