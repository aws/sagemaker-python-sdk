"""Unit tests for sagemaker.serve.deployment_progress module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
from sagemaker.serve.deployment_progress import (
    EndpointDeploymentProgress,
    _deploy_done_with_progress,
    _live_logging_deploy_done_with_progress,
)


class TestEndpointDeploymentProgress(unittest.TestCase):
    """Test cases for EndpointDeploymentProgress class."""

    def test_init(self):
        """Test EndpointDeploymentProgress initialization."""
        progress = EndpointDeploymentProgress("test-endpoint")
        self.assertEqual(progress.endpoint_name, "test-endpoint")
        self.assertEqual(progress.current_status, "Creating")
        self.assertIsNone(progress.live)
        self.assertIsNotNone(progress.console)
        self.assertIsNotNone(progress.progress)
        self.assertIsNotNone(progress.status)

    @patch('sagemaker.serve.deployment_progress.Live')
    def test_context_manager_enter(self, mock_live):
        """Test entering context manager."""
        progress = EndpointDeploymentProgress("test-endpoint")
        with progress as p:
            self.assertIsNotNone(p.live)
            mock_live.return_value.start.assert_called_once()

    @patch('sagemaker.serve.deployment_progress.Live')
    def test_context_manager_exit(self, mock_live):
        """Test exiting context manager."""
        progress = EndpointDeploymentProgress("test-endpoint")
        with progress:
            pass
        mock_live.return_value.stop.assert_called_once()

    @patch('sagemaker.serve.deployment_progress.Console')
    def test_log_message(self, mock_console_class):
        """Test logging a message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        progress = EndpointDeploymentProgress("test-endpoint")
        progress.log("Test message")
        
        mock_console.print.assert_called_once_with("Test message")

    def test_update_status(self):
        """Test updating deployment status."""
        progress = EndpointDeploymentProgress("test-endpoint")
        progress.update_status("InService")
        
        self.assertEqual(progress.current_status, "InService")


class TestDeployDoneWithProgress(unittest.TestCase):
    """Test cases for _deploy_done_with_progress function."""

    def test_deploy_done_creating_status(self):
        """Test deployment in Creating status."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "Creating"
        }
        
        result = _deploy_done_with_progress(mock_client, "test-endpoint")
        
        self.assertIsNone(result)
        mock_client.describe_endpoint.assert_called_once_with(EndpointName="test-endpoint")

    def test_deploy_done_updating_status(self):
        """Test deployment in Updating status."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "Updating"
        }
        
        result = _deploy_done_with_progress(mock_client, "test-endpoint")
        
        self.assertIsNone(result)

    def test_deploy_done_inservice_status(self):
        """Test deployment in InService status."""
        mock_client = Mock()
        expected_desc = {"EndpointStatus": "InService"}
        mock_client.describe_endpoint.return_value = expected_desc
        
        result = _deploy_done_with_progress(mock_client, "test-endpoint")
        
        self.assertEqual(result, expected_desc)

    def test_deploy_done_with_progress_tracker(self):
        """Test deployment with progress tracker."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        mock_tracker = Mock()
        
        result = _deploy_done_with_progress(
            mock_client, "test-endpoint", progress_tracker=mock_tracker
        )
        
        mock_tracker.update_status.assert_called_once_with("InService")
        self.assertIsNotNone(result)


class TestLiveLoggingDeployDoneWithProgress(unittest.TestCase):
    """Test cases for _live_logging_deploy_done_with_progress function."""

    def test_endpoint_not_found(self):
        """Test when endpoint doesn't exist yet."""
        mock_client = Mock()
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}},
            "describe_endpoint"
        )
        mock_paginator = Mock()
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 5
        )
        
        self.assertIsNone(result)

    def test_endpoint_creating_status(self):
        """Test endpoint in Creating status."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "Creating"
        }
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = []
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 5
        )
        
        self.assertIsNone(result)

    def test_endpoint_inservice_status(self):
        """Test endpoint in InService status."""
        mock_client = Mock()
        expected_desc = {"EndpointStatus": "InService"}
        mock_client.describe_endpoint.return_value = expected_desc
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = []
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 5
        )
        
        self.assertEqual(result, expected_desc)

    def test_with_progress_tracker_and_logs(self):
        """Test with progress tracker and CloudWatch logs."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = [
            {
                "events": [
                    {"message": "Log line 1"},
                    {"message": "Log line 2"}
                ]
            }
        ]
        mock_tracker = Mock()
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 5, mock_tracker
        )
        
        # Should log success message when InService
        self.assertGreaterEqual(mock_tracker.log.call_count, 1)
        mock_tracker.update_status.assert_called_once_with("InService")

    def test_resource_not_found_exception(self):
        """Test ResourceNotFoundException during log fetching."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "Creating"
        }
        mock_paginator = Mock()
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "paginate"
        )
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, {}, 5
        )
        
        self.assertIsNone(result)

    def test_pagination_with_next_token(self):
        """Test pagination with nextToken."""
        mock_client = Mock()
        mock_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        mock_paginator = Mock()
        paginator_config = {}
        mock_paginator.paginate.return_value = [
            {
                "nextToken": "token123",
                "events": [{"message": "Log 1"}]
            }
        ]
        
        result = _live_logging_deploy_done_with_progress(
            mock_client, "test-endpoint", mock_paginator, paginator_config, 5
        )
        
        self.assertEqual(paginator_config.get("StartingToken"), "token123")


if __name__ == "__main__":
    unittest.main()
