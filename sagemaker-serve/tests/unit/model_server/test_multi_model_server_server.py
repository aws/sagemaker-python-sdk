"""Unit tests for multi_model_server server.py module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestLocalMultiModelServer(unittest.TestCase):
    """Test LocalMultiModelServer class."""

    @patch('sagemaker.serve.model_server.multi_model_server.server.Path')
    def test_start_serving_creates_container(self, mock_path):
        """Test _start_serving creates and configures container."""
        from sagemaker.serve.model_server.multi_model_server.server import LocalMultiModelServer
        
        server = LocalMultiModelServer()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        
        server._start_serving(
            client=mock_client,
            image="test-image:latest",
            model_path="/path/to/model",
            secret_key="test-secret",
            env_vars={"CUSTOM_VAR": "value"}
        )
        
        self.assertEqual(server.container, mock_container)
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", call_kwargs["environment"])
        self.assertEqual(call_kwargs["environment"]["SAGEMAKER_SERVE_SECRET_KEY"], "test-secret")

    @patch('sagemaker.serve.model_server.multi_model_server.server.Path')
    def test_start_serving_with_no_env_vars(self, mock_path):
        """Test _start_serving with no custom env vars."""
        from sagemaker.serve.model_server.multi_model_server.server import LocalMultiModelServer
        
        server = LocalMultiModelServer()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        
        server._start_serving(
            client=mock_client,
            image="test-image:latest",
            model_path="/path/to/model",
            secret_key="test-secret",
            env_vars=None
        )
        
        call_kwargs = mock_client.containers.run.call_args[1]
        self.assertIn("SAGEMAKER_SUBMIT_DIRECTORY", call_kwargs["environment"])
        self.assertIn("SAGEMAKER_PROGRAM", call_kwargs["environment"])

    @patch('sagemaker.serve.model_server.multi_model_server.server.requests.post')
    @patch('sagemaker.serve.model_server.multi_model_server.server.get_docker_host')
    def test_invoke_multi_model_server_serving_success(self, mock_get_host, mock_post):
        """Test _invoke_multi_model_server_serving successful request."""
        from sagemaker.serve.model_server.multi_model_server.server import LocalMultiModelServer
        
        server = LocalMultiModelServer()
        mock_get_host.return_value = "localhost"
        mock_response = Mock()
        mock_response.content = b'{"result": "success"}'
        mock_post.return_value = mock_response
        
        result = server._invoke_multi_model_server_serving(
            request='{"input": "data"}',
            content_type="application/json",
            accept="application/json"
        )
        
        self.assertEqual(result, b'{"result": "success"}')
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(call_kwargs["headers"]["Accept"], "application/json")

    @patch('sagemaker.serve.model_server.multi_model_server.server.requests.post')
    @patch('sagemaker.serve.model_server.multi_model_server.server.get_docker_host')
    def test_invoke_multi_model_server_serving_failure(self, mock_get_host, mock_post):
        """Test _invoke_multi_model_server_serving handles errors."""
        from sagemaker.serve.model_server.multi_model_server.server import LocalMultiModelServer
        
        server = LocalMultiModelServer()
        mock_get_host.return_value = "localhost"
        mock_post.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception) as context:
            server._invoke_multi_model_server_serving(
                request='{"input": "data"}',
                content_type="application/json",
                accept="application/json"
            )
        self.assertIn("Unable to send request", str(context.exception))


class TestSageMakerMultiModelServer(unittest.TestCase):
    """Test SageMakerMultiModelServer class."""

    @patch('sagemaker.serve.model_server.multi_model_server.server.S3Uploader')
    @patch('sagemaker.serve.model_server.multi_model_server.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.multi_model_server.server.fw_utils')
    @patch('sagemaker.serve.model_server.multi_model_server.server._is_s3_uri')
    def test_upload_server_artifacts_with_s3_path(self, mock_is_s3, mock_fw_utils, mock_determine, mock_uploader):
        """Test _upload_server_artifacts with S3 path."""
        from sagemaker.serve.model_server.multi_model_server.server import SageMakerMultiModelServer
        
        server = SageMakerMultiModelServer()
        mock_is_s3.return_value = True
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        model_data, env_vars = server._upload_server_artifacts(
            model_path="s3://bucket/model",
            secret_key="test-key",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertIsNotNone(model_data)
        self.assertEqual(model_data["S3DataSource"]["S3Uri"], "s3://bucket/model/")

    @patch('sagemaker.serve.model_server.multi_model_server.server.S3Uploader')
    @patch('sagemaker.serve.model_server.multi_model_server.server.s3_path_join')
    @patch('sagemaker.serve.model_server.multi_model_server.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.multi_model_server.server.parse_s3_url')
    @patch('sagemaker.serve.model_server.multi_model_server.server.fw_utils')
    @patch('sagemaker.serve.model_server.multi_model_server.server._is_s3_uri')
    @patch('sagemaker.serve.model_server.multi_model_server.server.Path')
    def test_upload_server_artifacts_uploads_to_s3(self, mock_path, mock_is_s3, mock_fw_utils, 
                                                     mock_parse, mock_determine, mock_s3_join, mock_uploader):
        """Test _upload_server_artifacts uploads artifacts to S3."""
        from sagemaker.serve.model_server.multi_model_server.server import SageMakerMultiModelServer
        
        server = SageMakerMultiModelServer()
        mock_is_s3.return_value = False
        mock_parse.return_value = ("bucket", "prefix")
        mock_determine.return_value = ("bucket", "code_prefix")
        mock_s3_join.return_value = "s3://bucket/code_prefix/code"
        mock_uploader.upload.return_value = "s3://bucket/code_prefix/code"
        
        mock_path_obj = Mock()
        mock_code_dir = Mock()
        mock_path_obj.joinpath.return_value = mock_code_dir
        mock_path.return_value = mock_path_obj
        
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        model_data, env_vars = server._upload_server_artifacts(
            model_path="/local/model",
            secret_key="test-key",
            sagemaker_session=mock_session,
            s3_model_data_url="s3://bucket/prefix",
            image="test-image",
            should_upload_artifacts=True
        )
        
        self.assertIsNotNone(model_data)
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        self.assertEqual(env_vars["SAGEMAKER_SERVE_SECRET_KEY"], "test-key")

    @patch('sagemaker.serve.model_server.multi_model_server.server._is_s3_uri')
    def test_upload_server_artifacts_no_upload(self, mock_is_s3):
        """Test _upload_server_artifacts without uploading."""
        from sagemaker.serve.model_server.multi_model_server.server import SageMakerMultiModelServer
        
        server = SageMakerMultiModelServer()
        mock_is_s3.return_value = False
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        model_data, env_vars = server._upload_server_artifacts(
            model_path="/local/model",
            secret_key="test-key",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertIsNone(model_data)
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)


class TestUpdateEnvVars(unittest.TestCase):
    """Test _update_env_vars function."""

    def test_update_env_vars_with_none(self):
        """Test _update_env_vars with None input."""
        from sagemaker.serve.model_server.multi_model_server.server import _update_env_vars
        
        result = _update_env_vars(None)
        self.assertIsInstance(result, dict)

    def test_update_env_vars_with_custom_vars(self):
        """Test _update_env_vars with custom variables."""
        from sagemaker.serve.model_server.multi_model_server.server import _update_env_vars
        
        custom_vars = {"CUSTOM_KEY": "custom_value"}
        result = _update_env_vars(custom_vars)
        
        self.assertIn("CUSTOM_KEY", result)
        self.assertEqual(result["CUSTOM_KEY"], "custom_value")


if __name__ == "__main__":
    unittest.main()
