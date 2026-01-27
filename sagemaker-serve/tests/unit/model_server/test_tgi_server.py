"""Unit tests for tgi server.py module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestLocalTgiServing(unittest.TestCase):
    """Test LocalTgiServing class."""

    @patch('sagemaker.serve.model_server.tgi.server.Path')
    @patch('sagemaker.serve.model_server.tgi.server.DeviceRequest')
    def test_start_tgi_serving_jumpstart(self, mock_device_req, mock_path):
        """Test _start_tgi_serving with jumpstart=True."""
        from sagemaker.serve.model_server.tgi.server import LocalTgiServing
        
        server = LocalTgiServing()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        mock_device_req.return_value = Mock()
        
        server._start_tgi_serving(
            client=mock_client,
            image="test-image:latest",
            model_path="/path/to/model",
            env_vars={"CUSTOM_VAR": "value"},
            jumpstart=True
        )
        
        self.assertEqual(server.container, mock_container)
        mock_client.containers.run.assert_called_once()
        call_args = mock_client.containers.run.call_args
        # Check that the command includes --model-id
        self.assertEqual(call_args[0][1][0], "--model-id")

    @patch('sagemaker.serve.model_server.tgi.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tgi.server.Path')
    @patch('sagemaker.serve.model_server.tgi.server.DeviceRequest')
    def test_start_tgi_serving_non_jumpstart(self, mock_device_req, mock_path, mock_update_env):
        """Test _start_tgi_serving with jumpstart=False."""
        from sagemaker.serve.model_server.tgi.server import LocalTgiServing
        
        server = LocalTgiServing()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        mock_device_req.return_value = Mock()
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        
        server._start_tgi_serving(
            client=mock_client,
            image="test-image:latest",
            model_path="/path/to/model",
            env_vars={"CUSTOM_VAR": "value"},
            jumpstart=False
        )
        
        self.assertEqual(server.container, mock_container)
        mock_update_env.assert_called_once()

    @patch('sagemaker.serve.model_server.tgi.server.requests.post')
    @patch('sagemaker.serve.model_server.tgi.server.get_docker_host')
    def test_invoke_tgi_serving_success(self, mock_get_host, mock_post):
        """Test _invoke_tgi_serving successful request."""
        from sagemaker.serve.model_server.tgi.server import LocalTgiServing
        
        server = LocalTgiServing()
        mock_get_host.return_value = "localhost"
        mock_response = Mock()
        mock_response.content = b'{"generated_text": "result"}'
        mock_post.return_value = mock_response
        
        result = server._invoke_tgi_serving(
            request='{"inputs": "test"}',
            content_type="application/json",
            accept="application/json"
        )
        
        self.assertEqual(result, b'{"generated_text": "result"}')
        mock_post.assert_called_once()

    @patch('sagemaker.serve.model_server.tgi.server.requests.post')
    @patch('sagemaker.serve.model_server.tgi.server.get_docker_host')
    def test_invoke_tgi_serving_failure(self, mock_get_host, mock_post):
        """Test _invoke_tgi_serving handles errors."""
        from sagemaker.serve.model_server.tgi.server import LocalTgiServing
        
        server = LocalTgiServing()
        mock_get_host.return_value = "localhost"
        mock_post.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception) as context:
            server._invoke_tgi_serving(
                request='{"inputs": "test"}',
                content_type="application/json",
                accept="application/json"
            )
        self.assertIn("Unable to send request", str(context.exception))


class TestSageMakerTgiServing(unittest.TestCase):
    """Test SageMakerTgiServing class."""

    @patch('sagemaker.serve.model_server.tgi.server._is_s3_uri')
    def test_upload_tgi_artifacts_with_s3_path(self, mock_is_s3):
        """Test _upload_tgi_artifacts with S3 path."""
        from sagemaker.serve.model_server.tgi.server import SageMakerTgiServing
        
        server = SageMakerTgiServing()
        mock_is_s3.return_value = True
        mock_session = Mock()
        
        model_data, env_vars = server._upload_tgi_artifacts(
            model_path="s3://bucket/model",
            sagemaker_session=mock_session,
            jumpstart=False,
            should_upload_artifacts=False
        )
        
        self.assertIsNotNone(model_data)
        self.assertEqual(model_data["S3DataSource"]["S3Uri"], "s3://bucket/model/")

    @patch('sagemaker.serve.model_server.tgi.server._is_s3_uri')
    def test_upload_tgi_artifacts_jumpstart(self, mock_is_s3):
        """Test _upload_tgi_artifacts with jumpstart=True."""
        from sagemaker.serve.model_server.tgi.server import SageMakerTgiServing
        
        server = SageMakerTgiServing()
        mock_is_s3.return_value = True
        mock_session = Mock()
        
        model_data, env_vars = server._upload_tgi_artifacts(
            model_path="s3://bucket/model",
            sagemaker_session=mock_session,
            jumpstart=True,
            should_upload_artifacts=False
        )
        
        self.assertIsNotNone(model_data)
        self.assertEqual(env_vars, {})

    @patch('sagemaker.serve.model_server.tgi.server.S3Uploader')
    @patch('sagemaker.serve.model_server.tgi.server.s3_path_join')
    @patch('sagemaker.serve.model_server.tgi.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.tgi.server.parse_s3_url')
    @patch('sagemaker.serve.model_server.tgi.server.fw_utils')
    @patch('sagemaker.serve.model_server.tgi.server._is_s3_uri')
    @patch('sagemaker.serve.model_server.tgi.server.Path')
    def test_upload_tgi_artifacts_uploads_to_s3(self, mock_path, mock_is_s3, mock_fw_utils,
                                                  mock_parse, mock_determine, mock_s3_join, mock_uploader):
        """Test _upload_tgi_artifacts uploads to S3."""
        from sagemaker.serve.model_server.tgi.server import SageMakerTgiServing
        
        server = SageMakerTgiServing()
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
        
        model_data, env_vars = server._upload_tgi_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            jumpstart=False,
            s3_model_data_url="s3://bucket/prefix",
            image="test-image",
            env_vars={"CUSTOM": "var"},
            should_upload_artifacts=True
        )
        
        self.assertIsNotNone(model_data)
        mock_uploader.upload.assert_called_once()


class TestUpdateEnvVars(unittest.TestCase):
    """Test _update_env_vars function."""

    def test_update_env_vars_with_none(self):
        """Test _update_env_vars with None input."""
        from sagemaker.serve.model_server.tgi.server import _update_env_vars
        
        result = _update_env_vars(None)
        self.assertIn("HF_HOME", result)
        self.assertIn("HUGGINGFACE_HUB_CACHE", result)

    def test_update_env_vars_with_custom_vars(self):
        """Test _update_env_vars with custom variables."""
        from sagemaker.serve.model_server.tgi.server import _update_env_vars
        
        custom_vars = {"CUSTOM_KEY": "custom_value"}
        result = _update_env_vars(custom_vars)
        
        self.assertIn("CUSTOM_KEY", result)
        self.assertIn("HF_HOME", result)
        self.assertEqual(result["CUSTOM_KEY"], "custom_value")


if __name__ == "__main__":
    unittest.main()
