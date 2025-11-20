"""Unit tests for tei server.py module."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path


class TestLocalTeiServing(unittest.TestCase):
    """Test LocalTeiServing class."""

    @patch('sagemaker.serve.model_server.tei.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tei.server.Path')
    @patch('sagemaker.serve.model_server.tei.server.DeviceRequest')
    def test_start_tei_serving(self, mock_device_req, mock_path, mock_update_env):
        """Test _start_tei_serving creates container."""
        from sagemaker.serve.model_server.tei.server import LocalTeiServing
        
        server = LocalTeiServing()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        mock_device_req.return_value = Mock()
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        
        server._start_tei_serving(
            client=mock_client,
            image="tei:latest",
            model_path="/path/to/model",
            secret_key="test-secret",
            env_vars={"CUSTOM_VAR": "value"}
        )
        
        self.assertEqual(server.container, mock_container)
        mock_client.containers.run.assert_called_once()

    @patch('sagemaker.serve.model_server.tei.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tei.server.Path')
    @patch('sagemaker.serve.model_server.tei.server.DeviceRequest')
    def test_start_tei_serving_adds_secret_key(self, mock_device_req, mock_path, mock_update_env):
        """Test _start_tei_serving adds secret key to env vars."""
        from sagemaker.serve.model_server.tei.server import LocalTeiServing
        
        server = LocalTeiServing()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value.joinpath.return_value = mock_path_obj
        mock_device_req.return_value = Mock()
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        
        env_vars = {"CUSTOM_VAR": "value"}
        server._start_tei_serving(
            client=mock_client,
            image="tei:latest",
            model_path="/path/to/model",
            secret_key="test-secret",
            env_vars=env_vars
        )
        
        # Verify secret key was added to env_vars
        self.assertEqual(env_vars["SAGEMAKER_SERVE_SECRET_KEY"], "test-secret")

    @patch('sagemaker.serve.model_server.tei.server.requests.post')
    @patch('sagemaker.serve.model_server.tei.server.get_docker_host')
    def test_invoke_tei_serving_success(self, mock_get_host, mock_post):
        """Test _invoke_tei_serving successful request."""
        from sagemaker.serve.model_server.tei.server import LocalTeiServing
        
        server = LocalTeiServing()
        mock_get_host.return_value = "localhost"
        mock_response = Mock()
        mock_response.content = b'{"embeddings": [[0.1, 0.2]]}'
        mock_post.return_value = mock_response
        
        result = server._invoke_tei_serving(
            request='{"inputs": "test text"}',
            content_type="application/json",
            accept="application/json"
        )
        
        self.assertEqual(result, b'{"embeddings": [[0.1, 0.2]]}')
        mock_post.assert_called_once()

    @patch('sagemaker.serve.model_server.tei.server.requests.post')
    @patch('sagemaker.serve.model_server.tei.server.get_docker_host')
    def test_invoke_tei_serving_failure(self, mock_get_host, mock_post):
        """Test _invoke_tei_serving handles errors."""
        from sagemaker.serve.model_server.tei.server import LocalTeiServing
        
        server = LocalTeiServing()
        mock_get_host.return_value = "localhost"
        mock_post.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception) as context:
            server._invoke_tei_serving(
                request='{"inputs": "test"}',
                content_type="application/json",
                accept="application/json"
            )
        self.assertIn("Unable to send request", str(context.exception))


class TestSageMakerTeiServing(unittest.TestCase):
    """Test SageMakerTeiServing class."""

    @patch('sagemaker.serve.model_server.tei.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tei.server._is_s3_uri')
    def test_upload_tei_artifacts_with_s3_path(self, mock_is_s3, mock_update_env):
        """Test _upload_tei_artifacts with S3 path."""
        from sagemaker.serve.model_server.tei.server import SageMakerTeiServing
        
        server = SageMakerTeiServing()
        mock_is_s3.return_value = True
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        mock_session = Mock()
        
        model_data, env_vars = server._upload_tei_artifacts(
            model_path="s3://bucket/model",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertIsNotNone(model_data)
        self.assertEqual(model_data["S3DataSource"]["S3Uri"], "s3://bucket/model/")

    @patch('sagemaker.serve.model_server.tei.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tei.server.S3Uploader')
    @patch('sagemaker.serve.model_server.tei.server.s3_path_join')
    @patch('sagemaker.serve.model_server.tei.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.tei.server.parse_s3_url')
    @patch('sagemaker.serve.model_server.tei.server.fw_utils')
    @patch('sagemaker.serve.model_server.tei.server._is_s3_uri')
    @patch('sagemaker.serve.model_server.tei.server.Path')
    def test_upload_tei_artifacts_uploads_to_s3(self, mock_path, mock_is_s3, mock_fw_utils,
                                                  mock_parse, mock_determine, mock_s3_join, 
                                                  mock_uploader, mock_update_env):
        """Test _upload_tei_artifacts uploads to S3."""
        from sagemaker.serve.model_server.tei.server import SageMakerTeiServing
        
        server = SageMakerTeiServing()
        mock_is_s3.return_value = False
        mock_parse.return_value = ("bucket", "prefix")
        mock_determine.return_value = ("bucket", "code_prefix")
        mock_s3_join.return_value = "s3://bucket/code_prefix/code"
        mock_uploader.upload.return_value = "s3://bucket/code_prefix/code"
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        
        mock_path_obj = Mock()
        mock_code_dir = Mock()
        mock_path_obj.joinpath.return_value = mock_code_dir
        mock_path.return_value = mock_path_obj
        
        mock_session = Mock()
        
        model_data, env_vars = server._upload_tei_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            s3_model_data_url="s3://bucket/prefix",
            image="test-image",
            env_vars={"CUSTOM": "var"},
            should_upload_artifacts=True
        )
        
        self.assertIsNotNone(model_data)
        mock_uploader.upload.assert_called_once()

    @patch('sagemaker.serve.model_server.tei.server._update_env_vars')
    @patch('sagemaker.serve.model_server.tei.server._is_s3_uri')
    def test_upload_tei_artifacts_no_upload(self, mock_is_s3, mock_update_env):
        """Test _upload_tei_artifacts without uploading."""
        from sagemaker.serve.model_server.tei.server import SageMakerTeiServing
        
        server = SageMakerTeiServing()
        mock_is_s3.return_value = False
        mock_update_env.return_value = {"HF_HOME": "/opt/ml/model/"}
        mock_session = Mock()
        
        model_data, env_vars = server._upload_tei_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertIsNone(model_data)


class TestUpdateEnvVars(unittest.TestCase):
    """Test _update_env_vars function."""

    def test_update_env_vars_with_none(self):
        """Test _update_env_vars with None input."""
        from sagemaker.serve.model_server.tei.server import _update_env_vars
        
        result = _update_env_vars(None)
        self.assertIn("HF_HOME", result)
        self.assertIn("HUGGINGFACE_HUB_CACHE", result)

    def test_update_env_vars_with_custom_vars(self):
        """Test _update_env_vars with custom variables."""
        from sagemaker.serve.model_server.tei.server import _update_env_vars
        
        custom_vars = {"CUSTOM_KEY": "custom_value"}
        result = _update_env_vars(custom_vars)
        
        self.assertIn("CUSTOM_KEY", result)
        self.assertIn("HF_HOME", result)
        self.assertEqual(result["CUSTOM_KEY"], "custom_value")


if __name__ == "__main__":
    unittest.main()
