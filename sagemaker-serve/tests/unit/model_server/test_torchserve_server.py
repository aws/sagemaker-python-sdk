"""Unit tests for torchserve server.py module."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path


class TestLocalTorchServe(unittest.TestCase):
    """Test LocalTorchServe class."""

    @patch('sagemaker.serve.model_server.torchserve.server.Path')
    def test_start_torch_serve(self, mock_path):
        """Test _start_torch_serve creates container."""
        from sagemaker.serve.model_server.torchserve.server import LocalTorchServe
        
        server = LocalTorchServe()
        mock_client = Mock()
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        mock_path_obj = Mock()
        mock_path.return_value = mock_path_obj
        
        server._start_torch_serve(
            client=mock_client,
            image="torchserve:latest",
            model_path="/path/to/model",
            env_vars={"CUSTOM_VAR": "value"}
        )
        
        self.assertEqual(server.container, mock_container)
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", call_kwargs["environment"])
        self.assertEqual(call_kwargs["environment"]["CUSTOM_VAR"], "value")

    @patch('sagemaker.serve.model_server.torchserve.server.requests.post')
    @patch('sagemaker.serve.model_server.torchserve.server.get_docker_host')
    def test_invoke_torch_serve_success(self, mock_get_host, mock_post):
        """Test _invoke_torch_serve successful request."""
        from sagemaker.serve.model_server.torchserve.server import LocalTorchServe
        
        server = LocalTorchServe()
        mock_get_host.return_value = "localhost"
        mock_response = Mock()
        mock_response.content = b'{"predictions": [0.1, 0.9]}'
        mock_post.return_value = mock_response
        
        result = server._invoke_torch_serve(
            request='{"data": [1, 2, 3]}',
            content_type="application/json",
            accept="application/json"
        )
        
        self.assertEqual(result, b'{"predictions": [0.1, 0.9]}')
        mock_post.assert_called_once()

    @patch('sagemaker.serve.model_server.torchserve.server.requests.post')
    @patch('sagemaker.serve.model_server.torchserve.server.get_docker_host')
    def test_invoke_torch_serve_failure(self, mock_get_host, mock_post):
        """Test _invoke_torch_serve handles errors."""
        from sagemaker.serve.model_server.torchserve.server import LocalTorchServe
        
        server = LocalTorchServe()
        mock_get_host.return_value = "localhost"
        mock_post.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception) as context:
            server._invoke_torch_serve(
                request='{"data": [1, 2, 3]}',
                content_type="application/json",
                accept="application/json"
            )
        self.assertIn("Unable to send request", str(context.exception))


class TestSageMakerTorchServe(unittest.TestCase):
    """Test SageMakerTorchServe class."""

    @patch('sagemaker.serve.model_server.torchserve.server._is_s3_uri')
    def test_upload_torchserve_artifacts_with_s3_path(self, mock_is_s3):
        """Test _upload_torchserve_artifacts with S3 path."""
        from sagemaker.serve.model_server.torchserve.server import SageMakerTorchServe
        
        server = SageMakerTorchServe()
        mock_is_s3.return_value = True
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_torchserve_artifacts(
            model_path="s3://bucket/model",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertEqual(s3_path, "s3://bucket/model")
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)

    @patch('sagemaker.serve.model_server.torchserve.server.upload')
    @patch('sagemaker.serve.model_server.torchserve.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.torchserve.server.parse_s3_url')
    @patch('sagemaker.serve.model_server.torchserve.server.fw_utils')
    @patch('sagemaker.serve.model_server.torchserve.server._is_s3_uri')
    def test_upload_torchserve_artifacts_uploads_to_s3(self, mock_is_s3, mock_fw_utils,
                                                         mock_parse, mock_determine, mock_upload):
        """Test _upload_torchserve_artifacts uploads to S3."""
        from sagemaker.serve.model_server.torchserve.server import SageMakerTorchServe
        
        server = SageMakerTorchServe()
        mock_is_s3.return_value = False
        mock_parse.return_value = ("bucket", "prefix")
        mock_determine.return_value = ("bucket", "code_prefix")
        mock_upload.return_value = "s3://bucket/code_prefix/model.tar.gz"
        
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_torchserve_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            s3_model_data_url="s3://bucket/prefix",
            image="test-image",
            should_upload_artifacts=True
        )
        
        self.assertEqual(s3_path, "s3://bucket/code_prefix/model.tar.gz")
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        mock_upload.assert_called_once()

    @patch('sagemaker.serve.model_server.torchserve.server._is_s3_uri')
    def test_upload_torchserve_artifacts_no_upload(self, mock_is_s3):
        """Test _upload_torchserve_artifacts without uploading."""
        from sagemaker.serve.model_server.torchserve.server import SageMakerTorchServe
        
        server = SageMakerTorchServe()
        mock_is_s3.return_value = False
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_torchserve_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            should_upload_artifacts=False
        )
        
        self.assertIsNone(s3_path)
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)


if __name__ == "__main__":
    unittest.main()
