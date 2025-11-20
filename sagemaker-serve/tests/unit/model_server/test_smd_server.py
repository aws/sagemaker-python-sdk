"""Unit tests for smd server.py module."""

import unittest
from unittest.mock import Mock, patch


class TestSageMakerSmdServer(unittest.TestCase):
    """Test SageMakerSmdServer class."""

    @patch('sagemaker.serve.model_server.smd.server._is_s3_uri')
    def test_upload_smd_artifacts_with_s3_path(self, mock_is_s3):
        """Test _upload_smd_artifacts with S3 path."""
        from sagemaker.serve.model_server.smd.server import SageMakerSmdServer
        
        server = SageMakerSmdServer()
        mock_is_s3.return_value = True
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_smd_artifacts(
            model_path="s3://bucket/model",
            sagemaker_session=mock_session,
            secret_key="test-key",
            should_upload_artifacts=False
        )
        
        self.assertEqual(s3_path, "s3://bucket/model")
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        self.assertEqual(env_vars["SAGEMAKER_SERVE_SECRET_KEY"], "test-key")
        self.assertIn("SAGEMAKER_INFERENCE_CODE_DIRECTORY", env_vars)

    @patch('sagemaker.serve.model_server.smd.server.upload')
    @patch('sagemaker.serve.model_server.smd.server.determine_bucket_and_prefix')
    @patch('sagemaker.serve.model_server.smd.server.parse_s3_url')
    @patch('sagemaker.serve.model_server.smd.server.fw_utils')
    @patch('sagemaker.serve.model_server.smd.server._is_s3_uri')
    def test_upload_smd_artifacts_uploads_to_s3(self, mock_is_s3, mock_fw_utils,
                                                  mock_parse, mock_determine, mock_upload):
        """Test _upload_smd_artifacts uploads to S3."""
        from sagemaker.serve.model_server.smd.server import SageMakerSmdServer
        
        server = SageMakerSmdServer()
        mock_is_s3.return_value = False
        mock_parse.return_value = ("bucket", "prefix")
        mock_determine.return_value = ("bucket", "code_prefix")
        mock_upload.return_value = "s3://bucket/code_prefix/model.tar.gz"
        
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_smd_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            secret_key="test-key",
            s3_model_data_url="s3://bucket/prefix",
            image="test-image",
            should_upload_artifacts=True
        )
        
        self.assertEqual(s3_path, "s3://bucket/code_prefix/model.tar.gz")
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        self.assertIn("SAGEMAKER_INFERENCE_CODE", env_vars)
        mock_upload.assert_called_once()

    @patch('sagemaker.serve.model_server.smd.server._is_s3_uri')
    def test_upload_smd_artifacts_no_upload(self, mock_is_s3):
        """Test _upload_smd_artifacts without uploading."""
        from sagemaker.serve.model_server.smd.server import SageMakerSmdServer
        
        server = SageMakerSmdServer()
        mock_is_s3.return_value = False
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        
        s3_path, env_vars = server._upload_smd_artifacts(
            model_path="/local/model",
            sagemaker_session=mock_session,
            secret_key="test-key",
            should_upload_artifacts=False
        )
        
        self.assertIsNone(s3_path)
        self.assertIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)


if __name__ == "__main__":
    unittest.main()
