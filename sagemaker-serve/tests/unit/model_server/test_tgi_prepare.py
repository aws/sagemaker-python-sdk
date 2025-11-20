"""Unit tests for tgi prepare.py module."""

import unittest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import shutil
import json


class TestTgiPrepare(unittest.TestCase):
    """Test TGI prepare module functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('tarfile.open')
    @patch('sagemaker.serve.model_server.tgi.prepare.custom_extractall_tarfile')
    def test_extract_js_resource(self, mock_extract, mock_tarfile):
        """Test _extract_js_resource extracts tarball."""
        from sagemaker.serve.model_server.tgi.prepare import _extract_js_resource
        
        js_model_dir = self.temp_dir
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        # Create a dummy tar file
        tar_path = Path(js_model_dir) / "infer-prepack-test-id.tar.gz"
        tar_path.touch()
        
        mock_tar = Mock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        
        _extract_js_resource(js_model_dir, code_dir, "test-id")
        
        mock_extract.assert_called_once_with(mock_tar, code_dir)

    @patch('sagemaker.serve.model_server.tgi.prepare.S3Downloader')
    @patch('sagemaker.serve.model_server.tgi.prepare._tmpdir')
    @patch('sagemaker.serve.model_server.tgi.prepare._extract_js_resource')
    def test_copy_jumpstart_artifacts_with_tarball(self, mock_extract, mock_tmpdir, mock_s3_downloader):
        """Test _copy_jumpstart_artifacts with tar.gz file."""
        from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
        
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        # Create config.json
        config_file = code_dir / "config.json"
        config_data = {"model_type": "gpt2"}
        config_file.write_text(json.dumps(config_data))
        
        mock_tmpdir.return_value.__enter__.return_value = self.temp_dir
        mock_downloader_instance = Mock()
        mock_s3_downloader.return_value = mock_downloader_instance
        
        result = _copy_jumpstart_artifacts(
            model_data="s3://bucket/model.tar.gz",
            js_id="test-id",
            code_dir=code_dir
        )
        
        self.assertEqual(result, (config_data, True))
        mock_downloader_instance.download.assert_called_once()

    @patch('sagemaker.serve.model_server.tgi.prepare.S3Downloader')
    def test_copy_jumpstart_artifacts_uncompressed(self, mock_s3_downloader):
        """Test _copy_jumpstart_artifacts with uncompressed data."""
        from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
        
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        config_file = code_dir / "config.json"
        config_data = {"model_type": "bert"}
        config_file.write_text(json.dumps(config_data))
        
        mock_downloader_instance = Mock()
        mock_s3_downloader.return_value = mock_downloader_instance
        
        result = _copy_jumpstart_artifacts(
            model_data="s3://bucket/model/",
            js_id="test-id",
            code_dir=code_dir
        )
        
        self.assertEqual(result, (config_data, True))

    @patch('sagemaker.serve.model_server.tgi.prepare.S3Downloader')
    def test_copy_jumpstart_artifacts_with_dict(self, mock_s3_downloader):
        """Test _copy_jumpstart_artifacts with dict model_data."""
        from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
        
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        config_file = code_dir / "config.json"
        config_file.write_text(json.dumps({"model_type": "t5"}))
        
        mock_downloader_instance = Mock()
        mock_s3_downloader.return_value = mock_downloader_instance
        
        model_data = {
            "S3DataSource": {
                "S3Uri": "s3://bucket/model/"
            }
        }
        
        result = _copy_jumpstart_artifacts(
            model_data=model_data,
            js_id="test-id",
            code_dir=code_dir
        )
        
        self.assertIsNotNone(result)
        mock_downloader_instance.download.assert_called_once_with("s3://bucket/model/", code_dir)

    @patch('sagemaker.serve.model_server.tgi.prepare.S3Downloader')
    def test_copy_jumpstart_artifacts_invalid_format(self, mock_s3_downloader):
        """Test _copy_jumpstart_artifacts raises error for invalid format."""
        from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
        
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        mock_downloader_instance = Mock()
        mock_s3_downloader.return_value = mock_downloader_instance
        
        with self.assertRaises(ValueError):
            _copy_jumpstart_artifacts(
                model_data={"invalid": "format"},
                js_id="test-id",
                code_dir=code_dir
            )

    @patch('sagemaker.serve.model_server.tgi.prepare.S3Downloader')
    def test_copy_jumpstart_artifacts_no_config(self, mock_s3_downloader):
        """Test _copy_jumpstart_artifacts when config.json doesn't exist."""
        from sagemaker.serve.model_server.tgi.prepare import _copy_jumpstart_artifacts
        
        code_dir = Path(self.temp_dir) / "code"
        code_dir.mkdir()
        
        mock_downloader_instance = Mock()
        mock_s3_downloader.return_value = mock_downloader_instance
        
        result = _copy_jumpstart_artifacts(
            model_data="s3://bucket/model/",
            js_id="test-id",
            code_dir=code_dir
        )
        
        self.assertEqual(result, (None, True))

    @patch('sagemaker.serve.model_server.tgi.prepare._check_docker_disk_usage')
    @patch('sagemaker.serve.model_server.tgi.prepare._check_disk_space')
    def test_create_dir_structure(self, mock_disk_space, mock_docker_disk):
        """Test _create_dir_structure creates directories."""
        from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
        
        model_path = Path(self.temp_dir) / "model"
        model_path_obj, code_dir = _create_dir_structure(str(model_path))
        
        self.assertTrue(model_path.exists())
        self.assertTrue(code_dir.exists())
        mock_disk_space.assert_called_once()
        mock_docker_disk.assert_called_once()

    @patch('sagemaker.serve.model_server.tgi.prepare._check_docker_disk_usage')
    @patch('sagemaker.serve.model_server.tgi.prepare._check_disk_space')
    def test_create_dir_structure_raises_on_file(self, mock_disk_space, mock_docker_disk):
        """Test _create_dir_structure raises ValueError for file path."""
        from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
        
        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()
        
        with self.assertRaises(ValueError):
            _create_dir_structure(str(file_path))

    @patch('sagemaker.serve.model_server.tgi.prepare._copy_jumpstart_artifacts')
    @patch('sagemaker.serve.model_server.tgi.prepare._create_dir_structure')
    def test_prepare_tgi_js_resources(self, mock_create_dir, mock_copy_js):
        """Test prepare_tgi_js_resources."""
        from sagemaker.serve.model_server.tgi.prepare import prepare_tgi_js_resources
        
        mock_model_path = Path(self.temp_dir) / "model"
        mock_code_dir = mock_model_path / "code"
        mock_create_dir.return_value = (mock_model_path, mock_code_dir)
        mock_copy_js.return_value = ({"config": "data"}, True)
        
        result = prepare_tgi_js_resources(
            model_path=str(mock_model_path),
            js_id="test-js-id",
            model_data="s3://bucket/model.tar.gz"
        )
        
        mock_create_dir.assert_called_once()
        mock_copy_js.assert_called_once()
        self.assertEqual(result, ({"config": "data"}, True))


if __name__ == "__main__":
    unittest.main()
