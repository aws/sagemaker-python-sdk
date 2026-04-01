"""Unit tests for multi_model_server prepare.py module."""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil


class TestMultiModelServerPrepare(unittest.TestCase):
    """Test Multi Model Server prepare module functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_server.multi_model_server.prepare._check_docker_disk_usage')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare._check_disk_space')
    def test_create_dir_structure_creates_directories(self, mock_disk_space, mock_docker_disk):
        """Test _create_dir_structure creates model and code directories."""
        from sagemaker.serve.model_server.multi_model_server.prepare import _create_dir_structure
        
        model_path = Path(self.temp_dir) / "model"
        model_path_obj, code_dir = _create_dir_structure(str(model_path))
        
        self.assertTrue(model_path.exists())
        self.assertTrue(code_dir.exists())
        self.assertEqual(code_dir, model_path / "code")
        mock_disk_space.assert_called_once()
        mock_docker_disk.assert_called_once()

    @patch('sagemaker.serve.model_server.multi_model_server.prepare._check_docker_disk_usage')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare._check_disk_space')
    def test_create_dir_structure_raises_on_file(self, mock_disk_space, mock_docker_disk):
        """Test _create_dir_structure raises ValueError when path is a file."""
        from sagemaker.serve.model_server.multi_model_server.prepare import _create_dir_structure
        
        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()
        
        with self.assertRaises(ValueError) as context:
            _create_dir_structure(str(file_path))
        self.assertIn("not a valid directory", str(context.exception))

    @patch('sagemaker.serve.model_server.multi_model_server.prepare._copy_jumpstart_artifacts')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure')
    def test_prepare_mms_js_resources(self, mock_create_dir, mock_copy_js):
        """Test prepare_mms_js_resources calls necessary functions."""
        from sagemaker.serve.model_server.multi_model_server.prepare import prepare_mms_js_resources
        
        mock_model_path = Path(self.temp_dir) / "model"
        mock_code_dir = mock_model_path / "code"
        mock_create_dir.return_value = (mock_model_path, mock_code_dir)
        mock_copy_js.return_value = ({"config": "data"}, True)
        
        result = prepare_mms_js_resources(
            model_path=str(mock_model_path),
            js_id="test-js-id",
            model_data="s3://bucket/model.tar.gz"
        )
        
        mock_create_dir.assert_called_once_with(str(mock_model_path))
        mock_copy_js.assert_called_once_with("s3://bucket/model.tar.gz", "test-js-id", mock_code_dir)
        self.assertEqual(result, ({"config": "data"}, True))

    @patch('builtins.input', return_value='')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_mms_creates_structure(self, mock_copy, mock_capture, mock_hash, mock_input):
        """Test prepare_for_mms creates directory structure and files."""
        from sagemaker.serve.model_server.multi_model_server.prepare import prepare_for_mms
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        # Create serve.pkl file
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        mock_inference_spec = Mock()
        
        secret_key = prepare_for_mms(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={},
            session=mock_session,
            image_uri="test-image",
            inference_spec=mock_inference_spec
        )
        
        # Should return None now (no longer returns secret key)
        self.assertIsNone(secret_key)
        mock_inference_spec.prepare.assert_called_once_with(str(model_path))
        mock_capture.assert_called_once()

    @patch('builtins.input', return_value='')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_mms_raises_on_invalid_dir(self, mock_copy, mock_capture, mock_hash, mock_input):
        """Test prepare_for_mms raises exception for invalid directory."""
        from sagemaker.serve.model_server.multi_model_server.prepare import prepare_for_mms
        
        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()
        
        mock_session = Mock()
        
        with self.assertRaises(Exception) as context:
            prepare_for_mms(
                model_path=str(file_path),
                shared_libs=[],
                dependencies={},
                session=mock_session,
                image_uri="test-image"
            )
        self.assertIn("not a valid directory", str(context.exception))

    @patch('builtins.input', return_value='')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.multi_model_server.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_mms_copies_shared_libs(self, mock_copy, mock_capture, mock_hash, mock_input):
        """Test prepare_for_mms copies shared libraries."""
        from sagemaker.serve.model_server.multi_model_server.prepare import prepare_for_mms
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        shared_lib = Path(self.temp_dir) / "lib.so"
        shared_lib.touch()
        
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        
        with patch('builtins.open', mock_open(read_data=b"test data")):
            prepare_for_mms(
                model_path=str(model_path),
                shared_libs=[str(shared_lib)],
                dependencies={},
                session=mock_session,
                image_uri="test-image"
            )
        
        # Verify copy2 was called for shared lib
        self.assertTrue(any(str(shared_lib) in str(call) for call in mock_copy.call_args_list))


if __name__ == "__main__":
    unittest.main()
