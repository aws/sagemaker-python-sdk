"""Unit tests for torchserve prepare.py module."""

import unittest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import shutil


class TestTorchServePrepare(unittest.TestCase):
    """Test TorchServe prepare module functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_server.torchserve.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.torchserve.prepare.capture_dependencies')
    @patch('sagemaker.serve.model_server.torchserve.prepare.is_1p_image_uri')
    @patch('shutil.copy2')
    def test_prepare_for_torchserve_standard_image(self, mock_copy, mock_is_1p, mock_capture,
                                                     mock_hash):
        """Test prepare_for_torchserve with standard image."""
        from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_is_1p.return_value = True
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        mock_inference_spec = Mock()
        
        secret_key = prepare_for_torchserve(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={},
            session=mock_session,
            image_uri="test-pytorch-image",
            inference_spec=mock_inference_spec
        )
        
        self.assertEqual(secret_key, "")
        mock_inference_spec.prepare.assert_called_once_with(str(model_path))
        mock_capture.assert_called_once()

    @patch('os.rename')
    @patch('sagemaker.serve.model_server.torchserve.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.torchserve.prepare.capture_dependencies')
    @patch('sagemaker.serve.model_server.torchserve.prepare.is_1p_image_uri')
    @patch('shutil.copy2')
    def test_prepare_for_torchserve_xgboost_image(self, mock_copy, mock_is_1p, mock_capture,
                                                    mock_hash, mock_rename):
        """Test prepare_for_torchserve with xgboost image."""
        from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_is_1p.return_value = True
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        
        secret_key = prepare_for_torchserve(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={},
            session=mock_session,
            image_uri="xgboost-image",
            inference_spec=None
        )
        
        self.assertEqual(secret_key, "")
        mock_rename.assert_called_once()
        mock_capture.assert_called_once()

    @patch('sagemaker.serve.model_server.torchserve.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.torchserve.prepare.capture_dependencies')
    @patch('sagemaker.serve.model_server.torchserve.prepare.is_1p_image_uri')
    @patch('shutil.copy2')
    def test_prepare_for_torchserve_with_shared_libs(self, mock_copy, mock_is_1p, mock_capture,
                                                       mock_hash):
        """Test prepare_for_torchserve copies shared libraries."""
        from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        shared_lib = Path(self.temp_dir) / "lib.so"
        shared_lib.touch()
        
        mock_is_1p.return_value = False
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        
        with patch('builtins.open', mock_open(read_data=b"test data")):
            prepare_for_torchserve(
                model_path=str(model_path),
                shared_libs=[str(shared_lib)],
                dependencies={},
                session=mock_session,
                image_uri="test-image"
            )
        
        # Verify copy2 was called for shared lib
        self.assertTrue(any(str(shared_lib) in str(call) for call in mock_copy.call_args_list))

    @patch('sagemaker.serve.model_server.torchserve.prepare.is_1p_image_uri')
    def test_prepare_for_torchserve_invalid_dir(self, mock_is_1p):
        """Test prepare_for_torchserve raises exception for invalid directory."""
        from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
        
        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()
        
        mock_session = Mock()
        
        with self.assertRaises(Exception) as context:
            prepare_for_torchserve(
                model_path=str(file_path),
                shared_libs=[],
                dependencies={},
                session=mock_session,
                image_uri="test-image"
            )
        self.assertIn("not a valid directory", str(context.exception))

    @patch('sagemaker.serve.model_server.torchserve.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.torchserve.prepare.capture_dependencies')
    @patch('sagemaker.serve.model_server.torchserve.prepare.is_1p_image_uri')
    @patch('shutil.copy2')
    def test_prepare_for_torchserve_no_inference_spec(self, mock_copy, mock_is_1p, mock_capture,
                                                        mock_hash):
        """Test prepare_for_torchserve without inference_spec."""
        from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_is_1p.return_value = False
        mock_hash.return_value = "test-hash"
        mock_session = Mock()
        
        secret_key = prepare_for_torchserve(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={},
            session=mock_session,
            image_uri="test-image",
            inference_spec=None
        )
        
        self.assertEqual(secret_key, "")


if __name__ == "__main__":
    unittest.main()
