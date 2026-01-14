"""Unit tests for tensorflow_serving prepare.py module."""

import unittest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import shutil


class TestTensorflowServingPrepare(unittest.TestCase):
    """Test TensorFlow Serving prepare module functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._move_contents')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._get_saved_model_path_for_tensorflow_and_keras_flavor')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_tf_serving_success(self, mock_copy, mock_capture, mock_hash, 
                                              mock_get_saved, mock_move):
        """Test prepare_for_tf_serving creates structure successfully."""
        from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_hash.return_value = "test-hash"
        mock_get_saved.return_value = Path(self.temp_dir) / "saved_model"
        
        secret_key = prepare_for_tf_serving(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={}
        )
        
        self.assertEqual(secret_key, "")
        mock_capture.assert_called_once()
        mock_move.assert_called_once()

    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._get_saved_model_path_for_tensorflow_and_keras_flavor')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_tf_serving_no_saved_model(self, mock_copy, mock_capture, mock_hash,
                                                     mock_get_saved):
        """Test prepare_for_tf_serving raises error when SavedModel not found."""
        from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        mock_hash.return_value = "test-hash"
        mock_get_saved.return_value = None
        
        with self.assertRaises(ValueError) as context:
            prepare_for_tf_serving(
                model_path=str(model_path),
                shared_libs=[],
                dependencies={}
            )
        self.assertIn("SavedModel is not found", str(context.exception))

    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._move_contents')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._get_saved_model_path_for_tensorflow_and_keras_flavor')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._move_contents')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._get_saved_model_path_for_tensorflow_and_keras_flavor')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_tf_serving_with_shared_libs(self, mock_copy, mock_capture, mock_hash,
                                                       mock_get_saved, mock_move):
        """Test prepare_for_tf_serving copies shared libraries."""
        from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
        
        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")
        
        shared_lib = Path(self.temp_dir) / "lib.so"
        shared_lib.touch()
        
        mock_hash.return_value = "test-hash"
        mock_get_saved.return_value = Path(self.temp_dir) / "saved_model"
        
        prepare_for_tf_serving(
            model_path=str(model_path),
            shared_libs=[str(shared_lib)],
            dependencies={}
        )
        
        # Verify copy2 was called for shared lib
        self.assertTrue(any(str(shared_lib) in str(call) for call in mock_copy.call_args_list))

    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare._get_saved_model_path_for_tensorflow_and_keras_flavor')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash')
    @patch('sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies')
    @patch('shutil.copy2')
    def test_prepare_for_tf_serving_invalid_dir(self, mock_copy, mock_capture, mock_hash,
                                                  mock_get_saved):
        """Test prepare_for_tf_serving raises exception for invalid directory."""
        from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
        
        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()
        
        with self.assertRaises(Exception) as context:
            prepare_for_tf_serving(
                model_path=str(file_path),
                shared_libs=[],
                dependencies={}
            )
        self.assertIn("not a valid directory", str(context.exception))


if __name__ == "__main__":
    unittest.main()
