"""Unit tests for smd prepare.py module."""

import unittest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import shutil


class TestSmdPrepare(unittest.TestCase):
    """Test SMD prepare module functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("sagemaker.serve.model_server.smd.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    @patch("shutil.copy2")
    def test_prepare_for_smd_with_inference_spec(self, mock_copy, mock_capture, mock_hash):
        """Test prepare_for_smd with InferenceSpec."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
        from sagemaker.serve.spec.inference_spec import InferenceSpec

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)

        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")

        mock_hash.return_value = "test-hash"
        mock_inference_spec = Mock(spec=InferenceSpec)

        with patch("builtins.open", mock_open(read_data=b"test data")):
            secret_key = prepare_for_smd(
                model_path=str(model_path),
                shared_libs=[],
                dependencies={},
                inference_spec=mock_inference_spec,
            )

        mock_inference_spec.prepare.assert_called_once_with(str(model_path))
        self.assertEqual(secret_key, "test-hash")

    @patch("os.rename")
    @patch("sagemaker.serve.model_server.smd.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    @patch("shutil.copy2")
    def test_prepare_for_smd_with_custom_orchestrator(
        self, mock_copy, mock_capture, mock_hash, mock_rename
    ):
        """Test prepare_for_smd with CustomOrchestrator."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
        from sagemaker.serve.spec.inference_base import CustomOrchestrator

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)

        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")

        mock_hash.return_value = "test-hash"
        mock_orchestrator = Mock(spec=CustomOrchestrator)

        with patch("builtins.open", mock_open(read_data=b"test data")):
            secret_key = prepare_for_smd(
                model_path=str(model_path),
                shared_libs=[],
                dependencies={},
                inference_spec=mock_orchestrator,
            )

        # Verify custom_execution_inference.py was copied and renamed
        mock_rename.assert_called_once()
        self.assertEqual(secret_key, "test-hash")

    @patch("sagemaker.serve.model_server.smd.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    @patch("shutil.copy2")
    def test_prepare_for_smd_with_shared_libs(self, mock_copy, mock_capture, mock_hash):
        """Test prepare_for_smd copies shared libraries."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)

        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test data")

        shared_lib = Path(self.temp_dir) / "lib.so"
        shared_lib.touch()

        mock_hash.return_value = "test-hash"

        with patch("builtins.open", mock_open(read_data=b"test data")):
            prepare_for_smd(
                model_path=str(model_path), shared_libs=[str(shared_lib)], dependencies={}
            )

        # Verify copy2 was called for shared lib
        self.assertTrue(any(str(shared_lib) in str(call) for call in mock_copy.call_args_list))

    def test_prepare_for_smd_invalid_dir(self):
        """Test prepare_for_smd raises exception for invalid directory."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd

        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()

        with self.assertRaises(Exception) as context:
            prepare_for_smd(model_path=str(file_path), shared_libs=[], dependencies={})
        self.assertIn("not a valid directory", str(context.exception))

    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    @patch("shutil.copy2")
    def test_prepare_for_smd_returns_hash_value(self, mock_copy, mock_capture):
        """Test prepare_for_smd returns a valid SHA-256 hash string."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)

        # Create a real serve.pkl file
        serve_pkl = code_dir / "serve.pkl"
        serve_pkl.write_bytes(b"test pickle data")

        result = prepare_for_smd(
            model_path=str(model_path), shared_libs=[], dependencies={}
        )

        # Verify the return value is a valid 64-character hex string (SHA-256)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
        # Verify it's a valid hex string
        int(result, 16)


if __name__ == "__main__":
    unittest.main()
