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
            prepare_for_smd(
                model_path=str(model_path),
                shared_libs=[],
                dependencies={},
                inference_spec=mock_inference_spec,
            )

        mock_inference_spec.prepare.assert_called_once_with(str(model_path))

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
            prepare_for_smd(
                model_path=str(model_path),
                shared_libs=[],
                dependencies={},
                inference_spec=mock_orchestrator,
            )

        # Verify custom_execution_inference.py was copied and renamed
        mock_rename.assert_called_once()

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

    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    def test_prepare_for_smd_returns_none(self, mock_capture):
        """prepare_for_smd records the digest in metadata.json and returns nothing.

        The secret key it used to return was removed for CVE-2026-8596; the digest is
        verified against metadata.json in the container instead. See issue #6200 --
        callers must not resurrect ``secret_key = prepare_for_smd(...)``.
        """
        import hashlib
        import json

        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "serve.pkl").write_bytes(b"test data")

        result = prepare_for_smd(model_path=str(model_path), shared_libs=[], dependencies={})

        self.assertIsNone(result)
        metadata = json.loads((code_dir / "metadata.json").read_bytes())
        self.assertEqual(metadata["sha256_hash"], hashlib.sha256(b"test data").hexdigest())

    @patch("sagemaker.serve.model_server.smd.prepare.capture_dependencies")
    def test_prepare_for_smd_writes_sagemaker_free_entrypoint(self, mock_capture):
        """The copied inference.py must not import the container's sagemaker package.

        Regression test for issue #6200: the container's bundled sagemaker version is set
        by the SMD image, not by the SDK that built the artifacts.
        """
        import ast

        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
        from sagemaker.serve.spec.inference_base import CustomOrchestrator

        model_path = Path(self.temp_dir) / "model"
        code_dir = model_path / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "serve.pkl").write_bytes(b"test data")

        class _Orchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data

        prepare_for_smd(
            model_path=str(model_path),
            shared_libs=[],
            dependencies={},
            inference_spec=_Orchestrator(),
        )

        tree = ast.parse((code_dir / "inference.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        self.assertEqual(
            sorted(n for n in imported if n == "sagemaker" or n.startswith("sagemaker.")), []
        )

    def test_prepare_for_smd_invalid_dir(self):
        """Test prepare_for_smd raises exception for invalid directory."""
        from sagemaker.serve.model_server.smd.prepare import prepare_for_smd

        file_path = Path(self.temp_dir) / "file.txt"
        file_path.touch()

        with self.assertRaises(Exception) as context:
            prepare_for_smd(model_path=str(file_path), shared_libs=[], dependencies={})
        self.assertIn("not a valid directory", str(context.exception))


if __name__ == "__main__":
    unittest.main()
