"""
Unit tests for sagemaker.serve.model_format.mlflow.utils module.

Tests utility functions for MLflow model format handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import tempfile
import shutil
import yaml
from pathlib import Path

from sagemaker.serve.model_format.mlflow.utils import (
    _get_default_model_server_for_mlflow,
    _get_default_image_for_mlflow,
    _generate_mlflow_artifact_path,
    _get_all_flavor_metadata,
    _get_framework_version_from_requirements,
    _get_deployment_flavor,
    _get_python_version_from_parsed_mlflow_model_file,
    _download_s3_artifacts,
    _copy_directory_contents,
    _select_container_for_mlflow_model,
    _validate_input_for_mlflow,
    _get_saved_model_path_for_tensorflow_and_keras_flavor,
    _move_contents
)
from sagemaker.serve.utils.types import ModelServer


class TestGetDefaultModelServerForMlflow(unittest.TestCase):
    """Test _get_default_model_server_for_mlflow function."""

    def test_tensorflow_flavor_returns_tensorflow_serving(self):
        """Test that tensorflow flavor returns TENSORFLOW_SERVING."""
        result = _get_default_model_server_for_mlflow("tensorflow")
        self.assertEqual(result, ModelServer.TENSORFLOW_SERVING)

    def test_keras_flavor_returns_tensorflow_serving(self):
        """Test that keras flavor returns TENSORFLOW_SERVING."""
        result = _get_default_model_server_for_mlflow("keras")
        self.assertEqual(result, ModelServer.TENSORFLOW_SERVING)

    def test_pytorch_flavor_returns_torchserve(self):
        """Test that pytorch flavor returns TORCHSERVE."""
        result = _get_default_model_server_for_mlflow("pytorch")
        self.assertEqual(result, ModelServer.TORCHSERVE)

    def test_sklearn_flavor_returns_torchserve(self):
        """Test that sklearn flavor returns TORCHSERVE."""
        result = _get_default_model_server_for_mlflow("sklearn")
        self.assertEqual(result, ModelServer.TORCHSERVE)

    def test_xgboost_flavor_returns_torchserve(self):
        """Test that xgboost flavor returns TORCHSERVE."""
        result = _get_default_model_server_for_mlflow("xgboost")
        self.assertEqual(result, ModelServer.TORCHSERVE)


class TestGetDefaultImageForMlflow(unittest.TestCase):
    """Test _get_default_image_for_mlflow function."""

    @patch('sagemaker.serve.model_format.mlflow.utils.image_uris')
    def test_get_default_image_success(self, mock_image_uris):
        """Test successful retrieval of default image."""
        mock_image_uris.retrieve.return_value = "123456789.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-cpu-py39"
        
        result = _get_default_image_for_mlflow("3.9.0", "us-east-1", "ml.m5.xlarge")
        
        self.assertIn("pytorch-inference", result)
        mock_image_uris.retrieve.assert_called_once()
        call_args = mock_image_uris.retrieve.call_args[1]
        self.assertEqual(call_args['framework'], 'pytorch')
        self.assertEqual(call_args['region'], 'us-east-1')
        self.assertEqual(call_args['py_version'], 'py39')

    @patch('sagemaker.serve.model_format.mlflow.utils.image_uris')
    def test_get_default_image_python_38(self, mock_image_uris):
        """Test image retrieval for Python 3.8."""
        mock_image_uris.retrieve.return_value = "123456789.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-cpu-py38"
        
        result = _get_default_image_for_mlflow("3.8.10", "us-west-2", "ml.t2.medium")
        
        call_args = mock_image_uris.retrieve.call_args[1]
        self.assertEqual(call_args['py_version'], 'py38')

    @patch('sagemaker.serve.model_format.mlflow.utils.image_uris')
    def test_get_default_image_failure_raises_error(self, mock_image_uris):
        """Test that ValueError is raised when image cannot be retrieved."""
        mock_image_uris.retrieve.side_effect = ValueError("No image found")
        
        with self.assertRaises(ValueError) as context:
            _get_default_image_for_mlflow("3.11.0", "us-east-1", "ml.m5.xlarge")
        
        self.assertIn("Unable to find default image", str(context.exception))


class TestGenerateMlflowArtifactPath(unittest.TestCase):
    """Test _generate_mlflow_artifact_path function."""

    def test_generate_artifact_path_success(self):
        """Test successful artifact path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_file = os.path.join(tmpdir, "MLmodel")
            with open(artifact_file, 'w') as f:
                f.write("test content")
            
            result = _generate_mlflow_artifact_path(tmpdir, "MLmodel")
            
            self.assertEqual(result, artifact_file)
            self.assertTrue(os.path.isfile(result))

    def test_generate_artifact_path_file_not_found(self):
        """Test that FileNotFoundError is raised when artifact doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as context:
                _generate_mlflow_artifact_path(tmpdir, "nonexistent.txt")
            
            self.assertIn("does not exist", str(context.exception))


class TestGetAllFlavorMetadata(unittest.TestCase):
    """Test _get_all_flavor_metadata function."""

    def test_get_flavor_metadata_success(self):
        """Test successful parsing of MLmodel file."""
        mlmodel_content = """
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.8.10
  sklearn:
    pickled_model: model.pkl
    sklearn_version: 1.0.2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mlmodel_content)
            f.flush()
            
            try:
                result = _get_all_flavor_metadata(f.name)
                
                self.assertIn('python_function', result)
                self.assertIn('sklearn', result)
                self.assertEqual(result['python_function']['python_version'], '3.8.10')
            finally:
                os.unlink(f.name)

    def test_get_flavor_metadata_file_not_found(self):
        """Test that ValueError is raised when file doesn't exist."""
        with self.assertRaises(ValueError) as context:
            _get_all_flavor_metadata("/nonexistent/path/MLmodel")
        
        self.assertIn("File does not exist", str(context.exception))

    def test_get_flavor_metadata_missing_flavors_key(self):
        """Test that ValueError is raised when 'flavors' key is missing."""
        mlmodel_content = """
artifact_path: model
run_id: abc123
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mlmodel_content)
            f.flush()
            
            try:
                with self.assertRaises(ValueError) as context:
                    _get_all_flavor_metadata(f.name)
                
                self.assertIn("'flavors' key is missing", str(context.exception))
            finally:
                os.unlink(f.name)

    def test_get_flavor_metadata_invalid_yaml(self):
        """Test that ValueError is raised for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            
            try:
                with self.assertRaises(ValueError) as context:
                    _get_all_flavor_metadata(f.name)
                
                self.assertIn("Error parsing the file as YAML", str(context.exception))
            finally:
                os.unlink(f.name)


class TestGetFrameworkVersionFromRequirements(unittest.TestCase):
    """Test _get_framework_version_from_requirements function."""

    def test_get_version_with_double_equals(self):
        """Test version extraction with == operator."""
        requirements_content = """
numpy==1.21.0
scikit-learn==1.0.2
pandas==1.3.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            f.flush()
            
            try:
                result = _get_framework_version_from_requirements("sklearn", f.name)
                self.assertEqual(result, "1.0.2")
            finally:
                os.unlink(f.name)

    def test_get_version_with_greater_equals(self):
        """Test version extraction with >= operator."""
        requirements_content = """
tensorflow>=2.8.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            f.flush()
            
            try:
                result = _get_framework_version_from_requirements("tensorflow", f.name)
                self.assertEqual(result, "2.8.0")
            finally:
                os.unlink(f.name)

    def test_get_version_with_less_equals(self):
        """Test version extraction with <= operator."""
        requirements_content = """
torch<=1.13.1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            f.flush()
            
            try:
                result = _get_framework_version_from_requirements("pytorch", f.name)
                self.assertEqual(result, "1.13.1")
            finally:
                os.unlink(f.name)

    def test_get_version_not_found(self):
        """Test when framework is not in requirements."""
        requirements_content = """
numpy==1.21.0
pandas==1.3.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            f.flush()
            
            try:
                result = _get_framework_version_from_requirements("sklearn", f.name)
                self.assertIsNone(result)
            finally:
                os.unlink(f.name)

    def test_get_version_file_not_found(self):
        """Test that ValueError is raised when file doesn't exist."""
        with self.assertRaises(ValueError) as context:
            _get_framework_version_from_requirements("sklearn", "/nonexistent/requirements.txt")
        
        self.assertIn("File not found", str(context.exception))

    def test_get_version_unsupported_flavor(self):
        """Test with unsupported flavor returns None."""
        requirements_content = "numpy==1.21.0"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            f.flush()
            
            try:
                result = _get_framework_version_from_requirements("unsupported_flavor", f.name)
                self.assertIsNone(result)
            finally:
                os.unlink(f.name)


class TestGetDeploymentFlavor(unittest.TestCase):
    """Test _get_deployment_flavor function."""

    def test_get_deployment_flavor_with_sklearn(self):
        """Test deployment flavor extraction with sklearn."""
        flavor_metadata = {
            "python_function": {"python_version": "3.8.10"},
            "sklearn": {"sklearn_version": "1.0.2"}
        }
        
        result = _get_deployment_flavor(flavor_metadata)
        self.assertEqual(result, "sklearn")

    def test_get_deployment_flavor_with_pytorch(self):
        """Test deployment flavor extraction with pytorch."""
        flavor_metadata = {
            "python_function": {"python_version": "3.9.0"},
            "pytorch": {"pytorch_version": "1.13.1"}
        }
        
        result = _get_deployment_flavor(flavor_metadata)
        self.assertEqual(result, "pytorch")

    def test_get_deployment_flavor_pyfunc_only(self):
        """Test deployment flavor defaults to pyfunc when only pyfunc exists."""
        flavor_metadata = {
            "python_function": {"python_version": "3.8.10"}
        }
        
        result = _get_deployment_flavor(flavor_metadata)
        self.assertEqual(result, "python_function")

    def test_get_deployment_flavor_none_raises_error(self):
        """Test that ValueError is raised when flavor_metadata is None."""
        with self.assertRaises(ValueError) as context:
            _get_deployment_flavor(None)
        
        self.assertIn("Flavor metadata is not found", str(context.exception))

    def test_get_deployment_flavor_empty_dict_raises_error(self):
        """Test that ValueError is raised when flavor_metadata is empty."""
        with self.assertRaises(ValueError) as context:
            _get_deployment_flavor({})
        
        self.assertIn("Flavor metadata is not found", str(context.exception))


class TestGetPythonVersionFromParsedMlflowModelFile(unittest.TestCase):
    """Test _get_python_version_from_parsed_mlflow_model_file function."""

    def test_get_python_version_success(self):
        """Test successful Python version extraction."""
        parsed_metadata = {
            "python_function": {"python_version": "3.8.10"},
            "sklearn": {"sklearn_version": "1.0.2"}
        }
        
        result = _get_python_version_from_parsed_mlflow_model_file(parsed_metadata)
        self.assertEqual(result, "3.8.10")

    def test_get_python_version_missing_pyfunc_raises_error(self):
        """Test that ValueError is raised when python_function is missing."""
        parsed_metadata = {
            "sklearn": {"sklearn_version": "1.0.2"}
        }
        
        with self.assertRaises(ValueError) as context:
            _get_python_version_from_parsed_mlflow_model_file(parsed_metadata)
        
        self.assertIn("python_function cannot be found", str(context.exception))


class TestDownloadS3Artifacts(unittest.TestCase):
    """Test _download_s3_artifacts function."""

    @patch('sagemaker.serve.model_format.mlflow.utils.os.makedirs')
    def test_download_s3_artifacts_invalid_path(self, mock_makedirs):
        """Test that ValueError is raised for invalid S3 path."""
        mock_session = Mock()
        
        with self.assertRaises(ValueError) as context:
            _download_s3_artifacts("/local/path", "/dst/path", mock_session)
        
        self.assertIn("Invalid S3 path", str(context.exception))

    @patch('sagemaker.serve.model_format.mlflow.utils.os.makedirs')
    def test_download_s3_artifacts_success(self, mock_makedirs):
        """Test successful S3 artifact download."""
        mock_session = Mock()
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client
        
        # Mock paginator
        mock_paginator = Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "model/MLmodel"},
                    {"Key": "model/model.pkl"}
                ]
            }
        ]
        
        _download_s3_artifacts("s3://my-bucket/model", "/local/dst", mock_session)
        
        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        self.assertEqual(mock_s3_client.download_file.call_count, 2)


class TestCopyDirectoryContents(unittest.TestCase):
    """Test _copy_directory_contents function."""

    def test_copy_directory_contents_success(self):
        """Test successful directory copy."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dest_dir:
                # Create test files in source
                test_file = os.path.join(src_dir, "test.txt")
                with open(test_file, 'w') as f:
                    f.write("test content")
                
                sub_dir = os.path.join(src_dir, "subdir")
                os.makedirs(sub_dir)
                sub_file = os.path.join(sub_dir, "sub.txt")
                with open(sub_file, 'w') as f:
                    f.write("sub content")
                
                _copy_directory_contents(src_dir, dest_dir)
                
                # Verify files were copied
                self.assertTrue(os.path.exists(os.path.join(dest_dir, "test.txt")))
                self.assertTrue(os.path.exists(os.path.join(dest_dir, "subdir", "sub.txt")))

    def test_copy_directory_same_source_and_dest(self):
        """Test that no action is taken when source and dest are the same."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            # Should not raise error
            _copy_directory_contents(tmpdir, tmpdir)
            
            # File should still exist
            self.assertTrue(os.path.exists(test_file))


class TestValidateInputForMlflow(unittest.TestCase):
    """Test _validate_input_for_mlflow function."""

    def test_validate_torchserve_success(self):
        """Test validation passes for TORCHSERVE."""
        # Should not raise error
        _validate_input_for_mlflow(ModelServer.TORCHSERVE, "sklearn")

    def test_validate_tensorflow_serving_with_tensorflow_flavor(self):
        """Test validation passes for TENSORFLOW_SERVING with tensorflow flavor."""
        # Should not raise error
        _validate_input_for_mlflow(ModelServer.TENSORFLOW_SERVING, "tensorflow")

    def test_validate_tensorflow_serving_with_keras_flavor(self):
        """Test validation passes for TENSORFLOW_SERVING with keras flavor."""
        # Should not raise error
        _validate_input_for_mlflow(ModelServer.TENSORFLOW_SERVING, "keras")

    def test_validate_unsupported_model_server_raises_error(self):
        """Test that ValueError is raised for unsupported model server."""
        with self.assertRaises(ValueError) as context:
            _validate_input_for_mlflow(ModelServer.DJL_SERVING, "sklearn")
        
        self.assertIn("is currently not supported", str(context.exception))

    def test_validate_tensorflow_serving_with_wrong_flavor_raises_error(self):
        """Test that ValueError is raised for TF Serving with incompatible flavor."""
        with self.assertRaises(ValueError) as context:
            _validate_input_for_mlflow(ModelServer.TENSORFLOW_SERVING, "sklearn")
        
        self.assertIn("Tensorflow Serving is currently only supported", str(context.exception))


class TestGetSavedModelPathForTensorflowAndKerasFlavor(unittest.TestCase):
    """Test _get_saved_model_path_for_tensorflow_and_keras_flavor function."""

    def test_find_saved_model_pb_success(self):
        """Test successful finding of saved_model.pb."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model", "data")
            os.makedirs(model_dir)
            saved_model_file = os.path.join(model_dir, "saved_model.pb")
            with open(saved_model_file, 'w') as f:
                f.write("test")
            
            result = _get_saved_model_path_for_tensorflow_and_keras_flavor(tmpdir)
            
            self.assertEqual(result, model_dir)

    def test_find_saved_model_pb_not_found(self):
        """Test when saved_model.pb is not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_saved_model_path_for_tensorflow_and_keras_flavor(tmpdir)
            self.assertIsNone(result)

    def test_find_saved_model_pb_in_nested_directory(self):
        """Test finding saved_model.pb in deeply nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "a", "b", "c", "model")
            os.makedirs(nested_dir)
            saved_model_file = os.path.join(nested_dir, "saved_model.pb")
            with open(saved_model_file, 'w') as f:
                f.write("test")
            
            result = _get_saved_model_path_for_tensorflow_and_keras_flavor(tmpdir)
            
            self.assertEqual(result, nested_dir)


class TestMoveContents(unittest.TestCase):
    """Test _move_contents function."""

    def test_move_contents_success(self):
        """Test successful moving of directory contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            
            # Create test files
            test_file = os.path.join(src_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            sub_dir = os.path.join(src_dir, "subdir")
            os.makedirs(sub_dir)
            sub_file = os.path.join(sub_dir, "sub.txt")
            with open(sub_file, 'w') as f:
                f.write("sub content")
            
            _move_contents(src_dir, dest_dir)
            
            # Verify files were moved
            self.assertTrue(os.path.exists(os.path.join(dest_dir, "test.txt")))
            self.assertTrue(os.path.exists(os.path.join(dest_dir, "subdir", "sub.txt")))
            
            # Verify source directory was removed
            self.assertFalse(os.path.exists(src_dir))

    def test_move_contents_with_path_objects(self):
        """Test moving contents using Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            dest_dir = Path(tmpdir) / "dest"
            src_dir.mkdir()
            
            test_file = src_dir / "test.txt"
            test_file.write_text("test content")
            
            _move_contents(src_dir, dest_dir)
            
            self.assertTrue((dest_dir / "test.txt").exists())
            self.assertFalse(src_dir.exists())


class TestSelectContainerForMlflowModel(unittest.TestCase):
    """Test _select_container_for_mlflow_model function."""

    @patch('sagemaker.serve.model_format.mlflow.utils._get_framework_version_from_requirements')
    @patch('sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata')
    @patch('sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path')
    @patch('sagemaker.serve.model_format.mlflow.utils.image_uris')
    @patch('sagemaker.serve.model_format.mlflow.utils._cast_to_compatible_version')
    def test_select_container_pytorch_success(self, mock_cast, mock_image_uris, mock_gen_path,
                                                mock_get_metadata, mock_get_version):
        """Test successful container selection for PyTorch."""
        mock_gen_path.side_effect = ["/path/requirements.txt", "/path/MLmodel"]
        mock_get_metadata.return_value = {
            "python_function": {"python_version": "3.9.0"},
            "pytorch": {"pytorch_version": "1.13.1"}
        }
        mock_get_version.return_value = "1.13.1"
        mock_cast.return_value = ("1.13.1",)
        mock_image_uris.retrieve.return_value = "pytorch-inference:1.13.1-cpu-py39"
        
        result = _select_container_for_mlflow_model(
            "/path/to/model",
            "pytorch",
            "us-east-1",
            "ml.m5.xlarge"
        )
        
        self.assertIn("pytorch-inference", result)

    @patch('sagemaker.serve.model_format.mlflow.utils._get_framework_version_from_requirements')
    @patch('sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata')
    @patch('sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path')
    @patch('sagemaker.serve.model_format.mlflow.utils._get_default_image_for_mlflow')
    def test_select_container_unsupported_flavor_uses_default(self, mock_default_image, mock_gen_path,
                                                               mock_get_metadata, mock_get_version):
        """Test that unsupported flavor falls back to default image."""
        mock_gen_path.side_effect = ["/path/requirements.txt", "/path/MLmodel"]
        mock_get_metadata.return_value = {
            "python_function": {"python_version": "3.8.10"}
        }
        mock_default_image.return_value = "default-image:latest"
        
        result = _select_container_for_mlflow_model(
            "/path/to/model",
            "unsupported_flavor",
            "us-east-1",
            "ml.m5.xlarge"
        )
        
        self.assertEqual(result, "default-image:latest")

    @patch('sagemaker.serve.model_format.mlflow.utils._get_framework_version_from_requirements')
    @patch('sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata')
    @patch('sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path')
    def test_select_container_no_framework_version_raises_error(self, mock_gen_path,
                                                                  mock_get_metadata, mock_get_version):
        """Test that ValueError is raised when framework version cannot be detected."""
        mock_gen_path.side_effect = ["/path/requirements.txt", "/path/MLmodel"]
        mock_get_metadata.return_value = {
            "python_function": {"python_version": "3.9.0"},
            "sklearn": {"sklearn_version": "1.0.2"}
        }
        mock_get_version.return_value = None
        
        with self.assertRaises(ValueError) as context:
            _select_container_for_mlflow_model(
                "/path/to/model",
                "sklearn",
                "us-east-1",
                "ml.m5.xlarge"
            )
        
        self.assertIn("Unable to auto detect framework version", str(context.exception))


if __name__ == '__main__':
    unittest.main()
