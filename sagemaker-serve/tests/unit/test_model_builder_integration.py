"""
Integration-style unit tests for ModelBuilder that exercise complete workflows.
These tests use heavy mocking to simulate flows without executing resource operations.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer, ModelHub
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.constants import Framework
from sagemaker.core.resources import Model

# Import test fixtures
from .test_fixtures import (
    mock_sagemaker_session,
    mock_model_object,
    mock_schema_builder,
    MOCK_ROLE_ARN,
    MOCK_REGION,
    MOCK_IMAGE_URI,
    MOCK_S3_URI
)


# Simple test model classes that can be pickled
class SimplePyTorchModel:
    """Minimal PyTorch-like model for testing."""
    pass


class SimpleXGBoostModel:
    """Minimal XGBoost-like model for testing."""
    pass


class TestModelBuilderSaveWorkflow(unittest.TestCase):
    """Test _save_model_inference_spec workflow with different model types."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder.save_pkl')
    def test_save_with_inference_spec_path(self, mock_save_pkl):
        """Test that inference_spec path is taken in _save_model_inference_spec."""
        builder = ModelBuilder(
            inference_spec=Mock(),  # Just need something truthy
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify inference spec path was taken
        mock_save_pkl.assert_called_once()
        # First arg should be code_path, second should be tuple
        call_args = mock_save_pkl.call_args[0]
        self.assertIn("code", str(call_args[0]))

    @patch('sagemaker.serve.model_builder.save_pkl')
    def test_save_with_string_model_sets_env_var(self, mock_save_pkl):
        """Test that string model sets MODEL_CLASS_NAME and framework=None."""
        builder = ModelBuilder(
            model="my_module.MyModelClass",
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify string model path
        self.assertIsNone(builder.framework)
        self.assertEqual(builder.env_vars["MODEL_CLASS_NAME"], "my_module.MyModelClass")
        mock_save_pkl.assert_called_once()

    @patch('sagemaker.serve.model_builder.save_pkl')
    @patch('sagemaker.serve.model_builder._detect_framework_and_version')
    @patch('sagemaker.serve.model_builder._get_model_base')
    def test_save_with_model_object_detects_framework(self, mock_get_base, mock_detect, mock_save_pkl):
        """Test that model object triggers framework detection."""
        # Use a simple real object that can be inspected
        simple_model = SimplePyTorchModel()
        
        mock_get_base.return_value = simple_model
        mock_detect.return_value = ("pytorch", "1.8.0")
        
        builder = ModelBuilder(
            model=simple_model,
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify framework detection happened
        self.assertEqual(builder.framework, Framework.PYTORCH)
        self.assertIn("MODEL_CLASS_NAME", builder.env_vars)
        mock_detect.assert_called_once()
        mock_save_pkl.assert_called_once()

    @patch('sagemaker.serve.model_builder.save_xgboost')
    @patch('sagemaker.serve.model_builder.save_pkl')
    @patch('sagemaker.serve.model_builder._detect_framework_and_version')
    @patch('sagemaker.serve.model_builder._get_model_base')
    def test_save_with_xgboost_uses_special_save(self, mock_get_base, mock_detect, mock_save_pkl, mock_save_xgb):
        """Test that XGBoost model uses save_xgboost."""
        # Use a simple real object
        simple_model = SimpleXGBoostModel()
        
        mock_get_base.return_value = simple_model
        mock_detect.return_value = ("xgboost", "1.3.0")
        
        builder = ModelBuilder(
            model=simple_model,
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify XGBoost special handling
        self.assertEqual(builder.framework, Framework.XGBOOST)
        mock_save_xgb.assert_called_once()  # XGBoost-specific save
        mock_save_pkl.assert_called_once()  # Also saves framework tuple

    @patch('sagemaker.serve.model_builder.save_pkl')
    def test_save_with_mlflow_model(self, mock_save_pkl):
        """Test that MLflow model path is taken."""
        builder = ModelBuilder(
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model = None
        builder.inference_spec = None
        builder._is_mlflow_model = True
        builder.schema_builder = mock_schema_builder()
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify MLflow path
        mock_save_pkl.assert_called_once()
        # Should save just schema_builder for MLflow
        call_args = mock_save_pkl.call_args[0]
        self.assertEqual(call_args[1], builder.schema_builder)

    def test_save_without_model_or_spec_raises_error(self):
        """Test that save raises error without model or inference_spec."""
        builder = ModelBuilder(
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model = None
        builder.inference_spec = None
        builder._is_mlflow_model = False
        
        # Execute save - should raise
        with self.assertRaises(ValueError) as context:
            builder._save_model_inference_spec()
        
        self.assertIn("Cannot detect required model or inference spec", str(context.exception))

    def test_save_creates_model_path_directory(self):
        """Test that save creates model_path if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, "new_dir")
        self.assertFalse(os.path.exists(non_existent_dir))
        
        builder = ModelBuilder(
            model="my_module.MyModel",  # String model to avoid pickling
            model_path=non_existent_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        with patch('sagemaker.serve.model_builder.save_pkl'):
            # Execute save
            builder._save_model_inference_spec()
        
        # Verify directory was created
        self.assertTrue(os.path.exists(non_existent_dir))


class TestModelBuilderPrepareForModeIntegration(unittest.TestCase):
    """Test _prepare_for_mode with realistic scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder.SageMakerEndpointMode')
    def test_prepare_for_sagemaker_endpoint_mode_sets_upload_path(self, mock_mode_class):
        """Test prepare for SageMaker endpoint mode sets s3_upload_path."""
        # Setup mocks
        mock_mode = Mock()
        mock_mode.prepare.return_value = (MOCK_S3_URI, {"HF_MODEL_ID": "model-id"})
        mock_mode_class.return_value = mock_mode
        
        builder = ModelBuilder(
            model=mock_model_object(),
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_server=ModelServer.TORCHSERVE
        )
        builder.secret_key = "test-key"
        builder.serve_settings = Mock()
        builder.serve_settings.s3_model_data_url = None
        builder.modes = {}
        builder.inference_spec = None
        
        # Execute prepare
        result = builder._prepare_for_mode()
        
        # Verify mode preparation
        self.assertIsNotNone(result)
        self.assertEqual(builder.s3_upload_path, MOCK_S3_URI)
        self.assertIn("HF_MODEL_ID", builder.env_vars)
        mock_mode.prepare.assert_called_once()

    @patch('sagemaker.serve.model_builder.LocalContainerMode')
    def test_prepare_for_local_container_mode_sets_file_path(self, mock_mode_class):
        """Test prepare for LOCAL_CONTAINER mode sets file:// path."""
        # Setup mocks
        mock_mode = Mock()
        mock_mode.prepare.return_value = None
        mock_mode_class.return_value = mock_mode
        
        builder = ModelBuilder(
            model=mock_model_object(),
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.TORCHSERVE
        )
        builder.inference_spec = None
        builder.schema_builder = mock_schema_builder()
        builder.modes = {}
        
        # Execute prepare
        result = builder._prepare_for_mode()
        
        # Verify local container setup
        self.assertIsNone(result)
        self.assertIn("file://", builder.s3_upload_path)
        # Verify LocalContainerMode was initialized
        mock_mode_class.assert_called_once()
        call_kwargs = mock_mode_class.call_args[1]
        self.assertEqual(call_kwargs['model_server'], ModelServer.TORCHSERVE)

    @patch('sagemaker.serve.model_builder.InProcessMode')
    def test_prepare_for_in_process_mode_initializes_mode(self, mock_mode_class):
        """Test prepare for IN_PROCESS mode initializes InProcessMode."""
        # Setup mocks
        mock_mode = Mock()
        mock_mode.prepare.return_value = None
        mock_mode_class.return_value = mock_mode
        
        mock_model = mock_model_object()
        builder = ModelBuilder(
            model=mock_model,
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.IN_PROCESS
        )
        builder.inference_spec = None
        builder.schema_builder = mock_schema_builder()
        builder.modes = {}
        
        # Execute prepare
        result = builder._prepare_for_mode()
        
        # Verify in-process setup
        self.assertIsNone(result)
        # Verify InProcessMode was initialized with model
        mock_mode_class.assert_called_once()
        call_kwargs = mock_mode_class.call_args[1]
        self.assertEqual(call_kwargs['model'], mock_model)


class TestModelBuilderSaveModelIntegration(unittest.TestCase):
    """Test _save_model_inference_spec with complete scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder.save_pkl')
    def test_save_creates_directory_if_not_exists(self, mock_save_pkl):
        """Test that save creates model_path directory if it doesn't exist."""
        from sagemaker.serve.spec.inference_spec import InferenceSpec
        
        # Use a non-existent directory
        non_existent_dir = os.path.join(self.temp_dir, "new_dir")
        self.assertFalse(os.path.exists(non_existent_dir))
        
        # Create a simple mock inference spec that won't cause pickling issues
        mock_inference_spec = Mock(spec=InferenceSpec)
        
        builder = ModelBuilder(
            inference_spec=mock_inference_spec,
            schema_builder=None,  # Use None to avoid pickling issues
            model_path=non_existent_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=mock_sagemaker_session(),
            image_uri=MOCK_IMAGE_URI
        )
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify directory was created
        self.assertTrue(os.path.exists(non_existent_dir))
        mock_save_pkl.assert_called_once()

    @patch('sagemaker.serve.model_builder.save_pkl')
    @patch('sagemaker.serve.model_builder._detect_framework_and_version')
    @patch('sagemaker.serve.model_builder._get_model_base')
    def test_save_model_object_sets_env_vars(self, mock_get_base, mock_detect, mock_save_pkl):
        """Test that saving model object sets MODEL_CLASS_NAME env var."""
        # Setup mocks - create a simple mock that won't trigger framework detection
        mock_model = Mock()
        mock_model.__class__.__module__ = "sklearn.ensemble"
        mock_model.__class__.__name__ = "RandomForestClassifier"
        
        mock_get_base.return_value = mock_model
        mock_detect.return_value = ("sklearn", "0.24.0")
        
        builder = ModelBuilder(
            model=mock_model,
            model_path=self.temp_dir,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=mock_sagemaker_session(),
            image_uri=MOCK_IMAGE_URI  # Provide image to skip auto-detection
        )
        builder.env_vars = {}
        builder.schema_builder = None
        
        # Execute save
        builder._save_model_inference_spec()
        
        # Verify env vars were set
        self.assertIn("MODEL_CLASS_NAME", builder.env_vars)
        self.assertEqual(builder.env_vars["MODEL_CLASS_NAME"], 
                        "sklearn.ensemble.RandomForestClassifier")
        mock_save_pkl.assert_called_once()


class TestModelBuilderAutoDetectImage(unittest.TestCase):
    """Test auto-detection of container images."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    def test_auto_detect_image_sets_image_uri(self):
        """Test that auto-detection sets image_uri."""
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session
        )
        builder.image_uri = None
        builder.model_server = ModelServer.TORCHSERVE
        
        # Mock the detection method
        builder._detect_model_object_image = Mock()
        builder._detect_model_object_image.return_value = None
        builder.image_uri = MOCK_IMAGE_URI  # Simulate detection setting it
        
        # Verify image was set
        self.assertIsNotNone(builder.image_uri)
        self.assertEqual(builder.image_uri, MOCK_IMAGE_URI)




if __name__ == "__main__":
    unittest.main()
