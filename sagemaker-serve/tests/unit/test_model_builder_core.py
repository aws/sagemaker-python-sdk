"""
Unit tests for ModelBuilder core functionality - initialization, validation, and build methods.
Focuses on increasing coverage for model_builder.py
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import tempfile
import os

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.resources import TrainingJob, Model
from sagemaker.core.training.configs import Compute, Networking, SourceCode


class TestModelBuilderInitialization(unittest.TestCase):
    """Test ModelBuilder initialization and __post_init__ logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.settings = Mock()
        self.mock_session.settings.include_jumpstart_tags = False
        
        mock_credentials = Mock()
        mock_credentials.access_key = "test-key"
        mock_credentials.secret_key = "test-secret"
        mock_credentials.token = None
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.get_credentials.return_value = mock_credentials
        self.mock_session.boto_session.region_name = "us-west-2"

    def test_initialization_with_compute_config(self):
        """Test initialization with Compute configuration."""
        compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
        
        builder = ModelBuilder(
            model=Mock(),
            compute=compute,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.instance_type, "ml.m5.xlarge")
        self.assertEqual(builder.instance_count, 2)

    def test_initialization_with_network_isolation(self):
        """Test initialization with network isolation enabled."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # By default network isolation should be False
        self.assertFalse(builder._enable_network_isolation)

    def test_initialization_creates_default_model_path(self):
        """Test that default model_path is created."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertIsNotNone(builder.model_path)
        self.assertIn("/tmp/sagemaker/model-builder/", builder.model_path)

    def test_initialization_with_model_metadata(self):
        """Test initialization with model_metadata."""
        metadata = {
            "HF_TASK": "text-generation",
            "MLFLOW_MODEL_PATH": "s3://bucket/model"
        }
        
        builder = ModelBuilder(
            model=Mock(),
            model_metadata=metadata,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.model_metadata, metadata)

    def test_initialization_sets_region_from_session(self):
        """Test that region is set from sagemaker_session."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.region, "us-west-2")

    @patch('sagemaker.core.helper.session_helper.get_execution_role')
    def test_initialization_gets_default_role(self, mock_get_role):
        """Test that default role is retrieved when not provided."""
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/DefaultRole"
        
        # Mock the session to return a proper role ARN
        self.mock_session.get_caller_identity_arn = Mock(return_value="arn:aws:iam::123456789012:role/DefaultRole")
        
        builder = ModelBuilder(
            model=Mock(),
            sagemaker_session=self.mock_session
        )
        
        # The role should be set from get_execution_role
        self.assertIsNotNone(builder.role_arn)

    def test_deprecated_parameters_warning(self):
        """Test that deprecated parameters trigger warnings."""
        with self.assertWarns(DeprecationWarning):
            builder = ModelBuilder(
                model=Mock(),
                shared_libs=["lib1.so"],
                role_arn="arn:aws:iam::123456789012:role/TestRole",
                sagemaker_session=self.mock_session
            )

    def test_initialization_with_content_and_accept_types(self):
        """Test initialization with content_type and accept_type."""
        builder = ModelBuilder(
            model=Mock(),
            content_type="application/json",
            accept_type="application/json",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.content_type, "application/json")
        self.assertEqual(builder.accept_type, "application/json")


class TestModelBuilderValidations(unittest.TestCase):
    """Test ModelBuilder validation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}

    def test_build_validations_model_trainer_without_inference_spec(self):
        """Test validation fails for non-JumpStart ModelTrainer without InferenceSpec."""
        mock_trainer = Mock(spec=ModelTrainer)
        mock_trainer._jumpstart_config = None
        
        builder = ModelBuilder(
            model=mock_trainer,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        
        self.assertIn("InferenceSpec is required", str(context.exception))

    def test_build_validations_model_and_inference_spec_conflict(self):
        """Test validation fails when both model and inference_spec are provided."""
        builder = ModelBuilder(
            model=Mock(),
            inference_spec=Mock(spec=InferenceSpec),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        
        self.assertIn("Can only set one", str(context.exception))

    @patch('sagemaker.serve.validations.check_image_uri.is_1p_image_uri')
    def test_build_validations_passthrough_with_1p_image(self, mock_is_1p):
        """Test passthrough mode with first-party image."""
        mock_is_1p.return_value = True
        
        builder = ModelBuilder(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.8.0-gpu-py3",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        builder._build_validations()
        
        self.assertTrue(builder._passthrough)

    @patch('sagemaker.serve.validations.check_image_uri.is_1p_image_uri')
    def test_build_validations_custom_image_requires_model_server(self, mock_is_1p):
        """Test validation fails for custom image without model_server."""
        mock_is_1p.return_value = False
        
        builder = ModelBuilder(
            model=Mock(),
            image_uri="custom-registry.com/my-image:latest",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        
        self.assertIn("Model_server must be set", str(context.exception))


class TestModelBuilderHelperMethods(unittest.TestCase):
    """Test ModelBuilder helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.config = {}

    def test_enable_network_isolation_true(self):
        """Test enable_network_isolation returns True when set."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder._enable_network_isolation = True
        
        self.assertTrue(builder.enable_network_isolation())

    def test_enable_network_isolation_false(self):
        """Test enable_network_isolation returns False when not set."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder._enable_network_isolation = False
        
        self.assertFalse(builder.enable_network_isolation())

    def test_convert_model_data_source_to_local_none(self):
        """Test _convert_model_data_source_to_local with None."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._convert_model_data_source_to_local(None)
        
        self.assertIsNone(result)

    def test_convert_model_data_source_to_local_with_s3_source(self):
        """Test _convert_model_data_source_to_local with S3 data source."""
        mock_source = Mock()
        mock_s3_source = Mock()
        mock_s3_source.s3_uri = "s3://bucket/model.tar.gz"
        mock_s3_source.s3_data_type = "S3Prefix"
        mock_s3_source.compression_type = "Gzip"
        mock_s3_source.model_access_config = None
        mock_source.s3_data_source = mock_s3_source
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._convert_model_data_source_to_local(mock_source)
        
        self.assertIsNotNone(result)
        self.assertIn("S3DataSource", result)
        self.assertEqual(result["S3DataSource"]["S3Uri"], "s3://bucket/model.tar.gz")

    def test_convert_additional_sources_to_local_none(self):
        """Test _convert_additional_sources_to_local with None."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._convert_additional_sources_to_local(None)
        
        self.assertIsNone(result)

    def test_convert_additional_sources_to_local_with_sources(self):
        """Test _convert_additional_sources_to_local with sources."""
        mock_source = Mock()
        mock_source.channel_name = "extra-data"
        mock_s3_source = Mock()
        mock_s3_source.s3_uri = "s3://bucket/extra.tar.gz"
        mock_s3_source.s3_data_type = "S3Prefix"
        mock_s3_source.compression_type = "Gzip"
        mock_s3_source.model_access_config = None
        mock_source.s3_data_source = mock_s3_source
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._convert_additional_sources_to_local([mock_source])
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ChannelName"], "extra-data")

    def test_build_default_async_inference_config(self):
        """Test _build_default_async_inference_config sets default paths."""
        from sagemaker.core.inference_config import AsyncInferenceConfig
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.model_name = "test-model"
        
        async_config = AsyncInferenceConfig()
        result = builder._build_default_async_inference_config(async_config)
        
        self.assertIsNotNone(result.output_path)
        self.assertIn("s3://", result.output_path)
        self.assertIn("async-endpoint-outputs", result.output_path)
        self.assertIsNotNone(result.failure_path)
        self.assertIn("async-endpoint-failures", result.failure_path)


class TestModelBuilderScriptMode(unittest.TestCase):
    """Test ModelBuilder script mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.local_mode = False

    def test_script_mode_env_vars_with_uploaded_code(self):
        """Test _script_mode_env_vars with uploaded code."""
        from sagemaker.core import fw_utils
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.region = "us-west-2"
        builder.container_log_level = 20
        builder.env_vars = {}
        builder.uploaded_code = fw_utils.UploadedCode(
            s3_prefix="s3://bucket/code",
            script_name="inference.py"
        )
        builder.repacked_model_data = None
        builder._enable_network_isolation = False
        
        result = builder._script_mode_env_vars()
        
        self.assertEqual(result["SAGEMAKER_PROGRAM"], "inference.py")
        self.assertEqual(result["SAGEMAKER_SUBMIT_DIRECTORY"], "s3://bucket/code")
        self.assertEqual(result["SAGEMAKER_REGION"], "us-west-2")

    def test_script_mode_env_vars_with_repacked_model(self):
        """Test _script_mode_env_vars with repacked model data."""
        from sagemaker.core import fw_utils
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.region = "us-west-2"
        builder.container_log_level = 20
        builder.env_vars = {}
        builder.uploaded_code = fw_utils.UploadedCode(
            s3_prefix="s3://bucket/code",
            script_name="train.py"
        )
        builder.repacked_model_data = "s3://bucket/repacked.tar.gz"
        builder._enable_network_isolation = False
        
        result = builder._script_mode_env_vars()
        
        self.assertEqual(result["SAGEMAKER_PROGRAM"], "train.py")
        self.assertEqual(result["SAGEMAKER_SUBMIT_DIRECTORY"], "/opt/ml/model/code")

    def test_script_mode_env_vars_with_network_isolation(self):
        """Test _script_mode_env_vars with network isolation enabled."""
        from sagemaker.core import fw_utils
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.region = "us-west-2"
        builder.container_log_level = 20
        builder.env_vars = {}
        builder.uploaded_code = fw_utils.UploadedCode(
            s3_prefix="s3://bucket/code",
            script_name="inference.py"
        )
        builder.repacked_model_data = None
        builder._enable_network_isolation = True
        
        result = builder._script_mode_env_vars()
        
        self.assertEqual(result["SAGEMAKER_SUBMIT_DIRECTORY"], "/opt/ml/model/code")

    def test_script_mode_env_vars_with_entry_point_only(self):
        """Test _script_mode_env_vars with entry_point but no uploaded_code."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.region = "us-west-2"
        builder.container_log_level = 20
        builder.env_vars = {}
        builder.uploaded_code = None
        builder.entry_point = "inference.py"
        builder.source_dir = "/local/path"
        
        result = builder._script_mode_env_vars()
        
        self.assertEqual(result["SAGEMAKER_PROGRAM"], "inference.py")
        self.assertEqual(result["SAGEMAKER_SUBMIT_DIRECTORY"], "file:///local/path")


class TestModelBuilderPrepareForMode(unittest.TestCase):
    """Test ModelBuilder _prepare_for_mode method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_prepare_for_mode_sagemaker_endpoint_sets_upload_path(self):
        """Test _prepare_for_mode for SAGEMAKER_ENDPOINT mode sets s3_upload_path."""
        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.model_path = self.temp_dir
        
        # Initially s3_upload_path should be None
        self.assertIsNone(builder.s3_upload_path)

    @patch('sagemaker.serve.mode.local_container_mode.LocalContainerMode')
    def test_prepare_for_mode_local_container(self, mock_mode_class):
        """Test _prepare_for_mode for LOCAL_CONTAINER mode."""
        mock_mode = Mock()
        mock_mode.prepare.return_value = None
        mock_mode_class.return_value = mock_mode
        
        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.model_path = self.temp_dir
        builder.inference_spec = None
        builder.schema_builder = None
        builder.env_vars = {}
        builder.model_server = ModelServer.TORCHSERVE
        builder.modes = {}
        
        result = builder._prepare_for_mode()
        
        self.assertIsNone(result)
        self.assertIn("file://", builder.s3_upload_path)

    @patch('sagemaker.serve.mode.in_process_mode.InProcessMode')
    def test_prepare_for_mode_in_process(self, mock_mode_class):
        """Test _prepare_for_mode for IN_PROCESS mode."""
        mock_mode = Mock()
        mock_mode.prepare.return_value = None
        mock_mode_class.return_value = mock_mode
        
        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.IN_PROCESS,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.model_path = self.temp_dir
        builder.inference_spec = None
        builder.schema_builder = None
        builder.env_vars = {}
        builder.modes = {}
        
        result = builder._prepare_for_mode()
        
        self.assertIsNone(result)

    def test_prepare_for_mode_unsupported_mode(self):
        """Test _prepare_for_mode raises error for unsupported mode."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        builder.mode = "UNSUPPORTED_MODE"
        builder.modes = {}
        
        with self.assertRaises(ValueError) as context:
            builder._prepare_for_mode()
        
        self.assertIn("Unsupported deployment mode", str(context.exception))


if __name__ == "__main__":
    unittest.main()
