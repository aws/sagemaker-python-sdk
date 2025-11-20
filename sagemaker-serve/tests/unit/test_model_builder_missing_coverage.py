"""
Unit tests for untested lines in ModelBuilder based on coverage report.
Focuses on lines: 376-378, 416, 442-448, 461, 464, 472-476, 492-493, 516-518, 544, 556-569, etc.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


class TestModelBuilderMissingCoverage(unittest.TestCase):
    """Test untested lines in ModelBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-east-1"
        self.mock_session.default_bucket = Mock(return_value="test-bucket")
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.settings = Mock()
        self.mock_session.settings.include_jumpstart_tags = False
        self.mock_session.settings._local_download_dir = None
        
        mock_credentials = Mock()
        mock_credentials.access_key = "test-key"
        mock_credentials.secret_key = "test-secret"
        mock_credentials.token = None
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-east-1"
        self.mock_session.boto_session.get_credentials = Mock(return_value=mock_credentials)

    def test_create_session_with_region(self):
        """Test _create_session_with_region when region is set (line 376-378)."""
        with patch('sagemaker.serve.model_builder.Session') as mock_session_class:
            builder = ModelBuilder(
                model="test-model",
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=self.mock_session
            )
            builder.region = "us-west-2"
            session = builder._create_session_with_region()
            mock_session_class.assert_called_once()

    def test_warn_deprecated_shared_libs(self):
        """Test deprecation warning for shared_libs (line 416)."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder = ModelBuilder(
                model=Mock(),
                shared_libs=["lib1.so"],
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=self.mock_session
            )
            assert len(w) > 0
            assert "shared_libs" in str(w[0].message)

    def test_initialize_compute_no_instance_type(self):
        """Test _initialize_compute_config when no instance_type (lines 442-448)."""
        with patch.object(ModelBuilder, '_get_default_instance_type', return_value='ml.m5.large'):
            builder = ModelBuilder(
                model=Mock(),
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=self.mock_session
            )
            assert builder.instance_type == 'ml.m5.large'

    def test_initialize_network_config_with_subnets(self):
        """Test _initialize_network_config with subnets (line 461)."""
        from sagemaker.core.training.configs import Networking
        network = Mock()
        network.vpc_config = None
        network.subnets = ["subnet-123"]
        network.security_group_ids = ["sg-456"]
        network.enable_network_isolation = False
        
        builder = ModelBuilder(
            model=Mock(),
            network=network,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        assert builder.vpc_config is not None
        assert "Subnets" in builder.vpc_config

    def test_initialize_defaults_region_from_boto3(self):
        """Test _initialize_defaults region fallback to boto3 (lines 472-476)."""
        with patch('boto3.Session') as mock_boto_session:
            mock_boto_session.return_value.region_name = "eu-west-1"
            builder = ModelBuilder(
                model=Mock(),
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=None
            )
            # Region should be set from boto3 session

    def test_initialize_jumpstart_hub_arn_generation(self):
        """Test _initialize_jumpstart_config hub_arn generation (lines 492-493)."""
        with patch('sagemaker.core.jumpstart.hub.utils.generate_hub_arn_for_init_kwargs') as mock_gen:
            mock_gen.return_value = "arn:aws:sagemaker:us-east-1:123456789012:hub/test"
            builder = ModelBuilder(
                model=Mock(),
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=self.mock_session
            )
            builder.hub_name = "test-hub"
            builder._initialize_jumpstart_config()
            assert builder.hub_arn is not None

    def test_initialize_jumpstart_model_type_detection(self):
        """Test _initialize_jumpstart_config model type detection (lines 516-518)."""
        with patch('sagemaker.core.jumpstart.utils.validate_model_id_and_get_type') as mock_validate:
            from sagemaker.core.jumpstart.enums import JumpStartModelType
            mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
            
            builder = ModelBuilder(
                model="test-model-id",
                role_arn="arn:aws:iam::123456789012:role/test",
                sagemaker_session=self.mock_session
            )
            builder.model_version = None
            builder.hub_arn = None
            builder._initialize_jumpstart_config()
            mock_validate.assert_called()



    def test_get_client_translators_numpy(self):
        """Test _get_client_translators with numpy content type (line 622)."""
        builder = ModelBuilder(
            model=Mock(),
            content_type="application/x-npy",
            accept_type="application/json",
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        builder.framework = "pytorch"
        serializer, deserializer = builder._get_client_translators()
        assert serializer is not None
        assert deserializer is not None

    def test_get_client_translators_torch_tensor(self):
        """Test _get_client_translators with torch tensor (line 624)."""
        builder = ModelBuilder(
            model=Mock(),
            content_type="tensor/pt",
            accept_type="tensor/pt",
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        builder.framework = "pytorch"
        serializer, deserializer = builder._get_client_translators()
        assert serializer is not None
        assert deserializer is not None

    def test_build_validations_model_trainer_without_inference_spec(self):
        """Test _build_validations with ModelTrainer without InferenceSpec (line 730)."""
        from sagemaker.train.model_trainer import ModelTrainer
        mock_trainer = Mock(spec=ModelTrainer)
        mock_trainer._jumpstart_config = None
        
        builder = ModelBuilder(
            model=mock_trainer,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        assert "InferenceSpec is required" in str(context.exception)

    def test_build_validations_passthrough_1p_image(self):
        """Test _build_validations passthrough with 1P image (line 741)."""
        builder = ModelBuilder(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.8.0",
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        builder._build_validations()
        assert builder._passthrough is True

    def test_enable_network_isolation(self):
        """Test enable_network_isolation method (line 802)."""
        network = Mock()
        network.vpc_config = None
        network.subnets = []
        network.security_group_ids = []
        network.enable_network_isolation = True
        
        builder = ModelBuilder(
            model=Mock(),
            network=network,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        assert builder.enable_network_isolation() is True

    def test_convert_model_data_source_to_local(self):
        """Test _convert_model_data_source_to_local (line 827)."""
        mock_source = Mock()
        mock_source.s3_data_source = Mock()
        mock_source.s3_data_source.s3_uri = "s3://bucket/model.tar.gz"
        mock_source.s3_data_source.s3_data_type = "S3Prefix"
        mock_source.s3_data_source.compression_type = "Gzip"
        mock_source.s3_data_source.model_access_config = None
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        result = builder._convert_model_data_source_to_local(mock_source)
        assert result is not None
        assert "S3DataSource" in result

    def test_is_repack_with_model_trainer(self):
        """Test is_repack with ModelTrainer and InferenceSpec (line 903)."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.serve.spec.inference_spec import InferenceSpec
        
        mock_trainer = Mock(spec=ModelTrainer)
        mock_spec = Mock(spec=InferenceSpec)
        
        builder = ModelBuilder(
            model=mock_trainer,
            inference_spec=mock_spec,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        builder.source_dir = "/path/to/code"
        builder.entry_point = "inference.py"
        
        assert builder.is_repack() is False

    def test_to_string_with_pipeline_variable(self):
        """Test to_string with PipelineVariable (line 893)."""
        mock_pipeline_var = Mock()
        mock_pipeline_var.to_string = Mock(return_value="pipeline_value")
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        
        with patch('sagemaker.serve.model_builder.is_pipeline_variable', return_value=True):
            result = builder.to_string(mock_pipeline_var)
            assert result == "pipeline_value"


    def test_initialize_script_mode_with_source_code(self):
        """Test _initialize_script_mode_variables with source_code (line 556-569)."""
        source_code = Mock()
        source_code.entry_script = "inference.py"
        source_code.source_dir = "/path/to/code"
        source_code.requirements = None
        
        builder = ModelBuilder(
            model=Mock(),
            source_code=source_code,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        assert builder.entry_point == "inference.py"
        assert builder.source_dir == "/path/to/code"

    def test_get_source_code_env_vars(self):
        """Test _get_source_code_env_vars (line 862-883)."""
        source_code = Mock()
        source_code.entry_script = "inference.py"
        source_code.source_dir = "/local/path"
        
        builder = ModelBuilder(
            model=Mock(),
            source_code=source_code,
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        
        env_vars = builder._get_source_code_env_vars()
        assert "SAGEMAKER_PROGRAM" in env_vars
        assert env_vars["SAGEMAKER_PROGRAM"] == "inference.py"
        assert "SAGEMAKER_SUBMIT_DIRECTORY" in env_vars

    def test_build_default_async_inference_config(self):
        """Test _build_default_async_inference_config (line 776-802)."""
        from sagemaker.core.inference_config import AsyncInferenceConfig
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/test",
            sagemaker_session=self.mock_session
        )
        builder.model_name = "test-model"
        
        async_config = AsyncInferenceConfig()
        result = builder._build_default_async_inference_config(async_config)
        
        assert result.output_path is not None
        assert result.failure_path is not None
        assert "s3://" in result.output_path


if __name__ == "__main__":
    unittest.main()
