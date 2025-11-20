"""
Unit tests for ModelBuilder to improve coverage.
Targets specific uncovered lines from coverage report.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import tempfile

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.training.configs import Compute, Networking


class TestModelBuilderInit(unittest.TestCase):
    """Test ModelBuilder initialization."""

    def test_init_with_compute(self):
        """Test initialization with Compute config."""
        compute = Compute(instance_type="ml.m5.large", instance_count=2)
        
        mb = ModelBuilder(model=Mock(), compute=compute)
        
        self.assertEqual(mb.instance_type, "ml.m5.large")
        self.assertEqual(mb.instance_count, 2)

    def test_init_with_deprecated_params(self):
        """Test initialization with deprecated parameters."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mb = ModelBuilder(
                model=Mock(),
                shared_libs=["lib1.so"],
                dependencies={"custom": ["dep1"]},
                image_config={"key": "value"}
            )
            
            # Should have deprecation warnings
            self.assertTrue(any("deprecated" in str(warning.message).lower() for warning in w))


class TestGetClientTranslators(unittest.TestCase):
    """Test _get_client_translators method."""

    def test_get_client_translators_with_schema_builder(self):
        """Test getting translators with schema builder."""
        schema_builder = Mock()
        schema_builder.input_serializer = Mock()
        schema_builder.output_deserializer = Mock()
        
        mb = ModelBuilder(model=Mock(), schema_builder=schema_builder)
        
        serializer, deserializer = mb._get_client_translators()
        
        self.assertIsNotNone(serializer)
        self.assertIsNotNone(deserializer)

    def test_get_client_translators_no_schema(self):
        """Test getting translators without schema builder."""
        mb = ModelBuilder(model=Mock())
        mb.framework = "pytorch"
        mb.content_type = None
        mb.accept_type = None
        
        serializer, deserializer = mb._get_client_translators()
        
        self.assertIsNotNone(serializer)
        self.assertIsNotNone(deserializer)


class TestIsRepack(unittest.TestCase):
    """Test is_repack method."""

    def test_is_repack_true(self):
        """Test is_repack returns True."""
        mb = ModelBuilder(model=Mock())
        mb.source_dir = "/path/to/source"
        mb.entry_point = "inference.py"
        mb.key_prefix = None
        mb.git_config = None
        
        result = mb.is_repack()
        
        self.assertTrue(result)

    def test_is_repack_false_no_source(self):
        """Test is_repack returns False without source."""
        mb = ModelBuilder(model=Mock())
        mb.source_dir = None
        mb.entry_point = None
        
        result = mb.is_repack()
        
        self.assertFalse(result)


class TestEnableNetworkIsolation(unittest.TestCase):
    """Test enable_network_isolation method."""

    def test_enable_network_isolation_true(self):
        """Test network isolation enabled."""
        mb = ModelBuilder(model=Mock())
        mb._enable_network_isolation = True
        
        result = mb.enable_network_isolation()
        
        self.assertTrue(result)

    def test_enable_network_isolation_false(self):
        """Test network isolation disabled."""
        mb = ModelBuilder(model=Mock())
        mb._enable_network_isolation = False
        
        result = mb.enable_network_isolation()
        
        self.assertFalse(result)


class TestToString(unittest.TestCase):
    """Test to_string method."""

    def test_to_string_regular_object(self):
        """Test to_string with regular object."""
        mb = ModelBuilder(model=Mock())
        
        result = mb.to_string("test_string")
        
        self.assertEqual(result, "test_string")

    def test_to_string_with_number(self):
        """Test to_string with number."""
        mb = ModelBuilder(model=Mock())
        
        result = mb.to_string(123)
        
        self.assertEqual(result, "123")


class TestBuildValidations(unittest.TestCase):
    """Test _build_validations method."""

    def test_build_validations_passthrough_1p_image(self):
        """Test validations for 1P image passthrough."""
        mb = ModelBuilder(image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13")
        mb.model = None
        mb.inference_spec = None
        
        mb._build_validations()
        
        self.assertTrue(mb._passthrough)

    def test_build_validations_non_1p_image_no_model_server(self):
        """Test validations fail for non-1P image without model_server."""
        mb = ModelBuilder(
            image_uri="custom-registry.com/my-image:latest",
            model=Mock()
        )
        mb.model_server = None
        
        with self.assertRaises(ValueError) as context:
            mb._build_validations()
        
        self.assertIn("Model_server must be set", str(context.exception))


class TestBuildForPassthrough(unittest.TestCase):
    """Test _build_for_passthrough method."""

    @patch.object(ModelBuilder, '_create_model')
    def test_build_for_passthrough(self, mock_create):
        """Test building for passthrough."""
        mock_model = Mock()
        mock_create.return_value = mock_model
        
        mb = ModelBuilder(image_uri="test-image:latest")
        
        result = mb._build_for_passthrough()
        
        self.assertEqual(result, mock_model)
        self.assertIsNone(mb.s3_upload_path)


class TestBuildDefaultAsyncInferenceConfig(unittest.TestCase):
    """Test _build_default_async_inference_config method."""

    def test_build_default_async_config(self):
        """Test building default async inference config."""
        from sagemaker.core.inference_config import AsyncInferenceConfig
        
        mb = ModelBuilder(model=Mock())
        mb.model_name = "test-model"
        mb.sagemaker_session = Mock()
        mb.sagemaker_session.default_bucket = Mock(return_value="test-bucket")
        mb.sagemaker_session.default_bucket_prefix = "prefix"
        
        async_config = AsyncInferenceConfig()
        
        result = mb._build_default_async_inference_config(async_config)
        
        self.assertIsNotNone(result.output_path)
        self.assertIsNotNone(result.failure_path)


class TestResetBuildState(unittest.TestCase):
    """Test _reset_build_state method."""

    def test_reset_build_state(self):
        """Test resetting build state."""
        mb = ModelBuilder(model=Mock())
        mb.built_model = Mock()
        mb.secret_key = "test-key"
        mb.prepared_for_djl = True
        mb.modes = {}
        
        mb._reset_build_state()
        
        self.assertIsNone(mb.built_model)
        self.assertEqual(mb.secret_key, "")
        self.assertFalse(hasattr(mb, 'prepared_for_djl'))
        self.assertFalse(hasattr(mb, 'modes'))


class TestConfigureForTorchServe(unittest.TestCase):
    """Test configure_for_torchserve method."""

    def test_configure_for_torchserve(self):
        """Test configuring for TorchServe."""
        mb = ModelBuilder(model=Mock())
        
        result = mb.configure_for_torchserve(
            shared_libs=["lib1.so"],
            dependencies={"auto": True},
            image_config={"key": "value"}
        )
        
        self.assertEqual(result.model_server, ModelServer.TORCHSERVE)
        self.assertEqual(result.shared_libs, ["lib1.so"])


class TestDoesICExist(unittest.TestCase):
    """Test _does_ic_exist method."""

    def test_does_ic_exist_true(self):
        """Test IC exists."""
        mb = ModelBuilder(model=Mock())
        mb.sagemaker_session = Mock()
        mb.sagemaker_session.describe_inference_component = Mock(return_value={})
        
        result = mb._does_ic_exist("test-ic")
        
        self.assertTrue(result)

    def test_does_ic_exist_false(self):
        """Test IC doesn't exist."""
        from botocore.exceptions import ClientError
        
        mb = ModelBuilder(model=Mock())
        mb.sagemaker_session = Mock()
        error_response = {"Error": {"Message": "Could not find inference component"}}
        mb.sagemaker_session.describe_inference_component = Mock(
            side_effect=ClientError(error_response, "DescribeInferenceComponent")
        )
        
        result = mb._does_ic_exist("test-ic")
        
        self.assertFalse(result)


class TestDisplayBenchmarkMetrics(unittest.TestCase):
    """Test display_benchmark_metrics method."""

    def test_display_benchmark_metrics_non_string_model(self):
        """Test display benchmark metrics with non-string model."""
        mb = ModelBuilder(model=Mock())
        
        with self.assertRaises(ValueError) as context:
            mb.display_benchmark_metrics()
        
        self.assertIn("only supported for JumpStart", str(context.exception))


class TestSetDeploymentConfig(unittest.TestCase):
    """Test set_deployment_config method."""

    def test_set_deployment_config_non_string_model(self):
        """Test set deployment config with non-string model."""
        mb = ModelBuilder(model=Mock())
        
        with self.assertRaises(ValueError) as context:
            mb.set_deployment_config("config-1", "ml.g5.xlarge")
        
        self.assertIn("only supported for JumpStart", str(context.exception))


class TestGetDeploymentConfig(unittest.TestCase):
    """Test get_deployment_config method."""

    def test_get_deployment_config_non_string_model(self):
        """Test get deployment config with non-string model."""
        mb = ModelBuilder(model=Mock())
        
        with self.assertRaises(ValueError) as context:
            mb.get_deployment_config()
        
        self.assertIn("only supported for JumpStart", str(context.exception))

    def test_get_deployment_config_no_config_name(self):
        """Test get deployment config without config_name."""
        mb = ModelBuilder(model="test-model")
        mb.config_name = None
        
        with patch.object(mb, '_is_jumpstart_model_id', return_value=True):
            result = mb.get_deployment_config()
        
        self.assertIsNone(result)


class TestListDeploymentConfigs(unittest.TestCase):
    """Test list_deployment_configs method."""

    def test_list_deployment_configs_non_string_model(self):
        """Test list deployment configs with non-string model."""
        mb = ModelBuilder(model=Mock())
        
        with self.assertRaises(ValueError) as context:
            mb.list_deployment_configs()
        
        self.assertIn("only supported for JumpStart", str(context.exception))


class TestTransformer(unittest.TestCase):
    """Test transformer method."""

    def test_transformer_without_built_model(self):
        """Test transformer without built model."""
        mb = ModelBuilder(model=Mock())
        
        with self.assertRaises(ValueError) as context:
            mb.transformer(
                instance_count=1,
                instance_type="ml.m5.large"
            )
        
        self.assertIn("Must call build()", str(context.exception))


class TestDeployLocal(unittest.TestCase):
    """Test deploy_local method."""

    def test_deploy_local_wrong_mode(self):
        """Test deploy_local with wrong mode."""
        mb = ModelBuilder(model=Mock(), mode=Mode.SAGEMAKER_ENDPOINT)
        
        with self.assertRaises(ValueError) as context:
            mb.deploy_local()
        
        self.assertIn("only supports LOCAL_CONTAINER and IN_PROCESS", str(context.exception))


class TestFromJumpStartConfig(unittest.TestCase):
    """Test from_jumpstart_config class method."""

    def test_from_jumpstart_config_basic(self):
        """Test creating ModelBuilder from JumpStart config."""
        from sagemaker.core.jumpstart.configs import JumpStartConfig
        
        js_config = JumpStartConfig(
            model_id="test-model",
            model_version="1.0.0"
        )
        
        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole"
        )
        
        self.assertEqual(mb.model, "test-model")
        self.assertEqual(mb.model_version, "1.0.0")


if __name__ == "__main__":
    unittest.main()
