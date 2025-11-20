"""
Unit tests for advanced ModelBuilder V3 features.

Tests for:
- optimize() method with quantization, compilation, sharding
- Inference Component deployments
- Custom orchestrator deployments
- Sharded model deployments
- Advanced deployment configurations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model, Endpoint
from sagemaker.core.enums import EndpointType
from sagemaker.core.inference_config import (
    AsyncInferenceConfig,
    ServerlessInferenceConfig,
    ResourceRequirements
)


class TestModelBuilderOptimize(unittest.TestCase):
    """Test ModelBuilder.optimize() method for model optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._generate_optimized_core_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._optimize_for_hf')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    def test_optimize_with_quantization_config(
        self, mock_build_single, mock_optimize_hf, mock_generate_optimized, mock_is_js
    ):
        """Test optimize() with quantization configuration."""
        mock_is_js.return_value = False  # Not a JumpStart model
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_optimize_hf.return_value = {
            "DeploymentInstanceType": "ml.g5.xlarge",
            "OptimizationConfigs": [{"ModelQuantizationConfig": {}}]
        }
        mock_generate_optimized.return_value = mock_model
        
        self.mock_session.wait_for_optimization_job.return_value = {"Status": "Completed"}
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.xlarge"
        )
        
        quantization_config = {"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}}
        
        result = builder.optimize(
            instance_type="ml.g5.xlarge",
            quantization_config=quantization_config,
            output_path="s3://bucket/output"
        )
        
        self.assertIsNotNone(result)
        mock_optimize_hf.assert_called_once()
        self.mock_client.create_optimization_job.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._generate_optimized_core_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._optimize_for_hf')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    def test_optimize_with_sharding_config(
        self, mock_build_single, mock_optimize_hf, mock_generate_optimized, mock_is_js
    ):
        """Test optimize() with sharding configuration."""
        mock_is_js.return_value = False  # Not a JumpStart model
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_optimize_hf.return_value = {
            "DeploymentInstanceType": "ml.g5.12xlarge",
            "OptimizationConfigs": [{"ModelShardingConfig": {}}]
        }
        mock_generate_optimized.return_value = mock_model
        
        self.mock_session.wait_for_optimization_job.return_value = {"Status": "Completed"}
        
        builder = ModelBuilder(
            model="meta-llama/Llama-2-70b",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.12xlarge"
        )
        
        sharding_config = {
            "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "4"}
        }
        
        result = builder.optimize(
            instance_type="ml.g5.12xlarge",
            sharding_config=sharding_config,
            output_path="s3://bucket/output"
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(builder._is_sharded_model)
        mock_optimize_hf.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    def test_optimize_sharding_requires_tensor_parallel_degree(self, mock_is_js):
        """Test that sharding config requires OPTION_TENSOR_PARALLEL_DEGREE."""
        mock_is_js.return_value = False  # Not a JumpStart model
        
        builder = ModelBuilder(
            model="meta-llama/Llama-2-70b",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.12xlarge"
        )
        
        sharding_config = {"OverrideEnvironment": {}}  # Missing OPTION_TENSOR_PARALLEL_DEGREE
        
        with self.assertRaises(ValueError) as context:
            builder.optimize(
                instance_type="ml.g5.12xlarge",
                sharding_config=sharding_config,
                output_path="s3://bucket/output"
            )
        
        self.assertIn("OPTION_TENSOR_PARALLEL_DEGREE", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    def test_optimize_sharding_mutually_exclusive_with_other_optimizations(self, mock_is_js):
        """Test that sharding cannot be combined with other optimizations."""
        mock_is_js.return_value = False  # Not a JumpStart model
        
        builder = ModelBuilder(
            model="meta-llama/Llama-2-70b",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.12xlarge"
        )
        
        sharding_config = {
            "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "4"}
        }
        quantization_config = {"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}}
        
        with self.assertRaises(ValueError) as context:
            builder.optimize(
                instance_type="ml.g5.12xlarge",
                sharding_config=sharding_config,
                quantization_config=quantization_config,
                output_path="s3://bucket/output"
            )
        
        self.assertIn("mutually exclusive", str(context.exception))

    @patch('sagemaker.serve.model_builder._validate_optimization_configuration')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    def test_optimize_only_supported_in_sagemaker_endpoint_mode(self, mock_build_single, mock_validate):
        """Test that optimize() only works in SAGEMAKER_ENDPOINT mode."""
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_validate.return_value = None  # Skip validation
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            mode=Mode.LOCAL_CONTAINER
        )
        
        with self.assertRaises(ValueError) as context:
            builder.optimize(
                instance_type="ml.g5.xlarge",
                output_path="s3://bucket/output"
            )
        
        # The actual error message from the code
        self.assertIn("only supported in Sagemaker Endpoint Mode", str(context.exception))


class TestModelBuilderInferenceComponents(unittest.TestCase):
    """Test ModelBuilder with Inference Components."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_client = Mock()
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_resource_requirements_for_inference_component(self, mock_deploy):
        """Test deploy() with ResourceRequirements for inference component deployment."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.xlarge"
        )
        builder.built_model = Mock(spec=Model)
        
        resource_requirements = ResourceRequirements(
            requests={"num_cpus": 2, "memory": 8192},
            limits={}
        )
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=resource_requirements,
            wait=False
        )
        
        self.assertIsNotNone(result)
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['endpoint_type'], EndpointType.INFERENCE_COMPONENT_BASED)
        self.assertEqual(call_kwargs['resources'], resource_requirements)

    def test_deploy_with_resource_requirements_rejects_update_endpoint(self):
        """Test that update_endpoint is not supported with ResourceRequirements."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.xlarge"
        )
        builder.built_model = Mock(spec=Model)
        
        resource_requirements = ResourceRequirements(
            requests={"num_cpus": 2, "memory": 8192},
            limits={}
        )
        
        with self.assertRaises(ValueError) as context:
            builder.deploy(
                endpoint_name="test-endpoint",
                inference_config=resource_requirements,
                update_endpoint=True,
                wait=False
            )
        
        self.assertIn("not supported for inference component", str(context.exception))


class TestModelBuilderShardedModels(unittest.TestCase):
    """Test ModelBuilder with sharded models."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_client = Mock()
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_sharded_model_sets_flag(self):
        """Test that _is_sharded_model flag is set correctly."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.12xlarge"
        )
        
        # Initially should be False
        self.assertFalse(builder._is_sharded_model)
        
        # Set to True (as would happen after optimize with sharding_config)
        builder._is_sharded_model = True
        self.assertTrue(builder._is_sharded_model)

    def test_sharded_model_rejects_network_isolation(self):
        """Test that sharded models cannot use network isolation."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.g5.12xlarge"
        )
        builder.built_model = Mock(spec=Model)
        builder._is_sharded_model = True
        builder._enable_network_isolation = True
        
        with self.assertRaises(ValueError) as context:
            builder._deploy(
                endpoint_name="test-endpoint",
                instance_type="ml.g5.12xlarge",
                initial_instance_count=1,
                wait=False
            )
        
        self.assertIn("EnableNetworkIsolation", str(context.exception))
        self.assertIn("Fast Model Loading", str(context.exception))


class TestModelBuilderAsyncInference(unittest.TestCase):
    """Test ModelBuilder with async inference configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        
        self.mock_client = Mock()
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_build_default_async_inference_config_sets_output_path(self):
        """Test _build_default_async_inference_config sets default output path."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        builder.model_name = "test-model"
        
        async_config = AsyncInferenceConfig()
        
        result = builder._build_default_async_inference_config(async_config)
        
        self.assertIsNotNone(result.output_path)
        self.assertIn("s3://", result.output_path)
        self.assertIn("async-endpoint-outputs", result.output_path)

    def test_build_default_async_inference_config_sets_failure_path(self):
        """Test _build_default_async_inference_config sets default failure path."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        builder.model_name = "test-model"
        
        async_config = AsyncInferenceConfig()
        
        result = builder._build_default_async_inference_config(async_config)
        
        self.assertIsNotNone(result.failure_path)
        self.assertIn("s3://", result.failure_path)
        self.assertIn("async-endpoint-failures", result.failure_path)

    def test_build_default_async_inference_config_preserves_existing_paths(self):
        """Test _build_default_async_inference_config preserves user-provided paths."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        builder.model_name = "test-model"
        
        custom_output = "s3://my-bucket/custom-output"
        custom_failure = "s3://my-bucket/custom-failure"
        async_config = AsyncInferenceConfig(
            output_path=custom_output,
            failure_path=custom_failure
        )
        
        result = builder._build_default_async_inference_config(async_config)
        
        self.assertEqual(result.output_path, custom_output)
        self.assertEqual(result.failure_path, custom_failure)


class TestModelBuilderDeployValidations(unittest.TestCase):
    """Test ModelBuilder deploy validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_client = Mock()
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_deploy_requires_role_arn(self):
        """Test that deploy requires role_arn to be set."""
        # Create builder with explicit role first, then set to None
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        builder.built_model = Mock(spec=Model)
        # Now set role_arn to None to test validation
        builder.role_arn = None
        
        with self.assertRaises(ValueError) as context:
            builder._deploy(
                endpoint_name="test-endpoint",
                instance_type="ml.m5.xlarge",
                initial_instance_count=1,
                wait=False
            )
        
        self.assertIn("Role can not be null", str(context.exception))

    def test_deploy_requires_instance_type_for_non_serverless(self):
        """Test that deploy requires instance_type for non-serverless deployments."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"  # Set initially
        )
        builder.built_model = Mock(spec=Model)
        # Clear instance_type to test validation
        builder.instance_type = None
        
        with self.assertRaises(ValueError) as context:
            builder._deploy(
                endpoint_name="test-endpoint",
                initial_instance_count=1,  # Provide count but not type
                wait=False
            )
        
        self.assertIn("Must specify instance type", str(context.exception))

    def test_deploy_validates_async_inference_config_type(self):
        """Test that deploy validates AsyncInferenceConfig type."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        builder.built_model = Mock(spec=Model)
        
        with self.assertRaises(ValueError) as context:
            builder._deploy(
                endpoint_name="test-endpoint",
                instance_type="ml.m5.xlarge",
                initial_instance_count=1,
                async_inference_config={"not": "a config object"},
                wait=False
            )
        
        self.assertIn("AsyncInferenceConfig object", str(context.exception))

    def test_deploy_validates_serverless_inference_config_type(self):
        """Test that deploy validates ServerlessInferenceConfig type."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        builder.built_model = Mock(spec=Model)
        
        with self.assertRaises(ValueError) as context:
            builder._deploy(
                endpoint_name="test-endpoint",
                serverless_inference_config={"not": "a config object"},
                wait=False
            )
        
        self.assertIn("ServerlessInferenceConfig object", str(context.exception))


class TestModelBuilderNetworkIsolation(unittest.TestCase):
    """Test ModelBuilder network isolation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_enable_network_isolation_returns_true_when_set(self):
        """Test enable_network_isolation() returns True when enabled."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        # Directly set the internal attribute that enable_network_isolation checks
        builder._enable_network_isolation = True
        
        self.assertTrue(builder.enable_network_isolation())

    def test_enable_network_isolation_returns_false_when_not_set(self):
        """Test enable_network_isolation() returns False when not enabled."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        self.assertFalse(builder.enable_network_isolation())


if __name__ == '__main__':
    unittest.main()
