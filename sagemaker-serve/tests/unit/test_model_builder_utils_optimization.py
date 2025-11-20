"""
Unit tests for optimization-related methods in _ModelBuilderUtils.
Targets uncovered optimization and deployment config functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.core.enums import Tag


class TestExtractOptimizationConfigAndEnv(unittest.TestCase):
    """Test _extract_optimization_config_and_env method."""

    def test_extract_with_quantization_only(self):
        """Test extracting optimization config with quantization only."""
        utils = _ModelBuilderUtils()
        quantization_config = {"OverrideEnvironment": {"KEY": "value"}}
        
        opt_config, quant_env, comp_env, shard_env = utils._extract_optimization_config_and_env(
            quantization_config=quantization_config
        )
        
        self.assertIn("ModelQuantizationConfig", opt_config)
        self.assertEqual(quant_env, {"KEY": "value"})
        self.assertIsNone(comp_env)
        self.assertIsNone(shard_env)

    def test_extract_with_compilation_only(self):
        """Test extracting optimization config with compilation only."""
        utils = _ModelBuilderUtils()
        compilation_config = {"OverrideEnvironment": {"KEY": "value"}}
        
        opt_config, quant_env, comp_env, shard_env = utils._extract_optimization_config_and_env(
            compilation_config=compilation_config
        )
        
        self.assertIn("ModelCompilationConfig", opt_config)
        self.assertIsNone(quant_env)
        self.assertEqual(comp_env, {"KEY": "value"})
        self.assertIsNone(shard_env)

    def test_extract_with_sharding_only(self):
        """Test extracting optimization config with sharding only."""
        utils = _ModelBuilderUtils()
        sharding_config = {"OverrideEnvironment": {"KEY": "value"}}
        
        opt_config, quant_env, comp_env, shard_env = utils._extract_optimization_config_and_env(
            sharding_config=sharding_config
        )
        
        self.assertIn("ModelShardingConfig", opt_config)
        self.assertIsNone(quant_env)
        self.assertIsNone(comp_env)
        self.assertEqual(shard_env, {"KEY": "value"})

    def test_extract_with_all_configs(self):
        """Test extracting optimization config with all configs."""
        utils = _ModelBuilderUtils()
        quantization_config = {"OverrideEnvironment": {"Q": "q"}}
        compilation_config = {"OverrideEnvironment": {"C": "c"}}
        sharding_config = {"OverrideEnvironment": {"S": "s"}}
        
        opt_config, quant_env, comp_env, shard_env = utils._extract_optimization_config_and_env(
            quantization_config=quantization_config,
            compilation_config=compilation_config,
            sharding_config=sharding_config
        )
        
        self.assertIn("ModelQuantizationConfig", opt_config)
        self.assertIn("ModelCompilationConfig", opt_config)
        self.assertIn("ModelShardingConfig", opt_config)

    def test_extract_with_no_configs(self):
        """Test extracting optimization config with no configs."""
        utils = _ModelBuilderUtils()
        
        opt_config, quant_env, comp_env, shard_env = utils._extract_optimization_config_and_env()
        
        self.assertIsNone(opt_config)
        self.assertIsNone(quant_env)
        self.assertIsNone(comp_env)
        self.assertIsNone(shard_env)


class TestCustomSpeculativeDecoding(unittest.TestCase):
    """Test _custom_speculative_decoding method."""

    def test_custom_speculative_decoding_s3_uri(self):
        """Test custom speculative decoding with S3 URI."""
        utils = _ModelBuilderUtils()
        utils.additional_model_data_sources = []
        utils.env_vars = {}
        utils._tags = []
        
        config = {"ModelSource": "s3://bucket/draft-model"}
        
        utils._custom_speculative_decoding(config, False)
        
        self.assertIn("OPTION_SPECULATIVE_DRAFT_MODEL", utils.env_vars)
        self.assertEqual(len(utils.additional_model_data_sources), 1)

    def test_custom_speculative_decoding_local_path(self):
        """Test custom speculative decoding with local path."""
        utils = _ModelBuilderUtils()
        utils.additional_model_data_sources = []
        utils.env_vars = {}
        utils._tags = []
        
        config = {"ModelSource": "/local/path/to/model"}
        
        utils._custom_speculative_decoding(config, False)
        
        self.assertIn("OPTION_SPECULATIVE_DRAFT_MODEL", utils.env_vars)
        self.assertEqual(utils.env_vars["OPTION_SPECULATIVE_DRAFT_MODEL"], "/local/path/to/model")

    def test_custom_speculative_decoding_with_eula(self):
        """Test custom speculative decoding with EULA acceptance."""
        utils = _ModelBuilderUtils()
        utils.additional_model_data_sources = []
        utils.env_vars = {}
        utils._tags = []
        
        config = {"ModelSource": "s3://bucket/draft-model", "AcceptEula": True}
        
        utils._custom_speculative_decoding(config, False)
        
        self.assertEqual(len(utils.additional_model_data_sources), 1)
        self.assertIn("ModelAccessConfig", utils.additional_model_data_sources[0]["S3DataSource"])


class TestJumpStartSpeculativeDecoding(unittest.TestCase):
    """Test _jumpstart_speculative_decoding method - skipped (requires ModelBuilder context)."""
    pass


class TestOptimizeForHF(unittest.TestCase):
    """Test _optimize_for_hf method."""

    @patch.object(_ModelBuilderUtils, '_jumpstart_speculative_decoding')
    def test_optimize_for_hf_with_speculative_jumpstart(self, mock_js_spec):
        """Test HF optimization with JumpStart speculative decoding."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = Mock()
        utils.instance_type = "ml.g5.xlarge"
        utils.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
        utils.env_vars = {}
        utils.s3_upload_path = "s3://bucket/model"
        
        config = {"ModelProvider": "JumpStart", "ModelID": "draft-model"}
        
        result = utils._optimize_for_hf(
            output_path="s3://bucket/output",
            job_name="test-job",
            speculative_decoding_config=config
        )
        
        mock_js_spec.assert_called_once()

    @patch.object(_ModelBuilderUtils, '_custom_speculative_decoding')
    def test_optimize_for_hf_with_speculative_custom(self, mock_custom_spec):
        """Test HF optimization with custom speculative decoding."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = Mock()
        utils.instance_type = "ml.g5.xlarge"
        utils.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
        utils.env_vars = {}
        utils.s3_upload_path = "s3://bucket/model"
        
        config = {"ModelProvider": "Custom", "ModelSource": "s3://bucket/draft"}
        
        result = utils._optimize_for_hf(
            output_path="s3://bucket/output",
            job_name="test-job",
            speculative_decoding_config=config
        )
        
        mock_custom_spec.assert_called_once()

    @patch.object(_ModelBuilderUtils, '_optimize_prepare_for_hf')
    @patch.object(_ModelBuilderUtils, '_generate_model_source')
    def test_optimize_for_hf_with_quantization(self, mock_gen_source, mock_prepare):
        """Test HF optimization with quantization config."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = Mock()
        utils.instance_type = "ml.g5.xlarge"
        utils.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
        utils.env_vars = {}
        utils.s3_upload_path = "s3://bucket/model"
        
        mock_gen_source.return_value = {"S3": {"S3Uri": "s3://bucket/model"}}
        
        result = utils._optimize_for_hf(
            output_path="s3://bucket/output",
            job_name="test-job",
            quantization_config={"OverrideEnvironment": {}}
        )
        
        self.assertIsNotNone(result)
        self.assertIn("OptimizationConfigs", result)


class TestOptimizePrepareForHF(unittest.TestCase):
    """Test _optimize_prepare_for_hf method - skipped (requires ModelBuilder context)."""
    pass


class TestIsGatedModel(unittest.TestCase):
    """Test _is_gated_model method."""

    def test_is_gated_model_true(self):
        """Test gated model detection - true."""
        utils = _ModelBuilderUtils()
        utils.s3_upload_path = "s3://jumpstart-private-cache/model"
        
        result = utils._is_gated_model()
        
        self.assertTrue(result)

    def test_is_gated_model_false(self):
        """Test gated model detection - false."""
        utils = _ModelBuilderUtils()
        utils.s3_upload_path = "s3://jumpstart-cache/model"
        
        result = utils._is_gated_model()
        
        self.assertFalse(result)

    def test_is_gated_model_dict(self):
        """Test gated model detection with dict."""
        utils = _ModelBuilderUtils()
        utils.s3_upload_path = {"S3DataSource": {"S3Uri": "s3://jumpstart-private-cache/model"}}
        
        result = utils._is_gated_model()
        
        self.assertTrue(result)

    def test_is_gated_model_none(self):
        """Test gated model detection with None."""
        utils = _ModelBuilderUtils()
        utils.s3_upload_path = None
        
        result = utils._is_gated_model()
        
        self.assertFalse(result)


class TestSetJSDeploymentConfig(unittest.TestCase):
    """Test set_js_deployment_config method - skipped (requires ModelBuilder context)."""
    pass


class TestSetAdditionalModelSource(unittest.TestCase):
    """Test _set_additional_model_source method - skipped (requires ModelBuilder context)."""
    pass


class TestFindCompatibleDeploymentConfig(unittest.TestCase):
    """Test _find_compatible_deployment_config method - skipped (requires ModelBuilder context)."""
    pass


class TestGetNeuronModelEnvVars(unittest.TestCase):
    """Test _get_neuron_model_env_vars method."""

    @patch.object(_ModelBuilderUtils, '_get_cached_model_specs')
    def test_get_neuron_model_env_vars_success(self, mock_specs):
        """Test getting Neuron model env vars."""
        utils = _ModelBuilderUtils()
        utils.config_name = "config-1"
        utils.region = "us-west-2"
        utils.sagemaker_session = Mock()
        utils._metadata_configs = {
            "config-1": Mock(
                resolved_config={
                    "supported_inference_instance_types": ["ml.inf2.xlarge"],
                    "hosting_neuron_model_id": "neuron-model",
                    "hosting_neuron_model_version": "1.0.0"
                }
            )
        }
        
        mock_specs.return_value = Mock()
        mock_specs.return_value.to_json.return_value = {
            "hosting_env_vars": {"NEURON_KEY": "value"}
        }
        
        result = utils._get_neuron_model_env_vars("ml.g5.xlarge")
        
        self.assertEqual(result, {"NEURON_KEY": "value"})

    def test_get_neuron_model_env_vars_no_metadata(self):
        """Test getting Neuron model env vars without metadata."""
        utils = _ModelBuilderUtils()
        utils._metadata_configs = None
        
        result = utils._get_neuron_model_env_vars("ml.g5.xlarge")
        
        self.assertIsNone(result)


class TestSetOptimizationImageDefault(unittest.TestCase):
    """Test _set_optimization_image_default method - skipped (requires ModelBuilder context)."""
    pass


class TestGetDefaultVLLMImage(unittest.TestCase):
    """Test _get_default_vllm_image method - skipped (requires ModelBuilder context)."""
    pass


class TestGenerateOptimizedCoreModel(unittest.TestCase):
    """Test _generate_optimized_core_model method - skipped (requires ModelBuilder context)."""
    pass


class TestDeploymentConfigResponseData(unittest.TestCase):
    """Test deployment_config_response_data method."""

    def test_deployment_config_response_data_empty(self):
        """Test deployment config response data with empty list."""
        utils = _ModelBuilderUtils()
        
        result = utils.deployment_config_response_data(None)
        
        self.assertEqual(result, [])

    def test_deployment_config_response_data_with_configs(self):
        """Test deployment config response data with configs."""
        utils = _ModelBuilderUtils()
        
        mock_config = Mock()
        mock_config.to_json.return_value = {
            "DeploymentConfigName": "config-1",
            "BenchmarkMetrics": {
                "ml.g5.xlarge": {"latency": 100}
            }
        }
        mock_config.deployment_args = Mock(instance_type="ml.g5.xlarge")
        
        result = utils.deployment_config_response_data([mock_config])
        
        self.assertEqual(len(result), 1)
        self.assertIn("BenchmarkMetrics", result[0])


if __name__ == "__main__":
    unittest.main()
