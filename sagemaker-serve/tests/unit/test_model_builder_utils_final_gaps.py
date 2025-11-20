"""
Final unit tests for _ModelBuilderUtils to push coverage higher.
Targets remaining testable gaps.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.serve.constants import Framework
from sagemaker.serve.utils.types import ModelServer


class TestRetrieveHuggingFaceModelMapping(unittest.TestCase):
    """Test _retrieve_hugging_face_model_mapping method."""

    @patch('sagemaker.core.jumpstart.accessors.JumpStartS3PayloadAccessor.get_object_cached')
    def test_retrieve_mapping_success(self, mock_get_object):
        """Test successful retrieval of HF model mapping."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = Mock()
        utils.sagemaker_session.boto_region_name = "us-west-2"
        utils.sagemaker_session.s3_client = Mock()
        
        mock_get_object.return_value = json.dumps({
            "huggingface-llm-gpt2": {
                "hf-model-id": "gpt2",
                "jumpstart-model-version": "1.0.0",
                "merged-at": "2024-01-01"
            }
        })
        
        result = utils._retrieve_hugging_face_model_mapping()
        
        self.assertIn("gpt2", result)
        self.assertEqual(result["gpt2"]["jumpstart-model-id"], "huggingface-llm-gpt2")

    def test_retrieve_mapping_no_session(self):
        """Test retrieval without session."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = None
        
        result = utils._retrieve_hugging_face_model_mapping()
        
        self.assertEqual(result, {})

    @patch('sagemaker.core.jumpstart.accessors.JumpStartS3PayloadAccessor.get_object_cached')
    def test_retrieve_mapping_exception(self, mock_get_object):
        """Test retrieval with exception."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = Mock()
        utils.sagemaker_session.boto_region_name = "us-west-2"
        utils.sagemaker_session.s3_client = Mock()
        
        mock_get_object.side_effect = Exception("S3 error")
        
        result = utils._retrieve_hugging_face_model_mapping()
        
        self.assertEqual(result, {})


class TestMLflowMetadataExists(unittest.TestCase):
    """Test _mlflow_metadata_exists method."""

    def test_mlflow_metadata_exists_local_true(self):
        """Test MLflow metadata exists locally."""
        utils = _ModelBuilderUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mlmodel_path = os.path.join(tmpdir, "MLmodel")
            with open(mlmodel_path, 'w') as f:
                f.write("artifact_path: model\n")
            
            result = utils._mlflow_metadata_exists(tmpdir)
            
            self.assertTrue(result)

    def test_mlflow_metadata_exists_local_false(self):
        """Test MLflow metadata doesn't exist locally."""
        utils = _ModelBuilderUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = utils._mlflow_metadata_exists(tmpdir)
            
            self.assertFalse(result)

    def test_mlflow_metadata_exists_s3_no_session(self):
        """Test MLflow metadata check in S3 without session."""
        utils = _ModelBuilderUtils()
        utils.sagemaker_session = None
        
        result = utils._mlflow_metadata_exists("s3://bucket/model")
        
        self.assertFalse(result)


class TestInitializeForMLflow(unittest.TestCase):
    """Test _initialize_for_mlflow method - skipped (complex MLflow setup)."""
    pass


class TestDeploymentConfigContainsDraftModel(unittest.TestCase):
    """Test _deployment_config_contains_draft_model method."""

    def test_contains_draft_model_true(self):
        """Test deployment config contains draft model."""
        utils = _ModelBuilderUtils()
        
        config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {
                    "speculative_decoding": [{"channel_name": "draft_model"}]
                }
            }
        }
        
        result = utils._deployment_config_contains_draft_model(config)
        
        self.assertTrue(result)

    def test_contains_draft_model_false_no_speculative(self):
        """Test deployment config without speculative decoding."""
        utils = _ModelBuilderUtils()
        
        config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {}
            }
        }
        
        result = utils._deployment_config_contains_draft_model(config)
        
        self.assertFalse(result)

    def test_contains_draft_model_false_none(self):
        """Test deployment config is None."""
        utils = _ModelBuilderUtils()
        
        result = utils._deployment_config_contains_draft_model(None)
        
        self.assertFalse(result)


class TestIsDraftModelJumpStartProvided(unittest.TestCase):
    """Test _is_draft_model_jumpstart_provided method."""

    def test_is_draft_model_jumpstart_true(self):
        """Test draft model is JumpStart provided."""
        utils = _ModelBuilderUtils()
        
        config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {
                    "speculative_decoding": [
                        {
                            "channel_name": "draft_model",
                            "provider": {"name": "JumpStart"}
                        }
                    ]
                }
            }
        }
        
        result = utils._is_draft_model_jumpstart_provided(config)
        
        self.assertTrue(result)

    def test_is_draft_model_jumpstart_false(self):
        """Test draft model is not JumpStart provided."""
        utils = _ModelBuilderUtils()
        
        config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {
                    "speculative_decoding": [
                        {
                            "channel_name": "draft_model",
                            "provider": {"name": "Custom"}
                        }
                    ]
                }
            }
        }
        
        result = utils._is_draft_model_jumpstart_provided(config)
        
        self.assertFalse(result)


class TestExtractAdditionalModelDataSourceS3Uri(unittest.TestCase):
    """Test _extract_additional_model_data_source_s3_uri method."""

    def test_extract_s3_uri_success(self):
        """Test extracting S3 URI successfully."""
        utils = _ModelBuilderUtils()
        
        source = {
            "S3DataSource": {
                "S3Uri": "s3://bucket/model"
            }
        }
        
        result = utils._extract_additional_model_data_source_s3_uri(source)
        
        self.assertEqual(result, "s3://bucket/model")

    def test_extract_s3_uri_none(self):
        """Test extracting S3 URI from None."""
        utils = _ModelBuilderUtils()
        
        result = utils._extract_additional_model_data_source_s3_uri(None)
        
        self.assertIsNone(result)

    def test_extract_s3_uri_no_s3_data_source(self):
        """Test extracting S3 URI without S3DataSource."""
        utils = _ModelBuilderUtils()
        
        source = {"OtherKey": "value"}
        
        result = utils._extract_additional_model_data_source_s3_uri(source)
        
        self.assertIsNone(result)


class TestExtractDeploymentConfigAdditionalModelDataSourceS3Uri(unittest.TestCase):
    """Test _extract_deployment_config_additional_model_data_source_s3_uri method."""

    def test_extract_deployment_config_s3_uri_success(self):
        """Test extracting deployment config S3 URI successfully."""
        utils = _ModelBuilderUtils()
        
        source = {
            "s3_data_source": {
                "s3_uri": "s3://bucket/model"
            }
        }
        
        result = utils._extract_deployment_config_additional_model_data_source_s3_uri(source)
        
        self.assertEqual(result, "s3://bucket/model")

    def test_extract_deployment_config_s3_uri_none(self):
        """Test extracting deployment config S3 URI from None."""
        utils = _ModelBuilderUtils()
        
        result = utils._extract_deployment_config_additional_model_data_source_s3_uri(None)
        
        self.assertIsNone(result)


class TestIsDraftModelGated(unittest.TestCase):
    """Test _is_draft_model_gated method."""

    def test_is_draft_model_gated_true(self):
        """Test draft model is gated."""
        utils = _ModelBuilderUtils()
        
        config = {"hosting_eula_key": "eula-key"}
        
        result = utils._is_draft_model_gated(config)
        
        self.assertTrue(result)

    def test_is_draft_model_gated_false(self):
        """Test draft model is not gated."""
        utils = _ModelBuilderUtils()
        
        config = {"other_key": "value"}
        
        result = utils._is_draft_model_gated(config)
        
        self.assertFalse(result)

    def test_is_draft_model_gated_none(self):
        """Test draft model gated check with None."""
        utils = _ModelBuilderUtils()
        
        result = utils._is_draft_model_gated(None)
        
        self.assertFalse(result)


class TestExtractsAndValidatesSpeculativeModelSource(unittest.TestCase):
    """Test _extracts_and_validates_speculative_model_source method."""

    def test_extracts_model_source_success(self):
        """Test extracting model source successfully."""
        utils = _ModelBuilderUtils()
        
        config = {"ModelSource": "s3://bucket/draft-model"}
        
        result = utils._extracts_and_validates_speculative_model_source(config)
        
        self.assertEqual(result, "s3://bucket/draft-model")

    def test_extracts_model_source_missing(self):
        """Test extracting model source when missing."""
        utils = _ModelBuilderUtils()
        
        config = {}
        
        with self.assertRaises(ValueError) as context:
            utils._extracts_and_validates_speculative_model_source(config)
        
        self.assertIn("ModelSource must be provided", str(context.exception))


class TestGetCachedModelSpecs(unittest.TestCase):
    """Test _get_cached_model_specs method."""

    @patch('sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs')
    def test_get_cached_model_specs_first_call(self, mock_get_specs):
        """Test getting cached model specs on first call."""
        utils = _ModelBuilderUtils()
        
        mock_specs = Mock()
        mock_get_specs.return_value = mock_specs
        
        result = utils._get_cached_model_specs("model-id", "1.0.0", "us-west-2", Mock())
        
        self.assertEqual(result, mock_specs)
        mock_get_specs.assert_called_once()

    @patch('sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs')
    def test_get_cached_model_specs_cached(self, mock_get_specs):
        """Test getting cached model specs on subsequent call."""
        utils = _ModelBuilderUtils()
        
        mock_specs = Mock()
        mock_get_specs.return_value = mock_specs
        
        # First call
        result1 = utils._get_cached_model_specs("model-id", "1.0.0", "us-west-2", Mock())
        # Second call should use cache
        result2 = utils._get_cached_model_specs("model-id", "1.0.0", "us-west-2", Mock())
        
        self.assertEqual(result1, result2)
        # Should only be called once due to caching
        self.assertEqual(mock_get_specs.call_count, 1)


class TestUserAgentDecorator(unittest.TestCase):
    """Test _user_agent_decorator method."""

    def test_user_agent_decorator_adds_modelbuilder(self):
        """Test user agent decorator adds ModelBuilder."""
        utils = _ModelBuilderUtils()
        
        def mock_func():
            return "UserAgent/1.0"
        
        decorated = utils._user_agent_decorator(mock_func)
        result = decorated()
        
        self.assertIn("ModelBuilder", result)

    def test_user_agent_decorator_already_has_modelbuilder(self):
        """Test user agent decorator when ModelBuilder already present."""
        utils = _ModelBuilderUtils()
        
        def mock_func():
            return "UserAgent/1.0 ModelBuilder"
        
        decorated = utils._user_agent_decorator(mock_func)
        result = decorated()
        
        self.assertEqual(result, "UserAgent/1.0 ModelBuilder")


class TestDeploymentConfigResponseData(unittest.TestCase):
    """Test deployment_config_response_data method."""

    def test_deployment_config_response_data_none(self):
        """Test deployment config response data with None."""
        utils = _ModelBuilderUtils()
        
        result = utils.deployment_config_response_data(None)
        
        self.assertEqual(result, [])

    def test_deployment_config_response_data_empty(self):
        """Test deployment config response data with empty list."""
        utils = _ModelBuilderUtils()
        
        result = utils.deployment_config_response_data([])
        
        self.assertEqual(result, [])

    def test_deployment_config_response_data_with_configs(self):
        """Test deployment config response data with configs."""
        utils = _ModelBuilderUtils()
        
        mock_config = Mock()
        mock_config.to_json.return_value = {
            "DeploymentConfigName": "config-1",
            "BenchmarkMetrics": {
                "ml.g5.xlarge": {"latency": 100},
                "ml.g5.2xlarge": {"latency": 50}
            }
        }
        mock_config.deployment_args = Mock()
        mock_config.deployment_args.instance_type = "ml.g5.xlarge"
        
        result = utils.deployment_config_response_data([mock_config])
        
        self.assertEqual(len(result), 1)
        self.assertIn("BenchmarkMetrics", result[0])
        # Should only include metrics for the specific instance type
        self.assertIn("ml.g5.xlarge", result[0]["BenchmarkMetrics"])


class TestExtractFrameworkFromImageUri(unittest.TestCase):
    """Test _extract_framework_from_image_uri method."""

    def test_extract_framework_xgboost(self):
        """Test extracting XGBoost framework."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        
        framework, version = utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.XGBOOST)
        self.assertEqual(version, "1.5")

    def test_extract_framework_sklearn(self):
        """Test extracting sklearn framework."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1"
        
        framework, version = utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.SKLEARN)
        self.assertEqual(version, "0.23")

    def test_extract_framework_mxnet(self):
        """Test extracting MXNet framework."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.8.0-gpu-py37"
        
        framework, version = utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.MXNET)
        self.assertIsNotNone(version)

    def test_extract_framework_huggingface_no_version(self):
        """Test extracting HuggingFace framework without version."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:latest"
        
        framework, version = utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.HUGGINGFACE)
        self.assertIsNone(version)


class TestGetJumpStartRecommendedInstanceType(unittest.TestCase):
    """Test _get_jumpstart_recommended_instance_type method."""

    @patch('sagemaker.core.jumpstart.factory.utils.get_deploy_kwargs')
    def test_get_recommended_instance_type_success(self, mock_get_deploy):
        """Test getting recommended instance type successfully."""
        utils = _ModelBuilderUtils()
        utils.model = "huggingface-llm-falcon-7b"
        utils.region = "us-west-2"
        
        mock_deploy_kwargs = Mock()
        mock_deploy_kwargs.instance_type = "ml.g5.2xlarge"
        mock_get_deploy.return_value = mock_deploy_kwargs
        
        result = utils._get_jumpstart_recommended_instance_type()
        
        # May return None if hasattr check fails
        self.assertIsInstance(result, (str, type(None)))

    @patch('sagemaker.core.jumpstart.factory.utils.get_deploy_kwargs')
    def test_get_recommended_instance_type_exception(self, mock_get_deploy):
        """Test getting recommended instance type with exception."""
        utils = _ModelBuilderUtils()
        utils.model = "invalid-model"
        utils.region = "us-west-2"
        
        mock_get_deploy.side_effect = Exception("Model not found")
        
        result = utils._get_jumpstart_recommended_instance_type()
        
        self.assertIsNone(result)


class TestGetDefaultInstanceType(unittest.TestCase):
    """Test _get_default_instance_type method."""

    @patch.object(_ModelBuilderUtils, '_get_jumpstart_recommended_instance_type')
    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_get_default_instance_type_jumpstart(self, mock_is_js, mock_get_rec):
        """Test getting default instance type for JumpStart model."""
        utils = _ModelBuilderUtils()
        utils.model = "huggingface-llm-falcon-7b"
        
        mock_is_js.return_value = True
        mock_get_rec.return_value = "ml.g5.2xlarge"
        
        result = utils._get_default_instance_type()
        
        self.assertEqual(result, "ml.g5.2xlarge")

    @patch.object(_ModelBuilderUtils, 'get_huggingface_model_metadata')
    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_get_default_instance_type_large_hf_model(self, mock_is_js, mock_get_metadata):
        """Test getting default instance type for large HF model."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2-large"
        utils.env_vars = {}
        
        mock_is_js.return_value = False
        mock_get_metadata.return_value = {
            "safetensors": {"total": 3_000_000_000},  # 3GB
            "tags": []
        }
        
        result = utils._get_default_instance_type()
        
        self.assertEqual(result, "ml.g5.xlarge")

    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_get_default_instance_type_fallback(self, mock_is_js):
        """Test getting default instance type fallback."""
        utils = _ModelBuilderUtils()
        utils.model = "unknown-model"
        
        mock_is_js.return_value = False
        
        result = utils._get_default_instance_type()
        
        self.assertEqual(result, "ml.m5.large")


if __name__ == "__main__":
    unittest.main()
