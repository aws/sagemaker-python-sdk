"""
Additional unit tests for sagemaker.serve.model_builder_utils module.

Tests utility functions for ModelBuilder including:
- Session management
- Instance type detection
- Image URI extraction
- HuggingFace utilities
- Resource requirements
- MLflow utilities
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from typing import Optional

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.serve.constants import Framework
from sagemaker.serve.utils.types import ModelServer


class TestSessionManagement(unittest.TestCase):
    """Test session management utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()
        self.utils.sagemaker_session = None
        self.utils.region = "us-west-2"

    @patch('sagemaker.serve.model_builder_utils.LocalSession')
    def test_init_session_for_local_instance(self, mock_local_session):
        """Test session initialization for local instance type."""
        self.utils.instance_type = "local"
        
        self.utils._init_sagemaker_session_if_does_not_exist()
        
        mock_local_session.assert_called_once()
        self.assertIsNotNone(self.utils.sagemaker_session)

    @patch('sagemaker.serve.model_builder_utils.LocalSession')
    def test_init_session_for_local_gpu_instance(self, mock_local_session):
        """Test session initialization for local_gpu instance type."""
        self.utils.instance_type = "local_gpu"
        
        self.utils._init_sagemaker_session_if_does_not_exist()
        
        mock_local_session.assert_called_once()

    @patch('sagemaker.serve.model_builder_utils.Session')
    @patch('boto3.Session')
    def test_init_session_for_remote_instance(self, mock_boto3_session, mock_session):
        """Test session initialization for remote instance type."""
        self.utils.instance_type = "ml.m5.large"
        mock_boto_session = Mock()
        mock_boto3_session.return_value = mock_boto_session
        
        self.utils._init_sagemaker_session_if_does_not_exist()
        
        mock_boto3_session.assert_called_once_with(region_name="us-west-2")
        mock_session.assert_called_once()

    def test_init_session_does_not_override_existing(self):
        """Test that existing session is not overridden."""
        existing_session = Mock()
        self.utils.sagemaker_session = existing_session
        
        self.utils._init_sagemaker_session_if_does_not_exist()
        
        self.assertEqual(self.utils.sagemaker_session, existing_session)

    @patch('sagemaker.serve.model_builder_utils.Session')
    def test_init_session_with_instance_type_parameter(self, mock_session):
        """Test session initialization with instance_type parameter."""
        self.utils._init_sagemaker_session_if_does_not_exist(instance_type="ml.g5.xlarge")
        
        mock_session.assert_called_once()


class TestInstanceTypeDetection(unittest.TestCase):
    """Test instance type detection utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()
        self.utils.region = "us-east-1"

    @patch('sagemaker.serve.model_builder_utils.get_deploy_kwargs')
    def test_get_jumpstart_recommended_instance_type_success(self, mock_get_deploy_kwargs):
        """Test successful retrieval of JumpStart recommended instance type."""
        self.utils.model = "huggingface-llm-falcon-7b-bf16"
        mock_deploy_kwargs = Mock()
        mock_deploy_kwargs.instance_type = "ml.g5.2xlarge"
        mock_get_deploy_kwargs.return_value = mock_deploy_kwargs
        
        result = self.utils._get_jumpstart_recommended_instance_type()
        
        self.assertEqual(result, "ml.g5.2xlarge")

    @patch('sagemaker.serve.model_builder_utils.get_deploy_kwargs')
    def test_get_jumpstart_recommended_instance_type_no_recommendation(self, mock_get_deploy_kwargs):
        """Test when JumpStart has no recommended instance type."""
        self.utils.model = "some-model"
        mock_deploy_kwargs = Mock()
        mock_deploy_kwargs.instance_type = None
        mock_get_deploy_kwargs.return_value = mock_deploy_kwargs
        
        result = self.utils._get_jumpstart_recommended_instance_type()
        
        self.assertIsNone(result)

    @patch('sagemaker.serve.model_builder_utils.get_deploy_kwargs')
    def test_get_jumpstart_recommended_instance_type_exception(self, mock_get_deploy_kwargs):
        """Test exception handling in JumpStart instance type retrieval."""
        self.utils.model = "invalid-model"
        mock_get_deploy_kwargs.side_effect = Exception("Model not found")
        
        result = self.utils._get_jumpstart_recommended_instance_type()
        
        self.assertIsNone(result)

    def test_get_default_instance_type_fallback(self):
        """Test default instance type fallback."""
        self.utils.model = "some-model"
        self.utils._is_jumpstart_model_id = Mock(return_value=False)
        
        result = self.utils._get_default_instance_type()
        
        self.assertEqual(result, "ml.m5.large")


class TestImageURIExtraction(unittest.TestCase):
    """Test image URI extraction utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_extract_framework_from_pytorch_image(self):
        """Test framework extraction from PyTorch image URI."""
        self.utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.PYTORCH)
        self.assertEqual(version, "1.13.1")

    def test_extract_framework_from_tensorflow_image(self):
        """Test framework extraction from TensorFlow image URI."""
        self.utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.11.0-cpu"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.TENSORFLOW)
        self.assertEqual(version, "2.11.0")

    def test_extract_framework_from_xgboost_image(self):
        """Test framework extraction from XGBoost image URI."""
        self.utils.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.XGBOOST)
        self.assertEqual(version, "1.5")

    def test_extract_framework_from_sklearn_image(self):
        """Test framework extraction from scikit-learn image URI."""
        self.utils.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.SKLEARN)
        self.assertEqual(version, "1.0")

    def test_extract_framework_from_huggingface_image(self):
        """Test framework extraction from HuggingFace image URI."""
        self.utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        # HuggingFace images with pytorch in the name are detected as PyTorch
        self.assertEqual(framework, Framework.PYTORCH)
        self.assertEqual(version, "1.13.1")

    def test_extract_framework_from_mxnet_image(self):
        """Test framework extraction from MXNet image URI."""
        self.utils.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertEqual(framework, Framework.MXNET)
        self.assertEqual(version, "1.9.0")

    def test_extract_framework_no_image_uri(self):
        """Test framework extraction when no image URI is set."""
        self.utils.image_uri = None
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertIsNone(framework)
        self.assertIsNone(version)

    def test_extract_framework_unknown_image(self):
        """Test framework extraction from unknown image URI."""
        self.utils.image_uri = "123456789.dkr.ecr.us-west-2.amazonaws.com/custom-image:latest"
        
        framework, version = self.utils._extract_framework_from_image_uri()
        
        self.assertIsNone(framework)
        self.assertIsNone(version)


class TestHuggingFaceUtilities(unittest.TestCase):
    """Test HuggingFace-related utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_is_huggingface_model_with_slash(self):
        """Test HuggingFace model detection with organization/model format."""
        self.utils.model = "bert-base-uncased"
        self.utils._is_jumpstart_model_id = Mock(return_value=False)
        
        result = self.utils._is_huggingface_model()
        
        self.assertTrue(result)

    def test_is_huggingface_model_with_explicit_type(self):
        """Test HuggingFace model detection with explicit model_type."""
        self.utils.model = "my-model"
        self.utils.model_type = "huggingface"
        
        result = self.utils._is_huggingface_model()
        
        self.assertTrue(result)

    def test_is_not_huggingface_model_non_string(self):
        """Test HuggingFace model detection with non-string model."""
        self.utils.model = Mock()
        
        result = self.utils._is_huggingface_model()
        
        self.assertFalse(result)

    def test_get_supported_version_success(self):
        """Test successful extraction of supported framework version."""
        hf_config = {
            "versions": {
                "4.26.0": {
                    "pytorch1.13.1": {"py_versions": ["py39"]},
                    "pytorch1.12.1": {"py_versions": ["py38"]}
                }
            }
        }
        
        result = self.utils._get_supported_version(hf_config, "4.26.0", "pytorch")
        
        self.assertEqual(result, "1.13.1")

    def test_get_supported_version_no_versions_raises_error(self):
        """Test that ValueError is raised when no supported versions found."""
        hf_config = {
            "versions": {
                "4.26.0": {
                    "tensorflow2.11.0": {"py_versions": ["py39"]}
                }
            }
        }
        
        with self.assertRaises(ValueError) as context:
            self.utils._get_supported_version(hf_config, "4.26.0", "pytorch")
        
        self.assertIn("No supported versions found", str(context.exception))

    @patch('sagemaker.serve.model_builder_utils._ModelBuilderUtils.get_huggingface_model_metadata')
    def test_prepare_hf_model_for_upload_creates_directory(self, mock_get_metadata):
        """Test that HF model preparation creates necessary directories."""
        self.utils.model = "bert-base-uncased"
        self.utils.model_path = None
        self.utils.env_vars = {}
        
        with patch('sagemaker.serve.model_builder_utils._ModelBuilderUtils.download_huggingface_model_metadata'):
            self.utils._prepare_hf_model_for_upload()
        
        self.assertIsNotNone(self.utils.model_path)
        self.assertIn("bert-base-uncased", self.utils.model_path)


class TestResourceRequirements(unittest.TestCase):
    """Test resource requirement utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_get_processing_unit_cpu_default(self):
        """Test processing unit detection defaults to CPU."""
        self.utils.resource_requirements = None
        
        result = self.utils._get_processing_unit()
        
        self.assertEqual(result, "cpu")

    def test_get_processing_unit_gpu_from_resource_requirements(self):
        """Test processing unit detection from resource_requirements."""
        mock_resource_req = Mock()
        mock_resource_req.num_accelerators = 1
        self.utils.resource_requirements = mock_resource_req
        
        result = self.utils._get_processing_unit()
        
        self.assertEqual(result, "gpu")

    def test_get_processing_unit_gpu_from_modelbuilder_list(self):
        """Test processing unit detection from modelbuilder_list."""
        self.utils.resource_requirements = None
        mock_ic = Mock()
        mock_ic_resource_req = Mock()
        mock_ic_resource_req.num_accelerators = 2
        mock_ic.resource_requirements = mock_ic_resource_req
        self.utils.modelbuilder_list = [mock_ic]
        
        result = self.utils._get_processing_unit()
        
        self.assertEqual(result, "gpu")

    def test_get_processing_unit_cpu_zero_accelerators(self):
        """Test processing unit detection with zero accelerators."""
        mock_resource_req = Mock()
        mock_resource_req.num_accelerators = 0
        self.utils.resource_requirements = mock_resource_req
        
        result = self.utils._get_processing_unit()
        
        self.assertEqual(result, "cpu")


class TestMLflowUtilities(unittest.TestCase):
    """Test MLflow-related utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_has_mlflow_arguments_with_inference_spec_returns_false(self):
        """Test MLflow argument detection returns False when inference_spec present."""
        self.utils.inference_spec = Mock()
        self.utils.model = None
        self.utils.model_metadata = {"MLFLOW_MODEL_PATH": "/path/to/model"}
        
        result = self.utils._has_mlflow_arguments()
        
        self.assertFalse(result)

    def test_has_mlflow_arguments_with_model_returns_false(self):
        """Test MLflow argument detection returns False when model present."""
        self.utils.inference_spec = None
        self.utils.model = "some-model"
        self.utils.model_metadata = {"MLFLOW_MODEL_PATH": "/path/to/model"}
        
        result = self.utils._has_mlflow_arguments()
        
        self.assertFalse(result)

    def test_has_mlflow_arguments_no_metadata_returns_false(self):
        """Test MLflow argument detection returns False when no metadata."""
        self.utils.inference_spec = None
        self.utils.model = None
        self.utils.model_metadata = None
        
        result = self.utils._has_mlflow_arguments()
        
        self.assertFalse(result)

    def test_has_mlflow_arguments_no_mlflow_path_returns_false(self):
        """Test MLflow argument detection returns False when no MLFLOW_MODEL_PATH."""
        self.utils.inference_spec = None
        self.utils.model = None
        self.utils.model_metadata = {"OTHER_KEY": "value"}
        
        result = self.utils._has_mlflow_arguments()
        
        self.assertFalse(result)

    def test_has_mlflow_arguments_valid_returns_true(self):
        """Test MLflow argument detection returns True with valid arguments."""
        self.utils.inference_spec = None
        self.utils.model = None
        self.utils.model_metadata = {"MLFLOW_MODEL_PATH": "/path/to/model"}
        
        result = self.utils._has_mlflow_arguments()
        
        self.assertTrue(result)

    def test_get_artifact_path_direct_path(self):
        """Test artifact path retrieval for direct file path."""
        mlflow_model_path = "/local/path/to/model"
        
        result = self.utils._get_artifact_path(mlflow_model_path)
        
        self.assertEqual(result, mlflow_model_path)

    def test_get_artifact_path_run_id_without_tracking_arn_raises_error(self):
        """Test that ValueError is raised for run ID path without tracking ARN."""
        mlflow_model_path = "runs:/abc123/model"
        self.utils.model_metadata = {}
        
        with self.assertRaises(ValueError) as context:
            self.utils._get_artifact_path(mlflow_model_path)
        
        self.assertIn("MLFLOW_TRACKING_ARN", str(context.exception))

    def test_mlflow_metadata_exists_local_file_exists(self):
        """Test MLflow metadata existence check for local file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlmodel_file = os.path.join(tmpdir, "MLmodel")
            with open(mlmodel_file, 'w') as f:
                f.write("test content")
            
            result = self.utils._mlflow_metadata_exists(tmpdir)
            
            self.assertTrue(result)

    def test_mlflow_metadata_exists_local_file_not_exists(self):
        """Test MLflow metadata existence check when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.utils._mlflow_metadata_exists(tmpdir)
            
            self.assertFalse(result)

    @patch('sagemaker.serve.model_builder_utils.S3Downloader')
    def test_mlflow_metadata_exists_s3_path_exists(self, mock_s3_downloader):
        """Test MLflow metadata existence check for S3 path."""
        mock_downloader_instance = Mock()
        mock_downloader_instance.list.return_value = ["s3://bucket/path/MLmodel"]
        mock_s3_downloader.return_value = mock_downloader_instance
        
        self.utils.sagemaker_session = Mock()
        
        result = self.utils._mlflow_metadata_exists("s3://bucket/path")
        
        self.assertTrue(result)

    @patch('sagemaker.serve.model_builder_utils.S3Downloader')
    def test_mlflow_metadata_exists_s3_path_not_exists(self, mock_s3_downloader):
        """Test MLflow metadata existence check when S3 file doesn't exist."""
        mock_downloader_instance = Mock()
        mock_downloader_instance.list.return_value = []
        mock_s3_downloader.return_value = mock_downloader_instance
        
        self.utils.sagemaker_session = Mock()
        
        result = self.utils._mlflow_metadata_exists("s3://bucket/path")
        
        self.assertFalse(result)


class TestSerializationUtilities(unittest.TestCase):
    """Test serialization-related utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    @patch('sagemaker.serve.model_builder_utils.DEFAULT_SERIALIZERS_BY_FRAMEWORK')
    def test_fetch_serializer_for_known_framework(self, mock_default_serializers):
        """Test fetching serializer for known framework."""
        mock_serializer = Mock()
        mock_deserializer = Mock()
        mock_default_serializers.__getitem__.return_value = (mock_serializer, mock_deserializer)
        mock_default_serializers.__contains__.return_value = True
        
        serializer, deserializer = self.utils._fetch_serializer_and_deserializer_for_framework("pytorch")
        
        self.assertEqual(serializer, mock_serializer)
        self.assertEqual(deserializer, mock_deserializer)

    def test_fetch_serializer_for_unknown_framework(self):
        """Test fetching serializer for unknown framework returns defaults."""
        from sagemaker.core.serializers import NumpySerializer
        from sagemaker.core.deserializers import JSONDeserializer
        
        serializer, deserializer = self.utils._fetch_serializer_and_deserializer_for_framework("unknown")
        
        self.assertIsInstance(serializer, NumpySerializer)
        self.assertIsInstance(deserializer, JSONDeserializer)


class TestOptimizationUtilities(unittest.TestCase):
    """Test optimization-related utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_is_inferentia_or_trainium_with_inf1(self):
        """Test Inferentia detection for inf1 instance."""
        result = self.utils._is_inferentia_or_trainium("ml.inf1.xlarge")
        
        self.assertTrue(result)

    def test_is_inferentia_or_trainium_with_inf2(self):
        """Test Inferentia detection for inf2 instance."""
        result = self.utils._is_inferentia_or_trainium("ml.inf2.xlarge")
        
        self.assertTrue(result)

    def test_is_inferentia_or_trainium_with_trn1(self):
        """Test Trainium detection for trn1 instance."""
        result = self.utils._is_inferentia_or_trainium("ml.trn1.2xlarge")
        
        self.assertTrue(result)

    def test_is_inferentia_or_trainium_with_regular_instance(self):
        """Test Inferentia/Trainium detection for regular instance."""
        result = self.utils._is_inferentia_or_trainium("ml.m5.large")
        
        self.assertFalse(result)

    def test_is_inferentia_or_trainium_with_none(self):
        """Test Inferentia/Trainium detection with None."""
        result = self.utils._is_inferentia_or_trainium(None)
        
        self.assertFalse(result)

    def test_is_image_compatible_with_optimization_djl_lmi(self):
        """Test image compatibility check for DJL LMI image."""
        result = self.utils._is_image_compatible_with_optimization_job(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        
        self.assertTrue(result)

    def test_is_image_compatible_with_optimization_djl_neuronx(self):
        """Test image compatibility check for DJL Neuronx image."""
        result = self.utils._is_image_compatible_with_optimization_job(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.27.0-neuronx-sdk2.18.1"
        )
        
        self.assertTrue(result)

    def test_is_image_compatible_with_optimization_incompatible(self):
        """Test image compatibility check for incompatible image."""
        result = self.utils._is_image_compatible_with_optimization_job(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39"
        )
        
        self.assertFalse(result)

    def test_is_image_compatible_with_optimization_none(self):
        """Test image compatibility check with None."""
        result = self.utils._is_image_compatible_with_optimization_job(None)
        
        # None is treated as compatible (returns True)
        self.assertTrue(result)

    def test_deployment_config_contains_draft_model_true(self):
        """Test draft model detection in deployment config."""
        deployment_config = {
            "DeploymentArgs": {
                "AdditionalDataSources": "speculative_decoding"
            }
        }
        
        result = self.utils._deployment_config_contains_draft_model(deployment_config)
        
        self.assertTrue(result)

    def test_deployment_config_contains_draft_model_false(self):
        """Test draft model detection when not present."""
        deployment_config = {
            "DeploymentArgs": {
                "AdditionalDataSources": "other_data"
            }
        }
        
        result = self.utils._deployment_config_contains_draft_model(deployment_config)
        
        self.assertFalse(result)

    def test_deployment_config_contains_draft_model_none(self):
        """Test draft model detection with None config."""
        result = self.utils._deployment_config_contains_draft_model(None)
        
        self.assertFalse(result)

    def test_is_s3_uri_valid(self):
        """Test S3 URI validation for valid URI."""
        result = self.utils._is_s3_uri("s3://my-bucket/path/to/model")
        
        self.assertTrue(result)

    def test_is_s3_uri_invalid(self):
        """Test S3 URI validation for invalid URI."""
        result = self.utils._is_s3_uri("/local/path/to/model")
        
        self.assertFalse(result)

    def test_is_s3_uri_none(self):
        """Test S3 URI validation with None."""
        result = self.utils._is_s3_uri(None)
        
        self.assertFalse(result)


class TestEnvironmentVariableUtilities(unittest.TestCase):
    """Test environment variable utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_update_environment_variables_both_none(self):
        """Test updating environment variables when both are None."""
        result = self.utils._update_environment_variables(None, None)
        
        self.assertIsNone(result)

    def test_update_environment_variables_env_none(self):
        """Test updating environment variables when env is None."""
        new_env = {"KEY1": "value1"}
        
        result = self.utils._update_environment_variables(None, new_env)
        
        self.assertEqual(result, new_env)

    def test_update_environment_variables_new_env_none(self):
        """Test updating environment variables when new_env is None."""
        env = {"KEY1": "value1"}
        
        result = self.utils._update_environment_variables(env, None)
        
        self.assertEqual(result, env)

    def test_update_environment_variables_merge(self):
        """Test merging environment variables."""
        env = {"KEY1": "value1", "KEY2": "value2"}
        new_env = {"KEY2": "new_value2", "KEY3": "value3"}
        
        result = self.utils._update_environment_variables(env, new_env)
        
        self.assertEqual(result["KEY1"], "value1")
        self.assertEqual(result["KEY2"], "new_value2")  # Should be overwritten
        self.assertEqual(result["KEY3"], "value3")


class TestChannelNameGeneration(unittest.TestCase):
    """Test channel name generation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.utils = _ModelBuilderUtils()

    def test_generate_channel_name_no_existing_sources(self):
        """Test channel name generation with no existing sources."""
        result = self.utils._generate_channel_name(None)
        
        # Default channel name is "draft_model"
        self.assertEqual(result, "draft_model")

    def test_generate_channel_name_with_existing_sources(self):
        """Test channel name generation with existing sources."""
        existing_sources = [
            {"ChannelName": "additional-model-data-source-0"},
            {"ChannelName": "additional-model-data-source-1"}
        ]
        
        result = self.utils._generate_channel_name(existing_sources)
        
        # Returns the first channel name from existing sources
        self.assertEqual(result, "additional-model-data-source-0")

    def test_generate_channel_name_with_custom_name(self):
        """Test channel name generation with custom channel name."""
        existing_sources = [
            {"ChannelName": "custom-name"}
        ]
        
        result = self.utils._generate_channel_name(existing_sources)
        
        # Returns the first channel name from existing sources
        self.assertEqual(result, "custom-name")


if __name__ == '__main__':
    unittest.main()
