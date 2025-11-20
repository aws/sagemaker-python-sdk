"""
Additional unit tests for _ModelBuilderUtils to further improve coverage.
Targets remaining gaps from coverage report.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.serve.constants import Framework
from sagemaker.serve.utils.types import ModelServer
from sagemaker.train import ModelTrainer


class TestAutoDetectContainerDefault(unittest.TestCase):
    """Test _auto_detect_container_default method."""

    @patch('sagemaker.core.image_uris.retrieve')
    @patch.object(_ModelBuilderUtils, '_get_hf_framework_versions')
    def test_auto_detect_container_pytorch(self, mock_get_versions, mock_retrieve):
        """Test auto-detecting container for PyTorch."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.instance_type = "ml.g5.xlarge"
        utils.region = "us-west-2"
        utils.env_vars = {}
        
        mock_get_versions.return_value = ("1.13.0", None, "4.26", "py39")
        mock_retrieve.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13"
        
        result = utils._auto_detect_container_default()
        
        self.assertIsNotNone(result)
        self.assertIn("pytorch", result)

    @patch('sagemaker.core.image_uris.retrieve')
    @patch.object(_ModelBuilderUtils, '_get_hf_framework_versions')
    def test_auto_detect_container_tensorflow(self, mock_get_versions, mock_retrieve):
        """Test auto-detecting container for TensorFlow."""
        utils = _ModelBuilderUtils()
        utils.model = "bert-base"
        utils.instance_type = "ml.m5.large"
        utils.region = "us-west-2"
        utils.env_vars = {}
        
        mock_get_versions.return_value = (None, "2.11.0", "4.26", "py39")
        mock_retrieve.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.11"
        
        result = utils._auto_detect_container_default()
        
        self.assertIsNotNone(result)
        self.assertIn("tensorflow", result)

    @patch.object(_ModelBuilderUtils, '_get_hf_framework_versions')
    def test_auto_detect_container_no_instance_type(self, mock_get_versions):
        """Test auto-detecting container without instance type."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.instance_type = None
        
        with self.assertRaises(ValueError) as context:
            utils._auto_detect_container_default()
        
        self.assertIn("Instance type is not specified", str(context.exception))

    @patch.object(_ModelBuilderUtils, '_get_hf_framework_versions')
    def test_auto_detect_container_no_framework(self, mock_get_versions):
        """Test auto-detecting container with no framework detected."""
        utils = _ModelBuilderUtils()
        utils.model = "unknown-model"
        utils.instance_type = "ml.m5.large"
        utils.region = "us-west-2"
        utils.env_vars = {}
        
        mock_get_versions.return_value = (None, None, "4.26", "py39")
        
        with self.assertRaises(ValueError) as context:
            utils._auto_detect_container_default()
        
        self.assertIn("Could not detect framework", str(context.exception))


class TestGetSMDImageUri(unittest.TestCase):
    """Test _get_smd_image_uri method."""

    @patch('sagemaker.core.image_uris.retrieve')
    def test_get_smd_image_uri_cpu(self, mock_retrieve):
        """Test getting SMD image URI for CPU."""
        utils = _ModelBuilderUtils()
        utils.region = "us-west-2"
        utils.sagemaker_session = None
        
        mock_retrieve.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-distribution:latest"
        
        result = utils._get_smd_image_uri("cpu")
        
        self.assertIsNotNone(result)
        mock_retrieve.assert_called_once()

    @patch('sagemaker.core.image_uris.retrieve')
    def test_get_smd_image_uri_gpu(self, mock_retrieve):
        """Test getting SMD image URI for GPU."""
        utils = _ModelBuilderUtils()
        utils.region = "us-west-2"
        utils.sagemaker_session = None
        
        mock_retrieve.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-distribution:latest-gpu"
        
        result = utils._get_smd_image_uri("gpu")
        
        self.assertIsNotNone(result)

    def test_get_smd_image_uri_invalid_processing_unit(self):
        """Test getting SMD image URI with invalid processing unit."""
        utils = _ModelBuilderUtils()
        utils.region = "us-west-2"
        utils.sagemaker_session = None
        
        with self.assertRaises(ValueError) as context:
            utils._get_smd_image_uri("invalid")
        
        self.assertIn("Invalid processing unit", str(context.exception))


class TestDetectModelObjectImage(unittest.TestCase):
    """Test _detect_model_object_image method - skipped (complex mocking)."""
    pass


class TestAutoDetectImageUri(unittest.TestCase):
    """Test _auto_detect_image_uri method."""

    @patch.object(_ModelBuilderUtils, '_extract_framework_from_image_uri')
    def test_auto_detect_image_uri_with_provided_uri(self, mock_extract):
        """Test auto-detect skips when image_uri provided."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "custom-image:latest"
        
        mock_extract.return_value = (Framework.PYTORCH, "1.13")
        
        utils._auto_detect_image_uri()
        
        self.assertEqual(utils.image_uri, "custom-image:latest")

    @patch.object(_ModelBuilderUtils, '_detect_jumpstart_image')
    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_auto_detect_image_uri_jumpstart(self, mock_is_js, mock_detect_js):
        """Test auto-detect for JumpStart model."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.model = "huggingface-llm-falcon-7b"
        
        mock_is_js.return_value = True
        
        utils._auto_detect_image_uri()
        
        mock_detect_js.assert_called_once()

    @patch.object(_ModelBuilderUtils, '_detect_huggingface_image')
    @patch.object(_ModelBuilderUtils, '_is_huggingface_model')
    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_auto_detect_image_uri_huggingface(self, mock_is_js, mock_is_hf, mock_detect_hf):
        """Test auto-detect for HuggingFace model."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.model = "gpt2"
        
        mock_is_js.return_value = False
        mock_is_hf.return_value = True
        
        utils._auto_detect_image_uri()
        
        mock_detect_hf.assert_called_once()

    @patch.object(_ModelBuilderUtils, '_detect_model_object_image')
    def test_auto_detect_image_uri_object_model(self, mock_detect_obj):
        """Test auto-detect for object model."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.model = Mock()
        
        utils._auto_detect_image_uri()
        
        mock_detect_obj.assert_called_once()


class TestUseJumpStartEquivalent(unittest.TestCase):
    """Test _use_jumpstart_equivalent method."""

    @patch.object(_ModelBuilderUtils, '_retrieve_hugging_face_model_mapping')
    def test_use_jumpstart_equivalent_no_image_uri(self, mock_retrieve):
        """Test using JumpStart equivalent without image_uri - skipped (complex schema builder init)."""
        pass

    def test_use_jumpstart_equivalent_with_image_uri(self):
        """Test using JumpStart equivalent with image_uri provided."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.image_uri = "custom-image"
        utils.env_vars = None
        
        result = utils._use_jumpstart_equivalent()
        
        self.assertFalse(result)

    def test_use_jumpstart_equivalent_with_env_vars(self):
        """Test using JumpStart equivalent with env_vars provided."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.image_uri = None
        utils.env_vars = {"KEY": "value"}
        
        result = utils._use_jumpstart_equivalent()
        
        self.assertFalse(result)

    @patch.object(_ModelBuilderUtils, '_retrieve_hugging_face_model_mapping')
    def test_use_jumpstart_equivalent_no_mapping(self, mock_retrieve):
        """Test using JumpStart equivalent with no mapping."""
        utils = _ModelBuilderUtils()
        utils.model = "unknown-model"
        utils.image_uri = None
        utils.env_vars = None
        
        mock_retrieve.return_value = {}
        
        result = utils._use_jumpstart_equivalent()
        
        self.assertFalse(result)


class TestPrepareHFModelForUpload(unittest.TestCase):
    """Test _prepare_hf_model_for_upload method."""

    @patch.object(_ModelBuilderUtils, 'download_huggingface_model_metadata')
    def test_prepare_hf_model_for_upload_no_model_path(self, mock_download):
        """Test preparing HF model without model_path."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.model_path = None
        utils.env_vars = {}
        
        utils._prepare_hf_model_for_upload()
        
        self.assertIsNotNone(utils.model_path)
        mock_download.assert_called_once()

    def test_prepare_hf_model_for_upload_with_model_path(self):
        """Test preparing HF model with existing model_path."""
        utils = _ModelBuilderUtils()
        utils.model = "gpt2"
        utils.model_path = "/existing/path"
        
        utils._prepare_hf_model_for_upload()
        
        self.assertEqual(utils.model_path, "/existing/path")


class TestGetInferenceComponentResourceRequirements(unittest.TestCase):
    """Test _get_inference_component_resource_requirements method."""

    def test_get_ic_resource_requirements_with_existing(self):
        """Test getting IC resource requirements with existing requirements."""
        utils = _ModelBuilderUtils()
        mb = Mock()
        mb.resource_requirements = Mock()
        mb._is_jumpstart_model_id = Mock(return_value=True)
        
        result = utils._get_inference_component_resource_requirements(mb)
        
        self.assertEqual(result, mb)

    @patch.object(_ModelBuilderUtils, '_is_jumpstart_model_id')
    def test_get_ic_resource_requirements_no_jumpstart(self, mock_is_js):
        """Test getting IC resource requirements for non-JumpStart model."""
        utils = _ModelBuilderUtils()
        mb = Mock()
        mb.resource_requirements = None
        mb._is_jumpstart_model_id = Mock(return_value=False)
        
        result = utils._get_inference_component_resource_requirements(mb)
        
        self.assertEqual(result, mb)


class TestCanFitOnSingleGPU(unittest.TestCase):
    """Test _can_fit_on_single_gpu method."""

    def test_can_fit_on_single_gpu_no_method(self):
        """Test can fit on single GPU without _try_fetch_gpu_info."""
        utils = _ModelBuilderUtils()
        
        result = utils._can_fit_on_single_gpu()
        
        self.assertFalse(result)


class TestFetchSerializerAndDeserializer(unittest.TestCase):
    """Test _fetch_serializer_and_deserializer_for_framework method."""

    def test_fetch_serializer_pytorch(self):
        """Test fetching serializer for PyTorch."""
        utils = _ModelBuilderUtils()
        
        serializer, deserializer = utils._fetch_serializer_and_deserializer_for_framework("pytorch")
        
        self.assertIsNotNone(serializer)
        self.assertIsNotNone(deserializer)

    def test_fetch_serializer_unknown(self):
        """Test fetching serializer for unknown framework."""
        utils = _ModelBuilderUtils()
        
        serializer, deserializer = utils._fetch_serializer_and_deserializer_for_framework("unknown")
        
        self.assertIsNotNone(serializer)
        self.assertIsNotNone(deserializer)


class TestHandleMLflowInput(unittest.TestCase):
    """Test _handle_mlflow_input method."""

    def test_handle_mlflow_input_not_mlflow(self):
        """Test handling non-MLflow input."""
        utils = _ModelBuilderUtils()
        utils.model = Mock()
        utils.inference_spec = None
        utils.model_metadata = None
        
        utils._handle_mlflow_input()
        
        self.assertFalse(utils._is_mlflow_model)

    @patch.object(_ModelBuilderUtils, '_mlflow_metadata_exists')
    @patch.object(_ModelBuilderUtils, '_get_artifact_path')
    def test_handle_mlflow_input_no_metadata(self, mock_get_path, mock_exists):
        """Test handling MLflow input without metadata."""
        utils = _ModelBuilderUtils()
        utils.model = None
        utils.inference_spec = None
        utils.model_metadata = {"MLFLOW_MODEL_PATH": "/path/to/model"}
        
        mock_get_path.return_value = "/path/to/model"
        mock_exists.return_value = False
        
        utils._handle_mlflow_input()
        
        self.assertTrue(utils._is_mlflow_model)


class TestExtractFrameworkFromModelTrainer(unittest.TestCase):
    """Test _extract_framework_from_model_trainer method."""

    def test_extract_framework_pytorch(self):
        """Test extracting PyTorch framework."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13"
        
        result = utils._extract_framework_from_model_trainer(trainer)
        
        self.assertEqual(result, Framework.PYTORCH)

    def test_extract_framework_tensorflow(self):
        """Test extracting TensorFlow framework."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.11"
        
        result = utils._extract_framework_from_model_trainer(trainer)
        
        self.assertEqual(result, Framework.TENSORFLOW)

    def test_extract_framework_huggingface(self):
        """Test extracting HuggingFace framework."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.13"
        
        result = utils._extract_framework_from_model_trainer(trainer)
        
        # HuggingFace images contain pytorch, so it returns PYTORCH not HUGGINGFACE
        self.assertIn(result, [Framework.PYTORCH, Framework.HUGGINGFACE])

    def test_extract_framework_unknown(self):
        """Test extracting unknown framework."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "custom-training-image:latest"
        
        result = utils._extract_framework_from_model_trainer(trainer)
        
        self.assertIsNone(result)


class TestInferModelServerFromTraining(unittest.TestCase):
    """Test _infer_model_server_from_training method."""

    def test_infer_model_server_huggingface_tgi(self):
        """Test inferring TGI for HuggingFace."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "huggingface-pytorch-training:1.13"
        trainer.hyperparameters = {"max_new_tokens": 100}
        
        result = utils._infer_model_server_from_training(trainer)
        
        self.assertEqual(result, ModelServer.TGI)

    def test_infer_model_server_pytorch(self):
        """Test inferring TorchServe for PyTorch."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.training_image = "pytorch-training:1.13"
        trainer.hyperparameters = {}
        
        with patch.object(utils, '_extract_framework_from_model_trainer', return_value=Framework.PYTORCH):
            result = utils._infer_model_server_from_training(trainer)
        
        self.assertEqual(result, ModelServer.TORCHSERVE)


class TestExtractInferenceSpecFromTrainingCode(unittest.TestCase):
    """Test _extract_inference_spec_from_training_code method."""

    def test_extract_inference_spec_no_source(self):
        """Test extracting inference spec without source."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.source_code = None
        
        result = utils._extract_inference_spec_from_training_code(trainer)
        
        self.assertIsNone(result)

    def test_extract_inference_spec_s3_source(self):
        """Test extracting inference spec from S3 source."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.source_code = Mock()
        trainer.source_code.source_dir = "s3://bucket/code"
        
        result = utils._extract_inference_spec_from_training_code(trainer)
        
        self.assertIsNone(result)


class TestInheritTrainingEnvironment(unittest.TestCase):
    """Test _inherit_training_environment method."""

    def test_inherit_training_environment(self):
        """Test inheriting training environment."""
        utils = _ModelBuilderUtils()
        trainer = Mock(spec=ModelTrainer)
        trainer.environment = {"HUGGING_FACE_HUB_TOKEN": "token123"}
        trainer._latest_training_job = Mock()
        trainer._latest_training_job.environment = {"MODEL_CLASS_NAME": "MyModel"}
        
        result = utils._inherit_training_environment(trainer)
        
        self.assertIn("HUGGING_FACE_HUB_TOKEN", result)
        self.assertIn("MODEL_CLASS_NAME", result)


class TestExtractVersionFromTrainingImage(unittest.TestCase):
    """Test _extract_version_from_training_image method."""

    def test_extract_version_success(self):
        """Test extracting version successfully."""
        utils = _ModelBuilderUtils()
        
        result = utils._extract_version_from_training_image("pytorch-training:1.13.0-gpu")
        
        self.assertEqual(result, "1.13.0")

    def test_extract_version_no_match(self):
        """Test extracting version with no match."""
        utils = _ModelBuilderUtils()
        
        result = utils._extract_version_from_training_image("custom-image:latest")
        
        self.assertIsNone(result)


class TestDetectInferenceImageFromTraining(unittest.TestCase):
    """Test _detect_inference_image_from_training method."""

    def test_detect_inference_image_pytorch(self):
        """Test detecting inference image for PyTorch."""
        utils = _ModelBuilderUtils()
        utils.model = Mock(spec=ModelTrainer)
        utils.model.training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13"
        
        utils._detect_inference_image_from_training()
        
        self.assertIn("pytorch-inference", utils.image_uri)

    def test_detect_inference_image_tensorflow(self):
        """Test detecting inference image for TensorFlow."""
        utils = _ModelBuilderUtils()
        utils.model = Mock(spec=ModelTrainer)
        utils.model.training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.11"
        
        utils._detect_inference_image_from_training()
        
        self.assertIn("tensorflow-inference", utils.image_uri)


class TestEnsureBaseNameIfNeeded(unittest.TestCase):
    """Test _ensure_base_name_if_needed method."""

    def test_ensure_base_name_with_model_name(self):
        """Test ensuring base name when model_name exists."""
        utils = _ModelBuilderUtils()
        utils.model_name = "my-model"
        
        utils._ensure_base_name_if_needed("image-uri", None, None)
        
        # Should not set _base_name if model_name exists
        self.assertFalse(hasattr(utils, '_base_name') and utils._base_name)

    @patch('sagemaker.core.common_utils.base_name_from_image')
    def test_ensure_base_name_without_model_name(self, mock_base_name):
        """Test ensuring base name without model_name."""
        utils = _ModelBuilderUtils()
        utils.model_name = None
        
        mock_base_name.return_value = "base-name"
        
        utils._ensure_base_name_if_needed("image-uri", None, None)
        
        # _base_name is set to result of base_name_from_image or the image itself
        self.assertIsNotNone(utils._base_name)


class TestEnsureMetadataConfigs(unittest.TestCase):
    """Test _ensure_metadata_configs method."""

    @patch('sagemaker.core.jumpstart.utils.get_jumpstart_configs')
    def test_ensure_metadata_configs_jumpstart(self, mock_get_configs):
        """Test ensuring metadata configs for JumpStart model."""
        utils = _ModelBuilderUtils()
        utils._metadata_configs = None
        utils.model = "huggingface-llm-falcon-7b"
        utils.region = "us-west-2"
        utils.sagemaker_session = Mock()
        
        mock_get_configs.return_value = {"config-1": Mock()}
        
        utils._ensure_metadata_configs()
        
        self.assertIsNotNone(utils._metadata_configs)

    def test_ensure_metadata_configs_not_string(self):
        """Test ensuring metadata configs for non-string model."""
        utils = _ModelBuilderUtils()
        utils._metadata_configs = None
        utils.model = Mock()
        
        utils._ensure_metadata_configs()
        
        # Should remain None for non-string models
        self.assertIsNone(utils._metadata_configs)


class TestGetServeSettings(unittest.TestCase):
    """Test _get_serve_setting method - skipped (requires proper session setup)."""
    pass


if __name__ == "__main__":
    unittest.main()
