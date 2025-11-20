"""Unit tests for _ModelBuilderUtils class utility methods."""
import unittest
from unittest.mock import Mock, patch
from typing import Optional, Dict

from sagemaker.serve.constants import Framework


class TestModelBuilderUtilsInferentiaTrainium(unittest.TestCase):
    """Test _ModelBuilderUtils._is_inferentia_or_trainium method."""

    def test_is_inferentia_instance(self):
        """Test detection of Inferentia instances."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        # Test various Inferentia instance types
        inf_instances = [
            "ml.inf1.xlarge",
            "ml.inf1.2xlarge",
            "ml.inf2.xlarge",
            "ml.inf2.24xlarge",
        ]
        
        for instance in inf_instances:
            result = _ModelBuilderUtils._is_inferentia_or_trainium(mock_builder, instance)
            self.assertTrue(result, f"Failed for {instance}")

    def test_is_trainium_instance(self):
        """Test detection of Trainium instances."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        # Test various Trainium instance types
        trn_instances = [
            "ml.trn1.2xlarge",
            "ml.trn1.32xlarge",
            "ml.trn1n.32xlarge",
        ]
        
        for instance in trn_instances:
            result = _ModelBuilderUtils._is_inferentia_or_trainium(mock_builder, instance)
            self.assertTrue(result, f"Failed for {instance}")

    def test_is_not_inferentia_or_trainium(self):
        """Test non-Inferentia/Trainium instances return False."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        # Test various non-Inferentia/Trainium instance types
        other_instances = [
            "ml.m5.xlarge",
            "ml.g5.2xlarge",
            "ml.p3.8xlarge",
            "ml.c5.large",
        ]
        
        for instance in other_instances:
            result = _ModelBuilderUtils._is_inferentia_or_trainium(mock_builder, instance)
            self.assertFalse(result, f"Failed for {instance}")

    def test_is_inferentia_or_trainium_with_none(self):
        """Test with None instance type."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._is_inferentia_or_trainium(mock_builder, None)
        self.assertFalse(result)

    def test_is_inferentia_or_trainium_with_invalid_format(self):
        """Test with invalid instance type format."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        invalid_instances = [
            "invalid-instance",
            "m5.xlarge",  # Missing ml. prefix
            "",
        ]
        
        for instance in invalid_instances:
            result = _ModelBuilderUtils._is_inferentia_or_trainium(mock_builder, instance)
            self.assertFalse(result, f"Failed for {instance}")


class TestModelBuilderUtilsImageCompatibility(unittest.TestCase):
    """Test _ModelBuilderUtils._is_image_compatible_with_optimization_job method."""

    def test_is_compatible_with_djl_lmi_image(self):
        """Test compatibility with DJL LMI images."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        compatible_images = [
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-lmi",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-lmi7.0.0-cu118",
        ]
        
        for image in compatible_images:
            result = _ModelBuilderUtils._is_image_compatible_with_optimization_job(mock_builder, image)
            self.assertTrue(result, f"Failed for {image}")

    def test_is_compatible_with_djl_neuronx_image(self):
        """Test compatibility with DJL Neuronx images."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        compatible_images = [
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-neuronx-sdk2.13.0",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-",
        ]
        
        for image in compatible_images:
            result = _ModelBuilderUtils._is_image_compatible_with_optimization_job(mock_builder, image)
            self.assertTrue(result, f"Failed for {image}")

    def test_is_not_compatible_with_other_images(self):
        """Test incompatibility with non-DJL images."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        incompatible_images = [
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.9.1-cpu",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.0-tgi0.8.2-gpu-py39-cu118",
        ]
        
        for image in incompatible_images:
            result = _ModelBuilderUtils._is_image_compatible_with_optimization_job(mock_builder, image)
            self.assertFalse(result, f"Failed for {image}")

    def test_is_compatible_with_none_image(self):
        """Test that None image URI is considered compatible."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._is_image_compatible_with_optimization_job(mock_builder, None)
        self.assertTrue(result)


class TestModelBuilderUtilsDeploymentConfig(unittest.TestCase):
    """Test _ModelBuilderUtils deployment config methods."""

    def test_deployment_config_contains_draft_model_true(self):
        """Test detection of draft model in deployment config."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        deployment_config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {
                    "speculative_decoding": [
                        {"channel_name": "draft_model"}
                    ]
                }
            }
        }
        
        result = _ModelBuilderUtils._deployment_config_contains_draft_model(
            mock_builder, deployment_config
        )
        self.assertTrue(result)

    def test_deployment_config_contains_draft_model_false(self):
        """Test when deployment config doesn't contain draft model."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        deployment_config = {
            "DeploymentArgs": {
                "AdditionalDataSources": {
                    "other_data": []
                }
            }
        }
        
        result = _ModelBuilderUtils._deployment_config_contains_draft_model(
            mock_builder, deployment_config
        )
        self.assertFalse(result)

    def test_deployment_config_contains_draft_model_none(self):
        """Test with None deployment config."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._deployment_config_contains_draft_model(
            mock_builder, None
        )
        self.assertFalse(result)

    def test_deployment_config_contains_draft_model_no_additional_sources(self):
        """Test when deployment config has no AdditionalDataSources."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        deployment_config = {
            "DeploymentArgs": {}
        }
        
        result = _ModelBuilderUtils._deployment_config_contains_draft_model(
            mock_builder, deployment_config
        )
        self.assertFalse(result)

    def test_is_draft_model_jumpstart_provided_true(self):
        """Test detection of JumpStart-provided draft model."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        deployment_config = {
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
        
        result = _ModelBuilderUtils._is_draft_model_jumpstart_provided(
            mock_builder, deployment_config
        )
        self.assertTrue(result)

    def test_is_draft_model_jumpstart_provided_false(self):
        """Test when draft model is not JumpStart-provided."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        deployment_config = {
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
        
        result = _ModelBuilderUtils._is_draft_model_jumpstart_provided(
            mock_builder, deployment_config
        )
        self.assertFalse(result)

    def test_is_draft_model_jumpstart_provided_none(self):
        """Test with None deployment config."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._is_draft_model_jumpstart_provided(
            mock_builder, None
        )
        self.assertFalse(result)


class TestModelBuilderUtilsFrameworkNormalization(unittest.TestCase):
    """Test _ModelBuilderUtils._normalize_framework_to_enum method."""

    def test_normalize_framework_with_enum(self):
        """Test normalization when input is already Framework enum."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._normalize_framework_to_enum(
            mock_builder, Framework.PYTORCH
        )
        self.assertEqual(result, Framework.PYTORCH)

    def test_normalize_framework_with_none(self):
        """Test normalization with None input."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, None)
        self.assertIsNone(result)

    def test_normalize_pytorch_variants(self):
        """Test normalization of PyTorch string variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        pytorch_variants = ["pytorch", "PyTorch", "PYTORCH", "torch", "Torch"]
        
        for variant in pytorch_variants:
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, Framework.PYTORCH, f"Failed for {variant}")

    def test_normalize_tensorflow_variants(self):
        """Test normalization of TensorFlow string variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        tf_variants = ["tensorflow", "TensorFlow", "TENSORFLOW", "tf", "TF"]
        
        for variant in tf_variants:
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, Framework.TENSORFLOW, f"Failed for {variant}")

    def test_normalize_sklearn_variants(self):
        """Test normalization of scikit-learn string variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        sklearn_variants = ["sklearn", "scikit-learn", "scikit_learn", "sk-learn"]
        
        for variant in sklearn_variants:
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, Framework.SKLEARN, f"Failed for {variant}")

    def test_normalize_huggingface_variants(self):
        """Test normalization of HuggingFace string variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        hf_variants = ["huggingface", "HuggingFace", "hf", "transformers"]
        
        for variant in hf_variants:
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, Framework.HUGGINGFACE, f"Failed for {variant}")

    def test_normalize_xgboost_variants(self):
        """Test normalization of XGBoost string variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        xgb_variants = ["xgboost", "XGBoost", "xgb", "XGB"]
        
        for variant in xgb_variants:
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, Framework.XGBOOST, f"Failed for {variant}")

    def test_normalize_other_frameworks(self):
        """Test normalization of other framework variants."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        framework_map = {
            "mxnet": Framework.MXNET,
            "chainer": Framework.CHAINER,
            "djl": Framework.DJL,
            "sparkml": Framework.SPARKML,
            "spark": Framework.SPARKML,
            "lda": Framework.LDA,
            "ntm": Framework.NTM,
            "smd": Framework.SMD,
            "sagemaker-distribution": Framework.SMD,
        }
        
        for variant, expected in framework_map.items():
            result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, variant)
            self.assertEqual(result, expected, f"Failed for {variant}")

    def test_normalize_unknown_framework(self):
        """Test normalization with unknown framework string."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._normalize_framework_to_enum(
            mock_builder, "unknown_framework"
        )
        self.assertIsNone(result)

    def test_normalize_invalid_type(self):
        """Test normalization with invalid input type."""
        from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
        
        mock_builder = Mock(spec=_ModelBuilderUtils)
        
        result = _ModelBuilderUtils._normalize_framework_to_enum(mock_builder, 123)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
