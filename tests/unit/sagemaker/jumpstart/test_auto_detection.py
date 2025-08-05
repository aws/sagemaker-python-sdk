# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Tests for JumpStart inference configuration auto-detection."""

from __future__ import absolute_import
import unittest
from unittest.mock import Mock, patch
import copy

from sagemaker.jumpstart.artifacts.image_uris import _retrieve_image_uri
from sagemaker.jumpstart.artifacts.model_uris import _retrieve_model_uri
from sagemaker.jumpstart.artifacts.environment_variables import _retrieve_default_environment_variables
from sagemaker.jumpstart.artifacts.resource_requirements import _retrieve_default_resources
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartModelType
from sagemaker.jumpstart.types import JumpStartModelSpecs

# Mock spec with multiple inference configurations
MULTI_CONFIG_SPEC = {
    "model_id": "test-multi-config-model",
    "version": "1.0.0",
    "min_sdk_version": "2.189.0",
    "hosting_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.10.0",
        "py_version": "py38",
    },
    "hosting_artifact_key": "default/artifacts/",
    "inference_configs": {
        "tgi": {
                "component_names": ["tgi"],
                "resolved_config": {
                    "hosting_instance_type_variants": {
                        "regional_aliases": {
                            "us-west-2": {
                                "tgi_image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04"
                            }
                        },
                        "variants": {
                            "g5": {
                                "regional_properties": {
                                    "image_uri": "$tgi_image"
                                },
                                "properties": {
                                    "artifact_key": "artifacts/tgi/inference-prepack/v1.0.0/",
                                    "environment_variables": {
                                        "HF_MODEL_ID": "/opt/ml/model",
                                        "OPTION_GPU_MEMORY_UTILIZATION": "0.85",
                                        "SM_NUM_GPUS": "1"
                                    }
                                }
                            },
                            "ml.g5.12xlarge": {
                                "regional_properties": {
                                    "image_uri": "$tgi_image"
                                },
                                "properties": {
                                    "artifact_key": "artifacts/tgi/inference-prepack/v1.0.0/",
                                    "environment_variables": {
                                        "HF_MODEL_ID": "/opt/ml/model",
                                        "OPTION_GPU_MEMORY_UTILIZATION": "0.85",
                                        "SM_NUM_GPUS": "1"
                                    },
                                    "resource_requirements": {
                                        "num_accelerators": 4,
                                        "min_memory": 98304
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "neuron": {
                "component_names": ["neuron"],
                "resolved_config": {
                    "hosting_instance_type_variants": {
                        "regional_aliases": {
                            "us-west-2": {
                                "neuron_image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1"
                            }
                        },
                        "variants": {
                            "inf2": {
                                "regional_properties": {
                                    "image_uri": "$neuron_image"
                                },
                                "properties": {
                                    "artifact_key": "artifacts/neuron/inference-prepack/v1.0.0/",
                                    "environment_variables": {
                                        "OPTION_TENSOR_PARALLEL_DEGREE": "12",
                                        "OPTION_N_POSITIONS": "4096",
                                        "OPTION_DTYPE": "fp16",
                                        "OPTION_NEURON_OPTIMIZE_LEVEL": "2"
                                    }
                                }
                            },
                            "ml.inf2.24xlarge": {
                                "regional_properties": {
                                    "image_uri": "$neuron_image"
                                },
                                "properties": {
                                    "artifact_key": "artifacts/neuron/inference-prepack/v1.0.0/",
                                    "environment_variables": {
                                        "OPTION_TENSOR_PARALLEL_DEGREE": "12",
                                        "OPTION_N_POSITIONS": "4096",
                                        "OPTION_DTYPE": "fp16",
                                        "OPTION_NEURON_OPTIMIZE_LEVEL": "2"
                                    },
                                    "resource_requirements": {
                                        "num_accelerators": 6,
                                        "min_memory": 196608
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    "inference_config_rankings": {
        "overall": {
            "description": "default",
            "rankings": ["tgi", "lmi", "lmi-optimized", "neuron"]
        }
    },
    "inference_environment_variables": [
        {
            "name": "SAGEMAKER_PROGRAM",
            "type": "text",
            "default": "inference.py",
            "scope": "container",
            "required_for_model_class": True,
        }
    ],
    "hosting_resource_requirements": {"num_accelerators": 1, "min_memory_mb": 8192},
}


class AutoDetectionTestCase(unittest.TestCase):
    """Base test case for auto-detection functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        self.model_id = "test-multi-config-model"
        self.model_version = "1.0.0"
        self.region = "us-west-2"
        self.mock_session = Mock(boto_region_name=self.region)

    def _get_mock_model_specs(self, config_name=None):
        """Get mock model specs with optional config selection."""
        # Create simple mock that avoids JumpStartModelSpecs parsing complexity
        mock_spec = Mock()
        
        if config_name is None:
            # Full spec with inference_configs for auto-detection
            mock_spec.inference_configs = Mock()
            mock_spec.inference_configs.configs = {
                "tgi": Mock(resolved_config={
                    "hosting_instance_type_variants": {
                        "regional_aliases": {"us-west-2": {"tgi_image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04"}},
                        "variants": {
                            "g5": {"regional_properties": {"image_uri": "$tgi_image"}, "properties": {"artifact_key": "artifacts/tgi/inference-prepack/v1.0.0/", "environment_variables": {"HF_MODEL_ID": "/opt/ml/model", "OPTION_GPU_MEMORY_UTILIZATION": "0.85", "SM_NUM_GPUS": "1"}}},
                            "ml.g5.12xlarge": {"regional_properties": {"image_uri": "$tgi_image"}, "properties": {"artifact_key": "artifacts/tgi/inference-prepack/v1.0.0/", "environment_variables": {"HF_MODEL_ID": "/opt/ml/model", "OPTION_GPU_MEMORY_UTILIZATION": "0.85", "SM_NUM_GPUS": "1"}, "resource_requirements": {"num_accelerators": 4, "min_memory": 98304}}}
                        }
                    }
                }),
                "neuron": Mock(resolved_config={
                    "hosting_instance_type_variants": {
                        "regional_aliases": {"us-west-2": {"neuron_image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1"}},
                        "variants": {
                            "inf2": {"regional_properties": {"image_uri": "$neuron_image"}, "properties": {"artifact_key": "artifacts/neuron/inference-prepack/v1.0.0/", "environment_variables": {"OPTION_TENSOR_PARALLEL_DEGREE": "12", "OPTION_N_POSITIONS": "4096", "OPTION_DTYPE": "fp16", "OPTION_NEURON_OPTIMIZE_LEVEL": "2"}}},
                            "ml.inf2.24xlarge": {"regional_properties": {"image_uri": "$neuron_image"}, "properties": {"artifact_key": "artifacts/neuron/inference-prepack/v1.0.0/", "environment_variables": {"OPTION_TENSOR_PARALLEL_DEGREE": "12", "OPTION_N_POSITIONS": "4096", "OPTION_DTYPE": "fp16", "OPTION_NEURON_OPTIMIZE_LEVEL": "2"}, "resource_requirements": {"num_accelerators": 6, "min_memory": 196608}}}
                        }
                    }
                })
            }
            mock_spec.inference_config_rankings = Mock()
            mock_spec.inference_config_rankings.get.return_value = Mock(rankings=["tgi", "lmi", "lmi-optimized", "neuron"])
        else:
            # Config-specific spec (inference_configs removed)
            mock_spec.inference_configs = None
            mock_spec.inference_config_rankings = None
            
            # Mock the hosting_instance_type_variants based on selected config
            if config_name == "neuron":
                mock_spec.hosting_instance_type_variants = Mock()
                mock_spec.hosting_instance_type_variants.get_image_uri.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1"
                mock_spec.hosting_instance_type_variants.get_instance_specific_artifact_key.return_value = "artifacts/neuron/inference-prepack/v1.0.0/"
                mock_spec.hosting_instance_type_variants.get_instance_specific_environment_variables.return_value = {
                    "OPTION_TENSOR_PARALLEL_DEGREE": "12", "OPTION_N_POSITIONS": "4096", "OPTION_DTYPE": "fp16", "OPTION_NEURON_OPTIMIZE_LEVEL": "2"
                }
                mock_spec.hosting_instance_type_variants.get_instance_specific_resource_requirements.return_value = {"num_accelerators": 6, "min_memory_mb": 196608}
                # Additional needed attributes
                mock_spec.inference_environment_variables = []
                mock_spec.hosting_resource_requirements = {"num_accelerators": 1, "min_memory_mb": 8192}
                mock_spec.dynamic_container_deployment_supported = True
            elif config_name == "tgi":
                mock_spec.hosting_instance_type_variants = Mock()
                mock_spec.hosting_instance_type_variants.get_image_uri.return_value = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04"
                mock_spec.hosting_instance_type_variants.get_instance_specific_artifact_key.return_value = "artifacts/tgi/inference-prepack/v1.0.0/"
                mock_spec.hosting_instance_type_variants.get_instance_specific_environment_variables.return_value = {
                    "HF_MODEL_ID": "/opt/ml/model", "OPTION_GPU_MEMORY_UTILIZATION": "0.85", "SM_NUM_GPUS": "1"
                }
                mock_spec.hosting_instance_type_variants.get_instance_specific_resource_requirements.return_value = {"num_accelerators": 4, "min_memory_mb": 98304}
                # Additional needed attributes
                mock_spec.inference_environment_variables = []
                mock_spec.hosting_resource_requirements = {"num_accelerators": 1, "min_memory_mb": 8192}
                mock_spec.dynamic_container_deployment_supported = True
        
        return mock_spec


class ImageUriAutoDetectionTest(AutoDetectionTestCase):
    """Test auto-detection for image URIs."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_neuron_instance_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that neuron instances automatically select neuron config."""
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        # First call (config_name=None) returns full spec
        # Second call (config_name="neuron") returns neuron-specific spec
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # First call for auto-detection
            self._get_mock_model_specs("neuron")  # Second call with detected config
        ]

        result = _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return neuron image
        self.assertIn("djl-inference", result)
        self.assertIn("neuronx", result)
        
        # Verify calls
        self.assertEqual(mock_verify_specs.call_count, 2)
        # First call should have config_name=None for auto-detection
        first_call_kwargs = mock_verify_specs.call_args_list[0][1]
        self.assertIsNone(first_call_kwargs.get("config_name"))
        # Second call should have detected config_name="neuron"
        second_call_kwargs = mock_verify_specs.call_args_list[1][1]
        self.assertEqual(second_call_kwargs.get("config_name"), "neuron")

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_gpu_instance_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that GPU instances automatically select TGI config."""
        mock_instance_family.return_value = "g5"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # First call for auto-detection
            self._get_mock_model_specs("tgi")  # Second call with detected config
        ]

        result = _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.g5.12xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return TGI image
        self.assertIn("huggingface-pytorch-tgi-inference", result)
        
        # Verify second call used detected config
        second_call_kwargs = mock_verify_specs.call_args_list[1][1]
        self.assertEqual(second_call_kwargs.get("config_name"), "tgi")

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_explicit_config_still_does_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that explicit config_name still goes through auto-detection but uses the explicit config."""
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        # Auto-detection should still run and confirm neuron is the right choice
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("neuron")  # Final call with explicit config
        ]

        result = _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            config_name="neuron",  # Explicit config
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should still return neuron image
        self.assertIn("djl-inference", result)
        self.assertIn("neuronx", result)
        
        # Should call verify_specs twice (auto-detection still runs)
        self.assertEqual(mock_verify_specs.call_count, 2)
        # Final call should use the detected config (which matches explicit config)
        second_call_kwargs = mock_verify_specs.call_args_list[1][1]
        self.assertEqual(second_call_kwargs.get("config_name"), "neuron")


class ModelUriAutoDetectionTest(AutoDetectionTestCase):
    """Test auto-detection for model URIs."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.model_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_neuron_instance_model_uri_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that neuron instances get correct model artifacts."""
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("neuron")  # Detected config call
        ]

        result = _retrieve_model_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            model_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return neuron artifacts path
        self.assertIn("neuron", result)
        self.assertIn("inference-prepack", result)

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.model_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_gpu_instance_model_uri_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that GPU instances get correct model artifacts."""
        mock_instance_family.return_value = "g5"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("tgi")  # Detected config call
        ]

        result = _retrieve_model_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            model_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.g5.12xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return TGI artifacts path
        self.assertIn("tgi", result)
        self.assertIn("inference-prepack", result)


class EnvironmentVariablesAutoDetectionTest(AutoDetectionTestCase):
    """Test auto-detection for environment variables."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.environment_variables.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_neuron_instance_env_vars_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that neuron instances get correct environment variables."""
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("neuron")  # Detected config call
        ]

        result = _retrieve_default_environment_variables(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            script=JumpStartScriptScope.INFERENCE,
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should contain neuron-specific environment variables
        self.assertIn("OPTION_TENSOR_PARALLEL_DEGREE", result)
        self.assertEqual(result["OPTION_TENSOR_PARALLEL_DEGREE"], "12")
        self.assertIn("OPTION_NEURON_OPTIMIZE_LEVEL", result)
        
        # Should NOT contain GPU-specific variables
        self.assertNotIn("OPTION_GPU_MEMORY_UTILIZATION", result)
        self.assertNotIn("SM_NUM_GPUS", result)

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.environment_variables.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_gpu_instance_env_vars_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that GPU instances get correct environment variables."""
        mock_instance_family.return_value = "g5"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("tgi")  # Detected config call
        ]

        result = _retrieve_default_environment_variables(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            instance_type="ml.g5.12xlarge",
            script=JumpStartScriptScope.INFERENCE,
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should contain GPU-specific environment variables
        self.assertIn("OPTION_GPU_MEMORY_UTILIZATION", result)
        self.assertEqual(result["OPTION_GPU_MEMORY_UTILIZATION"], "0.85")
        self.assertIn("SM_NUM_GPUS", result)
        
        # Should NOT contain neuron-specific variables
        self.assertNotIn("OPTION_TENSOR_PARALLEL_DEGREE", result)
        self.assertNotIn("OPTION_NEURON_OPTIMIZE_LEVEL", result)


class ResourceRequirementsAutoDetectionTest(AutoDetectionTestCase):
    """Test auto-detection for resource requirements."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.resource_requirements.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_neuron_instance_resources_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that neuron instances get correct resource requirements."""
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("neuron")  # Detected config call
        ]

        result = _retrieve_default_resources(
            model_id=self.model_id,
            model_version=self.model_version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return neuron-specific resource requirements
        self.assertEqual(result.num_accelerators, 6)
        self.assertEqual(result.min_memory, 196608)

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.resource_requirements.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_gpu_instance_resources_auto_detection(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that GPU instances get correct resource requirements."""
        mock_instance_family.return_value = "g5"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs("tgi")  # Detected config call
        ]

        result = _retrieve_default_resources(
            model_id=self.model_id,
            model_version=self.model_version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.g5.12xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should return TGI-specific resource requirements
        self.assertEqual(result.num_accelerators, 4)
        self.assertEqual(result.min_memory, 98304)


class RankingSystemTest(AutoDetectionTestCase):
    """Test that the ranking system works correctly."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_ranking_priority_respected(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that higher priority configs are selected when multiple configs support instance type."""
        # Create mock spec where both TGI and neuron support the same instance type
        mock_spec_with_both = Mock()
        mock_spec_with_both.inference_configs = Mock()
        mock_spec_with_both.inference_configs.configs = {
            "tgi": Mock(resolved_config={
                "hosting_instance_type_variants": {
                    "variants": {"g5": {"regional_properties": {"image_uri": "$tgi_image"}}}
                }
            }),
            "neuron": Mock(resolved_config={
                "hosting_instance_type_variants": {
                    "variants": {"g5": {"regional_properties": {"image_uri": "$neuron_image"}}}
                }
            })
        }
        mock_spec_with_both.inference_config_rankings = Mock()
        mock_spec_with_both.inference_config_rankings.get.return_value = Mock(rankings=["tgi", "lmi", "lmi-optimized", "neuron"])
        
        mock_instance_family.return_value = "g5"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            mock_spec_with_both,  # Auto-detection call
            self._get_mock_model_specs("tgi")  # Should select TGI (higher priority)
        ]

        result = _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.g5.12xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should select TGI (higher priority) even though neuron also supports g5
        self.assertIn("huggingface-pytorch-tgi-inference", result)
        
        # Verify TGI was selected
        second_call_kwargs = mock_verify_specs.call_args_list[1][1]
        self.assertEqual(second_call_kwargs.get("config_name"), "tgi")

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_no_ranking_fallback(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test fallback behavior when no rankings are available."""
        # Create spec without rankings
        mock_spec_no_rankings = Mock()
        mock_spec_no_rankings.inference_configs = Mock()
        mock_spec_no_rankings.inference_configs.configs = {
            "neuron": Mock(resolved_config={
                "hosting_instance_type_variants": {
                    "variants": {"inf2": {"regional_properties": {"image_uri": "$neuron_image"}}}
                }
            })
        }
        mock_spec_no_rankings.inference_config_rankings = None
        
        mock_instance_family.return_value = "inf2"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        mock_verify_specs.side_effect = [
            mock_spec_no_rankings,  # Auto-detection call
            self._get_mock_model_specs("neuron")  # Should still select neuron (first match)
        ]

        result = _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.inf2.24xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should still work and return neuron image
        self.assertIn("djl-inference", result)
        self.assertIn("neuronx", result)


class EdgeCaseTest(AutoDetectionTestCase):
    """Test edge cases and error conditions."""

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    def test_no_instance_type_skips_auto_detection(self, mock_verify_specs, mock_validate):
        """Test that missing instance_type skips auto-detection."""
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        mock_verify_specs.return_value = self._get_mock_model_specs()

        _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type=None,  # No instance type
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Should only call verify_specs once with original config_name (None)
        self.assertEqual(mock_verify_specs.call_count, 1)
        call_kwargs = mock_verify_specs.call_args_list[0][1]
        self.assertIsNone(call_kwargs.get("config_name"))

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
    @patch("sagemaker.utils.get_instance_type_family")
    def test_unsupported_instance_type_uses_default(self, mock_instance_family, mock_verify_specs, mock_validate):
        """Test that unsupported instance types fall back to default config."""
        mock_instance_family.return_value = "unsupported"
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        
        # Auto-detection should find no matching configs and use default
        mock_verify_specs.side_effect = [
            self._get_mock_model_specs(),  # Auto-detection call
            self._get_mock_model_specs()  # Default call (config_name=None)
        ]

        _retrieve_image_uri(
            model_id=self.model_id,
            model_version=self.model_version,
            image_scope=JumpStartScriptScope.INFERENCE,
            region=self.region,
            instance_type="ml.unsupported.xlarge",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

        # Second call should still have config_name=None (no match found)
        second_call_kwargs = mock_verify_specs.call_args_list[1][1]
        self.assertIsNone(second_call_kwargs.get("config_name"))


if __name__ == "__main__":
    unittest.main()