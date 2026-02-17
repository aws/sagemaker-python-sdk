"""
Unit tests for ModelBuilder compute requirements resolution.
Tests the _resolve_compute_requirements method with various scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements


class TestComputeRequirementsResolution(unittest.TestCase):
    """Test compute requirements resolution - Requirements 2.1, 3.1, 3.2, 3.4"""

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

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_resolve_with_defaults_only(self, mock_get_resources, mock_fetch_hub):
        """Test resolving compute requirements with only JumpStart defaults."""
        # Setup: Hub document with default compute requirements
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)  # 8 CPUs, 32GB RAM
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.2xlarge",
            user_resource_requirements=None
        )
        
        # Verify: Should use defaults from JumpStart
        assert requirements.number_of_cpu_cores_required == 4
        assert requirements.min_memory_required_in_mb == 8192
        # Check that accelerator count is not set (should be Unassigned)
        from sagemaker.core.utils.utils import Unassigned
        assert isinstance(requirements.number_of_accelerator_devices_required, Unassigned)

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_resolve_with_user_override(self, mock_get_resources, mock_fetch_hub):
        """Test that user-provided requirements take precedence over defaults."""
        # Setup: Hub document with defaults
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User provides custom requirements
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 8,
                "memory": 16384,
                "num_accelerators": 2
            },
            limits={
                "memory": 32768
            }
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.g5.12xlarge",
            user_resource_requirements=user_requirements
        )
        
        # Verify: Should use user-provided values
        assert requirements.number_of_cpu_cores_required == 8
        assert requirements.min_memory_required_in_mb == 16384
        assert requirements.max_memory_required_in_mb == 32768
        assert requirements.number_of_accelerator_devices_required == 2

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_resolve_with_partial_user_override(self, mock_get_resources, mock_fetch_hub):
        """Test merging user requirements with defaults (partial override)."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User only overrides memory
        user_requirements = ResourceRequirements(
            requests={
                "memory": 16384
            }
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.2xlarge",
            user_resource_requirements=user_requirements
        )
        
        # Verify: Should use user memory, default CPUs
        assert requirements.number_of_cpu_cores_required == 4  # From default
        assert requirements.min_memory_required_in_mb == 16384  # From user

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_infer_accelerator_count_for_gpu_instance(self, mock_get_resources, mock_fetch_hub):
        """Test automatic accelerator count inference for GPU instances."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                        # No accelerator count specified
                    }
                }
            ]
        }
        mock_get_resources.return_value = (48, 196608)  # g5.12xlarge specs
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.g5.12xlarge",
            user_resource_requirements=None
        )
        
        # Verify: Should automatically infer 4 GPUs for g5.12xlarge
        assert requirements.number_of_accelerator_devices_required == 4

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_error_when_gpu_instance_accelerator_count_unknown(self, mock_get_resources, mock_fetch_hub):
        """Test error when GPU instance type has unknown accelerator count."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute with a real GPU instance type that's not in our mapping
        # Use ml.g5.metal which is a valid GPU instance pattern but not in our map
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.g5.metal",  # Valid GPU pattern but not in mapping
                user_resource_requirements=None
            )
        
        # Verify error message
        error_msg = str(exc_info.value)
        assert "requires accelerator device count specification" in error_msg
        assert "ResourceRequirements" in error_msg
        assert "num_accelerators" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_validate_cpu_requirements_exceed_instance_capacity(self, mock_get_resources, mock_fetch_hub):
        """Test validation error when CPU requirements exceed instance capacity."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (2, 8192)  # Only 2 CPUs available
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more CPUs than available
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 16,  # More than available
                "memory": 8192
            }
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.t3.small",
                user_resource_requirements=user_requirements
            )
        
        error_msg = str(exc_info.value)
        assert "Resource requirements incompatible" in error_msg
        assert "16 CPUs" in error_msg
        assert "2 CPUs" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_validate_memory_requirements_exceed_instance_capacity(self, mock_get_resources, mock_fetch_hub):
        """Test validation error when memory requirements exceed instance capacity."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 8192)  # Only 8GB RAM
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more memory than available
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 4,
                "memory": 32768  # More than available
            }
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.m5.large",
                user_resource_requirements=user_requirements
            )
        
        error_msg = str(exc_info.value)
        assert "Resource requirements incompatible" in error_msg
        assert "32768 MB memory" in error_msg
        assert "8192 MB memory" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_adjust_cpu_count_when_default_exceeds_capacity(self, mock_get_resources, mock_fetch_hub):
        """Test automatic CPU adjustment when default exceeds instance capacity."""
        # Setup: Default requests 8 CPUs but instance only has 4
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 8,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (4, 16384)  # Only 4 CPUs
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"  # Provide instance_type to avoid auto-detection
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.xlarge",
            user_resource_requirements=None
        )
        
        # Verify: Should adjust to instance capacity
        assert requirements.number_of_cpu_cores_required == 4

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_adjust_memory_when_default_exceeds_capacity(self, mock_get_resources, mock_fetch_hub):
        """Test automatic memory adjustment when default exceeds instance capacity."""
        # Setup: Default requests 32GB but instance only has 8GB
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 32768  # 32GB requested
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 8192)  # Only 8GB RAM
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.large"  # Provide instance_type to avoid auto-detection
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.large",
            user_resource_requirements=None
        )
        
        # Verify: Should adjust to instance capacity
        assert requirements.min_memory_required_in_mb == 8192

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_missing_hosting_configs_error(self, mock_get_resources, mock_fetch_hub):
        """Test error when hub document has no hosting configs."""
        # Setup: No hosting configs
        mock_fetch_hub.return_value = {}
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.m5.xlarge",
                user_resource_requirements=None
            )
        
        error_msg = str(exc_info.value)
        assert "Unable to resolve compute requirements" in error_msg
        assert "does not have hosting configuration" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_various_gpu_instance_types(self, mock_get_resources, mock_fetch_hub):
        """Test accelerator count inference for various GPU instance types."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Test various GPU instance types
        test_cases = [
            ("ml.g5.xlarge", 1),
            ("ml.g5.12xlarge", 4),
            ("ml.g5.48xlarge", 8),
            ("ml.p3.2xlarge", 1),
            ("ml.p3.8xlarge", 4),
            ("ml.p4d.24xlarge", 8),
            ("ml.g4dn.xlarge", 1),
            ("ml.g4dn.12xlarge", 4),
        ]
        
        for instance_type, expected_gpus in test_cases:
            mock_get_resources.return_value = (8, 32768)
            
            requirements = builder._resolve_compute_requirements(
                instance_type=instance_type,
                user_resource_requirements=None
            )
            
            assert requirements.number_of_accelerator_devices_required == expected_gpus, \
                f"Expected {expected_gpus} GPUs for {instance_type}, got {requirements.number_of_accelerator_devices_required}"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_cpu_instance_no_accelerator_count(self, mock_get_resources, mock_fetch_hub):
        """Test that CPU instances don't get accelerator count."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute with CPU instance
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.2xlarge",
            user_resource_requirements=None
        )
        
        # Verify: Should not have accelerator count
        from sagemaker.core.utils.utils import Unassigned
        assert isinstance(requirements.number_of_accelerator_devices_required, Unassigned)

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_default_accelerator_count_from_metadata(self, mock_get_resources, mock_fetch_hub):
        """Test using default accelerator count from JumpStart metadata."""
        # Setup: Metadata includes accelerator count
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192,
                        "NumberOfAcceleratorDevicesRequired": 2
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.g5.12xlarge",
            user_resource_requirements=None
        )
        
        # Verify: Should use metadata value, not inferred value
        assert requirements.number_of_accelerator_devices_required == 2


    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_missing_accelerator_count_for_unknown_gpu_instance(self, mock_get_resources, mock_fetch_hub):
        """Test error when GPU instance type has no accelerator count in metadata or mapping."""
        # Setup: No accelerator count in metadata
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                        # No NumberOfAcceleratorDevicesRequired
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute with a GPU instance not in the mapping
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.g5.unknown",  # Not in mapping
                user_resource_requirements=None
            )
        
        # Verify error message provides guidance
        error_msg = str(exc_info.value)
        assert "requires accelerator device count specification" in error_msg
        assert "ResourceRequirements" in error_msg
        assert "num_accelerators" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_incompatible_accelerator_requirements(self, mock_get_resources, mock_fetch_hub):
        """Test validation when user requests more accelerators than available."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more accelerators than ml.g5.xlarge has (1 GPU)
        user_requirements = ResourceRequirements(
            requests={
                "num_accelerators": 8,  # More than available
                "memory": 8192
            }
        )
        
        # Execute - should succeed but with warning (we don't validate accelerator count against instance)
        # This is because accelerator validation is complex and AWS will validate at deployment time
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.g5.xlarge",
            user_resource_requirements=user_requirements
        )
        
        # Verify: Should use user-provided accelerator count
        assert requirements.number_of_accelerator_devices_required == 8

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_validation_error_message_format_cpu(self, mock_get_resources, mock_fetch_hub):
        """Test that CPU validation error messages are properly formatted."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (4, 16384)  # 4 CPUs available
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more CPUs than available
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 16,
                "memory": 8192
            }
        )
        
        # Execute and verify error message format
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.m5.xlarge",
                user_resource_requirements=user_requirements
            )
        
        error_msg = str(exc_info.value)
        # Verify error message contains all required information
        assert "Resource requirements incompatible" in error_msg
        assert "ml.m5.xlarge" in error_msg
        assert "Requested: 16 CPUs" in error_msg
        assert "Available: 4 CPUs" in error_msg
        assert "reduce CPU requirements" in error_msg or "larger instance type" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_validation_error_message_format_memory(self, mock_get_resources, mock_fetch_hub):
        """Test that memory validation error messages are properly formatted."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 16384)  # 16GB available
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more memory than available
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 4,
                "memory": 65536  # 64GB requested, only 16GB available
            }
        )
        
        # Execute and verify error message format
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.m5.2xlarge",
                user_resource_requirements=user_requirements
            )
        
        error_msg = str(exc_info.value)
        # Verify error message contains all required information
        assert "Resource requirements incompatible" in error_msg
        assert "ml.m5.2xlarge" in error_msg
        assert "Requested: 65536 MB memory" in error_msg
        assert "Available: 16384 MB memory" in error_msg
        assert "reduce memory requirements" in error_msg or "larger instance type" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_accelerator_error_message_includes_example_code(self, mock_get_resources, mock_fetch_hub):
        """Test that accelerator count error includes example code snippet."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute with unknown GPU instance
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.g5.custom",  # Not in mapping
                user_resource_requirements=None
            )
        
        error_msg = str(exc_info.value)
        # Verify error message includes example code
        assert "ResourceRequirements" in error_msg
        assert "num_accelerators" in error_msg
        assert "requests" in error_msg
        # Should show how to create ResourceRequirements
        assert "from sagemaker.core.inference_config import ResourceRequirements" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_both_cpu_and_memory_incompatible(self, mock_get_resources, mock_fetch_hub):
        """Test error when both CPU and memory requirements exceed capacity."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (2, 4096)  # Small instance
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User requests more than available
        user_requirements = ResourceRequirements(
            requests={
                "num_cpus": 8,  # More than 2 available
                "memory": 16384  # More than 4096 available
            }
        )
        
        # Execute - should fail on CPU first (checked first in code)
        with pytest.raises(ValueError) as exc_info:
            builder._resolve_compute_requirements(
                instance_type="ml.t3.small",
                user_resource_requirements=user_requirements
            )
        
        error_msg = str(exc_info.value)
        # Should report CPU incompatibility (checked first)
        assert "Resource requirements incompatible" in error_msg
        assert "CPUs" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_zero_accelerator_count_explicit(self, mock_get_resources, mock_fetch_hub):
        """Test that explicitly setting 0 accelerators works for CPU instances."""
        # Setup
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192
                    }
                }
            ]
        }
        mock_get_resources.return_value = (8, 32768)
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # User explicitly sets 0 accelerators
        user_requirements = ResourceRequirements(
            requests={
                "num_accelerators": 0,
                "num_cpus": 4,
                "memory": 8192
            }
        )
        
        # Execute
        requirements = builder._resolve_compute_requirements(
            instance_type="ml.m5.2xlarge",
            user_resource_requirements=user_requirements
        )
        
        # Verify: Should accept 0 accelerators
        assert requirements.number_of_accelerator_devices_required == 0


if __name__ == "__main__":
    unittest.main()
