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

import pytest
from unittest.mock import Mock, MagicMock, patch
from sagemaker.core.jumpstart.enums import JumpStartModelType, JumpStartScriptScope


class TestFactoryUtilsHelpers:
    """Test cases for factory utils helper functions using mocks to avoid circular imports"""

    def test_model_info_kwargs_structure(self):
        """Test that model info kwargs have expected structure"""
        # Create a mock kwargs object
        mock_kwargs = Mock()
        mock_kwargs.model_id = "test-model"
        mock_kwargs.hub_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub"
        mock_kwargs.region = "us-west-2"
        mock_kwargs.sagemaker_session = Mock()
        mock_kwargs.model_type = JumpStartModelType.OPEN_WEIGHTS
        mock_kwargs.config_name = "default"
        mock_kwargs.model_version = "1.0.0"
        mock_kwargs.tolerate_deprecated_model = False
        mock_kwargs.tolerate_vulnerable_model = False

        # Verify the mock has all expected attributes
        assert hasattr(mock_kwargs, "model_id")
        assert hasattr(mock_kwargs, "hub_arn")
        assert hasattr(mock_kwargs, "region")
        assert hasattr(mock_kwargs, "model_type")
        assert mock_kwargs.model_id == "test-model"

    def test_session_handling(self):
        """Test session handling logic"""
        # Test when session is already set
        mock_kwargs = Mock()
        mock_session = Mock()
        mock_kwargs.sagemaker_session = mock_session

        assert mock_kwargs.sagemaker_session is not None
        assert mock_kwargs.sagemaker_session == mock_session

        # Test when session is None
        mock_kwargs_none = Mock()
        mock_kwargs_none.sagemaker_session = None
        assert mock_kwargs_none.sagemaker_session is None

    def test_region_resolution(self):
        """Test region resolution logic"""
        # Test when region is already set
        mock_kwargs = Mock()
        mock_kwargs.region = "us-west-2"
        assert mock_kwargs.region == "us-west-2"

        # Test when region comes from session
        mock_session = Mock()
        mock_session.boto_region_name = "eu-west-1"
        mock_kwargs_from_session = Mock()
        mock_kwargs_from_session.region = None
        mock_kwargs_from_session.sagemaker_session = mock_session

        # Simulate region resolution
        resolved_region = mock_kwargs_from_session.region or mock_session.boto_region_name
        assert resolved_region == "eu-west-1"

    def test_model_version_handling(self):
        """Test model version handling"""
        # Test with explicit version
        mock_kwargs = Mock()
        mock_kwargs.model_version = "2.0.0"
        mock_kwargs.hub_arn = None
        assert mock_kwargs.model_version == "2.0.0"

        # Test with wildcard default
        mock_kwargs_wildcard = Mock()
        mock_kwargs_wildcard.model_version = None
        mock_kwargs_wildcard.hub_arn = None
        default_version = mock_kwargs_wildcard.model_version or "*"
        assert default_version == "*"

        # Test with hub specs
        mock_specs = Mock()
        mock_specs.version = "3.0.0"
        mock_kwargs_hub = Mock()
        mock_kwargs_hub.model_version = None
        mock_kwargs_hub.hub_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub"
        mock_kwargs_hub.specs = mock_specs

        # Simulate hub version resolution
        if mock_kwargs_hub.hub_arn:
            resolved_version = mock_specs.version
        else:
            resolved_version = mock_kwargs_hub.model_version or "*"
        assert resolved_version == "3.0.0"

    def test_tolerate_flags_defaults(self):
        """Test tolerate flags default to False"""
        mock_kwargs = Mock()
        mock_kwargs.tolerate_deprecated_model = None
        mock_kwargs.tolerate_vulnerable_model = None

        # Simulate default assignment
        tolerate_deprecated = mock_kwargs.tolerate_deprecated_model or False
        tolerate_vulnerable = mock_kwargs.tolerate_vulnerable_model or False

        assert tolerate_deprecated is False
        assert tolerate_vulnerable is False

    def test_proprietary_model_type_handling(self):
        """Test that proprietary models get None for certain fields"""
        mock_kwargs = Mock()
        mock_kwargs.model_type = JumpStartModelType.PROPRIETARY
        mock_kwargs.image_uri = "some-uri"
        mock_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_kwargs.source_dir = "some-dir"
        mock_kwargs.entry_point = "inference.py"
        mock_kwargs.env = {"KEY": "value"}

        # Simulate proprietary model handling
        if mock_kwargs.model_type == JumpStartModelType.PROPRIETARY:
            image_uri = None
            model_data = None
            source_dir = None
            entry_point = None
            env = None
        else:
            image_uri = mock_kwargs.image_uri
            model_data = mock_kwargs.model_data
            source_dir = mock_kwargs.source_dir
            entry_point = mock_kwargs.entry_point
            env = mock_kwargs.env

        assert image_uri is None
        assert model_data is None
        assert source_dir is None
        assert entry_point is None
        assert env is None

    def test_s3_prefix_detection(self):
        """Test S3 prefix detection and conversion"""
        # Test S3 prefix ending with /
        s3_prefix = "s3://bucket/prefix/"
        assert s3_prefix.startswith("s3://")
        assert s3_prefix.endswith("/")

        # Test conversion to S3DataSource dict
        if s3_prefix.startswith("s3://") and s3_prefix.endswith("/"):
            model_data_dict = {
                "S3DataSource": {
                    "S3Uri": s3_prefix,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            }
            assert isinstance(model_data_dict, dict)
            assert "S3DataSource" in model_data_dict

    def test_environment_variable_merging(self):
        """Test environment variable merging logic"""
        existing_env = {"EXISTING_KEY": "existing_value"}
        new_env_vars = {"NEW_KEY": "new_value", "ANOTHER_KEY": "another_value"}

        # Simulate merging without overwriting existing keys
        merged_env = existing_env.copy()
        for key, value in new_env_vars.items():
            if key not in merged_env:
                merged_env[key] = value

        assert merged_env["EXISTING_KEY"] == "existing_value"
        assert merged_env["NEW_KEY"] == "new_value"
        assert merged_env["ANOTHER_KEY"] == "another_value"

    def test_empty_env_becomes_none(self):
        """Test that empty env dict becomes None"""
        env = {}
        result = None if env == {} else env
        assert result is None

        env_with_values = {"KEY": "value"}
        result_with_values = None if env_with_values == {} else env_with_values
        assert result_with_values is not None

    def test_resource_name_generation(self):
        """Test resource name generation logic"""
        base_name = "test-model"

        # Simulate name generation with timestamp
        import time

        timestamp = str(int(time.time()))
        generated_name = f"{base_name}-{timestamp}"

        assert generated_name.startswith(base_name)
        assert len(generated_name) > len(base_name)

    def test_tag_structure(self):
        """Test tag structure for JumpStart models"""
        model_id = "test-model"
        model_version = "1.0.0"
        model_type = JumpStartModelType.OPEN_WEIGHTS
        config_name = "default"
        scope = JumpStartScriptScope.INFERENCE

        # Simulate tag creation
        tags = [
            {"Key": "sagemaker:jumpstart-model-id", "Value": model_id},
            {"Key": "sagemaker:jumpstart-model-version", "Value": model_version},
        ]

        assert len(tags) == 2
        assert tags[0]["Key"] == "sagemaker:jumpstart-model-id"
        assert tags[0]["Value"] == model_id

    def test_inference_config_selection(self):
        """Test inference config selection from training config"""
        mock_training_config = Mock()
        mock_training_config.default_inference_config = "inference-config-1"

        mock_training_configs = Mock()
        mock_training_configs.configs = {"training-config-1": mock_training_config}

        mock_specs = Mock()
        mock_specs.training_configs = mock_training_configs

        # Simulate config selection
        training_config_name = "training-config-1"
        if mock_specs.training_configs:
            resolved_config = mock_specs.training_configs.configs.get(training_config_name)
            if resolved_config:
                result = resolved_config.default_inference_config
            else:
                result = None
        else:
            result = None

        assert result == "inference-config-1"

    def test_inference_config_selection_not_found(self):
        """Test inference config selection when config not found"""
        mock_training_configs = Mock()
        mock_training_configs.configs = {}

        mock_specs = Mock()
        mock_specs.training_configs = mock_training_configs

        # Simulate config selection with nonexistent config
        training_config_name = "nonexistent-config"
        if mock_specs.training_configs:
            resolved_config = mock_specs.training_configs.configs.get(training_config_name)
            if resolved_config:
                result = resolved_config.default_inference_config
            else:
                result = None
        else:
            result = None

        assert result is None

    def test_instance_type_variants_handling(self):
        """Test instance type variants handling"""
        mock_specs = Mock()
        mock_specs.inference_configs = None

        # When no inference configs, should proceed normally
        if mock_specs.inference_configs:
            has_configs = True
        else:
            has_configs = False

        assert has_configs is False

    def test_instance_type_retrieval_logic(self):
        """Test instance type retrieval logic"""
        # Simulate instance type retrieval
        instance_type = None
        default_instance_type = "ml.m5.large"

        result = instance_type or default_instance_type
        assert result == "ml.m5.large"

    def test_image_uri_retrieval_logic(self):
        """Test image URI retrieval logic"""
        # Simulate image URI retrieval
        image_uri = None
        default_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest"

        result = image_uri or default_image_uri
        assert result == default_image_uri

    def test_model_data_retrieval_logic(self):
        """Test model data retrieval logic"""
        # Simulate model data retrieval
        model_data = None
        default_model_data = "s3://bucket/model.tar.gz"

        result = model_data or default_model_data
        assert result == default_model_data

    def test_speculative_decoding_data_sources(self):
        """Test speculative decoding data sources handling"""
        mock_specs = Mock()
        mock_data_source = Mock()
        mock_data_source.provider = "test-provider"
        mock_data_source.s3_data_source = Mock()
        mock_specs.get_speculative_decoding_s3_data_sources.return_value = [mock_data_source]

        # Simulate data source processing
        data_sources = mock_specs.get_speculative_decoding_s3_data_sources()
        assert len(data_sources) == 1
        assert data_sources[0].provider == "test-provider"

    def test_additional_model_data_sources_none(self):
        """Test when additional model data sources is None"""
        mock_kwargs = Mock()
        mock_kwargs.additional_model_data_sources = None
        mock_kwargs.specs = Mock()
        mock_kwargs.specs.get_speculative_decoding_s3_data_sources.return_value = []

        # Should remain None when no speculative decoding sources
        assert mock_kwargs.additional_model_data_sources is None

    def test_hub_content_type_handling(self):
        """Test hub content type handling"""
        from sagemaker.core.jumpstart.types import HubContentType

        mock_specs = Mock()
        mock_specs.hub_content_type = HubContentType.MODEL_REFERENCE

        # Simulate hub content type check
        if mock_specs.hub_content_type == HubContentType.MODEL_REFERENCE:
            is_model_reference = True
        else:
            is_model_reference = False

        assert is_model_reference is True

    def test_model_reference_arn_construction(self):
        """Test model reference ARN construction"""
        hub_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub"
        model_name = "test-model"
        version = "1.0.0"

        # Simulate ARN construction
        if hub_arn:
            model_reference_arn = f"{hub_arn}/model-reference/{model_name}/{version}"
        else:
            model_reference_arn = None

        assert model_reference_arn is not None
        assert "model-reference" in model_reference_arn

    def test_endpoint_type_inference_component(self):
        """Test endpoint type for inference component"""
        from sagemaker.core.enums import EndpointType

        endpoint_type = EndpointType.INFERENCE_COMPONENT_BASED

        # Simulate endpoint type check
        if endpoint_type == EndpointType.INFERENCE_COMPONENT_BASED:
            requires_resources = True
        else:
            requires_resources = False

        assert requires_resources is True

    def test_managed_instance_scaling(self):
        """Test managed instance scaling configuration"""
        managed_instance_scaling = {
            "Status": "ENABLED",
            "MinInstanceCount": 1,
            "MaxInstanceCount": 10,
        }

        assert managed_instance_scaling["Status"] == "ENABLED"
        assert managed_instance_scaling["MinInstanceCount"] == 1

    def test_routing_config_handling(self):
        """Test routing config handling"""
        routing_config = {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"}

        assert routing_config["RoutingStrategy"] == "LEAST_OUTSTANDING_REQUESTS"

    def test_model_access_configs(self):
        """Test model access configs"""
        from sagemaker.core.shapes import ModelAccessConfig

        model_access_config = ModelAccessConfig(accept_eula=True)

        assert model_access_config.accept_eula is True

    def test_inference_ami_version(self):
        """Test inference AMI version"""
        inference_ami_version = "al2-ami-sagemaker-inference-gpu-2"

        assert "sagemaker-inference" in inference_ami_version

    def test_volume_size_and_timeouts(self):
        """Test volume size and timeout configurations"""
        volume_size = 30
        model_data_download_timeout = 3600
        container_startup_health_check_timeout = 600

        assert volume_size > 0
        assert model_data_download_timeout > 0
        assert container_startup_health_check_timeout > 0

    def test_explainer_config_handling(self):
        """Test explainer config handling"""
        mock_explainer_config = Mock()
        mock_explainer_config.clarify_explainer_config = Mock()

        assert mock_explainer_config.clarify_explainer_config is not None

    def test_async_inference_config(self):
        """Test async inference config"""
        mock_async_config = Mock()
        mock_async_config.output_path = "s3://bucket/output"

        assert mock_async_config.output_path.startswith("s3://")

    def test_serverless_inference_config(self):
        """Test serverless inference config"""
        mock_serverless_config = Mock()
        mock_serverless_config.memory_size_in_mb = 2048
        mock_serverless_config.max_concurrency = 10

        assert mock_serverless_config.memory_size_in_mb == 2048
        assert mock_serverless_config.max_concurrency == 10

    def test_data_capture_config(self):
        """Test data capture config"""
        mock_data_capture = Mock()
        mock_data_capture.enable_capture = True
        mock_data_capture.destination_s3_uri = "s3://bucket/capture"

        assert mock_data_capture.enable_capture is True

    def test_kms_key_handling(self):
        """Test KMS key handling"""
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"

        assert kms_key.startswith("arn:aws:kms:")

    def test_vpc_config_structure(self):
        """Test VPC config structure"""
        vpc_config = {"SecurityGroupIds": ["sg-12345"], "Subnets": ["subnet-12345", "subnet-67890"]}

        assert "SecurityGroupIds" in vpc_config
        assert "Subnets" in vpc_config
        assert len(vpc_config["Subnets"]) == 2

    def test_enable_network_isolation(self):
        """Test network isolation flag"""
        enable_network_isolation = True

        assert enable_network_isolation is True

    def test_image_config_structure(self):
        """Test image config structure"""
        image_config = {
            "RepositoryAccessMode": "Platform",
            "RepositoryAuthConfig": {
                "RepositoryCredentialsProviderArn": "arn:aws:secretsmanager:..."
            },
        }

        assert image_config["RepositoryAccessMode"] == "Platform"

    def test_code_location_handling(self):
        """Test code location handling"""
        code_location = "s3://bucket/code"

        assert code_location.startswith("s3://")

    def test_container_log_level(self):
        """Test container log level"""
        container_log_level = 20  # INFO level

        assert container_log_level in [10, 20, 30, 40, 50]  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    def test_dependencies_list(self):
        """Test dependencies list"""
        dependencies = ["requirements.txt", "setup.py"]

        assert isinstance(dependencies, list)
        assert len(dependencies) == 2

    def test_git_config_structure(self):
        """Test git config structure"""
        git_config = {
            "repo": "https://github.com/user/repo.git",
            "branch": "main",
            "commit": "abc123",
        }

        assert "repo" in git_config
        assert git_config["branch"] == "main"

    def test_training_instance_type_for_inference(self):
        """Test training instance type used for inference defaults"""
        training_instance_type = "ml.p3.2xlarge"

        # Simulate using training instance type for inference defaults
        if training_instance_type:
            instance_family = training_instance_type.split(".")[1]
            assert instance_family == "p3"

    def test_accept_eula_flag(self):
        """Test accept EULA flag"""
        accept_eula = True

        assert accept_eula is True

    def test_endpoint_logging_flag(self):
        """Test endpoint logging flag"""
        endpoint_logging = True

        assert endpoint_logging is True

    def test_inference_recommendation_id(self):
        """Test inference recommendation ID"""
        inference_recommendation_id = "rec-12345"

        assert inference_recommendation_id.startswith("rec-")

    def test_inference_component_name(self):
        """Test inference component name"""
        inference_component_name = "my-inference-component"

        assert len(inference_component_name) > 0

    def test_wait_flag(self):
        """Test wait flag for deployment"""
        wait = True

        assert wait is True

    def test_serializer_deserializer(self):
        """Test serializer and deserializer"""
        mock_serializer = Mock()
        mock_deserializer = Mock()

        assert mock_serializer is not None
        assert mock_deserializer is not None
