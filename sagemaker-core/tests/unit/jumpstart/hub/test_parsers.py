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
from unittest.mock import Mock, patch
from sagemaker.core.jumpstart.hub.parsers import (
    _to_json,
    get_model_spec_arg_keys,
    get_model_spec_kwargs_from_hub_model_document,
    make_model_specs_from_describe_hub_content_response,
)
from sagemaker.core.jumpstart.enums import ModelSpecKwargType, NamingConventionType
from sagemaker.core.jumpstart.types import HubContentType, JumpStartDataHolderType
from sagemaker.core.jumpstart.hub.interfaces import DescribeHubContentResponse, HubModelDocument


class MockDataHolder(JumpStartDataHolderType):
    """Mock data holder for testing"""

    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {"value": self.value}


class TestParsers:
    """Test cases for hub parsers"""

    def test_to_json_simple_dict(self):
        """Test _to_json with simple dictionary"""
        data = {"key1": "value1", "key2": 123}

        result = _to_json(data)

        assert result == data

    def test_to_json_with_data_holder(self):
        """Test _to_json with JumpStartDataHolderType"""
        data = {"holder": MockDataHolder("test")}

        result = _to_json(data)

        assert "holder" in result
        assert result["holder"]["Value"] == "test"

    def test_to_json_with_list_of_data_holders(self):
        """Test _to_json with list containing data holders"""
        data = {"holders": [MockDataHolder("test1"), MockDataHolder("test2")]}

        result = _to_json(data)

        assert len(result["holders"]) == 2
        assert result["holders"][0]["Value"] == "test1"
        assert result["holders"][1]["Value"] == "test2"

    def test_to_json_with_nested_dict(self):
        """Test _to_json with nested dictionary containing data holders"""
        data = {"nested": {"holder": MockDataHolder("nested_value")}}

        result = _to_json(data)

        assert result["nested"]["holder"]["Value"] == "nested_value"

    def test_get_model_spec_arg_keys_deploy(self):
        """Test get_model_spec_arg_keys for DEPLOY type"""
        keys = get_model_spec_arg_keys(ModelSpecKwargType.DEPLOY)

        assert "ModelDataDownloadTimeout" in keys
        assert "ContainerStartupHealthCheckTimeout" in keys
        assert "InferenceAmiVersion" in keys

    def test_get_model_spec_arg_keys_deploy_snake_case(self):
        """Test get_model_spec_arg_keys for DEPLOY type with snake_case"""
        keys = get_model_spec_arg_keys(
            ModelSpecKwargType.DEPLOY, naming_convention=NamingConventionType.SNAKE_CASE
        )

        assert "model_data_download_timeout" in keys
        assert "container_startup_health_check_timeout" in keys
        assert "inference_ami_version" in keys

    def test_get_model_spec_arg_keys_estimator(self):
        """Test get_model_spec_arg_keys for ESTIMATOR type"""
        keys = get_model_spec_arg_keys(ModelSpecKwargType.ESTIMATOR)

        assert "EncryptInterContainerTraffic" in keys
        assert "MaxRuntimeInSeconds" in keys
        assert "DisableOutputCompression" in keys
        assert "ModelDir" in keys

    def test_get_model_spec_arg_keys_model(self):
        """Test get_model_spec_arg_keys for MODEL type"""
        keys = get_model_spec_arg_keys(ModelSpecKwargType.MODEL)

        assert len(keys) == 0

    def test_get_model_spec_arg_keys_fit(self):
        """Test get_model_spec_arg_keys for FIT type"""
        keys = get_model_spec_arg_keys(ModelSpecKwargType.FIT)

        assert len(keys) == 0

    def test_get_model_spec_arg_keys_invalid_convention(self):
        """Test get_model_spec_arg_keys raises error for invalid naming convention"""
        with pytest.raises(ValueError, match="valid naming convention"):
            get_model_spec_arg_keys(ModelSpecKwargType.DEPLOY, naming_convention="invalid")

    def test_get_model_spec_kwargs_from_hub_model_document_deploy(self):
        """Test get_model_spec_kwargs_from_hub_model_document for DEPLOY"""
        document = {
            "ModelDataDownloadTimeout": 600,
            "ContainerStartupHealthCheckTimeout": 300,
            "InferenceAmiVersion": "1.0",
            "OtherField": "ignored",
        }

        result = get_model_spec_kwargs_from_hub_model_document(ModelSpecKwargType.DEPLOY, document)

        assert result["ModelDataDownloadTimeout"] == 600
        assert result["ContainerStartupHealthCheckTimeout"] == 300
        assert result["InferenceAmiVersion"] == "1.0"
        assert "OtherField" not in result

    def test_get_model_spec_kwargs_from_hub_model_document_empty(self):
        """Test get_model_spec_kwargs_from_hub_model_document with no matching keys"""
        document = {"OtherField": "value"}

        result = get_model_spec_kwargs_from_hub_model_document(ModelSpecKwargType.DEPLOY, document)

        assert len(result) == 0

    def test_get_model_spec_kwargs_from_hub_model_document_partial(self):
        """Test get_model_spec_kwargs_from_hub_model_document with partial keys"""
        document = {"ModelDataDownloadTimeout": 600, "OtherField": "ignored"}

        result = get_model_spec_kwargs_from_hub_model_document(ModelSpecKwargType.DEPLOY, document)

        assert result["ModelDataDownloadTimeout"] == 600
        assert len(result) == 1

    def test_make_model_specs_from_describe_hub_content_response_minimal(self):
        """Test make_model_specs_from_describe_hub_content_response with minimal data"""
        response = Mock(spec=DescribeHubContentResponse)
        response.hub_content_type = HubContentType.MODEL
        response.hub_content_name = "test-model"
        response.hub_content_version = "1.0"
        response.get_hub_region = Mock(return_value="us-west-2")

        hub_doc = Mock(spec=HubModelDocument)
        hub_doc.url = "https://example.com/model"
        hub_doc.min_sdk_version = "2.0"
        hub_doc.model_types = ["text-generation"]
        hub_doc.capabilities = ["inference"]
        hub_doc.training_supported = False
        hub_doc.incremental_training_supported = False
        hub_doc.hosting_ecr_uri = "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
        hub_doc.inference_configs = []
        hub_doc.inference_config_components = {}
        hub_doc.inference_config_rankings = []
        hub_doc.hosting_artifact_uri = None
        hub_doc.hosting_script_uri = None
        hub_doc.inference_environment_variables = []
        hub_doc.inference_dependencies = []
        hub_doc.default_inference_instance_type = "ml.m5.xlarge"
        hub_doc.supported_inference_instance_types = ["ml.m5.xlarge"]
        hub_doc.dynamic_container_deployment_supported = False
        hub_doc.hosting_resource_requirements = {}
        hub_doc.hosting_prepacked_artifact_uri = None
        hub_doc.sage_maker_sdk_predictor_specifications = {}
        hub_doc.default_payloads = None
        hub_doc.gated_bucket = False
        hub_doc.inference_volume_size = 30
        hub_doc.inference_enable_network_isolation = False
        hub_doc.resource_name_base = "test-model"
        hub_doc.hosting_eula_uri = None
        hub_doc.hosting_model_package_arn = None
        hub_doc.model_subscription_link = None
        hub_doc.hosting_use_script_uri = False
        hub_doc.hosting_instance_type_variants = {}
        hub_doc.to_json = Mock(return_value={})

        response.hub_content_document = hub_doc

        result = make_model_specs_from_describe_hub_content_response(response)

        assert result is not None

    def test_make_model_specs_from_describe_hub_content_response_invalid_type(self):
        """Test make_model_specs_from_describe_hub_content_response with invalid content type"""
        response = Mock(spec=DescribeHubContentResponse)
        response.hub_content_type = "INVALID_TYPE"

        with pytest.raises(AttributeError, match="Invalid content type"):
            make_model_specs_from_describe_hub_content_response(response)

    def test_make_model_specs_from_describe_hub_content_response_with_artifacts(self):
        """Test make_model_specs_from_describe_hub_content_response with artifact URIs"""
        response = Mock(spec=DescribeHubContentResponse)
        response.hub_content_type = HubContentType.MODEL
        response.hub_content_name = "test-model"
        response.hub_content_version = "1.0"
        response.get_hub_region = Mock(return_value="us-west-2")

        hub_doc = Mock(spec=HubModelDocument)
        hub_doc.url = "https://example.com/model"
        hub_doc.min_sdk_version = "2.0"
        hub_doc.model_types = ["text-generation"]
        hub_doc.capabilities = ["inference"]
        hub_doc.training_supported = False
        hub_doc.incremental_training_supported = False
        hub_doc.hosting_ecr_uri = "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
        hub_doc.inference_configs = []
        hub_doc.inference_config_components = {}
        hub_doc.inference_config_rankings = []
        hub_doc.hosting_artifact_uri = "s3://bucket/model.tar.gz"
        hub_doc.hosting_script_uri = "s3://bucket/inference.py"
        hub_doc.inference_environment_variables = []
        hub_doc.inference_dependencies = []
        hub_doc.default_inference_instance_type = "ml.m5.xlarge"
        hub_doc.supported_inference_instance_types = ["ml.m5.xlarge"]
        hub_doc.dynamic_container_deployment_supported = False
        hub_doc.hosting_resource_requirements = {}
        hub_doc.hosting_prepacked_artifact_uri = "s3://bucket/prepacked.tar.gz"
        hub_doc.sage_maker_sdk_predictor_specifications = {}
        hub_doc.default_payloads = None
        hub_doc.gated_bucket = False
        hub_doc.inference_volume_size = 30
        hub_doc.inference_enable_network_isolation = False
        hub_doc.resource_name_base = "test-model"
        hub_doc.hosting_eula_uri = "s3://bucket/eula.txt"
        hub_doc.hosting_model_package_arn = "arn:aws:sagemaker:us-west-2:123:model-package/test"
        hub_doc.model_subscription_link = "https://example.com/subscribe"
        hub_doc.hosting_use_script_uri = True
        hub_doc.hosting_instance_type_variants = {}
        hub_doc.to_json = Mock(return_value={})

        response.hub_content_document = hub_doc

        result = make_model_specs_from_describe_hub_content_response(response)

        assert result is not None

    def test_make_model_specs_from_describe_hub_content_response_with_training(self):
        """Test make_model_specs_from_describe_hub_content_response with training support"""
        response = Mock(spec=DescribeHubContentResponse)
        response.hub_content_type = HubContentType.MODEL
        response.hub_content_name = "test-model"
        response.hub_content_version = "1.0"
        response.get_hub_region = Mock(return_value="us-west-2")

        hub_doc = Mock(spec=HubModelDocument)
        hub_doc.url = "https://example.com/model"
        hub_doc.min_sdk_version = "2.0"
        hub_doc.model_types = ["text-generation"]
        hub_doc.capabilities = ["inference", "training"]
        hub_doc.training_supported = True
        hub_doc.incremental_training_supported = True
        hub_doc.hosting_ecr_uri = "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
        hub_doc.inference_configs = []
        hub_doc.inference_config_components = {}
        hub_doc.inference_config_rankings = []
        hub_doc.hosting_artifact_uri = None
        hub_doc.hosting_script_uri = None
        hub_doc.inference_environment_variables = []
        hub_doc.inference_dependencies = []
        hub_doc.default_inference_instance_type = "ml.m5.xlarge"
        hub_doc.supported_inference_instance_types = ["ml.m5.xlarge"]
        hub_doc.dynamic_container_deployment_supported = False
        hub_doc.hosting_resource_requirements = {}
        hub_doc.hosting_prepacked_artifact_uri = None
        hub_doc.sage_maker_sdk_predictor_specifications = {}
        hub_doc.default_payloads = None
        hub_doc.gated_bucket = False
        hub_doc.inference_volume_size = 30
        hub_doc.inference_enable_network_isolation = False
        hub_doc.resource_name_base = "test-model"
        hub_doc.hosting_eula_uri = None
        hub_doc.hosting_model_package_arn = None
        hub_doc.model_subscription_link = None
        hub_doc.hosting_use_script_uri = False
        hub_doc.hosting_instance_type_variants = {}

        # Training-specific fields
        hub_doc.training_ecr_uri = "123.dkr.ecr.us-west-2.amazonaws.com/training:latest"
        hub_doc.training_artifact_uri = "s3://bucket/training.tar.gz"
        hub_doc.training_script_uri = "s3://bucket/train.py"
        hub_doc.training_configs = []
        hub_doc.training_config_components = {}
        hub_doc.training_config_rankings = []
        hub_doc.training_dependencies = []
        hub_doc.default_training_instance_type = "ml.p3.2xlarge"
        hub_doc.supported_training_instance_types = ["ml.p3.2xlarge"]
        hub_doc.training_metrics = []
        hub_doc.training_prepacked_script_uri = None
        hub_doc.hyperparameters = {}
        hub_doc.training_volume_size = 50
        hub_doc.training_enable_network_isolation = False
        hub_doc.training_model_package_artifact_uri = None
        hub_doc.training_instance_type_variants = {}
        hub_doc.default_training_dataset_uri = None

        hub_doc.to_json = Mock(return_value={})

        response.hub_content_document = hub_doc

        result = make_model_specs_from_describe_hub_content_response(response)

        assert result is not None

    def test_make_model_specs_from_describe_hub_content_response_with_payloads(self):
        """Test make_model_specs_from_describe_hub_content_response with default payloads"""
        response = Mock(spec=DescribeHubContentResponse)
        response.hub_content_type = HubContentType.MODEL_REFERENCE
        response.hub_content_name = "test-model"
        response.hub_content_version = "1.0"
        response.get_hub_region = Mock(return_value="us-west-2")

        mock_payload = Mock()
        mock_payload.to_json = Mock(return_value={"ContentType": "application/json"})

        hub_doc = Mock(spec=HubModelDocument)
        hub_doc.url = "https://example.com/model"
        hub_doc.min_sdk_version = "2.0"
        hub_doc.model_types = ["text-generation"]
        hub_doc.capabilities = ["inference"]
        hub_doc.training_supported = False
        hub_doc.incremental_training_supported = False
        hub_doc.hosting_ecr_uri = "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
        hub_doc.inference_configs = []
        hub_doc.inference_config_components = {}
        hub_doc.inference_config_rankings = []
        hub_doc.hosting_artifact_uri = None
        hub_doc.hosting_script_uri = None
        hub_doc.inference_environment_variables = []
        hub_doc.inference_dependencies = []
        hub_doc.default_inference_instance_type = "ml.m5.xlarge"
        hub_doc.supported_inference_instance_types = ["ml.m5.xlarge"]
        hub_doc.dynamic_container_deployment_supported = False
        hub_doc.hosting_resource_requirements = {}
        hub_doc.hosting_prepacked_artifact_uri = None
        hub_doc.sage_maker_sdk_predictor_specifications = {}
        hub_doc.default_payloads = {"default": mock_payload}
        hub_doc.gated_bucket = False
        hub_doc.inference_volume_size = 30
        hub_doc.inference_enable_network_isolation = False
        hub_doc.resource_name_base = "test-model"
        hub_doc.hosting_eula_uri = None
        hub_doc.hosting_model_package_arn = None
        hub_doc.model_subscription_link = None
        hub_doc.hosting_use_script_uri = False
        hub_doc.hosting_instance_type_variants = {}
        hub_doc.to_json = Mock(return_value={})

        response.hub_content_document = hub_doc

        result = make_model_specs_from_describe_hub_content_response(response)

        assert result is not None
