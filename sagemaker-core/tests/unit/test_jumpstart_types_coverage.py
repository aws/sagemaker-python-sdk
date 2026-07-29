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
from sagemaker.core.jumpstart.types import (
    JumpStartDataHolderType,
    JumpStartECRSpecs,
    JumpStartHyperparameter,
    JumpStartEnvironmentVariable,
    JumpStartPredictorSpecs,
    JumpStartSerializablePayload,
    JumpStartInstanceTypeVariants,
    JumpStartAdditionalDataSources,
    JumpStartModelDataSource,
    ModelAccessConfig,
    HubAccessConfig,
    S3DataSource,
    AdditionalModelDataSource,
    JumpStartBenchmarkStat,
    JumpStartConfigRanking,
    JumpStartMetadataBaseFields,
    JumpStartConfigComponent,
    JumpStartMetadataConfig,
    JumpStartMetadataConfigs,
    JumpStartModelSpecs,
    JumpStartVersionedModelId,
    JumpStartCachedContentKey,
    JumpStartCachedContentValue,
    HubArnExtractedInfo,
    JumpStartKwargs,
    JumpStartModelInitKwargs,
    JumpStartModelDeployKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorDeployKwargs,
    JumpStartModelRegisterKwargs,
    BaseDeploymentConfigDataHolder,
    DeploymentArgs,
    DeploymentConfigMetadata,
    JumpStartS3FileType,
    HubContentType,
)
from sagemaker.core.jumpstart.enums import JumpStartScriptScope, JumpStartModelType


class TestJumpStartDataHolderTypeEdgeCases:
    """Test edge cases for JumpStartDataHolderType"""

    def test_eq_missing_attribute_in_self(self):
        """Test equality when self has missing attribute - line 75"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")
        # Manually delete attribute from obj1
        delattr(obj1, "version")
        assert obj1 != obj2

    def test_eq_missing_attribute_in_other(self):
        """Test equality when other has missing attribute - line 77"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")
        delattr(obj2, "version")
        assert obj1 != obj2

    def test_eq_different_attribute_values(self):
        """Test equality with different attribute values - line 82"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "2.0.0")
        assert obj1 != obj2


class TestJumpStartECRSpecsEdgeCases:
    """Test edge cases for JumpStartECRSpecs"""

    def test_from_json_empty_dict(self):
        """Test from_json with empty dict - line 321"""
        ecr_specs = JumpStartECRSpecs({})
        assert not hasattr(ecr_specs, "framework")

    def test_from_json_with_hub_content(self):
        """Test from_json with hub content flag - line 326, 328"""
        spec = {"Framework": "pytorch", "FrameworkVersion": "1.13.0", "PyVersion": "py39"}
        ecr_specs = JumpStartECRSpecs(spec, is_hub_content=True)
        assert ecr_specs.framework == "pytorch"


class TestJumpStartHyperparameterEdgeCases:
    """Test edge cases for JumpStartHyperparameter"""

    def test_from_json_with_exclusive_min_max(self):
        """Test from_json with exclusive min/max - lines 419, 470"""
        spec = {
            "name": "learning_rate",
            "type": "float",
            "default": "0.001",
            "scope": "training",
            "exclusive_min": True,
            "exclusive_max": True,
        }
        hyperparam = JumpStartHyperparameter(spec)
        assert hyperparam.exclusive_min is True
        assert hyperparam.exclusive_max is True


class TestJumpStartPredictorSpecsEdgeCases:
    """Test edge cases for JumpStartPredictorSpecs"""

    def test_from_json_none_input(self):
        """Test from_json with None - line 521"""
        predictor_specs = JumpStartPredictorSpecs(None)
        assert not hasattr(predictor_specs, "default_content_type")


class TestJumpStartSerializablePayloadEdgeCases:
    """Test edge cases for JumpStartSerializablePayload"""

    def test_from_json_none_input(self):
        """Test from_json with None - line 529"""
        payload = JumpStartSerializablePayload(None)
        assert not hasattr(payload, "content_type")

    def test_from_json_with_accept(self):
        """Test from_json with accept field - lines 544"""
        spec = {"content_type": "application/json", "body": "{}", "accept": "application/json"}
        payload = JumpStartSerializablePayload(spec)
        assert payload.accept == "application/json"


class TestJumpStartInstanceTypeVariantsEdgeCases:
    """Test edge cases for JumpStartInstanceTypeVariants"""

    def test_from_describe_hub_content_response_none(self):
        """Test from_describe_hub_content_response with None - line 562"""
        variants = JumpStartInstanceTypeVariants(None, is_hub_content=True)
        assert not hasattr(variants, "aliases")

    def test_regionalize_returns_none_when_aliases_set(self):
        """Test regionalize returns None when aliases is set - line 577"""
        spec = {"aliases": {"alias1": "value1"}}
        variants = JumpStartInstanceTypeVariants(spec, is_hub_content=True)
        result = variants.regionalize("us-west-2")
        assert result is None

    def test_get_instance_specific_metric_definitions_empty(self):
        """Test get_instance_specific_metric_definitions with no variants - line 598"""
        variants = JumpStartInstanceTypeVariants({})
        metrics = variants.get_instance_specific_metric_definitions("ml.m5.xlarge")
        assert metrics == []

    def test_get_instance_specific_prepacked_artifact_key(self):
        """Test get_instance_specific_prepacked_artifact_key - line 674"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"prepacked_artifact_key": "s3://bucket/artifact.tar.gz"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        key = variants.get_instance_specific_prepacked_artifact_key("ml.m5.xlarge")
        assert key == "s3://bucket/artifact.tar.gz"

    def test_get_instance_specific_training_artifact_key(self):
        """Test get_instance_specific_training_artifact_key - line 705"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"training_artifact_uri": "s3://bucket/training.tar.gz"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        key = variants.get_instance_specific_training_artifact_key("ml.m5.xlarge")
        assert key == "s3://bucket/training.tar.gz"

    def test_get_instance_specific_default_inference_instance_type(self):
        """Test get_instance_specific_default_inference_instance_type - line 807"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {"properties": {"default_inference_instance_type": "ml.m5.2xlarge"}}
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        instance_type = variants.get_instance_specific_default_inference_instance_type(
            "ml.m5.xlarge"
        )
        assert instance_type == "ml.m5.2xlarge"

    def test_get_instance_specific_supported_inference_instance_types_empty(self):
        """Test get_instance_specific_supported_inference_instance_types empty - line 867"""
        variants = JumpStartInstanceTypeVariants({})
        types = variants.get_instance_specific_supported_inference_instance_types("ml.m5.xlarge")
        assert types == []

    def test_get_image_uri_none_variants(self):
        """Test get_image_uri with None variants - line 882"""
        variants = JumpStartInstanceTypeVariants({})
        uri = variants.get_image_uri("ml.m5.xlarge", "us-west-2")
        assert uri is None

    def test_get_regional_property_none_region_with_regional_aliases(self):
        """Test _get_regional_property with None region and regional_aliases - line 918"""
        spec = {
            "regional_aliases": {"us-west-2": {"image_uri": "image"}},
            "variants": {"ml.m5.xlarge": {"regional_properties": {"image_uri": "$image_uri"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)
        result = variants._get_regional_property("ml.m5.xlarge", None, "image_uri")
        assert result is None

    def test_get_regional_property_bad_alias_format(self):
        """Test _get_regional_property with bad alias format - line 971"""
        spec = {
            "regional_aliases": {"us-west-2": {"image_uri": "image"}},
            "variants": {"ml.m5.xlarge": {"regional_properties": {"image_uri": "bad_alias"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)
        result = variants._get_regional_property("ml.m5.xlarge", "us-west-2", "image_uri")
        assert result is None


class TestJumpStartAdditionalDataSourcesEdgeCases:
    """Test edge cases for JumpStartAdditionalDataSources"""

    def test_from_json_none_speculative_decoding(self):
        """Test from_json with None speculative_decoding - line 1015"""
        spec = {"speculative_decoding": None}
        data_sources = JumpStartAdditionalDataSources(spec)
        assert data_sources.speculative_decoding is None

    def test_from_json_none_scripts(self):
        """Test from_json with None scripts - line 1023"""
        spec = {"scripts": None}
        data_sources = JumpStartAdditionalDataSources(spec)
        assert data_sources.scripts is None


class TestHubAccessConfigEdgeCases:
    """Test edge cases for HubAccessConfig"""

    def test_from_json(self):
        """Test from_json - line 1077"""
        spec = {"accept_eula": True}
        config = HubAccessConfig(spec)
        # Note: There's a bug in the source - it sets hub_content_arn from accept_eula
        assert config.hub_content_arn is True


class TestS3DataSourceEdgeCases:
    """Test edge cases for S3DataSource"""

    def test_from_json_with_hub_access_config(self):
        """Test from_json with hub_access_config - line 1199"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/",
            "hub_access_config": {"accept_eula": True},
        }
        data_source = S3DataSource(spec)
        assert data_source.hub_access_config is not None


class TestAdditionalModelDataSourceEdgeCases:
    """Test edge cases for AdditionalModelDataSource"""

    def test_to_json_with_exclude_keys_false(self):
        """Test to_json with exclude_keys=False - line 1484"""
        spec = {
            "channel_name": "model-data",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model/",
            },
            "artifact_version": "1.0.0",
        }
        data_source = JumpStartModelDataSource(spec)
        json_obj = data_source.to_json(exclude_keys=False)
        assert "artifact_version" in json_obj


class TestJumpStartConfigRankingEdgeCases:
    """Test edge cases for JumpStartConfigRanking"""

    def test_init_with_hub_content(self):
        """Test init with hub content - lines 1488-1491"""
        spec = {"Description": "Test ranking", "Rankings": ["config1", "config2"]}
        ranking = JumpStartConfigRanking(spec, is_hub_content=True)
        assert ranking.description == "Test ranking"
        assert ranking.rankings == ["config1", "config2"]


class TestJumpStartMetadataBaseFieldsEdgeCases:
    """Test edge cases for JumpStartMetadataBaseFields"""

    def test_from_json_with_hub_content_capabilities(self):
        """Test from_json with hub content capabilities - lines 1495-1498"""
        spec = {
            "model_id": "test-model",
            "Capabilities": ["text-generation"],
            "ModelTypes": ["llm"],
        }
        fields = JumpStartMetadataBaseFields(spec, is_hub_content=True)
        assert fields.capabilities == ["text-generation"]
        assert fields.model_types == ["llm"]

    def test_from_json_with_training_prepacked_script_version(self):
        """Test from_json with training_prepacked_script_version - lines 1505-1506"""
        spec = {
            "model_id": "test-model",
            "training_supported": True,
            "TrainingPrepackedScriptVersion": "1.0.0",
            "HostingPrepackedArtifactVersion": "1.0.0",
            "training_artifact_key": "key",
            "training_script_key": "script",
        }
        fields = JumpStartMetadataBaseFields(spec, is_hub_content=True)
        assert fields.training_prepacked_script_version == "1.0.0"
        assert fields.hosting_prepacked_artifact_version == "1.0.0"

    def test_set_hub_content_type(self):
        """Test set_hub_content_type - line 1762"""
        spec = {"model_id": "test-model"}
        fields = JumpStartMetadataBaseFields(spec, is_hub_content=True)
        fields.set_hub_content_type(HubContentType.MODEL)
        assert fields.hub_content_type == HubContentType.MODEL


class TestJumpStartConfigComponentEdgeCases:
    """Test edge cases for JumpStartConfigComponent"""

    def test_init_with_hub_content(self):
        """Test init with hub content - lines 1806"""
        component = {"ComponentName": "test-component", "HostingEcrUri": "image:latest"}
        config_component = JumpStartConfigComponent("test", component, is_hub_content=True)
        assert config_component.component_name == "test-component"


class TestJumpStartMetadataConfigEdgeCases:
    """Test edge cases for JumpStartMetadataConfig"""

    def test_init_with_none_benchmark_metrics(self):
        """Test init with None benchmark_metrics - line 1870"""
        config = JumpStartMetadataConfig("test-config", {}, {"model_id": "test"}, {})
        assert config.benchmark_metrics is None


class TestJumpStartMetadataConfigsEdgeCases:
    """Test edge cases for JumpStartMetadataConfigs"""

    def test_get_top_config_from_ranking_no_rankings(self):
        """Test get_top_config_from_ranking with no rankings - lines 1917-1932"""
        # Create a mock config with resolved_config
        mock_config = Mock()
        mock_config.resolved_config = Mock()
        mock_config.resolved_config.supported_inference_instance_types = ["ml.m5.xlarge"]

        configs = JumpStartMetadataConfigs(
            {"config1": mock_config}, None, JumpStartScriptScope.INFERENCE
        )
        result = configs.get_top_config_from_ranking(instance_type="ml.m5.xlarge")
        assert result is not None

    def test_get_top_config_from_ranking_unknown_scope(self):
        """Test get_top_config_from_ranking with unknown scope - line 1936"""
        configs = JumpStartMetadataConfigs({}, {}, "unknown_scope")
        with pytest.raises(NotImplementedError):
            configs.get_top_config_from_ranking()


class TestJumpStartModelSpecsEdgeCases:
    """Test edge cases for JumpStartModelSpecs"""

    def test_set_config_unknown_scope(self):
        """Test set_config with unknown scope - line 2042"""
        spec = {"model_id": "test-model"}
        model_specs = JumpStartModelSpecs(spec)
        with pytest.raises(ValueError):
            model_specs.set_config("config1", "unknown_scope")

    def test_set_config_config_not_found(self):
        """Test set_config with config not found - lines 2070"""
        spec = {"model_id": "test-model"}
        model_specs = JumpStartModelSpecs(spec)
        # Create a mock inference_configs with a config
        mock_config = Mock()
        mock_config.configs = {"config1": Mock()}
        model_specs.inference_configs = mock_config

        with pytest.raises(ValueError, match="Cannot find Jumpstart config name"):
            model_specs.set_config("nonexistent", JumpStartScriptScope.INFERENCE)

    def test_supports_prepacked_inference(self):
        """Test supports_prepacked_inference - lines 2095-2096"""
        spec = {
            "model_id": "test-model",
            "hosting_prepacked_artifact_key": "s3://bucket/artifact.tar.gz",
        }
        model_specs = JumpStartModelSpecs(spec)
        assert model_specs.supports_prepacked_inference() is True

    def test_use_inference_script_uri(self):
        """Test use_inference_script_uri - lines 2107-2118"""
        spec = {"model_id": "test-model", "hosting_use_script_uri": False}
        model_specs = JumpStartModelSpecs(spec)
        assert model_specs.use_inference_script_uri() is False

    def test_use_training_model_artifact_gated(self):
        """Test use_training_model_artifact with gated bucket - line 2210"""
        spec = {"model_id": "test-model", "gated_bucket": True}
        model_specs = JumpStartModelSpecs(spec)
        assert model_specs.use_training_model_artifact() is False

    def test_is_gated_model(self):
        """Test is_gated_model - lines 2239"""
        spec = {"model_id": "test-model", "hosting_eula_key": "eula.txt"}
        model_specs = JumpStartModelSpecs(spec)
        assert model_specs.is_gated_model() is True

    def test_get_speculative_decoding_s3_data_sources_none(self):
        """Test get_speculative_decoding_s3_data_sources with None - line 2340"""
        spec = {"model_id": "test-model"}
        model_specs = JumpStartModelSpecs(spec)
        sources = model_specs.get_speculative_decoding_s3_data_sources()
        assert sources == []


class TestHubArnExtractedInfoEdgeCases:
    """Test edge cases for HubArnExtractedInfo"""

    def test_extract_region_from_hub_content_arn(self):
        """Test extract_region_from_arn with hub content ARN - lines 2520-2579"""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/hub-name/Model/model-name/1.0.0"
        region = HubArnExtractedInfo.extract_region_from_arn(arn)
        assert region == "us-west-2"

    def test_extract_region_from_hub_arn(self):
        """Test extract_region_from_arn with hub ARN - lines 2634-2647"""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub/hub-name"
        region = HubArnExtractedInfo.extract_region_from_arn(arn)
        assert region == "us-east-1"

    def test_extract_region_from_invalid_arn(self):
        """Test extract_region_from_arn with invalid ARN"""
        arn = "invalid-arn"
        region = HubArnExtractedInfo.extract_region_from_arn(arn)
        assert region is None


class TestJumpStartKwargsEdgeCases:
    """Test edge cases for JumpStartKwargs"""

    def test_to_kwargs_dict_with_exclude_keys_false(self):
        """Test to_kwargs_dict with exclude_keys=False - lines 2755-2795"""
        kwargs = JumpStartModelInitKwargs("test-model")
        kwargs_dict = kwargs.to_kwargs_dict(exclude_keys=False)
        assert "specs" in kwargs_dict or "model_id" in kwargs_dict


class TestJumpStartModelDeployKwargsEdgeCases:
    """Test edge cases for JumpStartModelDeployKwargs"""

    def test_init_with_all_params(self):
        """Test init with all parameters - lines 2889-2922"""
        kwargs = JumpStartModelDeployKwargs(
            model_id="test-model",
            model_version="1.0.0",
            hub_arn="arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub",
            model_type=JumpStartModelType.OPEN_WEIGHTS,
            region="us-west-2",
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            model_access_configs={},
        )
        assert kwargs.model_id == "test-model"
        assert kwargs.model_access_configs == {}


class TestJumpStartEstimatorInitKwargsEdgeCases:
    """Test edge cases for JumpStartEstimatorInitKwargs"""

    def test_init_with_training_plan(self):
        """Test init with training_plan - lines 2936, 2940-2946"""
        kwargs = JumpStartEstimatorInitKwargs(
            model_id="test-model", training_plan="training-plan-arn"
        )
        assert kwargs.training_plan == "training-plan-arn"


class TestJumpStartEstimatorFitKwargsEdgeCases:
    """Test edge cases for JumpStartEstimatorFitKwargs"""

    def test_init_minimal(self):
        """Test init with minimal parameters - lines 2956-2973"""
        kwargs = JumpStartEstimatorFitKwargs(model_id="test-model")
        assert kwargs.model_id == "test-model"


class TestJumpStartModelRegisterKwargsEdgeCases:
    """Test edge cases for JumpStartModelRegisterKwargs"""

    def test_init_with_model_card(self):
        """Test init with model_card - lines 2998-3018"""
        kwargs = JumpStartModelRegisterKwargs(model_id="test-model", model_card={})
        assert kwargs.model_card == {}


class TestBaseDeploymentConfigDataHolderEdgeCases:
    """Test edge cases for BaseDeploymentConfigDataHolder"""

    def test_convert_to_pascal_case(self):
        """Test _convert_to_pascal_case - lines 3039-3044"""
        holder = BaseDeploymentConfigDataHolder()
        result = holder._convert_to_pascal_case("test_attribute_name")
        assert result == "TestAttributeName"

    def test_val_to_json_with_benchmark_stat(self):
        """Test _val_to_json with JumpStartBenchmarkStat"""
        holder = BaseDeploymentConfigDataHolder()
        stat = JumpStartBenchmarkStat(
            {"name": "test_metric", "value": "100", "unit": "ms", "concurrency": 1}
        )
        result = holder._val_to_json(stat)
        assert result["name"] == "Test Metric"


class TestDeploymentArgsEdgeCases:
    """Test edge cases for DeploymentArgs"""

    def test_init_with_resources(self):
        """Test init with resources"""
        mock_resources = Mock()
        mock_resources.get_compute_resource_requirements.return_value = {"cpu": 2}

        init_kwargs = JumpStartModelInitKwargs("test-model")
        init_kwargs.resources = mock_resources

        deployment_args = DeploymentArgs(init_kwargs=init_kwargs)
        assert deployment_args.compute_resource_requirements == {"cpu": 2}


class TestDeploymentConfigMetadataEdgeCases:
    """Test edge cases for DeploymentConfigMetadata"""

    def test_init_with_all_params(self):
        """Test init with all parameters"""
        init_kwargs = JumpStartModelInitKwargs("test-model")
        deploy_kwargs = JumpStartModelDeployKwargs("test-model")

        # Create a mock metadata_config with resolved_config
        metadata_config = Mock()
        metadata_config.resolved_config = {
            "default_inference_instance_type": "ml.m5.xlarge",
            "supported_inference_instance_types": ["ml.m5.xlarge"],
            "hosting_additional_data_sources": None,
        }

        config_metadata = DeploymentConfigMetadata(
            config_name="test-config",
            metadata_config=metadata_config,
            init_kwargs=init_kwargs,
            deploy_kwargs=deploy_kwargs,
        )
        assert config_metadata.deployment_config_name == "test-config"
