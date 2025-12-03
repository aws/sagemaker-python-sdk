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

"""Extended unit tests for sagemaker.core.jumpstart.utils module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from sagemaker.core.jumpstart import utils, constants, enums
from sagemaker.core.jumpstart.types import JumpStartModelSpecs
from sagemaker.core.jumpstart.exceptions import (
    DeprecatedJumpStartModelError,
    VulnerableJumpStartModelError,
)


class TestGetJumpStartLaunchedRegionsMessage:
    """Test cases for get_jumpstart_launched_regions_message"""

    @patch.object(constants, "JUMPSTART_REGION_NAME_SET", set())
    def test_no_regions(self):
        """Test with no regions"""
        message = utils.get_jumpstart_launched_regions_message()
        assert "not available in any region" in message

    @patch.object(constants, "JUMPSTART_REGION_NAME_SET", {"us-west-2"})
    def test_single_region(self):
        """Test with single region"""
        message = utils.get_jumpstart_launched_regions_message()
        assert "us-west-2" in message
        assert "region." in message

    @patch.object(constants, "JUMPSTART_REGION_NAME_SET", {"us-west-2", "us-east-1"})
    def test_two_regions(self):
        """Test with two regions"""
        message = utils.get_jumpstart_launched_regions_message()
        assert "us-west-2" in message or "us-east-1" in message
        assert " and " in message

    @patch.object(constants, "JUMPSTART_REGION_NAME_SET", {"us-west-2", "us-east-1", "eu-west-1"})
    def test_multiple_regions(self):
        """Test with multiple regions"""
        message = utils.get_jumpstart_launched_regions_message()
        assert "us-west-2" in message
        assert ", " in message


class TestGetJumpStartContentBucket:
    """Test cases for get_jumpstart_content_bucket"""

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT")
    def test_get_content_bucket_success(self, mock_region_dict):
        """Test successful bucket retrieval"""
        mock_region_info = Mock()
        mock_region_info.content_bucket = "jumpstart-cache-prod-us-west-2"
        mock_region_dict.__getitem__.return_value = mock_region_info

        bucket = utils.get_jumpstart_content_bucket("us-west-2")
        assert bucket == "jumpstart-cache-prod-us-west-2"

    @patch.dict("os.environ", {"AWS_JUMPSTART_CONTENT_BUCKET_OVERRIDE": "my-custom-bucket"})
    def test_get_content_bucket_with_override(self):
        """Test bucket retrieval with environment override"""
        from sagemaker.core.jumpstart import accessors

        accessors.JumpStartModelsAccessor.set_jumpstart_content_bucket(None)

        bucket = utils.get_jumpstart_content_bucket("us-west-2")
        assert bucket == "my-custom-bucket"

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT", {})
    def test_get_content_bucket_invalid_region(self):
        """Test with invalid region"""
        with pytest.raises(ValueError, match="Unable to get content bucket"):
            utils.get_jumpstart_content_bucket("invalid-region")


class TestGetJumpStartGatedContentBucket:
    """Test cases for get_jumpstart_gated_content_bucket"""

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT")
    def test_get_gated_bucket_success(self, mock_region_dict):
        """Test successful gated bucket retrieval"""
        mock_region_info = Mock()
        mock_region_info.gated_content_bucket = "jumpstart-private-cache-prod-us-west-2"
        mock_region_dict.__getitem__.return_value = mock_region_info

        bucket = utils.get_jumpstart_gated_content_bucket("us-west-2")
        assert bucket == "jumpstart-private-cache-prod-us-west-2"

    @patch.dict("os.environ", {"AWS_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE": "my-gated-bucket"})
    def test_get_gated_bucket_with_override(self):
        """Test gated bucket with environment override"""
        from sagemaker.core.jumpstart import accessors

        accessors.JumpStartModelsAccessor.set_jumpstart_gated_content_bucket(None)

        bucket = utils.get_jumpstart_gated_content_bucket("us-west-2")
        assert bucket == "my-gated-bucket"

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT")
    def test_get_gated_bucket_none(self, mock_region_dict):
        """Test when gated bucket is None"""
        mock_region_info = Mock()
        mock_region_info.gated_content_bucket = None
        mock_region_dict.__getitem__.return_value = mock_region_info

        with pytest.raises(ValueError, match="No private content bucket"):
            utils.get_jumpstart_gated_content_bucket("us-west-2")


class TestIsJumpStartModelInput:
    """Test cases for is_jumpstart_model_input"""

    def test_both_none(self):
        """Test with both None"""
        assert utils.is_jumpstart_model_input(None, None) is False

    def test_both_provided(self):
        """Test with both provided"""
        assert utils.is_jumpstart_model_input("model-id", "1.0.0") is True

    def test_only_model_id(self):
        """Test with only model_id"""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            utils.is_jumpstart_model_input("model-id", None)

    def test_only_version(self):
        """Test with only version"""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            utils.is_jumpstart_model_input(None, "1.0.0")


class TestIsJumpStartModelUri:
    """Test cases for is_jumpstart_model_uri"""

    @patch.object(constants, "JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET", {"jumpstart-bucket"})
    def test_jumpstart_uri(self):
        """Test with JumpStart S3 URI"""
        assert utils.is_jumpstart_model_uri("s3://jumpstart-bucket/model.tar.gz") is True

    @patch.object(constants, "JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET", {"jumpstart-bucket"})
    def test_non_jumpstart_uri(self):
        """Test with non-JumpStart S3 URI"""
        assert utils.is_jumpstart_model_uri("s3://my-bucket/model.tar.gz") is False

    def test_none_uri(self):
        """Test with None URI"""
        assert utils.is_jumpstart_model_uri(None) is False

    def test_non_string_uri(self):
        """Test with non-string URI"""
        assert utils.is_jumpstart_model_uri(123) is False

    def test_non_s3_uri(self):
        """Test with non-S3 URI"""
        assert utils.is_jumpstart_model_uri("file:///path/to/model") is False


class TestTagKeyInArray:
    """Test cases for tag_key_in_array"""

    def test_key_exists(self):
        """Test when key exists"""
        tags = [{"Key": "model_id", "Value": "test-model"}]
        assert utils.tag_key_in_array("model_id", tags) is True

    def test_key_not_exists(self):
        """Test when key doesn't exist"""
        tags = [{"Key": "model_id", "Value": "test-model"}]
        assert utils.tag_key_in_array("version", tags) is False

    def test_empty_array(self):
        """Test with empty array"""
        assert utils.tag_key_in_array("model_id", []) is False


class TestGetTagValue:
    """Test cases for get_tag_value"""

    def test_get_existing_tag(self):
        """Test getting existing tag value"""
        tags = [
            {"Key": "model_id", "Value": "test-model"},
            {"Key": "version", "Value": "1.0.0"},
        ]
        value = utils.get_tag_value("model_id", tags)
        assert value == "test-model"

    def test_get_nonexistent_tag(self):
        """Test getting non-existent tag"""
        tags = [{"Key": "model_id", "Value": "test-model"}]
        with pytest.raises(KeyError, match="Cannot get value of tag"):
            utils.get_tag_value("version", tags)

    def test_duplicate_keys(self):
        """Test with duplicate keys"""
        tags = [
            {"Key": "model_id", "Value": "test-model-1"},
            {"Key": "model_id", "Value": "test-model-2"},
        ]
        with pytest.raises(KeyError, match="found 2 number of matches"):
            utils.get_tag_value("model_id", tags)


class TestAddSingleJumpStartTag:
    """Test cases for add_single_jumpstart_tag"""

    def test_add_tag_to_none(self):
        """Test adding tag when curr_tags is None"""
        result = utils.add_single_jumpstart_tag(
            "test-model",
            enums.JumpStartTag.MODEL_ID,
            None,
            is_uri=False,
        )
        assert len(result) == 1
        assert result[0]["Key"] == enums.JumpStartTag.MODEL_ID
        assert result[0]["Value"] == "test-model"

    def test_add_tag_to_existing(self):
        """Test adding tag to existing tags"""
        curr_tags = [{"Key": "existing", "Value": "tag"}]
        result = utils.add_single_jumpstart_tag(
            "test-model",
            enums.JumpStartTag.MODEL_ID,
            curr_tags,
            is_uri=False,
        )
        assert len(result) == 2

    def test_skip_duplicate_tag(self):
        """Test skipping duplicate tag"""
        curr_tags = [{"Key": enums.JumpStartTag.MODEL_ID, "Value": "old-model"}]
        result = utils.add_single_jumpstart_tag(
            "test-model",
            enums.JumpStartTag.MODEL_ID,
            curr_tags,
            is_uri=False,
        )
        assert len(result) == 1
        assert result[0]["Value"] == "old-model"

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri", return_value=True)
    def test_add_uri_tag(self, mock_is_jumpstart):
        """Test adding URI tag"""
        result = utils.add_single_jumpstart_tag(
            "s3://jumpstart-bucket/model.tar.gz",
            enums.JumpStartTag.INFERENCE_MODEL_URI,
            None,
            is_uri=True,
        )
        assert len(result) == 1


class TestAddJumpStartModelInfoTags:
    """Test cases for add_jumpstart_model_info_tags"""

    def test_add_model_info_tags(self):
        """Test adding model info tags"""
        tags = utils.add_jumpstart_model_info_tags(
            None,
            "test-model",
            "1.0.0",
        )
        assert len(tags) >= 2
        model_id_tag = next(t for t in tags if t["Key"] == enums.JumpStartTag.MODEL_ID)
        assert model_id_tag["Value"] == "test-model"

    def test_skip_wildcard_version(self):
        """Test that wildcard version is not added"""
        tags = utils.add_jumpstart_model_info_tags(
            None,
            "test-model",
            "*",
        )
        version_tags = [t for t in tags if t["Key"] == enums.JumpStartTag.MODEL_VERSION]
        assert len(version_tags) == 0

    def test_add_proprietary_model_type(self):
        """Test adding proprietary model type tag"""
        tags = utils.add_jumpstart_model_info_tags(
            None,
            "test-model",
            "1.0.0",
            model_type=enums.JumpStartModelType.PROPRIETARY,
        )
        type_tags = [t for t in tags if t["Key"] == enums.JumpStartTag.MODEL_TYPE]
        assert len(type_tags) == 1

    def test_add_config_name_inference(self):
        """Test adding inference config name"""
        tags = utils.add_jumpstart_model_info_tags(
            None,
            "test-model",
            "1.0.0",
            config_name="default",
            scope=enums.JumpStartScriptScope.INFERENCE,
        )
        config_tags = [t for t in tags if t["Key"] == enums.JumpStartTag.INFERENCE_CONFIG_NAME]
        assert len(config_tags) == 1


class TestAddJumpStartUriTags:
    """Test cases for add_jumpstart_uri_tags"""

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri", return_value=True)
    def test_add_inference_model_uri(self, mock_is_jumpstart):
        """Test adding inference model URI tag"""
        tags = utils.add_jumpstart_uri_tags(
            inference_model_uri="s3://jumpstart-bucket/model.tar.gz"
        )
        assert len(tags) == 1
        assert tags[0]["Key"] == enums.JumpStartTag.INFERENCE_MODEL_URI

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri", return_value=True)
    def test_add_multiple_uri_tags(self, mock_is_jumpstart):
        """Test adding multiple URI tags"""
        tags = utils.add_jumpstart_uri_tags(
            inference_model_uri="s3://jumpstart-bucket/model.tar.gz",
            inference_script_uri="s3://jumpstart-bucket/script.tar.gz",
        )
        assert len(tags) == 2

    @patch("sagemaker.core.jumpstart.utils.is_pipeline_variable", return_value=True)
    def test_skip_pipeline_variable(self, mock_is_pipeline):
        """Test skipping pipeline variables"""
        with patch("sagemaker.core.jumpstart.utils.logging") as mock_logging:
            tags = utils.add_jumpstart_uri_tags(inference_model_uri=Mock())  # Pipeline variable
            assert tags is None or len(tags) == 0


class TestGetEulaMessage:
    """Test cases for get_eula_message"""

    def test_no_eula_key(self):
        """Test when no EULA key"""
        model_specs = Mock()
        model_specs.hosting_eula_key = None

        message = utils.get_eula_message(model_specs, "us-west-2")
        assert message == ""

    @patch("sagemaker.core.jumpstart.utils.get_jumpstart_content_bucket")
    @patch("sagemaker.core.jumpstart.utils.get_domain_for_region")
    def test_with_eula_key(self, mock_get_domain, mock_get_bucket):
        """Test with EULA key"""
        mock_get_bucket.return_value = "jumpstart-bucket"
        mock_get_domain.return_value = "amazonaws.com"

        model_specs = Mock()
        model_specs.model_id = "test-model"
        model_specs.hosting_eula_key = "eula/test-model.txt"

        message = utils.get_eula_message(model_specs, "us-west-2")
        assert "test-model" in message
        assert "end-user license agreement" in message


class TestVerifyModelRegionAndReturnSpecs:
    """Test cases for verify_model_region_and_return_specs"""

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_verify_success(self, mock_get_specs):
        """Test successful verification"""
        mock_specs = Mock(spec=JumpStartModelSpecs)
        mock_specs.training_supported = True
        mock_specs.deprecated = False
        mock_specs.inference_vulnerable = False
        mock_specs.training_vulnerable = False
        mock_get_specs.return_value = mock_specs

        result = utils.verify_model_region_and_return_specs(
            "test-model",
            "1.0.0",
            "training",
            "us-west-2",
        )
        assert result == mock_specs

    def test_verify_none_scope(self):
        """Test with None scope"""
        with pytest.raises(ValueError, match="Must specify `model_scope`"):
            utils.verify_model_region_and_return_specs(
                "test-model",
                "1.0.0",
                None,
                "us-west-2",
            )

    def test_verify_unsupported_scope(self):
        """Test with unsupported scope"""
        with pytest.raises(NotImplementedError, match="only support scopes"):
            utils.verify_model_region_and_return_specs(
                "test-model",
                "1.0.0",
                "invalid_scope",
                "us-west-2",
            )

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_verify_training_not_supported(self, mock_get_specs):
        """Test when training is not supported"""
        mock_specs = Mock(spec=JumpStartModelSpecs)
        mock_specs.training_supported = False
        mock_get_specs.return_value = mock_specs

        with pytest.raises(ValueError, match="does not support training"):
            utils.verify_model_region_and_return_specs(
                "test-model",
                "1.0.0",
                "training",
                "us-west-2",
            )

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_verify_deprecated_model(self, mock_get_specs):
        """Test with deprecated model"""
        mock_specs = Mock(spec=JumpStartModelSpecs)
        mock_specs.deprecated = True
        mock_specs.deprecated_message = "Model is deprecated"
        mock_specs.training_supported = True
        mock_specs.inference_vulnerable = False
        mock_specs.training_vulnerable = False
        mock_get_specs.return_value = mock_specs

        with pytest.raises(DeprecatedJumpStartModelError):
            utils.verify_model_region_and_return_specs(
                "test-model",
                "1.0.0",
                "training",
                "us-west-2",
                tolerate_deprecated_model=False,
            )

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_verify_vulnerable_model(self, mock_get_specs):
        """Test with vulnerable model"""
        mock_specs = Mock(spec=JumpStartModelSpecs)
        mock_specs.deprecated = False
        mock_specs.training_supported = True
        mock_specs.training_vulnerable = True
        mock_specs.training_vulnerabilities = ["CVE-2021-1234"]
        mock_get_specs.return_value = mock_specs

        with pytest.raises(VulnerableJumpStartModelError):
            utils.verify_model_region_and_return_specs(
                "test-model",
                "1.0.0",
                "training",
                "us-west-2",
                tolerate_vulnerable_model=False,
            )


class TestUpdateDictIfKeyNotPresent:
    """Test cases for update_dict_if_key_not_present"""

    def test_add_to_none(self):
        """Test adding to None dict"""
        result = utils.update_dict_if_key_not_present(None, "key", "value")
        assert result == {"key": "value"}

    def test_add_new_key(self):
        """Test adding new key"""
        result = utils.update_dict_if_key_not_present({"existing": "value"}, "new", "new_value")
        assert result == {"existing": "value", "new": "new_value"}

    def test_skip_existing_key(self):
        """Test skipping existing key"""
        result = utils.update_dict_if_key_not_present({"key": "old"}, "key", "new")
        assert result == {"key": "old"}


class TestGetSagemakerVersion:
    """Test cases for get_sagemaker_version"""

    @patch("sagemaker.core.jumpstart.accessors.SageMakerSettings.get_sagemaker_version")
    @patch("sagemaker.core.jumpstart.accessors.SageMakerSettings.set_sagemaker_version")
    @patch("sagemaker.core.jumpstart.utils.parse_sagemaker_version")
    def test_get_sagemaker_version_not_set(self, mock_parse, mock_set, mock_get):
        """Test getting version when not set"""
        mock_get.return_value = ""
        mock_parse.return_value = "2.100.0"

        version = utils.get_sagemaker_version()

        mock_parse.assert_called_once()
        mock_set.assert_called_once_with("2.100.0")

    @patch("sagemaker.core.jumpstart.accessors.SageMakerSettings.get_sagemaker_version")
    def test_get_sagemaker_version_already_set(self, mock_get):
        """Test getting version when already set"""
        mock_get.return_value = "2.100.0"

        version = utils.get_sagemaker_version()

        assert version == "2.100.0"


class TestParseSagemakerVersion:
    """Test cases for parse_sagemaker_version"""

    def test_parse_version_three_parts(self):
        """Test parsing version with three parts"""
        with patch.object(utils.sagemaker, "__version__", "2.100.0", create=True):
            version = utils.parse_sagemaker_version()
            assert version == "2.100.0"

    def test_parse_version_four_parts(self):
        """Test parsing version with four parts"""
        with patch.object(utils.sagemaker, "__version__", "2.100.0.dev0", create=True):
            version = utils.parse_sagemaker_version()
            assert version == "2.100.0"

    def test_parse_version_invalid_periods(self):
        """Test parsing version with invalid number of periods"""
        with patch.object(utils.sagemaker, "__version__", "2.100", create=True):
            with pytest.raises(RuntimeError, match="Bad value for SageMaker version"):
                utils.parse_sagemaker_version()


class TestGetFormattedManifest:
    """Test cases for get_formatted_manifest"""

    def test_get_formatted_manifest(self):
        """Test formatting manifest"""
        manifest = [
            {
                "model_id": "test-model-1",
                "version": "1.0.0",
                "min_version": "2.0.0",
                "spec_key": "specs/test-model-1.json",
            },
            {
                "model_id": "test-model-2",
                "version": "2.0.0",
                "min_version": "2.0.0",
                "spec_key": "specs/test-model-2.json",
            },
        ]

        result = utils.get_formatted_manifest(manifest)

        assert len(result) == 2
        from sagemaker.core.jumpstart.types import JumpStartVersionedModelId

        key1 = JumpStartVersionedModelId("test-model-1", "1.0.0")
        assert key1 in result


class TestGetNeoContentBucket:
    """Test cases for get_neo_content_bucket"""

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT")
    def test_get_neo_bucket_success(self, mock_region_dict):
        """Test successful Neo bucket retrieval"""
        mock_region_info = Mock()
        mock_region_info.neo_content_bucket = "neo-cache-prod-us-west-2"
        mock_region_dict.__getitem__.return_value = mock_region_info

        bucket = utils.get_neo_content_bucket("us-west-2")
        assert bucket == "neo-cache-prod-us-west-2"

    @patch.dict("os.environ", {"AWS_NEO_CONTENT_BUCKET_OVERRIDE": "my-neo-bucket"})
    def test_get_neo_bucket_with_override(self):
        """Test Neo bucket with environment override"""
        bucket = utils.get_neo_content_bucket("us-west-2")
        assert bucket == "my-neo-bucket"

    @patch.dict("os.environ", {}, clear=True)
    @patch.object(constants, "JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT", {})
    def test_get_neo_bucket_invalid_region(self):
        """Test with invalid region"""
        with pytest.raises(ValueError, match="Unable to get content bucket for Neo"):
            utils.get_neo_content_bucket("invalid-region")


class TestGetJumpStartBaseNameIfJumpStartModel:
    """Test cases for get_jumpstart_base_name_if_jumpstart_model"""

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri")
    def test_with_jumpstart_uri(self, mock_is_jumpstart):
        """Test with JumpStart URI"""
        mock_is_jumpstart.return_value = True

        result = utils.get_jumpstart_base_name_if_jumpstart_model(
            "s3://jumpstart-bucket/model.tar.gz"
        )

        assert result == constants.JUMPSTART_RESOURCE_BASE_NAME

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri")
    def test_with_non_jumpstart_uri(self, mock_is_jumpstart):
        """Test with non-JumpStart URI"""
        mock_is_jumpstart.return_value = False

        result = utils.get_jumpstart_base_name_if_jumpstart_model("s3://my-bucket/model.tar.gz")

        assert result is None

    @patch("sagemaker.core.jumpstart.utils.is_jumpstart_model_uri")
    def test_with_multiple_uris(self, mock_is_jumpstart):
        """Test with multiple URIs"""
        mock_is_jumpstart.side_effect = [False, True]

        result = utils.get_jumpstart_base_name_if_jumpstart_model(
            "s3://my-bucket/model.tar.gz", "s3://jumpstart-bucket/model.tar.gz"
        )

        assert result == constants.JUMPSTART_RESOURCE_BASE_NAME


class TestAddHubContentArnTags:
    """Test cases for add_hub_content_arn_tags"""

    def test_add_hub_content_arn_tag(self):
        """Test adding hub content ARN tag"""
        tags = utils.add_hub_content_arn_tags(
            None, "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1"
        )

        assert len(tags) == 1
        assert tags[0]["Key"] == enums.JumpStartTag.HUB_CONTENT_ARN

    def test_add_hub_content_arn_tag_to_existing(self):
        """Test adding hub content ARN tag to existing tags"""
        existing_tags = [{"Key": "existing", "Value": "tag"}]
        tags = utils.add_hub_content_arn_tags(
            existing_tags,
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1",
        )

        assert len(tags) == 2


class TestAddBedrockStoreTags:
    """Test cases for add_bedrock_store_tags"""

    def test_add_bedrock_store_tag(self):
        """Test adding bedrock store tag"""
        tags = utils.add_bedrock_store_tags(None, "bedrock-compatible")

        assert len(tags) == 1
        assert tags[0]["Key"] == enums.JumpStartTag.BEDROCK


class TestUpdateInferenceTagsWithJumpStartTrainingTags:
    """Test cases for update_inference_tags_with_jumpstart_training_tags"""

    def test_update_with_training_tags(self):
        """Test updating inference tags with training tags"""
        training_tags = [
            {"Key": enums.JumpStartTag.MODEL_ID, "Value": "test-model"},
            {"Key": enums.JumpStartTag.MODEL_VERSION, "Value": "1.0.0"},
        ]

        inference_tags = utils.update_inference_tags_with_jumpstart_training_tags(
            None, training_tags
        )

        assert len(inference_tags) == 2

    def test_update_with_no_training_tags(self):
        """Test updating when no training tags"""
        result = utils.update_inference_tags_with_jumpstart_training_tags(None, None)

        assert result is None

    def test_skip_duplicate_tags(self):
        """Test skipping duplicate tags"""
        training_tags = [
            {"Key": enums.JumpStartTag.MODEL_ID, "Value": "test-model"},
        ]
        inference_tags = [
            {"Key": enums.JumpStartTag.MODEL_ID, "Value": "old-model"},
        ]

        result = utils.update_inference_tags_with_jumpstart_training_tags(
            inference_tags, training_tags
        )

        # Should keep old value
        model_id_tags = [t for t in result if t["Key"] == enums.JumpStartTag.MODEL_ID]
        assert len(model_id_tags) == 1
        assert model_id_tags[0]["Value"] == "old-model"


class TestEmitLogsBasedOnModelSpecs:
    """Test cases for emit_logs_based_on_model_specs"""

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.core.jumpstart.utils.get_eula_message")
    def test_emit_logs_with_eula(self, mock_get_eula, mock_get_manifest):
        """Test emitting logs with EULA"""
        mock_get_eula.return_value = "EULA message"
        mock_get_manifest.return_value = []

        model_specs = Mock()
        model_specs.hosting_eula_key = "eula/test.txt"
        model_specs.version = "1.0.0"
        model_specs.model_id = "test-model"
        model_specs.deprecated = False
        model_specs.deprecate_warn_message = None
        model_specs.usage_info_message = None
        model_specs.inference_vulnerable = False
        model_specs.training_vulnerable = False

        mock_s3_client = Mock()

        with patch("sagemaker.core.jumpstart.constants.JUMPSTART_LOGGER") as mock_logger:
            utils.emit_logs_based_on_model_specs(model_specs, "us-west-2", mock_s3_client)

            mock_logger.info.assert_called()

    @patch("sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    def test_emit_logs_deprecated_model(self, mock_get_manifest):
        """Test emitting logs for deprecated model"""
        mock_get_manifest.return_value = []

        model_specs = Mock()
        model_specs.hosting_eula_key = None
        model_specs.version = "1.0.0"
        model_specs.model_id = "test-model"
        model_specs.deprecated = True
        model_specs.deprecated_message = "Model is deprecated"
        model_specs.deprecate_warn_message = None
        model_specs.usage_info_message = None
        model_specs.inference_vulnerable = False
        model_specs.training_vulnerable = False

        mock_s3_client = Mock()

        with patch("sagemaker.core.jumpstart.constants.JUMPSTART_LOGGER") as mock_logger:
            utils.emit_logs_based_on_model_specs(model_specs, "us-west-2", mock_s3_client)

            mock_logger.warning.assert_called()


class TestGetFormattedEulaMessageTemplate:
    """Test cases for get_formatted_eula_message_template"""

    @patch("sagemaker.core.jumpstart.utils.get_jumpstart_content_bucket")
    @patch("sagemaker.core.jumpstart.utils.get_domain_for_region")
    def test_get_formatted_eula_message(self, mock_get_domain, mock_get_bucket):
        """Test getting formatted EULA message"""
        mock_get_bucket.return_value = "jumpstart-bucket"
        mock_get_domain.return_value = "amazonaws.com"

        message = utils.get_formatted_eula_message_template(
            "test-model", "us-west-2", "eula/test-model.txt"
        )

        assert "test-model" in message
        assert "end-user license agreement" in message
        assert "jumpstart-bucket" in message
