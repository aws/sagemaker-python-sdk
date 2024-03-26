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
from __future__ import absolute_import

from unittest.mock import Mock
from sagemaker.jumpstart.types import HubArnExtractedInfo
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.curated_hub import utils
from unittest.mock import patch
from sagemaker.jumpstart.curated_hub.types import (
    CuratedHubUnsupportedFlag,
    summary_from_list_api_response,
    summary_list_from_list_api_response,
)
from sagemaker.jumpstart.curated_hub.interfaces import HubContentInfo


def test_get_info_from_hub_resource_arn():
    model_arn = (
        "arn:aws:sagemaker:us-west-2:000000000000:hub-content/MockHub/Model/my-mock-model/1.0.2"
    )
    assert utils.get_info_from_hub_resource_arn(model_arn) == HubArnExtractedInfo(
        partition="aws",
        region="us-west-2",
        account_id="000000000000",
        hub_name="MockHub",
        hub_content_type="Model",
        hub_content_name="my-mock-model",
        hub_content_version="1.0.2",
    )

    notebook_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub-content/MockHub/Notebook/my-mock-notebook/1.0.2"
    assert None is utils.get_info_from_hub_resource_arn(notebook_arn)

    hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/MockHub"
    assert utils.get_info_from_hub_resource_arn(hub_arn) == HubArnExtractedInfo(
        partition="aws",
        region="us-west-2",
        account_id="000000000000",
        hub_name="MockHub",
    )

    invalid_arn = "arn:aws:sagemaker:us-west-2:000000000000:endpoint/my-endpoint-123"
    assert None is utils.get_info_from_hub_resource_arn(invalid_arn)

    invalid_arn = "nonsense-string"
    assert None is utils.get_info_from_hub_resource_arn(invalid_arn)

    invalid_arn = ""
    assert None is utils.get_info_from_hub_resource_arn(invalid_arn)


def test_construct_hub_arn_from_name():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.boto_region_name = "us-west-2"
    hub_name = "my-cool-hub"

    assert (
        utils.construct_hub_arn_from_name(hub_name=hub_name, session=mock_sagemaker_session)
        == "arn:aws:sagemaker:us-west-2:123456789123:hub/my-cool-hub"
    )

    assert (
        utils.construct_hub_arn_from_name(
            hub_name=hub_name, region="us-east-1", session=mock_sagemaker_session
        )
        == "arn:aws:sagemaker:us-east-1:123456789123:hub/my-cool-hub"
    )


def test_construct_hub_model_arn_from_inputs():
    model_name, version = "pytorch-ic-imagenet-v2", "1.0.2"
    hub_arn = "arn:aws:sagemaker:us-west-2:123456789123:hub/my-mock-hub"

    assert (
        utils.construct_hub_model_arn_from_inputs(hub_arn, model_name, version)
        == "arn:aws:sagemaker:us-west-2:123456789123:hub-content/my-mock-hub/Model/pytorch-ic-imagenet-v2/1.0.2"
    )

    version = "*"
    assert (
        utils.construct_hub_model_arn_from_inputs(hub_arn, model_name, version)
        == "arn:aws:sagemaker:us-west-2:123456789123:hub-content/my-mock-hub/Model/pytorch-ic-imagenet-v2/*"
    )


def test_generate_hub_arn_for_init_kwargs():
    hub_name = "my-hub-name"
    hub_arn = "arn:aws:sagemaker:us-west-2:12346789123:hub/my-awesome-hub"
    # Mock default session with default values
    mock_default_session = Mock()
    mock_default_session.account_id.return_value = "123456789123"
    mock_default_session.boto_region_name = JUMPSTART_DEFAULT_REGION_NAME
    # Mock custom session with custom values
    mock_custom_session = Mock()
    mock_custom_session.account_id.return_value = "000000000000"
    mock_custom_session.boto_region_name = "us-east-2"

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_name, session=mock_default_session)
        == "arn:aws:sagemaker:us-west-2:123456789123:hub/my-hub-name"
    )

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_name, "us-east-1", session=mock_default_session)
        == "arn:aws:sagemaker:us-east-1:123456789123:hub/my-hub-name"
    )

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_name, "eu-west-1", mock_custom_session)
        == "arn:aws:sagemaker:eu-west-1:000000000000:hub/my-hub-name"
    )

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_name, None, mock_custom_session)
        == "arn:aws:sagemaker:us-east-2:000000000000:hub/my-hub-name"
    )

    assert utils.generate_hub_arn_for_init_kwargs(hub_arn, session=mock_default_session) == hub_arn

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_arn, "us-east-1", session=mock_default_session)
        == hub_arn
    )

    assert (
        utils.generate_hub_arn_for_init_kwargs(hub_arn, "us-east-1", mock_custom_session) == hub_arn
    )

    assert utils.generate_hub_arn_for_init_kwargs(hub_arn, None, mock_custom_session) == hub_arn


def test_generate_default_hub_bucket_name():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.boto_region_name = "us-east-1"

    assert (
        utils.generate_default_hub_bucket_name(sagemaker_session=mock_sagemaker_session)
        == "sagemaker-hubs-us-east-1-123456789123"
    )


def test_create_hub_bucket_if_it_does_not_exist_hub_arn():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.client("sts").get_caller_identity.return_value = {
        "Account": "123456789123"
    }
    hub_arn = "arn:aws:sagemaker:us-west-2:12346789123:hub/my-awesome-hub"
    # Mock custom session with custom values
    mock_custom_session = Mock()
    mock_custom_session.account_id.return_value = "000000000000"
    mock_custom_session.boto_region_name = "us-east-2"
    mock_sagemaker_session.boto_session.resource("s3").Bucket().creation_date = None
    mock_sagemaker_session.boto_region_name = "us-east-1"

    bucket_name = "sagemaker-hubs-us-east-1-123456789123"
    created_hub_bucket_name = utils.create_hub_bucket_if_it_does_not_exist(
        sagemaker_session=mock_sagemaker_session
    )

    mock_sagemaker_session.boto_session.resource("s3").create_bucketassert_called_once()
    assert created_hub_bucket_name == bucket_name
    assert utils.generate_hub_arn_for_init_kwargs(hub_arn, None, mock_custom_session) == hub_arn


def test_is_gated_bucket():
    assert utils.is_gated_bucket("jumpstart-private-cache-prod-us-west-2") is True

    assert utils.is_gated_bucket("jumpstart-private-cache-prod-us-east-1") is True

    assert utils.is_gated_bucket("jumpstart-cache-prod-us-west-2") is False

    assert utils.is_gated_bucket("") is False


def test_create_hub_bucket_if_it_does_not_exist():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.client("sts").get_caller_identity.return_value = {
        "Account": "123456789123"
    }
    mock_sagemaker_session.boto_session.resource("s3").Bucket().creation_date = None
    mock_sagemaker_session.boto_region_name = "us-east-1"
    bucket_name = "sagemaker-hubs-us-east-1-123456789123"
    created_hub_bucket_name = utils.create_hub_bucket_if_it_does_not_exist(
        sagemaker_session=mock_sagemaker_session
    )

    mock_sagemaker_session.boto_session.resource("s3").create_bucketassert_called_once()
    assert created_hub_bucket_name == bucket_name


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_find_tags_for_jumpstart_model_version(mock_spec_util):
    mock_sagemaker_session = Mock()
    mock_specs = Mock()
    mock_specs.deprecated = True
    mock_specs.inference_vulnerable = True
    mock_specs.training_vulnerable = True
    mock_spec_util.return_value = mock_specs

    tags = utils.find_unsupported_flags_for_model_version(
        model_id="test", version="test", region="test", session=mock_sagemaker_session
    )

    mock_spec_util.assert_called_once_with(
        model_id="test",
        version="test",
        region="test",
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=mock_sagemaker_session,
    )

    assert tags == [
        CuratedHubUnsupportedFlag.DEPRECATED_VERSIONS,
        CuratedHubUnsupportedFlag.INFERENCE_VULNERABLE_VERSIONS,
        CuratedHubUnsupportedFlag.TRAINING_VULNERABLE_VERSIONS,
    ]


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_find_tags_for_jumpstart_model_version_some_false(mock_spec_util):
    mock_sagemaker_session = Mock()
    mock_specs = Mock()
    mock_specs.deprecated = True
    mock_specs.inference_vulnerable = False
    mock_specs.training_vulnerable = False
    mock_spec_util.return_value = mock_specs

    tags = utils.find_unsupported_flags_for_model_version(
        model_id="test", version="test", region="test", session=mock_sagemaker_session
    )

    mock_spec_util.assert_called_once_with(
        model_id="test",
        version="test",
        region="test",
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=mock_sagemaker_session,
    )

    assert tags == [CuratedHubUnsupportedFlag.DEPRECATED_VERSIONS]


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_find_tags_for_jumpstart_model_version_all_false(mock_spec_util):
    mock_sagemaker_session = Mock()
    mock_specs = Mock()
    mock_specs.deprecated = False
    mock_specs.inference_vulnerable = False
    mock_specs.training_vulnerable = False
    mock_spec_util.return_value = mock_specs

    tags = utils.find_unsupported_flags_for_model_version(
        model_id="test", version="test", region="test", session=mock_sagemaker_session
    )

    mock_spec_util.assert_called_once_with(
        model_id="test",
        version="test",
        region="test",
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=mock_sagemaker_session,
    )

    assert tags == []


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_find_all_tags_for_jumpstart_model_filters_non_jumpstart_models(mock_spec_util):
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.list_hub_content_versions.return_value = {
        "HubContentSummaries": [
            {
                "CreationTime": "test_creation_timestamp_0",
                "HubContentArn": "test_arn_0",
                "HubContentName": "test_name",
                "HubContentVersion": "1.0.0",
                "HubContentType": "Model",
                "DocumentSchemaVersion": "test_schema",
                "HubContentStatus": "test",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-one-pytorch",
                    "@jumpstart-model-version:1.0.3",
                ],
            },
            {
                "HubContentName": "test_name",
                "CreationTime": "test_creation_timestamp_1",
                "HubContentVersion": "2.0.0",
                "HubContentArn": "test_arn_1",
                "HubContentType": "Model",
                "DocumentSchemaVersion": "test_schema",
                "HubContentStatus": "test",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-four-huggingface",
                    "@jumpstart-model-version:2.0.2",
                ],
            },
            {
                "CreationTime": "test_creation_timestamp_3",
                "HubContentArn": "test_arn_3",
                "HubContentName": "test_name",
                "HubContentType": "Model",
                "DocumentSchemaVersion": "test_schema",
                "HubContentVersion": "3.0.0",
                "HubContentStatus": "test",
                "HubContentSearchKeywords": [],
            },
        ]
    }

    mock_specs = Mock()
    mock_specs.deprecated = True
    mock_specs.inference_vulnerable = True
    mock_specs.training_vulnerable = True
    mock_spec_util.return_value = mock_specs

    tags = utils.find_deprecated_vulnerable_flags_for_hub_content(
        hub_name="test", hub_content_name="test", region="test", session=mock_sagemaker_session
    )

    mock_sagemaker_session.list_hub_content_versions.assert_called_once_with(
        hub_name="test",
        hub_content_type="Model",
        hub_content_name="test",
    )

    assert tags == [
        {
            "Key": "deprecated_versions",
            "Value": str(["1.0.0", "2.0.0"]),
        },
        {
            "Key": "inference_vulnerable_versions",
            "Value": str(["1.0.0", "2.0.0"]),
        },
        {
            "Key": "training_vulnerable_versions",
            "Value": str(["1.0.0", "2.0.0"]),
        },
    ]


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_summary_from_list_api_response(mock_spec_util):
    test = summary_from_list_api_response(
        {
            "HubContentArn": "test_arn",
            "HubContentName": "test_name",
            "HubContentVersion": "test_version",
            "HubContentType": "Model",
            "DocumentSchemaVersion": "test_schema",
            "HubContentStatus": "test",
            "HubContentDescription": "test_description",
            "HubContentSearchKeywords": ["test"],
            "CreationTime": "test_creation",
        }
    )

    assert test == HubContentInfo(
        {
            "HubContentArn": "test_arn",
            "HubContentName": "test_name",
            "HubContentVersion": "test_version",
            "HubContentType": "Model",
            "DocumentSchemaVersion": "test_schema",
            "HubContentStatus": "test",
            "HubContentDescription": "test_description",
            "HubContentSearchKeywords": ["test"],
            "CreationTime": "test_creation",
        }
    )


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
def test_summaries_from_list_api_response(mock_spec_util):
    test = summary_list_from_list_api_response(
        {
            "HubContentSummaries": [
                {
                    "HubContentArn": "test",
                    "HubContentName": "test",
                    "HubContentVersion": "test",
                    "HubContentType": "Model",
                    "DocumentSchemaVersion": "test",
                    "HubContentStatus": "test",
                    "HubContentDescription": "test",
                    "HubContentSearchKeywords": ["test", "test_2"],
                    "CreationTime": "test",
                },
                {
                    "HubContentArn": "test_2",
                    "HubContentName": "test_2",
                    "HubContentVersion": "test_2",
                    "HubContentType": "Model",
                    "DocumentSchemaVersion": "test_2",
                    "HubContentStatus": "test_2",
                    "HubContentDescription": "test_2",
                    "HubContentSearchKeywords": ["test_2", "test_2_2"],
                    "CreationTime": "test_2",
                },
            ]
        }
    )

    assert test == [
        HubContentInfo(
            {
                "HubContentArn": "test",
                "HubContentName": "test",
                "HubContentVersion": "test",
                "HubContentType": "Model",
                "DocumentSchemaVersion": "test",
                "HubContentStatus": "test",
                "HubContentDescription": "test",
                "HubContentSearchKeywords": ["test", "test_2"],
                "CreationTime": "test",
            }
        ),
        HubContentInfo(
            {
                "HubContentArn": "test_2",
                "HubContentName": "test_2",
                "HubContentVersion": "test_2",
                "HubContentType": "Model",
                "DocumentSchemaVersion": "test_2",
                "HubContentStatus": "test_2",
                "HubContentDescription": "test_2",
                "HubContentSearchKeywords": ["test_2", "test_2_2"],
                "CreationTime": "test_2",
            }
        ),
    ]
