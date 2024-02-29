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
from sagemaker.jumpstart.curated_hub import utils
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.curated_hub.types import HubArnExtractedInfo


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
    assert utils.get_info_from_hub_resource_arn(notebook_arn) == HubArnExtractedInfo(
        partition="aws",
        region="us-west-2",
        account_id="000000000000",
        hub_name="MockHub",
        hub_content_type="Notebook",
        hub_content_name="my-mock-notebook",
        hub_content_version="1.0.2",
    )

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

    assert (
        utils.generate_hub_arn_for_estimator_init_kwargs(hub_arn, None, mock_custom_session)
        == hub_arn
    )


def test_generate_default_hub_bucket_name():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.boto_region_name = "us-east-1"

    assert (
        utils.generate_default_hub_bucket_name(sagemaker_session=mock_sagemaker_session)
        == "sagemaker-hubs-us-east-1-123456789123"
    )


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
    assert utils.generate_hub_arn_for_init_kwargs(hub_arn, None, mock_custom_session) == hub_arn


def test_generate_default_hub_bucket_name():
    mock_sagemaker_session = Mock()
    mock_sagemaker_session.account_id.return_value = "123456789123"
    mock_sagemaker_session.boto_region_name = "us-east-1"

    assert (
        utils.generate_default_hub_bucket_name(sagemaker_session=mock_sagemaker_session)
        == "sagemaker-hubs-us-east-1-123456789123"
    )


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
