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

import boto3
from mock.mock import patch, Mock
import pytest

from sagemaker import image_uris
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from sagemaker.jumpstart import constants as sagemaker_constants


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_common_image_uri(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)

    image_uris.retrieve(
        framework=None,
        region="us-west-2",
        image_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2",
        model_id="pytorch-ic-mobilenet-v2",
        hub_arn=None,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    image_uris.retrieve(
        framework=None,
        region="us-west-2",
        image_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2",
        model_id="pytorch-ic-mobilenet-v2",
        hub_arn=None,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    image_uris.retrieve(
        framework=None,
        region=None,
        image_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        hub_arn=None,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    image_uris.retrieve(
        framework=None,
        region=None,
        image_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        hub_arn=None,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    with pytest.raises(NotImplementedError):
        image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="BAD_SCOPE",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
            instance_type="ml.m5.xlarge",
        )

    with pytest.raises(KeyError):
        image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="training",
            model_id="blah",
            model_version="*",
            instance_type="ml.m5.xlarge",
        )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=None,
            region="mars-south-1",
            image_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
            instance_type="ml.m5.xlarge",
        )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=None,
            region="us-west-2",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
            instance_type="ml.m5.xlarge",
        )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="training",
            model_version="*",
            instance_type="ml.m5.xlarge",
        )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            region="us-west-2",
            framework=None,
            image_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
            instance_type="ml.m5.xlarge",
        )


@patch("sagemaker.image_uris.JUMPSTART_LOGGER.info")
@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_image_uri_logging_extra_fields(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
    patched_info_log,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)

    image_uris.retrieve(
        framework=None,
        region="us-west-2",
        image_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )

    patched_info_log.assert_not_called()

    image_uris.retrieve(
        framework="framework",
        container_version="1.2.3",
        region="us-west-2",
        image_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )

    patched_info_log.assert_called_once_with(
        "Ignoring the following fields "
        "when retriving image uri for "
        "JumpStart model id '%s': %s",
        "pytorch-ic-mobilenet-v2",
        "{'framework': 'framework', 'container_version': '1.2.3'}",
    )
