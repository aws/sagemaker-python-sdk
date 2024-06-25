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

import boto3
from mock.mock import patch
import pytest

from sagemaker import model_uris
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from sagemaker.jumpstart import constants as sagemaker_constants


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.artifacts.model_uris.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_common_model_uri(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    mock_client = boto3.client("s3")
    region = "us-west-2"
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)

    model_uris.retrieve(
        model_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    model_uris.retrieve(
        model_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    model_uris.retrieve(
        region="us-west-2",
        model_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2",
        model_id="pytorch-ic-mobilenet-v2",
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    patched_get_model_specs.reset_mock()
    patched_verify_model_region_and_return_specs.reset_mock()

    model_uris.retrieve(
        region="us-west-2",
        model_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
        sagemaker_session=mock_session,
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2",
        model_id="pytorch-ic-mobilenet-v2",
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )
    patched_verify_model_region_and_return_specs.assert_called_once()

    with pytest.raises(NotImplementedError):
        model_uris.retrieve(
            region="us-west-2",
            model_scope="BAD_SCOPE",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(KeyError):
        model_uris.retrieve(
            region="us-west-2",
            model_scope="training",
            model_id="blah",
            model_version="*",
        )

    with pytest.raises(ValueError):
        model_uris.retrieve(
            region="mars-south-1",
            model_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(ValueError):
        model_uris.retrieve(
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(ValueError):
        model_uris.retrieve(
            model_scope="training",
            model_version="*",
        )

    with pytest.raises(ValueError):
        model_uris.retrieve(
            model_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
        )


@patch("sagemaker.jumpstart.artifacts.model_uris.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch.dict(
    "sagemaker.jumpstart.cache.os.environ",
    {
        sagemaker_constants.ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE: "some-cool-bucket-name"
    },
)
def test_jumpstart_artifact_bucket_override(
    patched_get_model_specs, patched_verify_model_region_and_return_specs
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_spec_from_base_spec

    uri = model_uris.retrieve(
        model_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    assert uri == "s3://some-cool-bucket-name/pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz"
