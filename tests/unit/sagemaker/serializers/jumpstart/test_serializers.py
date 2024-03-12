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

from sagemaker import base_serializers, serializers
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.artifacts.predictors.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_default_serializers(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    model_id, model_version = "predictor-specs-model", "*"
    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client)

    default_serializer = serializers.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        sagemaker_session=mock_session,
    )
    assert isinstance(default_serializer, base_serializers.IdentitySerializer)

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )

    patched_get_model_specs.reset_mock()


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.artifacts.predictors.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_serializer_options(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client)

    model_id, model_version = "predictor-specs-model", "*"
    region = "us-west-2"

    serializer_options = serializers.retrieve_options(
        region=region,
        model_id=model_id,
        model_version=model_version,
        sagemaker_session=mock_session,
    )
    assert len(serializer_options) == 1
    assert all(
        [
            isinstance(serializer, base_serializers.IdentitySerializer)
            for serializer in serializer_options
        ]
    )

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )
