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

from sagemaker import instance_types
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_instance_types(patched_get_model_specs, patched_validate_model_id_and_get_type):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    model_id, model_version = "huggingface-eqa-bert-base-cased", "*"
    region = "us-west-2"

    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)

    default_training_instance_types = instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="training",
        sagemaker_session=mock_session,
    )
    assert default_training_instance_types == "ml.m5.xlarge"

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
        hub_arn=None,
    )

    patched_get_model_specs.reset_mock()

    default_inference_instance_types = instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
    )
    assert default_inference_instance_types == "ml.m5.large"

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    default_training_instance_types = instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="training",
        sagemaker_session=mock_session,
    )
    assert default_training_instance_types == ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.m4.xlarge"]

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    default_inference_instance_types = instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
    )
    assert default_inference_instance_types == [
        "ml.m5.large",
        "ml.m5.xlarge",
        "ml.c5.xlarge",
        "ml.c5.2xlarge",
        "ml.m4.large",
        "ml.m4.xlarge",
    ]

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    with pytest.raises(ValueError):
        instance_types.retrieve_default(
            region=region,
            model_id=model_id,
            model_version="*",
        )

    with pytest.raises(ValueError):
        instance_types.retrieve(
            region=region,
            model_id=model_id,
            model_version="*",
        )

    with pytest.raises(KeyError):
        instance_types.retrieve_default(
            region=region, model_id="blah", model_version="*", scope="inference"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve_default(
            region="mars-south-1", model_id=model_id, model_version="*", scope="training"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve_default(model_version="*", scope="inference")

    with pytest.raises(ValueError):
        instance_types.retrieve_default(model_id=model_id, scope="training")

    with pytest.raises(KeyError):
        instance_types.retrieve(
            region=region, model_id="blah", model_version="*", scope="inference"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve(
            region="mars-south-1", model_id=model_id, model_version="*", scope="training"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve(model_version="*", scope="inference")

    with pytest.raises(ValueError):
        instance_types.retrieve(model_id=model_id, scope="training")


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_inference_instance_type_variants(patched_get_model_specs):
    patched_get_model_specs.side_effect = get_special_model_spec

    mock_client = boto3.client("s3")
    region = "us-west-2"
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)
    model_id, model_version = "inference-instance-types-variant-model", "*"
    region = "us-west-2"

    assert ["ml.inf1.2xlarge", "ml.inf1.xlarge"] == instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.trn1.xlarge",
    )

    assert ["ml.inf1.2xlarge", "ml.inf1.xlarge"] == instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.trn1.12xlarge",
    )

    assert ["ml.p2.xlarge", "ml.p3.xlarge", "ml.p5.xlarge"] == instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.p2.12xlarge",
    )

    assert ["ml.p4de.24xlarge"] == instance_types.retrieve(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.p29s.12xlarge",
    )

    assert "ml.inf1.xlarge" == instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.trn1.xlarge",
    )

    assert "ml.inf1.xlarge" == instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.trn1.12xlarge",
    )

    assert "ml.p5.xlarge" == instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.p2.12xlarge",
    )

    assert "ml.p4de.24xlarge" == instance_types.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        training_instance_type="ml.p29s.12xlarge",
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_no_supported_instance_types(patched_get_model_specs):
    patched_get_model_specs.side_effect = get_special_model_spec

    model_id, model_version = "no-supported-instance-types-model", "*"
    region = "us-west-2"

    with pytest.raises(ValueError):
        instance_types.retrieve_default(
            region=region, model_id=model_id, model_version=model_version, scope="training"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve_default(
            region=region, model_id=model_id, model_version=model_version, scope="inference"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve(
            region=region, model_id=model_id, model_version=model_version, scope="training"
        )

    with pytest.raises(ValueError):
        instance_types.retrieve(
            region=region, model_id=model_id, model_version=model_version, scope="inference"
        )
