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

from sagemaker import resource_requirements
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.jumpstart.artifacts.resource_requirements import (
    REQUIREMENT_TYPE_TO_SPEC_FIELD_NAME_TO_RESOURCE_REQUIREMENT_NAME_MAP,
)

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_resource_requirements(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client)

    model_id, model_version = "huggingface-llm-mistral-7b-instruct", "*"
    default_inference_resource_requirements = resource_requirements.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
    )
    assert default_inference_resource_requirements.requests["num_accelerators"] == 1
    assert default_inference_resource_requirements.requests["memory"] == 34360

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
    )
    patched_get_model_specs.reset_mock()


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_resource_requirements_instance_type_variants(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec
    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client)

    model_id, model_version = "variant-model", "*"
    default_inference_resource_requirements = resource_requirements.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        instance_type="ml.g5.xlarge",
    )
    assert default_inference_resource_requirements.requests == {
        "memory": 81999,
        "num_accelerators": 10,
    }

    default_inference_resource_requirements = resource_requirements.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        instance_type="ml.g5.555xlarge",
    )
    assert default_inference_resource_requirements.requests == {
        "memory": 81999,
        "num_accelerators": 888810,
    }

    default_inference_resource_requirements = resource_requirements.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
        instance_type="ml.f9.555xlarge",
    )
    assert default_inference_resource_requirements.requests == {
        "memory": 81999,
        "num_accelerators": 1,
    }


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_no_supported_resource_requirements(patched_get_model_specs):
    patched_get_model_specs.side_effect = get_special_model_spec

    model_id, model_version = "no-supported-instance-types-model", "*"
    region = "us-west-2"
    mock_client = boto3.client("s3")
    mock_session = Mock(s3_client=mock_client)

    default_inference_resource_requirements = resource_requirements.retrieve_default(
        region=region,
        model_id=model_id,
        model_version=model_version,
        scope="inference",
        sagemaker_session=mock_session,
    )
    assert default_inference_resource_requirements is None

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version=model_version,
        s3_client=mock_client,
    )
    patched_get_model_specs.reset_mock()

    with pytest.raises(NotImplementedError):
        resource_requirements.retrieve_default(
            region=region, model_id=model_id, model_version=model_version, scope="training"
        )


def test_jumpstart_supports_all_resource_requirement_fields():

    all_tracked_resource_requirement_fields = {
        field
        for requirements in REQUIREMENT_TYPE_TO_SPEC_FIELD_NAME_TO_RESOURCE_REQUIREMENT_NAME_MAP.values()
        for _, field in requirements.values()
    }

    excluded_resource_requirement_fields = {"requests", "limits"}
    assert (
        set(ResourceRequirements().__dict__.keys()) - excluded_resource_requirement_fields
        == all_tracked_resource_requirement_fields
    )
