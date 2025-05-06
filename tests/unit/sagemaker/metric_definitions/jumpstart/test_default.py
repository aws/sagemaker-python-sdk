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

from sagemaker import metric_definitions
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec

mock_client = boto3.client("s3")
region = "us-west-2"
mock_session = Mock(s3_client=mock_client, boto_region_name=region)


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_default_metric_definitions(
    patched_get_model_specs, patched_validate_model_id_and_get_type
):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    mock_client = boto3.client("s3")
    region = "us-west-2"
    mock_session = Mock(s3_client=mock_client, boto_region_name=region)

    model_id = "pytorch-ic-mobilenet-v2"
    region = "us-west-2"

    definitions = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
    )
    assert definitions == [
        {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
    ]

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
        hub_arn=None,
    )

    patched_get_model_specs.reset_mock()

    definitions = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
        sagemaker_session=mock_session,
    )
    assert definitions == [
        {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
    ]

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=mock_session,
        hub_arn=None,
    )

    patched_get_model_specs.reset_mock()

    with pytest.raises(KeyError):
        metric_definitions.retrieve_default(
            region=region,
            model_id="blah",
            model_version="*",
        )

    with pytest.raises(ValueError):
        metric_definitions.retrieve_default(
            region="mars-south-1",
            model_id=model_id,
            model_version="*",
        )

    with pytest.raises(ValueError):
        metric_definitions.retrieve_default(
            model_version="*",
        )

    with pytest.raises(ValueError):
        metric_definitions.retrieve_default(
            model_id=model_id,
        )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_metric_definitions_instance_type_overrides(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id = "variant-model"
    region = "us-west-2"

    # assert that we can add metric definitions to default
    metrics = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p2.48xlarge",
    )
    assert metrics == [
        {
            "Name": "huggingface-textgeyyyuyuyuyneration:train-loss",
            "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:wtafigo",
            "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
        },
        {"Name": "huggingface-textgeneration:eval-loss", "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)"},
        {
            "Name": "huggingface-textgeneration:train-loss",
            "Regex": "'instance family specific': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:noneyourbusiness-loss",
            "Regex": "'loss-noyb': ([0-9]+\\.[0-9]+)",
        },
    ]

    # assert that we can override default metric definitions (instance family + instance type
    # specific)
    metrics = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p2.12xlarge",
    )
    assert metrics == [
        {
            "Name": "huggingface-textgeyyyuyuyuyneration:train-loss",
            "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:instance-typemetric-loss",
            "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
        },
        {"Name": "huggingface-textgeneration:eval-loss", "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)"},
        {
            "Name": "huggingface-textgeneration:train-loss",
            "Regex": "'instance type specific': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:noneyourbusiness-loss",
            "Regex": "'loss-noyb instance specific': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:wtafigo",
            "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
        },
    ]

    # assert that we can return default metric definitions for unrecognized instance
    metrics = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p9999.48xlarge",
    )

    assert metrics == [
        {
            "Name": "huggingface-textgeneration:train-loss",
            "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeyyyuyuyuyneration:train-loss",
            "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
        },
    ]
