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


from mock.mock import patch
import pytest

from sagemaker import metric_definitions

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_default_metric_definitions(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_spec_from_base_spec

    model_id = "pytorch-ic-mobilenet-v2"
    region = "us-west-2"

    definitions = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
    )
    assert definitions == [
        {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
    ]

    patched_get_model_specs.assert_called_once_with(region=region, model_id=model_id, version="*")

    patched_get_model_specs.reset_mock()

    definitions = metric_definitions.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
    )
    assert definitions == [
        {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
    ]

    patched_get_model_specs.assert_called_once_with(region=region, model_id=model_id, version="1.*")

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
