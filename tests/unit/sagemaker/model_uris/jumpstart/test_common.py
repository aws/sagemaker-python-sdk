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

from sagemaker import model_uris

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from sagemaker.jumpstart import constants as sagemaker_constants


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_common_model_uri(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_spec_from_base_spec

    model_uris.retrieve(
        model_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        version="*",
    )

    patched_get_model_specs.reset_mock()

    model_uris.retrieve(
        model_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
    )
    patched_get_model_specs.assert_called_once_with(
        region=sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME,
        model_id="pytorch-ic-mobilenet-v2",
        version="1.*",
    )

    patched_get_model_specs.reset_mock()

    model_uris.retrieve(
        region="us-west-2",
        model_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    patched_get_model_specs.reset_mock()

    model_uris.retrieve(
        region="us-west-2",
        model_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="1.*",
    )
    patched_get_model_specs.assert_called_once_with(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="1.*"
    )

    with pytest.raises(ValueError):
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
