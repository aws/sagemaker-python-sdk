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

from sagemaker import script_uris
import pytest

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart import constants as sagemaker_constants


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_script_uri(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    uri = script_uris.retrieve(
        region="us-west-2",
        script_scope="inference",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    assert (
        uri == f"s3://{get_jumpstart_content_bucket('us-west-2')}/"
        "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz"
    )
    patched_get_model_specs.assert_called_once_with("us-west-2", "pytorch-ic-mobilenet-v2", "*")

    patched_get_model_specs.reset_mock()

    uri = script_uris.retrieve(
        region="us-west-2",
        script_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    assert (
        uri == f"s3://{get_jumpstart_content_bucket('us-west-2')}/"
        "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz"
    )
    patched_get_model_specs.assert_called_once_with("us-west-2", "pytorch-ic-mobilenet-v2", "*")
    patched_get_model_specs.reset_mock()

    script_uris.retrieve(
        script_scope="training",
        model_id="pytorch-ic-mobilenet-v2",
        model_version="*",
    )
    patched_get_model_specs.assert_called_once_with(
        sagemaker_constants.JUMPSTART_DEFAULT_REGION_NAME, "pytorch-ic-mobilenet-v2", "*"
    )

    with pytest.raises(ValueError):
        script_uris.retrieve(
            region="us-west-2",
            script_scope="BAD_SCOPE",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(ValueError):
        script_uris.retrieve(
            region="mars-south-1",
            script_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(ValueError):
        script_uris.retrieve(
            model_id="pytorch-ic-mobilenet-v2",
            model_version="*",
        )

    with pytest.raises(ValueError):
        script_uris.retrieve(
            script_scope="training",
            model_version="*",
        )

    with pytest.raises(ValueError):
        script_uris.retrieve(
            script_scope="training",
            model_id="pytorch-ic-mobilenet-v2",
        )
