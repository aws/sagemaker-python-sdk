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

from sagemaker import image_uris
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


@patch("sagemaker.jumpstart.artifacts.image_uris.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_variants_image_uri(
    patched_get_model_specs, patched_verify_model_region_and_return_specs
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    assert (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
        "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
        == image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="inference",
            model_id="variant-model",
            model_version="*",
            instance_type="ml.p2.xlarge",
        )
    )

    assert "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah" == image_uris.retrieve(
        framework=None,
        region="us-west-2",
        image_scope="inference",
        model_id="variant-model",
        model_version="*",
        instance_type="ml.c2.xlarge",
    )

    assert (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.5.0-cpu-py3"
        == image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="inference",
            model_id="variant-model",
            model_version="*",
            instance_type="ml.c200000.xlarge",
        )
    )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=None,
            region="us-west-29",
            image_scope="inference",
            model_id="variant-model",
            model_version="*",
            instance_type="ml.c2.xlarge",
        )

    assert (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.5.0-gpu-py3"
        == image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="training",
            model_id="variant-model",
            model_version="*",
            instance_type="ml.g4dn.2xlarge",
        )
    )
