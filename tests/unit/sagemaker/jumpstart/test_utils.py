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
from mock.mock import patch
import pytest
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.constants import REGION_NAME_SET
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartVersionedModelId
from sagemaker.jumpstart.types import JumpStartLaunchedRegionInfo


def test_get_jumpstart_content_bucket():
    bad_region = "bad_region"
    assert bad_region not in REGION_NAME_SET
    with pytest.raises(RuntimeError):
        utils.get_jumpstart_content_bucket(bad_region)


def test_get_jumpstart_launched_regions_string():

    with patch("sagemaker.jumpstart.constants.REGION_NAME_SET", {}):
        assert (
            utils.get_jumpstart_launched_regions_string()
            == "JumpStart is not available in any region."
        )

    with patch("sagemaker.jumpstart.constants.REGION_NAME_SET", {"some_region"}):
        assert (
            utils.get_jumpstart_launched_regions_string()
            == "JumpStart is available in some_region region."
        )

    with patch("sagemaker.jumpstart.constants.REGION_NAME_SET", {"some_region1", "some_region2"}):
        assert (
            utils.get_jumpstart_launched_regions_string()
            == "JumpStart is available in some_region1 and some_region2 regions."
        )

    with patch("sagemaker.jumpstart.constants.REGION_NAME_SET", {"a", "b", "c"}):
        assert (
            utils.get_jumpstart_launched_regions_string()
            == "JumpStart is available in a, b, and c regions."
        )


def test_get_formatted_manifest():
    mock_manifest = [
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        },
    ]

    assert utils.get_formatted_manifest(mock_manifest) == {
        JumpStartVersionedModelId(
            "tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"
        ): JumpStartModelHeader(mock_manifest[0])
    }

    assert utils.get_formatted_manifest([]) == {}


def test_get_sagemaker_version():

    with patch("sagemaker.__version__", "1.2.3"):
        assert utils.get_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3.3332j"):
        assert utils.get_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3."):
        assert utils.get_sagemaker_version() == "1.2.3"

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2.3dfsdfs"):
            utils.get_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2"):
            utils.get_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1"):
            utils.get_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", ""):
            utils.get_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2.3.4.5"):
            utils.get_sagemaker_version()
