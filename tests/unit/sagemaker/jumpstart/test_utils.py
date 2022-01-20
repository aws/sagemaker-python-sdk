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
from mock.mock import Mock, patch
import pytest
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.constants import JUMPSTART_REGION_NAME_SET, JumpStartScriptScope
from sagemaker.jumpstart.exceptions import (
    DeprecatedJumpStartModelError,
    VulnerableJumpStartModelError,
)
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartVersionedModelId
from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


def test_get_jumpstart_content_bucket():
    bad_region = "bad_region"
    assert bad_region not in JUMPSTART_REGION_NAME_SET
    with pytest.raises(ValueError):
        utils.get_jumpstart_content_bucket(bad_region)


def test_get_jumpstart_launched_regions_message():

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {}):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is not available in any region."
        )

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"some_region"}):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is available in some_region region."
        )

    with patch(
        "sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"some_region1", "some_region2"}
    ):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is available in some_region1 and some_region2 regions."
        )

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"a", "b", "c"}):
        assert (
            utils.get_jumpstart_launched_regions_message()
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


def test_parse_sagemaker_version():

    with patch("sagemaker.__version__", "1.2.3"):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3.3332j"):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3."):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with pytest.raises(ValueError):
        with patch("sagemaker.__version__", "1.2.3dfsdfs"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", ""):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2.3.4.5"):
            utils.parse_sagemaker_version()


@patch("sagemaker.jumpstart.utils.parse_sagemaker_version")
@patch("sagemaker.jumpstart.accessors.SageMakerSettings._parsed_sagemaker_version", "")
def test_get_sagemaker_version(patched_parse_sm_version: Mock):
    utils.get_sagemaker_version()
    utils.get_sagemaker_version()
    utils.get_sagemaker_version()
    assert patched_parse_sm_version.called_only_once()


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_vulnerable_model(patched_get_model_specs):
    def make_vulnerable_inference_spec(*largs, **kwargs):
        spec = get_spec_from_base_spec(*largs, **kwargs)
        spec.inference_vulnerable = True
        spec.inference_vulnerabilities = ["some", "vulnerability"]
        return spec

    patched_get_model_specs.side_effect = make_vulnerable_inference_spec

    with pytest.raises(VulnerableJumpStartModelError) as e:
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.INFERENCE.value,
            region="us-west-2",
        )
    assert (
        "Version '*' of JumpStart model 'pytorch-eqa-bert-base-cased' has at least 1 "
        "vulnerable dependency in the inference script. "
        "Please try targetting a higher version of the model. "
        "List of vulnerabilities: some, vulnerability"
    ) == str(e.value.message)

    assert (
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.INFERENCE.value,
            region="us-west-2",
            tolerate_vulnerable_model=True,
        )
        is not None
    )

    def make_vulnerable_training_spec(*largs, **kwargs):
        spec = get_spec_from_base_spec(*largs, **kwargs)
        spec.training_vulnerable = True
        spec.training_vulnerabilities = ["some", "vulnerability"]
        return spec

    patched_get_model_specs.side_effect = make_vulnerable_training_spec

    with pytest.raises(VulnerableJumpStartModelError) as e:
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.TRAINING.value,
            region="us-west-2",
        )
    assert (
        "Version '*' of JumpStart model 'pytorch-eqa-bert-base-cased' has at least 1 "
        "vulnerable dependency in the training script. "
        "Please try targetting a higher version of the model. "
        "List of vulnerabilities: some, vulnerability"
    ) == str(e.value.message)

    assert (
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.TRAINING.value,
            region="us-west-2",
            tolerate_vulnerable_model=True,
        )
        is not None
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_deprecated_model(patched_get_model_specs):
    def make_deprecated_spec(*largs, **kwargs):
        spec = get_spec_from_base_spec(*largs, **kwargs)
        spec.deprecated = True
        return spec

    patched_get_model_specs.side_effect = make_deprecated_spec

    with pytest.raises(DeprecatedJumpStartModelError) as e:
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.INFERENCE.value,
            region="us-west-2",
        )
    assert "Version '*' of JumpStart model 'pytorch-eqa-bert-base-cased' is deprecated. "
    "Please try targetting a higher version of the model." == str(e.value.message)

    assert (
        utils.verify_model_region_and_return_specs(
            model_id="pytorch-eqa-bert-base-cased",
            version="*",
            scope=JumpStartScriptScope.INFERENCE.value,
            region="us-west-2",
            tolerate_deprecated_model=True,
        )
        is not None
    )
