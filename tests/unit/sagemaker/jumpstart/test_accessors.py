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

from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.utils import (
    get_header_from_base_header,
    get_spec_from_base_spec,
)
from importlib import reload


def test_jumpstart_sagemaker_settings():

    assert "" == accessors.SageMakerSettings.get_sagemaker_version()
    accessors.SageMakerSettings.set_sagemaker_version("1.0.1")
    assert "1.0.1" == accessors.SageMakerSettings.get_sagemaker_version()
    assert "1.0.1" == accessors.SageMakerSettings.get_sagemaker_version()
    accessors.SageMakerSettings.set_sagemaker_version("1.0.2")
    assert "1.0.2" == accessors.SageMakerSettings.get_sagemaker_version()

    # necessary because accessors is a static module
    reload(accessors)


@patch("sagemaker.jumpstart.cache.JumpStartModelsCache.get_header", get_header_from_base_header)
@patch("sagemaker.jumpstart.cache.JumpStartModelsCache.get_specs", get_spec_from_base_spec)
def test_jumpstart_models_cache_get_fxs():

    assert JumpStartModelHeader(
        {
            "model_id": "pytorch-ic-mobilenet-v2",
            "version": "*",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == accessors.JumpStartModelsCache.get_model_header(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )
    assert JumpStartModelSpecs(
        {
            "model_id": "pytorch-ic-mobilenet-v2",
            "version": "*",
            "min_sdk_version": "2.49.0",
            "incremental_training_supported": True,
            "hosting_ecr_specs": {
                "py_version": "py3",
                "framework": "pytorch",
                "framework_version": "1.5.0",
            },
            "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
            "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
            "training_supported": True,
            "training_ecr_specs": {
                "py_version": "py3",
                "framework": "pytorch",
                "framework_version": "1.5.0",
            },
            "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
            "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
            "hyperparameters": {
                "adam-learning-rate": {"type": "float", "default": 0.05, "min": 1e-08, "max": 1},
                "epochs": {"type": "int", "default": 3, "min": 1, "max": 1000},
                "batch-size": {"type": "int", "default": 4, "min": 1, "max": 1024},
            },
        }
    ) == accessors.JumpStartModelsCache.get_model_specs(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    # necessary because accessors is a static module
    reload(accessors)


@patch("sagemaker.jumpstart.cache.JumpStartModelsCache")
def test_jumpstart_models_cache_set_reset_fxs(mock_model_cache: Mock):

    # test change of region resets cache
    accessors.JumpStartModelsCache.get_model_header(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    accessors.JumpStartModelsCache.get_model_specs(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsCache.get_model_header(
        region="us-east-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsCache.get_model_specs(
        region="us-west-1", model_id="pytorch-ic-mobilenet-v2", version="*"
    )
    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    # test set_cache_kwargs
    accessors.JumpStartModelsCache.set_cache_kwargs(cache_kwargs={"some": "kwarg"})
    mock_model_cache.assert_called_once_with(some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsCache.set_cache_kwargs(
        region="us-west-2", cache_kwargs={"some": "kwarg"}
    )
    mock_model_cache.assert_called_once_with(region="us-west-2", some="kwarg")
    mock_model_cache.reset_mock()

    # test reset cache
    accessors.JumpStartModelsCache.reset_cache(cache_kwargs={"some": "kwarg"})
    mock_model_cache.assert_called_once_with(some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsCache.reset_cache(region="us-west-2", cache_kwargs={"some": "kwarg"})
    mock_model_cache.assert_called_once_with(region="us-west-2", some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsCache.reset_cache()
    mock_model_cache.assert_called_once_with()
    mock_model_cache.reset_mock()

    # validate region and cache kwargs utility
    assert {"some": "kwarg"} == accessors.JumpStartModelsCache._validate_region_cache_kwargs(
        {"some": "kwarg"}, "us-west-2"
    )
    assert {"some": "kwarg"} == accessors.JumpStartModelsCache._validate_region_cache_kwargs(
        {"some": "kwarg", "region": "us-west-2"}, "us-west-2"
    )

    with pytest.raises(ValueError) as e:
        accessors.JumpStartModelsCache._validate_region_cache_kwargs(
            {"some": "kwarg", "region": "us-east-2"}, "us-west-2"
        )

    # necessary because accessors is a static module
    reload(accessors)
