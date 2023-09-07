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
from tests.unit.sagemaker.jumpstart.constants import BASE_MANIFEST
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


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
def test_jumpstart_models_cache_get_fxs(mock_cache):

    mock_cache.get_manifest = Mock(return_value=BASE_MANIFEST)
    mock_cache.get_header = Mock(side_effect=get_header_from_base_header)
    mock_cache.get_specs = Mock(side_effect=get_spec_from_base_spec)

    assert get_header_from_base_header(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    ) == accessors.JumpStartModelsAccessor.get_model_header(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )
    assert get_spec_from_base_spec(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    ) == accessors.JumpStartModelsAccessor.get_model_specs(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    assert len(accessors.JumpStartModelsAccessor._get_manifest()) > 0

    # necessary because accessors is a static module
    reload(accessors)


@patch("sagemaker.jumpstart.cache.JumpStartModelsCache")
def test_jumpstart_models_cache_set_reset_fxs(mock_model_cache: Mock):

    # test change of region resets cache
    accessors.JumpStartModelsAccessor.get_model_header(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    accessors.JumpStartModelsAccessor.get_model_specs(
        region="us-west-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsAccessor.get_model_header(
        region="us-east-2", model_id="pytorch-ic-mobilenet-v2", version="*"
    )

    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsAccessor.get_model_specs(
        region="us-west-1", model_id="pytorch-ic-mobilenet-v2", version="*"
    )
    mock_model_cache.assert_called_once()
    mock_model_cache.reset_mock()

    # test set_cache_kwargs
    accessors.JumpStartModelsAccessor.set_cache_kwargs(cache_kwargs={"some": "kwarg"})
    mock_model_cache.assert_called_once_with(some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsAccessor.set_cache_kwargs(
        region="us-west-2", cache_kwargs={"some": "kwarg"}
    )
    mock_model_cache.assert_called_once_with(region="us-west-2", some="kwarg")
    mock_model_cache.reset_mock()

    # test reset cache
    accessors.JumpStartModelsAccessor.reset_cache(cache_kwargs={"some": "kwarg"})
    mock_model_cache.assert_called_once_with(some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsAccessor.reset_cache(
        region="us-west-2", cache_kwargs={"some": "kwarg"}
    )
    mock_model_cache.assert_called_once_with(region="us-west-2", some="kwarg")
    mock_model_cache.reset_mock()

    accessors.JumpStartModelsAccessor.reset_cache()
    mock_model_cache.assert_called_once_with()
    mock_model_cache.reset_mock()

    # validate region and cache kwargs utility
    assert {
        "some": "kwarg"
    } == accessors.JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
        {"some": "kwarg"}, "us-west-2"
    )
    assert {
        "some": "kwarg"
    } == accessors.JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
        {"some": "kwarg", "region": "us-west-2"}, "us-west-2"
    )

    with pytest.raises(ValueError):
        accessors.JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            {"some": "kwarg", "region": "us-east-2"}, "us-west-2"
        )

    # necessary because accessors is a static module
    reload(accessors)
