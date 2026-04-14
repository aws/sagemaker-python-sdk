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
"""Unit tests for _resolve_data_cache_config and _resolve_container_spec."""
from __future__ import absolute_import

import pytest

from sagemaker.core.shapes import (
    InferenceComponentDataCacheConfig,
    InferenceComponentContainerSpecification,
)
from sagemaker.serve.model_builder_utils import _ModelBuilderUtils


class ConcreteUtils(_ModelBuilderUtils):
    """Concrete class to test mixin methods."""
    pass


@pytest.fixture
def utils():
    return ConcreteUtils()


# ============================================================
# Tests for _resolve_data_cache_config
# ============================================================

class TestResolveDataCacheConfig:
    def test_none_returns_none(self, utils):
        assert utils._resolve_data_cache_config(None) is None

    def test_already_typed_passthrough(self, utils):
        config = InferenceComponentDataCacheConfig(enable_caching=True)
        result = utils._resolve_data_cache_config(config)
        assert result is config
        assert result.enable_caching is True

    def test_dict_with_enable_caching_true(self, utils):
        result = utils._resolve_data_cache_config({"enable_caching": True})
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is True

    def test_dict_with_enable_caching_false(self, utils):
        result = utils._resolve_data_cache_config({"enable_caching": False})
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is False

    def test_dict_missing_enable_caching_raises(self, utils):
        with pytest.raises(ValueError, match="must contain the required 'enable_caching' key"):
            utils._resolve_data_cache_config({})

    def test_dict_with_extra_keys_still_works(self, utils):
        """Extra keys are ignored; only enable_caching is required."""
        result = utils._resolve_data_cache_config(
            {"enable_caching": True, "extra_key": "ignored"}
        )
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is True

    def test_invalid_type_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config("invalid")

    def test_invalid_type_int_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config(42)

    def test_invalid_type_list_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config([True])


# ============================================================
# Tests for _resolve_container_spec
# ============================================================

class TestResolveContainerSpec:
    def test_none_returns_none(self, utils):
        assert utils._resolve_container_spec(None) is None

    def test_already_typed_passthrough(self, utils):
        spec = InferenceComponentContainerSpecification(
            image="my-image:latest",
            artifact_url="s3://bucket/artifact",
            environment={"KEY": "VALUE"},
        )
        result = utils._resolve_container_spec(spec)
        assert result is spec

    def test_dict_full(self, utils):
        result = utils._resolve_container_spec({
            "image": "my-image:latest",
            "artifact_url": "s3://bucket/artifact",
            "environment": {"KEY": "VALUE"},
        })
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "my-image:latest"
        assert result.artifact_url == "s3://bucket/artifact"
        assert result.environment == {"KEY": "VALUE"}

    def test_dict_image_only(self, utils):
        result = utils._resolve_container_spec({"image": "my-image:latest"})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "my-image:latest"

    def test_dict_artifact_url_only(self, utils):
        result = utils._resolve_container_spec({"artifact_url": "s3://bucket/model.tar.gz"})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.artifact_url == "s3://bucket/model.tar.gz"

    def test_dict_environment_only(self, utils):
        result = utils._resolve_container_spec({"environment": {"A": "B"}})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.environment == {"A": "B"}

    def test_dict_empty(self, utils):
        """Empty dict creates a spec with no fields set."""
        result = utils._resolve_container_spec({})
        assert isinstance(result, InferenceComponentContainerSpecification)

    def test_dict_with_extra_keys(self, utils):
        """Extra keys are ignored."""
        result = utils._resolve_container_spec({
            "image": "img",
            "unknown_key": "ignored",
        })
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "img"

    def test_invalid_type_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec("invalid")

    def test_invalid_type_int_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec(123)

    def test_invalid_type_list_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec([{"image": "img"}])
