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
"""Unit tests for merged model (is_checkpoint=False) deployment path."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock


class TestFetchPeftMergedModel:
    """Tests for _fetch_peft with is_checkpoint=False."""

    def _make_model_builder_with_model_package(self, is_checkpoint, recipe_name):
        """Helper to create a ModelBuilder with mocked model package."""
        from sagemaker.serve.model_builder import ModelBuilder
        from sagemaker.core.resources import ModelPackage

        mock_mp = MagicMock()
        mock_container = MagicMock()
        mock_container.is_checkpoint = is_checkpoint
        mock_base_model = MagicMock()
        mock_base_model.recipe_name = recipe_name
        mock_container.base_model = mock_base_model
        mock_mp.inference_specification.containers = [mock_container]
        mock_mp.model_package_arn = "arn:aws:sagemaker:us-west-2:123:model-package/test/1"

        with patch.object(ModelBuilder, '__post_init__', lambda self: None):
            mb = ModelBuilder.__new__(ModelBuilder)
            # Use a real ModelPackage instance as mb.model so isinstance works on all Python versions
            # but override _fetch_model_package to return our mock with the test attributes
            real_mp = ModelPackage.__new__(ModelPackage)
            mb.model = real_mp
            mb._fetch_model_package = MagicMock(return_value=mock_mp)
        return mb

    def test_merged_model_returns_none(self):
        """is_checkpoint=False should return None even with lora in recipe name."""
        mb = self._make_model_builder_with_model_package(
            is_checkpoint=False, recipe_name="mtrl-gpt-oss-20b-lora"
        )
        assert mb._fetch_peft() is None

    def test_checkpoint_model_returns_lora(self):
        """is_checkpoint=True with lora recipe should return LORA."""
        mb = self._make_model_builder_with_model_package(
            is_checkpoint=True, recipe_name="mtrl-gpt-oss-20b-lora"
        )
        assert mb._fetch_peft() == "LORA"

    def test_checkpoint_none_with_lora_recipe(self):
        """is_checkpoint=None (Unassigned) with lora recipe should return LORA."""
        mb = self._make_model_builder_with_model_package(
            is_checkpoint=None, recipe_name="mtrl-gpt-oss-20b-lora"
        )
        assert mb._fetch_peft() == "LORA"

    def test_merged_model_non_lora_recipe(self):
        """is_checkpoint=False with non-lora recipe should return None."""
        mb = self._make_model_builder_with_model_package(
            is_checkpoint=False, recipe_name="full-finetune-gpt-oss-20b"
        )
        assert mb._fetch_peft() is None

    def test_checkpoint_non_lora_recipe(self):
        """is_checkpoint=True with non-lora recipe should return None."""
        mb = self._make_model_builder_with_model_package(
            is_checkpoint=True, recipe_name="full-finetune-gpt-oss-20b"
        )
        assert mb._fetch_peft() is None


class TestResolveModelArtifactUriMerged:
    """Tests for _resolve_model_artifact_uri with merged models."""

    def _make_model_builder(self, is_checkpoint, s3_uri):
        """Helper to create a ModelBuilder with mocked model package for artifact resolution."""
        from sagemaker.serve.model_builder import ModelBuilder
        from sagemaker.core.resources import ModelPackage

        mock_mp = MagicMock()
        mock_container = MagicMock()
        mock_container.is_checkpoint = is_checkpoint
        mock_container.model_data_source.s3_data_source.s3_uri = s3_uri
        mock_container.base_model = MagicMock()
        mock_container.base_model.recipe_name = "mtrl-gpt-oss-20b-lora"
        mock_mp.inference_specification.containers = [mock_container]

        with patch.object(ModelBuilder, '__post_init__', lambda self: None):
            mb = ModelBuilder.__new__(ModelBuilder)
            real_mp = ModelPackage.__new__(ModelPackage)
            mb.model = real_mp
            mb._fetch_model_package = MagicMock(return_value=mock_mp)
            mb._is_model_customization = MagicMock(return_value=True)
            mb._fetch_peft = MagicMock(return_value=None)
        return mb

    def test_merged_model_resolves_to_hf_merged(self):
        """is_checkpoint=False should resolve to checkpoints/hf_merged/ path."""
        mb = self._make_model_builder(
            is_checkpoint=False,
            s3_uri="s3://bucket/path/model/",
        )
        result = mb._resolve_model_artifact_uri()
        assert result == "s3://bucket/path/model/checkpoints/hf_merged/"

    def test_non_merged_returns_raw_uri(self):
        """is_checkpoint=True should return the raw s3_uri."""
        mb = self._make_model_builder(
            is_checkpoint=True,
            s3_uri="s3://bucket/path/model/",
        )
        result = mb._resolve_model_artifact_uri()
        assert result == "s3://bucket/path/model/"
