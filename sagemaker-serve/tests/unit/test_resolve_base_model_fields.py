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
"""Tests for _resolve_base_model_fields and related Unassigned handling."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from sagemaker.core.utils.utils import Unassigned


def _make_model_builder(**kwargs):
    """Create a ModelBuilder instance with mocked session to avoid real AWS calls."""
    with patch("sagemaker.serve.model_builder.Session"):
        with patch("sagemaker.serve.model_builder.get_execution_role", return_value="arn:aws:iam::123456789012:role/SageMakerRole"):
            from sagemaker.serve.model_builder import ModelBuilder
            defaults = dict(
                model="dummy-model",
                role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            )
            defaults.update(kwargs)
            mb = ModelBuilder(**defaults)
            # Reset the resolution flag so tests can trigger it
            mb._base_model_fields_resolved = False
            return mb


def _make_base_model(hub_content_name=None, hub_content_version=None, recipe_name=None):
    """Create a mock BaseModel with the given fields."""
    base_model = MagicMock()
    base_model.hub_content_name = hub_content_name if hub_content_name is not None else Unassigned()
    base_model.hub_content_version = hub_content_version if hub_content_version is not None else Unassigned()
    base_model.recipe_name = recipe_name if recipe_name is not None else Unassigned()
    return base_model


def _make_model_package(base_model):
    """Create a mock ModelPackage with the given base_model."""
    container = MagicMock()
    container.base_model = base_model
    container.model_data_source = MagicMock()
    container.model_data_source.s3_data_source = MagicMock()
    container.model_data_source.s3_data_source.s3_uri = "s3://bucket/path"

    model_package = MagicMock()
    model_package.inference_specification.containers = [container]
    return model_package


def _make_hub_content(hub_content_version="1.0.0", hub_content_document=None):
    """Create a mock HubContent object."""
    hc = MagicMock()
    hc.hub_content_version = hub_content_version
    if hub_content_document is None:
        hub_content_document = json.dumps({
            "RecipeCollection": [
                {"Name": "auto-resolved-recipe", "HostingConfigs": []}
            ],
            "HostingConfigs": [],
        })
    hc.hub_content_document = hub_content_document
    return hc


class TestResolveBaseModelFields:
    """Tests for _resolve_base_model_fields method."""

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_missing_hub_content_version(self, mock_hub_content_cls):
        """When hub_content_version is Unassigned, it should be resolved from HubContent.get."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,  # Will be Unassigned
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        mock_hc = _make_hub_content(hub_content_version="2.5.0")
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.hub_content_version == "2.5.0"
        # recipe_name should remain unchanged since it was already set
        assert base_model.recipe_name == "some-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_missing_recipe_name(self, mock_hub_content_cls):
        """When recipe_name is Unassigned, it should be resolved from RecipeCollection."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name=None,  # Will be Unassigned
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        hub_doc = json.dumps({
            "RecipeCollection": [
                {"Name": "verl-grpo-rlaif-qwen-3-32b-lora", "HostingConfigs": []}
            ],
        })
        mock_hc = _make_hub_content(hub_content_version="1.0.0", hub_content_document=hub_doc)
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.recipe_name == "verl-grpo-rlaif-qwen-3-32b-lora"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_noop_when_all_fields_present(self, mock_hub_content_cls):
        """When all fields are present, HubContent.get should not be called."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()
        assert base_model.hub_content_version == "1.0.0"
        assert base_model.recipe_name == "some-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_both_version_and_recipe(self, mock_hub_content_cls):
        """When both hub_content_version and recipe_name are Unassigned, both should be resolved."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        hub_doc = json.dumps({
            "RecipeCollection": [
                {"Name": "auto-resolved-recipe", "HostingConfigs": []}
            ],
        })
        mock_hc = _make_hub_content(hub_content_version="3.0.0", hub_content_document=hub_doc)
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.hub_content_version == "3.0.0"
        assert base_model.recipe_name == "auto-resolved-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_fetch_hub_document_works_after_resolution(self, mock_hub_content_cls):
        """_fetch_hub_document_for_custom_model should work when hub_content_version was Unassigned."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        hub_doc = json.dumps({"HostingConfigs": [{"Profile": "Default"}]})
        mock_hc = _make_hub_content(hub_content_version="1.0.0", hub_content_document=hub_doc)
        mock_hub_content_cls.get.return_value = mock_hc

        result = mb._fetch_hub_document_for_custom_model()

        assert result == {"HostingConfigs": [{"Profile": "Default"}]}

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_no_base_model_is_noop(self, mock_hub_content_cls):
        """When containers[0] has no base_model, method should return without error."""
        mb = _make_model_builder()
        container = MagicMock()
        container.base_model = None
        model_package = MagicMock()
        model_package.inference_specification.containers = [container]
        mb._fetch_model_package = MagicMock(return_value=model_package)

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_no_hub_content_name_is_noop(self, mock_hub_content_cls):
        """When hub_content_name is Unassigned, method should return without calling HubContent.get."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name=None,  # Will be Unassigned
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_is_nova_model_with_unassigned_fields_does_not_crash(self, mock_hub_content_cls):
        """_is_nova_model should return False without raising when fields are Unassigned."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name=None,  # Unassigned
            hub_content_version=None,
            recipe_name=None,  # Unassigned
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        result = mb._is_nova_model()

        assert result is False

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_fetch_and_cache_recipe_config_raises_when_recipe_unresolvable(self, mock_hub_content_cls):
        """When recipe_name cannot be resolved, _fetch_and_cache_recipe_config should raise ValueError."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name=None,  # Unassigned
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(return_value=model_package)

        # Hub document with empty RecipeCollection - recipe cannot be resolved
        hub_doc = json.dumps({"RecipeCollection": [], "HostingConfigs": []})
        mock_hc = _make_hub_content(hub_content_version="1.0.0", hub_content_document=hub_doc)
        mock_hub_content_cls.get.return_value = mock_hc

        with pytest.raises(ValueError, match="recipe_name is missing"):
            mb._fetch_and_cache_recipe_config()
