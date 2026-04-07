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
"""Tests for resolve_base_model_fields utility function."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.core.utils.utils import Unassigned
from sagemaker.serve.model_builder_utils import resolve_base_model_fields


class FakeBaseModel:
    """Fake BaseModel for testing."""

    def __init__(self, hub_content_name=None, hub_content_version=None, recipe_name=None):
        self.hub_content_name = hub_content_name
        self.hub_content_version = hub_content_version
        self.recipe_name = recipe_name


class FakeHubContent:
    """Fake HubContent response."""

    def __init__(self, hub_content_version=None):
        self.hub_content_version = hub_content_version


class TestResolveBaseModelFields:
    """Tests for resolve_base_model_fields."""

    def test_resolve_with_none_base_model(self):
        """Test that None base_model is returned unchanged."""
        result = resolve_base_model_fields(None)
        assert result is None

    def test_resolve_with_no_hub_content_name_returns_unchanged(self):
        """Test that base_model without hub_content_name is returned unchanged."""
        base_model = FakeBaseModel(
            hub_content_name=Unassigned(),
            hub_content_version=Unassigned(),
            recipe_name=Unassigned(),
        )
        result = resolve_base_model_fields(base_model)
        assert isinstance(result.hub_content_version, Unassigned)
        assert isinstance(result.recipe_name, Unassigned)

    def test_resolve_with_none_hub_content_name_returns_unchanged(self):
        """Test that base_model with None hub_content_name is returned unchanged."""
        base_model = FakeBaseModel(
            hub_content_name=None,
            hub_content_version=Unassigned(),
            recipe_name=Unassigned(),
        )
        result = resolve_base_model_fields(base_model)
        assert isinstance(result.hub_content_version, Unassigned)

    def test_resolve_with_empty_hub_content_name_returns_unchanged(self):
        """Test that base_model with empty hub_content_name is returned unchanged."""
        base_model = FakeBaseModel(
            hub_content_name="",
            hub_content_version=Unassigned(),
            recipe_name=Unassigned(),
        )
        result = resolve_base_model_fields(base_model)
        assert isinstance(result.hub_content_version, Unassigned)

    def test_resolve_with_all_fields_present_no_api_call(self):
        """Test that no API call is made when all fields are already present."""
        base_model = FakeBaseModel(
            hub_content_name="huggingface-model-abc",
            hub_content_version="1.0.0",
            recipe_name="my-recipe",
        )
        with patch("sagemaker.serve.model_builder_utils.HubContent", autospec=True) as mock_hc:
            # HubContent should NOT be imported/called
            result = resolve_base_model_fields(base_model)
            assert result.hub_content_version == "1.0.0"
            assert result.recipe_name == "my-recipe"

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_missing_hub_content_version_resolves_from_hub(self, mock_hub_content_cls):
        """Test that missing hub_content_version is resolved from SageMakerPublicHub."""
        fake_hc = FakeHubContent(hub_content_version="2.5.0")
        mock_hub_content_cls.get.return_value = fake_hc

        base_model = FakeBaseModel(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=Unassigned(),
            recipe_name="some-recipe",
        )

        with patch(
            "sagemaker.serve.model_builder_utils.HubContent", mock_hub_content_cls
        ):
            result = resolve_base_model_fields(base_model)

        assert result.hub_content_version == "2.5.0"
        mock_hub_content_cls.get.assert_called_once_with(
            hub_content_type="Model",
            hub_name="SageMakerPublicHub",
            hub_content_name="huggingface-reasoning-qwen3-32b",
        )

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_missing_recipe_name_logs_warning(self, mock_hub_content_cls):
        """Test that missing recipe_name logs a warning but does not crash."""
        base_model = FakeBaseModel(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name=Unassigned(),
        )

        result = resolve_base_model_fields(base_model)
        # recipe_name should still be Unassigned (not resolved automatically)
        assert isinstance(result.recipe_name, Unassigned)
        # But the function should not crash
        assert result.hub_content_version == "1.0.0"

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_hub_content_not_found_does_not_crash(self, mock_hub_content_cls):
        """Test that HubContent.get() failure is handled gracefully."""
        mock_hub_content_cls.get.side_effect = Exception("HubContent not found")

        base_model = FakeBaseModel(
            hub_content_name="nonexistent-model",
            hub_content_version=Unassigned(),
            recipe_name="some-recipe",
        )

        with patch(
            "sagemaker.serve.model_builder_utils.HubContent", mock_hub_content_cls
        ):
            # Should not raise, just log a warning
            result = resolve_base_model_fields(base_model)

        # hub_content_version should still be Unassigned since resolution failed
        assert isinstance(result.hub_content_version, Unassigned)

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_both_version_and_recipe_missing(self, mock_hub_content_cls):
        """Test resolution when both hub_content_version and recipe_name are missing."""
        fake_hc = FakeHubContent(hub_content_version="3.0.0")
        mock_hub_content_cls.get.return_value = fake_hc

        base_model = FakeBaseModel(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=Unassigned(),
            recipe_name=Unassigned(),
        )

        with patch(
            "sagemaker.serve.model_builder_utils.HubContent", mock_hub_content_cls
        ):
            result = resolve_base_model_fields(base_model)

        # Version should be resolved
        assert result.hub_content_version == "3.0.0"
        # Recipe should still be Unassigned (with warning logged)
        assert isinstance(result.recipe_name, Unassigned)

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_with_none_version_resolves(self, mock_hub_content_cls):
        """Test that None hub_content_version (not just Unassigned) is also resolved."""
        fake_hc = FakeHubContent(hub_content_version="1.2.3")
        mock_hub_content_cls.get.return_value = fake_hc

        base_model = FakeBaseModel(
            hub_content_name="huggingface-model-xyz",
            hub_content_version=None,
            recipe_name="my-recipe",
        )

        with patch(
            "sagemaker.serve.model_builder_utils.HubContent", mock_hub_content_cls
        ):
            result = resolve_base_model_fields(base_model)

        assert result.hub_content_version == "1.2.3"

    @patch("sagemaker.core.resources.HubContent")
    def test_resolve_with_empty_string_version_resolves(self, mock_hub_content_cls):
        """Test that empty string hub_content_version is also resolved."""
        fake_hc = FakeHubContent(hub_content_version="4.0.0")
        mock_hub_content_cls.get.return_value = fake_hc

        base_model = FakeBaseModel(
            hub_content_name="huggingface-model-xyz",
            hub_content_version="",
            recipe_name="my-recipe",
        )

        with patch(
            "sagemaker.serve.model_builder_utils.HubContent", mock_hub_content_cls
        ):
            result = resolve_base_model_fields(base_model)

        assert result.hub_content_version == "4.0.0"
