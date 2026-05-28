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
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.core.utils.utils import Unassigned


def _make_model_builder(**kwargs):
    """Create a ModelBuilder with mocked session to avoid real AWS calls."""
    with patch("sagemaker.serve.model_builder.Session"):
        with patch(
            "sagemaker.serve.model_builder.get_execution_role",
            return_value=(
                "arn:aws:iam::123456789012:role/SageMakerRole"
            ),
        ):
            from sagemaker.serve.model_builder import ModelBuilder
            defaults = dict(
                model="dummy-model",
                role_arn=(
                    "arn:aws:iam::123456789012:role/SageMakerRole"
                ),
            )
            defaults.update(kwargs)
            mb = ModelBuilder(**defaults)
            # Reset the resolution flag so tests can trigger it
            mb._base_model_fields_resolved = False
            return mb


def _make_base_model(
    hub_content_name=None,
    hub_content_version=None,
    recipe_name=None,
):
    """Create a mock BaseModel with the given fields."""
    base_model = MagicMock()
    base_model.hub_content_name = (
        hub_content_name
        if hub_content_name is not None
        else Unassigned()
    )
    base_model.hub_content_version = (
        hub_content_version
        if hub_content_version is not None
        else Unassigned()
    )
    base_model.recipe_name = (
        recipe_name
        if recipe_name is not None
        else Unassigned()
    )
    return base_model


def _make_model_package(base_model):
    """Create a mock ModelPackage with the given base_model."""
    container = MagicMock()
    container.base_model = base_model
    container.model_data_source = MagicMock()
    container.model_data_source.s3_data_source = MagicMock()
    container.model_data_source.s3_data_source.s3_uri = (
        "s3://bucket/path"
    )

    model_package = MagicMock()
    model_package.inference_specification.containers = [container]
    return model_package


def _make_hub_content(
    hub_content_version="1.0.0",
    hub_content_document=None,
):
    """Create a mock HubContent object."""
    hc = MagicMock()
    hc.hub_content_version = hub_content_version
    if hub_content_document is None:
        hub_content_document = json.dumps({
            "RecipeCollection": [
                {
                    "Name": "auto-resolved-recipe",
                    "HostingConfigs": [],
                }
            ],
            "HostingConfigs": [],
        })
    hc.hub_content_document = hub_content_document
    return hc


class TestResolveBaseModelFields:
    """Tests for _resolve_base_model_fields method."""

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_missing_hub_content_version(
        self, mock_hub_content_cls
    ):
        """hub_content_version Unassigned => resolved from HubContent.get."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        mock_hc = _make_hub_content(hub_content_version="2.5.0")
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.hub_content_version == "2.5.0"
        assert base_model.recipe_name == "some-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_missing_recipe_name(
        self, mock_hub_content_cls
    ):
        """recipe_name Unassigned => resolved from RecipeCollection."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        hub_doc = json.dumps({
            "RecipeCollection": [
                {
                    "Name": "verl-grpo-rlaif-qwen-3-32b-lora",
                    "HostingConfigs": [],
                }
            ],
        })
        mock_hc = _make_hub_content(
            hub_content_version="1.0.0",
            hub_content_document=hub_doc,
        )
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.recipe_name == (
            "verl-grpo-rlaif-qwen-3-32b-lora"
        )

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_noop_when_all_fields_present(
        self, mock_hub_content_cls
    ):
        """All fields present => HubContent.get not called."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()
        assert base_model.hub_content_version == "1.0.0"
        assert base_model.recipe_name == "some-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_both_version_and_recipe(
        self, mock_hub_content_cls
    ):
        """Both Unassigned => both resolved."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        hub_doc = json.dumps({
            "RecipeCollection": [
                {
                    "Name": "auto-resolved-recipe",
                    "HostingConfigs": [],
                }
            ],
        })
        mock_hc = _make_hub_content(
            hub_content_version="3.0.0",
            hub_content_document=hub_doc,
        )
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()

        assert base_model.hub_content_version == "3.0.0"
        assert base_model.recipe_name == "auto-resolved-recipe"

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_fetch_hub_document_works_after_resolution(
        self, mock_hub_content_cls
    ):
        """_fetch_hub_document_for_custom_model works after resolution."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        hub_doc = json.dumps(
            {"HostingConfigs": [{"Profile": "Default"}]}
        )
        mock_hc = _make_hub_content(
            hub_content_version="1.0.0",
            hub_content_document=hub_doc,
        )
        mock_hub_content_cls.get.return_value = mock_hc

        result = mb._fetch_hub_document_for_custom_model()

        assert result == {
            "HostingConfigs": [{"Profile": "Default"}]
        }

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_no_base_model_is_noop(
        self, mock_hub_content_cls
    ):
        """No base_model => method returns without error."""
        mb = _make_model_builder()
        container = MagicMock()
        container.base_model = None
        model_package = MagicMock()
        model_package.inference_specification.containers = [
            container
        ]
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_no_hub_content_name_is_noop(
        self, mock_hub_content_cls
    ):
        """hub_content_name Unassigned => no HubContent.get call."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name=None,
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        mb._resolve_base_model_fields()

        mock_hub_content_cls.get.assert_not_called()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_is_nova_model_with_unassigned_fields(
        self, mock_hub_content_cls
    ):
        """_is_nova_model returns False when fields are Unassigned."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name=None,
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        result = mb._is_nova_model()

        assert result is False

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_fetch_and_cache_recipe_raises_when_unresolvable(
        self, mock_hub_content_cls
    ):
        """recipe_name unresolvable => ValueError from _fetch_and_cache."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version="1.0.0",
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        hub_doc = json.dumps(
            {"RecipeCollection": [], "HostingConfigs": []}
        )
        mock_hc = _make_hub_content(
            hub_content_version="1.0.0",
            hub_content_document=hub_doc,
        )
        mock_hub_content_cls.get.return_value = mock_hc

        with pytest.raises(ValueError, match="recipe_name is missing"):
            mb._fetch_and_cache_recipe_config()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_graceful_on_hub_content_get_failure(
        self, mock_hub_content_cls
    ):
        """When HubContent.get fails, resolution returns early.

        Downstream code (_fetch_and_cache_recipe_config) should still
        raise the appropriate ValueError because recipe_name remains
        Unassigned.
        """
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        # Simulate HubContent.get raising a ClientError
        mock_hub_content_cls.get.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Hub content not found",
                }
            },
            operation_name="DescribeHubContent",
        )

        # _resolve_base_model_fields should not raise
        mb._resolve_base_model_fields()

        # Fields should still be Unassigned
        assert isinstance(
            base_model.hub_content_version, Unassigned
        )
        assert isinstance(base_model.recipe_name, Unassigned)
        # The flag should be set so it doesn't retry
        assert mb._base_model_fields_resolved is True

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_resolve_failure_then_fetch_and_cache_raises(
        self, mock_hub_content_cls
    ):
        """When resolution fails, _fetch_and_cache_recipe_config raises.

        This tests the full flow: resolution fails gracefully, then
        downstream code raises ValueError because recipe_name is still
        Unassigned.
        """
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name=None,
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        # First call (from _resolve_base_model_fields) fails
        # Second call (from _fetch_hub_document_for_custom_model)
        # succeeds but returns empty RecipeCollection
        hub_doc = json.dumps(
            {"RecipeCollection": [], "HostingConfigs": []}
        )
        mock_hc = _make_hub_content(
            hub_content_version="1.0.0",
            hub_content_document=hub_doc,
        )

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ClientError(
                    error_response={
                        "Error": {
                            "Code": "ResourceNotFoundException",
                            "Message": "Not found",
                        }
                    },
                    operation_name="DescribeHubContent",
                )
            return mock_hc

        mock_hub_content_cls.get.side_effect = side_effect

        with pytest.raises(
            ValueError, match="recipe_name is missing"
        ):
            mb._fetch_and_cache_recipe_config()

    @patch("sagemaker.serve.model_builder.HubContent")
    def test_idempotent_second_call_is_noop(
        self, mock_hub_content_cls
    ):
        """Second call to _resolve_base_model_fields is a no-op."""
        mb = _make_model_builder()
        base_model = _make_base_model(
            hub_content_name="huggingface-reasoning-qwen3-32b",
            hub_content_version=None,
            recipe_name="some-recipe",
        )
        model_package = _make_model_package(base_model)
        mb._fetch_model_package = MagicMock(
            return_value=model_package
        )

        mock_hc = _make_hub_content(hub_content_version="2.0.0")
        mock_hub_content_cls.get.return_value = mock_hc

        mb._resolve_base_model_fields()
        assert base_model.hub_content_version == "2.0.0"

        # Reset mock to verify no additional calls
        mock_hub_content_cls.get.reset_mock()

        mb._resolve_base_model_fields()
        mock_hub_content_cls.get.assert_not_called()
