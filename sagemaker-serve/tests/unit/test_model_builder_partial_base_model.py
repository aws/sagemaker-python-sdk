"""Regression tests for partially populated BaseModel metadata in ModelBuilder."""

import json
import unittest
from dataclasses import fields
from unittest.mock import Mock, patch

from sagemaker.core.utils.utils import Unassigned
from sagemaker.serve.model_builder import ModelBuilder


def _model_package(
    *, hub_content_name="test-model", hub_content_version=Unassigned(), recipe_name=Unassigned()
):
    base_model = Mock()
    base_model.hub_content_name = hub_content_name
    base_model.hub_content_version = hub_content_version
    base_model.recipe_name = recipe_name
    container = Mock()
    container.base_model = base_model
    model_package = Mock()
    model_package.inference_specification.containers = [container]
    return model_package


def _builder(*, recipe_name=None):
    builder = ModelBuilder.__new__(ModelBuilder)
    builder.recipe_name = recipe_name
    return builder


def test_fetch_hub_document_omits_absent_hub_content_version():
    """An absent BaseModel version asks HubContent.get for the latest version."""
    for version in (None, "", Unassigned()):
        with unittest.TestCase().subTest(version=version):
            builder = _builder()
            builder._fetch_model_package = Mock(
                return_value=_model_package(hub_content_version=version)
            )

            with patch("sagemaker.serve.model_builder.HubContent.get") as get_hub_content:
                get_hub_content.return_value.hub_content_document = json.dumps(
                    {"HostingConfigs": []}
                )
                builder._fetch_hub_document_for_custom_model()

            get_hub_content.assert_called_once_with(
                hub_content_type="Model",
                hub_name="SageMakerPublicHub",
                hub_content_name="test-model",
            )


def test_fetch_hub_document_rejects_absent_hub_content_name():
    for hub_content_name in (None, "", Unassigned()):
        with unittest.TestCase().subTest(hub_content_name=hub_content_name):
            builder = _builder()
            builder._fetch_model_package = Mock(
                return_value=_model_package(hub_content_name=hub_content_name)
            )

            with patch("sagemaker.serve.model_builder.HubContent.get") as get_hub_content:
                with unittest.TestCase().assertRaisesRegex(
                    ValueError, "does not specify hub_content_name"
                ):
                    builder._fetch_hub_document_for_custom_model()

            get_hub_content.assert_not_called()


def test_fetch_hub_document_preserves_explicit_hub_content_version():
    builder = _builder()
    builder._fetch_model_package = Mock(return_value=_model_package(hub_content_version="2.0.1"))

    with patch("sagemaker.serve.model_builder.HubContent.get") as get_hub_content:
        get_hub_content.return_value.hub_content_document = json.dumps({"HostingConfigs": []})
        builder._fetch_hub_document_for_custom_model()

    get_hub_content.assert_called_once_with(
        hub_content_type="Model",
        hub_name="SageMakerPublicHub",
        hub_content_name="test-model",
        hub_content_version="2.0.1",
    )


def test_recipe_name_preserves_existing_constructor_parameters():
    """The additive override does not shift any existing positional parameter."""
    init_fields = [field.name for field in fields(ModelBuilder) if field.init]
    assert init_fields.index("recipe_name") > init_fields.index("image_config")


def test_recipe_absent_prefers_top_level_hosting_configs():
    builder = _builder()
    top_level = [{"InstanceType": "ml.top.xlarge"}]
    recipe_level = [{"InstanceType": "ml.recipe.xlarge"}]
    hub_document = {
        "HostingConfigs": top_level,
        "RecipeCollection": [{"Name": "recipe-a", "HostingConfigs": recipe_level}],
    }

    assert builder._extract_hosting_configs_from_hub(hub_document, None) == top_level


def test_recipe_absent_infers_unique_recipe_with_hosting_configs():
    builder = _builder()
    expected = [{"InstanceType": "ml.g5.xlarge"}]
    hub_document = {
        "RecipeCollection": [
            {"Name": "without-hosting"},
            {"Name": "deployable-recipe", "HostingConfigs": expected},
        ]
    }
    builder._fetch_hub_document_for_custom_model = Mock(return_value=hub_document)
    builder._fetch_model_package = Mock(return_value=_model_package())

    assert builder._resolve_recipe_hosting_configs() == expected


def test_recipe_name_override_selects_explicit_recipe():
    builder = _builder(recipe_name="recipe-b")
    expected = [{"InstanceType": "ml.p5.48xlarge"}]
    hub_document = {
        "RecipeCollection": [
            {"Name": "recipe-a", "HostingConfigs": [{"InstanceType": "ml.g5.xlarge"}]},
            {"Name": "recipe-b", "HostingConfigs": expected},
        ]
    }
    builder._fetch_hub_document_for_custom_model = Mock(return_value=hub_document)
    builder._fetch_model_package = Mock(return_value=_model_package(recipe_name=None))

    assert builder._resolve_recipe_hosting_configs() == expected


def test_recipe_absent_rejects_multiple_hosting_candidates():
    builder = _builder()
    hub_document = {
        "RecipeCollection": [
            {"Name": "recipe-a", "HostingConfigs": [{"InstanceType": "ml.g5.xlarge"}]},
            {"Name": "recipe-b", "HostingConfigs": [{"InstanceType": "ml.p5.48xlarge"}]},
        ]
    }
    builder._fetch_hub_document_for_custom_model = Mock(return_value=hub_document)
    builder._fetch_model_package = Mock(return_value=_model_package(recipe_name=""))

    with unittest.TestCase().assertRaisesRegex(ValueError, "recipe-a.*recipe-b.*recipe_name"):
        builder._resolve_recipe_hosting_configs()


def test_nova_endpoint_tags_omit_unassigned_recipe_name():
    builder = _builder()
    builder.instance_type = "ml.g5.12xlarge"
    builder.built_model = Mock(model_name="model-name")
    builder._base_model_name = Mock(return_value="nova-textgeneration-lite")
    builder._fetch_model_package = Mock(return_value=_model_package())
    builder._tags = None
    endpoint = Mock()

    with patch("sagemaker.serve.model_builder.EndpointConfig.create"), patch(
        "sagemaker.serve.model_builder.Endpoint.create", return_value=endpoint
    ) as create_endpoint:
        assert builder._deploy_nova_model("endpoint", wait=False) is endpoint

    assert create_endpoint.call_args.kwargs["tags"] == [
        {
            "key": "sagemaker-sdk:jumpstart-model-id",
            "value": "nova-textgeneration-lite",
        }
    ]
