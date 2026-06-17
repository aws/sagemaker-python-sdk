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

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
import yaml

from sagemaker.train.common_utils.data_mixing_utils import (
    HyperPodTemplateContext,
    build_hyperpod_datamix_recipe_from_context,
    resolve_hyperpod_datamix_context,
    validate_data_mixing_categories,
    _DATAMIX_NOVA_PREFIX,
    _DATAMIX_PERCENT_SUFFIX,
)
from sagemaker.train.data_mixing_config import DataMixingConfig


@pytest.mark.parametrize(
    "categories",
    [
        {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0},
        {"single": 100.0},
        {"a": 0.0, "b": 50.5, "c": 49.5},
        {"hyphenated-name": 33.33, "another-one": 66.67},
    ],
    ids=[
        "four_standard_categories",
        "single_category_100_percent",
        "three_categories_with_zero",
        "hyphenated_category_names",
    ],
)
def test_category_extraction_round_trip(categories):
    overrides = {}
    for cat_name, value in categories.items():
        field_key = f"{_DATAMIX_NOVA_PREFIX}{cat_name}{_DATAMIX_PERCENT_SUFFIX}"
        overrides[field_key] = {"default": value, "type": "float"}

    # Non-matching field should be ignored
    overrides["other_param"] = {"default": "ignored"}

    session = MagicMock()
    session.boto_session.region_name = "us-west-2"

    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

    s3_client = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return sts_client
        elif service_name == "s3":
            return s3_client
        return MagicMock()

    session.boto_session.client.side_effect = client_factory

    template_content = "simple template content"
    template_body = MagicMock()
    template_body.read.return_value = template_content.encode("utf-8")

    overrides_body = MagicMock()
    overrides_body.read.return_value = json.dumps(overrides).encode("utf-8")

    s3_client.get_object.side_effect = [
        {"Body": template_body},
        {"Body": overrides_body},
    ]

    hub_metadata = {
        "hub_content_document": {
            "RecipeCollection": [
                {
                    "Name": "nova_pro_sft_lora_text_with_datamix",
                    "CustomizationTechnique": "SFT",
                    "HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml",
                    "HpEksOverrideParamsS3Uri": "s3://bucket/overrides.json",
                }
            ]
        }
    }

    with patch(
        "sagemaker.train.common_utils.data_mixing_utils._get_hub_content_metadata",
        return_value=hub_metadata,
    ), patch(
        "sagemaker.train.common_utils.data_mixing_utils.get_sagemaker_hub_name",
        return_value="SageMakerPublicHub",
    ):
        ctx = resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

    assert ctx.categories == categories


@pytest.mark.parametrize(
    "recipe_name,uri_field",
    [
        ("nova_pro_sft_lora_datamix", "HpEksPayloadTemplateS3Uri"),
        ("nova_pro_sft_lora_datamix", "HpEksOverrideParamsS3Uri"),
        ("custom_lora_recipe", "HpEksPayloadTemplateS3Uri"),
    ],
    ids=[
        "missing_template_uri_standard_recipe",
        "missing_overrides_uri_standard_recipe",
        "missing_template_uri_custom_recipe",
    ],
)
def test_missing_uri_error_contains_field_and_recipe(recipe_name, uri_field):
    recipe = {
        "Name": recipe_name,
        "CustomizationTechnique": "SFT",
    }
    # Add the other URI field so only the target one is missing
    if uri_field == "HpEksPayloadTemplateS3Uri":
        recipe["HpEksOverrideParamsS3Uri"] = "s3://bucket/overrides.json"
    else:
        recipe["HpEksPayloadTemplateS3Uri"] = "s3://bucket/template.yaml"

    # Make recipe name contain the datamix keyword for matching
    recipe["Name"] = f"{recipe_name}_text_with_datamix"

    hub_metadata = {
        "hub_content_document": {
            "RecipeCollection": [recipe]
        }
    }

    session = MagicMock()
    session.boto_session.region_name = "us-west-2"

    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

    s3_client = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return sts_client
        elif service_name == "s3":
            return s3_client
        return MagicMock()

    session.boto_session.client.side_effect = client_factory

    # Template download must succeed before the missing overrides URI is checked
    if uri_field == "HpEksOverrideParamsS3Uri":
        template_body = MagicMock()
        template_body.read.return_value = b"template content"
        s3_client.get_object.return_value = {"Body": template_body}

    with patch(
        "sagemaker.train.common_utils.data_mixing_utils._get_hub_content_metadata",
        return_value=hub_metadata,
    ), patch(
        "sagemaker.train.common_utils.data_mixing_utils.get_sagemaker_hub_name",
        return_value="SageMakerPublicHub",
    ):
        with pytest.raises(ValueError) as exc_info:
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

    error_message = str(exc_info.value)
    assert uri_field in error_message, (
        f"Error message should contain '{uri_field}', got: {error_message}"
    )
    full_recipe_name = f"{recipe_name}_text_with_datamix"
    assert full_recipe_name in error_message, (
        f"Error message should contain recipe name '{full_recipe_name}', got: {error_message}"
    )


@pytest.mark.parametrize(
    "model_name,keyword,technique,training_type",
    [
        ("nova-pro", "text_with_datamix", "SFT", "LORA"),
        ("nova-lite", "mm_with_datamix", "CPT", "FULL"),
    ],
    ids=[
        "text_sft_lora_no_match",
        "multimodal_cpt_full_no_match",
    ],
)
def test_unmatched_recipe_error_contains_all_identifiers(
    model_name, keyword, technique, training_type
):
    hub_metadata = {
        "hub_content_document": {
            "RecipeCollection": []
        }
    }

    is_multimodal = keyword == "mm_with_datamix"

    session = MagicMock()
    session.boto_session.region_name = "us-west-2"

    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return sts_client
        return MagicMock()

    session.boto_session.client.side_effect = client_factory

    with patch(
        "sagemaker.train.common_utils.data_mixing_utils._get_hub_content_metadata",
        return_value=hub_metadata,
    ), patch(
        "sagemaker.train.common_utils.data_mixing_utils.get_sagemaker_hub_name",
        return_value="SageMakerPublicHub",
    ):
        with pytest.raises(ValueError) as exc_info:
            resolve_hyperpod_datamix_context(
                model_name=model_name,
                is_multimodal=is_multimodal,
                sagemaker_session=session,
                training_type=training_type,
                customization_technique=technique,
            )

    error_message = str(exc_info.value)
    assert model_name in error_message, (
        f"Error message should contain model_name '{model_name}', got: {error_message}"
    )
    assert keyword in error_message, (
        f"Error message should contain keyword '{keyword}', got: {error_message}"
    )
    assert technique in error_message, (
        f"Error message should contain technique '{technique}', got: {error_message}"
    )
    assert training_type in error_message, (
        f"Error message should contain training_type '{training_type}', got: {error_message}"
    )


@pytest.mark.parametrize(
    "customer_percent,nova_percents",
    [
        (70.0, {"code": 50.0, "math": 50.0}),
        (30.0, {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0}),
        (100.0, {"code": 0.0, "math": 0.0}),
    ],
    ids=[
        "two_categories_70_customer",
        "four_categories_30_customer",
        "all_customer_zero_nova",
    ],
)
def test_data_mixing_values_injected_faithfully(customer_percent, nova_percents):
    all_cats = list(nova_percents.keys())
    nova_data_yaml_lines = "\n".join(
        f"          {cat}: '{{{{{cat}}}}}'" for cat in all_cats
    )

    helm_template = (
        "---\n"
        "# Source: nova-training/templates/training-config.yaml\n"
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: training-config\n"
        "data:\n"
        "  config.yaml: |-\n"
        "    model:\n"
        "      name: nova-pro\n"
        "    data_mixing:\n"
        "      sources:\n"
        "        customer_data:\n"
        "          percent: '{{percent}}'\n"
        "        nova_data:\n"
        f"{nova_data_yaml_lines}\n"
        "---\n"
    )

    # Build overrides
    overrides = {
        "percent": {"default": 50.0, "type": "float"},
    }
    for cat in all_cats:
        nova_key = f"{_DATAMIX_NOVA_PREFIX}{cat}{_DATAMIX_PERCENT_SUFFIX}"
        overrides[nova_key] = {"default": 25.0, "type": "float"}

    context = HyperPodTemplateContext(
        raw_template=helm_template,
        overrides=overrides,
        recipe_name="nova_pro_sft_lora_text_with_datamix",
        datamix_keyword="text_with_datamix",
        categories={cat: 25.0 for cat in all_cats},
        image_uri=None,
    )

    validated_config = DataMixingConfig(
        customer_data_percent=customer_percent,
        nova_data_percentages=nova_percents,
    )

    written_content = {}

    def capture_write(path, mode="r", **kwargs):
        if mode == "w":
            sio = StringIO()
            sio.name = path
            written_content["buffer"] = sio
            sio.close = lambda: None
            return sio
        raise FileNotFoundError(f"Not mocking reads: {path}")

    mock_hyperpod_cli = MagicMock()
    mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

    with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
        with patch("os.makedirs"):
            with patch("builtins.open", side_effect=capture_write):
                build_hyperpod_datamix_recipe_from_context(context, validated_config)

    assert "buffer" in written_content
    written_yaml = written_content["buffer"].getvalue()
    parsed = yaml.safe_load(written_yaml)

    sources = parsed["data_mixing"]["sources"]

    expected_customer = (
        int(customer_percent) if customer_percent == int(customer_percent) else customer_percent
    )
    assert sources["customer_data"]["percent"] == expected_customer

    for cat, expected_val in nova_percents.items():
        expected = int(expected_val) if expected_val == int(expected_val) else expected_val
        assert sources["nova_data"][cat] == expected, (
            f"nova_data['{cat}'] should be {expected}, got {sources['nova_data'][cat]}"
        )


@pytest.mark.parametrize(
    "unsupported_cats,template_cats",
    [
        ({"nonexistent"}, {"code", "math"}),
        ({"bad1", "bad2"}, {"code", "math", "en-entertainment"}),
    ],
    ids=[
        "one_unsupported_category",
        "two_unsupported_categories",
    ],
)
def test_unsupported_categories_named_in_error(unsupported_cats, template_cats):
    nova_data_yaml_lines = "\n".join(
        f"          {cat}: '{{{{cat}}}}'" for cat in sorted(template_cats)
    )

    helm_template = (
        "---\n"
        "# Source: nova-training/templates/training-config.yaml\n"
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: training-config\n"
        "data:\n"
        "  config.yaml: |-\n"
        "    model:\n"
        "      name: nova-pro\n"
        "    data_mixing:\n"
        "      sources:\n"
        "        customer_data:\n"
        "          percent: '{{percent}}'\n"
        "        nova_data:\n"
        f"{nova_data_yaml_lines}\n"
        "---\n"
    )

    # Build overrides
    overrides = {"percent": {"default": 50.0, "type": "float"}}
    for cat in template_cats:
        nova_key = f"{_DATAMIX_NOVA_PREFIX}{cat}{_DATAMIX_PERCENT_SUFFIX}"
        overrides[nova_key] = {"default": 25.0, "type": "float"}

    context = HyperPodTemplateContext(
        raw_template=helm_template,
        overrides=overrides,
        recipe_name="nova_pro_sft_lora_text_with_datamix",
        datamix_keyword="text_with_datamix",
        categories={cat: 25.0 for cat in template_cats},
        image_uri=None,
    )

    # Config with non-zero values for unsupported categories
    nova_data_percentages = {}
    for cat in template_cats:
        nova_data_percentages[cat] = 0.0
    for cat in unsupported_cats:
        nova_data_percentages[cat] = 50.0 / len(unsupported_cats)

    validated_config = DataMixingConfig(
        customer_data_percent=100.0,
        nova_data_percentages=nova_data_percentages,
    )

    mock_hyperpod_cli = MagicMock()
    mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

    with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
        with patch("os.makedirs"):
            with patch("builtins.open", MagicMock()):
                with pytest.raises(ValueError) as exc_info:
                    build_hyperpod_datamix_recipe_from_context(context, validated_config)

    error_message = str(exc_info.value)

    for cat in sorted(unsupported_cats):
        assert cat in error_message, (
            f"Error message should contain unsupported category '{cat}', got: {error_message}"
        )

    for cat in sorted(template_cats):
        assert cat in error_message, (
            f"Error message should contain valid category '{cat}', got: {error_message}"
        )


@pytest.mark.parametrize(
    "recipe_categories",
    [
        {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0},
        {"single_cat": 100.0},
    ],
    ids=[
        "four_standard_categories",
        "single_category_100_percent",
    ],
)
def test_defaults_populated_when_nova_percentages_none(recipe_categories):
    config = DataMixingConfig(
        customer_data_percent=50.0,
        nova_data_percentages=None,
    )

    result = validate_data_mixing_categories(config, recipe_categories)

    assert result.nova_data_percentages == recipe_categories
    assert result.customer_data_percent == 50.0


@pytest.mark.parametrize(
    "is_multimodal,expected_keyword,recipe_name",
    [
        (False, "text_with_datamix", "nova_pro_sft_lora_text_with_datamix"),
        (True, "mm_with_datamix", "nova_pro_sft_lora_mm_with_datamix"),
    ],
    ids=[
        "text_only_selects_text_with_datamix",
        "multimodal_selects_mm_with_datamix",
    ],
)
def test_datamix_keyword_selection(is_multimodal, expected_keyword, recipe_name):
    """is_multimodal flag selects the correct datamix keyword for recipe matching."""
    overrides = {
        "nova_code_percent": {"default": 50.0, "type": "float"},
        "nova_math_percent": {"default": 50.0, "type": "float"},
    }

    session = MagicMock()
    session.boto_session.region_name = "us-west-2"

    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

    s3_client = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return sts_client
        elif service_name == "s3":
            return s3_client
        return MagicMock()

    session.boto_session.client.side_effect = client_factory

    template_body = MagicMock()
    template_body.read.return_value = b"template content"

    overrides_body = MagicMock()
    overrides_body.read.return_value = json.dumps(overrides).encode("utf-8")

    s3_client.get_object.side_effect = [
        {"Body": template_body},
        {"Body": overrides_body},
    ]

    # Both recipes present — only the one matching expected_keyword should be selected
    hub_metadata = {
        "hub_content_document": {
            "RecipeCollection": [
                {
                    "Name": "nova_pro_sft_lora_text_with_datamix",
                    "CustomizationTechnique": "SFT",
                    "HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml",
                    "HpEksOverrideParamsS3Uri": "s3://bucket/overrides.json",
                },
                {
                    "Name": "nova_pro_sft_lora_mm_with_datamix",
                    "CustomizationTechnique": "SFT",
                    "HpEksPayloadTemplateS3Uri": "s3://bucket/mm_template.yaml",
                    "HpEksOverrideParamsS3Uri": "s3://bucket/mm_overrides.json",
                },
            ]
        }
    }

    with patch(
        "sagemaker.train.common_utils.data_mixing_utils._get_hub_content_metadata",
        return_value=hub_metadata,
    ), patch(
        "sagemaker.train.common_utils.data_mixing_utils.get_sagemaker_hub_name",
        return_value="SageMakerPublicHub",
    ):
        ctx = resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=is_multimodal,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

    assert ctx.recipe_name == recipe_name
    assert ctx.datamix_keyword == expected_keyword
    assert expected_keyword in ctx.recipe_name
