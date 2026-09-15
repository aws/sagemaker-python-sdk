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
"""Utility functions for data mixing validation and resolution."""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import uuid
from dataclasses import dataclass
from typing import Dict, Set

import yaml
from botocore.exceptions import ClientError

from sagemaker.train.common_utils.model_aliases import MODEL_NAME_ALIASES
from sagemaker.train.common_utils.recipe_utils import _get_hub_content_metadata
from sagemaker.train.common_utils.finetune_utils import extract_image_from_hyperpod_template
from sagemaker.train.constants import get_sagemaker_hub_name
from sagemaker.core.training.constants import TrainingPlatform
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.s3.utils import resolve_s3_uri_placeholders

logger = logging.getLogger(__name__)

_DATAMIX_NOVA_PREFIX = "nova_"
_DATAMIX_PERCENT_SUFFIX = "_percent"


@dataclass(frozen=True)
class HyperPodTemplateContext:
    """Cached S3 content from the resolve phase for use in the build phase."""

    raw_template: str  # Full Helm chart YAML from S3
    overrides: dict  # Parsed overrides JSON from S3
    recipe_name: str  # Recipe Name from RecipeCollection
    datamix_keyword: str  # "mm_with_datamix" or "text_with_datamix"
    categories: dict[str, float]  # {category_name: default_percent} stripped of prefix/suffix
    image_uri: str | None = None  # Container image URI from template (None if not found)

    def __post_init__(self):
        if not self.raw_template:
            raise ValueError("raw_template must not be empty")
        if not self.recipe_name:
            raise ValueError("recipe_name must not be empty")


def resolve_hyperpod_datamix_context(
    model_name: str,
    is_multimodal: bool,
    sagemaker_session,
    training_type: str = "LORA",
    customization_technique: str = "SFT",
) -> HyperPodTemplateContext:
    """Download HyperPod datamix recipe template and overrides from S3.

    Performs exactly 2 S3 GetObject calls and returns a frozen context
    containing all data needed for the build phase.

    Args:
        model_name: The resolved model hub content name.
        is_multimodal: Whether training data is multimodal.
        sagemaker_session: SageMaker session for API calls.
        training_type: "LORA" or "FULL".
        customization_technique: "SFT" or "CPT".

    Returns:
        HyperPodTemplateContext with cached template, overrides, and categories.

    Raises:
        ValueError: If no matching recipe found (message includes model_name,
            datamix_keyword, customization_technique, training_type).
        ValueError: If HpEksPayloadTemplateS3Uri missing (message includes field
            name + recipe name).
        ValueError: If S3 GetObject fails (message includes original error,
            Forge reference, s3:GetObject).
        ValueError: If HpEksOverrideParamsS3Uri missing (message includes field
            name + recipe name).
        ValueError: If overrides has no nova_*_percent fields.
    """
    hub_name = get_sagemaker_hub_name()
    datamix_keyword = "mm_with_datamix" if is_multimodal else "text_with_datamix"

    hub_metadata = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_name=model_name,
        hub_content_type="Model",
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_session.region_name,
    )

    hub_content_document = hub_metadata.get("hub_content_document", {})
    recipe_collection = hub_content_document.get("RecipeCollection", [])

    datamix_recipe = None
    is_lora = training_type.upper() == "LORA"
    for recipe in recipe_collection:
        recipe_name = recipe.get("Name", "")
        if datamix_keyword not in recipe_name:
            continue
        recipe_technique = recipe.get("CustomizationTechnique", "")
        if recipe_technique and recipe_technique.upper() != customization_technique.upper():
            continue
        has_lora = "lora" in recipe_name.lower()
        if is_lora == has_lora:
            datamix_recipe = recipe
            break

    if datamix_recipe is None:
        raise ValueError(
            f"No HyperPod datamix recipe found for model '{model_name}'. "
            f"No recipe matching datamix_keyword='{datamix_keyword}', "
            f"customization_technique='{customization_technique}', "
            f"training_type='{training_type}' was found in the model's RecipeCollection."
        )

    matched_recipe_name = datamix_recipe.get("Name", "unknown")

    hp_template_s3_uri = datamix_recipe.get("HpEksPayloadTemplateS3Uri")
    if not hp_template_s3_uri:
        raise ValueError(
            f"Recipe '{matched_recipe_name}' is missing required field "
            f"'HpEksPayloadTemplateS3Uri'."
        )

    hp_overrides_s3_uri = datamix_recipe.get("HpEksOverrideParamsS3Uri")
    if not hp_overrides_s3_uri:
        raise ValueError(
            f"Recipe '{matched_recipe_name}' is missing required field "
            f"'HpEksOverrideParamsS3Uri'."
        )

    hp_template_s3_uri = resolve_s3_uri_placeholders(hp_template_s3_uri, sagemaker_session)
    hp_overrides_s3_uri = resolve_s3_uri_placeholders(hp_overrides_s3_uri, sagemaker_session)

    s3_client = sagemaker_session.boto_session.client("s3")

    try:
        template_s3_path = hp_template_s3_uri.replace("s3://", "")
        if template_s3_path.startswith("arn:"):
            arn_parts = template_s3_path.split("/", 2)
            bucket = arn_parts[0] + "/" + arn_parts[1]
            key = arn_parts[2] if len(arn_parts) > 2 else ""
        else:
            bucket, key = template_s3_path.split("/", 1)

        obj = s3_client.get_object(Bucket=bucket, Key=key)
        template_content = obj["Body"].read().decode("utf-8")
    except ClientError as e:
        raise ValueError(
            f"Failed to download from S3 Access Point: {e}"
            f"\nVerify your account has a Forge subscription. "
            f"Refer: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-access.html"
            f"\nAlso ensure your IAM role has the right 's3:GetObject'."
        ) from e

    try:
        overrides_s3_path = hp_overrides_s3_uri.replace("s3://", "")
        if overrides_s3_path.startswith("arn:"):
            arn_parts = overrides_s3_path.split("/", 2)
            o_bucket = arn_parts[0] + "/" + arn_parts[1]
            o_key = arn_parts[2] if len(arn_parts) > 2 else ""
        else:
            o_bucket, o_key = overrides_s3_path.split("/", 1)

        o_obj = s3_client.get_object(Bucket=o_bucket, Key=o_key)
        overrides_template = json.loads(o_obj["Body"].read())
    except ClientError as e:
        raise ValueError(
            f"Failed to download from S3 Access Point: {e}"
            f"\nVerify your account has a Forge subscription. "
            f"Refer: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-access.html"
            f"\nAlso ensure your IAM role has the right 's3:GetObject'."
        ) from e

    categories: Dict[str, float] = {}
    for field_name, field_spec in overrides_template.items():
        if (
            field_name.startswith(_DATAMIX_NOVA_PREFIX)
            and field_name.endswith(_DATAMIX_PERCENT_SUFFIX)
        ):
            category = field_name[len(_DATAMIX_NOVA_PREFIX):-len(_DATAMIX_PERCENT_SUFFIX)]
            if isinstance(field_spec, dict) and "default" in field_spec:
                categories[category] = float(field_spec["default"])
            else:
                categories[category] = float(field_spec) if field_spec is not None else 0.0

    if not categories:
        raise ValueError(
            f"No Nova data category fields found in override parameters for recipe "
            f"'{matched_recipe_name}'. Expected fields matching pattern "
            f"'nova_<category>_percent'."
        )

    image_uri = None
    if "training-config.yaml" in template_content:
        image_uri = extract_image_from_hyperpod_template(template_content)

    logger.info(
        "Resolved HyperPod datamix context for model '%s' "
        "(multimodal=%s, technique=%s, type=%s): %d categories found.",
        model_name,
        is_multimodal,
        customization_technique,
        training_type,
        len(categories),
    )

    return HyperPodTemplateContext(
        raw_template=template_content,
        overrides=overrides_template,
        recipe_name=matched_recipe_name,
        datamix_keyword=datamix_keyword,
        categories=categories,
        image_uri=image_uri,
    )


# Platforms that support data mixing
DATA_MIXING_SUPPORTED_PLATFORMS: Set[str] = {
    TrainingPlatform.SAGEMAKER_HYPERPOD,
    TrainingPlatform.SAGEMAKER_TRAINING_JOB_SERVERLESS,
}


def validate_data_mixing_platform(platform: str) -> None:
    """Raise ValueError if platform does not support data mixing.

    Platform is derived from the compute type passed to the trainer:
    - HyperPodCompute instance → SAGEMAKER_HYPERPOD
    - Compute instance (not HyperPodCompute) → SAGEMAKER_TRAINING_JOB
    - None (no compute specified) → SAGEMAKER_TRAINING_JOB_SERVERLESS

    Data mixing is supported on SAGEMAKER_HYPERPOD and
    SAGEMAKER_TRAINING_JOB_SERVERLESS only.

    Args:
        platform: A TrainingPlatform constant.

    Raises:
        ValueError: If platform does not support data mixing.
    """
    if platform not in DATA_MIXING_SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Data mixing is only supported on HyperPod and Serverless platforms, "
            f"but training targets '{platform}'."
        )


def validate_data_mixing_model(model_name: str) -> None:
    """Raise ValueError if model is not in the Nova model family.

    Data mixing is only supported for Amazon Nova models. Model membership
    is determined by checking if the model name matches any key or value in
    MODEL_NAME_ALIASES (the canonical set of supported Nova models), using
    normalized comparison (lowercase, dots treated as dashes).

    Args:
        model_name: The resolved model name or identifier.

    Raises:
        ValueError: If model is not in the Nova family.
    """
    known_models = set(MODEL_NAME_ALIASES.keys()) | set(MODEL_NAME_ALIASES.values())
    normalized_name = model_name.lower().replace(".", "-")

    if not any(
        known_id.lower().replace(".", "-") in normalized_name
        for known_id in known_models
    ):
        raise ValueError(
            f"Data mixing is only supported for Nova models "
            f"({', '.join(sorted(known_models))}), "
            f"but model is '{model_name}'."
        )


def validate_data_mixing_categories(
    data_mixing_config: "DataMixingConfig",
    recipe_categories: Dict[str, float],
) -> "DataMixingConfig":
    """Validate category keys against recipe and resolve defaults/zeroing.

    Behavior:
    - If nova_data_percentages is None: return a new config with all recipe defaults.
    - If nova_data_percentages is provided: validate keys exist in recipe,
      then set all unspecified recipe categories to 0.

    Args:
        data_mixing_config: The user-provided DataMixingConfig.
        recipe_categories: Mapping of category_name -> default_percent from the recipe.

    Returns:
        A new DataMixingConfig with all recipe categories populated.

    Raises:
        ValueError: If config contains category keys not in the recipe.
    """
    valid_categories = set(recipe_categories.keys())

    if data_mixing_config.nova_data_percentages is None:
        return DataMixingConfig(
            customer_data_percent=data_mixing_config.customer_data_percent,
            nova_data_percentages=dict(recipe_categories),
        )

    user_categories = set(data_mixing_config.nova_data_percentages.keys())
    invalid = user_categories - valid_categories

    if invalid:
        raise ValueError(
            f"Unrecognized data mixing categories: {sorted(invalid)}. "
            f"Valid categories from recipe: {sorted(valid_categories)}"
        )

    merged = {category: 0.0 for category in valid_categories}
    merged.update(data_mixing_config.nova_data_percentages)

    return DataMixingConfig(
        customer_data_percent=data_mixing_config.customer_data_percent,
        nova_data_percentages=merged,
    )


def resolve_datamix_recipe(
    model_name: str,
    is_multimodal: bool,
    sagemaker_session,
) -> Dict[str, float]:
    """Fetch the datamix recipe from SageMaker Hub and return category defaults.

    Looks up the model's hub content, finds a recipe whose Name contains
    "mm_with_datamix" (multimodal) or "text_with_datamix" (text) based on
    the is_multimodal flag, downloads the override parameters from S3,
    and extracts category names and their default percentages.

    Category fields in the override params follow the naming convention
    ``nova_<category>_percent`` (e.g., ``nova_code_percent``). This function
    strips the prefix and suffix to return clean category names
    (e.g., ``"code"``).

    This function is used for the serverless (SMTJ) path. HyperPod uses
    resolve_hyperpod_datamix_context instead.

    Args:
        model_name: The resolved model hub content name.
        is_multimodal: Whether training data is multimodal.
        sagemaker_session: SageMaker session for API calls.

    Returns:
        Dict mapping category names to their default percentages from the recipe.
        For example: {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0}

    Raises:
        ValueError: If no datamix recipe is found for the model, or if the S3
            download fails (e.g., due to missing Forge subscription).
    """
    hub_name = get_sagemaker_hub_name()
    datamix_keyword = "mm_with_datamix" if is_multimodal else "text_with_datamix"

    hub_metadata = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_name=model_name,
        hub_content_type="Model",
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_session.region_name,
    )

    hub_content_document = hub_metadata.get("hub_content_document", {})
    recipe_collection = hub_content_document.get("RecipeCollection", [])

    if not recipe_collection:
        raise ValueError(
            f"No data mixing recipe found for model '{model_name}'. "
            f"The model has no recipes in its RecipeCollection."
        )

    # Find recipe matching the datamix keyword
    datamix_recipe = None
    for recipe in recipe_collection:
        recipe_name = recipe.get("Name", "")
        if datamix_keyword in recipe_name:
            datamix_recipe = recipe
            break

    if datamix_recipe is None:
        raise ValueError(
            f"No data mixing recipe found for model '{model_name}'. "
            f"No recipe with '{datamix_keyword}' in its name was found in the "
            f"model's RecipeCollection."
        )

    override_params_s3_uri = datamix_recipe.get("SmtjOverrideParamsS3Uri")
    if not override_params_s3_uri:
        raise ValueError(
            f"No data mixing recipe found for model '{model_name}'. "
            f"The datamix recipe '{datamix_recipe.get('Name')}' is missing "
            f"'SmtjOverrideParamsS3Uri'."
        )

    override_params_s3_uri = resolve_s3_uri_placeholders(
        override_params_s3_uri, sagemaker_session
    )

    s3_client = sagemaker_session.boto_session.client("s3")
    s3_path = override_params_s3_uri.replace("s3://", "")

    try:
        if s3_path.startswith("arn:"):
            arn_parts = s3_path.split("/", 2)
            bucket = arn_parts[0] + "/" + arn_parts[1]
            key = arn_parts[2] if len(arn_parts) > 2 else ""
        else:
            bucket, key = s3_path.split("/", 1)

        obj = s3_client.get_object(Bucket=bucket, Key=key)
        override_params = json.loads(obj["Body"].read())
    except ClientError as e:
        raise ValueError(
            f"Failed to download from S3 Access Point {e}"
            f"\nVerify your account has a Forge subscription. "
            f"Refer: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-access.html"
            f"\nAlso ensure your IAM role has the right 's3:GetObject'."
        ) from e

    categories: Dict[str, float] = {}
    for field_name, field_spec in override_params.items():
        if (
            field_name.startswith(_DATAMIX_NOVA_PREFIX)
            and field_name.endswith(_DATAMIX_PERCENT_SUFFIX)
        ):
            category = field_name[len(_DATAMIX_NOVA_PREFIX):-len(_DATAMIX_PERCENT_SUFFIX)]
            if isinstance(field_spec, dict) and "default" in field_spec:
                categories[category] = float(field_spec["default"])
            else:
                categories[category] = float(field_spec) if field_spec is not None else 0.0

    if not categories:
        raise ValueError(
            f"No data mixing recipe found for model '{model_name}'. "
            f"The datamix recipe override params contain no Nova data category fields "
            f"(expected fields matching pattern 'nova_<category>_percent')."
        )

    logger.info(
        "Resolved datamix recipe for model '%s' (multimodal=%s): %d categories found.",
        model_name,
        is_multimodal,
        len(categories),
    )

    return categories


def build_hyperpod_datamix_recipe_from_context(
    context: HyperPodTemplateContext,
    validated_config: "DataMixingConfig",
) -> tuple[str, str | None]:
    """Build and write the HyperPod datamix recipe from pre-resolved context.

    No S3 calls are made. Takes the cached template/overrides from the resolve
    phase and a validated DataMixingConfig (output of validate_data_mixing_categories).

    Steps:
    1. Verify hyperpod_cli is installed
    2. Parse raw_template as YAML (extract training-config section)
    3. Resolve Jinja placeholders using the overrides dictionary
    4. Inject customer_data_percent and nova_data_percentages into data_mixing.sources
    5. Validate non-zero categories exist in template's nova_data section
    6. Write final YAML to HyperPod CLI recipes directory
    7. Return (recipe_path, image_uri)

    Args:
        context: The HyperPodTemplateContext from resolve_hyperpod_datamix_context.
        validated_config: A DataMixingConfig validated via validate_data_mixing_categories.

    Returns:
        Tuple of (recipe_path, image_uri). recipe_path is the absolute filesystem
        path (including the .yaml extension) of the generated recipe written under
        the HyperPod CLI recipes_collection/recipes directory; it is consumed by the
        recipe resolver as a user recipe file. image_uri is the container image URI
        from the context (or None).

    Raises:
        RuntimeError: If hyperpod_cli is not installed.
        ValueError: If template cannot be parsed as YAML.
        ValueError: If validated_config contains non-zero categories not in template's
            nova_data section (message includes sorted unsupported and sorted valid names).
    """
    try:
        import hyperpod_cli
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The HyperPod CLI is a required dependency for running HyperPod jobs. "
            "Install it with: pip install hyperpod"
        ) from e

    template_content = context.raw_template
    recipe_yaml_str = template_content

    if "training-config.yaml" in template_content:
        recipe_pattern = (
            r"# Source: .*/training-config\.yaml.*?config\.yaml: \|-\n(.*?)(?=---|\Z)"
        )
        recipe_match = re.search(recipe_pattern, template_content, re.DOTALL)
        if recipe_match:
            recipe_yaml_str = textwrap.dedent(recipe_match.group(1)).strip()
        else:
            raise ValueError(
                "Unable to extract training config from HyperPod recipe template. "
                "Expected 'training-config.yaml' section not found in expected format."
            )

    try:
        recipe_dict = yaml.safe_load(recipe_yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Template content is malformed and cannot be parsed as valid YAML: {e}"
        ) from e

    if recipe_dict is None:
        raise ValueError(
            "Template content is malformed and cannot be parsed as valid YAML: "
            "parsed result is empty."
        )

    # Resolve Jinja placeholders using overrides
    overrides_template = dict(context.overrides)

    if validated_config.nova_data_percentages:
        for category, percent in validated_config.nova_data_percentages.items():
            nova_key = f"{_DATAMIX_NOVA_PREFIX}{category}{_DATAMIX_PERCENT_SUFFIX}"
            overrides_template[category] = {"default": float(percent), "type": "float"}
            if nova_key in overrides_template:
                overrides_template[nova_key]["default"] = float(percent)

    # Add customer_data percent
    overrides_template["percent"] = {
        "default": float(validated_config.customer_data_percent), "type": "float"
    }
    if "customer_data_percent" in overrides_template:
        overrides_template["customer_data_percent"]["default"] = float(
            validated_config.customer_data_percent
        )

    # Add short keys for nova fields not already present (use template defaults)
    for ov_key, ov_val in list(overrides_template.items()):
        if (
            ov_key.startswith(_DATAMIX_NOVA_PREFIX)
            and ov_key.endswith(_DATAMIX_PERCENT_SUFFIX)
        ):
            short_key = ov_key[len(_DATAMIX_NOVA_PREFIX):-len(_DATAMIX_PERCENT_SUFFIX)]
            if short_key not in overrides_template:
                if isinstance(ov_val, dict) and "default" in ov_val:
                    overrides_template[short_key] = {"default": ov_val["default"], "type": "float"}

    def _apply_overrides(recipe: dict, overrides: dict) -> dict:
        for key, value in recipe.items():
            if isinstance(value, dict):
                recipe[key] = _apply_overrides(value, overrides)
            else:
                if key == "name" and value == "distributed_fused_adam":
                    continue
                if isinstance(value, str) and "{{" in value:
                    placeholder_match = re.match(r"^\{\{([\w\-]+)\}\}$", value.strip())
                    if placeholder_match:
                        placeholder_key = placeholder_match.group(1)
                        if placeholder_key in overrides:
                            default_val = overrides[placeholder_key]
                            if isinstance(default_val, dict) and "default" in default_val:
                                recipe[key] = default_val["default"]
                            else:
                                recipe[key] = default_val
                        elif key in overrides:
                            default_val = overrides[key]
                            if isinstance(default_val, dict) and "default" in default_val:
                                recipe[key] = default_val["default"]
                        else:
                            recipe[key] = 0
                elif key in overrides:
                    default_value = overrides[key]
                    if isinstance(default_value, dict) and "default" in default_value:
                        recipe[key] = default_value["default"]
        return recipe

    recipe_dict = _apply_overrides(recipe_dict, overrides_template)

    if "data_mixing" not in recipe_dict:
        recipe_dict["data_mixing"] = {}
    if "sources" not in recipe_dict["data_mixing"]:
        recipe_dict["data_mixing"]["sources"] = {}

    sources = recipe_dict["data_mixing"]["sources"]

    if "customer_data" not in sources:
        sources["customer_data"] = {}
    sources["customer_data"]["percent"] = (
        int(validated_config.customer_data_percent)
        if validated_config.customer_data_percent == int(validated_config.customer_data_percent)
        else validated_config.customer_data_percent
    )

    if "nova_data" not in sources:
        sources["nova_data"] = {}

    if validated_config.nova_data_percentages:
        existing_categories = set(sources["nova_data"].keys())

        specified_categories = {
            k for k, v in validated_config.nova_data_percentages.items() if v > 0
        }
        invalid_for_recipe = specified_categories - existing_categories
        if invalid_for_recipe:
            raise ValueError(
                f"The following data mixing categories are not supported by this model's "
                f"HyperPod recipe: {sorted(invalid_for_recipe)}. "
                f"Valid categories for this recipe: {sorted(existing_categories)}"
            )

        for cat_key in existing_categories:
            sources["nova_data"][cat_key] = 0

        for category, percent in validated_config.nova_data_percentages.items():
            if category in existing_categories:
                value = int(percent) if percent == int(percent) else percent
                sources["nova_data"][category] = value

    recipe_output = yaml.dump(recipe_dict, default_flow_style=False, sort_keys=False, width=120)

    HYPERPOD_RECIPE_PATH = os.path.join(
        "sagemaker_hyperpod_recipes", "recipes_collection", "recipes"
    )
    hp_cli_recipes_dir = os.path.join(
        os.path.dirname(hyperpod_cli.__file__), HYPERPOD_RECIPE_PATH
    )

    recipe_dir = os.path.join(hp_cli_recipes_dir, "fine-tuning", "nova")
    os.makedirs(recipe_dir, exist_ok=True)

    recipe_filename = f"{context.recipe_name}-{str(uuid.uuid4())[:4]}.yaml"
    recipe_path = os.path.join(recipe_dir, recipe_filename)
    with open(recipe_path, "w") as f:
        f.write(recipe_output)

    logger.info(
        "Generated HyperPod datamix recipe at '%s' from context '%s'.",
        recipe_path,
        context.recipe_name,
    )

    return recipe_path, context.image_uri

