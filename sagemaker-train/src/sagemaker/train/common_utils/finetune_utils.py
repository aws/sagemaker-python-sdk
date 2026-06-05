"""Common utilities for fine-tuning trainers."""

import os
import re
import time
import logging
import json
from typing import Any, Dict, Optional, Union
import time
import boto3
from sagemaker.core.resources import ModelPackage, ModelPackageGroup
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.utils import resolve_s3_uri_placeholders
from sagemaker.train.common_utils.recipe_utils import _get_hub_content_metadata
# Single source of truth for Lambda-ARN detection, shared with the reward verifier
# so both code paths agree on what counts as a Lambda ARN.
from sagemaker.train.common_utils.rlvr_reward_verifier import LAMBDA_ARN_REGEX
from sagemaker.train.common import TrainingType, CustomizationTechnique, JOB_TYPE, FineTuningOptions
from sagemaker.core.shapes import ServerlessJobConfig, Channel, DataSource, ModelPackageConfig, MlflowConfig
from sagemaker.core.training.configs import HyperPodCompute, TrainingJobCompute
from sagemaker.train.configs import InputData, OutputDataConfig
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.constants import get_sagemaker_hub_name

logger = logging.getLogger(__name__)

# Region mappings for model availability
OPEN_WEIGHTS_REGIONS = ['us-east-1', 'us-west-2', 'ap-northeast-1', 'eu-west-1']  # IAD, PDX, NRT, DUB
NOVA_REGIONS = ['us-east-1', 'us-west-2']  # IAD, PDX
# Constants
DEFAULT_REGION = "us-west-2"


from sagemaker.train.common_utils.model_aliases import normalize_model_name as _normalize_model_name


def _select_recipe_by_training_type(recipes: list, training_type, with_fallback: bool = True):
    """Select a recipe from a list based on training type (LORA/FULL).

    Selects non-subscription recipes first, then falls back to subscription
    recipes if with_fallback is True.

    Args:
        recipes: Filtered list of recipe entries from RecipeCollection.
        training_type: TrainingType enum or string ("LORA", "FULL").
        with_fallback: If True, falls back to subscription recipes when
            no non-subscription recipe matches.

    Returns:
        The matching recipe dict, or None if no match found.
    """
    is_lora = (isinstance(training_type, TrainingType) and training_type == TrainingType.LORA) or training_type == "LORA"
    is_full = (isinstance(training_type, TrainingType) and training_type == TrainingType.FULL) or training_type == "FULL"

    if is_lora:
        recipe = next((r for r in recipes if r.get("Peft") and not r.get("IsSubscriptionModel")), None)
        if not recipe and with_fallback:
            recipe = next((r for r in recipes if r.get("Peft")), None)
    elif is_full:
        recipe = next((r for r in recipes if not r.get("Peft") and not r.get("IsSubscriptionModel")), None)
        if not recipe and with_fallback:
            recipe = next((r for r in recipes if not r.get("Peft")), None)
    else:
        recipe = None

    return recipe


def validate_eval_platform_compatibility(model_path: Optional[str], compute) -> None:
    """Validate that the model checkpoint platform matches the evaluation compute.

    SMHP-trained checkpoints can only be evaluated on HyperPod.
    SMTJ-trained checkpoints can only be evaluated on SMTJ.
    If the checkpoint platform cannot be determined, a warning is logged.

    Args:
        model_path: S3 path to the model checkpoint, or None for base model eval.
        compute: The compute configuration (Compute, HyperPodCompute, or None for serverless).

    Raises:
        ValueError: If the checkpoint platform doesn't match the evaluation platform.
    """
    if model_path is None:
        return  # Base model evaluation — no platform restriction

    from sagemaker.core.training.configs import HyperPodCompute as _HyperPodCompute

    # Detect checkpoint platform from S3 path
    if "-hp-" in model_path:
        checkpoint_platform = "hyperpod"
    elif "-smtj-" in model_path:
        checkpoint_platform = "smtj"
    else:
        logger.warning(
            f"Cannot determine training platform from model path: {model_path}. "
            f"If the checkpoint was trained on a different platform, evaluation may fail."
        )
        return

    # Determine execution platform
    if isinstance(compute, _HyperPodCompute):
        exec_platform = "hyperpod"
    else:
        exec_platform = "smtj"

    if checkpoint_platform != exec_platform:
        raise ValueError(
            f"Platform mismatch: model checkpoint was trained on {checkpoint_platform} "
            f"but evaluation is configured to run on {exec_platform}. "
            f"Checkpoints must be evaluated on the same platform they were trained on."
        )


def _validate_model_region_availability(model_name: str, region_name: str):
    """Validate if the model is available in the specified region."""
    if "nova" in model_name.lower():
        if region_name not in NOVA_REGIONS:
            raise ValueError(
                f"""
Region '{region_name}' does not support model customization.
Currently supported regions for this feature are: {', '.join(NOVA_REGIONS)}
Please choose one of the supported regions or check our documentation for updates.
            """
            )
    else:
        # Open weights models
        if region_name not in OPEN_WEIGHTS_REGIONS:
            raise ValueError(
                f"""
Region '{region_name}' does not support model customization.
Currently supported regions for this feature are: {', '.join(OPEN_WEIGHTS_REGIONS)}
Please choose one of the supported regions or check our documentation for updates.
            """
            )

def _get_beta_session():
    """Create a SageMaker session with beta endpoint for demo purposes."""
    sm_client = boto3.client('sagemaker', region_name=DEFAULT_REGION)
    return Session(sagemaker_client=sm_client)


def _read_domain_id_from_metadata() -> Optional[str]:
    """Read domain ID from Studio metadata file.
    
    This is the standard location for domain information in Studio with Spaces.
    Returns None if not running in Studio or if metadata file doesn't exist.
    """
    try:
        metadata_path = '/opt/ml/metadata/resource-metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('DomainId')
    except Exception as e:
        logger.debug(f"Could not read Studio metadata file: {e}")
    return None


def _get_current_domain_id(sagemaker_session) -> Optional[str]:
    """Get current SageMaker Studio domain ID.
    
    Checks multiple sources in order of reliability:
    1. Studio metadata file (Studio with Spaces - newer architecture)
    2. User profile ARN (Studio Classic with User Profiles - legacy)
    
    Returns None if not running in a Studio environment with domain.
    """
    # Try metadata file first (Studio with Spaces)
    domain_id = _read_domain_id_from_metadata()
    if domain_id:
        return domain_id
    
    # Fallback to original logic (Studio Classic with User Profiles)
    try:
        user_profile_arn = sagemaker_session.get_caller_identity_arn()
        if user_profile_arn and 'user-profile' in user_profile_arn:
            # ARN format: arn:aws:sagemaker:region:account:user-profile/domain-id/profile-name
            return user_profile_arn.split('/')[1]
    except Exception as e:
        logger.debug(f"Could not extract domain ID from user profile ARN: {e}")
    
    return None


def _get_prod_sm_client(sagemaker_session) -> "boto3.client":
    """Create a boto3 SageMaker client that always hits prod (no custom endpoint_url)."""
    region = sagemaker_session.boto_session.region_name
    return boto3.client("sagemaker", region_name=region)


def _resolve_mlflow_resource_arn(sagemaker_session, mlflow_resource_arn: Optional[str] = None, min_mlflow_version: Optional[str] = None) -> Optional[str]:
    """Resolve MLflow resource ARN using default experience logic.

    All MLflow API calls use a raw boto3 client against prod (no custom endpoint),
    since MLflow apps are a prod-only resource.

    Args:
        sagemaker_session: SageMaker session.
        mlflow_resource_arn: Explicit ARN to use (returned as-is if provided).
        min_mlflow_version: Minimum required MLflow version (e.g. "3.10").
            If the resolved app's version is below this, a new app is created.
    """
    if mlflow_resource_arn:
        return mlflow_resource_arn

    try:
        sm_client = _get_prod_sm_client(sagemaker_session)
        mlflow_apps_list = []
        paginator = sm_client.get_paginator("list_mlflow_apps")
        for page in paginator.paginate():
            mlflow_apps_list.extend(page.get("Summaries", []))

        logger.info("Found %d MLflow apps: %s", len(mlflow_apps_list),
                    [(a.get("Name", "?"), a.get("Status", "?"), a.get("MlflowVersion", "?")) for a in mlflow_apps_list])
        current_domain_id = _get_current_domain_id(sagemaker_session)

        # Check for domain match
        resolved_app = None
        if current_domain_id:
            resolved_app = next((app for app in mlflow_apps_list
                               if current_domain_id in app.get("DefaultDomainIdList", [])), None)

        # Check for account default
        if not resolved_app:
            resolved_app = next((app for app in mlflow_apps_list
                              if app.get("AccountDefaultStatus") == "ENABLED"), None)

        # Use first available with ready status
        if not resolved_app and mlflow_apps_list:
            resolved_app = next((app for app in mlflow_apps_list
                            if app.get("Status") in ["Created", "Updated"]), None)

        # Check resolved app status
        if resolved_app:
            resolved_arn = resolved_app["Arn"]
            logger.info("Resolved MLflow app: %s (status: %s, version: %s)",
                        resolved_arn, resolved_app.get("Status"),
                        resolved_app.get("MlflowVersion", "unknown"))
            if resolved_app.get("Status") in ["Failed", "CreateFailed", "DeleteFailed", "Stopped"]:
                logger.warning("Resolved MLflow app %s is in failed state: %s. Skipping.",
                               resolved_arn, resolved_app.get("Status"))
                resolved_app = None

        # Version check: if resolved app is below min version, create a new one as default
        if resolved_app and min_mlflow_version and not _mlflow_version_meets_minimum_dict(resolved_app, min_mlflow_version):
            resolved_arn = resolved_app["Arn"]
            logger.info(
                "Existing MLflow app %s has version below %s. Creating new app as default.",
                resolved_arn, min_mlflow_version
            )
            new_arn = _create_mlflow_app_as_upgrade(sagemaker_session, resolved_app, current_domain_id)
            if new_arn:
                logger.info("Created new MLflow app as default: %s", new_arn)
                return new_arn
            logger.warning("Failed to create upgraded MLflow app. Using existing app.")
            return resolved_arn

        if resolved_app:
            return resolved_app["Arn"]

        # Create new app
        new_arn = _create_mlflow_app(sagemaker_session)
        if new_arn:
            logger.info("Created new MLflow app: %s", new_arn)
            return new_arn

        logger.warning("Failed to create MLflow app. MLflow tracking disabled.")
        return None

    except Exception as e:
        logger.error("Error resolving MLflow resource ARN: %s", e)
        return None


def _mlflow_version_meets_minimum_dict(app: dict, min_version: str) -> bool:
    """Check if an MLflow app dict's version meets the minimum requirement."""
    app_version = app.get("MlflowVersion")
    if not app_version:
        return False
    try:
        app_parts = [int(x) for x in str(app_version).split(".")]
        min_parts = [int(x) for x in min_version.split(".")]
        return app_parts >= min_parts
    except (ValueError, AttributeError):
        return False


def _wait_for_mlflow_app_ready_boto(sm_client, arn: str, timeout: int = 600) -> Optional[str]:
    """Poll until MLflow app reaches a ready or terminal state using boto3."""
    logger.info("Waiting for MLflow app to be ready (timeout=%ds)...", timeout)
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = sm_client.describe_mlflow_app(Arn=arn)
        status = resp.get("Status")
        logger.info("MLflow app status: %s (elapsed: %.0fs)", status, time.time() - start_time)
        if status in ["Created", "Updated"]:
            return arn
        if status in ["Failed", "Stopped", "CreateFailed", "DeleteFailed"]:
            logger.error("MLflow app failed with status: %s, reason: %s",
                         status, resp.get("FailureReason"))
            return None
        time.sleep(60)
    logger.warning("Timed out waiting for MLflow app to be ready.")
    return None


def _create_mlflow_app_as_upgrade(
    sagemaker_session, old_app: dict, domain_id: Optional[str]
) -> Optional[str]:
    """Unset default from old app and create a new MLflow app as the default. Returns ARN or None."""
    try:
        sm_client = _get_prod_sm_client(sagemaker_session)
        region = sagemaker_session.boto_session.region_name
        account_id = sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]
        old_arn = old_app["Arn"]

        # Unset default from old app
        if domain_id and domain_id in old_app.get("DefaultDomainIdList", []):
            logger.info("Unsetting domain default from old MLflow app: %s", old_arn)
            sm_client.update_mlflow_app(Arn=old_arn, DefaultDomainIdList=[])
        elif not domain_id and old_app.get("AccountDefaultStatus") == "ENABLED":
            logger.info("Unsetting account default from old MLflow app: %s", old_arn)
            sm_client.update_mlflow_app(Arn=old_arn, AccountDefaultStatus="DISABLED")

        artifact_store_uri = old_app.get("ArtifactStoreUri") or \
            f"s3://sagemaker-{region}-{account_id}/mlflow-artifacts"
        role_arn = old_app.get("RoleArn") or \
            TrainDefaults.get_role(role=None, sagemaker_session=sagemaker_session)
        old_name = old_app.get("Name", "mlflow-app")
        app_name = f"{old_name}-upgraded-{int(time.time())}"

        create_kwargs = {
            "Name": app_name,
            "ArtifactStoreUri": artifact_store_uri,
            "RoleArn": role_arn,
        }
        if domain_id:
            create_kwargs["DefaultDomainIdList"] = [domain_id]
        else:
            create_kwargs["AccountDefaultStatus"] = "ENABLED"

        logger.info("Creating upgraded MLflow app: %s", create_kwargs)
        resp = sm_client.create_mlflow_app(**create_kwargs)
        new_arn = resp["Arn"]
        return _wait_for_mlflow_app_ready_boto(sm_client, new_arn)

    except Exception as e:
        logger.error("Failed to create upgraded MLflow app: %s", e)
        return None


def _create_mlflow_app(sagemaker_session) -> Optional[str]:
    """Create a new MLflow app with minimal configuration. Returns ARN or None."""
    try:
        sm_client = _get_prod_sm_client(sagemaker_session)
        region = sagemaker_session.boto_session.region_name
        account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
        artifact_store_uri = f"s3://sagemaker-{region}-{account_id}/mlflow-artifacts"
        role_arn = TrainDefaults.get_role(role=None, sagemaker_session=sagemaker_session)
        app_name = f"finetune-mlflow-{int(time.time())}"

        # Ensure S3 bucket and prefix exist
        s3_client = sagemaker_session.boto_session.client('s3')
        bucket_name = f"sagemaker-{region}-{account_id}"

        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="mlflow-artifacts/", MaxKeys=1)
            if 'Contents' not in response:
                s3_client.put_object(Bucket=bucket_name, Key="mlflow-artifacts/")
        except s3_client.exceptions.NoSuchBucket:
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            s3_client.put_object(Bucket=bucket_name, Key="mlflow-artifacts/")

        resp = sm_client.create_mlflow_app(
            Name=app_name,
            ArtifactStoreUri=artifact_store_uri,
            RoleArn=role_arn,
            AccountDefaultStatus="DISABLED",
        )
        new_arn = resp["Arn"]
        return _wait_for_mlflow_app_ready_boto(sm_client, new_arn)

    except Exception as e:
        logger.error("Failed to create MLflow app: %s", e)
        return None


def _validate_dataset_arn(dataset: str, param_name: str):
    """Validate that dataset is in correct ARN format."""
    arn_pattern = r"^arn:aws:sagemaker:[^:]+:\d+:hub-content/[^/]+/DataSet/[^/]+/[\d\.]+$"
    if not dataset.startswith("arn:aws:sagemaker:") or not re.match(arn_pattern, dataset):
        raise ValueError(f"{param_name} must be a valid SageMaker hub-content DataSet ARN")


def _validate_evaluator_arn(evaluator_arn: str, param_name: str):
    """Validate that evaluator_arn is in correct ARN format."""
    arn_pattern = r"^arn:aws:sagemaker:[^:]+:\d+:hub-content/[^/]+/JsonDoc/[^/]+/[\d\.]+$"
    if not evaluator_arn.startswith("arn:aws:sagemaker:") or not re.match(arn_pattern, evaluator_arn):
        raise ValueError(f"{param_name} must be a valid SageMaker hub-content evaluator ARN")


def _validate_model_package_group_requirement(model, model_package_group_name):
    """Validate model_package_group_name when source_model_package_arn is not available."""
    if not isinstance(model, ModelPackage) and not model_package_group_name:
        raise ValueError("model_package_group_name must be provided when source_model_package_arn is not available")


def _resolve_model_package_group_arn(model_package_group_name_or_arn, sagemaker_session) -> str:
    """Resolve model package group name, ARN, or ModelPackageGroup object to ARN."""
    if isinstance(model_package_group_name_or_arn, str):
        # Check if it's already an ARN using pattern matching
        arn_pattern = r"^arn:aws:sagemaker:[^:]+:\d+:model-package-group/[^/]+$"
        
        if re.match(arn_pattern, model_package_group_name_or_arn):
            # It's already an ARN
            return model_package_group_name_or_arn
        else:
            # It's a name, resolve to ARN
            model_package_group = ModelPackageGroup.get(
                model_package_group_name=model_package_group_name_or_arn,
                session=sagemaker_session.boto_session,
                region=sagemaker_session.boto_session.region_name
            )
            return model_package_group.model_package_group_arn
    else:
        # It's a ModelPackageGroup object
        return model_package_group_name_or_arn.model_package_group_arn


def _get_default_s3_output_path(sagemaker_session) -> str:
    """Generate default S3 output path: s3://sagemaker-<region>-<account-id>/output"""
    account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
    region = sagemaker_session.boto_session.region_name
    return f"s3://sagemaker-{region}-{account_id}/output"


def _extract_dataset_source(dataset, param_name: str = "dataset"):
    """Extract and validate dataset source from string, S3 URI, or DataSet object."""
    if isinstance(dataset, str):
        # Validate ARN format if it's not an S3 URI
        if not dataset.startswith("s3://"):
            _validate_dataset_arn(dataset, param_name)
        return dataset
    else:
        # It's a DataSet object, extract ARN (already valid)
        return dataset.arn


def _is_lambda_arn(arn: str) -> bool:
    """Check if the given string is a valid AWS Lambda function ARN."""
    return bool(LAMBDA_ARN_REGEX.match(arn))


def _is_nova_model(model_name: str) -> bool:
    """Check if the model is a Nova model based on model name.

    Args:
        model_name: The model name string.

    Returns:
        True if the model name contains 'nova' (case-insensitive).
    """
    return "nova" in model_name.lower()


def _extract_evaluator_arn(evaluator, param_name: str = "custom_reward_function"):
    """Extract and validate evaluator ARN from string or Evaluator object."""
    if isinstance(evaluator, str):
        _validate_evaluator_arn(evaluator, param_name)
        return evaluator
    else:
        # It's an Evaluator object, extract ARN (already valid)
        return evaluator.arn


def _get_lambda_arn_from_evaluator_arn(evaluator_arn: str, sagemaker_session=None) -> str:
    """Extract the Lambda ARN from an evaluator's hub content entry.

    Parses the hub name, content type, content name, and version directly from the
    evaluator ARN, then calls describe_hub_content on the SageMaker client to retrieve
    the stored Lambda function reference. This bypasses AIRHub (which hardcodes a
    single hub name) so it works for evaluators stored in any hub.

    Args:
        evaluator_arn: Full evaluator ARN
            (e.g., arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/JsonDoc/my-eval/0.0.1)
        sagemaker_session: Optional SageMaker session

    Returns:
        The Lambda function ARN stored as the evaluator's reference

    Raises:
        ValueError: If the ARN cannot be parsed or the evaluator has no Lambda reference
    """
    # Parse: arn:aws:sagemaker:region:account:hub-content/HubName/ContentType/ContentName/Version
    try:
        resource_part = evaluator_arn.split(":", 5)[5]  # "hub-content/HubName/Type/Name/Version"
        parts = resource_part.split("/")
        # parts[0] = "hub-content", parts[1] = HubName, parts[2] = ContentType,
        # parts[3] = ContentName, parts[4] = Version (optional)
        hub_name = parts[1]
        hub_content_type = parts[2]
        hub_content_name = parts[3]
        hub_content_version = parts[4] if len(parts) > 4 else None
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Failed to parse evaluator ARN '{evaluator_arn}': {str(e)}"
        )

    sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
    client = sagemaker_session.sagemaker_client

    request = {
        "HubName": hub_name,
        "HubContentType": hub_content_type,
        "HubContentName": hub_content_name,
    }
    if hub_content_version:
        request["HubContentVersion"] = hub_content_version

    response = client.describe_hub_content(**request)

    # Parse the document: HubContentDocument -> JsonContent -> Reference
    doc = json.loads(response["HubContentDocument"])
    json_content = json.loads(doc.get("JsonContent", "{}"))
    reference = json_content.get("Reference", "")

    if not reference:
        raise ValueError(
            f"Evaluator '{hub_content_name}' in hub '{hub_name}' does not have a "
            "Lambda ARN reference."
        )

    return reference


def _resolve_model_name(model_package) -> str:
    """Resolve model_name from model_package if needed."""
    if model_package:
        try:
            # Extract base model from InferenceSpecification
            if (model_package.inference_specification and 
                model_package.inference_specification.containers):
                container = model_package.inference_specification.containers[0]
                if hasattr(container, 'base_model') and container.base_model:
                    return container.base_model.hub_content_name
            
            raise ValueError("Continued fine tuning is only allowed on model packages fine tuned with sagemaker 1p models")
        except Exception as e:
            logger.error("Failed to resolve model_name from model package: %s", e)
            raise
    
    raise ValueError("model name or package must be provided")


def _resolve_model_package_arn(model_package) -> Optional[str]:
    """Resolve model package ARN from model package."""
    try:
        return model_package.model_package_arn
    except Exception as e:
        logger.error("Failed to resolve model package ARN: %s", e)
        return None


def _get_fine_tuning_options_and_model_arn(model_name: str, customization_technique: str, training_type, sagemaker_session,
                                         hub_name: Optional[str] = None, compute: Optional[Union[HyperPodCompute, TrainingJobCompute]] = None) -> tuple:
    """Get fine-tuning options and model ARN for given customization technique.
    Returns:
        tuple: (FineTuningOptions, model_arn, is_gated_model)
    """

    is_hyperpod = isinstance(compute, HyperPodCompute)

    if hub_name is None:
        hub_name = get_sagemaker_hub_name()

    try:

        hub_content = _get_hub_content_metadata(
            hub_name=hub_name,
            hub_content_type="Model", 
            hub_content_name=model_name,
            session=sagemaker_session.boto_session,
            region=sagemaker_session.boto_session.region_name
        )
        
        model_arn = hub_content.get('hub_content_arn')
        document = hub_content.get('hub_content_document')
        
        # Check if model is gated
        is_gated_model = document.get("GatedBucket", False)
        
        recipe_collection = document.get("RecipeCollection", [])
        
        # Filter recipes by customization technique
        matching_recipes = [r for r in recipe_collection if r.get("CustomizationTechnique") == customization_technique]
        
        if not matching_recipes:
            raise ValueError(f"No recipes found for model '{model_name}' with customization technique: {customization_technique}")
        
        # Filter recipes based on compute type:
        # - HyperPodCompute: filter by HpEksPayloadTemplateS3Uri
        # - Default (serverless/SMTJ): filter by SmtjRecipeTemplateS3Uri
        if is_hyperpod:
            recipe_template_key = "HpEksPayloadTemplateS3Uri"
            override_params_key = "HpEksOverrideParamsS3Uri"
            platform_label = "HyperPod"
        else:
            recipe_template_key = "SmtjRecipeTemplateS3Uri"
            override_params_key = "SmtjOverrideParamsS3Uri"
            platform_label = "Smtj"

        recipes_with_template = [r for r in matching_recipes if r.get(recipe_template_key)]
        
        if not recipes_with_template:
            raise ValueError(f"No recipes found with {platform_label} for technique: {customization_technique}")

        # Select recipe based on training type
        recipe = _select_recipe_by_training_type(recipes_with_template, training_type)

        if not recipe:
            raise ValueError(f"No recipes found with {platform_label} for technique: {customization_technique},training_type:{training_type}")

        # Start with the recipe's override_params (platform-specific key)
        options_dict = {}
        if recipe.get(override_params_key):
            s3_uri = recipe[override_params_key]
            # Handle {customer_id} placeholder (subscription recipes use access point ARNs)
            if "{customer_id}" in s3_uri:
                s3_uri = s3_uri.replace("{customer_id}", sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"])
            s3 = sagemaker_session.boto_session.client("s3")
            uri_path = s3_uri.replace("s3://", "")
            # Handle access point ARN URIs (subscription recipes use S3 access points).
            # Format: arn:aws:s3:<region>:<account>:accesspoint/<name>/<key>
            # S3 GetObject expects Bucket=full_ARN_up_to_accesspoint_name, Key=remainder
            if uri_path.startswith("arn:"):
                arn_parts = uri_path.split("/", 2)
                bucket = arn_parts[0] + "/" + arn_parts[1]  # arn:...:accesspoint/<name>
                key = arn_parts[2] if len(arn_parts) > 2 else ""
            else:
                bucket, key = uri_path.split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            options_dict = json.loads(obj["Body"].read())

        # Auto-detect and merge subscription recipe's override_params if available
        if (isinstance(training_type, TrainingType) and training_type == TrainingType.LORA) or training_type == "LORA":
            sub_recipe = next((r for r in recipes_with_template if r.get("Peft") and r.get("IsSubscriptionModel")), None)
        else:
            sub_recipe = next((r for r in recipes_with_template if not r.get("Peft") and r.get("IsSubscriptionModel")), None)

        if sub_recipe and sub_recipe.get(override_params_key):
            try:
                sub_s3_uri = sub_recipe[override_params_key].replace("{customer_id}", sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"])
                sub_uri_path = sub_s3_uri.replace("s3://", "")
                # Handle access point ARN URIs
                if sub_uri_path.startswith("arn:"):
                    arn_parts = sub_uri_path.split("/", 2)
                    sub_bucket = arn_parts[0] + "/" + arn_parts[1]
                    sub_key = arn_parts[2] if len(arn_parts) > 2 else ""
                else:
                    sub_bucket, sub_key = sub_uri_path.split("/", 1)
                s3_sub = sagemaker_session.boto_session.client("s3")
                sub_obj = s3_sub.get_object(Bucket=sub_bucket, Key=sub_key)
                sub_options = json.loads(sub_obj["Body"].read())
                # Merge: subscription params into _specs only (don't set defaults)
                # This makes them settable but not serialized unless user explicitly sets them
                for k, v in sub_options.items():
                    if k not in options_dict:
                        v_copy = v.copy() if isinstance(v, dict) else v
                        if isinstance(v_copy, dict):
                            v_copy['default'] = None  # No default — won't appear in to_dict() unless set
                        options_dict[k] = v_copy
            except Exception as e:
                logger.debug(f"Could not fetch subscription recipe override_params: {type(e).__name__}: {e}")

        if options_dict:
            return FineTuningOptions(options_dict), model_arn, is_gated_model
        else:
            return FineTuningOptions({}), model_arn, is_gated_model
            
    except Exception as e:
        logger.debug("Exception getting fine-tuning options: %s", e)
        raise


def _resolve_base_model_weights_s3_uri(model_name: str, sagemaker_session) -> Optional[str]:
    """Resolve the S3 URI of the base model weights for OSS/LLMFT SMTJ training.

    The LLMFT container feeds ``model_name_or_path`` straight into
    ``AutoModelForCausalLM.from_pretrained(...)``. It therefore needs HF-format
    weights on the local filesystem, delivered via a SageMaker input channel.
    This returns the S3 location of those weights so the caller can mount them
    as a channel and point ``model_name_or_path`` at the local mount.

    We deliberately prefer the SageMaker-prepared ``TrainingArtifactUri`` over the
    public HuggingFace id: for models like gpt-oss the HF weights ship MXFP4-quantized
    (carrying a ``quantization_config`` that breaks the training container), whereas
    the SageMaker training artifact is the dequantized, fine-tuning-ready bf16 copy.

    Args:
        model_name: Hub content name (e.g., "openai-reasoning-gpt-oss-20b")
        sagemaker_session: SageMaker session for API calls

    Returns:
        str or None: S3 URI of the base model weights, or None if unresolvable.
    """
    try:
        hub_name = get_sagemaker_hub_name()
        hub_content = _get_hub_content_metadata(
            hub_name=hub_name,
            hub_content_type="Model",
            hub_content_name=model_name,
            session=sagemaker_session.boto_session,
            region=sagemaker_session.boto_session.region_name,
        )
        document = hub_content.get("hub_content_document", {})

        training_artifact_uri = document.get("TrainingArtifactUri")
        if training_artifact_uri and training_artifact_uri.startswith("s3://"):
            logger.info(
                f"Resolved base model weights from TrainingArtifactUri: {training_artifact_uri}"
            )
            return training_artifact_uri

    except Exception as e:
        logger.debug(f"Could not resolve base model weights for '{model_name}': {e}")

    return None


def _create_input_channels(dataset: str, content_type: Optional[str] = None, 
                         input_compression_type: Optional[str] = None,
                         record_wrapper_type: Optional[str] = None,
                         input_mode: Optional[str] = None):
    """Create input channels from dataset (S3 URI or dataset ARN).
    
    Args:
        dataset: S3 URI (s3://bucket/key) or dataset ARN (arn:aws:sagemaker:...)
        
    Returns:
        list: List of Channel objects
    """

    channels = []

    
    if dataset.startswith("s3://"):
        # S3 URI - create S3DataSource
        data_source = DataSource(
            s3_data_source={
                "s3_uri": dataset,
                "s3_data_type": "S3Prefix",
                "s3_data_distribution_type": "FullyReplicated"
            }
        )
    else:
        # Dataset ARN - validate and create dataset source
        _validate_dataset_arn(dataset, "dataset")
        data_source = DataSource(
            dataset_source={"dataset_arn": dataset}
        )
    
    channel = Channel(
        channel_name="train",
        data_source=data_source,
        content_type=content_type,
        compression_type=input_compression_type,
        record_wrapper_type=record_wrapper_type,
        input_mode=input_mode
        )
    channels.append(channel)
    
    return channels


def _resolve_model_with_checkpoint(model, base_model_name, compute, sagemaker_session=None, resolve_fn=None):
    """Resolve model identity and checkpoint source from model param.

    Handles the S3 checkpoint detection: when model is an S3 URI, base_model_name
    is used for recipe lookup and model becomes the checkpoint source.

    Args:
        model: Model name, ARN, ModelPackage, or S3 URI (checkpoint).
        base_model_name: Required when model is S3 URI. Identifies the model for recipe lookup.
        compute: Compute config. Required when model is S3 URI.
        sagemaker_session: SageMaker session for API calls.
        resolve_fn: Model resolution function. Defaults to _resolve_model_and_name.
            Pass the trainer's imported copy so test patches are respected.

    Returns:
        tuple: (resolved_model, model_name, model_source)
            - resolved_model: Model identity (name or ModelPackage)
            - model_name: Normalized model name for Hub lookups
            - model_source: S3 checkpoint path, or None
    """
    if resolve_fn is None:
        resolve_fn = _resolve_model_and_name

    if isinstance(model, str) and model.startswith("s3://"):
        if not base_model_name:
            raise ValueError(
                "base_model_name is required when 'model' is an S3 URI "
                "(e.g., base_model_name='amazon.nova-2-lite-v1')."
            )
        if compute is None:
            raise ValueError(
                "S3 checkpoint model is only supported with HyperPodCompute or "
                "TrainingJobCompute. For serverless, pass a model name or model package ARN instead."
            )
        _, model_name = resolve_fn(base_model_name, sagemaker_session)
        return base_model_name, model_name, model
    else:
        if base_model_name:
            logger.warning(
                "base_model_name is ignored when 'model' is not an S3 URI. "
                "'model' parameter will be used."
            )
        resolved_model, model_name = resolve_fn(model, sagemaker_session)
        return resolved_model, model_name, None


def _resolve_model_and_name(model, sagemaker_session=None):
    """Resolve model and extract model name from string, ARN, or ModelPackage object.
    
    Args:
        model: Can be a model name (str), model package ARN (str), or ModelPackage object
        sagemaker_session: SageMaker session for API calls (required for ARN resolution)
    
    Returns:
        tuple: (resolved_model, model_name)
    """
    # Get region for validation
    region_name = None
    if sagemaker_session:
        region_name = sagemaker_session.boto_region_name
    else:
        # Try to get region from SAGEMAKER_REGION env var, then boto3 session, then AWS_DEFAULT_REGION
        region_name = os.environ.get('SAGEMAKER_REGION')
        if not region_name:
            try:
                import boto3
                region_name = boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION')
            except:
                pass
    
    if isinstance(model, str):
        # Check if it's a model package ARN
        if model.startswith("arn:aws:sagemaker:") and ":model-package/" in model:
            # Get ModelPackage object from ARN
            model_package = ModelPackage.get(
                model_package_name=model,
                session=sagemaker_session.boto_session if sagemaker_session else None,
                region=sagemaker_session.boto_session.region_name if sagemaker_session else None
            )
            model_name = _resolve_model_name(model_package)
            # Validate region availability
            if region_name:
                _validate_model_region_availability(model_name, region_name)
            return model_package, model_name
        else:
            # It's a regular model name string
            # Normalize Bedrock-style model IDs to Hub content names
            model_name = _normalize_model_name(model)
            # Validate region availability
            if region_name:
                _validate_model_region_availability(model_name, region_name)
            return model_name, model_name
    else:
        # It's a ModelPackage object
        model_name = _resolve_model_name(model)
        # Validate region availability
        if region_name:
            _validate_model_region_availability(model_name, region_name)
        return model, model_name


def _create_serverless_config(model_arn, customization_technique,
                           training_type, accept_eula, evaluator_arn=None, job_type=JOB_TYPE) -> Optional['ServerlessJobConfig']:
    """Create serverless job configuration for fine-tuning.
    
    Args:
        model_arn: ARN of the base model
        customization_technique: Technique used (e.g., "SFT", "DPO", "RLVR", "RLAIF")
        training_type: Training type (TrainingType enum or string)
        accept_eula: Boolean indicating if EULA is accepted
        evaluator_arn: Optional evaluator ARN for RLVR/RLAIF
        job_type: Type of job (default: "FineTuning")
    
    Returns:
        ServerlessJobConfig object or None if required parameters are missing
    """
    peft = None if (isinstance(training_type, TrainingType) and training_type == TrainingType.FULL) \
        else (training_type.value if isinstance(training_type, TrainingType) else training_type)

    # Create ServerlessJobConfig using shapes
    serverless_config = ServerlessJobConfig(
        job_type=job_type,
        base_model_arn=model_arn,
        customization_technique=customization_technique,
        peft=peft,
        evaluator_arn=evaluator_arn,
        accept_eula=accept_eula
    )

    return serverless_config


def _create_input_data_config(training_dataset, validation_dataset=None):
    """Create input data configuration from training and validation datasets.
    
    Args:
        training_dataset: Training dataset (method parameter takes priority over class attribute)
        validation_dataset: Validation dataset (method parameter takes priority over class attribute)
    
    Returns:
        List of InputData objects for training job configuration
    """
    # Extract and validate training dataset
    final_training_dataset = _extract_dataset_source(training_dataset, "training_dataset")
    
    input_data_config = [
        InputData(channel_name="train", data_source=final_training_dataset)
    ]
    
    # Add validation dataset if provided
    if validation_dataset:
        final_validation_dataset = _extract_dataset_source(validation_dataset, "validation_dataset")
        input_data_config.append(
            InputData(channel_name="validation", data_source=final_validation_dataset)
        )
    
    return input_data_config



def _create_model_package_config(model_package_group_name, model, sagemaker_session):
    """Create model package configuration with resolved ARNs.
    
    Args:
        model_package_group_name: Model package group name to resolve
        model: Model object (used to resolve source model package ARN if it's a ModelPackage)
        sagemaker_session: SageMaker session for API calls
    
    Returns:
        ModelPackageConfig object or None if no model package group name provided
    """
    
    model_package_group_arn = None
    if model_package_group_name:
        model_package_group_arn = _resolve_model_package_group_arn(
            model_package_group_name, sagemaker_session
        )

    source_model_package_arn = None
    if isinstance(model, ModelPackage):
        source_model_package_arn = _resolve_model_package_arn(model)

    return ModelPackageConfig(
        model_package_group_arn=model_package_group_arn,
        source_model_package_arn=source_model_package_arn,
    )



def _create_mlflow_config(sagemaker_session, mlflow_resource_arn=None, 
                       mlflow_experiment_name=None, mlflow_run_name=None):
    """Create MLflow configuration with resolved resource ARN.
    
    Args:
        sagemaker_session: SageMaker session for resolving MLflow ARN
        mlflow_resource_arn: MLflow resource ARN (if None, uses default experience)
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
    
    Returns:
        MlflowConfig object or None if no MLflow resource ARN is resolved
    """

    
    # Derive mlflow_resource_arn with default experience
    resolved_mlflow_arn = _resolve_mlflow_resource_arn(sagemaker_session, mlflow_resource_arn)
    logger.info(f"MLflow resource ARN: {resolved_mlflow_arn}")

    # Create MlflowConfig using shapes
    mlflow_config = None
    if resolved_mlflow_arn:
        mlflow_config = MlflowConfig(
            mlflow_resource_arn=resolved_mlflow_arn,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
        )
    
    return mlflow_config


def _create_output_config(sagemaker_session, s3_output_path=None, kms_key_id=None, disable_output_compression=False):
    """Create output data configuration with default S3 path if needed.

    Args:
        s3_output_path: S3 output path (if None, generates default path)
        sagemaker_session: SageMaker session for generating default path
        kms_key_id: Optional KMS key ID for encryption
        disable_output_compression: If True, sets compression_type to "NONE"

    Returns:
        OutputDataConfig object
    """

    # Use default S3 output path if none provided
    if s3_output_path is None:
        s3_output_path = _get_default_s3_output_path(sagemaker_session)

    # Validate S3 path exists
    _validate_s3_path_exists(s3_output_path, sagemaker_session, kms_key_id=kms_key_id)

    config_kwargs = {
        "s3_output_path": s3_output_path,
        "kms_key_id": kms_key_id,
    }
    if disable_output_compression:
        config_kwargs["compression_type"] = "NONE"

    return OutputDataConfig(**config_kwargs)


def _convert_input_data_to_channels(input_data_config, s3_data_type="S3Prefix"):
    """Convert InputData objects to Channel objects with S3 and dataset ARN support.
    
    Args:
        input_data_config: List of InputData objects
        s3_data_type: The S3 data type to use for S3 data sources. Use "Converse"
            for Nova SFT/DPO multimodal datasets so the SageMaker data agent
            downloads images and rewrites "uri" to "localPath" in the JSONL.
            Defaults to "S3Prefix".
    
    Returns:
        List of Channel objects
    """
    
    channels = []
    for input_data in input_data_config:
        if input_data.data_source.startswith("s3://"):
            # S3 URI - create S3DataSource
            data_source = DataSource(
                s3_data_source={
                    "s3_uri": input_data.data_source,
                    "s3_data_type": s3_data_type,
                    "s3_data_distribution_type": "FullyReplicated"
                }
            )
        else:
            # Dataset ARN - create dataset source
            data_source = DataSource(
                dataset_source={"dataset_arn": input_data.data_source}
            )

        channel = Channel(
            channel_name=input_data.channel_name,
            data_source=data_source,
        )
        channels.append(channel)
    
    return channels


def _validate_and_resolve_model_package_group(model, model_package_group_name):
    """Validate and resolve model_package_group_name from ModelPackage if needed.
    
    Only called for serverless compute paths where model_package_group is required.
    """
    # If model_package_group_name is already provided, return it as-is
    if model_package_group_name:
        return model_package_group_name

    # Try to resolve from ModelPackage if available
    if isinstance(model, ModelPackage):
        return model.model_package_group_name

    raise ValueError("model_package_group_name must be provided when model given is "
                     "not a ModelPackage artifact/not continued finetuning")


def _validate_eula_for_gated_model(model, accept_eula, is_gated_model):
    """Validate EULA acceptance for gated models.
    
    Args:
        model: Original model input (string, ARN, or ModelPackage)
        accept_eula: Boolean indicating if EULA is accepted
        is_gated_model: Boolean indicating if the model is gated
    
    Returns:
        bool: True if EULA is accepted (either explicitly or by default for ARN/ModelPackage)
    
    Raises:
        ValueError: If model is gated but accept_eula is False
    """
    # For ModelPackage/ARN inputs, EULA is assumed accepted by default
    if isinstance(model, ModelPackage) or (isinstance(model, str) and model.startswith("arn:aws:sagemaker:")):
        return True
    
    # Validate EULA acceptance for gated models
    if is_gated_model and not accept_eula:
        raise ValueError(
            f"Model '{model}' is a gated model and requires EULA acceptance. "
            "Please set accept_eula=True to proceed with training."
        )
    
    return accept_eula


def _validate_s3_path_exists(s3_path: str, sagemaker_session, kms_key_id: Optional[str] = None):
    """Validate S3 path and create bucket/prefix if they don't exist.

    Args:
        s3_path: S3 path to validate.
        sagemaker_session: SageMaker session used to create the S3 client.
        kms_key_id: Optional KMS key ID. When provided, objects created to
            initialize a missing prefix are written with SSE-KMS encryption so
            that validation does not fail on buckets enforcing KMS encryption.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path format: {s3_path}")
    
    # Parse S3 URI
    s3_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket_name = s3_parts[0]
    prefix = s3_parts[1] if len(s3_parts) > 1 else ""
    
    s3_client = sagemaker_session.boto_session.client('s3')
    
    try:
        # Check if bucket exists, create if it doesn't
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except Exception as e:
            if "NoSuchBucket" in str(e) or "Not Found" in str(e):
                # Create bucket
                region = sagemaker_session.boto_region_name
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
            else:
                raise
        
        # If prefix is provided, check if it exists, create if it doesn't
        if prefix:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
            if 'Contents' not in response:
                # Create the prefix by putting an empty object
                if not prefix.endswith('/'):
                    prefix += '/'
                put_object_kwargs = {"Bucket": bucket_name, "Key": prefix, "Body": b''}
                if kms_key_id:
                    put_object_kwargs["ServerSideEncryption"] = "aws:kms"
                    put_object_kwargs["SSEKMSKeyId"] = kms_key_id
                s3_client.put_object(**put_object_kwargs)
                
    except Exception as e:
        raise ValueError(f"Failed to validate/create S3 path '{s3_path}': {str(e)}")


def _validate_hyperparameter_values(hyperparameters: dict):
    """Validate hyperparameter values for allowed characters."""
    import re
    allowed_chars = r"^[a-zA-Z0-9/_.:,\-\s'\"\[\]]*$"
    for key, value in hyperparameters.items():
        if isinstance(value, str) and not re.match(allowed_chars, value):
            raise ValueError(
                f"Hyperparameter '{key}' value '{value}' contains invalid characters. "
                f"Only a-z, A-Z, 0-9, /, _, ., :, \\, -, space, ', \", [, ] and , are allowed."
            )


def get_recipe_s3_uri(model_name: str, customization_technique: str, training_type,
                      sagemaker_session, hub_name: Optional[str] = None) -> str:
    """Resolve the SmtjRecipeTemplateS3Uri for a model + technique + training_type.

    This is used by the SMTJ compute path to download the recipe and pass it to
    ModelTrainer.from_recipe().

    Args:
        model_name: Hub content name or Bedrock-style model ID (e.g. "amazon.nova-lite-v2")
        customization_technique: e.g. "RLVR", "SFT", "DPO"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session for API calls
        hub_name: Optional hub name override

    Returns:
        str: S3 URI of the recipe template YAML

    Raises:
        ValueError: If no matching recipe is found
    """
    model_name = _normalize_model_name(model_name)

    if hub_name is None:
        hub_name = get_sagemaker_hub_name()

    hub_content = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_type="Model",
        hub_content_name=model_name,
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_session.region_name
    )

    document = hub_content.get('hub_content_document')
    recipe_collection = document.get("RecipeCollection", [])

    # Filter recipes by customization technique
    matching_recipes = [r for r in recipe_collection if r.get("CustomizationTechnique") == customization_technique]

    if not matching_recipes:
        raise ValueError(
            f"No recipes found for model '{model_name}' with customization technique: {customization_technique}"
        )

    # Filter recipes that have SmtjRecipeTemplateS3Uri
    recipes_with_template = [r for r in matching_recipes if r.get("SmtjRecipeTemplateS3Uri")]

    if not recipes_with_template:
        raise ValueError(
            f"No SMTJ recipes found for model '{model_name}' with technique: {customization_technique}"
        )

    # Select recipe based on training type
    recipe = _select_recipe_by_training_type(recipes_with_template, training_type, with_fallback=True)

    if not recipe:
        raise ValueError(
            f"No SMTJ recipe found for technique: {customization_technique}, training_type: {training_type}"
        )

    return resolve_s3_uri_placeholders(recipe["SmtjRecipeTemplateS3Uri"], sagemaker_session)


def _get_recipe_entry_and_override_spec(
    model_name: str,
    customization_technique: str,
    training_type,
    sagemaker_session,
    platform: str = "smtj",
    hub_name: Optional[str] = None,
    display_name_filter: Optional[str] = None,
) -> tuple:
    """Resolve the matching recipe entry and its override spec from SageMaker Hub.

    Shared logic for both SMTJ and HyperPod paths. Fetches hub content,
    filters the RecipeCollection by technique/platform/training_type, downloads
    the override params JSON, and adds infrastructure field defaults.

    Args:
        model_name: Hub content name or Bedrock-style model ID
        customization_technique: e.g. "SFT", "RLVR", "DPO", "CPT"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session for API calls
        platform: "smtj" or "hyperpod" — determines which S3 URI keys to use
        hub_name: Optional hub name override

    Returns:
        tuple: (recipe_entry dict, override_spec dict)

    Raises:
        ValueError: If no matching recipe is found
    """
    model_name = _normalize_model_name(model_name)

    if hub_name is None:
        hub_name = get_sagemaker_hub_name()

    hub_content = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_type="Model",
        hub_content_name=model_name,
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_session.region_name
    )

    document = hub_content.get('hub_content_document')
    recipe_collection = document.get("RecipeCollection", [])

    # Filter by customization technique
    # Evaluation recipes use "Type": "Evaluation" in the Hub rather than
    # "CustomizationTechnique": "Evaluation", so check both fields.
    if customization_technique == "Evaluation":
        matching_recipes = [r for r in recipe_collection
                            if r.get("CustomizationTechnique") == customization_technique
                            or r.get("Type") == "Evaluation"]
    else:
        matching_recipes = [r for r in recipe_collection
                            if r.get("CustomizationTechnique") == customization_technique]

    if not matching_recipes:
        raise ValueError(
            f"No recipes found for model '{model_name}' with customization technique: "
            f"{customization_technique}"
        )

    # Filter by platform
    if platform == "hyperpod":
        template_key = "HpEksPayloadTemplateS3Uri"
        override_key = "HpEksOverrideParamsS3Uri"
    else:
        template_key = "SmtjRecipeTemplateS3Uri"
        override_key = "SmtjOverrideParamsS3Uri"

    platform_recipes = [r for r in matching_recipes if r.get(template_key)]

    if not platform_recipes:
        raise ValueError(
            f"No {platform} recipes found for model '{model_name}' with technique: "
            f"{customization_technique}"
        )

    # Filter by display name if specified (e.g., "benchmark" for general text benchmark eval)
    if display_name_filter:
        filtered = [r for r in platform_recipes
                    if display_name_filter.lower() in r.get("DisplayName", "").lower()]
        if filtered:
            platform_recipes = filtered

    # Select by training type
    recipe = _select_recipe_by_training_type(platform_recipes, training_type)

    if not recipe:
        raise ValueError(
            f"No {platform} recipe found for technique: {customization_technique}, "
            f"training_type: {training_type}"
        )

    # Download override params spec from S3
    override_spec = {}
    override_s3_uri = recipe.get(override_key)
    if override_s3_uri:
        override_s3_uri = resolve_s3_uri_placeholders(override_s3_uri, sagemaker_session)
        s3_client = sagemaker_session.boto_session.client("s3")
        uri_path = override_s3_uri.replace("s3://", "")

        # Handle S3 access point ARN URIs
        if uri_path.startswith("arn:"):
            match = re.match(r'(arn:aws:s3:[^:]*:[^:]*:accesspoint/[^/]+)/(.*)', uri_path)
            if match:
                bucket = match.group(1)
                key = match.group(2)
            else:
                raise ValueError(f"Cannot parse S3 access point ARN: {uri_path}")
        else:
            bucket, key = uri_path.split("/", 1)

        response = s3_client.get_object(Bucket=bucket, Key=key)
        override_spec = json.loads(response["Body"].read())

    # Add infrastructure fields not in the spec but present in recipe templates
    for infra_key in ("name", "data_s3_path", "output_s3_path",
                      "mlflow_tracking_uri", "mlflow_experiment_name", "mlflow_run_name"):
        if infra_key not in override_spec:
            override_spec[infra_key] = {"default": "", "type": "string"}

    return recipe, override_spec


def _get_smtj_override_spec(model_name: str, customization_technique: str, training_type,
                            sagemaker_session, hub_name: Optional[str] = None) -> dict:
    """Fetch the SMTJ override params spec JSON for a model recipe.

    Returns the parsed override spec dict from SmtjOverrideParamsS3Uri,
    with infrastructure fields added for rendering.

    Args:
        model_name: Hub content name (e.g. "amazon.nova-lite-v2")
        customization_technique: e.g. "SFT", "RLVR", "DPO"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session for API calls
        hub_name: Optional hub name override

    Returns:
        dict: Override spec mapping parameter names to their metadata.
    """
    _, override_spec = _get_recipe_entry_and_override_spec(
        model_name=model_name,
        customization_technique=customization_technique,
        training_type=training_type,
        sagemaker_session=sagemaker_session,
        platform="smtj",
        hub_name=hub_name,
    )
    return override_spec


def _get_smhp_replicas_enum(model_name: str, customization_technique: str, training_type,
                            sagemaker_session, hub_name: Optional[str] = None) -> Optional[list]:
    """Fetch the replicas enum from the SMHP override spec for the same model/technique.

    SMTJ hub content does not include a replicas enum in its override spec, but
    the SMHP recipe for the same configuration does. This function retrieves that
    enum so it can be applied to SMTJ recipe validation.

    Returns:
        List of valid instance counts, or None if unavailable.
    """
    try:
        _, smhp_override_spec = _get_recipe_entry_and_override_spec(
            model_name=model_name,
            customization_technique=customization_technique,
            training_type=training_type,
            sagemaker_session=sagemaker_session,
            platform="hyperpod",
            hub_name=hub_name,
        )
        replicas_meta = smhp_override_spec.get("replicas", {})
        enum_val = replicas_meta.get("enum")
        if isinstance(enum_val, list) and enum_val:
            return enum_val
    except Exception as e:
        logger.warning(
            f"Could not fetch valid instance counts from SMHP recipe for "
            f"{model_name}/{customization_technique}: {e}. "
            "Instance count validation will be skipped."
        )
    return None


def _extract_recipe_from_helm_template(template_content: str) -> str:
    """Extract the training config YAML from a HyperPod Helm chart template.

    The HpEksPayloadTemplateS3Uri contains a full Helm chart (multi-document YAML
    with ``---`` separators). The HyperPod CLI expects a single-document recipe YAML.
    This function extracts just the ``config.yaml`` content section.

    Args:
        template_content: Raw Helm chart template string from S3.

    Returns:
        str: Single-document recipe YAML content.

    Raises:
        ValueError: If the training config section cannot be extracted.
    """
    import textwrap

    if "training-config.yaml" not in template_content:
        raise ValueError(
            "Could not extract training config from HyperPod recipe template. "
            "Expected 'training-config.yaml' section not found."
        )

    recipe_pattern = (
        r"# Source: .*/training-config\.yaml.*?config\.yaml: \|-\n(.*?)(?=---|\Z)"
    )
    recipe_match = re.search(recipe_pattern, template_content, re.DOTALL)
    if not recipe_match:
        raise ValueError(
            "Could not extract training config from HyperPod recipe template. "
            "The template format may have changed."
        )

    return textwrap.dedent(recipe_match.group(1)).strip()


def _render_recipe_placeholders(recipe_content: str, override_spec: dict) -> str:
    """Replace {{placeholder}} tokens in a recipe with default values from the spec.

    The extracted HyperPod recipe template contains ``{{key}}`` placeholders that
    must be rendered with actual typed values before the training container can
    validate the recipe.

    Args:
        recipe_content: Recipe YAML string with ``{{placeholder}}`` tokens.
        override_spec: Dict mapping parameter names to their spec
            (with ``default``, ``type``, etc.).

    Returns:
        str: Recipe YAML with all placeholders replaced by default values.
    """
    for key, spec in override_spec.items():
        default_value = spec.get("default") if isinstance(spec, dict) else spec
        placeholder = "{{" + key + "}}"
        placeholder_single_quoted = "'" + placeholder + "'"
        placeholder_double_quoted = '"' + placeholder + '"'

        if default_value is None:
            # For string-typed fields, use empty string instead of null
            expected_type = spec.get("type", "") if isinstance(spec, dict) else ""
            yaml_value = "''" if expected_type == "string" else "null"
        elif isinstance(default_value, bool):
            yaml_value = "true" if default_value else "false"
        elif isinstance(default_value, str):
            yaml_value = "''" if default_value == "" else default_value
        else:
            yaml_value = str(default_value)

        recipe_content = recipe_content.replace(placeholder_single_quoted, yaml_value)
        recipe_content = recipe_content.replace(placeholder_double_quoted, yaml_value)
        recipe_content = recipe_content.replace(placeholder, yaml_value)

    # Validate no unresolved {{...}} placeholders remain.
    # All training config placeholders should have been resolved from the override spec.
    # Infrastructure fields (mlflow, data paths) should also be in the spec or absent.
    remaining = re.findall(r"[{][{]([^}]+)[}][}]", recipe_content)
    if remaining:
        raise ValueError(
            f"Recipe template has unresolved placeholders not found in override spec: "
            f"{remaining}. The HpEksOverrideParamsS3Uri JSON may be incomplete."
        )

    return recipe_content


def get_hyperpod_recipe_path(model_name: str, customization_technique: str, training_type,
                             sagemaker_session, job_name: str,
                             hub_name: Optional[str] = None,
                             display_name_filter: Optional[str] = None,
                             additional_overrides: Optional[Dict[str, Any]] = None) -> str:
    """Resolve and write the HyperPod recipe for a model from SageMaker Hub.

    Downloads the recipe template from Hub (HpEksPayloadTemplateS3Uri), writes it
    to the HyperPod CLI's local recipe directory, and returns the relative recipe
    path that ``hyperpod start-job --recipe`` expects.

    This mirrors the Nova Forge SDK approach: the recipe YAML is fetched from Hub,
    placed in the CLI's recipes_collection directory, and the CLI resolves it by
    relative name.

    Args:
        model_name: Hub content name (e.g. "nova-textgeneration-lite-v2")
        customization_technique: e.g. "RLVR", "SFT", "DPO", "CPT", "Evaluation"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session for API calls
        job_name: Unique job name used as recipe filename
        hub_name: Optional hub name override
        display_name_filter: Optional filter to match against recipe DisplayName
            (e.g., "benchmark" to prefer general text benchmark eval recipes)
        additional_overrides: Optional dict of extra key-value pairs to inject
            into the override spec before rendering. These take precedence over
            Hub defaults. Keys should be flat spec keys (e.g., "save_steps").

    Returns:
        str: Relative recipe path for the HyperPod CLI (without .yaml extension)

    Raises:
        ValueError: If no matching HyperPod recipe is found in Hub metadata
        RuntimeError: If the HyperPod CLI is not installed
    """
    import uuid
    import yaml

    recipe, override_spec = _get_recipe_entry_and_override_spec(
        model_name=model_name,
        customization_technique=customization_technique,
        training_type=training_type,
        sagemaker_session=sagemaker_session,
        platform="hyperpod",
        hub_name=hub_name,
        display_name_filter=display_name_filter,
    )

    hp_template_s3_uri = recipe["HpEksPayloadTemplateS3Uri"]
    logger.info(f"HyperPod recipe template S3 URI: {hp_template_s3_uri}")

    # Download recipe template from S3
    s3_client = sagemaker_session.boto_session.client("s3")
    uri_path = hp_template_s3_uri.replace("s3://", "")
    bucket, key = uri_path.split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    recipe_content = response["Body"].read().decode("utf-8")

    # Extract the training config from the Helm chart template
    recipe_content = _extract_recipe_from_helm_template(recipe_content)

    # Inject additional overrides into spec before rendering
    if additional_overrides:
        for k, v in additional_overrides.items():
            entry = override_spec.get(k)
            if isinstance(entry, dict):
                entry["default"] = v
            else:
                override_spec[k] = {"default": v, "type": "string"}

    # Render {{placeholder}} values with defaults from the override params spec.
    recipe_content = _render_recipe_placeholders(recipe_content, override_spec)

    # Resolve the HyperPod CLI's recipe directory
    try:
        import hyperpod_cli
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The HyperPod CLI is a required dependency for running HyperPod jobs. "
            "Install it with: pip install hyperpod"
        ) from e

    HYPERPOD_RECIPE_PATH = os.path.join(
        "sagemaker_hyperpod_recipes", "recipes_collection", "recipes"
    )
    hp_cli_recipes_dir = os.path.join(
        os.path.dirname(hyperpod_cli.__file__), HYPERPOD_RECIPE_PATH
    )

    # Build recipe subdirectory based on technique
    technique_lower = customization_technique.lower()
    if technique_lower in ("sft", "dpo", "rlvr", "rft"):
        subdir = "fine-tuning"
    elif technique_lower == "cpt":
        subdir = "training"
    elif technique_lower == "evaluation":
        subdir = "evaluation"
    else:
        subdir = "fine-tuning"

    # Construct a unique filename
    recipe_filename = f"{job_name}-{str(uuid.uuid4())[:4]}.yaml"
    recipe_dir = os.path.join(hp_cli_recipes_dir, subdir, "nova")
    os.makedirs(recipe_dir, exist_ok=True)

    recipe_path = os.path.join(recipe_dir, recipe_filename)

    # Write recipe to the HyperPod CLI directory
    with open(recipe_path, "w") as f:
        f.write(recipe_content)

    logger.info(f"HyperPod recipe written to: {recipe_path}")

    # Return relative path (strip prefix + .yaml extension) as expected by CLI
    relative_path = (
        recipe_path.split(HYPERPOD_RECIPE_PATH, 1)[1]
        .lstrip("/").lstrip("\\")
        .removesuffix(".yaml")
    )

    return relative_path


def get_training_image(model_name: str, customization_technique: str, training_type,
                       sagemaker_session, hub_name: Optional[str] = None) -> Optional[str]:
    """Resolve the training image URI for SMTJ from hub model metadata.

    Args:
        model_name: Hub content name or Bedrock-style model ID (e.g. "amazon.nova-lite-v2")
        customization_technique: e.g. "RLVR", "SFT", "DPO"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session
        hub_name: Optional hub name override

    Returns:
        str or None: Training image URI if specified in recipe metadata
    """
    model_name = _normalize_model_name(model_name)

    if hub_name is None:
        hub_name = get_sagemaker_hub_name()

    hub_content = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_type="Model",
        hub_content_name=model_name,
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_session.region_name
    )

    document = hub_content.get('hub_content_document')
    recipe_collection = document.get("RecipeCollection", [])

    matching_recipes = [r for r in recipe_collection if r.get("CustomizationTechnique") == customization_technique]
    if not matching_recipes and customization_technique == "Evaluation":
        matching_recipes = [r for r in recipe_collection if r.get("Type") == "Evaluation"]
    recipes_with_template = [r for r in matching_recipes if r.get("SmtjRecipeTemplateS3Uri")]

    recipe = _select_recipe_by_training_type(recipes_with_template, training_type, with_fallback=True)

    if recipe:
        return recipe.get("SmtjImageUri")
    return None


def get_hyperpod_training_image(model_name: str, customization_technique: str, training_type,
                                sagemaker_session, hub_name: Optional[str] = None) -> Optional[str]:
    """Resolve the training image URI for HyperPod from the EKS payload template.

    Downloads the HpEksPayloadTemplateS3Uri for the matching recipe and extracts
    the container image from the Helm chart template.

    Args:
        model_name: Hub content name or Bedrock-style model ID (e.g. "amazon.nova-lite-v2")
        customization_technique: e.g. "RLVR", "SFT", "DPO", "CPT"
        training_type: TrainingType enum or string ("LORA", "FULL")
        sagemaker_session: SageMaker session
        hub_name: Optional hub name override

    Returns:
        str or None: Training image URI if found in the template
    """
    try:
        recipe, _override_spec = _get_recipe_entry_and_override_spec(
            model_name=model_name,
            customization_technique=customization_technique,
            training_type=training_type,
            sagemaker_session=sagemaker_session,
            platform="hyperpod",
            hub_name=hub_name,
        )
    except ValueError:
        return None

    hp_template_s3_uri = recipe.get("HpEksPayloadTemplateS3Uri")
    if not hp_template_s3_uri:
        return None

    # Download recipe template from S3
    s3_client = sagemaker_session.boto_session.client("s3")
    uri_path = hp_template_s3_uri.replace("s3://", "")
    bucket, key = uri_path.split("/", 1)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        template_content = response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.warning("Failed to download HyperPod template from s3://%s/%s: %s", bucket, key, e)
        return None

    return extract_image_from_hyperpod_template(template_content)


def extract_image_from_hyperpod_template(template_content: str) -> Optional[str]:
    """Extract the container image URI from a HyperPod Helm chart template.

    Parses the raw template content looking for the pytorch container image
    definition in the training-config.yaml section.

    Args:
        template_content: Raw Helm chart template string from S3.

    Returns:
        str or None: The image URI if found, None otherwise.
    """
    if "training-config.yaml" in template_content:
        image_pattern = r"name:\s*pytorch\s*\n\s*image:\s*(.+?)(?:\s|$)"
        image_match = re.search(image_pattern, template_content, re.MULTILINE)
        if image_match:
            return image_match.group(1).strip()
    return None
