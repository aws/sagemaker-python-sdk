"""Common utilities for fine-tuning trainers."""

import os
import re
import time
import logging
import json
from typing import Optional
import time
import boto3
from sagemaker.core.resources import MlflowApp, ModelPackage, ModelPackageGroup
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.common_utils.recipe_utils import _get_hub_content_metadata
from sagemaker.train.common import TrainingType, CustomizationTechnique, JOB_TYPE, FineTuningOptions
from sagemaker.core.shapes import ServerlessJobConfig, Channel, DataSource, ModelPackageConfig, MlflowConfig
from sagemaker.train.configs import InputData, OutputDataConfig
from sagemaker.train.defaults import TrainDefaults

logger = logging.getLogger(__name__)

# Region mappings for model availability
OPEN_WEIGHTS_REGIONS = ['us-east-1', 'us-west-2', 'ap-northeast-1', 'eu-west-1']  # IAD, PDX, NRT, DUB
NOVA_REGIONS = ['us-east-1']  # IAD only
# Constants
DEFAULT_REGION = "us-west-2"

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


def _resolve_mlflow_resource_arn(sagemaker_session, mlflow_resource_arn: Optional[str] = None) -> Optional[str]:
    """Resolve MLflow resource ARN using default experience logic."""
    if mlflow_resource_arn:
        return mlflow_resource_arn
    
    try:
        mlflow_apps = MlflowApp.get_all(
            session=sagemaker_session.boto_session,
            region=sagemaker_session.boto_session.region_name
        )
        
        mlflow_apps_list = list(mlflow_apps)
        current_domain_id = _get_current_domain_id(sagemaker_session)
        
        # Check for domain match
        if current_domain_id:
            domain_match = next((app for app in mlflow_apps_list
                               if isinstance(app.default_domain_id_list, list) 
                               and current_domain_id in app.default_domain_id_list), None)
            if domain_match:
                logger.info("Using domain-matched MLflow app: %s", domain_match.arn)
                return domain_match.arn
        
        # Check for account default
        account_default = next((app for app in mlflow_apps_list 
                              if app.account_default_status == "ENABLED"), None)
        if account_default:
            logger.info("Using account default MLflow app: %s", account_default.arn)
            return account_default.arn
        
        # Use first available with ready status
        if mlflow_apps_list:
            ready_app = next((app for app in mlflow_apps_list 
                            if app.status in ["Created", "Updated"]), None)
            if ready_app:
                logger.info("Using first available ready MLflow app: %s", ready_app.arn)
                return ready_app.arn
        
        # Create new app
        new_app = _create_mlflow_app(sagemaker_session)
        if new_app:
            logger.info("Created new MLflow app: %s", new_app.arn)
            return new_app.arn
        
        logger.warning("Failed to create MLflow app. MLflow tracking disabled.")
        return None
        
    except Exception as e:
        logger.error("Error resolving MLflow resource ARN: %s", e)
        return None


def _create_mlflow_app(sagemaker_session) -> Optional[MlflowApp]:
    """Create a new MLflow app with minimal configuration."""
    try:
        app_name = f"finetune-mlflow-{int(time.time())}"
        account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
        region = sagemaker_session.boto_session.region_name
        artifact_store_uri = f"s3://sagemaker-{region}-{account_id}/mlflow-artifacts"
        role_arn = TrainDefaults.get_role(role=None, sagemaker_session=sagemaker_session)
        
        # Ensure S3 bucket and prefix exist
        s3_client = sagemaker_session.boto_session.client('s3')
        bucket_name = f"sagemaker-{region}-{account_id}"
        
        try:
            # Check if prefix exists
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="mlflow-artifacts/", MaxKeys=1)
            if 'Contents' not in response:
                s3_client.put_object(Bucket=bucket_name, Key="mlflow-artifacts/")
        except s3_client.exceptions.NoSuchBucket:
            # Bucket doesn't exist, create bucket and prefix
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            s3_client.put_object(Bucket=bucket_name, Key="mlflow-artifacts/")
        
        new_app = MlflowApp.create(
            name=app_name,
            artifact_store_uri=artifact_store_uri,
            role_arn=role_arn,
            account_default_status="DISABLED",
            session=sagemaker_session.boto_session,
            region=region
        )
        
        # Wait for app to reach Created/Updated state
        max_wait_time = 600  # 10 minutes
        poll_interval = 10   # 10 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            new_app.refresh()
            if new_app.status in ["Created", "Updated"]:
                return new_app
            elif new_app.status in ["Failed", "Stopped"]:
                # Get detailed error from MLflow app
                error_msg = f"MLflow app creation failed with status: {new_app.status}"
                if hasattr(new_app, 'failure_reason') and new_app.failure_reason:
                    error_msg += f". Reason: {new_app.failure_reason}"
                raise RuntimeError(error_msg)
            time.sleep(poll_interval)
        
        # Timeout case - get current status and any error details
        new_app.refresh()
        error_msg = f"MLflow app creation failed. Current status: {new_app.status}"
        if hasattr(new_app, 'failure_reason') and new_app.failure_reason:
            error_msg += f". Reason: {new_app.failure_reason}"
        raise RuntimeError(error_msg)
            
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


def _extract_evaluator_arn(evaluator, param_name: str = "custom_reward_function"):
    """Extract and validate evaluator ARN from string or Evaluator object."""
    if isinstance(evaluator, str):
        _validate_evaluator_arn(evaluator, param_name)
        return evaluator
    else:
        # It's an Evaluator object, extract ARN (already valid)
        return evaluator.arn


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
                                         hub_name: str = "SageMakerPublicHub") -> tuple:
    """Get fine-tuning options and model ARN for given customization technique.
    
    Returns:
        tuple: (FineTuningOptions, model_arn, is_gated_model)
    """
    
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
            raise ValueError(f"No recipes found for customization technique: {customization_technique}")
        
        # Filter recipes that have SmtjRecipeTemplateS3Uri key
        recipes_with_template = [r for r in matching_recipes if r.get("SmtjRecipeTemplateS3Uri")]
        
        if not recipes_with_template:
            raise ValueError(f"No recipes found with Smtj for technique: {customization_technique}")

        # Select recipe based on training type
        recipe = None
        if (isinstance(training_type, TrainingType) and training_type == TrainingType.LORA) or training_type == "LORA":
            recipe = next((r for r in recipes_with_template if r.get("Peft")), None)
        elif (isinstance(training_type, TrainingType) and training_type == TrainingType.FULL) or training_type == "FULL":
            recipe = next((r for r in recipes_with_template if not r.get("Peft")), None)

        if not recipe:
            raise ValueError(f"No recipes found with Smtj for technique: {customization_technique},training_type:{training_type}")

        elif recipe and recipe.get("SmtjOverrideParamsS3Uri"):
            s3_uri = recipe["SmtjOverrideParamsS3Uri"]
            s3 = boto3.client("s3")
            bucket, key = s3_uri.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            options_dict = json.loads(obj["Body"].read())
            return FineTuningOptions(options_dict), model_arn, is_gated_model
        else:
            return FineTuningOptions({}), model_arn, is_gated_model
            
    except Exception as e:
        logger.error("Exception getting fine-tuning options: %s", e)


def _create_input_channels(dataset: str, content_type: Optional[str] = None, 
                         input_compression_type: Optional[str] = None,
                         record_wrapper_type: Optional[str] = None,
                         input_mode: Optional[str] = None,
                         enable_ffm: Optional[bool] = None):
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
        input_mode=input_mode,
        enable_ffm=enable_ffm,
    )
    channels.append(channel)
    
    return channels


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
            # Validate region availability
            if region_name:
                _validate_model_region_availability(model, region_name)
            return model, model
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


def _create_output_config(sagemaker_session,s3_output_path=None, kms_key_id=None):
    """Create output data configuration with default S3 path if needed.
    
    Args:
        s3_output_path: S3 output path (if None, generates default path)
        sagemaker_session: SageMaker session for generating default path
        kms_key_id: Optional KMS key ID for encryption
    
    Returns:
        OutputDataConfig object
    """

    # Use default S3 output path if none provided
    if s3_output_path is None:
        s3_output_path = _get_default_s3_output_path(sagemaker_session)
    
    # Validate S3 path exists
    _validate_s3_path_exists(s3_output_path, sagemaker_session)

    return OutputDataConfig(
        s3_output_path=s3_output_path,
        kms_key_id=kms_key_id,
    )


def _convert_input_data_to_channels(input_data_config ):
    """Convert InputData objects to Channel objects with S3 and dataset ARN support.
    
    Args:
        input_data_config: List of InputData objects
    
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
                    "s3_data_type": "S3Prefix",
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
    """Validate and resolve model_package_group_name from ModelPackage if needed."""
    # If model_package_group_name is already provided, return it as-is
    if model_package_group_name:
        return model_package_group_name
    
    # Try to resolve from ModelPackage if available
    if isinstance(model, ModelPackage):
        return model.model_package_group_name
    
    # Only validate if model_package_group_name is None and model is not ModelPackage
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


def _validate_s3_path_exists(s3_path: str, sagemaker_session):
    """Validate S3 path and create bucket/prefix if they don't exist."""
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
                s3_client.put_object(Bucket=bucket_name, Key=prefix, Body=b'')
                
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
