"""
Common utilities for fetching recipe metadata and override parameters from JumpStart Hub.

This module provides reusable functionality for retrieving evaluation recipe configurations
and inference parameters from SageMaker Hub content.
"""

import json
import logging
from typing import Dict, Any, Optional
import boto3

from sagemaker.core.resources import HubContent

logger = logging.getLogger(__name__)


def _is_nova_model(model_id: str) -> bool:
    """Check if the model ID is a Nova model.
    
    Args:
        model_id: The model identifier/hub content name
        
    Returns:
        True if the model ID contains "nova" (case-insensitive), False otherwise
        
    Example:
        >>> _is_nova_model("amazon-nova-pro")
        True
        >>> _is_nova_model("meta-textgeneration-llama-3-2-1b-instruct")
        False
    """
    return "nova" in model_id.lower()


def _get_hub_content_metadata(
    hub_name: str,
    hub_content_name: str,
    hub_content_type: str = "Model",
    region: Optional[str] = None,
    session: Optional[Any] = None
) -> Dict[str, Any]:
    """Internal: Get hub content metadata using SageMaker Core HubContent.get
    
    Args:
        hub_name: Name of the SageMaker Hub (e.g., "SageMakerPublicHub")
        hub_content_name: Name of the hub content (e.g., model name)
        hub_content_type: Type of hub content (default: "Model")
        region: AWS region (optional)
        session: Boto3 session (optional)
    
    Returns:
        Dict containing hub content metadata including RecipeCollection
        
    Example:
        >>> metadata = get_hub_content_metadata(
        ...     hub_name="SageMakerPublicHub",
        ...     hub_content_name="meta-textgeneration-llama-3-2-1b-instruct",
        ...     hub_content_type="Model"
        ... )
        >>> print(metadata['HubContentName'])
    """
    hub_content = HubContent.get(
        hub_name=hub_name,
        hub_content_type=hub_content_type,
        hub_content_name=hub_content_name,
        region=region,
        session=session
    )
    
    # Convert to dict for easier access
    hub_content_dict = hub_content.__dict__
    
    # Parse HubContentDocument if it's a JSON string
    if 'hub_content_document' in hub_content_dict:
        hub_content_document = hub_content_dict['hub_content_document']
        if isinstance(hub_content_document, str):
            try:
                hub_content_dict['hub_content_document'] = json.loads(hub_content_document)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, leave it as is
                pass
    
    return hub_content_dict


def _download_s3_json(s3_uri: str, region: Optional[str] = None) -> Dict[str, Any]:
    """Internal: Download and parse JSON file from S3
    
    Args:
        s3_uri: S3 URI of the JSON file (e.g., s3://bucket/path/file.json)
        region: AWS region (optional)
    
    Returns:
        Dict containing parsed JSON content
        
    Example:
        >>> params = download_s3_json("s3://bucket/path/override_params.json")
        >>> print(params)
    """
    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    s3_path = s3_uri[5:]  # Remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    # Download from S3
    s3_client = boto3.client('s3', region_name=region)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    
    return json.loads(content)


def _find_evaluation_recipe(
    recipe_collection: list,
    recipe_type: str = "Evaluation",
    evaluation_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Internal: Find evaluation recipe in recipe collection
    
    Args:
        recipe_collection: List of recipes from hub content document
        recipe_type: Type of recipe to find (default: "Evaluation")
        evaluation_type: Optional evaluation type filter (e.g., "DeterministicEvaluation")
    
    Returns:
        Recipe dict if found, None otherwise
        
    Example:
        >>> # Find any evaluation recipe
        >>> recipe = find_evaluation_recipe(
        ...     recipe_collection=metadata['HubContentDocument']['RecipeCollection'],
        ...     recipe_type="Evaluation"
        ... )
        >>> 
        >>> # Find deterministic evaluation recipe for benchmarks
        >>> recipe = find_evaluation_recipe(
        ...     recipe_collection=metadata['HubContentDocument']['RecipeCollection'],
        ...     recipe_type="Evaluation",
        ...     evaluation_type="DeterministicEvaluation"
        ... )
        >>> print(recipe['Name'])
    """
    for recipe in recipe_collection:
        if recipe.get('Type') == recipe_type:
            # If evaluation_type is specified, also check that
            if evaluation_type is not None:
                if recipe.get('EvaluationType') == evaluation_type:
                    return recipe
            else:
                return recipe
    return None


def _get_evaluation_override_params(
    hub_content_name: str,
    hub_name: str = "SageMakerPublicHub",
    hub_content_type: str = "Model",
    evaluation_type: str = "DeterministicEvaluation",
    region: Optional[str] = None,
    session: Optional[Any] = None
) -> Dict[str, Any]:
    """Internal: Get evaluation recipe override parameters from hub content
    
    This function retrieves the hub content metadata, finds the evaluation recipe
    (filtered by EvaluationType for deterministic benchmarks), downloads the override 
    parameters from S3, and returns them.
    
    Args:
        hub_content_name: Name of the hub content (e.g., model name)
        hub_name: Name of the SageMaker Hub (default: "SageMakerPublicHub")
        hub_content_type: Type of hub content (default: "Model")
        evaluation_type: Evaluation type filter (default: "DeterministicEvaluation")
        region: AWS region (optional)
        session: Boto3 session (optional)
    
    Returns:
        Dict containing override parameters with structure:
        {
                'max_new_tokens': {'default': 8192, ...},
                'temperature': {'default': 0, ...},
                'top_k': {'default': -1, ...},
                'top_p': {'default': 1.0, ...},
                ...
        }
            
    Raises:
        ValueError: If evaluation recipe is not found or SmtjOverrideParamsS3Uri is missing
        
    Example:
        >>> # For benchmark evaluation (DeterministicEvaluation)
        >>> params = get_evaluation_override_params(
        ...     hub_content_name="meta-textgeneration-llama-3-2-1b-instruct"
        ... )
        >>> max_tokens = params['max_new_tokens']['default']
        >>> print(max_tokens)  # 8192
    """
    # Get hub content metadata
    logger.info(f"Fetching hub content metadata for {hub_content_name} from {hub_name}")
    hub_metadata = _get_hub_content_metadata(
        hub_name=hub_name,
        hub_content_name=hub_content_name,
        hub_content_type=hub_content_type,
        region=region,
        session=session
    )
    
    # Extract recipe collection from hub content document
    hub_content_document = hub_metadata.get('hub_content_document', {})
    recipe_collection = hub_content_document.get('RecipeCollection', [])
    
    if not recipe_collection:
        raise ValueError(
            f"Unsupported Base Model. No recipes found in hub content '{hub_content_name}'. "
            f"RecipeCollection is empty or missing."
        )
    
    # Find evaluation recipe with specific evaluation type
    logger.info(f"Searching for evaluation recipe with Type='Evaluation' and EvaluationType='{evaluation_type}'")
    evaluation_recipe = _find_evaluation_recipe(
        recipe_collection, 
        recipe_type="Evaluation",
        evaluation_type=evaluation_type
    )
    
    if not evaluation_recipe:
        raise ValueError(
            f"Model '{hub_content_name}' is not supported for evaluation. "
            f"The model does not have an evaluation recipe with EvaluationType='{evaluation_type}'. "
            f"Please use a model that supports evaluation or contact AWS support for assistance."
        )
    
    # Get SmtjOverrideParamsS3Uri
    override_params_s3_uri = evaluation_recipe.get('SmtjOverrideParamsS3Uri')
    
    if not override_params_s3_uri:
        raise ValueError(
            f"Model '{hub_content_name}' is not supported for evaluation. "
            f"The evaluation recipe is missing required configuration parameters. "
            f"Please use a model that supports evaluation or contact AWS support for assistance."
        )
    
    # Download override parameters from S3
    logger.info(f"Downloading override parameters from {override_params_s3_uri}")
    override_params = _download_s3_json(override_params_s3_uri, region=region)
    
    return override_params


def _extract_eval_override_options(
    override_params: Dict[str, Any],
    param_names: Optional[list] = None,
    return_full_spec: bool = False
) -> Dict[str, Any]:
    """Internal: Extract evaluation override options from override parameters JSON
    
    Extracts evaluation override options from the parameters JSON. Can return either
    just the default values as strings (for pipeline templates) or the full parameter
    specifications (for FineTuningOptions objects).
    
    The override_params structure has parameters at the root level, where each parameter
    has a 'default' key and optionally type, min, max, enum, etc.:
    {
      "max_new_tokens": {"default": 8192, "type": "integer", "min": 1, "max": 16384, ...},
      "temperature": {"default": 0, "type": "integer", "min": 0, "max": 2, ...},
      ...
    }
    
    Args:
        override_params: The override parameters JSON from _get_evaluation_override_params()
        param_names: Optional list of parameter names to extract. 
                    If None, extracts common evaluation override options:
                    ['max_new_tokens', 'temperature', 'top_k', 'top_p', 'aggregation', 'postprocessing', 'max_model_len']
        return_full_spec: If True, returns full parameter specifications (dict with type, min, max, etc.).
                         If False, returns only default values as strings.
    
    Returns:
        Dict mapping parameter names to either:
        - Their default values as strings (if return_full_spec=False)
        - Their full specifications as dicts (if return_full_spec=True)
        
    Example:
        >>> override_params = _get_evaluation_override_params("meta-textgeneration-llama-3-2-1b-instruct")
        >>> 
        >>> # Get default values only (for pipeline templates)
        >>> params = _extract_eval_override_options(override_params)
        >>> print(params)
        >>> # {'max_new_tokens': '8192', 'temperature': '0', ...}
        >>> 
        >>> # Get full specifications (for FineTuningOptions)
        >>> specs = _extract_eval_override_options(override_params, return_full_spec=True)
        >>> print(specs)
        >>> # {'max_new_tokens': {'default': 8192, 'type': 'integer', 'min': 1, ...}, ...}
    """
    if param_names is None:
        param_names = ['max_new_tokens', 'temperature', 'top_k', 'top_p', 'aggregation', 'postprocessing', 'max_model_len']
    
    extracted_params = {}
    for param_name in param_names:
        # Parameters are at root level in override_params
        param_config = override_params.get(param_name, {})
        
        if isinstance(param_config, dict) and 'default' in param_config:
            if return_full_spec:
                # Return full parameter specification
                extracted_params[param_name] = param_config.copy()
            else:
                # Return only default value as string
                extracted_params[param_name] = str(param_config['default'])
        else:
            logger.debug(
                f"Parameter '{param_name}' not found in override_params or has no default value. "
                f"Will use fallback value if needed."
            )
    
    return extracted_params
