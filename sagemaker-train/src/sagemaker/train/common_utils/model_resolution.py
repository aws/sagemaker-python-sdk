"""
Internal utilities for resolving model information from various input types.

This module provides common functionality for resolving model metadata from:
- JumpStart model IDs (strings like "llama3-2-1b-instruct")
- ModelPackage objects or ARNs (fine-tuned models)
"""

import os
import json
import boto3
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re


class _ModelType(Enum):
    """Internal enum for model type classification."""
    JUMPSTART = "jumpstart"
    FINE_TUNED = "fine_tuned"


@dataclass
class _ModelInfo:
    """
    Internal dataclass containing resolved model information.
    
    Attributes:
        base_model_name: Human-readable model name
        base_model_arn: ARN of the base model
        source_model_package_arn: ARN of source model package (None for JumpStart models)
        model_type: Type of model (JUMPSTART or FINE_TUNED)
        hub_content_name: Name in the hub (for JumpStart models)
        additional_metadata: Any additional metadata extracted during resolution
    """
    base_model_name: str
    base_model_arn: str
    source_model_package_arn: Optional[str]
    model_type: _ModelType
    hub_content_name: Optional[str]
    additional_metadata: Dict[str, Any]


class _ModelResolver:
    """
    Internal utility class for resolving model information.
    
    Handles resolution of model metadata from both JumpStart model IDs 
    and fine-tuned ModelPackage objects/ARNs.
    """
    
    DEFAULT_HUB_NAME = "SageMakerPublicHub"
    
    def __init__(self, sagemaker_session=None):
        """
        Initialize the resolver.
        
        Args:
            sagemaker_session: SageMaker session to use for API calls.
                             If None, will be created with beta endpoint if configured.
        """
        self.sagemaker_session = sagemaker_session
        self._beta_endpoint = os.environ.get('SAGEMAKER_ENDPOINT')
    
    def resolve_model_info(
        self, 
        base_model: Union[str, 'ModelPackage'],
        hub_name: Optional[str] = None
    ) -> _ModelInfo:
        """
        Resolve model information from various input types.
        
        Args:
            base_model: Either a JumpStart model ID (str) or ModelPackage object/ARN
            hub_name: Optional hub name for JumpStart models (defaults to SageMakerPublicHub)
        
        Returns:
            _ModelInfo: Resolved model information
            
        Raises:
            ValueError: If model input is invalid or resolution fails
        """
        # Check if it's a string first
        if isinstance(base_model, str):
            # Check if it's a model package ARN or JumpStart model ID
            if base_model.startswith("arn:aws:sagemaker:") and ":model-package/" in base_model:
                return self._resolve_model_package_arn(base_model)
            else:
                return self._resolve_jumpstart_model(base_model, hub_name or self.DEFAULT_HUB_NAME)
        else:
            # Not a string, so assume it's a ModelPackage object
            # Check if it has the expected attributes of a ModelPackage
            if hasattr(base_model, 'model_package_arn') or hasattr(base_model, 'inference_specification'):
                return self._resolve_model_package_object(base_model)
            else:
                raise ValueError(
                    f"base_model must be a string (JumpStart model ID or ModelPackage ARN) "
                    f"or ModelPackage object, got {type(base_model)}"
                )
    
    def _resolve_jumpstart_model(self, model_id: str, hub_name: str) -> _ModelInfo:
        """
        Resolve JumpStart model information from Hub API.
        
        Args:
            model_id: JumpStart model identifier
            hub_name: Hub name to query
            
        Returns:
            _ModelInfo: Resolved model information
        """
        from sagemaker.core.resources import HubContent
        
        session = self._get_session()
        
        try:
            hub_content = HubContent.get(
                hub_name=hub_name,
                hub_content_type="Model",
                hub_content_name=model_id,
                session=session.boto_session,
                region=session.boto_session.region_name
            )
            
            # Parse additional metadata from hub content document
            additional_metadata = {}
            if hub_content.hub_content_document:
                try:
                    additional_metadata = json.loads(hub_content.hub_content_document)
                except json.JSONDecodeError:
                    pass
            
            return _ModelInfo(
                base_model_name=model_id,
                base_model_arn=hub_content.hub_content_arn,
                source_model_package_arn=None,
                model_type=_ModelType.JUMPSTART,
                hub_content_name=model_id,
                additional_metadata=additional_metadata
            )
            
        except Exception as e:
            raise ValueError(
                f"Failed to resolve JumpStart model '{model_id}' from hub '{hub_name}': {e}"
            )
    
    def _resolve_model_package_object(self, model_package: 'ModelPackage') -> _ModelInfo:
        """
        Resolve model information from ModelPackage object.
        
        Args:
            model_package: ModelPackage object
            
        Returns:
            _ModelInfo: Resolved model information
            
        Raises:
            ValueError: If model package doesn't have base_model metadata
        """
        # Extract base model info from inference specification
        base_model_name = None
        base_model_arn = None
        hub_content_name = None
        
        # Check if inference specification exists
        if not hasattr(model_package, 'inference_specification') or not model_package.inference_specification:
            raise ValueError(
                f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                f"The provided model package (ARN: {getattr(model_package, 'model_package_arn', 'unknown')}) "
                f"does not have an inference_specification."
            )
        
        # Check if containers exist
        if not model_package.inference_specification.containers:
            raise ValueError(
                f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                f"The provided model package (ARN: {getattr(model_package, 'model_package_arn', 'unknown')}) "
                f"does not have any containers in its inference_specification."
            )
        
        container = model_package.inference_specification.containers[0]
        
        # Try to get base model information - this is critical
        if hasattr(container, 'base_model') and container.base_model:
            if hasattr(container.base_model, 'hub_content_name'):
                hub_content_name = container.base_model.hub_content_name
                base_model_name = hub_content_name
            if hasattr(container.base_model, 'hub_content_arn'):
                base_model_arn = container.base_model.hub_content_arn
        
        # If we couldn't extract base model ARN, this is not a supported model package
        if not base_model_arn:
            raise ValueError(
                f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                f"The provided model package (ARN: {getattr(model_package, 'model_package_arn', 'unknown')}) "
                f"does not have base_model metadata in its inference_specification.containers[0]. "
                f"Please ensure the model was created using SageMaker's fine-tuning capabilities."
            )
        
        # If we couldn't extract base model name, use package name as fallback
        if not base_model_name:
            if hasattr(model_package, 'model_package_arn'):
                arn_parts = model_package.model_package_arn.split('/')
                if len(arn_parts) >= 2:
                    base_model_name = arn_parts[-2]  # Get the group name
                else:
                    base_model_name = getattr(model_package, 'model_package_name', 'unknown')
            else:
                base_model_name = getattr(model_package, 'model_package_name', 'unknown')
            
        return _ModelInfo(
            base_model_name=base_model_name,
            base_model_arn=base_model_arn,
            source_model_package_arn=getattr(model_package, 'model_package_arn', None),
            model_type=_ModelType.FINE_TUNED,
            hub_content_name=hub_content_name,
            additional_metadata={}
        )
    
    def _resolve_model_package_arn(self, model_package_arn: str) -> _ModelInfo:
        """
        Resolve model information from ModelPackage ARN.
        
        Args:
            model_package_arn: ARN of the model package
            
        Returns:
            _ModelInfo: Resolved model information
        """
        session = self._get_session()
        
        try:
            # Validate ARN format
            self._validate_model_package_arn(model_package_arn)
            
            # TODO: Switch to sagemaker_core ModelPackage.get() once the bug is fixed
            # Currently, ModelPackage.get() has a Pydantic validation issue where 
            # the transform() function doesn't include model_package_name in the response,
            # causing: "1 validation error for ModelPackage - model_package_name: Field required"
            # Using boto3 directly as a workaround.
            
            # Use the sagemaker client from the session (which has the correct endpoint configured)
            sm_client = session.sagemaker_client if hasattr(session, 'sagemaker_client') else session.boto_session.client('sagemaker')
            response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
            
            # Extract base model info from response
            base_model_name = None
            base_model_arn = None
            hub_content_name = None
            
            # Check inference specification
            if 'InferenceSpecification' not in response:
                raise ValueError(
                    f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                    f"The provided model package (ARN: {model_package_arn}) "
                    f"does not have an inference_specification."
                )
            
            inf_spec = response['InferenceSpecification']
            if 'Containers' not in inf_spec or len(inf_spec['Containers']) == 0:
                raise ValueError(
                    f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                    f"The provided model package (ARN: {model_package_arn}) "
                    f"does not have any containers in its inference_specification."
                )
            
            container = inf_spec['Containers'][0]
            
            # Extract base model info
            if 'BaseModel' not in container:
                raise ValueError(
                    f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                    f"The provided model package (ARN: {model_package_arn}) "
                    f"does not have base_model metadata in its inference_specification.containers[0]. "
                    f"Please ensure the model was created using SageMaker's fine-tuning capabilities."
                )
            
            base_model_info = container['BaseModel']
            hub_content_name = base_model_info.get('HubContentName')
            hub_content_version = base_model_info.get('HubContentVersion')
            base_model_arn = base_model_info.get('HubContentArn')
            
            # If HubContentArn is None, construct it from HubContentName and version
            # This handles cases where the API doesn't return the full ARN
            if not base_model_arn and hub_content_name and hub_content_version:
                # Extract region from model_package_arn
                arn_parts = model_package_arn.split(':')
                if len(arn_parts) >= 4:
                    region = arn_parts[3]
                    # Construct hub content ARN for SageMaker public hub
                    base_model_arn = f"arn:aws:sagemaker:{region}:aws:hub-content/SageMakerPublicHub/Model/{hub_content_name}/{hub_content_version}"
            
            if not base_model_arn:
                raise ValueError(
                    f"NotSupported: Evaluation is only supported for model packages customized by SageMaker's fine-tuning flows. "
                    f"The provided model package (ARN: {model_package_arn}) "
                    f"does not have base_model metadata with HubContentArn or sufficient information to construct it. "
                    f"Please ensure the model was created using SageMaker's fine-tuning capabilities."
                )
            
            # Use hub_content_name as base_model_name
            base_model_name = hub_content_name if hub_content_name else response.get('ModelPackageGroupName', 'unknown')
            
            return _ModelInfo(
                base_model_name=base_model_name,
                base_model_arn=base_model_arn,
                source_model_package_arn=model_package_arn,
                model_type=_ModelType.FINE_TUNED,
                hub_content_name=hub_content_name,
                additional_metadata={}
            )
            
        except ValueError:
            # Re-raise ValueError as-is (our custom error messages)
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to resolve model package ARN '{model_package_arn}': {e}"
            )
    
    def _validate_model_package_arn(self, arn: str) -> bool:
        """
        Validate ModelPackage ARN format.
        
        Args:
            arn: ARN to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If ARN format is invalid
        """
        pattern = r'^arn:aws[a-z\-]*:sagemaker:[a-z0-9\-]+:\d{12}:model-package/.*$'
        if not re.match(pattern, arn):
            raise ValueError(
                f"Invalid ModelPackage ARN format: {arn}. "
                f"Expected format matching regex: {pattern}"
            )
        return True
    
    def _get_session(self):
        """
        Get or create SageMaker session with beta endpoint support.
        
        Returns:
            SageMaker session
        """
        if self.sagemaker_session:
            return self.sagemaker_session
        
        from sagemaker.core.helper.session_helper import Session
        
        # Check for beta endpoint in environment variable
        if self._beta_endpoint:
            sm_client = boto3.client(
                'sagemaker',
                endpoint_url=self._beta_endpoint,
                region_name=os.environ.get('AWS_REGION', 'us-west-2')
            )
            return Session(sagemaker_client=sm_client)
        
        # Default session
        return Session()


def _resolve_base_model(
    base_model: Union[str, 'ModelPackage'],
    sagemaker_session=None,
    hub_name: Optional[str] = None
) -> _ModelInfo:
    """
    Convenience function to resolve model information.
    
    This is the main entry point for model resolution. It handles both:
    - JumpStart model IDs (e.g., "llama3-2-1b-instruct")
    - ModelPackage objects or ARNs (fine-tuned models)
    
    Args:
        base_model: Either a JumpStart model ID (str) or ModelPackage object/ARN
        sagemaker_session: Optional SageMaker session for API calls
        hub_name: Optional hub name for JumpStart models
        
    Returns:
        _ModelInfo: Resolved model information containing base_model_name, 
                   base_model_arn, and other metadata
                   
    Raises:
        ValueError: If model input is invalid or resolution fails
        
    Example:
        >>> # Resolve JumpStart model
        >>> info = _resolve_base_model("llama3-2-1b-instruct")
        >>> print(info.base_model_name)  # "llama3-2-1b-instruct"
        >>> print(info.base_model_arn)   # "arn:aws:sagemaker:..."
        
        >>> # Resolve from ModelPackage ARN
        >>> info = _resolve_base_model("arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1")
        >>> print(info.source_model_package_arn)  # Original ARN
        >>> print(info.base_model_arn)            # Base model ARN
    """
    resolver = _ModelResolver(sagemaker_session)
    return resolver.resolve_model_info(base_model, hub_name)
