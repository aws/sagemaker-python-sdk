"""Base evaluator module for SageMaker Model Evaluation.

This module provides the base class for all evaluators in the SageMaker Model Evaluation Module.
It handles common functionality such as model resolution, MLflow integration, and AWS resource
configuration.
"""

from __future__ import absolute_import

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, validator

from sagemaker.core.common_utils import TagsDict
from sagemaker.core.resources import ModelPackageGroup, ModelPackage
from sagemaker.core.shapes import VpcConfig

if TYPE_CHECKING:
    from sagemaker.core.helper.session_helper import Session

from sagemaker.train.base_trainer import BaseTrainer
# Module-level logger
_logger = logging.getLogger(__name__)

# Regex patterns for ARN validation
_MODEL_PACKAGE_GROUP_ARN_PATTERN = (
    r"^arn:aws(-cn|-us-gov|-iso-f)?:sagemaker:[a-z0-9\-]{9,16}:[0-9]{12}:"
    r"model-package-group/[\S]{1,2048}$"
)
_MLFLOW_ARN_PATTERN = (
    r"^arn:aws[a-z\-]*:sagemaker:[a-z0-9\-]*:[0-9]{12}:(mlflow-tracking-server|mlflow-app)/.*$"
)
_MODEL_PACKAGE_ARN_PATTERN = (
    r"^arn:aws[a-z\-]*:sagemaker:([a-z0-9\-]+):([0-9]{12}):model-package/([^/]+)/[^/]+$"
)
_HUB_CONTENT_DATASET_ARN_PATTERN = r'arn:.*:hub-content/.*/DataSet/.*'
_S3_URI_PATTERN = r's3://.*'


class BaseEvaluator(BaseModel):
    """Base class for SageMaker model evaluators.

    Provides common functionality for all evaluators including model resolution,
    MLflow integration, and AWS resource configuration. Subclasses must implement
    the evaluate() method.

    Attributes:
        region (Optional[str]): AWS region for evaluation jobs. If not provided, will use
            SAGEMAKER_REGION env var or default region.
        sagemaker_session (Optional[Any]): SageMaker session object. If not provided, a default
            session will be created automatically.
        model (Union[str, Any]): Model for evaluation. Can be:
            - JumpStart model ID (str): e.g., 'llama3-2-1b-instruct'
            - ModelPackage object: A fine-tuned model package
            - ModelPackage ARN (str): e.g., 'arn:aws:sagemaker:region:account:model-package/name/version'
            - BaseTrainer object: A completed training job (i.e., it must have _latest_training_job with output_model_package_arn populated)
        base_eval_name (Optional[str]): Optional base name for evaluation jobs. This name is used
            as the PipelineExecutionDisplayName when creating the SageMaker pipeline execution.
            The actual display name will be "{base_eval_name}-{timestamp}". This parameter can
            be used to cross-reference the pipeline execution ARN with a human-readable display
            name in the SageMaker console. If not provided, a unique name will be generated
            automatically in the format "eval-{model_name}-{uuid}".
        s3_output_path (str): S3 location for evaluation outputs. Required.
        mlflow_resource_arn (Optional[str]): MLflow resource ARN for experiment tracking.
            Optional. If not provided, the system will attempt to resolve it using the default
            MLflow app experience (checks domain match, account default, or creates a new app).
            Supported formats:
            - MLflow tracking server: arn:aws:sagemaker:region:account:mlflow-tracking-server/name
            - MLflow app: arn:aws:sagemaker:region:account:mlflow-app/app-id
        mlflow_experiment_name (Optional[str]): Optional MLflow experiment name for tracking
            evaluation runs.
        mlflow_run_name (Optional[str]): Optional MLflow run name for tracking individual
            evaluation executions.
        networking (Optional[VpcConfig]): VPC configuration for evaluation jobs. Accepts a
            sagemaker_core.shapes.VpcConfig object with security_group_ids and subnets attributes.
            When provided, evaluation jobs will run within the specified VPC for enhanced security
            and access to private resources.
        kms_key_id (Optional[str]): AWS KMS key ID for encrypting output data. When provided,
            evaluation job outputs will be encrypted using this KMS key for enhanced data security.
        model_package_group (Optional[Union[str, ModelPackageGroup]]): Model package group. Accepts:
            1. ARN string (e.g., 'arn:aws:sagemaker:region:account:model-package-group/name')
            2. ModelPackageGroup object (ARN will be extracted from model_package_group_arn attribute)
            3. Model package group name string (will fetch the object and extract ARN)
            Required when model is a JumpStart model ID. Optional when model is a ModelPackage
            ARN/object (will be inferred automatically).
    """
    
    region: Optional[str] = None
    sagemaker_session: Optional[Any] = None
    model: Union[str, BaseTrainer, ModelPackage]
    base_eval_name: Optional[str] = None
    s3_output_path: str
    mlflow_resource_arn: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    networking: Optional[VpcConfig] = None
    kms_key_id: Optional[str] = None
    model_package_group: Optional[Union[str, ModelPackageGroup]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @staticmethod
    def _validate_and_resolve_dataset(v: Any) -> str:
        """Validate and resolve dataset to string (S3 URI or ARN).
        
        This static method provides common dataset validation logic that can be
        reused by all evaluator subclasses in their `_resolve_dataset` validators.
        
        Args:
            v: Dataset value to validate. Can be:
                - DataSet object with 'arn' attribute
                - String (S3 URI or hub-content DataSet ARN)
        
        Returns:
            str: Validated dataset string (S3 URI or ARN)
        
        Raises:
            ValueError: If dataset format is invalid
        
        Example usage in subclass validator:
            @validator('dataset', pre=True)
            def _resolve_dataset(cls, v):
                return BaseEvaluator._validate_and_resolve_dataset(v)
        """
        # Check if it's a DataSet object by checking for 'arn' attribute
        if hasattr(v, 'arn'):
            _logger.info(f"Resolving DataSet object to ARN: {v.arn}")
            dataset_str = v.arn
        else:
            dataset_str = v
        
        # Validate the resolved dataset string matches expected patterns
        if not isinstance(dataset_str, str):
            raise ValueError(
                f"Dataset must be a string (S3 URI or hub-content DataSet ARN) or a DataSet object. "
                f"Got {type(dataset_str).__name__}"
            )
        
        # Check if it matches hub-content DataSet ARN pattern or S3 URI pattern
        is_hub_content_arn = re.match(_HUB_CONTENT_DATASET_ARN_PATTERN, dataset_str)
        is_s3_uri = re.match(_S3_URI_PATTERN, dataset_str)
        
        if not (is_hub_content_arn or is_s3_uri):
            raise ValueError(
                f"Invalid dataset format: '{dataset_str}'. "
                f"Dataset must be either:\n"
                f"  1. A hub-content DataSet ARN matching pattern: arn:*:hub-content/*/DataSet/*\n"
                f"     Example: arn:aws:sagemaker:us-east-1:123456789012:hub-content/AIRegistry/DataSet/my-dataset/1.0\n"
                f"  2. An S3 URI matching pattern: s3://*\n"
                f"     Example: s3://my-bucket/path/to/dataset.jsonl"
            )
        
        return dataset_str
    
    @validator('mlflow_resource_arn', pre=True, always=True)
    def _resolve_mlflow_arn(cls, v, values):
        """Resolve MLflow resource ARN using default experience logic if not provided."""
        from ..common_utils.finetune_utils import _resolve_mlflow_resource_arn
        
        # Get sagemaker_session from values
        sagemaker_session = values.get('sagemaker_session')
        if sagemaker_session is None:
            # If session is not available yet during validation, return as-is
            # It will be resolved later in the evaluate() method
            return v
        
        # Resolve MLflow ARN using the utility function
        resolved_arn = _resolve_mlflow_resource_arn(sagemaker_session, v)
        if resolved_arn:
            _logger.info(f"Resolved MLflow resource ARN: {resolved_arn}")
        else:
            _logger.warning("Could not resolve MLflow resource ARN. MLflow tracking will be disabled.")
        
        return resolved_arn
    
    @validator('model_package_group', pre=True)
    def _validate_and_resolve_model_package_group(cls, v, values):
        r"""Validate and resolve model_package_group to ARN string.
        
        Accepts three input types:
        1. ARN string matching pattern: arn:aws(-cn|-us-gov|-iso-f)?:sagemaker:[a-z0-9\-]{9,16}:[0-9]{12}:model-package-group/[\S]{1,2048}
        2. ModelPackageGroup object - extracts ARN from object.model_package_group_arn
        3. Model package group name string - fetches object via ModelPackageGroup.get() and extracts ARN
        
        Args:
            v: Input value (ARN, object, or name)
            values: Dictionary of already-validated fields
            
        Returns:
            Optional[str]: Resolved model package group ARN or None
            
        Raises:
            ValueError: If ARN format is invalid or object/name resolution fails
        """
        if v is None:
            return None
        
        # Case 1: Already an ARN string
        if isinstance(v, str):
            # Check if it matches ARN pattern
            if re.match(_MODEL_PACKAGE_GROUP_ARN_PATTERN, v):
                _logger.info(f"Model package group provided as ARN: {v}")
                return v
            
            # Case 3: Treat as model package group name - fetch the object
            try:
                _logger.info(f"Model package group provided as name: {v}. Fetching ModelPackageGroup object...")
                
                # Get session for region
                session = values.get('sagemaker_session')
                region = values.get('region')
                if not region and session:
                    region = (session.boto_region_name 
                             if hasattr(session, 'boto_region_name') 
                             else 'us-west-2')
                
                # Fetch the object
                obj = ModelPackageGroup.get(
                    model_package_group_name=v,
                    region=region
                )
                
                # Extract ARN
                if hasattr(obj, 'model_package_group_arn'):
                    arn = obj.model_package_group_arn
                    _logger.info(f"Resolved model package group name '{v}' to ARN: {arn}")
                    return arn
                else:
                    raise ValueError(
                        f"ModelPackageGroup object for name '{v}' does not have model_package_group_arn attribute"
                    )
                    
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve model package group name '{v}': {e}. "
                    f"Please provide either a valid ARN or ensure the model package group exists."
                )
        
        # Case 2: ModelPackageGroup object
        if hasattr(v, 'model_package_group_arn'):
            arn = v.model_package_group_arn
            _logger.info(f"Resolved ModelPackageGroup object to ARN: {arn}")
            return arn
        
        # Invalid type
        raise ValueError(
            f"model_package_group must be either:\n"
            f"1. ARN string (e.g., 'arn:aws:sagemaker:region:account:model-package-group/name')\n"
            f"2. ModelPackageGroup object with model_package_group_arn attribute\n"
            f"3. Model package group name string\n"
            f"Got type: {type(v).__name__}"
        )
    
    @validator('mlflow_resource_arn')
    def _validate_mlflow_arn_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate MLFlow resource ARN format if provided.
        
        Args:
            v (Optional[str]): The MLflow resource ARN to validate.
            
        Returns:
            Optional[str]: The validated MLflow resource ARN or None.
            
        Raises:
            ValueError: If the ARN format is invalid.
        
        Expected formats:
            - MLflow tracking server: arn:aws[a-z-]*:sagemaker:[region]:[account-id]:mlflow-tracking-server/[name]
            - MLflow app: arn:aws[a-z-]*:sagemaker:[region]:[account-id]:mlflow-app/[app-id]
        """
        if v is not None and not re.match(_MLFLOW_ARN_PATTERN, v):
            raise ValueError(
                f"Invalid MLFlow resource ARN format: {v}. "
                f"Expected formats:\n"
                f"  - MLflow tracking server: arn:aws[a-z-]*:sagemaker:[region]:[account-id]:mlflow-tracking-server/[name]\n"
                f"  - MLflow app: arn:aws[a-z-]*:sagemaker:[region]:[account-id]:mlflow-app/[app-id]"
            )
        return v
    
    @validator('model')
    def _resolve_model_info(cls, v: Union[str, BaseTrainer, ModelPackage], values: dict) -> Union[str, Any]:
        """Resolve model information from various input types.
        
        This validator uses the common model resolution utility to extract:
        - base_model_name: Human-readable model name for job naming
        - base_model_arn: ARN of the base model
        - source_model_package_arn: ARN of source model package (if fine-tuned model)
        
        The resolved information is stored in private attributes for use by subclasses.
        
        Args:
            v (Union[str, BaseTrainer, ModelPackage]): Model identifier (JumpStart ID, ModelPackage, ARN, or BaseTrainer).
            values (dict): Dictionary of already-validated fields.
            
        Returns:
            Union[str, Any]: The validated model identifier.
            
        Raises:
            ValueError: If model resolution fails or base model is not supported.
        """
        from sagemaker.train.common_utils.model_resolution import _resolve_base_model
        import os
        
        try:
            # Get the session for resolution (may not be created yet due to validator order)
            session = values.get('sagemaker_session')
            
            # Resolve model information
            model_info = _resolve_base_model(
                base_model=v,
                sagemaker_session=session
            )
            
            # Check if model is GPT OSS (not supported for evaluation)
            if model_info.base_model_name in ["openai-reasoning-gpt-oss-20b", "openai-reasoning-gpt-oss-120b"]:
                raise ValueError(
                    "Evaluation is currently not supported for models created from GPT OSS 20B base model"
                )
            
            # If model is a ModelPackage object or ARN (has source_model_package_arn),
            # validate that the resolved base_model_arn is a hub content ARN
            if model_info.source_model_package_arn:
                # Check if base_model_arn is a hub content ARN
                # Format: arn:aws:sagemaker:region:aws:hub-content/...
                if not model_info.base_model_arn or ':hub-content/' not in model_info.base_model_arn:
                    raise ValueError(
                        f"Base model is not supported. When using a ModelPackage, the base model "
                        f"must be a JumpStart hub content model. "
                        f"Resolved base model ARN: {model_info.base_model_arn}"
                    )
            
            # Store resolved information in the values dict so it's available during init
            # Note: We can't directly set private attributes here, so we'll do it in __init__
            values['_resolved_model_info'] = model_info
            
            return v
            
        except Exception as e:
            raise ValueError(f"Failed to resolve model: {e}")
    
    @validator('sagemaker_session', always=True, pre=True)
    def _create_default_session(cls, v: Optional[Any], values: dict) -> Any:
        """Create a default SageMaker session if not provided.
        
        Respects SAGEMAKER_ENDPOINT and SAGEMAKER_REGION environment variables.
        
        Args:
            v (Optional[Any]): The sagemaker_session if provided, None otherwise.
            values (dict): Dictionary of already-validated fields.
            
        Returns:
            Any: SageMaker session object (provided or newly created).
        """
        if v is None:
            import os
            import boto3
            from sagemaker.core.helper.session_helper import Session
            
            # Get region from parameter or environment
            region = values.get('region') or os.environ.get('SAGEMAKER_REGION') or os.environ.get('AWS_REGION', 'us-west-2')
            
            # Check for beta endpoint in environment variable
            endpoint = os.environ.get('SAGEMAKER_ENDPOINT')
            sm_client = boto3.client(
                'sagemaker',
                endpoint_url=endpoint if endpoint else None,
                region_name=region
            )
            return Session(sagemaker_client=sm_client)
        return v
    
    def __init__(self, **data: Any) -> None:
        """Initialize evaluator and set resolved model information.
        
        Args:
            **data: Keyword arguments for initializing the evaluator fields.
        """
        super().__init__(**data)
        # Get resolved model info from validator if available, otherwise cache as None
        resolved_info = data.get('_resolved_model_info', None)
        object.__setattr__(self, '_resolved_model_info_cache', resolved_info)
    
    def _get_resolved_model_info(self) -> Any:
        """Lazily resolve and cache model information.
        
        Returns:
            Any: Resolved model information object containing base_model_name,
                base_model_arn, and source_model_package_arn attributes.
        """
        if not hasattr(self, '_resolved_model_info_cache') or self._resolved_model_info_cache is None:
            from sagemaker.train.common_utils.model_resolution import _resolve_base_model
            # Don't catch exceptions silently - let them propagate
            info = _resolve_base_model(self.model, self.sagemaker_session)
            object.__setattr__(self, '_resolved_model_info_cache', info)
        return self._resolved_model_info_cache
    
    @property
    def _base_model_name(self) -> Optional[str]:
        """Get the resolved base model name."""
        info = self._get_resolved_model_info()
        return info.base_model_name if info else None
    
    @property
    def _base_model_arn(self) -> Optional[str]:
        """Get the resolved base model ARN."""
        info = self._get_resolved_model_info()
        return info.base_model_arn if info else None
    
    @property
    def _source_model_package_arn(self) -> Optional[str]:
        """Get the resolved source model package ARN (None for JumpStart models)."""
        info = self._get_resolved_model_info()
        return info.source_model_package_arn if info else None

    @property
    def _is_jumpstart_model(self) -> bool:
        """Determine if model is a JumpStart model"""
        from sagemaker.train.common_utils.model_resolution import _ModelType
        info = self._get_resolved_model_info()
        return info.model_type == _ModelType.JUMPSTART
    
    def _infer_model_package_group_arn(self) -> Optional[str]:
        """Infer model package group ARN from source model package ARN.
        
        Extracts the model package group name from a model package ARN and constructs
        the corresponding model package group ARN.
        
        Model package ARN format:
            arn:aws:sagemaker:region:account:model-package/package-group-name/version
        
        Model package group ARN format:
            arn:aws:sagemaker:region:account:model-package-group/package-group-name
        
        Returns:
            Optional[str]: Model package group ARN if source model package ARN exists, None otherwise
        """
        if not self._source_model_package_arn:
            return None
        
        try:
            match = re.match(_MODEL_PACKAGE_ARN_PATTERN, self._source_model_package_arn)
            
            if not match:
                _logger.warning(f"Invalid model package ARN format: {self._source_model_package_arn}")
                return None
            
            region = match.group(1)
            account = match.group(2)
            package_group_name = match.group(3)
            
            # Construct model package group ARN
            model_package_group_arn = f"arn:aws:sagemaker:{region}:{account}:model-package-group/{package_group_name}"
            
            _logger.info(f"Inferred model package group ARN: {model_package_group_arn} from {self._source_model_package_arn}")
            return model_package_group_arn
            
        except Exception as e:
            _logger.warning(f"Failed to infer model package group ARN from {self._source_model_package_arn}: {e}")
            return None
    
    def _get_model_package_group_arn(self) -> Optional[str]:
        """Get or infer model_package_group ARN.
        
        This method handles all cases:
        1. If model_package_group was explicitly provided by user, use it (already resolved by validator)
        2. If using a ModelPackage (source_model_package_arn exists), try to infer it
        3. If using a JumpStart model ID (no source_model_package_arn), return None (it's optional)
        
        The validator handles three input types for model_package_group when provided:
        1. ARN string (validated against pattern)
        2. ModelPackageGroup object (extracts model_package_group_arn attribute)
        3. Model package group name string (fetches object and extracts ARN)
        
        Returns:
            Optional[str]: Model package group ARN (provided/resolved/inferred) or None for JumpStart models
            
        Raises:
            ValueError: If model_package_group cannot be determined for ModelPackage scenarios
        """
        # Case 1: User explicitly provided model_package_group - use it regardless of model type
        # The validator has already resolved it to an ARN string
        if self.model_package_group:
            _logger.info(f"Using user-provided model_package_group ARN: {self.model_package_group}")
            return self.model_package_group
        
        # Case 2: Using a ModelPackage (fine-tuned model) - try to infer from source_model_package_arn
        if self._source_model_package_arn:
            inferred_arn = self._infer_model_package_group_arn()
            if inferred_arn:
                _logger.info(f"Automatically inferred model_package_group from ModelPackage: {inferred_arn}")
                return inferred_arn
            else:
                raise ValueError(
                    f"Could not infer model_package_group from source_model_package_arn: {self._source_model_package_arn}. "
                    f"Please provide model_package_group explicitly."
                )
        
        # Case 3: Using a JumpStart model ID - model_package_group is optional, return None
        _logger.info("Using JumpStart model - model_package_group not required")
        return None
    
    def _get_or_create_artifact_arn(self, source_uri: str, region: str) -> str:
        """Get existing artifact or create new one for a model source URI.
        
        Uses sagemaker_core Artifact class to find or create artifacts.
        Supports both model package ARNs and base model (hub content) ARNs.
        
        Args:
            source_uri: Source URI to find/create artifact for. Can be either:
                - Model package ARN: arn:aws:sagemaker:region:account:model-package/name/version
                - Base model ARN: arn:aws:sagemaker:region:aws:hub-content/HubName/Model/name/version
            region: AWS region
            
        Returns:
            str: Artifact ARN (either existing or newly created)
        """
        from datetime import datetime

        from sagemaker.core.resources import Artifact
        from sagemaker.core.shapes import ArtifactSource, ArtifactSourceType
        
        # Determine source type from ARN
        is_model_package = ':model-package/' in source_uri
        is_hub_content = ':hub-content/' in source_uri
        
        source_type_label = "model package" if is_model_package else "base model" if is_hub_content else "model"
        
        # Try to find existing artifact using Artifact.get_all()
        try:
            _logger.info(f"Searching for existing artifact for {source_type_label}: {source_uri}")
            artifacts_iter = Artifact.get_all(
                source_uri=source_uri,
                region=region
            )
            
            # Get first artifact from iterator
            for artifact in artifacts_iter:
                artifact_arn = artifact.artifact_arn
                _logger.info(f"Found existing artifact: {artifact_arn}")
                return artifact_arn
        except Exception as e:
            _logger.info(f"Could not list artifacts: {e}")
        
        # Create new artifact if none exists
        try:
            _logger.info(f"Creating new artifact for {source_type_label}: {source_uri}")
            
            # Prepare properties based on source type
            properties = {}
            if is_model_package:
                properties['ModelPackageArn'] = source_uri
            elif is_hub_content:
                properties['HubContentArn'] = source_uri
            else:
                properties['SourceUri'] = source_uri

            _logger.info(f"source_uri: {source_uri}, region: {region}, properties: {properties}")
            
            # Create artifact using Artifact.create()
            artifact = Artifact.create(
                artifact_type='Model',
                source=ArtifactSource(
                    source_uri=source_uri,
                    source_types=[
                        ArtifactSourceType(
                            source_id_type='Custom',
                            value=datetime.utcnow().strftime('%a %b %d %H:%M:%S UTC %Y')
                        )
                    ]
                ),
                properties=properties,
                region=region
            )
            
            artifact_arn = artifact.artifact_arn
            _logger.info(f"Created new artifact: {artifact_arn}")
            return artifact_arn
        except Exception as e:
            _logger.error(f"Could not create artifact: {e}")
            # Raise the error - artifact creation should succeed
            raise RuntimeError(f"Failed to create artifact for {source_type_label} {source_uri}: {e}")
    
    @validator('base_eval_name', always=True)
    def _generate_default_eval_name(cls, v: Optional[str], values: dict) -> str:
        """Generate a unique eval name if not provided using format: eval-{model_name}-{uuid}.
        
        Adheres to AWS pipeline naming constraints:
        - Length: 1-256 characters
        - Pattern: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}
        
        Args:
            v (Optional[str]): The base_eval_name if provided, None otherwise.
            values (dict): Dictionary of already-validated fields.
            
        Returns:
            str: The evaluation name (provided or newly generated).
        """
        if v is None:
            import uuid
            import re
            # Generate shorter UUID (first 8 characters)
            short_uuid = str(uuid.uuid4())[:8]
            
            # Try to use resolved model name, fallback to model string representation
            model_name = 'model'
            if '_resolved_model_info' in values:
                model_name = values['_resolved_model_info'].base_model_name
            elif 'model' in values:
                model = values['model']
                if isinstance(model, str):
                    model_name = model
            
            # Take only the first part before hyphen
            model_name = model_name.split('-')[0]
            # Remove non-alphanumeric characters except hyphens
            model_name = re.sub(r'[^a-zA-Z0-9-]', '-', model_name)
            # Remove consecutive hyphens
            model_name = re.sub(r'-+', '-', model_name)
            # Remove leading/trailing hyphens
            model_name = model_name.strip('-')
            # Limit model name length (eval- is 5 chars, uuid is 8 chars, hyphens are 2 chars = 15 chars overhead)
            # Keep model name under 240 chars to stay well under 256 limit
            model_name = model_name[:240]
            return f"eval-{model_name}-{short_uuid}"
        return v
    
    def _get_aws_execution_context(self) -> Dict[str, str]:
        """Get AWS execution context (role ARN, region, account ID).
        
        Returns:
            dict: Dictionary containing:
                - role_arn (str): IAM role ARN for execution
                - region (str): AWS region
                - account_id (str): AWS account ID
        """
        # Get role ARN
        role_arn = (self.sagemaker_session.get_caller_identity_arn() 
                   if hasattr(self.sagemaker_session, 'get_caller_identity_arn') 
                   else self.sagemaker_session.expand_role())
        
        # Get region - prefer self.region if set, otherwise extract from session
        region = self.region or (self.sagemaker_session.boto_region_name 
                                 if hasattr(self.sagemaker_session, 'boto_region_name') 
                                 else 'us-west-2')
        
        # Extract account ID from role ARN
        account_id = role_arn.split(':')[4] if ':' in role_arn else '052150106756'
        
        return {
            'role_arn': role_arn,
            'region': region,
            'account_id': account_id
        }
    
    def _resolve_model_artifacts(self, region: str) -> Dict[str, str]:
        """Resolve model artifacts and create artifact ARN if needed.
        
        Args:
            region (str): AWS region
            
        Returns:
            dict: Dictionary containing:
                - artifact_source_uri (str): Source URI for artifact
                - resolved_model_artifact_arn (str): Resolved or created artifact ARN
        """
        # Determine artifact source URI - prefer model package, fallback to base model ARN
        artifact_source_uri = self._source_model_package_arn or self._base_model_arn or self.model
        
        # Get or create artifact ARN from the source URI
        _logger.info(f"Getting or creating artifact for source: {artifact_source_uri}")
        resolved_model_artifact_arn = self._get_or_create_artifact_arn(
            source_uri=artifact_source_uri,
            region=region
        )
        
        return {
            'artifact_source_uri': artifact_source_uri,
            'resolved_model_artifact_arn': resolved_model_artifact_arn
        }
    
    def _get_base_template_context(
        self,
        role_arn: str,
        region: str,
        account_id: str,
        model_package_group_arn: Optional[str],
        resolved_model_artifact_arn: str,
    ) -> Dict[str, Any]:
        """Build base template context with common fields.
        
        Args:
            role_arn (str): IAM role ARN
            region (str): AWS region
            account_id (str): AWS account ID
            model_package_group_arn (Optional[str]): Model package group ARN
            resolved_model_artifact_arn (str): Artifact ARN
            
        Returns:
            dict: Base template context dictionary
        """
        return {
            'role_arn': role_arn,
            'mlflow_resource_arn': self.mlflow_resource_arn,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'mlflow_run_name': self.mlflow_run_name,
            'model_package_group_arn': model_package_group_arn,
            'source_model_package_arn': self._source_model_package_arn,
            'base_model_arn': self._base_model_arn or self.model,
            's3_output_path': self.s3_output_path,
            'dataset_artifact_arn': resolved_model_artifact_arn,
            'action_arn_prefix': f"arn:aws:sagemaker:{region}:{account_id}:action",
            # Preserve pipeline_name placeholder for execution.py to replace during pipeline creation
            'pipeline_name': '{{ pipeline_name }}',
        }
    
    def _select_template(self, base_only_template: str, full_template: str) -> str:
        """Select appropriate template based on model type.
        
        Args:
            base_only_template (str): Template for JumpStart models (base-only)
            full_template (str): Template for ModelPackages (with custom model)
            
        Returns:
            str: Selected template string
        """
        if self._source_model_package_arn is None:
            _logger.info("Using base-only template for JumpStart model")
            return base_only_template
        else:
            _logger.info("Using full template for ModelPackage")
            return full_template
    
    def _add_vpc_and_kms_to_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add VPC and KMS configuration to template context if provided.
        
        Args:
            context (dict): Template context dictionary to modify
            
        Returns:
            dict: Modified context with VPC and KMS config added
        """
        # Add VPC configuration if provided
        if self.networking:
            context['vpc_config'] = True
            context['vpc_security_group_ids'] = self.networking.security_group_ids
            context['vpc_subnets'] = self.networking.subnets
        
        # Add KMS key ID if provided
        if self.kms_key_id:
            context['kms_key_id'] = self.kms_key_id
        
        return context
    
    @staticmethod
    def _render_pipeline_definition(template_str: str, context: Dict[str, Any]) -> str:
        """Render pipeline definition from Jinja2 template.
        
        Args:
            template_str (str): Jinja2 template string
            context (dict): Template context dictionary
            
        Returns:
            str: Rendered pipeline definition
        """
        from jinja2 import Template
        import json
        
        _logger.info(f"Resolved template parameters: {context}")
        
        template = Template(template_str)
        pipeline_definition = template.render(**context)
        
        # Pretty print the entire pipeline definition for debugging
        try:
            pipeline_dict = json.loads(pipeline_definition)
            pretty_pipeline = json.dumps(pipeline_dict, indent=2)
            _logger.info(f"Rendered pipeline definition:\n{pretty_pipeline}")
        except Exception as e:
            _logger.warning(f"Could not parse pipeline definition as JSON for pretty printing: {e}")
            _logger.info(f"Rendered pipeline definition (raw):\n{pipeline_definition}")
        
        return pipeline_definition
    
    def _start_execution(
        self,
        eval_type: Any,
        name: str,
        pipeline_definition: str,
        role_arn: str,
        region: str,
    ) -> Any:
        """Start evaluation pipeline execution.
        
        Args:
            eval_type: Evaluation type enum value
            name (str): Execution name
            pipeline_definition (str): Pipeline definition JSON/YAML
            role_arn (str): IAM role ARN
            region (str): AWS region
            
        Returns:
            EvaluationPipelineExecution: Started execution object
        """
        from .execution import EvaluationPipelineExecution

        tags: List[TagsDict] = []
        
        if self._is_jumpstart_model:
            from sagemaker.core.jumpstart.utils import add_jumpstart_model_info_tags
            tags = add_jumpstart_model_info_tags(tags, self.model, "*")
        
        execution = EvaluationPipelineExecution.start(
            eval_type=eval_type,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=role_arn,
            s3_output_path=self.s3_output_path,
            session=self.sagemaker_session.boto_session if hasattr(self.sagemaker_session, 'boto_session') else None,
            region=region,
            tags=tags
        )
        
        return execution
    
    def evaluate(self) -> Any:
        """Create and start an evaluation execution.
        
        This method must be implemented by subclasses to define the specific
        evaluation logic for different evaluation types (benchmark, custom scorer,
        LLM-as-judge, etc.).
        
        Returns:
            EvaluationPipelineExecution: The created evaluation execution object.
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
            
        Example:
            >>> # In a subclass implementation
            >>> class CustomEvaluator(BaseEvaluator):
            ...     def evaluate(self):
            ...         # Create pipeline definition
            ...         pipeline_definition = self._build_pipeline()
            ...         # Start execution
            ...         return EvaluationPipelineExecution.start(...)
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
