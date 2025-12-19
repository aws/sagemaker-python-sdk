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
"""ModelBuilder class for building and deploying machine learning models.

Provides a unified interface for building and deploying ML models across different
model servers and deployment modes.
"""
from __future__ import absolute_import, annotations

import json
import re 
import os
import copy
import logging
import uuid
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from botocore.exceptions import ClientError
import packaging.version

from sagemaker.core.resources import Model, Endpoint, TrainingJob, HubContent, InferenceComponent, EndpointConfig
from sagemaker.core.shapes import (
    ContainerDefinition,
    ModelMetrics,
    MetadataProperties,
    ModelLifeCycle,
    DriftCheckBaselines, InferenceComponentComputeResourceRequirements,
)
from sagemaker.core.resources import (
    ModelPackage,
    ModelPackageGroup,
    ModelCard,
    ModelPackageModelCard,
)
from sagemaker.core.utils.utils import logger
from sagemaker.core.helper import session_helper
from sagemaker.core.helper.session_helper import Session, get_execution_role, _wait_until, _deploy_done
from sagemaker.core.helper.pipeline_variable import StrPipeVar, PipelineVariable

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import Compute, Networking, SourceCode

from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.local_resources import LocalEndpoint
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator, CustomOrchestrator
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.core.transformer import Transformer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.mode.sagemaker_endpoint_mode import SageMakerEndpointMode
from sagemaker.serve.mode.in_process_mode import InProcessMode
from sagemaker.serve.utils.types import ModelServer, ModelHub
from sagemaker.serve.detector.image_detector import (
    _get_model_base, _detect_framework_and_version
)
from sagemaker.serve.detector.pickler import save_pkl, save_xgboost
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.serve.inference_recommendation_mixin import _InferenceRecommenderMixin
from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.serve.model_builder_servers import _ModelBuilderServers
from sagemaker.serve.validations.optimization import _validate_optimization_configuration
from sagemaker.core.enums import Tag
from sagemaker.core.model_registry import (
    get_model_package_args,
    create_model_package_from_containers,
)
from sagemaker.core.jumpstart.enums import JumpStartModelType

from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.jumpstart.utils import add_jumpstart_uri_tags
from sagemaker.core.jumpstart.artifacts.kwargs import _retrieve_model_deploy_kwargs

from sagemaker.core.inference_config import AsyncInferenceConfig, ServerlessInferenceConfig
from sagemaker.serve.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig

from sagemaker.core.serializers import (
    NumpySerializer,
    TorchTensorSerializer,
)
from sagemaker.core.deserializers import (
    JSONDeserializer,
    TorchTensorDeserializer,
)
from sagemaker.core import s3
from sagemaker.core.explainer.explainer_config import ExplainerConfig
from sagemaker.core.enums import EndpointType
from sagemaker.core.common_utils import (
    Tags,
    ModelApprovalStatusEnum,
    _resolve_routing_config,
    format_tags,
    resolve_value_from_config,
    unique_name_from_base,
    name_from_base,
    base_from_name,
    to_string,
    base_name_from_image,
    resolve_nested_dict_value_from_config,
    get_config_value,
    repack_model,
    update_container_with_inference_params,
)
from sagemaker.core.config.config_schema import (
    MODEL_ENABLE_NETWORK_ISOLATION_PATH, MODEL_EXECUTION_ROLE_ARN_PATH,
    MODEL_VPC_CONFIG_PATH, ENDPOINT_CONFIG_ASYNC_KMS_KEY_ID_PATH, MODEL_CONTAINERS_PATH
)
from sagemaker.serve.constants import SUPPORTED_MODEL_SERVERS, Framework
from sagemaker.core.workflow.pipeline_context import PipelineSession, runnable_by_pipeline
from sagemaker.core import fw_utils
from sagemaker.core.helper.session_helper import container_def
from sagemaker.core.workflow import is_pipeline_variable

from sagemaker.core import image_uris
from sagemaker.core.fw_utils import model_code_key_prefix
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

_LOWEST_MMS_VERSION = "1.2"
SCRIPT_PARAM_NAME = "sagemaker_program"
DIR_PARAM_NAME = "sagemaker_submit_directory"
CONTAINER_LOG_LEVEL_PARAM_NAME = "sagemaker_container_log_level"
JOB_NAME_PARAM_NAME = "sagemaker_job_name"
MODEL_SERVER_WORKERS_PARAM_NAME = "sagemaker_model_server_workers"
SAGEMAKER_REGION_PARAM_NAME = "sagemaker_region"
SAGEMAKER_OUTPUT_LOCATION = "sagemaker_s3_output"

@dataclass
class ModelBuilder(_InferenceRecommenderMixin, _ModelBuilderServers, _ModelBuilderUtils):
    """Unified interface for building and deploying machine learning models.
    
    ModelBuilder provides a streamlined workflow for preparing and deploying ML models to
    Amazon SageMaker. It supports multiple frameworks (PyTorch, TensorFlow, HuggingFace, etc.),
    model servers (TorchServe, TGI, Triton, etc.), and deployment modes (SageMaker endpoints,
    local containers, in-process).
    
    The typical workflow involves three steps:
    1. Initialize ModelBuilder with your model and configuration
    2. Call build() to create a deployable Model resource
    3. Call deploy() to create an Endpoint resource for inference
    
    Example:
        >>> from sagemaker.serve.model_builder import ModelBuilder
        >>> from sagemaker.serve.mode.function_pointers import Mode
        >>> 
        >>> # Initialize with a trained model
        >>> model_builder = ModelBuilder(
        ...     model=my_pytorch_model,
        ...     role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        ...     instance_type="ml.m5.xlarge"
        ... )
        >>> 
        >>> # Build the model (creates SageMaker Model resource)
        >>> model = model_builder.build()
        >>> 
        >>> # Deploy to endpoint (creates SageMaker Endpoint resource)
        >>> endpoint = model_builder.deploy(endpoint_name="my-endpoint")
        >>> 
        >>> # Make predictions
        >>> result = endpoint.invoke(data=input_data)
    
    Args:
        model: The model to deploy. Can be a trained model object, ModelTrainer, TrainingJob,
            ModelPackage, or JumpStart model ID string. Either model or inference_spec is required.
        model_path: Local directory path where model artifacts are stored or will be downloaded.
        inference_spec: Custom inference specification with load() and invoke() functions.
        schema_builder: Defines input/output schema for serialization and deserialization.
        modelbuilder_list: List of ModelBuilder objects for multi-model deployments.
        pipeline_models: List of Model objects for creating inference pipelines.
        role_arn: IAM role ARN for SageMaker to assume.
        sagemaker_session: Session object for managing SageMaker API interactions.
        image_uri: Container image URI. Auto-selected if not specified.
        s3_model_data_url: S3 URI where model artifacts are stored or will be uploaded.
        source_code: Source code configuration for custom inference code.
        env_vars: Environment variables to set in the container.
        model_server: Model server to use (TORCHSERVE, TGI, TRITON, etc.).
        model_metadata: Dictionary to override model metadata (HF_TASK, MLFLOW_MODEL_PATH, etc.).
        log_level: Logging level for ModelBuilder operations (default: logging.DEBUG).
        content_type: MIME type of input data. Auto-derived from schema_builder if provided.
        accept_type: MIME type of output data. Auto-derived from schema_builder if provided.
        compute: Compute configuration specifying instance type and count.
        network: Network configuration including VPC settings and network isolation.
        instance_type: EC2 instance type for deployment (e.g., 'ml.m5.large').
        mode: Deployment mode (SAGEMAKER_ENDPOINT, LOCAL_CONTAINER, or IN_PROCESS).
    
    Note:
        ModelBuilder returns sagemaker.core.resources.Model and sagemaker.core.resources.Endpoint
        objects, not the deprecated PySDK Model and Predictor classes. Use endpoint.invoke()
        instead of predictor.predict() for inference.
    """
    # ========================================
    # Core Model Definition
    # ========================================
    model: Optional[Union[object, str, ModelTrainer, BaseTrainer, TrainingJob, ModelPackage, List[Model]]] = field(
        default=None,
        metadata={
            "help": "The model object, JumpStart model ID, or training job from which to extract "
            "model artifacts. Can be a trained model object, ModelTrainer, TrainingJob, "
            "ModelPackage, JumpStart model ID string, or list of core models. Either model or inference_spec is required."
        },
    )
    model_path: Optional[str] = field(
        default_factory=lambda: "/tmp/sagemaker/model-builder/" + uuid.uuid1().hex,
        metadata={
            "help": "Local directory path where model artifacts are stored or will be downloaded. "
            "Defaults to a temporary directory under /tmp/sagemaker/model-builder/."
        },
    )
    inference_spec: Optional[InferenceSpec] = field(
        default=None,
        metadata={
            "help": "Custom inference specification with load() and invoke() functions for "
            "model loading and inference logic. Either model or inference_spec is required."
        },
    )
    schema_builder: Optional[SchemaBuilder] = field(
        default=None,
        metadata={
            "help": "Defines the input/output schema for the model. The schema builder handles "
            "serialization and deserialization of data between client and server. Can be omitted "
            "for certain HuggingFace models with supported task types."
        },
    )
    modelbuilder_list: Optional[List["ModelBuilder"]] = field(
        default=None,
        metadata={
            "help": "List of ModelBuilder objects for multi-model or inference component deployments. "
            "Used when deploying multiple models to a single endpoint."
        },
    )
    role_arn: Optional[str] = field(
        default=None,
        metadata={
            "help": "IAM role ARN for SageMaker to assume when creating models and endpoints. "
            "If not specified, attempts to use the default SageMaker execution role."
        },
    )
    sagemaker_session: Optional[Session] = field(
        default=None,
        metadata={
            "help": "Session object for managing interactions with SageMaker APIs and AWS services. "
            "If not specified, creates a session using the default AWS configuration."
        },
    )
    image_uri: Optional[StrPipeVar] = field(
        default=None,
        metadata={
            "help": "Container image URI for the model. If not specified, automatically selects "
            "an appropriate SageMaker-provided container based on framework and model server."
        },
    )
    s3_model_data_url: Optional[Union[str, PipelineVariable, Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": "S3 URI where model artifacts are stored or will be uploaded. If not specified, "
            "model artifacts are uploaded to a default S3 location."
        },
    )
    source_code: Optional[SourceCode] = field(
        default=None,
        metadata={
            "help": "Source code configuration for custom inference code, including source directory, "
            "entry point script, and dependencies."
        },
    )
    env_vars: Optional[Dict[str, StrPipeVar]] = field(
        default_factory=dict,
        metadata={
            "help": "Environment variables to set in the model container at runtime. Used to pass "
            "configuration and secrets to the inference code."
        },
    )
    model_server: Optional[ModelServer] = field(
        default=None,
        metadata={
            "help": "Model server to use for serving the model. Options include TORCHSERVE, MMS, "
            "TENSORFLOW_SERVING, DJL_SERVING, TRITON, TGI, and TEI. Required when using a custom image_uri."
        },
    )
    model_metadata: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary to override model metadata. Supported keys: HF_TASK (for HuggingFace "
            "models without task metadata), MLFLOW_MODEL_PATH (local or S3 path to MLflow artifacts), "
            "FINE_TUNING_MODEL_PATH (S3 path to fine-tuned model), FINE_TUNING_JOB_NAME (fine-tuning "
            "job name), and CUSTOM_MODEL_PATH (local or S3 path to custom model artifacts). "
            "FINE_TUNING_MODEL_PATH and FINE_TUNING_JOB_NAME are mutually exclusive."
        },
    )
    log_level: Optional[int] = field(
        default=logging.DEBUG,
        metadata={
            "help": "Logging level for ModelBuilder operations. Valid values are logging.CRITICAL, "
            "logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, and logging.NOTSET. "
            "Controls verbosity of ModelBuilder logs."
        },
    )
    content_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "MIME type of the input data for inference requests (e.g., 'application/json', "
            "'text/csv'). Automatically derived from the input sample if schema_builder is provided, "
            "but can be overridden."
        },
    )
    accept_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "MIME type of the output data from inference responses (e.g., 'application/json'). "
            "Automatically derived from the output sample if schema_builder is provided, but can be overridden."
        },
    )

    compute: Optional[Compute] = field(
        default=None,
        metadata={
            "help": "Compute configuration specifying instance type and instance count for deployment. "
            "Alternative to specifying instance_type separately."
        },
    )
    network: Optional[Networking] = field(
        default=None,
        metadata={
            "help": "Network configuration including VPC settings (subnets, security groups) and "
            "network isolation settings for the model and endpoint."
        },
    )

    instance_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "EC2 instance type for model deployment (e.g., 'ml.m5.large', 'ml.g5.xlarge'). "
            "Used to determine appropriate container images and for deployment."
        },
    )
    mode: Optional[Mode] = field(
        default=Mode.SAGEMAKER_ENDPOINT,
        metadata={
            "help": "Deployment mode for the model. Options: Mode.SAGEMAKER_ENDPOINT (deploy to "
            "SageMaker endpoint), Mode.LOCAL_CONTAINER (run locally in Docker container for testing), "
            "Mode.IN_PROCESS (run locally in current Python process for testing)."
        },
    )

    _base_name: Optional[str] = field(default=None, init=False)
    _is_sharded_model: Optional[bool] = field(default=False, init=False)
    _tags: Optional[Tags] = field(default=None, init=False)
    _optimizing: bool = field(default=False, init=False)
    _deployment_config: Optional[Dict[str, Any]] = field(default=None, init=False)


    shared_libs: List[str] = field(
        default_factory=list,
        metadata={"help": "DEPRECATED: Use configure_for_torchserve() instead"},
    )
    dependencies: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"auto": True},
        metadata={"help": "DEPRECATED: Use configure_for_torchserve() instead"},
    )
    image_config: Optional[Dict[str, StrPipeVar]] = field(
        default=None,
        metadata={"help": "DEPRECATED: Use configure_for_torchserve() instead"},
    )

    def _create_session_with_region(self):
        """Create a SageMaker session with the correct region."""
        if hasattr(self, "region") and self.region:
            import boto3
            boto_session = boto3.Session(region_name=self.region)
            return Session(boto_session=boto_session)
        return Session()

    def __post_init__(self) -> None:
        """Initialize ModelBuilder after instantiation."""
        import warnings

        if self.sagemaker_session is None:
            self.sagemaker_session = self._create_session_with_region()

        # Set logger level based on log_level parameter
        if self.log_level is not None:
            logger.setLevel(self.log_level)

        self._warn_about_deprecated_parameters(warnings)
        self._initialize_compute_config()
        self._initialize_network_config()
        self._initialize_defaults()
        self._initialize_jumpstart_config()
        self._initialize_script_mode_variables()

    def _warn_about_deprecated_parameters(self, warnings) -> None:
        """Issue deprecation warnings for legacy parameters."""
        if self.shared_libs:
            warnings.warn(
                "The 'shared_libs' parameter is deprecated. Use configure_for_torchserve() instead.",
                DeprecationWarning,
                stacklevel=3
            )

        if self.dependencies and self.dependencies != {"auto": False}:
            warnings.warn(
                "The 'dependencies' parameter is deprecated. Use configure_for_torchserve() instead.",
                DeprecationWarning,
                stacklevel=3
            )

        if self.image_config is not None:
            warnings.warn(
                "The 'image_config' parameter is deprecated. Use configure_for_torchserve() instead.",
                DeprecationWarning,
                stacklevel=3
            )

    def _initialize_compute_config(self) -> None:
        """Initialize compute configuration from Compute object."""
        if self.compute:
            self.instance_type = self.compute.instance_type
            self.instance_count = self.compute.instance_count or 1
        else:
            if not hasattr(self, 'instance_type') or self.instance_type is None:
                self.instance_type = None
            if not hasattr(self, 'instance_count') or self.instance_count is None:
                self.instance_count = 1

        self._user_provided_instance_type = bool(self.compute and self.compute.instance_type)

        if not self.instance_type:
            self.instance_type = self._get_default_instance_type()

    def _initialize_network_config(self) -> None:
        """Initialize network configuration from Networking object."""
        if self.network:
            if self.network.vpc_config:
                self.vpc_config = self.network.vpc_config
            else:
                self.vpc_config = {
                    'Subnets': self.network.subnets or [],
                    'SecurityGroupIds': self.network.security_group_ids or []
                } if (self.network.subnets or self.network.security_group_ids) else None
            self._enable_network_isolation = self.network.enable_network_isolation
        else:
            if not hasattr(self, 'vpc_config'):
                self.vpc_config = None
            if not hasattr(self, '_enable_network_isolation'):
                self._enable_network_isolation = False

    def _initialize_defaults(self) -> None:
        """Initialize default values for unset parameters."""
        if not hasattr(self, 'model_name') or self.model_name is None:
            self.model_name = "model-" + str(uuid.uuid4())[:8]

        if not hasattr(self, 'mode') or self.mode is None:
            self.mode = Mode.SAGEMAKER_ENDPOINT

        if not hasattr(self, 'env_vars') or self.env_vars is None:
            self.env_vars = {}

        # Set region with priority: user input > sagemaker session > AWS account region > default
        if not hasattr(self, "region") or not self.region:
            if self.sagemaker_session and self.sagemaker_session.boto_region_name:
                self.region = self.sagemaker_session.boto_region_name
            else:
                # Try to get region from boto3 session (AWS account config)
                try:
                    import boto3
                    self.region = boto3.Session().region_name or None
                except Exception:
                    self.region = None  # Default fallback

        # Set role_arn with priority: user input > execution role detection
        if not self.role_arn:
            self.role_arn = get_execution_role(self.sagemaker_session, use_default=True)

        self._metadata_configs = None
        self.s3_upload_path = None
        self.container_config = "host"
        self.inference_recommender_job_results = None
        self.container_log_level = logging.INFO
        
        if not hasattr(self, 'framework'):
            self.framework = None
        if not hasattr(self, 'framework_version'):
            self.framework_version = None

    def _fetch_default_instance_type_for_custom_model(self) -> str:
        hosting_configs = self._fetch_hosting_configs_for_custom_model()
        default_instance_type = hosting_configs.get("InstanceType")
        if not default_instance_type:
            raise ValueError(
                "Model is not supported for deployment. "
                "The hosting configuration does not specify a default instance type. "
                "Please specify an instance_type explicitly or use a different model."
            )
        logger.info(f"Fetching Instance Type from Hosting Configs - {default_instance_type}")
        return default_instance_type

    def _fetch_hub_document_for_custom_model(self) -> dict:
        from sagemaker.core.shapes import BaseModel as CoreBaseModel
        base_model: CoreBaseModel = self._fetch_model_package().inference_specification.containers[0].base_model
        hub_content = HubContent.get(
            hub_content_type="Model",
            hub_name="SageMakerPublicHub",
            hub_content_name=base_model.hub_content_name,
            hub_content_version=base_model.hub_content_version,
        )
        return json.loads(hub_content.hub_content_document)

    def _fetch_hosting_configs_for_custom_model(self) -> dict:
        hosting_configs = self._fetch_hub_document_for_custom_model().get("HostingConfigs")
        if not hosting_configs:
            raise ValueError(
                "Model is not supported for deployment. "
                "The model does not have hosting configuration. "
                "Please use a model that supports deployment or contact AWS support for assistance."
            )
        return hosting_configs


    def _get_instance_resources(self, instance_type: str) -> tuple:
        """Get CPU and memory for an instance type by querying EC2."""
        try:
            ec2_client = self.sagemaker_session.boto_session.client('ec2')
            ec2_instance_type = instance_type.replace('ml.', '')
            response = ec2_client.describe_instance_types(InstanceTypes=[ec2_instance_type])
            if response['InstanceTypes']:
                instance_info = response['InstanceTypes'][0]
                cpus = instance_info['VCpuInfo']['DefaultVCpus']
                memory_mb = instance_info['MemoryInfo']['SizeInMiB']
                return cpus, memory_mb
        except Exception as e:
            logger.warning(
                f"Could not query instance type {instance_type}: {e}. "
                f"Unable to validate CPU requirements. Proceeding with recipe defaults."
            )
        return None, None

    def _fetch_and_cache_recipe_config(self):
        """Fetch and cache image URI, compute requirements, and s3_upload_path from recipe during build."""
        hub_document = self._fetch_hub_document_for_custom_model()
        model_package = self._fetch_model_package()
        recipe_name = model_package.inference_specification.containers[0].base_model.recipe_name

        if not self.s3_upload_path:
            self.s3_upload_path = model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri

        for recipe in hub_document.get("RecipeCollection", []):
            if recipe.get("Name") == recipe_name:
                hosting_configs = recipe.get("HostingConfigs", [])
                if hosting_configs:
                    config = next(
                        (cfg for cfg in hosting_configs if cfg.get("Profile") == "Default"),
                        hosting_configs[0]
                    )
                    if not self.image_uri:
                        self.image_uri = config.get("EcrAddress")
                    if not self.instance_type:
                        self.instance_type = config.get("InstanceType") or config.get("DefaultInstanceType")
                    
                    compute_resource_requirements = config.get("ComputeResourceRequirements", {})
                    requested_cpus = compute_resource_requirements.get("NumberOfCpuCoresRequired", 1)
                    
                    # Get actual CPU count from instance type
                    actual_cpus, _ = self._get_instance_resources(self.instance_type)
                    if actual_cpus and requested_cpus > actual_cpus:
                        logger.warning(
                            f"Recipe requests {requested_cpus} CPUs but {self.instance_type} has {actual_cpus}. "
                            f"Adjusting to {actual_cpus}."
                        )
                        requested_cpus = actual_cpus
                    
                    self._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
                        min_memory_required_in_mb=1024,
                        number_of_cpu_cores_required=requested_cpus
                    )
                    return

        raise ValueError(
            f"Model with recipe '{recipe_name}' is not supported for deployment. "
            f"The recipe does not have hosting configuration. "
            f"Please use a model that supports deployment or contact AWS support for assistance."
        )

    def _initialize_jumpstart_config(self) -> None:
        """Initialize JumpStart-specific configuration."""
        if hasattr(self, "hub_name") and self.hub_name and not self.hub_arn:
            from sagemaker.core.jumpstart.hub.utils import generate_hub_arn_for_init_kwargs
            self.hub_arn = generate_hub_arn_for_init_kwargs(
                hub_name=self.hub_name,
                region=self.region,
                session=self.sagemaker_session
            )
        else:
            self.hub_name = None
            self.hub_arn = None

        if isinstance(self.model, str) and (not hasattr(self, "model_type") or not self.model_type):
            from sagemaker.core.jumpstart.utils import validate_model_id_and_get_type
            try:
                self.model_type = validate_model_id_and_get_type(
                    model_id=self.model,
                    model_version=self.model_version or "*",
                    region=self.region,
                    hub_arn=self.hub_arn,
                )
            except Exception:
                self.model_type = None

        if isinstance(self.model, str) and self.model_type:
            # Add tags for the JumpStart model
            from sagemaker.core.jumpstart.utils import add_jumpstart_model_info_tags
            from sagemaker.core.jumpstart.enums import JumpStartScriptScope
            self._tags = add_jumpstart_model_info_tags(
                self._tags,
                self.model,
                self.model_version or "*",
                self.model_type,
                self.config_name,
                JumpStartScriptScope.INFERENCE,
            )

        if not hasattr(self, "tolerate_vulnerable_model"):
            self.tolerate_vulnerable_model = None
        if not hasattr(self, "tolerate_deprecated_model"):
            self.tolerate_deprecated_model = None
        if not hasattr(self, "model_data_download_timeout"):
            self.model_data_download_timeout = None
        if not hasattr(self, "container_startup_health_check_timeout"):
            self.container_startup_health_check_timeout = None
        if not hasattr(self, "inference_ami_version"):
            self.inference_ami_version = None
        if not hasattr(self, "model_version"):
            self.model_version = None
        if not hasattr(self, "resource_requirements"):
            self.resource_requirements = None
        if not hasattr(self, "model_kms_key"):
            self.model_kms_key = None
        if not hasattr(self, "hub_name"):
            self.hub_name = None
        if not hasattr(self, "config_name"):
            self.config_name = None
        if not hasattr(self, "accept_eula"):
            self.accept_eula = None


    def _initialize_script_mode_variables(self) -> None:
        """Initialize script mode variables from source_code or defaults."""

        # Map SourceCode to model.py equivalents
        if self.source_code:
            self.entry_point = self.source_code.entry_script
            if hasattr(self.source_code, 'requirements'):
                self.script_dependencies = [self.source_code.requirements] if self.source_code.requirements else []
            else:
                self.script_dependencies = []
                logger.warning(
                    "No requirements.txt file found in source_code. "
                    "If you have any dependencies, please add them to requirements.txt"
                )
            # source_dir already exists as field, but ensure consistency
            if self.source_code.source_dir:
                self.source_dir = self.source_code.source_dir
            else:
                self.source_dir = None
        else:
            self.entry_point = None
            self.source_dir = None

        # Initialize missing script mode variables
        self.git_config = None
        self.key_prefix = None
        self.bucket = None
        self.uploaded_code = None
        self.repacked_model_data = None

    def _get_client_translators(self) -> tuple:
        """Get serializer and deserializer for client-side data translation."""
        serializer = None
        deserializer = None

        if self.content_type == "application/x-npy":
            serializer = NumpySerializer()
        elif self.content_type == "tensor/pt":
            serializer = TorchTensorSerializer()
        elif self.schema_builder and hasattr(self.schema_builder, "custom_input_translator"):
            serializer = self.schema_builder.custom_input_translator
        elif self.schema_builder:
            serializer = self.schema_builder.input_serializer

        if self.accept_type == "application/json":
            deserializer = JSONDeserializer()
        elif self.accept_type == "tensor/pt":
            deserializer = TorchTensorDeserializer()
        elif self.schema_builder and hasattr(self.schema_builder, "custom_output_translator"):
            deserializer = self.schema_builder.custom_output_translator
        elif self.schema_builder:
            deserializer = self.schema_builder.output_deserializer


        if serializer is None or deserializer is None:
            auto_serializer, auto_deserializer = self._fetch_serializer_and_deserializer_for_framework(self.framework)

            if serializer is None:
                serializer = auto_serializer
            if deserializer is None:
                deserializer = auto_deserializer


        if serializer is None:
            raise ValueError("Cannot determine serializer. Try providing a SchemaBuilder.")
        if deserializer is None:
            raise ValueError("Cannot determine deserializer. Try providing a SchemaBuilder.")

        return serializer, deserializer


    def _save_model_inference_spec(self) -> None:
        """Save model or inference specification to the model path."""

        # Skip saving for model customization - model artifacts already in S3
        if self._is_model_customization():
            return

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        code_path = Path(self.model_path).joinpath("code")

        if self.inference_spec:
            save_pkl(code_path, (self.inference_spec, self.schema_builder))
        elif self.model:
            if isinstance(self.model, str):
                self.framework = None
                self.env_vars.update({
                    "MODEL_CLASS_NAME": self.model
                })
            else:
                fw, _ = _detect_framework_and_version(str(_get_model_base(self.model)))
                self.framework = self._normalize_framework_to_enum(fw)
                self.env_vars.update({
                    "MODEL_CLASS_NAME": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
                })


            if self.framework == Framework.XGBOOST:
                save_xgboost(code_path, self.model)
                save_pkl(code_path, (self.framework, self.schema_builder))
            else:
                save_pkl(code_path, (self.model, self.schema_builder))
        elif self._is_mlflow_model:
            save_pkl(code_path, self.schema_builder)
        else:
            raise ValueError("Cannot detect required model or inference spec")


    def _prepare_for_mode(
        self, model_path: Optional[str] = None, should_upload_artifacts: Optional[bool] = False
    ) -> Optional[tuple]:
        """Prepare model artifacts for the specified deployment mode."""
        self.s3_upload_path = None

        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            self.modes[str(Mode.SAGEMAKER_ENDPOINT)] = SageMakerEndpointMode(
                inference_spec=self.inference_spec, model_server=self.model_server
            )
            self.s3_upload_path, env_vars_sagemaker = self.modes[
                str(Mode.SAGEMAKER_ENDPOINT)
            ].prepare(
                (model_path or self.model_path),
                self.secret_key,
                self.serve_settings.s3_model_data_url,
                self.sagemaker_session,
                self.image_uri,
                getattr(self, "model_hub", None) == ModelHub.JUMPSTART,
                should_upload_artifacts=should_upload_artifacts,
            )
            env_vars_sagemaker = env_vars_sagemaker or {}
            for key, value in env_vars_sagemaker.items():
                self.env_vars.setdefault(key, value)
            return self.s3_upload_path, env_vars_sagemaker

        elif self.mode == Mode.LOCAL_CONTAINER:
            self.modes[str(Mode.LOCAL_CONTAINER)] = LocalContainerMode(
                inference_spec=self.inference_spec,
                schema_builder=self.schema_builder,
                session=self.sagemaker_session,
                model_path=self.model_path,
                env_vars=self.env_vars,
                model_server=self.model_server,
            )
            self.modes[str(Mode.LOCAL_CONTAINER)].prepare()
            if self.model_path:
                self.s3_upload_path = f"file://{self.model_path}"

            return None

        elif self.mode == Mode.IN_PROCESS:
            self.modes[str(Mode.IN_PROCESS)] = InProcessMode(
                inference_spec=self.inference_spec,
                model=self.model,
                schema_builder=self.schema_builder,
                session=self.sagemaker_session,
                model_path=self.model_path,
                env_vars=self.env_vars,
            )
            self.modes[str(Mode.IN_PROCESS)].prepare()
            return None

        raise ValueError(
            f"Unsupported deployment mode: {self.mode}. "
            f"Supported modes: {Mode.LOCAL_CONTAINER}, {Mode.SAGEMAKER_ENDPOINT}, {Mode.IN_PROCESS}"
        )


    def _build_validations(self) -> None:
        """Validate ModelBuilder configuration before building."""
        if isinstance(self.model, ModelTrainer) and not self.inference_spec:
            # Check if this is a JumpStart ModelTrainer (which doesn't need InferenceSpec)
            if not (hasattr(self.model, '_jumpstart_config') and self.model._jumpstart_config is not None):
                raise ValueError(
                    "InferenceSpec is required when using ModelTrainer, "
                    "unless it's a JumpStart ModelTrainer created with from_jumpstart_config()"
                )

        if isinstance(self.model, ModelTrainer):
            is_jumpstart = hasattr(self.model, '_jumpstart_config') and self.model._jumpstart_config is not None

            if not is_jumpstart and not self.image_uri:
                logger.warning(
                    "Non-JumpStart ModelTrainer detected without image_uri. Consider providing image_uri "
                    "to skip auto-detection and improve build performance."
                )

            if is_jumpstart:
                logger.info(
                    "JumpStart ModelTrainer detected. InferenceSpec and image_uri are optional "
                    "as JumpStart provides built-in inference logic and container detection."
                )
            else:
                logger.info(
                    "Non-JumpStart ModelTrainer requires InferenceSpec and benefits from explicit image_uri "
                    "for optimal performance."
                )

        if self.inference_spec and self.model and not isinstance(self.model, ModelTrainer):
            raise ValueError("Can only set one of the following: model, inference_spec.")


        if self.image_uri and is_1p_image_uri(self.image_uri) and not self.model and not self.inference_spec and not getattr(self, '_is_mlflow_model', False):
            self._passthrough = True
            return


        if self.image_uri and not is_1p_image_uri(self.image_uri) and not self.model and not self.inference_spec and not getattr(self, '_is_mlflow_model', False):
            self._passthrough = True
            return

        self._passthrough = False
        if self.image_uri and not is_1p_image_uri(self.image_uri) and self.model_server is None:
            raise ValueError(
                f"Model_server must be set when non-first-party image_uri is set. "
                f"Supported model servers: {SUPPORTED_MODEL_SERVERS}"
            )


    def _build_for_passthrough(self) -> Model:
        """Build model for pass-through cases with image-only deployment."""
        if not self.image_uri:
            raise ValueError("image_uri is required for pass-through cases")

        self.s3_upload_path = None
        return self._create_model()


    def _build_default_async_inference_config(self, async_inference_config):
        """Build default async inference config and return ``AsyncInferenceConfig``"""
        unique_folder = unique_name_from_base(self.model_name)
        if async_inference_config.output_path is None:
            async_output_s3uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                "async-endpoint-outputs",
                unique_folder,
            )
            async_inference_config.output_path = async_output_s3uri

        if async_inference_config.failure_path is None:
            async_failure_s3uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                "async-endpoint-failures",
                unique_folder,
            )
            async_inference_config.failure_path = async_failure_s3uri

        return async_inference_config


    def enable_network_isolation(self):
        """Whether to enable network isolation when creating this Model

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return bool(self._enable_network_isolation)

    def _is_model_customization(self) -> bool:
        """Check if the model is from a model customization/fine-tuning job.

        Returns:
            bool: True if the model is from model customization, False otherwise.
        """
        from sagemaker.core.utils.utils import Unassigned

        if not self.model:
            return False

        # Direct ModelPackage input
        if isinstance(self.model, ModelPackage):
            return True

        # TrainingJob with model customization
        # Check both model_package_config (new location) and serverless_job_config (legacy)
        if isinstance(self.model, TrainingJob):
            # Check model_package_config first (new location)
            if (hasattr(self.model, 'model_package_config') and self.model.model_package_config != Unassigned
                    and getattr(self.model.model_package_config, 'source_model_package_arn', Unassigned) != Unassigned):
                return True
            # Fallback to serverless_job_config (legacy location)
            if (hasattr(self.model, 'serverless_job_config') and self.model.serverless_job_config != Unassigned
                    and hasattr(self.model, 'output_model_package_arn') and self.model.output_model_package_arn!= Unassigned):
                return True

        # ModelTrainer with model customization
        if isinstance(self.model, ModelTrainer) and hasattr(self.model, '_latest_training_job'):
            # Check model_package_config first (new location)
            if (hasattr(self.model._latest_training_job, 'model_package_config') and self.model._latest_training_job.model_package_config != Unassigned()
                    and getattr(self.model._latest_training_job.model_package_config, 'source_model_package_arn', Unassigned()) != Unassigned()):
                return True
            # Fallback to serverless_job_config (legacy location)
            if (hasattr(self.model._latest_training_job, 'serverless_job_config') and self.model._latest_training_job.serverless_job_config != Unassigned()
                    and hasattr(self.model._latest_training_job, 'output_model_package_arn') and self.model._latest_training_job.output_model_package_arn!= Unassigned()):
                return True

        # BaseTrainer with model customization
        if isinstance(self.model, BaseTrainer) and hasattr(self.model, '_latest_training_job'):
            # Check model_package_config first (new location)
            if (hasattr(self.model._latest_training_job, 'model_package_config') and self.model._latest_training_job.model_package_config != Unassigned()
                    and getattr(self.model._latest_training_job.model_package_config, 'source_model_package_arn', Unassigned()) != Unassigned()):
                return True
            # Fallback to serverless_job_config (legacy location)
            if (hasattr(self.model._latest_training_job, 'serverless_job_config') and self.model._latest_training_job.serverless_job_config != Unassigned()
                    and hasattr(self.model._latest_training_job, 'output_model_package_arn') and self.model._latest_training_job.output_model_package_arn!= Unassigned()):
                return True

        return False

    def _fetch_model_package_arn(self) -> Optional[str]:
        """Fetch the model package ARN from the model.

        Returns:
            Optional[str]: The model package ARN, or None if not available.
        """
        from sagemaker.core.utils.utils import Unassigned
        
        if isinstance(self.model, ModelPackage):
            return self.model.model_package_arn
        if isinstance(self.model, TrainingJob):
            # Try output_model_package_arn first (preferred)
            if hasattr(self.model, 'output_model_package_arn'):
                arn = self.model.output_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            # Fallback to model_package_config.source_model_package_arn
            if hasattr(self.model, 'model_package_config') and self.model.model_package_config != Unassigned and hasattr(self.model.model_package_config, 'source_model_package_arn'):
                arn = self.model.model_package_config.source_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            # Fallback to serverless_job_config.source_model_package_arn (legacy)
            if hasattr(self.model, 'serverless_job_config') and self.model.serverless_job_config != Unassigned and hasattr(self.model.serverless_job_config, 'source_model_package_arn'):
                arn = self.model.serverless_job_config.source_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            return None
        
        if isinstance(self.model, (ModelTrainer, BaseTrainer)) and hasattr(self.model, '_latest_training_job'):
            # Try output_model_package_arn first (preferred)
            if hasattr(self.model._latest_training_job, 'output_model_package_arn'):
                arn = self.model._latest_training_job.output_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            # Fallback to model_package_config.source_model_package_arn
            if hasattr(self.model._latest_training_job, 'model_package_config') and self.model._latest_training_job.model_package_config != Unassigned and hasattr(self.model._latest_training_job.model_package_config, 'source_model_package_arn'):
                arn = self.model._latest_training_job.model_package_config.source_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            # Fallback to serverless_job_config.source_model_package_arn (legacy)
            if hasattr(self.model._latest_training_job, 'serverless_job_config') and self.model._latest_training_job.serverless_job_config != Unassigned and hasattr(self.model._latest_training_job.serverless_job_config, 'source_model_package_arn'):
                arn = self.model._latest_training_job.serverless_job_config.source_model_package_arn
                if not isinstance(arn, Unassigned):
                    return arn
            
            return None
        
        return None

    def _fetch_model_package(self) -> Optional[ModelPackage]:
        """Fetch the ModelPackage resource.

        Returns:
            Optional[ModelPackage]: The ModelPackage resource, or None if not available.
        """
        if isinstance(self.model, ModelPackage):
            return self.model
        
        # Get the ARN and check if it's valid
        arn = self._fetch_model_package_arn()
        if arn:
            return ModelPackage.get(arn)
        return None

    def _convert_model_data_source_to_local(self, model_data_source):
        """Convert Core ModelDataSource to Local dictionary format."""
        if not model_data_source:
            return None

        result = {}
        if hasattr(model_data_source, 's3_data_source') and model_data_source.s3_data_source:
            s3_source = model_data_source.s3_data_source
            result["S3DataSource"] = {
                "S3Uri": s3_source.s3_uri,
                "S3DataType": s3_source.s3_data_type,
                "CompressionType": s3_source.compression_type,
            }

            # Handle ModelAccessConfig if present
            if hasattr(s3_source, 'model_access_config') and s3_source.model_access_config:
                result["S3DataSource"]["ModelAccessConfig"] = {
                    "AcceptEula": s3_source.model_access_config.accept_eula
                }

        return result

    def _convert_additional_sources_to_local(self, additional_sources):
        """Convert Core AdditionalModelDataSource list to Local dictionary format."""
        if not additional_sources:
            return None

        result = []
        for source in additional_sources:
            source_dict = {
                "ChannelName": source.channel_name,
            }

            if hasattr(source, 's3_data_source') and source.s3_data_source:
                s3_source = source.s3_data_source
                source_dict["S3DataSource"] = {
                    "S3Uri": s3_source.s3_uri,
                    "S3DataType": s3_source.s3_data_type,
                    "CompressionType": s3_source.compression_type,
                }

                # Handle ModelAccessConfig if present
                if hasattr(s3_source, 'model_access_config') and s3_source.model_access_config:
                    source_dict["S3DataSource"]["ModelAccessConfig"] = {
                        "AcceptEula": s3_source.model_access_config.accept_eula
                    }

            result.append(source_dict)

        return result

    def _get_source_code_env_vars(self) -> Dict[str, str]:
        """Convert SourceCode to Local Mode style for environment variables."""
        if not self.source_code:
            return {}


        script_name = self.source_code.entry_script
        dir_name = (
            self.source_code.source_dir
            if self.source_code.source_dir.startswith("s3://")
            else f"file://{self.source_code.source_dir}"
        )

        return {
            "SAGEMAKER_PROGRAM": script_name,
            "SAGEMAKER_SUBMIT_DIRECTORY": dir_name,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",  # INFO level
            "SAGEMAKER_REGION": self.region,
        }


    def to_string(self, obj: object):
        """Convert an object to string

        This helper function handles converting PipelineVariable object to string as well

        Args:
            obj (object): The object to be converted
        """
        return obj.to_string() if is_pipeline_variable(obj) else str(obj)

    def is_repack(self) -> bool:
        """Whether the source code needs to be repacked before uploading to S3.

        Returns:
            bool: if the source need to be repacked or not
        """
        if self.source_dir is None or self.entry_point is None:
            return False

        if isinstance(self.model, ModelTrainer) and self.inference_spec:
            return False

        return self.source_dir and self.entry_point and not (self.key_prefix or self.git_config)


    def _upload_code(self, key_prefix: str, repack: bool = False) -> None:
        """Uploads code to S3 to be used with script mode with SageMaker inference.

        Args:
            key_prefix (str): The S3 key associated with the ``code_location`` parameter of the
                ``Model`` class.
            repack (bool): Optional. Set to ``True`` to indicate that the source code and model
                artifact should be repackaged into a new S3 object. (default: False).
        """
        local_code = get_config_value("local.local_code", self.sagemaker_session.config)

        bucket, key_prefix = s3.determine_bucket_and_prefix(
            bucket=self.bucket,
            key_prefix=key_prefix,
            sagemaker_session=self.sagemaker_session,
        )

        if (self.sagemaker_session.local_mode and local_code) or self.entry_point is None:
            self.uploaded_code = None
        elif not repack:
            self.uploaded_code = fw_utils.tar_and_upload_dir(
                session=self.sagemaker_session.boto_session,
                bucket=bucket,
                s3_key_prefix=key_prefix,
                script=self.entry_point,
                directory=self.source_dir,
                dependencies=self.script_dependencies,
                kms_key=self.model_kms_key,
                settings=self.sagemaker_session.settings,
            )

        if repack and self.s3_model_data_url is not None and self.entry_point is not None:
            if isinstance(self.s3_model_data_url, dict):
                logging.warning("ModelDataSource currently doesn't support model repacking")
                return
            if is_pipeline_variable(self.s3_model_data_url):
                # model is not yet there, defer repacking to later during pipeline execution
                if not isinstance(self.sagemaker_session, PipelineSession):
                    logging.warning(
                        "The model_data is a Pipeline variable of type %s, "
                        "which should be used under `PipelineSession` and "
                        "leverage `ModelStep` to create or register model. "
                        "Otherwise some functionalities e.g. "
                        "runtime repack may be missing. For more, see: "
                        "https://sagemaker.readthedocs.io/en/stable/"
                        "amazon_sagemaker_model_building_pipeline.html#model-step",
                        type(self.s3_model_data_url),
                    )
                    return
                self.sagemaker_session.context.need_runtime_repack.add(id(self))
                self.sagemaker_session.context.runtime_repack_output_prefix = s3.s3_path_join(
                    "s3://", bucket, key_prefix
                )
                # Add the uploaded_code and repacked_model_data to update the container env
                self.repacked_model_data = self.s3_model_data_url
                self.uploaded_code = fw_utils.UploadedCode(
                    s3_prefix=self.repacked_model_data,
                    script_name=os.path.basename(self.entry_point),
                )
                return
            if local_code and self.s3_model_data_url.startswith("file://"):
                repacked_model_data = self.s3_model_data_url
            else:
                repacked_model_data = "s3://" + "/".join([bucket, key_prefix, "model.tar.gz"])
                self.uploaded_code = fw_utils.UploadedCode(
                    s3_prefix=repacked_model_data,
                    script_name=os.path.basename(self.entry_point),
                )

            logger.info(
                "Repacking model artifact (%s), script artifact "
                "(%s), and dependencies (%s) "
                "into single tar.gz file located at %s. "
                "This may take some time depending on model size...",
                self.s3_model_data_url,
                self.source_dir,
                self.dependencies,
                repacked_model_data,
            )

            repack_model(
                inference_script=self.entry_point,
                source_directory=self.source_dir,
                dependencies=self.dependencies,
                model_uri=self.s3_model_data_url,
                repacked_model_uri=repacked_model_data,
                sagemaker_session=self.sagemaker_session,
                kms_key=self.model_kms_key,
            )

            self.repacked_model_data = repacked_model_data

    def _script_mode_env_vars(self):
        """Returns a mapping of environment variables for script mode execution"""
        script_name = self.env_vars.get(SCRIPT_PARAM_NAME.upper(), "")
        dir_name = self.env_vars.get(DIR_PARAM_NAME.upper(), "")
        if self.uploaded_code:
            script_name = self.uploaded_code.script_name
            if self.repacked_model_data or self.enable_network_isolation():
                dir_name = "/opt/ml/model/code"
            else:
                dir_name = self.uploaded_code.s3_prefix
        elif self.entry_point is not None:
            script_name = self.entry_point
            if self.source_dir is not None:
                dir_name = (
                    self.source_dir
                    if self.source_dir.startswith("s3://")
                    else "file://" + self.source_dir
                )
        return {
            SCRIPT_PARAM_NAME.upper(): script_name,
            DIR_PARAM_NAME.upper(): dir_name,
            CONTAINER_LOG_LEVEL_PARAM_NAME.upper(): self.to_string(self.container_log_level),
            SAGEMAKER_REGION_PARAM_NAME.upper(): self.region,
        }


    def _is_mms_version(self):
        """Determines if the framework corresponds to an and using MMS.

        Whether the framework version corresponds to an inference image using
        the Multi-Model Server (https://github.com/awslabs/multi-model-server).

        Returns:
            bool: If the framework version corresponds to an image using MMS.
        """
        if self.framework_version is None:
            return False

        lowest_mms_version = packaging.version.Version(_LOWEST_MMS_VERSION)
        framework_version = packaging.version.Version(self.framework_version)
        return framework_version >= lowest_mms_version


    def _get_container_env(self):
        """Placeholder docstring."""
        if not self._container_log_level:
            return self.env

        if self._container_log_level not in self.LOG_LEVEL_MAP:
            logging.warning("ignoring invalid container log level: %s", self._container_log_level)
            return self.env

        env = dict(self.env)
        env[self.LOG_LEVEL_PARAM_NAME] = self.LOG_LEVEL_MAP[self._container_log_level]
        return env


    def _prepare_container_def_base(self):
        """Base container definition logic from your prepare_container_def_base.
            dict or list[dict]: A container definition object or list of container definitions
                usable with the CreateModel API.
        """
        # Handle pipeline models with multiple containers
        if isinstance(self.model, list):
            if not all(isinstance(m, Model) for m in self.model):
                raise ValueError(
                    "When model is a list, all elements must be sagemaker.core.resources.Model instances. "
                    "Found non-Model instances in the list."
                )
            return self._prepare_pipeline_container_defs()

        deploy_key_prefix = fw_utils.model_code_key_prefix(
            getattr(self, 'key_prefix', None),
            self.model_name,
            self.image_uri
        )

        deploy_env = copy.deepcopy(getattr(self, 'env_vars', {}))

        if (getattr(self, 'source_dir', None) or
            getattr(self, 'dependencies', None) or
            getattr(self, 'entry_point', None) or
            getattr(self, 'git_config', None)):

            self._upload_code(deploy_key_prefix, repack=getattr(self, 'is_repack', lambda: False)())
            deploy_env.update(self._script_mode_env_vars())

        # Determine model data URL: prioritize repacked > s3_upload_path > s3_model_data_url
        model_data_url = (getattr(self, 'repacked_model_data', None) or
                        getattr(self, 's3_upload_path', None) or
                        getattr(self, 's3_model_data_url', None))

        return container_def(
            self.image_uri,
            model_data_url,
            deploy_env,
            image_config=getattr(self, 'image_config', None),
            accept_eula=getattr(self, 'accept_eula', None),
            additional_model_data_sources=getattr(self, 'additional_model_data_sources', None),
            model_reference_arn=getattr(self, 'model_reference_arn', None),
        )


    def _handle_tf_repack(self, deploy_key_prefix, instance_type, serverless_inference_config):
        """Handle TensorFlow-specific repack logic."""
        bucket, key_prefix = s3.determine_bucket_and_prefix(
            bucket=getattr(self, 'bucket', None),
            key_prefix=deploy_key_prefix,
            sagemaker_session=self.sagemaker_session,
        )

        if self.entry_point and not is_pipeline_variable(getattr(self, 'model_data', None)):
            model_data = s3.s3_path_join("s3://", bucket, key_prefix, "model.tar.gz")

            repack_model(
                self.entry_point,
                getattr(self, 'source_dir', None),
                getattr(self, 'dependencies', None),
                getattr(self, 'model_data', None),
                model_data,
                self.sagemaker_session,
                kms_key=getattr(self, 'model_kms_key', None),
            )

            # Update model_data for container_def
            self.model_data = model_data

        elif self.entry_point and is_pipeline_variable(getattr(self, 'model_data', None)):
            # Handle pipeline variable case
            if isinstance(self.sagemaker_session, PipelineSession):
                self.sagemaker_session.context.need_runtime_repack.add(id(self))
                self.sagemaker_session.context.runtime_repack_output_prefix = s3.s3_path_join(
                    "s3://", bucket, key_prefix
                )
            else:
                logging.warning(
                    "The model_data is a Pipeline variable of type %s, "
                    "which should be used under `PipelineSession` and "
                    "leverage `ModelStep` to create or register model. "
                    "Otherwise some functionalities e.g. "
                    "runtime repack may be missing. For more, see: "
                    "https://sagemaker.readthedocs.io/en/stable/"
                    "amazon_sagemaker_model_building_pipeline.html#model-step",
                    type(getattr(self, 'model_data', None)),
                )


    def _prepare_container_def(self):
        """Unified container definition preparation for all frameworks."""
        if self.framework in [Framework.LDA, Framework.NTM, Framework.DJL, Framework.SPARKML] or self.framework is None:
            return self._prepare_container_def_base()

        # Framework-specific validations
        if self.framework == Framework.SKLEARN and self.accelerator_type:
            raise ValueError("Accelerator types are not supported for Scikit-Learn.")

        py_tuple = platform.python_version_tuple()
        self.py_version = f"py{py_tuple[0]}{py_tuple[1]}"

        # Image URI resolution
        deploy_image = self.image_uri
        if not deploy_image:
            if self.instance_type is None and self.serverless_inference_config is None:
                raise ValueError(
                    "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
                )

            # Framework-specific image retrieval parameters
            image_params = {
                "framework": self.framework.value,
                "region": self.region,
                "version": self.framework_version,
                "instance_type": self.instance_type,
                "accelerator_type": self.accelerator_type,
                "image_scope": "inference",
                "serverless_inference_config": self.serverless_inference_config,
            }

            # Add framework-specific parameters
            if self.framework in [Framework.PYTORCH, Framework.MXNET, Framework.CHAINER]:
                image_params["py_version"] = getattr(self, 'py_version', 'py3')
            elif self.framework == Framework.HUGGINGFACE:
                image_params["py_version"] = getattr(self, 'py_version', 'py3')
                # Use framework_version for both TensorFlow and PyTorch base versions
                if "tensorflow" in self.framework_version.lower():
                    image_params["base_framework_version"] = f"tensorflow{self.framework_version}"
                else:
                    image_params["base_framework_version"] = f"pytorch{self.framework_version}"
                if hasattr(self, 'inference_tool') and self.inference_tool:
                    image_params["inference_tool"] = self.inference_tool
            elif self.framework == Framework.SKLEARN:
                image_params["py_version"] = getattr(self, 'py_version', 'py3')

            deploy_image = image_uris.retrieve(**image_params)

        # Code upload logic
        deploy_key_prefix = model_code_key_prefix(
            getattr(self, 'key_prefix', None),
            self.model_name,
            deploy_image
        )

        # Framework-specific repack logic
        repack_logic = {
            Framework.PYTORCH: lambda: getattr(self, '_is_mms_version', lambda: False)(),
            Framework.MXNET: lambda: getattr(self, '_is_mms_version', lambda: False)(),
            Framework.CHAINER: lambda: True,
            Framework.XGBOOST: lambda: getattr(self, 'enable_network_isolation', lambda: False)(),
            Framework.SKLEARN: lambda: getattr(self, 'enable_network_isolation', lambda: False)(),
            Framework.HUGGINGFACE: lambda: True,
            Framework.TENSORFLOW: lambda: False,  # TF has special logic
        }

        if self.framework == Framework.TENSORFLOW:
            # TensorFlow has special repack logic
            self._handle_tf_repack(deploy_key_prefix, self.instance_type, self.serverless_inference_config)
        else:
            should_repack = repack_logic.get(self.framework, lambda: False)()
            self._upload_code(deploy_key_prefix, repack=should_repack)

        # Environment variables
        deploy_env = dict(getattr(self, 'env_vars', getattr(self, 'env', {})))

        # Add script mode env vars for frameworks that support it
        if self.framework != Framework.TENSORFLOW:  # TF handles this differently
            deploy_env.update(self._script_mode_env_vars())
        elif self.framework == Framework.TENSORFLOW:
            deploy_env = getattr(self, '_get_container_env', lambda: deploy_env)()

        # Add model server workers if supported
        if hasattr(self, 'model_server_workers') and self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = to_string(self.model_server_workers)

        # Model data resolution
        model_data_resolvers = {
            Framework.PYTORCH: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.MXNET: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.CHAINER: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.XGBOOST: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.SKLEARN: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.HUGGINGFACE: lambda: getattr(self, 'repacked_model_data', None) or getattr(self, 's3_upload_path', None) or getattr(self, 's3_model_data_url', None),
            Framework.TENSORFLOW: lambda: getattr(self, 'model_data', None),  # TF still has special handling
        }

        model_data = model_data_resolvers[self.framework]()

        # Build container definition
        container_params = {
            "image_uri": deploy_image,
            "model_data_url": model_data,
            "env": deploy_env,
            "accept_eula": getattr(self, 'accept_eula', None),
            "model_reference_arn": getattr(self, 'model_reference_arn', None),
        }

        # Add optional parameters if they exist
        if hasattr(self, 'image_config'):
            container_params["image_config"] = self.image_config
        if hasattr(self, 'additional_model_data_sources'):
            container_params["additional_model_data_sources"] = self.additional_model_data_sources

        return container_def(**container_params)

    def _prepare_pipeline_container_defs(self):
        """Prepare container definitions for inference pipeline.

        Extracts container definitions from sagemaker.core.resources.Model objects.

        Returns:
            list[dict]: List of container definitions.
        """
        containers = []
        for core_model in self.model:
            # Check if containers is set and is a list (not Unassigned)
            if hasattr(core_model, 'containers') and isinstance(core_model.containers, list):
                for c in core_model.containers:
                    containers.append(self._core_container_to_dict(c))
            elif hasattr(core_model, 'primary_container') and core_model.primary_container:
                containers.append(self._core_container_to_dict(core_model.primary_container))
        return containers

    def _core_container_to_dict(self, container):
        """Convert core ContainerDefinition to dict using container_def helper."""
        from sagemaker.core.utils.utils import Unassigned

        # Helper to check if value is Unassigned
        def get_value(obj, attr, default=None):
            if not hasattr(obj, attr):
                return default
            val = getattr(obj, attr)
            return default if isinstance(val, Unassigned) else val

        return container_def(
            container.image,
            get_value(container, 'model_data_url'),
            get_value(container, 'environment', {}),
            image_config=get_value(container, 'image_config'),
        )

    def _create_sagemaker_model(self):
        """Create a SageMaker Model Entity."""
        container_def = self._prepare_container_def()

        if not isinstance(self.sagemaker_session, PipelineSession):
            # _base_name, model_name are not needed under PipelineSession.
            # the model_data may be Pipeline variable
            # which may break the _base_name generation
            image_uri = container_def["Image"] if isinstance(container_def, dict) else container_def[0]["Image"]
            self._ensure_base_name_if_needed(
                image_uri=image_uri,
                script_uri=self.source_dir,
                model_uri=self._get_model_uri(),
            )

        self._init_sagemaker_session_if_does_not_exist(self.instance_type)
        # Depending on the instance type, a local session (or) a session is initialized.
        self.role_arn = resolve_value_from_config(
            self.role_arn,
            MODEL_EXECUTION_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.vpc_config = resolve_value_from_config(
            self.vpc_config,
            MODEL_VPC_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self._enable_network_isolation = resolve_value_from_config(
            self._enable_network_isolation,
            MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.env_vars = resolve_nested_dict_value_from_config(
            self.env_vars,
            ["Environment"],
            MODEL_CONTAINERS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        create_model_args = dict(
            name=self.model_name,
            role=self.role_arn,
            container_defs=container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=self._enable_network_isolation,
            tags=format_tags(self._tags),
        )
        self.sagemaker_session.create_model(**create_model_args)
        if isinstance(self.sagemaker_session, PipelineSession):
            return
        return Model.get(model_name=self.model_name, region=self.region)


    def _create_model(self):
        """Create a SageMaker Model instance from the current configuration."""
        if self._optimizing:
            return None

        execution_role = self.role_arn
        if not execution_role:
            execution_role = get_execution_role(self.sagemaker_session, use_default=True)
            self.role_arn = execution_role

        if self.mode == Mode.LOCAL_CONTAINER:
            from sagemaker.core.local.local_session import LocalSession
            local_session = LocalSession()

            primary_container = self._prepare_container_def()
            local_session.sagemaker_client.create_model(
                ModelName=self.model_name,
                PrimaryContainer=primary_container,
                ExecutionRoleArn=execution_role
            )

            return Model(
                model_name=self.model_name,
                primary_container=ContainerDefinition(
                    image=primary_container['Image'],
                    model_data_url=primary_container['ModelDataUrl'],
                    environment=primary_container['Environment']
                ),
                execution_role_arn=execution_role,
            )
        elif self.mode == Mode.IN_PROCESS:
            return Model(
                model_name=self.model_name,
                primary_container=ContainerDefinition(
                    image="dummy-in-process-image:latest",  # Not used in in-process mode
                    environment=self.env_vars or {},
                ),
                execution_role_arn=execution_role,
            )

        elif self.mode == Mode.SAGEMAKER_ENDPOINT:
            self._init_sagemaker_session_if_does_not_exist(self.instance_type)
            if not self.role_arn:
                self.role_arn = get_execution_role(self.sagemaker_session, use_default=True)

            self.role_arn = resolve_value_from_config(
                self.role_arn,
                MODEL_EXECUTION_ROLE_ARN_PATH,
                sagemaker_session=self.sagemaker_session,
            )
            self.vpc_config = resolve_value_from_config(
                self.vpc_config,
                MODEL_VPC_CONFIG_PATH,
                sagemaker_session=self.sagemaker_session,
            )
            self._enable_network_isolation = resolve_value_from_config(
                self._enable_network_isolation,
                MODEL_ENABLE_NETWORK_ISOLATION_PATH,
                sagemaker_session=self.sagemaker_session,
            )
            self._tags = format_tags(self._tags)

            if (
                getattr(self.sagemaker_session, "settings", None) is not None
                and self.sagemaker_session.settings.include_jumpstart_tags
            ):
                self._tags = add_jumpstart_uri_tags(
                    tags=self._tags,
                    inference_model_uri=(
                        self.s3_model_data_url if isinstance(self.s3_model_data_url, (str, dict)) else None
                    ),
                    inference_script_uri=self.source_dir,
                )

            if self.role_arn is None:
                raise ValueError("Role can not be null for deploying a model")

            return self._create_sagemaker_model()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.fetch_endpoint_names_for_base_model")
    def fetch_endpoint_names_for_base_model(self) -> Set[str]:
        """Fetches endpoint names for the base model.

        Returns:
            Set of endpoint names for the base model.
        """
        from sagemaker.core.resources import Tag as CoreTag
        if not self._is_model_customization():
            raise ValueError("This functionality is only supported for Model Customization use cases")
        recipe_name = self._fetch_model_package().inference_specification.containers[0].base_model.recipe_name
        endpoint_names = set()
        logger.error(f"recipe_name: {recipe_name}")
        for inference_component in InferenceComponent.get_all():
            logger.error(f"checking for {inference_component.inference_component_arn}")
            tags = CoreTag.get_all(resource_arn=inference_component.inference_component_arn)

            for tag in tags:
                if tag.key == "Base" and tag.value == recipe_name:
                    endpoint_names.add(inference_component.endpoint_name)
                    continue

        return endpoint_names

    def _build_single_modelbuilder(
        self,
        mode: Optional[Mode] = None,
        role_arn: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Model:
        """Create a deployable Model instance for single model deployment."""

        # Handle pipeline models early - they don't need normal model building
        if isinstance(self.model, list):
            if not all(isinstance(m, Model) for m in self.model):
                raise ValueError(
                    "When model is a list, all elements must be sagemaker.core.resources.Model instances. "
                    "Found non-Model instances in the list."
                )
            self.built_model = self._create_model()
            return self.built_model

        if mode:
            self.mode = mode
        if role_arn:
            self.role_arn = role_arn

        self.serve_settings = self._get_serve_setting()

        # Handle model customization (fine-tuned models)
        if self._is_model_customization():
            if mode is not None and mode != Mode.SAGEMAKER_ENDPOINT:
                raise ValueError("Only SageMaker Endpoint Mode is supported for Model Customization use cases")
            model_package = self._fetch_model_package()
            # Fetch recipe config first to set image_uri, instance_type, and s3_upload_path
            self._fetch_and_cache_recipe_config()
            self.s3_upload_path = model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri
            container_def = ContainerDefinition(
                image=self.image_uri,
                model_data_source={
                    "s3_data_source": {
                        "s3_uri": f"{self.s3_upload_path}/",
                        "s3_data_type": "S3Prefix",
                        "compression_type": "None"
                    }
                }
            )
            model_name = self.model_name or f"model-{uuid.uuid4().hex[:10]}"
            # Create model
            self.built_model = Model.create(
                execution_role_arn=self.role_arn,
                model_name=model_name,
                containers=[container_def]
            )
            return self.built_model

        self._serializer, self._deserializer = self._get_client_translators()
        self.modes = dict()

        if isinstance(self.model, TrainingJob):
            self.model_path = self.model.model_artifacts.s3_model_artifacts
            self.model = None
        elif isinstance(self.model, ModelTrainer):
            # Check if this is a JumpStart ModelTrainer
            if hasattr(self.model, '_jumpstart_config') and self.model._jumpstart_config is not None:
                # For JumpStart ModelTrainer, extract model_id and route to JumpStart flow
                jumpstart_config = self.model._jumpstart_config
                self.model_path = self.model._latest_training_job.model_artifacts.s3_model_artifacts
                self.model = jumpstart_config.model_id  # Set to model_id for JumpStart detection
                self.model_version = jumpstart_config.model_version
                self.accept_eula = jumpstart_config.accept_eula
                # Set internal flag to indicate this came from ModelTrainer
                self._from_jumpstart_model_trainer = True
            else:
                # Non-JumpStart ModelTrainer - use V2 flow
                self.model_path = self.model._latest_training_job.model_artifacts.s3_model_artifacts
                self.model = None

        self.sagemaker_session = sagemaker_session or self.sagemaker_session or self._create_session_with_region()
        self.sagemaker_session.settings._local_download_dir = self.model_path

        client = self.sagemaker_session.sagemaker_client
        client._user_agent_creator.to_string = self._user_agent_decorator(
            self.sagemaker_session.sagemaker_client._user_agent_creator.to_string
        )

        self._is_custom_image_uri = self.image_uri is not None


        self._handle_mlflow_input()
        self._build_validations()


        if self.env_vars.get("HUGGING_FACE_HUB_TOKEN") and not self.env_vars.get("HF_TOKEN"):
            self.env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
        elif self.env_vars.get("HF_TOKEN") and not self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
            self.env_vars["HUGGING_FACE_HUB_TOKEN"] = self.env_vars.get("HF_TOKEN")


        if getattr(self, '_passthrough', False):
            self.built_model = self._build_for_passthrough()
            return self.built_model

        if self.model_server and not (isinstance(self.model, str) and self._is_jumpstart_model_id()):
            self.built_model = self._build_for_model_server()
            return self.built_model


        if isinstance(self.model, str):
            model_task = None

            if self._is_jumpstart_model_id():
                if self.mode == Mode.IN_PROCESS:
                    raise ValueError(
                        f"{self.mode} is not supported for JumpStart models. "
                        "Please use LOCAL_CONTAINER mode to deploy a JumpStart model locally."
                    )
                self.model_hub = ModelHub.JUMPSTART
                logger.debug("Building for JumpStart model ID...")
                self.built_model = self._build_for_jumpstart()
                return self.built_model


            if self.mode != Mode.IN_PROCESS and self._use_jumpstart_equivalent():
                self.model_hub = ModelHub.JUMPSTART
                logger.debug("Building for JumpStart equivalent model ID...")
                self.built_model = self._build_for_jumpstart()
                return self.built_model

            if self._is_huggingface_model():
                self.model_hub = ModelHub.HUGGINGFACE

                if self.model_metadata:
                    model_task = self.model_metadata.get("HF_TASK")

                if self.model_server == ModelServer.DJL_SERVING:
                    self.built_model = self._build_for_djl()
                    return self.built_model
                else:
                    hf_model_md = self.get_huggingface_model_metadata(
                        self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                    )

                    if model_task is None:
                        model_task = hf_model_md.get("pipeline_tag")


                    if self.schema_builder is None and model_task is not None:
                        self._hf_schema_builder_init(model_task)


                    if model_task == "text-generation":
                        self.built_model = self._build_for_tgi()
                        return self.built_model
                    elif model_task in ["sentence-similarity", "feature-extraction"]:
                        self.built_model = self._build_for_tei()
                        return self.built_model
                    else:
                        self.built_model = self._build_for_transformers()
                        return self.built_model

            raise ValueError(f"Model {self.model} is not detected as HuggingFace or JumpStart model")


        if not self.model_server:
            if self.image_uri and is_1p_image_uri(self.image_uri):
                self.model_server = ModelServer.TRITON
                self.built_model = self._build_for_triton()
            else:
                self.model_server = ModelServer.TORCHSERVE
                self.built_model = self._build_for_torchserve()
            return self.built_model

        raise ValueError(f"Model server {self.model_server} is not supported")

    def _extract_and_extend_tags_from_model_trainer(self):
        if not isinstance(self.model, ModelTrainer):
            return

        # Check if tags attribute exists and is not None
        if not hasattr(self.model, 'tags') or not self.model.tags:
            return

        jumpstart_tags = [
            tag for tag in self.model.tags
            if tag.key in ["sagemaker-sdk:jumpstart-model-id", "sagemaker-sdk:jumpstart-model-version"]
        ]

        self._tags.extend(jumpstart_tags)


    def _deploy_local_endpoint(self, **kwargs):
        """Deploy the built model to a local endpoint."""

        # Extract parameters
        endpoint_name = kwargs.get("endpoint_name", getattr(self, 'endpoint_name', None))
        if "endpoint_name" in kwargs:
            self.endpoint_name = endpoint_name

        update_endpoint = kwargs.get("update_endpoint", False)

        endpoint_name = endpoint_name or self.endpoint_name

        from sagemaker.core.local.local_session import LocalSession
        local_session = LocalSession()
        endpoint_exists = False

        try:
            _ = local_session.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            endpoint_exists = True
        except Exception:
            endpoint_exists = False

        if not endpoint_exists:
            return LocalEndpoint.create(
                endpoint_name=endpoint_name,
                endpoint_config_name=None,
                model_server=self.model_server,
                local_model=self.built_model,
                local_session=local_session,
                container_timeout_seconds=kwargs.get("container_timeout_in_seconds", 300),
                secret_key=self.secret_key,
                local_container_mode_obj=self.modes[str(Mode.LOCAL_CONTAINER)],
                serializer=self._serializer,
                deserializer=self._deserializer,
                container_config=self.container_config
                )
        else:
            if update_endpoint:
                raise NotImplementedError("Update endpoint is not supported in local mode (V2 parity)")
            else:
                return LocalEndpoint.get(
                    endpoint_name=endpoint_name,
                    local_session=local_session
                )

    def _wait_for_endpoint(self, endpoint, poll=30, live_logging=False, show_progress=True, wait=True):
        """Enhanced wait with rich progress bar and status logging"""
        if not wait:
            logger.info("🚀 Deployment started: Endpoint '%s' using %s in %s mode (deployment in progress)",
                       endpoint, self.model_server, self.mode)
            return

        # Use the ModelBuilder's sagemaker_session client (which has correct region)
        sagemaker_client = self.sagemaker_session.sagemaker_client

        if show_progress and not live_logging:
            from sagemaker.serve.deployment_progress import (
                EndpointDeploymentProgress,
                _deploy_done_with_progress,
                _live_logging_deploy_done_with_progress
            )

            with EndpointDeploymentProgress(endpoint) as progress:
                # Check if we have permission for live logging
                from sagemaker.core.helper.session_helper import _has_permission_for_live_logging
                if _has_permission_for_live_logging(self.sagemaker_session.boto_session, endpoint):
                    # Use live logging with Rich progress tracker
                    cloudwatch_client = self.sagemaker_session.boto_session.client("logs")
                    paginator = cloudwatch_client.get_paginator("filter_log_events")
                    from sagemaker.core.helper.session_helper import create_paginator_config, EP_LOGGER_POLL
                    paginator_config = create_paginator_config()
                    desc = _wait_until(
                        lambda: _live_logging_deploy_done_with_progress(
                            sagemaker_client, endpoint, paginator, paginator_config, EP_LOGGER_POLL, progress
                        ),
                        poll=EP_LOGGER_POLL,
                    )
                else:
                    # Fallback to status-only progress
                    desc = _wait_until(
                        lambda: _deploy_done_with_progress(sagemaker_client, endpoint, progress),
                        poll
                    )
        else:
            # Existing implementation
            desc = _wait_until(lambda: _deploy_done(sagemaker_client, endpoint), poll)

        # Check final endpoint status and log accordingly
        try:
            endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint)
            endpoint_status = endpoint_desc['EndpointStatus']
            if endpoint_status == 'InService':
                endpoint_arn_info = f" (ARN: {endpoint_desc['EndpointArn']})" if self.mode == Mode.SAGEMAKER_ENDPOINT else ""
                logger.info("✅ Deployment successful: Endpoint '%s' using %s in %s mode%s",
                           endpoint, self.model_server, self.mode, endpoint_arn_info)
            else:
                logger.error("❌ Deployment failed: Endpoint '%s' status is '%s'", endpoint, endpoint_status)
        except Exception as e:
            logger.error("❌ Deployment failed: Unable to verify endpoint status - %s", str(e))

        return desc


    def _deploy_core_endpoint(self, **kwargs):
        # Extract and update self parameters
        initial_instance_count = kwargs.get("initial_instance_count", getattr(self, 'instance_count', None))
        if "initial_instance_count" in kwargs:
            self.instance_count = initial_instance_count

        instance_type = kwargs.get("instance_type", getattr(self, 'instance_type', None))
        if "instance_type" in kwargs:
            self.instance_type = instance_type

        accelerator_type = kwargs.get("accelerator_type", getattr(self, 'accelerator_type', None))
        if "accelerator_type" in kwargs:
            self.accelerator_type = accelerator_type

        endpoint_name = kwargs.get("endpoint_name", getattr(self, 'endpoint_name', None))
        if "endpoint_name" in kwargs:
            self.endpoint_name = endpoint_name

        tags = kwargs.get("tags", getattr(self, '_tags', None))
        if "tags" in kwargs:
            self._tags = tags

        kms_key = kwargs.get("kms_key", getattr(self, 'kms_key', None))
        if "kms_key" in kwargs:
            self.kms_key = kms_key

        async_inference_config = kwargs.get("async_inference_config", getattr(self, 'async_inference_config', None))
        if "async_inference_config" in kwargs:
            self.async_inference_config = async_inference_config

        serverless_inference_config = kwargs.get("serverless_inference_config", getattr(self, 'serverless_inference_config', None))
        if "serverless_inference_config" in kwargs:
            self.serverless_inference_config = serverless_inference_config

        model_data_download_timeout = kwargs.get("model_data_download_timeout", getattr(self, 'model_data_download_timeout', None))
        if "model_data_download_timeout" in kwargs:
            self.model_data_download_timeout = model_data_download_timeout

        resources = kwargs.get("resources", getattr(self, 'resource_requirements', None))
        if "resources" in kwargs:
            self.resource_requirements = resources

        inference_component_name = kwargs.get("inference_component_name", getattr(self, 'inference_component_name', None))
        if "inference_component_name" in kwargs:
            self.inference_component_name = inference_component_name

        container_startup_health_check_timeout = kwargs.get("container_startup_health_check_timeout", getattr(self, 'container_startup_health_check_timeout', None))
        inference_ami_version = kwargs.get("inference_ami_version", getattr(self, 'inference_ami_version', None))

        serializer = kwargs.get("serializer", None)
        deserializer = kwargs.get("deserializer", None)

        # Override _serializer and _deserializer if provided
        if serializer:
            self._serializer = serializer
        if deserializer:
            self._deserializer = deserializer

        data_capture_config = kwargs.get("data_capture_config", None)
        volume_size = kwargs.get("volume_size", None)
        inference_recommendation_id = kwargs.get("inference_recommendation_id", None)
        explainer_config = kwargs.get("explainer_config", None)
        endpoint_logging = kwargs.get("endpoint_logging", False)
        endpoint_type = kwargs.get("endpoint_type", EndpointType.MODEL_BASED)
        managed_instance_scaling = kwargs.get("managed_instance_scaling", None)
        routing_config = kwargs.get("routing_config", None)
        update_endpoint = kwargs.get("update_endpoint", False)
        wait = kwargs.get("wait", True)

        if tags:
            self.add_tags(tags)
            tags = format_tags(self._tags)
        else:
            tags = format_tags(self._tags)

        routing_config = _resolve_routing_config(routing_config)


        if (inference_recommendation_id is not None or self.inference_recommender_job_results is not None):
            instance_type, initial_instance_count = self._update_params(
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
                async_inference_config=async_inference_config,
                serverless_inference_config=serverless_inference_config,
                explainer_config=explainer_config,
                inference_recommendation_id=inference_recommendation_id,
                inference_recommender_job_results=self.inference_recommender_job_results,
            )


        is_async = async_inference_config is not None
        if is_async and not isinstance(async_inference_config, AsyncInferenceConfig):
            raise ValueError("async_inference_config needs to be a AsyncInferenceConfig object")

        is_explainer_enabled = explainer_config is not None
        if is_explainer_enabled and not isinstance(explainer_config, ExplainerConfig):
            raise ValueError("explainer_config needs to be a ExplainerConfig object")

        is_serverless = serverless_inference_config is not None
        if not is_serverless and not (instance_type and initial_instance_count):
            raise ValueError("Must specify instance type and instance count unless using serverless inference")

        if is_serverless and not isinstance(serverless_inference_config, ServerlessInferenceConfig):
            raise ValueError("serverless_inference_config needs to be a ServerlessInferenceConfig object")


        if self._is_sharded_model:
            if endpoint_type != EndpointType.INFERENCE_COMPONENT_BASED:
                logger.warning(
                    "Forcing INFERENCE_COMPONENT_BASED endpoint for sharded model. ADVISORY - "
                    "Use INFERENCE_COMPONENT_BASED endpoints over MODEL_BASED endpoints."
                )
                endpoint_type = EndpointType.INFERENCE_COMPONENT_BASED

            if self._enable_network_isolation:
                raise ValueError(
                    "EnableNetworkIsolation cannot be set to True since SageMaker Fast Model "
                    "Loading of model requires network access."
                )

            if resources and resources.num_cpus and resources.num_cpus > 0:
                logger.warning(
                    "NumberOfCpuCoresRequired should be 0 for the best experience with SageMaker "
                    "Fast Model Loading. Configure by setting `num_cpus` to 0 in `resources`."
                )

        if self.role_arn is None:
            raise ValueError("Role can not be null for deploying a model")

        routing_config = _resolve_routing_config(routing_config)

        if (
            inference_recommendation_id is not None
            or self.inference_recommender_job_results is not None
        ):
            instance_type, initial_instance_count = self._update_params(
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
                async_inference_config=async_inference_config,
                serverless_inference_config=serverless_inference_config,
                explainer_config=explainer_config,
                inference_recommendation_id=inference_recommendation_id,
                inference_recommender_job_results=self.inference_recommender_job_results,
            )

        is_async = async_inference_config is not None
        if is_async and not isinstance(async_inference_config, AsyncInferenceConfig):
            raise ValueError("async_inference_config needs to be a AsyncInferenceConfig object")

        is_explainer_enabled = explainer_config is not None
        if is_explainer_enabled and not isinstance(explainer_config, ExplainerConfig):
            raise ValueError("explainer_config needs to be a ExplainerConfig object")

        is_serverless = serverless_inference_config is not None
        if not is_serverless and not (instance_type and initial_instance_count):
            raise ValueError(
                "Must specify instance type and instance count unless using serverless inference"
            )

        if is_serverless and not isinstance(serverless_inference_config, ServerlessInferenceConfig):
            raise ValueError(
                "serverless_inference_config needs to be a ServerlessInferenceConfig object"
            )

        if self._is_sharded_model:
            if endpoint_type != EndpointType.INFERENCE_COMPONENT_BASED:
                logger.warning(
                    "Forcing INFERENCE_COMPONENT_BASED endpoint for sharded model. ADVISORY - "
                    "Use INFERENCE_COMPONENT_BASED endpoints over MODEL_BASED endpoints."
                )
                endpoint_type = EndpointType.INFERENCE_COMPONENT_BASED

            if self._enable_network_isolation:
                raise ValueError(
                    "EnableNetworkIsolation cannot be set to True since SageMaker Fast Model "
                    "Loading of model requires network access."
                )

            if resources and resources.num_cpus and resources.num_cpus > 0:
                logger.warning(
                    "NumberOfCpuCoresRequired should be 0 for the best experience with SageMaker "
                    "Fast Model Loading. Configure by setting `num_cpus` to 0 in `resources`."
                )


        if endpoint_type == EndpointType.INFERENCE_COMPONENT_BASED:
            if update_endpoint:
                raise ValueError(
                    "Currently update_endpoint is supported for single model endpoints"
                )
            if endpoint_name:
                self.endpoint_name = endpoint_name
            else:

                if self.model_name:
                    self.endpoint_name = name_from_base(self.model_name)
            # [TODO]: Refactor to a module
            managed_instance_scaling_config = {}
            if managed_instance_scaling:
                managed_instance_scaling_config["Status"] = "ENABLED"
                if "MaxInstanceCount" in managed_instance_scaling:
                    managed_instance_scaling_config["MaxInstanceCount"] = managed_instance_scaling[
                        "MaxInstanceCount"
                    ]
                if "MinInstanceCount" in managed_instance_scaling:
                    managed_instance_scaling_config["MinInstanceCount"] = managed_instance_scaling[
                        "MinInstanceCount"
                    ]
                else:
                    managed_instance_scaling_config["MinInstanceCount"] = initial_instance_count


            if not self.sagemaker_session.endpoint_in_service_or_not(self.endpoint_name):
                production_variant = session_helper.production_variant(
                    instance_type=instance_type,
                    initial_instance_count=initial_instance_count,
                    volume_size=volume_size,
                    model_data_download_timeout=model_data_download_timeout,
                    container_startup_health_check_timeout=container_startup_health_check_timeout,
                    managed_instance_scaling=managed_instance_scaling_config,
                    routing_config=routing_config,
                    inference_ami_version=inference_ami_version,
                )

                self.sagemaker_session.endpoint_from_production_variants(
                    name=self.endpoint_name,
                    production_variants=[production_variant],
                    tags=tags,
                    kms_key=kms_key,
                    vpc_config=self.vpc_config,
                    enable_network_isolation=self._enable_network_isolation,
                    role=self.role_arn,
                    live_logging=False,  # TODO: enable when IC supports this
                    wait=False,
                )
                self._wait_for_endpoint(endpoint=self.endpoint_name, show_progress=True, wait=wait)


            core_endpoint = Endpoint.get(
                endpoint_name=self.endpoint_name,
                session=self.sagemaker_session.boto_session,
                region=self.region
            )

            # [TODO]: Refactor to a module
            startup_parameters = {}
            if model_data_download_timeout:
                startup_parameters["ModelDataDownloadTimeoutInSeconds"] = (
                    model_data_download_timeout
                )
            if container_startup_health_check_timeout:
                startup_parameters["ContainerStartupHealthCheckTimeoutInSeconds"] = (
                    container_startup_health_check_timeout
                )

            inference_component_spec = {
                "ModelName": self.built_model.model_name,
                "StartupParameters": startup_parameters,
                "ComputeResourceRequirements": resources.get_compute_resource_requirements(),
            }
            runtime_config = {"CopyCount": resources.copy_count}
            self.inference_component_name = (
                inference_component_name
                or self.inference_component_name
                or unique_name_from_base(self.model_name)
            )

            # [TODO]: Add endpoint_logging support
            self.sagemaker_session.create_inference_component(
                inference_component_name=self.inference_component_name,
                endpoint_name=self.endpoint_name,
                variant_name="AllTraffic",  # default variant name
                specification=inference_component_spec,
                runtime_config=runtime_config,
                tags=tags,
                wait=False,
            )
            self._wait_for_endpoint(endpoint=self.endpoint_name, show_progress=True, wait=wait)

            return core_endpoint

        else:

            serverless_inference_config_dict = (
                serverless_inference_config._to_request_dict() if is_serverless else None
            )
            production_variant = session_helper.production_variant(
                self.model_name,
                instance_type,
                initial_instance_count,
                accelerator_type=accelerator_type,
                serverless_inference_config=serverless_inference_config_dict,
                volume_size=volume_size,
                model_data_download_timeout=model_data_download_timeout,
                container_startup_health_check_timeout=container_startup_health_check_timeout,
                routing_config=routing_config,
                inference_ami_version=inference_ami_version,
            )
            if endpoint_name:
                self.endpoint_name = endpoint_name
            else:
                base_endpoint_name = base_from_name(self.model_name)
                self.endpoint_name = name_from_base(base_endpoint_name)

            data_capture_config_dict = None
            if data_capture_config is not None:
                data_capture_config_dict = data_capture_config._to_request_dict()

            async_inference_config_dict = None
            if is_async:
                if (
                    async_inference_config.output_path is None
                    or async_inference_config.failure_path is None
                ):
                    async_inference_config = self._build_default_async_inference_config(
                        async_inference_config
                    )
                async_inference_config.kms_key_id = resolve_value_from_config(
                    async_inference_config.kms_key_id,
                    ENDPOINT_CONFIG_ASYNC_KMS_KEY_ID_PATH,
                    sagemaker_session=self.sagemaker_session,
                )
                async_inference_config_dict = async_inference_config._to_request_dict()

            explainer_config_dict = None
            if is_explainer_enabled:
                explainer_config_dict = explainer_config._to_request_dict()

            if update_endpoint:
                endpoint_config_name = self.sagemaker_session.create_endpoint_config(
                    name=self.model_name,
                    model_name=self.model_name,
                    initial_instance_count=initial_instance_count,
                    instance_type=instance_type,
                    accelerator_type=accelerator_type,
                    tags=tags,
                    kms_key=kms_key,
                    data_capture_config_dict=data_capture_config_dict,
                    volume_size=volume_size,
                    model_data_download_timeout=model_data_download_timeout,
                    container_startup_health_check_timeout=container_startup_health_check_timeout,
                    explainer_config_dict=explainer_config_dict,
                    async_inference_config_dict=async_inference_config_dict,
                    serverless_inference_config_dict=serverless_inference_config_dict,
                    routing_config=routing_config,
                    inference_ami_version=inference_ami_version,
                )
                self.sagemaker_session.update_endpoint(self.endpoint_name, endpoint_config_name)
            else:
                self.sagemaker_session.endpoint_from_production_variants(
                    name=self.endpoint_name,
                    production_variants=[production_variant],
                    tags=tags,
                    kms_key=kms_key,
                    wait=False,
                    data_capture_config_dict=data_capture_config_dict,
                    explainer_config_dict=explainer_config_dict,
                    async_inference_config_dict=async_inference_config_dict,
                    live_logging=endpoint_logging,
                )
                self._wait_for_endpoint(endpoint=self.endpoint_name, show_progress=True, wait=wait)

            # Create and return Endpoint
            core_endpoint = Endpoint.get(
                endpoint_name=self.endpoint_name,
                session=self.sagemaker_session.boto_session,
                region=self.region
            )

            return core_endpoint


    def _deploy(self, **kwargs):
        self.accept_eula = kwargs.get("accept_eula", getattr(self, 'accept_eula', False))
        self.built_model = kwargs.get("built_model", getattr(self, 'built_model', None))

        if not hasattr(self, 'built_model') or self.built_model is None:
            raise ValueError("Must call build() before deploy()")

        if hasattr(self, 'model_server') and self.model_server:
            wrapper_method = self._get_deploy_wrapper()
            if wrapper_method:
                endpoint = wrapper_method(**kwargs)
                return endpoint

        if self.mode == Mode.LOCAL_CONTAINER:
            endpoint = self._deploy_local_endpoint(**kwargs)
        elif self.mode == Mode.SAGEMAKER_ENDPOINT:
            endpoint = self._deploy_core_endpoint(**kwargs)
        elif self.mode == Mode.IN_PROCESS:
            endpoint = LocalEndpoint.create(
                endpoint_name=kwargs.get("endpoint_name"),
                model_server=self.model_server,
                in_process_mode=True,
                local_model=self.built_model,
                container_timeout_seconds=kwargs.get("container_timeout_in_seconds", 300),
                secret_key=self.secret_key,
                in_process_mode_obj=self.modes[str(Mode.IN_PROCESS)],
                serializer=self._serializer,
                deserializer=self._deserializer,
                container_config=self.container_config
                )
        else:
            raise ValueError(f"Deployment mode {self.mode} not supported")

        return endpoint


    def _get_deploy_wrapper(self):
        """Get the appropriate deploy wrapper method for the current model server."""
        if isinstance(self.model, str) and self._is_jumpstart_model_id():
            return self._js_builder_deploy_wrapper

        wrapper_map = {
            ModelServer.DJL_SERVING: self._djl_model_builder_deploy_wrapper,
            ModelServer.TGI: self._tgi_model_builder_deploy_wrapper,
            ModelServer.TEI: self._tei_model_builder_deploy_wrapper,
            ModelServer.MMS: self._transformers_model_builder_deploy_wrapper,
        }

        if self.model_server in wrapper_map:
            return wrapper_map.get(self.model_server)
        return None

    def _does_ic_exist(self, ic_name: str) -> bool:
        """Check if inference component exists."""
        try:
            self.sagemaker_session.describe_inference_component(inference_component_name=ic_name)
            return True
        except ClientError as e:
            return "Could not find inference component" not in e.response["Error"]["Message"]


    def _update_inference_component(self, ic_name: str, resource_requirements: ResourceRequirements, **kwargs):
        """Update existing inference component."""
        startup_parameters = {}
        if kwargs.get("model_data_download_timeout"):
            startup_parameters["ModelDataDownloadTimeoutInSeconds"] = kwargs["model_data_download_timeout"]
        if kwargs.get("container_timeout_in_seconds"):
            startup_parameters["ContainerStartupHealthCheckTimeoutInSeconds"] = kwargs["container_timeout_in_seconds"]

        compute_rr = resource_requirements.get_compute_resource_requirements()
        inference_component_spec = {
            "ModelName": self.model_name,
            "StartupParameters": startup_parameters,
            "ComputeResourceRequirements": compute_rr,
        }
        runtime_config = {"CopyCount": resource_requirements.copy_count}

        return self.sagemaker_session.update_inference_component(
            inference_component_name=ic_name,
            specification=inference_component_spec,
            runtime_config=runtime_config
        )

    def _deploy_for_ic(
        self,
        ic_data: Dict[str, Any],
        endpoint_name: str,
        **kwargs
    ) -> Endpoint:
        """Deploy/update inference component and return V3 Endpoint."""
        ic_name = ic_data.get("Name")
        resource_requirements = ic_data.get("ResourceRequirements")
        built_model = ic_data.get("Model")

        if self._does_ic_exist(ic_name):
            # Update existing IC
            self._update_inference_component(ic_name, resource_requirements, **kwargs)
            # Return existing endpoint
            return Endpoint.get(
                endpoint_name=endpoint_name,
                session=self.sagemaker_session.boto_session,
                region=self.region
            )
        else:
            # Create new IC via _deploy()
            return self._deploy(
                built_model=built_model,
                endpoint_name=endpoint_name,
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=resource_requirements,
                inference_component_name=ic_name,
                instance_type=kwargs.get('instance_type', self.instance_type),
                initial_instance_count=kwargs.get('initial_instance_count', 1),
                **kwargs
            )

    def _reset_build_state(self):
        """Reset all dynamically added build-related state."""
        # Core build state
        self.built_model = None
        self.secret_key = ""

        # JumpStart preparation flags
        for attr in ['prepared_for_djl', 'prepared_for_tgi', 'prepared_for_mms']:
            if hasattr(self, attr):
                delattr(self, attr)

        # JumpStart cached data
        for attr in ['js_model_config', 'existing_properties', '_cached_js_model_specs', '_cached_is_jumpstart']:
            if hasattr(self, attr):
                delattr(self, attr)

        # HuggingFace cached data
        if hasattr(self, 'hf_model_config'):
            delattr(self, 'hf_model_config')

        # Mode and serving state
        if hasattr(self, 'modes'):
            delattr(self, 'modes')
        if hasattr(self, 'serve_settings'):
            delattr(self, 'serve_settings')

        # Serialization state
        for attr in ['_serializer', '_deserializer']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Upload/packaging state
        self.s3_model_data_url = None
        self.s3_upload_path = None
        for attr in ['uploaded_code', 'repacked_model_data']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Image and passthrough flags
        for attr in ['_is_custom_image_uri', '_passthrough']:
            if hasattr(self, attr):
                delattr(self, attr)

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.build")
    @runnable_by_pipeline
    def build(
        self,
        model_name: Optional[str] = None,
        mode: Optional[Mode] = None,
        role_arn: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> Union[Model, "ModelBuilder", None]:
        """Build a deployable ``Model`` instance with ``ModelBuilder``.

        Creates a SageMaker ``Model`` resource with the appropriate container image,
        model artifacts, and configuration. This method prepares the model for deployment
        but does not deploy it to an endpoint. Use the deploy() method to create an endpoint.

        Note: This returns a ``sagemaker.core.resources.Model`` object, not the deprecated
        PySDK Model class.

        Args:
            model_name (str, optional): The name for the SageMaker model. If not specified,
                a unique name will be generated. (Default: None).
            mode (Mode, optional): The mode of operation. Options are SAGEMAKER_ENDPOINT,
                LOCAL_CONTAINER, or IN_PROCESS. (Default: None, uses mode from initialization).
            role_arn (str, optional): The IAM role ARN for SageMaker to assume when creating
                the model and endpoint. (Default: None).
            sagemaker_session (Session, optional): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified,
                uses the session from initialization or creates one using the default AWS
                configuration chain. (Default: None).
            region (str, optional): The AWS region for deployment. If specified and different
                from the current region, a new session will be created. (Default: None).

        Returns:
            Union[Model, ModelBuilder, None]: A ``sagemaker.core.resources.Model`` resource
                that represents the created SageMaker model, or a ``ModelBuilder`` instance
                for multi-model scenarios.

        Example:
            >>> model_builder = ModelBuilder(model=my_model, role_arn=role)
            >>> model = model_builder.build()  # Creates Model resource
            >>> endpoint = model_builder.deploy()  # Creates Endpoint resource
            >>> result = endpoint.invoke(data=input_data)
        """
        if hasattr(self, 'built_model') and self.built_model is not None:
            logger.warning(
                "ModelBuilder.build() has already been called. "
                "Reusing ModelBuilder objects is not recommended and may cause issues. "
                "Please create a new ModelBuilder instance for additional builds."
            )

            # Reset build variables if user chooses to do this. Cannot guarantee it will work
            self._reset_build_state()

        # Validate and set region
        if region and region != self.region:
            logger.warning("Changing region from '%s' to '%s' during build()", self.region, region)
            self.region = region
            # Recreate session with new region
            self.sagemaker_session = self._create_session_with_region()

        # Validate and set role_arn
        if role_arn and role_arn != self.role_arn:
            logger.debug("Updating role_arn during build()")
            self.role_arn = role_arn

        self.model_name = model_name or getattr(self, 'model_name', None)
        self.mode = mode or getattr(self, 'mode', None)
        self.instance_type = getattr(self, 'instance_type', None)
        self.s3_model_data_url = getattr(self, 's3_model_data_url', None)
        self.sagemaker_session = sagemaker_session or getattr(self, 'sagemaker_session', None) or self._create_session_with_region()
        self.framework = getattr(self, 'framework', None)
        self.framework_version = getattr(self, 'framework_version', None)
        self.git_config = getattr(self, 'git_config', None)
        self.model_kms_key = getattr(self, 'model_kms_key', None)
        self.model_server_workers = getattr(self, 'model_server_workers', None)
        self.serverless_inference_config = getattr(self, 'serverless_inference_config', None)
        self.accelerator_type = getattr(self, 'accelerator_type', None)
        self.model_reference_arn = getattr(self, 'model_reference_arn', None)
        self.accept_eula = getattr(self, 'accept_eula', None)
        self.container_log_level = getattr(self, 'container_log_level', None)

        deployables = {}

        if not self.modelbuilder_list and not isinstance(
            self.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)
        ):
            self.serve_settings = self._get_serve_setting()
            model = self._build_single_modelbuilder(
                mode=self.mode,
                role_arn=self.role_arn,
                sagemaker_session=self.sagemaker_session,
            )
            model_arn_info = f" (ARN: {self.built_model.model_arn})" if self.mode == Mode.SAGEMAKER_ENDPOINT and hasattr(self.built_model, 'model_arn') else ""
            logger.info("✅ Model has been created: '%s' using server %s in %s mode%s", self.model_name, self.model_server, self.mode, model_arn_info)
            return model


        built_ic_models = []
        if self.modelbuilder_list:
            logger.debug("Detected ModelBuilders in modelbuilder_list.")


            for mb in self.modelbuilder_list:
                if mb.mode == Mode.IN_PROCESS or mb.mode == Mode.LOCAL_CONTAINER:
                    raise ValueError(
                        "Bulk ModelBuilder building is only supported for SageMaker Endpoint Mode."
                    )

                if (not mb.resource_requirements and not mb.inference_component_name) and (
                    not mb.inference_spec
                    or not isinstance(
                        mb.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)
                    )
                ):
                    raise ValueError(
                        "Bulk ModelBuilder building is only supported for Inference Components "
                        + "and custom orchestrators."
                    )


            for mb in self.modelbuilder_list:

                mb.serve_settings = mb._get_serve_setting()

                logger.debug("Building ModelBuilder %s.", mb.model_name)


                mb = mb._get_inference_component_resource_requirements(mb=mb)

                built_model = mb._build_single_modelbuilder(
                    role_arn=self.role_arn, sagemaker_session=self.sagemaker_session
                )
                built_ic_models.append(
                    {
                        "Name": mb.inference_component_name,
                        "ResourceRequirements": mb.resource_requirements,
                        "Model": built_model,
                    }
                )
                model_arn_info = f" (ARN: {mb.built_model.model_arn})" if mb.mode == Mode.SAGEMAKER_ENDPOINT and hasattr(mb.built_model, 'model_arn') else ""
                logger.info("✅ Model build successful: '%s' using server %s in %s mode%s", mb.model_name, mb.model_server, mb.mode, model_arn_info)
            deployables["InferenceComponents"] = built_ic_models


        if isinstance(self.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)):
            logger.debug("Building custom orchestrator.")
            if self.mode == Mode.IN_PROCESS or self.mode == Mode.LOCAL_CONTAINER:
                raise ValueError(
                    "Custom orchestrator deployment is only supported for"
                    "SageMaker Endpoint Mode."
                )
            self.serve_settings = self._get_serve_setting()
            cpu_or_gpu_instance = self._get_processing_unit()
            self.image_uri = self._get_smd_image_uri(processing_unit=cpu_or_gpu_instance)
            self.model_server = ModelServer.SMD
            built_orchestrator = self._build_single_modelbuilder(
                mode=Mode.SAGEMAKER_ENDPOINT,
                role_arn=role_arn,
                sagemaker_session=sagemaker_session,
            )
            if not self.resource_requirements:
                logger.info(
                    "Custom orchestrator resource_requirements not found. "
                    "Building as a SageMaker Endpoint instead of Inference Component."
                )
                deployables["CustomOrchestrator"] = {
                    "Mode": "Endpoint",
                    "Model": built_orchestrator,
                }
            else:

                if built_ic_models:
                    if (
                        self.dependencies["auto"]
                        or "requirements" in self.dependencies
                        or "custom" in self.dependencies
                    ):
                        logger.warning(
                            "Custom orchestrator network isolation must be False when dependencies "
                            "are specified or using autocapture. To enable network isolation, "
                            "package all dependencies in the container or model artifacts "
                            "ahead of time."
                        )
                        built_orchestrator._enable_network_isolation = False
                        for model in built_ic_models:
                            model["Model"]._enable_network_isolation = False
                deployables["CustomOrchestrator"] = {
                    "Name": self.inference_component_name,
                    "Mode": "InferenceComponent",
                    "ResourceRequirements": self.resource_requirements,
                    "Model": built_orchestrator,
                }

            logger.info("✅ Custom orchestrator build successful: '%s' using server %s in %s mode", self.model_name, self.model_server, self.mode)

        self._deployables = deployables
        return self

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.configure_for_torchserve")
    def configure_for_torchserve(
        self,
        shared_libs: Optional[List[str]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, StrPipeVar]] = None,
    ) -> "ModelBuilder":
        """Configure ModelBuilder for TorchServe deployment."""
        if shared_libs is not None:
            self.shared_libs = shared_libs
        if dependencies is not None:
            self.dependencies = dependencies
        if image_config is not None:
            self.image_config = image_config

        self.model_server = ModelServer.TORCHSERVE
        return self


    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.from_jumpstart_config")
    def from_jumpstart_config(
        cls,
        jumpstart_config: JumpStartConfig,
        role_arn: Optional[str] = None,
        compute: Optional[Compute] = None,
        network: Optional[Networking] = None,
        image_uri: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        model_kms_key: Optional[str] = None,
        resource_requirements: Optional[ResourceRequirements] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        sagemaker_session: Optional[Session] = None,
        schema_builder: Optional[SchemaBuilder] = None,
    ) -> "ModelBuilder":
        """Create a ``ModelBuilder`` instance from a JumpStart configuration.

        This class method provides a convenient way to create a ModelBuilder for deploying
        pre-trained models from Amazon SageMaker JumpStart. It automatically retrieves the
        appropriate model artifacts, container images, and default configurations for the
        specified JumpStart model.

        Args:
            jumpstart_config (JumpStartConfig): Configuration object specifying the JumpStart
                model to use. Must include model_id and optionally model_version and
                inference_config_name.
            role_arn (str, optional): The IAM role ARN for SageMaker to assume when creating
                the model and endpoint. If not specified, attempts to use the default SageMaker
                execution role. (Default: None).
            compute (Compute, optional): Compute configuration specifying instance type and
                instance count for deployment. For example, Compute(instance_type='ml.g5.xlarge',
                instance_count=1). (Default: None).
            network (Networking, optional): Network configuration including VPC settings and
                network isolation. For example, Networking(vpc_config={'Subnets': [...],
                'SecurityGroupIds': [...]}, enable_network_isolation=False). (Default: None).
            image_uri (str, optional): Custom container image URI. If not specified, uses
                the default JumpStart container image for the model. (Default: None).
            env_vars (Dict[str, str], optional): Environment variables to set in the container.
                These will be merged with default JumpStart environment variables. (Default: None).
            model_kms_key (str, optional): KMS key ARN used to encrypt model artifacts when
                uploading to S3. (Default: None).
            resource_requirements (ResourceRequirements, optional): The compute resource
                requirements for deploying the model to an inference component based endpoint.
                (Default: None).
            tolerate_vulnerable_model (bool, optional): If True, allows deployment of models
                with known security vulnerabilities. Use with caution. (Default: None).
            tolerate_deprecated_model (bool, optional): If True, allows deployment of deprecated
                JumpStart models. (Default: None).
            sagemaker_session (Session, optional): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified,
                creates one using the default AWS configuration chain. (Default: None).
            schema_builder (SchemaBuilder, optional): Schema builder for defining input/output
                schemas. If not specified, uses default schemas for the JumpStart model.
                (Default: None).

        Returns:
            ModelBuilder: A configured ``ModelBuilder`` instance ready to build and deploy
                the specified JumpStart model.

        Example:
            >>> from sagemaker.core.jumpstart.configs import JumpStartConfig
            >>> from sagemaker.serve.model_builder import ModelBuilder
            >>>
            >>> js_config = JumpStartConfig(
            ...     model_id="huggingface-llm-mistral-7b",
            ...     model_version="*"
            ... )
            >>>
            >>> from sagemaker.core.training.configs import Compute
            >>>
            >>> model_builder = ModelBuilder.from_jumpstart_config(
            ...     jumpstart_config=js_config,
            ...     compute=Compute(instance_type="ml.g5.2xlarge", instance_count=1)
            ... )
            >>>
            >>> model = model_builder.build()  # Creates Model resource
            >>> endpoint = model_builder.deploy()  # Creates Endpoint resource
            >>> result = endpoint.invoke(data=input_data)  # Make predictions
        """
        deploy_kwargs = {}
        if compute and compute.instance_type:  # Only retrieve if instance_type is provided
            try:
                deploy_kwargs = _retrieve_model_deploy_kwargs(
                    model_id=jumpstart_config.model_id,
                    model_version=jumpstart_config.model_version or "*",
                    instance_type=compute.instance_type,
                    region=sagemaker_session.boto_region_name if sagemaker_session else None,
                    tolerate_vulnerable_model=tolerate_vulnerable_model or False,
                    tolerate_deprecated_model=tolerate_deprecated_model or False,
                    sagemaker_session=sagemaker_session or Session(),
                    config_name=jumpstart_config.inference_config_name,
                )
            except Exception:
                pass

        # Initialize JumpStart-Related Variables

        mb_instance = cls(
            model=jumpstart_config.model_id,
            role_arn=role_arn,
            compute=compute,
            network=network,
            image_uri=image_uri,
            env_vars=env_vars or {},
            sagemaker_session=sagemaker_session,
            schema_builder=schema_builder,
        )

        mb_instance.model_version = jumpstart_config.model_version or "*"
        mb_instance.resource_requirements = resource_requirements
        mb_instance.model_kms_key = model_kms_key
        mb_instance.hub_name = jumpstart_config.hub_name
        mb_instance.config_name=jumpstart_config.inference_config_name
        mb_instance.accept_eula = jumpstart_config.accept_eula
        mb_instance.tolerate_vulnerable_model=tolerate_vulnerable_model
        mb_instance.tolerate_deprecated_model=tolerate_deprecated_model
        mb_instance.model_data_download_timeout=deploy_kwargs.get("model_data_download_timeout")
        mb_instance.container_startup_health_check_timeout=deploy_kwargs.get("container_startup_health_check_timeout")
        mb_instance.inference_ami_version=deploy_kwargs.get("inference_ami_version")

        return mb_instance


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.transformer")
    def transformer(
        self,
        instance_count,
        instance_type,
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        env=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        volume_kms_key=None,
    ):
        """Return a ``Transformer`` that uses this Model.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): Environment variables to be set for use during the
                transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests
                to be made to each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP
                request to the container in MB.
            tags (Optional[Tags]): Tags for labeling a transform job. If
                none specified, then the tags used for the training job are used
                for the transform job.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
        """
        self._init_sagemaker_session_if_does_not_exist(self.instance_type)
        tags = format_tags(tags)

        # Ensure model has been built
        if not hasattr(self, 'built_model') or self.built_model is None:
            raise ValueError("Must call build() before creating transformer")

        # Network isolation disables custom environment variables
        if self._enable_network_isolation:
            env = None

        return Transformer(
            self.built_model.model_name,
            instance_count,
            instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            tags=tags,
            base_transform_job_name=self.model_name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.display_benchmark_metrics")
    def display_benchmark_metrics(self, **kwargs) -> None:
        """Display benchmark metrics for JumpStart models."""
        if not isinstance(self.model, str):
            raise ValueError("Benchmarking is only supported for JumpStart or HuggingFace models")
        if self._is_jumpstart_model_id() or self._use_jumpstart_equivalent():
            df = self.benchmark_metrics

            instance_type = kwargs.get("instance_type")
            if instance_type:
                df = df[df["Instance Type"].str.contains(instance_type)]

            print(df.to_markdown(index=False, floatfmt=".2f"))
        else:
            raise ValueError("This model does not have benchmark metrics available")


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.set_deployment_config")
    def set_deployment_config(self, config_name: str, instance_type: str) -> None:
        """Sets the deployment config to apply to the model."""
        if not isinstance(self.model, str):
            raise ValueError("Deployment config is only supported for JumpStart or HuggingFace models")

        if not (self._is_jumpstart_model_id() or self._use_jumpstart_equivalent()):
            raise ValueError(f"The deployment config {config_name} cannot be set on this model")


        self.config_name = config_name
        self.instance_type = instance_type


        self._deployment_config = None


        self._deployment_config = self.get_deployment_config()


        if self._deployment_config:
            deployment_args = self._deployment_config.get("DeploymentArgs", {})
            if deployment_args.get("AdditionalDataSources"):
                self.additional_model_data_sources = deployment_args["AdditionalDataSources"]


        if self.additional_model_data_sources:
            self.speculative_decoding_draft_model_source = "sagemaker"
            self.add_tags({"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "sagemaker"})
            self.remove_tag_with_key(Tag.OPTIMIZATION_JOB_NAME)
            self.remove_tag_with_key(Tag.FINE_TUNING_MODEL_PATH)
            self.remove_tag_with_key(Tag.FINE_TUNING_JOB_NAME)


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.get_deployment_config")
    def get_deployment_config(self) -> Optional[Dict[str, Any]]:
        """Gets the deployment config to apply to the model."""
        if not isinstance(self.model, str):
            raise ValueError("Deployment config is only supported for JumpStart or HuggingFace models")

        if not (self._is_jumpstart_model_id() or self._use_jumpstart_equivalent()):
            raise ValueError("This model does not have any deployment config yet")


        if self.config_name is None:
            return None


        if self._deployment_config is None:

            for config in self.list_deployment_configs():
                if config.get("DeploymentConfigName") == self.config_name:
                    self._deployment_config = config
                    break

        return self._deployment_config


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.list_deployment_configs")
    def list_deployment_configs(self) -> List[Dict[str, Any]]:
        """List deployment configs for the model in the current region."""
        if not isinstance(self.model, str):
            raise ValueError("Deployment config is only supported for JumpStart or HuggingFace models")

        if not (self._is_jumpstart_model_id() or self._use_jumpstart_equivalent()):
            raise ValueError("Deployment config is only supported for JumpStart models")


        return self.deployment_config_response_data(
            self._get_deployment_configs(self.config_name, self.instance_type)
        )  # Delegate to JumpStart builder



    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.optimize")
    # Add these methods to the current V3 ModelBuilder class:
    def optimize(
        self,
        model_name: Optional[str] = "optimize_model",
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        role_arn: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        region: Optional[str] = None,
        tags: Optional[Tags] = None,
        job_name: Optional[str] = None,
        accept_eula: Optional[bool] = None,
        quantization_config: Optional[Dict] = None,
        compilation_config: Optional[Dict] = None,
        speculative_decoding_config: Optional[Dict] = None,
        sharding_config: Optional[Dict] = None,
        env_vars: Optional[Dict] = None,
        vpc_config: Optional[Dict] = None,
        kms_key: Optional[str] = None,
        image_uri: Optional[str] = None,
        max_runtime_in_sec: Optional[int] = 36000,
    ) -> Model:
        """Create an optimized deployable ``Model`` instance with ``ModelBuilder``.

        Runs a SageMaker model optimization job to quantize, compile, or shard the model
        for improved inference performance. Returns a ``Model`` resource that can be deployed
        using the deploy() method.

        Note: This returns a ``sagemaker.core.resources.Model`` object.

        Args:
            output_path (str, optional): S3 URI where the optimized model artifacts will be stored.
                If not specified, uses the default output path. (Default: None).
            instance_type (str, optional): Target deployment instance type that the model is
                optimized for. For example, 'ml.p4d.24xlarge'. (Default: None).
            role_arn (str, optional): IAM execution role ARN for the optimization job.
                If not specified, uses the role from initialization. (Default: None).
            sagemaker_session (Session, optional): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified,
                uses the session from initialization or creates one using the default AWS
                configuration chain. (Default: None).
            region (str, optional): The AWS region for the optimization job. If specified and
                different from the current region, a new session will be created. (Default: None).
            model_name (str, optional): The name for the optimized SageMaker model. If not
                specified, a unique name will be generated. (Default: None).
            tags (Tags, optional): Tags for labeling the model optimization job. (Default: None).
            job_name (str, optional): The name of the model optimization job. If not specified,
                a unique name will be generated. (Default: None).
            accept_eula (bool, optional): For models that require a Model Access Config,
                specify True or False to indicate whether model terms of use have been accepted.
                The accept_eula value must be explicitly defined as True in order to accept
                the end-user license agreement (EULA) that some models require. (Default: None).
            quantization_config (Dict, optional): Quantization configuration specifying the
                quantization method and parameters. For example:
                {'OverrideEnvironment': {'OPTION_QUANTIZE': 'awq'}}. (Default: None).
            compilation_config (Dict, optional): Compilation configuration for optimizing
                the model for specific hardware. (Default: None).
            speculative_decoding_config (Dict, optional): Speculative decoding configuration
                for improving inference latency of large language models. (Default: None).
            sharding_config (Dict, optional): Model sharding configuration for distributing
                large models across multiple devices. (Default: None).
            env_vars (Dict, optional): Additional environment variables to pass to the
                optimization container. (Default: None).
            vpc_config (Dict, optional): VPC configuration for the optimization job.
                Should contain 'Subnets' and 'SecurityGroupIds' keys. (Default: None).
            kms_key (str, optional): KMS key ARN used to encrypt the optimized model artifacts
                when uploading to S3. (Default: None).
            image_uri (str, optional): Custom container image URI for the optimization job.
                If not specified, uses the default optimization container. (Default: None).
            max_runtime_in_sec (int): Maximum job execution time in seconds. The optimization
                job will be stopped if it exceeds this time. (Default: 36000).

        Returns:
            Model: A ``sagemaker.core.resources.Model`` resource containing the optimized
                model artifacts, ready for deployment.

        Example:
            >>> model_builder = ModelBuilder(model=my_model, role_arn=role)
            >>> optimized_model = model_builder.optimize(
            ...     instance_type="ml.g5.xlarge",
            ...     quantization_config={'OverrideEnvironment': {'OPTION_QUANTIZE': 'awq'}}
            ... )
            >>> endpoint = model_builder.deploy()  # Deploy the optimized model
            >>> result = endpoint.invoke(data=input_data)
        """

        # Update parameters if provided
        if region and region != self.region:
            logger.warning("Changing region from '%s' to '%s' during optimize()", self.region, region)
            self.region = region
            self.sagemaker_session = self._create_session_with_region()

        if role_arn and role_arn != self.role_arn:
            logger.debug("Updating role_arn during optimize()")
            self.role_arn = role_arn

        self.region = region or self.region
        if sagemaker_session:
            self.sagemaker_session = sagemaker_session

        self.model_name = model_name or getattr(self, 'model_name', None)
        self.framework = getattr(self, 'framework', None)
        self.framework_version = getattr(self, 'framework_version', None)
        self.accept_eula = accept_eula or getattr(self, 'accept_eula', None)
        self.instance_type = instance_type or getattr(self, 'instance_type', None)
        self.container_log_level = getattr(self, 'container_log_level', None)
        self.serve_settings = self._get_serve_setting()

        self._optimizing = True

        return self._model_builder_optimize_wrapper(
            output_path=output_path,
            instance_type=instance_type,
            role_arn=role_arn,
            tags=tags,
            job_name=job_name,
            accept_eula=accept_eula,
            quantization_config=quantization_config,
            compilation_config=compilation_config,
            speculative_decoding_config=speculative_decoding_config,
            sharding_config=sharding_config,
            env_vars=env_vars,
            vpc_config=vpc_config,
            kms_key=kms_key,
            image_uri=image_uri,
            max_runtime_in_sec=max_runtime_in_sec,
            sagemaker_session=sagemaker_session,
        )


    def _model_builder_optimize_wrapper(
        self,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        role_arn: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[Tags] = None,
        job_name: Optional[str] = None,
        accept_eula: Optional[bool] = None,
        quantization_config: Optional[Dict] = None,
        compilation_config: Optional[Dict] = None,
        speculative_decoding_config: Optional[Dict] = None,
        sharding_config: Optional[Dict] = None,
        env_vars: Optional[Dict] = None,
        vpc_config: Optional[Dict] = None,
        kms_key: Optional[str] = None,
        image_uri: Optional[str] = None,
        max_runtime_in_sec: Optional[int] = 36000,
        sagemaker_session: Optional[Session] = None,
    ) -> Model:
        """Runs a model optimization job."""
        if (
            hasattr(self, "_enable_network_isolation")
            and self._enable_network_isolation
            and sharding_config
        ):
            raise ValueError(
                "EnableNetworkIsolation cannot be set to True since SageMaker Fast Model "
                "Loading of model requires network access."
            )

        _validate_optimization_configuration(
            is_jumpstart=self._is_jumpstart_model_id(),
            instance_type=instance_type,
            quantization_config=quantization_config,
            compilation_config=compilation_config,
            sharding_config=sharding_config,
            speculative_decoding_config=speculative_decoding_config,
        )

        self.is_compiled = compilation_config is not None
        self.is_quantized = quantization_config is not None
        self.speculative_decoding_draft_model_source = self._extract_speculative_draft_model_provider(
            speculative_decoding_config
        )

        if self.mode != Mode.SAGEMAKER_ENDPOINT:
            raise ValueError("Model optimization is only supported in Sagemaker Endpoint Mode.")

        if sharding_config and (
            quantization_config or compilation_config or speculative_decoding_config
        ):
            raise ValueError(
                (
                    "Sharding config is mutually exclusive "
                    "and cannot be combined with any other optimization."
                )
            )

        if sharding_config:
            has_tensor_parallel_degree_in_env_vars = (
                env_vars and "OPTION_TENSOR_PARALLEL_DEGREE" in env_vars
            )
            has_tensor_parallel_degree_in_overrides = (
                sharding_config
                and sharding_config.get("OverrideEnvironment")
                and "OPTION_TENSOR_PARALLEL_DEGREE" in sharding_config.get("OverrideEnvironment")
            )
            if (
                not has_tensor_parallel_degree_in_env_vars
                and not has_tensor_parallel_degree_in_overrides
            ):
                raise ValueError(
                    (
                        "OPTION_TENSOR_PARALLEL_DEGREE is a required "
                        "environment variable with sharding config."
                    )
                )

        # Validate and set region
        if region and region != self.region:
            logger.warning("Changing region from '%s' to '%s' during optimize()", self.region, region)
            self.region = region
            # Recreate session with new region
            self.sagemaker_session = self._create_session_with_region()

        # Validate and set role_arn
        if role_arn and role_arn != self.role_arn:
            logger.debug("Updating role_arn during optimize()")
            self.role_arn = role_arn

        self.sagemaker_session = sagemaker_session or self.sagemaker_session or self._create_session_with_region()
        self.instance_type = instance_type or self.instance_type

        job_name = job_name or f"modelbuilderjob-{uuid.uuid4().hex}"

        if self._is_jumpstart_model_id():
            # Build using V3 method instead of self.build()
            self.built_model = self._build_single_modelbuilder(
                mode=self.mode,
                sagemaker_session=self.sagemaker_session
            )
            # Set deployment config on built_model if needed
            input_args = self._optimize_for_jumpstart(
                output_path=output_path,
                instance_type=instance_type,
                tags=tags,
                job_name=job_name,
                accept_eula=accept_eula,
                quantization_config=quantization_config,
                compilation_config=compilation_config,
                speculative_decoding_config=speculative_decoding_config,
                sharding_config=sharding_config,
                env_vars=env_vars,
                vpc_config=vpc_config,
                kms_key=kms_key,
                max_runtime_in_sec=max_runtime_in_sec,
            )

        else:
            if self.model_server != ModelServer.DJL_SERVING:
                logger.info("Overriding model server to DJL_SERVING.")
                self.model_server = ModelServer.DJL_SERVING

            # Build using V3 method instead of self.build()
            self.built_model = self._build_single_modelbuilder(
                mode=self.mode,
                sagemaker_session=self.sagemaker_session
            )
            input_args = self._optimize_for_hf(
                output_path=output_path,
                tags=tags,
                job_name=job_name,
                quantization_config=quantization_config,
                compilation_config=compilation_config,
                speculative_decoding_config=speculative_decoding_config,
                sharding_config=sharding_config,
                env_vars=env_vars,
                vpc_config=vpc_config,
                kms_key=kms_key,
                max_runtime_in_sec=max_runtime_in_sec,
            )

        if sharding_config:
            self._is_sharded_model = True

        if input_args:
            optimization_instance_type = input_args["DeploymentInstanceType"]


            gpu_instance_families = ["g5", "g6", "p4d", "p4de", "p5"]
            is_gpu_instance = optimization_instance_type and any(
                gpu_instance_family in optimization_instance_type
                for gpu_instance_family in gpu_instance_families
            )


            is_llama_3_plus = self.model and bool(
                re.search(r"llama-3[\.\-][1-9]\d*", self.model.lower())
            )

            if is_gpu_instance and self.model and self.is_compiled:
                if is_llama_3_plus:
                    raise ValueError(
                        "Compilation is not supported for models greater "
                        "than Llama-3.0 with a GPU instance."
                    )
                if speculative_decoding_config:
                    raise ValueError(
                        "Compilation is not supported with speculative decoding with "
                        "a GPU instance."
                    )

            if image_uri:
                input_args["OptimizationConfigs"][0]["ModelQuantizationConfig"]["Image"] = image_uri

            self.sagemaker_session.sagemaker_client.create_optimization_job(**input_args)
            job_status = self.sagemaker_session.wait_for_optimization_job(job_name)

            # KEY CHANGE: Generate optimized CoreModel instead of PySDK Model
            return self._generate_optimized_core_model(job_status)

        self._optimizing = False
        self.built_model = self._create_model()
        return self.built_model


    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.deploy")
    def deploy(
        self,
        endpoint_name: str = None,
        initial_instance_count: Optional[int] = 1,
        instance_type: Optional[str] = None,
        wait: bool = True,
        update_endpoint: Optional[bool] = False,
        container_timeout_in_seconds: int = 300,
        inference_config: Optional[
            Union[
                ServerlessInferenceConfig,
                AsyncInferenceConfig,
                BatchTransformInferenceConfig,
                ResourceRequirements,
            ]
        ] = None,
        custom_orchestrator_instance_type: str = None,
        custom_orchestrator_initial_instance_count: int = None,
        **kwargs,
    ) -> Union[Endpoint, LocalEndpoint, Transformer]:
        """Deploy the built model to an ``Endpoint``.

        Creates a SageMaker ``EndpointConfig`` and deploys an ``Endpoint`` resource from the
        model created by build(). The model must be built before calling deploy().

        Note: This returns a ``sagemaker.core.resources.Endpoint`` object, not the deprecated
        PySDK Predictor class. Use endpoint.invoke() to make predictions.

        Args:
            endpoint_name (str): The name of the endpoint to create. If not specified,
                a unique endpoint name will be created. (Default: "endpoint").
            initial_instance_count (int, optional): The initial number of instances to run
                in the endpoint. Required for instance-based endpoints. (Default: 1).
            instance_type (str, optional): The EC2 instance type to deploy this model to.
                For example, 'ml.p2.xlarge'. Required for instance-based endpoints unless
                using serverless inference. (Default: None).
            wait (bool): Whether the call should wait until the deployment completes.
                (Default: True).
            update_endpoint (bool): Flag to update the model in an existing Amazon SageMaker
                endpoint. If True, deploys a new EndpointConfig to an existing endpoint and
                deletes resources from the previous EndpointConfig. (Default: False).
            container_timeout_in_seconds (int): The timeout value, in seconds, for the container
                to respond to requests. (Default: 300).
            inference_config (Union[ServerlessInferenceConfig, AsyncInferenceConfig,
                BatchTransformInferenceConfig, ResourceRequirements], optional): Unified inference
                configuration parameter. Can be used instead of individual config parameters.
                (Default: None).
            custom_orchestrator_instance_type (str, optional): Instance type for custom
                orchestrator deployment. (Default: None).
            custom_orchestrator_initial_instance_count (int, optional): Initial instance count
                for custom orchestrator deployment. (Default: None).
        Returns:
            Union[Endpoint, LocalEndpoint, Transformer]: A ``sagemaker.core.resources.Endpoint``
                resource representing the deployed endpoint, a ``LocalEndpoint`` for local mode,
                or a ``Transformer`` for batch transform inference.

        Example:
            >>> model_builder = ModelBuilder(model=my_model, role_arn=role, instance_type="ml.m5.xlarge")
            >>> model = model_builder.build()  # Creates Model resource
            >>> endpoint = model_builder.deploy(endpoint_name="my-endpoint")  # Creates Endpoint resource
            >>> result = endpoint.invoke(data=input_data)  # Make predictions
        """
        if hasattr(self, '_deployed') and self._deployed:
            logger.warning(
                "ModelBuilder.deploy() has already been called. "
                "Reusing ModelBuilder objects for multiple deployments is not recommended. "
                "Please create a new ModelBuilder instance for additional deployments."
            )
        self._deployed = True

        if not hasattr(self, "built_model") and not hasattr(self, "_deployables"):
            raise ValueError("Model needs to be built before deploying")

        # Handle model customization deployment
        if self._is_model_customization():
            logger.info("Deploying Model Customization model")
            if not self.instance_type and not instance_type:
                self.instance_type = self._fetch_default_instance_type_for_custom_model()
            return self._deploy_model_customization(
                endpoint_name=endpoint_name,
                instance_type=instance_type or self.instance_type,
                initial_instance_count=initial_instance_count,
                wait=wait,
                container_timeout_in_seconds=container_timeout_in_seconds,
                **kwargs
            )

        if not update_endpoint:
            if not endpoint_name or endpoint_name == "endpoint":
                endpoint_name = (endpoint_name or "endpoint") + "-" + str(uuid.uuid4())[:8]
            self.endpoint_name = endpoint_name

        if not hasattr(self, "_deployables"):

            if not inference_config:
                deploy_kwargs = {
                    "instance_type": self.instance_type,
                    "initial_instance_count": initial_instance_count,
                    "endpoint_name": endpoint_name,
                    "update_endpoint": update_endpoint,
                    "container_timeout_in_seconds": container_timeout_in_seconds,
                    "wait": wait,
                    "endpoint_type": EndpointType.MODEL_BASED,
                }

                deploy_kwargs.update(kwargs)
                return self._deploy(**deploy_kwargs)

            if isinstance(inference_config, ServerlessInferenceConfig):
                deploy_kwargs = {
                    "serverless_inference_config": inference_config,
                    "endpoint_name": endpoint_name,
                    "update_endpoint": update_endpoint,
                    "container_timeout_in_seconds": container_timeout_in_seconds,
                    "wait": wait,
                    "endpoint_type": EndpointType.MODEL_BASED,
                }

                deploy_kwargs.update(kwargs)
                return self._deploy(**deploy_kwargs)

            if isinstance(inference_config, AsyncInferenceConfig):
                deploy_kwargs = {
                    "instance_type": self.instance_type,
                    "initial_instance_count": initial_instance_count,
                    "async_inference_config": inference_config,
                    "endpoint_name": endpoint_name,
                    "update_endpoint": update_endpoint,
                    "container_timeout_in_seconds": container_timeout_in_seconds,
                    "wait": wait,
                    "endpoint_type": EndpointType.MODEL_BASED,
                }

                deploy_kwargs.update(kwargs)
                return self._deploy(**deploy_kwargs)

            if isinstance(inference_config, BatchTransformInferenceConfig):
                return self.transformer(
                    instance_type=inference_config.instance_type,
                    output_path=inference_config.output_path,
                    instance_count=inference_config.instance_count,
                )

        if isinstance(inference_config, ResourceRequirements):
            if update_endpoint:
                raise ValueError(
                    "update_endpoint is not supported for inference component deployments"
                )
            deploy_kwargs = {
                "instance_type": self.instance_type,
                "mode": Mode.SAGEMAKER_ENDPOINT,
                "endpoint_type": EndpointType.INFERENCE_COMPONENT_BASED,
                "resources": inference_config,
                "initial_instance_count": initial_instance_count,
                "role": self.role_arn,
                "update_endpoint": update_endpoint,
                "container_timeout_in_seconds": container_timeout_in_seconds,
                "wait": wait,
            }

            deploy_kwargs.update(kwargs)
            return self._deploy(**deploy_kwargs)


        if hasattr(self, '_deployables') and self._deployables:
            endpoints = []
            for ic in self._deployables.get("InferenceComponents", []):
                endpoints.append(self._deploy_for_ic(ic_data=ic, endpoint_name=endpoint_name))

            # Handle custom orchestrator if present
            if self._deployables.get("CustomOrchestrator"):
                custom_orchestrator = self._deployables.get("CustomOrchestrator")
                if not custom_orchestrator_instance_type and not instance_type:
                    logger.warning(
                        "Deploying custom orchestrator as an endpoint but no instance type was "
                        "set. Defaulting to `ml.c5.xlarge`."
                    )
                    custom_orchestrator_instance_type = "ml.c5.xlarge"
                    custom_orchestrator_initial_instance_count = 1
                if custom_orchestrator["Mode"] == "Endpoint":
                    logger.info(
                        "Deploying custom orchestrator on instance type %s.",
                        custom_orchestrator_instance_type,
                    )
                    deploy_kwargs = {
                        "built_model": custom_orchestrator["Model"],
                        "instance_type": custom_orchestrator_instance_type,
                        "initial_instance_count": custom_orchestrator_initial_instance_count,
                        "endpoint_name": endpoint_name,
                        "container_timeout_in_seconds": container_timeout_in_seconds,
                        "wait": wait,
                        "endpoint_type": EndpointType.INFERENCE_COMPONENT_BASED,
                    }

                    deploy_kwargs.update(kwargs)
                    endpoints.append(self._deploy(**deploy_kwargs))
                elif custom_orchestrator["Mode"] == "InferenceComponent":
                    logger.info(
                        "Deploying custom orchestrator as an inference component "
                        f"to endpoint {endpoint_name}"
                    )
                    endpoints.append(
                        self._deploy_for_ic(
                            ic_data=custom_orchestrator,
                            container_timeout_in_seconds=container_timeout_in_seconds,
                            instance_type=custom_orchestrator_instance_type or instance_type,
                            initial_instance_count=custom_orchestrator_initial_instance_count
                            or initial_instance_count,
                            endpoint_name=endpoint_name,
                            **kwargs,
                        )
                    )

            return endpoints[0] if len(endpoints) == 1 else endpoints

        raise ValueError("Deployment Options not supported")


    def _deploy_model_customization(
        self,
        endpoint_name: str,
        initial_instance_count: int = 1,
        inference_component_name: Optional[str] = None,
        **kwargs
    ) -> Endpoint:
        """Deploy a model customization (fine-tuned) model to an endpoint with inference components.

        This method handles the special deployment flow for fine-tuned models, creating:
        1. Core Model resource
        2. EndpointConfig
        3. Endpoint
        4. InferenceComponent

        Args:
            endpoint_name (str): Name of the endpoint to create or update
            instance_type (str): EC2 instance type for deployment
            initial_instance_count (int): Number of instances (default: 1)
            wait (bool): Whether to wait for deployment to complete (default: True)
            container_timeout_in_seconds (int): Container timeout in seconds (default: 300)
            inference_component_name (Optional[str]): Name for the inference component
            **kwargs: Additional deployment parameters

        Returns:
            Endpoint: The deployed sagemaker.core.resources.Endpoint
        """
        from sagemaker.core.resources import Model as CoreModel, EndpointConfig as CoreEndpointConfig
        from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
        from sagemaker.core.shapes import (
            InferenceComponentSpecification,
            InferenceComponentContainerSpecification,
            InferenceComponentRuntimeConfig,
            InferenceComponentComputeResourceRequirements,
            ModelDataSource,
            S3ModelDataSource
        )
        from sagemaker.core.resources import InferenceComponent
        from sagemaker.core.utils.utils import Unassigned

        # Fetch model package
        model_package = self._fetch_model_package()

        # Check if endpoint exists
        is_existing_endpoint = self._does_endpoint_exist(endpoint_name)

        # Generate model name if not set
        model_name = self.model_name or f"model-{uuid.uuid4().hex[:10]}"

        if not is_existing_endpoint:
            EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[ProductionVariant(
                    variant_name=endpoint_name,
                    instance_type=self.instance_type,
                    initial_instance_count=initial_instance_count or 1
                )],
                execution_role_arn=self.role_arn
            )
            logger.info("Endpoint core call starting")
            endpoint = Endpoint.create(endpoint_name=endpoint_name, endpoint_config_name=endpoint_name)
            endpoint.wait_for_status("InService")
        else:
            endpoint = Endpoint.get(endpoint_name=endpoint_name)

        # Set inference component name
        if not inference_component_name:
            if not is_existing_endpoint:
                inference_component_name = f"{endpoint_name}-inference-component"
            else:
                inference_component_name = f"{endpoint_name}-inference-component-adapter"

        # Get PEFT type and base model recipe name
        peft_type = self._fetch_peft()
        base_model_recipe_name = model_package.inference_specification.containers[0].base_model.recipe_name
        base_inference_component_name = None
        tag = None

        # Handle tagging and base component lookup
        if not is_existing_endpoint:
            from sagemaker.core.resources import Tag as CoreTag
            tag = CoreTag(key="Base", value=base_model_recipe_name)
        elif peft_type == "LORA":
            from sagemaker.core.resources import Tag as CoreTag
            for component in InferenceComponent.get_all(endpoint_name_equals=endpoint_name, status_equals="InService"):
                component_tags = CoreTag.get_all(resource_arn=component.inference_component_arn)
                if any(t.key == "Base" and t.value == base_model_recipe_name for t in component_tags):
                    base_inference_component_name = component.inference_component_name
                    break

        artifact_url = None #if peft_type == "LORA" else self._fetch_model_package().inference_specification.containers[0].model_data_source.s3_data_source.s3_uri

        ic_spec = InferenceComponentSpecification(
            container=InferenceComponentContainerSpecification(
                image=self.image_uri,
                artifact_url=artifact_url,
                environment=self.env_vars
            )
        )

        if peft_type == "LORA":
            ic_spec.base_inference_component_name = base_inference_component_name
        ic_spec.compute_resource_requirements = self._cached_compute_requirements

        InferenceComponent.create(
            inference_component_name=inference_component_name,
            endpoint_name=endpoint_name,
            variant_name=endpoint_name,
            specification=ic_spec,
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
            tags=[{"key": tag.key, "value": tag.value}] if tag else []
        )

        # Create lineage tracking for new endpoints
        if not is_existing_endpoint:
            from sagemaker.core.resources import Action, Association, Artifact
            from sagemaker.core.shapes import ActionSource, MetadataProperties

            inference_component = InferenceComponent.get(inference_component_name=inference_component_name)

            action = Action.create(
                source=ActionSource(source_uri=self._fetch_model_package_arn(),
                                    source_type="SageMaker"),
                action_name=f"{endpoint_name}-action",
                action_type="ModelDeployment",
                properties={"EndpointConfigName": endpoint_name},
                metadata_properties=MetadataProperties(generated_by=inference_component.inference_component_arn)
            )

            artifacts = Artifact.get_all(source_uri=model_package.model_package_arn)
            for artifact in artifacts:
                Association.add(source_arn=artifact.artifact_arn, destination_arn=action.action_arn)
                break

        logger.info("✅ Model customization deployment successful: Endpoint '%s'", endpoint_name)
        return endpoint

    def _fetch_peft(self) -> Optional[str]:
        """Fetch the PEFT (Parameter-Efficient Fine-Tuning) type from the training job."""
        if isinstance(self.model, TrainingJob):
            training_job = self.model
        elif isinstance(self.model, ModelTrainer):
            training_job = self.model._latest_training_job
        else:
            return None

        from sagemaker.core.utils.utils import Unassigned
        if training_job.serverless_job_config != Unassigned() and training_job.serverless_job_config.job_spec != Unassigned():
            return training_job.serverless_job_config.job_spec.get("PEFT")
        return None

    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        """Check if an endpoint exists with the given name."""
        try:
            Endpoint.get(endpoint_name=endpoint_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                return False
            raise

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.deploy_local")
    def deploy_local(
        self,
        endpoint_name: str = "endpoint", 
        container_timeout_in_seconds: int = 300,
        **kwargs
    ) -> LocalEndpoint:
        """Deploy the built model to local mode for testing.
        
        Deploys the model locally using either LOCAL_CONTAINER mode (runs in a Docker container)
        or IN_PROCESS mode (runs in the current Python process). This is useful for testing and
        development before deploying to SageMaker endpoints. The model must be built with
        mode=Mode.LOCAL_CONTAINER or mode=Mode.IN_PROCESS before calling this method.
        
        Note: This returns a ``LocalEndpoint`` object for local inference, not a SageMaker
        Endpoint resource. Use local_endpoint.invoke() to make predictions.

        Args:
            endpoint_name (str): The name for the local endpoint. (Default: "endpoint").
            container_timeout_in_seconds (int): The timeout value, in seconds, for the container
                to respond to requests. (Default: 300).
        Returns:
            LocalEndpoint: A ``LocalEndpoint`` object for making local predictions.
            
        Raises:
            ValueError: If the model was not built with LOCAL_CONTAINER or IN_PROCESS mode.
            
        Example:
            >>> model_builder = ModelBuilder(
            ...     model=my_model,
            ...     role_arn=role,
            ...     mode=Mode.LOCAL_CONTAINER
            ... )
            >>> model = model_builder.build()
            >>> local_endpoint = model_builder.deploy_local()
            >>> result = local_endpoint.invoke(data=input_data)
        """
        if self.mode not in [Mode.LOCAL_CONTAINER, Mode.IN_PROCESS]:
            raise ValueError(f"deploy_local() only supports LOCAL_CONTAINER and IN_PROCESS modes, got {self.mode}")
        
        return self.deploy(
            endpoint_name=endpoint_name,
            container_timeout_in_seconds=container_timeout_in_seconds,
            **kwargs
        )
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="model_builder.register")
    @runnable_by_pipeline
    def register(
        self,
        model_package_name: Optional[StrPipeVar] = None,
        model_package_group_name: Optional[StrPipeVar] = None,
        content_types: List[StrPipeVar] = None,
        response_types: List[StrPipeVar] = None,
        inference_instances: Optional[List[StrPipeVar]] = None,
        transform_instances: Optional[List[StrPipeVar]] = None,
        model_metrics: Optional[ModelMetrics] = None,
        metadata_properties: Optional[MetadataProperties] = None,
        marketplace_cert: bool = False,
        approval_status: Optional[StrPipeVar] = None,
        description: Optional[str] = None,
        drift_check_baselines: Optional[DriftCheckBaselines] = None,
        customer_metadata_properties: Optional[Dict[str, StrPipeVar]] = None,
        validation_specification: Optional[StrPipeVar] = None,
        domain: Optional[StrPipeVar] = None,
        task: Optional[StrPipeVar] = None,
        sample_payload_url: Optional[StrPipeVar] = None,
        nearest_model_name: Optional[StrPipeVar] = None,
        data_input_configuration: Optional[StrPipeVar] = None,
        skip_model_validation: Optional[StrPipeVar] = None,
        source_uri: Optional[StrPipeVar] = None,
        model_card: Optional[Union[ModelPackageModelCard, ModelCard]] = None,
        model_life_cycle: Optional[ModelLifeCycle] = None,
        accept_eula: Optional[bool] = None,
        model_type: Optional[JumpStartModelType] = None,
    ) -> Union[ModelPackage, ModelPackageGroup]:
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the input data.
            response_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the output data.
            inference_instances (list[str] or list[PipelineVariable]): A list of the instance
                types that are used to generate inferences in real-time (default: None).
            transform_instances (list[str] or list[PipelineVariable]): A list of the instance
                types on which a transformation job can be run or on which an endpoint can be
                deployed (default: None).
            model_package_name (str or PipelineVariable): Model Package name, exclusive to
                `model_package_group_name`, using `model_package_name` makes the Model Package
                un-versioned (default: None).
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes
                the Model Package versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str or PipelineVariable): Model Approval Status, values can be
                "Approved", "Rejected", or "PendingManualApproval"
                (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str] or dict[str, PipelineVariable]):
                A dictionary of key-value paired metadata properties (default: None).
            domain (str or PipelineVariable): Domain values can be "COMPUTER_VISION",
                "NATURAL_LANGUAGE_PROCESSING", "MACHINE_LEARNING" (default: None).
            task (str or PipelineVariable): Task values which are supported by Inference Recommender
                are "FILL_MASK", "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION",
                "IMAGE_SEGMENTATION", "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            sample_payload_url (str or PipelineVariable): The S3 path where the sample
                payload is stored (default: None).
            nearest_model_name (str or PipelineVariable): Name of a pre-trained machine learning
                benchmarked by Amazon SageMaker Inference Recommender (default: None).
            data_input_configuration (str or PipelineVariable): Input object for the model
                (default: None).
            skip_model_validation (str or PipelineVariable): Indicates if you want to skip model
                validation. Values can be "All" or "None" (default: None).
            source_uri (str or PipelineVariable): The URI of the source for the model package
                (default: None).
            model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
                quantitative information about a model (default: None).
            model_life_cycle (ModelLifeCycle): ModelLifeCycle object (default: None).
            accept_eula (bool): For models that require a Model Access Config, specify True or
                False to indicate whether model terms of use have been accepted (default: None).
            model_type (JumpStartModelType): Type of JumpStart model (default: None).

        Returns:
            A `sagemaker.model.ModelPackage` instance or pipeline step arguments
            in case the Model instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        
        Note:
            The following parameters are inherited from ModelBuilder.__init__ and do not need
            to be passed to register():
            - image_uri: Use self.image_uri (defined in __init__)
            - framework: Use self.framework (defined in __init__)
            - framework_version: Use self.framework_version (defined in __init__)
        """
        if content_types is not None:
            self.content_types = content_types

        if response_types is not None:
            self.response_types = response_types

        if model_package_group_name is None and model_package_name is None:
            # If model package group and model package name is not set
            # then register to auto-generated model package group
            model_package_group_name = base_name_from_image(
                self.image_uri, default_base_name=ModelPackage.__name__
            )
        if (
            model_package_group_name is not None
            and model_type is not JumpStartModelType.PROPRIETARY
        ):
            container_def = self._prepare_container_def()
            # Handle pipeline models with multiple containers
            if isinstance(container_def, list):
                container_def = update_container_with_inference_params(
                    framework=self.framework,
                    framework_version=self.framework_version,
                    nearest_model_name=nearest_model_name,
                    data_input_configuration=data_input_configuration,
                    container_list=container_def,
                )
            else:
                container_def = update_container_with_inference_params(
                    framework=self.framework,
                    framework_version=self.framework_version,
                    nearest_model_name=nearest_model_name,
                    data_input_configuration=data_input_configuration,
                    container_def=container_def,
                )
        else:
            container_def = {
                "Image": self.image_uri,
            }

            if isinstance(self.s3_model_data_url, dict):
                raise ValueError(
                    "Un-versioned SageMaker Model Package currently cannot be "
                    "created with ModelDataSource."
                )

            if self.s3_model_data_url is not None:
                container_def["ModelDataUrl"] = self.s3_model_data_url

        # Ensure container_def_list is always a list
        container_def_list = container_def if isinstance(container_def, list) else [container_def]
        
        model_pkg_args = get_model_package_args(
            self.content_types,
            self.response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_name=model_package_name,
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            metadata_properties=metadata_properties,
            marketplace_cert=marketplace_cert,
            approval_status=approval_status,
            description=description,
            container_def_list=container_def_list,
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            validation_specification=validation_specification,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
            skip_model_validation=skip_model_validation,
            source_uri=source_uri,
            model_card=model_card,
            model_life_cycle=model_life_cycle,
        )

        model_package_response = create_model_package_from_containers(
            self.sagemaker_session,
            **model_pkg_args
        )

        if isinstance(self.sagemaker_session, PipelineSession):
            return

        # cannot return a ModelPackage object here as ModelPackage.get() needs model
        # package name, but versioned model package does not have a name
        logger.info(
            "ModelPackage created with model_package_arn=%s",
            model_package_response.get("ModelPackageArn"),
        )

        return model_package_response.get("ModelPackageArn")

