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
"""Utility functions and mixins for ModelBuilder.

This module provides utility functions for:
- Session management and initialization
- Instance type detection and optimization
- Container image auto-detection
- HuggingFace and JumpStart model handling
- Resource requirement calculation
- Framework serialization support
- MLflow model integration
- General model deployment utilities

Example:
    Basic usage as a mixin class::
    
        class MyModelBuilder(ModelBuilderUtils):
            def __init__(self):
                self.model = "huggingface-model-id"
                self.instance_type = "ml.g5.xlarge"
                
            def build(self):
                self._auto_detect_image_uri()
                return self.image_uri
"""
from __future__ import absolute_import, annotations

# Standard library imports
import importlib.util
import sys
import shutil
import json
import os
import platform
import re
import uuid
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
from packaging.version import Version

# SageMaker core imports
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.utils.utils import logger

from sagemaker.train import ModelTrainer

# SageMaker serve imports
from sagemaker.serve.compute_resource_requirements import ResourceRequirements
from sagemaker.serve.constants import (
    DEFAULT_SERIALIZERS_BY_FRAMEWORK,
    Framework,
)
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.builder.serve_settings import _ServeSettings
from sagemaker.serve.detector.image_detector import (
    _cast_to_compatible_version,
    _detect_framework_and_version,
    auto_detect_container,
    _get_model_base
)
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils import task
from sagemaker.serve.utils.exceptions import TaskNotFoundException
from sagemaker.serve.utils.hardware_detector import _total_inference_model_size_mib
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.resources import Model

# MLflow imports
from sagemaker.serve.model_format.mlflow.constants import (
    MLFLOW_METADATA_FILE,
    MLFLOW_MODEL_PATH,
    MLFLOW_PIP_DEPENDENCY_FILE,
    MLFLOW_REGISTRY_PATH_REGEX,
    MLFLOW_RUN_ID_REGEX,
    MLFLOW_TRACKING_ARN,
    MODEL_PACKAGE_ARN_REGEX,
)
from sagemaker.serve.model_format.mlflow.utils import (
    _copy_directory_contents,
    _download_s3_artifacts,
    _generate_mlflow_artifact_path,
    _get_all_flavor_metadata,
    _get_default_model_server_for_mlflow,
    _get_deployment_flavor,
    _select_container_for_mlflow_model,
    _validate_input_for_mlflow,
)

# SageMaker utils imports
from sagemaker.core.deserializers import JSONDeserializer
from sagemaker.core.jumpstart.accessors import JumpStartS3PayloadAccessor
from sagemaker.core.jumpstart.factory.utils import get_init_kwargs, get_deploy_kwargs
from sagemaker.core.jumpstart.utils import (
    get_jumpstart_base_name_if_jumpstart_model,
    get_jumpstart_content_bucket,
    get_eula_message,
    get_metrics_from_deployment_configs,
    add_instance_rate_stats_to_benchmark_metrics,
)
from sagemaker.core.jumpstart.types import DeploymentConfigMetadata
from sagemaker.core.jumpstart import accessors
from sagemaker.core.enums import Tag
from sagemaker.core.local.local_session import LocalSession
from sagemaker.core.s3 import S3Downloader
from sagemaker.core.serializers import NumpySerializer
from sagemaker.core.common_utils import (
    Tags,
    _validate_new_tags,
    base_name_from_image,
    remove_tag_with_key,
)
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core import model_uris
from sagemaker.serve.utils.local_hardware import _get_available_gpus
from sagemaker.core.base_serializers import JSONSerializer
from sagemaker.core.deserializers import JSONDeserializer
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.builder.requirements_manager import RequirementsManager
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)
from sagemaker.core.remote_function.core.serialization import _MetaData
from sagemaker.serve.model_server.triton.config_template import CONFIG_TEMPLATE

SPECULATIVE_DRAFT_MODEL = "/opt/ml/additional-model-data-sources"
_DJL_MODEL_BUILDER_ENTRY_POINT = "inference.py"
_NO_JS_MODEL_EX = "HuggingFace JumpStart Model ID not detected. Building for HuggingFace Model ID."
_JS_SCOPE = "inference"
_CODE_FOLDER = "code"
_JS_MINIMUM_VERSION_IMAGE = "{}:0.31.0-lmi13.0.0-cu124"

_INVALID_DJL_SAMPLE_DATA_EX = (
    'For djl-serving, sample input must be of {"inputs": str, "parameters": dict}, '
    'sample output must be of [{"generated_text": str,}]'
)
_INVALID_TGI_SAMPLE_DATA_EX = (
    'For tgi, sample input must be of {"inputs": str, "parameters": dict}, '
    'sample output must be of [{"generated_text": str,}]'
)

SUPPORTED_TRITON_MODE = {Mode.LOCAL_CONTAINER, Mode.SAGEMAKER_ENDPOINT, Mode.IN_PROCESS}
SUPPORTED_TRITON_FRAMEWORK = {Framework.PYTORCH, Framework.TENSORFLOW}
INPUT_NAME = "input_1"
OUTPUT_NAME = "output_1"

TRITON_IMAGE_ACCOUNT_ID_MAP = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}

GPU_INSTANCE_FAMILIES = {
    "ml.g4dn",
    "ml.g5",
    "ml.p3",
    "ml.p3dn",
    "ml.p4",
    "ml.p4d",
    "ml.p4de",
    "local_gpu",
}

TRITON_IMAGE_BASE = "{account_id}.dkr.ecr.{region}.{base}/sagemaker-tritonserver:{version}-py3"
LATEST_VERSION = "23.02"
VERSION_FOR_TF1 = "23.02"


class TritonSerializer(JSONSerializer):
    """A wrapper of JSONSerializer because Triton expects input to be certain format"""

    def __init__(self, input_serializer, dtype: str, content_type="application/json"):
        """Initialize TritonSerializer with input serializer and data type."""
        super().__init__(content_type)
        self.input_serializer = input_serializer
        self.dtype = dtype

    def serialize(self, data):
        """Serialize data into Triton-compatible format."""
        numpy_data = self.input_serializer.serialize(data)
        payload = {
            "inputs": [
                {
                    "name": INPUT_NAME,
                    "shape": numpy_data.shape,
                    "datatype": self.dtype,
                    "data": numpy_data.tolist(),
                }
            ]
        }

        return super().serialize(payload)

class _ModelBuilderUtils:
    """Utility mixin class providing common functionality for ModelBuilder.
    
    This class provides utility methods for:
    - Session management and initialization
    - Instance type detection and optimization  
    - Container image auto-detection
    - HuggingFace and JumpStart model handling
    - Resource requirement calculation
    - Framework serialization support
    - MLflow model integration
    - General model deployment utilities
    
    This class is designed to be used as a mixin with ModelBuilder classes.
    It expects certain attributes to be available on the instance:
    - sagemaker_session: SageMaker session object
    - model: Model identifier or object
    - instance_type: EC2 instance type
    - region: AWS region
    - env_vars: Environment variables dict
    
    Example:
        class MyModelBuilder(ModelBuilderUtils):
            def __init__(self):
                self.model = "huggingface-model-id"
                self.instance_type = "ml.g5.xlarge"
                self.sagemaker_session = None
                
            def build(self):
                self._init_sagemaker_session_if_does_not_exist()
                self._auto_detect_image_uri()
                return self.image_uri
    """

    # ========================================
    # Session Management
    # ========================================

    def _init_sagemaker_session_if_does_not_exist(self, instance_type: Optional[str] = None) -> None:
        """Initialize SageMaker session if it doesn't exist.

        Sets self.sagemaker_session to LocalSession for local instances,
        or regular Session for remote instances.

        Args:
            instance_type: EC2 instance type to determine session type.
                          If None, uses self.instance_type.
        """
        if self.sagemaker_session:
            return

        effective_instance_type = instance_type or getattr(self, 'instance_type', None)
        
        if effective_instance_type in ("local", "local_gpu"):
            self.sagemaker_session = LocalSession(
                sagemaker_config=getattr(self, '_sagemaker_config', None)
            )
        else:
            # Create session with correct region
            if hasattr(self, 'region') and self.region:
                import boto3
                boto_session = boto3.Session(region_name=self.region)
                self.sagemaker_session = Session(
                    boto_session=boto_session,
                    sagemaker_config=getattr(self, '_sagemaker_config', None)
                )
            else:
                self.sagemaker_session = Session(
                    sagemaker_config=getattr(self, '_sagemaker_config', None)
                )

    # ========================================
    # Instance Type Detection
    # ========================================

    def _get_jumpstart_recommended_instance_type(self) -> Optional[str]:
        """Get recommended instance type from JumpStart metadata.
        
        Returns:
            Recommended instance type string, or None if not available.
        """
        try:
            deploy_kwargs = get_deploy_kwargs(
                model_id=self.model,
                model_version=getattr(self, 'model_version', None) or "*",
                region=self.region,
                tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
                tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
            )
            
            # JumpStart provides recommended instance type
            if hasattr(deploy_kwargs, 'instance_type') and deploy_kwargs.instance_type:
                return deploy_kwargs.instance_type
                
        except Exception:
            pass
        
        return None

    def _get_default_instance_type(self) -> str:
        """Get optimal default instance type based on model characteristics.
        
        Analyzes the model to determine appropriate instance type:
        - JumpStart models: Use recommended instance type from metadata
        - HuggingFace models: Analyze model size and tags for GPU requirements
        - Fallback: ml.m5.large for CPU workloads
        
        Returns:
            Instance type string (e.g., 'ml.g5.xlarge', 'ml.m5.large').
        """
        logger.debug("Auto-detecting optimal instance type for model...")
        
        if isinstance(self.model, str) and self._is_jumpstart_model_id():
            recommended_type = self._get_jumpstart_recommended_instance_type()
            if recommended_type:
                logger.debug(f"Using JumpStart recommended instance type: {recommended_type}")
                return recommended_type
        
        # For HuggingFace models, use metadata to detect requirements
        elif isinstance(self.model, str):
            try:
                env_vars = getattr(self, 'env_vars', {}) or {}
                hf_model_md = self.get_huggingface_model_metadata(
                    self.model, 
                    env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )
                
                # Check model size from metadata
                model_size = hf_model_md.get("safetensors", {}).get("total", 0)
                model_tags = hf_model_md.get("tags", [])
                
                # Large models or specific tags indicate GPU need
                if (model_size > 2_000_000_000 or  # > 2GB
                    any(tag in model_tags for tag in ["7b", "13b", "70b"]) or
                    "7b" in self.model.lower() or "13b" in self.model.lower()):
                    logger.debug("Detected large model, using GPU instance type: ml.g5.xlarge")
                    return "ml.g5.xlarge"
                    
            except Exception as e:
                logger.debug(f"Could not get HF metadata for smart detection: {e}")
        
        # Default fallback
        logger.debug("Using default CPU instance type: ml.m5.large")
        return "ml.m5.large"
    
    # ========================================
    # Image Detection and Container Utils
    # ========================================

    def _auto_detect_container_default(self) -> str:
        """Auto-detect container image for framework-based models.
        
        Detects the appropriate Deep Learning Container (DLC) based on:
        - Model framework (PyTorch, TensorFlow)
        - Framework version from HuggingFace metadata
        - Python version compatibility
        - Instance type requirements
        
        Returns:
            Container image URI string.
            
        Raises:
            ValueError: If instance type not specified or no compatible image found.
        """
        from sagemaker.core import image_uris
        
        logger.debug("Auto-detecting image since image_uri was not provided in ModelBuilder()")

        if not getattr(self, 'instance_type', None):
            raise ValueError(
                "Instance type is not specified. "
                "Unable to detect if the container needs to be GPU or CPU."
            )

        logger.warning(
            "Auto detection is only supported for single models DLCs with a framework backend."
        )

        py_tuple = platform.python_version_tuple()
        env_vars = getattr(self, 'env_vars', {}) or {}
        
        torch_v, tf_v, base_hf_v, _ = self._get_hf_framework_versions(
            self.model, 
            env_vars.get("HUGGING_FACE_HUB_TOKEN")
        )
        
        if torch_v:
            fw, fw_version = "pytorch", torch_v
        elif tf_v:
            fw, fw_version = "tensorflow", tf_v
        else:
            raise ValueError("Could not detect framework from HuggingFace model metadata")

        logger.debug("Auto-detected framework: %s", fw)
        logger.debug("Auto-detected framework version: %s", fw_version)

        casted_versions = _cast_to_compatible_version(fw, fw_version) if fw_version else (None,)
        dlc = None

        for casted_version in filter(None, casted_versions):
            try:
                dlc = image_uris.retrieve(
                    framework=fw,
                    region=self.region,
                    version=casted_version,
                    image_scope="inference",
                    py_version=f"py{py_tuple[0]}{py_tuple[1]}",
                    instance_type=self.instance_type,
                )
                break
            except ValueError:
                pass

        if dlc:
            logger.debug("Auto-detected container: %s. Proceeding with deployment.", dlc)
            return dlc

        raise ValueError(
            f"Unable to auto-detect a DLC for framework {fw}, "
            f"framework version {fw_version} and python version py{py_tuple[0]}{py_tuple[1]}. "
            f"Please manually provide image_uri to ModelBuilder()"
        )
    

    def _get_smd_image_uri(self, processing_unit: Optional[str] = None) -> str:
        """Get SageMaker Distribution (SMD) inference image URI.
        
        Retrieves the appropriate SMD container image for custom orchestrator deployment.
        Requires Python >= 3.12 for SMD inference.
        
        Args:
            processing_unit: Target processing unit ('cpu' or 'gpu').
                           If None, defaults to 'cpu'.
        
        Returns:
            SMD inference image URI string.
            
        Raises:
            ValueError: If Python version < 3.12 or invalid processing unit.
        """
        import sys
        from sagemaker.core import image_uris

        if not self.sagemaker_session:
            if hasattr(self, 'region') and self.region:
                import boto3
                boto_session = boto3.Session(region_name=self.region)
                self.sagemaker_session = Session(boto_session=boto_session)
            else:
                self.sagemaker_session = Session()

        formatted_py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        if Version(f"{sys.version_info.major}{sys.version_info.minor}") < Version("3.12"):
            raise ValueError(
                f"Found Python version {formatted_py_version} but "
                f"custom orchestrator deployment requires Python version >= 3.12."
            )

        INSTANCE_TYPES = {"cpu": "ml.c5.xlarge", "gpu": "ml.g5.4xlarge"}
        effective_processing_unit = processing_unit or "cpu"
        
        if effective_processing_unit not in INSTANCE_TYPES:
            raise ValueError(
                f"Invalid processing unit '{effective_processing_unit}'. "
                f"Must be one of: {list(INSTANCE_TYPES.keys())}"
            )

        logger.debug("Finding SMD inference image URI for a %s instance.", effective_processing_unit)

        smd_uri = image_uris.retrieve(
            framework="sagemaker-distribution",
            image_scope="inference",
            instance_type=INSTANCE_TYPES[effective_processing_unit],
            region=self.region,
        )
        logger.debug("Found compatible image: %s", smd_uri)
        return smd_uri


    def _is_huggingface_model(self) -> bool:
        """Check if model is a HuggingFace model ID.
        
        Determines if the model string represents a HuggingFace model by:
        - Checking for organization/model-name format
        - Checking explicit model_type designation
        - Fallback: assume HuggingFace if not JumpStart
        
        Returns:
            True if model appears to be a HuggingFace model ID.
        """
        if not isinstance(self.model, str):
            return False
        
        # Simple pattern matching for HuggingFace model IDs
        # Format: "organization/model-name" or just "model-name"
        model_type = getattr(self, 'model_type', None)
        if "/" in self.model or model_type == "huggingface":
            return True
        
        # Additional check: if it's not a JumpStart model, assume HuggingFace
        return not self._is_jumpstart_model_id()
    

    def _get_supported_version(self, hf_config: Dict[str, Any], hugging_face_version: str, base_fw: str) -> str:
        """Extract supported framework version from HuggingFace config.
        
        Uses the HuggingFace JSON config to pick the best supported version
        for the given framework.
        
        Args:
            hf_config: HuggingFace configuration dictionary
            hugging_face_version: HuggingFace transformers version
            base_fw: Base framework name (e.g., 'pytorch', 'tensorflow')
            
        Returns:
            Best supported framework version string.
        """
        version_config = hf_config.get("versions", {}).get(hugging_face_version, {})
        versions_to_return = []
        
        for key in version_config.keys():
            if key.startswith(base_fw):
                base_fw_version = key[len(base_fw):]
                if len(hugging_face_version.split(".")) == 2:
                    base_fw_version = ".".join(base_fw_version.split(".")[:-1])
                versions_to_return.append(base_fw_version)
                
        if not versions_to_return:
            raise ValueError(f"No supported versions found for framework {base_fw}")
            
        return sorted(versions_to_return, reverse=True)[0]

    def _get_hf_framework_versions(self, model_id: str, hf_token: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str, str]:
        """Get HuggingFace framework versions for image_uris.retrieve().
        
        Analyzes HuggingFace model metadata to determine the appropriate
        framework versions for container image selection.
        
        Args:
            model_id: HuggingFace model identifier
            hf_token: Optional HuggingFace API token for private models
        
        Returns:
            Tuple of (pytorch_version, tensorflow_version, transformers_version, py_version).
            One of pytorch_version or tensorflow_version will be None.
            
        Raises:
            ValueError: If no supported framework versions found.
        """
        from sagemaker.core import image_uris
        
        # Get model metadata for framework detection
        hf_model_md = self.get_huggingface_model_metadata(model_id, hf_token)
        
        # Get HuggingFace framework configuration
        hf_config = image_uris.config_for_framework("huggingface").get("inference")
        config = hf_config["versions"]
        base_hf_version = sorted(config.keys(), key=lambda v: Version(v), reverse=True)[0]
        
        model_tags = hf_model_md.get("tags", [])
        
        # Detect framework from model tags
        if "pytorch" in model_tags:
            pytorch_version = self._get_supported_version(hf_config, base_hf_version, "pytorch")
            py_version = config[base_hf_version][f"pytorch{pytorch_version}"].get("py_versions", [])[-1]
            return pytorch_version, None, base_hf_version, py_version
            
        elif "keras" in model_tags or "tensorflow" in model_tags:
            tensorflow_version = self._get_supported_version(hf_config, base_hf_version, "tensorflow")
            py_version = config[base_hf_version][f"tensorflow{tensorflow_version}"].get("py_versions", [])[-1]
            return None, tensorflow_version, base_hf_version, py_version
            
        else:
            # Default to PyTorch if no framework detected (matches V2 behavior)
            pytorch_version = self._get_supported_version(hf_config, base_hf_version, "pytorch")
            py_version = config[base_hf_version][f"pytorch{pytorch_version}"].get("py_versions", [])[-1]
            return pytorch_version, None, base_hf_version, py_version
    

    def _detect_jumpstart_image(self) -> None:
        """Detect and set image URI for JumpStart models.
        
        Uses JumpStart metadata to determine the appropriate container image
        and framework information for the model.
        
        Raises:
            ValueError: If image URI cannot be determined or JumpStart lookup fails.
        """
        try:
            init_kwargs = get_init_kwargs(
                model_id=self.model,
                model_version=getattr(self, 'model_version', None) or "*",
                region=self.region,
                instance_type=getattr(self, 'instance_type', None),
                tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
                tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
            )
            
            self.image_uri = init_kwargs.get("image_uri")
            if not self.image_uri:
                raise ValueError(f"Could not determine image URI for JumpStart model: {self.model}")
                
            logger.debug("Auto-detected JumpStart image: %s", self.image_uri)
            self.framework, self.framework_version = self._extract_framework_from_image_uri()
        
        except Exception as e:
            raise ValueError(f"Failed to auto-detect image for JumpStart model {self.model}: {e}") from e

    
    def _detect_huggingface_image(self) -> None:
        """Detect and set image URI for HuggingFace models based on model server.
        
        Automatically selects the appropriate container image based on:
        - Explicit model_server setting
        - Model task type from HuggingFace metadata
        - Framework requirements and versions
        
        Raises:
            ValueError: If image detection fails or unsupported model server.
        """
        from sagemaker.core import image_uris
        
        try:
            env_vars = getattr(self, 'env_vars', {}) or {}
            
            # Determine which model server we're using
            model_server = getattr(self, 'model_server', None)
            if not model_server:
                # Auto-select model server based on HF model task
                hf_model_md = self.get_huggingface_model_metadata(
                    self.model, 
                    env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )
                model_task = hf_model_md.get("pipeline_tag")
                
                if model_task == "text-generation":
                    effective_model_server = ModelServer.TGI
                elif model_task in ["sentence-similarity", "feature-extraction"]:
                    effective_model_server = ModelServer.TEI
                else:
                    effective_model_server = ModelServer.MMS  # Transformers
            else:
                effective_model_server = model_server
            
            # Choose image based on effective model server
            if effective_model_server == ModelServer.TGI:
                # TGI: Use image_uris.retrieve with "huggingface-llm" framework
                self.image_uri = image_uris.retrieve(
                    "huggingface-llm",
                    region=self.region,
                    version=None,  # Use latest version
                    image_scope="inference",
                )
                self.framework = Framework.HUGGINGFACE

            elif effective_model_server == ModelServer.TEI:
                # TEI: Use image_uris.retrieve with "huggingface-tei" framework
                self.image_uri = image_uris.retrieve(
                    framework="huggingface-tei",
                    image_scope="inference",
                    instance_type=getattr(self, 'instance_type', None),
                    region=self.region,
                )
                self.framework = Framework.HUGGINGFACE
                
            elif effective_model_server == ModelServer.DJL_SERVING:
                # DJL: Use image_uris.retrieve with "djl-lmi" framework (matches DJLModel default)
                self.image_uri = image_uris.retrieve(
                    framework="djl-lmi",
                    region=self.region,
                    version="latest",
                    image_scope="inference",
                    instance_type=getattr(self, 'instance_type', None)
                )
                self.framework = Framework.DJL
                
            elif effective_model_server == ModelServer.MMS:  # Transformers
                # Transformers: Use HuggingFace framework with detected versions
                pytorch_version, tensorflow_version, transformers_version, py_version = \
                    self._get_hf_framework_versions(
                        self.model, 
                        env_vars.get("HUGGING_FACE_HUB_TOKEN")
                    )
                
                base_framework_version = (
                    f"pytorch{pytorch_version}" if pytorch_version 
                    else f"tensorflow{tensorflow_version}"
                )
                
                self.image_uri = image_uris.retrieve(
                    framework="huggingface",
                    region=self.region,
                    version=transformers_version,
                    py_version=py_version,
                    instance_type=getattr(self, 'instance_type', None),
                    image_scope="inference",
                    base_framework_version=base_framework_version,
                )
                self.framework = Framework.HUGGINGFACE
                
            elif effective_model_server == ModelServer.TORCHSERVE:
                # TorchServe: Use HuggingFace framework with detected versions
                pytorch_version, tensorflow_version, transformers_version, py_version = \
                    self._get_hf_framework_versions(
                        self.model, 
                        env_vars.get("HUGGING_FACE_HUB_TOKEN")
                    )
                
                base_framework_version = (
                    f"pytorch{pytorch_version}" if pytorch_version 
                    else f"tensorflow{tensorflow_version}"
                )
                
                self.image_uri = image_uris.retrieve(
                    framework="huggingface",
                    region=self.region,
                    version=transformers_version,
                    py_version=py_version,
                    instance_type=getattr(self, 'instance_type', None),
                    image_scope="inference",
                    base_framework_version=base_framework_version,
                )
                self.framework = Framework.HUGGINGFACE
                
            elif effective_model_server == ModelServer.TRITON:
                # Triton: Uses custom image construction (not image_uris.retrieve)
                raise ValueError(
                    "Triton image detection for HuggingFace models requires custom implementation"
                )
                
            elif effective_model_server == ModelServer.TENSORFLOW_SERVING:
                # TensorFlow Serving: V2 required explicit image_uri (no auto-detection)
                raise ValueError("TensorFlow Serving requires explicit image_uri specification")
                
            elif effective_model_server == ModelServer.SMD:
                # SMD: Uses _get_smd_image_uri helper
                cpu_or_gpu = self._get_processing_unit()
                self.image_uri = self._get_smd_image_uri(processing_unit=cpu_or_gpu)
                self.framework = Framework.SMD
                
            else:
                raise ValueError(
                    f"Unsupported model server for HuggingFace models: {effective_model_server}"
                )
                
            logger.debug("Auto-detected HuggingFace image: %s", self.image_uri)
            
        except Exception as e:
            raise ValueError(
                f"Failed to auto-detect image for HuggingFace model {self.model}: {e}"
            ) from e


    def _detect_model_object_image(self) -> None:
        """Detect image for legacy object-based models.
        
        Handles model objects (not string IDs) by using the auto_detect_container
        function to determine appropriate container image.
        
        Raises:
            ValueError: If neither model nor inference_spec available for detection.
        """
        model = getattr(self, 'model', None)
        inference_spec = getattr(self, 'inference_spec', None)
        model_path = getattr(self, 'model_path', None)
        
        if model:
            logger.debug(
                "Auto-detecting container URL for the provided model on instance %s",
                getattr(self, 'instance_type', None),
            )
            self.image_uri, fw, self.framework_version = auto_detect_container(
                model, 
                self.region, 
                getattr(self, 'instance_type', None)
            )
            self.framework = self._normalize_framework_to_enum(fw)

        elif inference_spec:
            logger.warning(
                "model_path provided with no image_uri. Attempting to auto-detect the image "
                "by loading the model using inference_spec.load()..."
            )
            self.image_uri, fw, self.framework_version = auto_detect_container(
                inference_spec.load(model_path),
                self.region,
                getattr(self, 'instance_type', None),
            )
            self.framework = self._normalize_framework_to_enum(fw)
        else:
            raise ValueError("Cannot detect required model or inference spec")
        

    def _auto_detect_image_uri(self) -> None:
        """Auto-detect container image URI based on model type.
        
        Determines the appropriate container image by:
        1. Using provided image_uri if available
        2. For string models: JumpStart vs HuggingFace detection
        3. For object models: Legacy auto-detection
        
        Sets self.image_uri, self.framework, and self.framework_version.
        
        Raises:
            ValueError: If image cannot be auto-detected for the model type.
        """
        image_uri = getattr(self, 'image_uri', None)
        if image_uri:
            self.framework, self.framework_version = self._extract_framework_from_image_uri()
            logger.debug("Skipping auto-detection as image_uri is provided: %s", image_uri)
            return

        if isinstance(self.model, ModelTrainer):
            self._detect_inference_image_from_training()
            return

        model = getattr(self, 'model', None)
        inference_spec = getattr(self, 'inference_spec', None)

        if isinstance(model, str):
            # V3: String-based model detection
            model_type = getattr(self, 'model_type', None)
            
            # First priority: Use model_type if it indicates JumpStart
            if model_type in ["open_weights", "proprietary"]:
                self._detect_jumpstart_image()
            else:
                # model_type is None - use pattern-based detection
                if self._is_jumpstart_model_id():
                    self._detect_jumpstart_image()
                elif self._is_huggingface_model():
                    self._detect_huggingface_image()
                else:
                    raise ValueError(f"Cannot auto-detect image for model: {model}")
        elif inference_spec and hasattr(inference_spec, 'get_model'):
            try:
                spec_model = inference_spec.get_model()
                if spec_model is None:
                    logger.warning(
                        "InferenceSpec.get_model() returned None. If you are using a JumpStar or HuggingFace model, you may need to implement get_model() in your InferenceSpec class")
                        
                if isinstance(spec_model, str):
                    # Temporarily set model for detection, then restore
                    original_model = self.model
                    self.model = spec_model
                    
                    # Use existing detection logic
                    if self._is_jumpstart_model_id():
                        self._detect_jumpstart_image()
                    elif self._is_huggingface_model():
                        self._detect_huggingface_image()
                    else:
                        raise ValueError(f"Cannot auto-detect image for inference_spec model: {spec_model}")
                    
                    # Restore original model
                    self.model = original_model
                    return
            except Exception as e:
                pass
        
            # Fall back to existing object detection
            self._detect_model_object_image()
        else:
            # V2: Object-based model detection
            self._detect_model_object_image()
    

    # ========================================
    # HuggingFace Jumpstart Utils
    # ========================================

    def _use_jumpstart_equivalent(self) -> bool:
        """Check if HuggingFace model has JumpStart equivalent and use it.
        
        Replaces the HuggingFace model with its JumpStart equivalent if available.
        Skips replacement if image_uri or env_vars are explicitly provided.
        
        Returns:
            True if JumpStart equivalent was found and used, False otherwise.
        """
        # Do not use the equivalent JS model if image_uri or env_vars is provided
        image_uri = getattr(self, 'image_uri', None)
        env_vars = getattr(self, 'env_vars', None)
        if image_uri or env_vars:
            return False
            
        if not hasattr(self, "_has_jumpstart_equivalent"):
            self._jumpstart_mapping = self._retrieve_hugging_face_model_mapping()
            self._has_jumpstart_equivalent = self.model in self._jumpstart_mapping
            
        if self._has_jumpstart_equivalent:
            # Use schema builder from HF model metadata
            schema_builder = getattr(self, 'schema_builder', None)
            if not schema_builder:
                model_task = None
                model_metadata = getattr(self, 'model_metadata', None)
                if model_metadata:
                    model_task = model_metadata.get("HF_TASK")
                    
                hf_model_md = self.get_huggingface_model_metadata(self.model)
                if not model_task:
                    model_task = hf_model_md.get("pipeline_tag")
                if model_task:
                    self._hf_schema_builder_init(model_task)

            huggingface_model_id = self.model
            jumpstart_model_id = self._jumpstart_mapping[huggingface_model_id]["jumpstart-model-id"]
            self.model = jumpstart_model_id
            merged_date = self._jumpstart_mapping[huggingface_model_id].get("merged-at")
            
            # Call _build_for_jumpstart if method exists
            if hasattr(self, '_build_for_jumpstart'):
                self._build_for_jumpstart()
                
            compare_model_diff_message = (
                "If you want to identify the differences between the two, "
                "please use model_uris.retrieve() to retrieve the model "
                "artifact S3 URI and compare them."
            )
            
            is_gated = (hasattr(self, '_is_gated_model') and self._is_gated_model())
            
            logger.warning(
                "Please note that for this model we are using the JumpStart's "
                f'local copy "{jumpstart_model_id}" '
                f'of the HuggingFace model "{huggingface_model_id}" you chose. '
                "We strive to keep our local copy synced with the HF model hub closely. "
                "This model was synced "
                f"{f'on {merged_date}' if merged_date else 'before 11/04/2024'}. "
                f"{compare_model_diff_message if not is_gated else ''}"
            )
            return True
        return False


    def _hf_schema_builder_init(self, model_task: str) -> None:
        """Initialize schema builder for HuggingFace model task.

        Attempts to load I/O schemas locally first, then falls back to remote
        schema retrieval for the given HuggingFace task.

        Args:
            model_task: HuggingFace task name (e.g., 'text-generation', 'text-classification')

        Raises:
            TaskNotFoundException: If I/O schema for the task cannot be found locally or remotely.
        """
        try:
            try:
                sample_inputs, sample_outputs = task.retrieve_local_schemas(model_task)
            except ValueError:
                # Samples could not be loaded locally, try to fetch remote HF schema
                from sagemaker_schema_inference_artifacts.huggingface import \
                    remote_schema_retriever

                if model_task in ("text-to-image", "automatic-speech-recognition"):
                    logger.warning(
                        "HF SchemaBuilder for %s is in beta mode, and is not guaranteed to work "
                        "with all models at this time.",
                        model_task,
                    )
                    
                remote_hf_schema_helper = remote_schema_retriever.RemoteSchemaRetriever()
                (
                    sample_inputs,
                    sample_outputs,
                ) = remote_hf_schema_helper.get_resolved_hf_schema_for_task(model_task)
                
            self.schema_builder = SchemaBuilder(sample_inputs, sample_outputs)
            
        except ValueError as e:
            raise TaskNotFoundException(
                f"HuggingFace Schema builder samples for {model_task} could not be found "
                f"locally or via remote."
            ) from e
        

    def _retrieve_hugging_face_model_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve and preprocess HuggingFace/JumpStart model mapping.
        
        Downloads the mapping file from S3 that contains the correspondence
        between HuggingFace model IDs and their JumpStart equivalents.
        
        Returns:
            Dictionary mapping HuggingFace model IDs to JumpStart model metadata.
            Empty dict if mapping cannot be retrieved.
        """
        converted_mapping = {}
        session = getattr(self, 'sagemaker_session', None)
        if not session:
            return converted_mapping
            
        region = session.boto_region_name
        try:
            mapping_json_object = JumpStartS3PayloadAccessor.get_object_cached(
                bucket=get_jumpstart_content_bucket(region),
                key="hf_model_id_map_cache.json",
                region=region,
                s3_client=session.s3_client,
            )
            mapping = json.loads(mapping_json_object)
        except Exception:
            return converted_mapping

        for k, v in mapping.items():
            converted_mapping[v["hf-model-id"]] = {
                "jumpstart-model-id": k,
                "jumpstart-model-version": v["jumpstart-model-version"],
                "merged-at": v.get("merged-at"),
                "hf-model-repo-sha": v.get("hf-model-repo-sha"),
            }
        return converted_mapping

    def _prepare_hf_model_for_upload(self) -> None:
        """Download HuggingFace model metadata for upload.
        
        Creates a temporary directory and downloads the necessary HuggingFace
        model metadata files if model_path is not already set.
        """
        model_path = getattr(self, 'model_path', None)
        if not model_path:
            self.model_path = f"/tmp/sagemaker/model-builder/{self.model}"
            env_vars = getattr(self, 'env_vars', {}) or {}
            self.download_huggingface_model_metadata(
                self.model,
                os.path.join(self.model_path, "code"),
                env_vars.get("HUGGING_FACE_HUB_TOKEN"),
            )
    



    # ========================================
    # Resource and Hardware Utils
    # ========================================

    def _get_processing_unit(self) -> str:
        """Detect if resource requirements are intended for CPU or GPU instance.
        
        Analyzes resource requirements to determine the target processing unit:
        - Checks for accelerator requirements in resource_requirements
        - Checks for accelerator requirements in modelbuilder_list items
        - Defaults to CPU if no accelerators specified
        
        Returns:
            'gpu' if accelerators are required, 'cpu' otherwise.
        """
        # Assume custom orchestrator will be deployed as an endpoint to a CPU instance
        resource_requirements = getattr(self, 'resource_requirements', None)
        if not resource_requirements or not getattr(resource_requirements, 'num_accelerators', None):
            modelbuilder_list = getattr(self, 'modelbuilder_list', None) or []
            for ic in modelbuilder_list:
                ic_resource_req = getattr(ic, 'resource_requirements', None)
                if ic_resource_req and getattr(ic_resource_req, 'num_accelerators', 0) > 0:
                    return "gpu"
            return "cpu"
            
        if getattr(resource_requirements, 'num_accelerators', 0) > 0:
            return "gpu"

        return "cpu"


    def _get_inference_component_resource_requirements(self, mb) -> None:
        """Fetch pre-benchmarked resource requirements from JumpStart.
        
        Attempts to retrieve and set resource requirements for inference components
        using JumpStart deployment configurations when available.
        
        Raises:
            ValueError: If no resource requirements provided and no JumpStart configs found.
        """
        resource_requirements = getattr(mb, 'resource_requirements', None)
        if mb._is_jumpstart_model_id() and not resource_requirements:
            if not hasattr(mb, 'list_deployment_configs'):
                return
                
            deployment_configs = mb.list_deployment_configs()
            if not deployment_configs:
                inference_component_name = getattr(mb, 'inference_component_name', 'Unknown')
                raise ValueError(
                    f"No resource requirements were provided for Inference Component "
                    f"{inference_component_name} and no default deployment "
                    f"configs were found in JumpStart."
                )
                
            compute_requirements = (
                deployment_configs[0].get("DeploymentArgs", {}).get("ComputeResourceRequirements", {})
            )
            
            logger.debug("Retrieved pre-benchmarked deployment configurations from JumpStart.")
            
            mb.resource_requirements = ResourceRequirements(
                requests={
                    "memory": compute_requirements.get("MinMemoryRequiredInMb"),
                    "num_accelerators": compute_requirements.get(
                        "NumberOfAcceleratorDevicesRequired", None
                    ),
                    "copies": 1,
                    "num_cpus": compute_requirements.get("NumberOfCpuCoresRequired", None),
                },
                limits={"memory": compute_requirements.get("MaxMemoryRequiredInMb", None)},
            )
            
        return mb
    

    def _can_fit_on_single_gpu(self) -> bool:
        """Check if model can fit on a single GPU.

        Compares the total inference model size with single GPU memory capacity
        to determine if the model can fit on a single GPU device.

        Returns:
            True if model size <= single GPU memory size, False otherwise.
        """
        try:
            if not hasattr(self, '_try_fetch_gpu_info'):
                return False
                
            single_gpu_size_mib = self._try_fetch_gpu_info()
            env_vars = getattr(self, 'env_vars', {}) or {}
            
            model_size_mib = _total_inference_model_size_mib(
                self.model, 
                env_vars.get("dtypes", "float32")
            )
            
            if model_size_mib <= single_gpu_size_mib:
                logger.debug(
                    "Total inference model size: %s MiB, single GPU size: %s MiB",
                    model_size_mib,
                    single_gpu_size_mib,
                )
                return True
            return False
            
        except ValueError:
            instance_type = getattr(self, 'instance_type', 'Unknown')
            logger.debug("Unable to determine single GPU size for instance %s", instance_type)
            return False



    # ========================================
    # Serialization Utils
    # ========================================

    def _extract_framework_from_image_uri(self) -> Tuple[Optional[Framework], Optional[str]]:
        """Extract framework and version information from SageMaker image URI.
        
        Analyzes the container image URI to determine the ML framework
        and version being used.
        
        Returns:
            Tuple of (Framework enum, version string). Both can be None if not detected.
        """
        image_uri = getattr(self, 'image_uri', None)
        if not image_uri:
            return None, None
        
        if "pytorch-inference" in image_uri or "pytorch-training" in image_uri:
            version_match = re.search(r'pytorch.*:(\d+\.\d+\.\d+)', image_uri)
            return Framework.PYTORCH, version_match.group(1) if version_match else None
        
        elif "tensorflow-inference" in image_uri or "tensorflow-training" in image_uri:
            version_match = re.search(r'tensorflow.*:(\d+\.\d+\.\d+)', image_uri)
            return Framework.TENSORFLOW, version_match.group(1) if version_match else None
        
        elif "sagemaker-xgboost" in image_uri:
            version_match = re.search(r'sagemaker-xgboost:(\d+\.\d+)', image_uri)
            return Framework.XGBOOST, version_match.group(1) if version_match else None
        
        elif "sagemaker-scikit-learn" in image_uri:
            version_match = re.search(r'scikit-learn:(\d+\.\d+)', image_uri)
            return Framework.SKLEARN, version_match.group(1) if version_match else None
        
        elif "huggingface" in image_uri:
            return Framework.HUGGINGFACE, None
        
        elif "mxnet" in image_uri:
            version_match = re.search(r'mxnet.*:(\d+\.\d+\.\d+)', image_uri)
            return Framework.MXNET, version_match.group(1) if version_match else None
        
        return None, None
    

    def _fetch_serializer_and_deserializer_for_framework(self, framework: str) -> Tuple[Any, Any]:
        """Fetch default serializer and deserializer for a framework.

        Args:
            framework: Framework name as string.

        Returns:
            Tuple containing (serializer, deserializer) instances.
            Defaults to (NumpySerializer, JSONDeserializer) if framework not found.
        """
        framework_enum = self._normalize_framework_to_enum(framework)
        if framework_enum and framework_enum in DEFAULT_SERIALIZERS_BY_FRAMEWORK:
            return DEFAULT_SERIALIZERS_BY_FRAMEWORK[framework_enum]
        return NumpySerializer(), JSONDeserializer()
    

    def _normalize_framework_to_enum(self, framework: Union[str, Framework, None]) -> Optional[Framework]:
        """Convert any framework input to Framework enum.
        
        Args:
            framework: Framework as string, enum, or None
            
        Returns:
            Framework enum or None if not found/None input
        """
        if framework is None:
            return None
        
        if isinstance(framework, Framework):
            return framework
        
        if not isinstance(framework, str):
            return None
        
        framework_mapping = {
            "xgboost": Framework.XGBOOST,
            "xgb": Framework.XGBOOST,
            "pytorch": Framework.PYTORCH,
            "torch": Framework.PYTORCH,
            "tensorflow": Framework.TENSORFLOW,
            "tf": Framework.TENSORFLOW,
            "sklearn": Framework.SKLEARN,
            "scikit-learn": Framework.SKLEARN,
            "scikit_learn": Framework.SKLEARN,
            "sk-learn": Framework.SKLEARN,
            "huggingface": Framework.HUGGINGFACE,
            "hf": Framework.HUGGINGFACE,
            "transformers": Framework.HUGGINGFACE,
            "mxnet": Framework.MXNET,
            "chainer": Framework.CHAINER,
            "djl": Framework.DJL,
            "sparkml": Framework.SPARKML,
            "spark": Framework.SPARKML,
            "lda": Framework.LDA,
            "ntm": Framework.NTM,
            "smd": Framework.SMD,
            "sagemaker-distribution": Framework.SMD,
        }
        
        return framework_mapping.get(framework.lower())


    # ========================================
    # MLflow Utils
    # ========================================

    def _handle_mlflow_input(self) -> None:
        """Check and handle MLflow model input if present.
        
        Detects MLflow model arguments, validates metadata existence,
        and initializes MLflow-specific configurations.
        """
        self._is_mlflow_model = self._has_mlflow_arguments()
        if not self._is_mlflow_model:
            return

        model_metadata = getattr(self, 'model_metadata', {})
        mlflow_model_path = model_metadata.get(MLFLOW_MODEL_PATH)
        if not mlflow_model_path:
            return
            
        artifact_path = self._get_artifact_path(mlflow_model_path)
        if not self._mlflow_metadata_exists(artifact_path):
            return

        self._initialize_for_mlflow(artifact_path)
        
        model_server = getattr(self, 'model_server', None)
        env_vars = getattr(self, 'env_vars', {}) or {}
        _validate_input_for_mlflow(model_server, env_vars.get("MLFLOW_MODEL_FLAVOR"))

    def _has_mlflow_arguments(self) -> bool:
        """Check whether MLflow model arguments are present.

        Returns:
            True if MLflow arguments are present and should be handled, False otherwise.
        """
        inference_spec = getattr(self, 'inference_spec', None)
        model = getattr(self, 'model', None)
        
        if inference_spec or model:
            logger.debug(
                "Either inference spec or model is provided. "
                "ModelBuilder is not handling MLflow model input"
            )
            return False

        model_metadata = getattr(self, 'model_metadata', None)
        if not model_metadata:
            logger.debug(
                "No ModelMetadata provided. ModelBuilder is not handling MLflow model input"
            )
            return False

        mlflow_model_path = model_metadata.get(MLFLOW_MODEL_PATH)
        if not mlflow_model_path:
            logger.debug(
                "%s is not provided in ModelMetadata. ModelBuilder is not handling MLflow model "
                "input",
                MLFLOW_MODEL_PATH,
            )
            return False

        return True

    def _get_artifact_path(self, mlflow_model_path: str) -> str:
        """Retrieve model artifact location from MLflow model path.

        Handles different MLflow path formats:
        - Run ID paths: runs:/<run_id>/<model_path>
        - Registry paths: models:/<model_name>/<version_or_alias>
        - Model package ARNs
        - Direct file paths

        Args:
            mlflow_model_path: MLflow model path input.

        Returns:
            Path to the model artifact.
            
        Raises:
            ValueError: If tracking ARN not provided for run/registry paths.
            ImportError: If sagemaker_mlflow not installed.
        """
        is_run_id_type = re.match(MLFLOW_RUN_ID_REGEX, mlflow_model_path)
        is_registry_type = re.match(MLFLOW_REGISTRY_PATH_REGEX, mlflow_model_path)
        
        if is_run_id_type or is_registry_type:
            model_metadata = getattr(self, 'model_metadata', {})
            mlflow_tracking_arn = model_metadata.get(MLFLOW_TRACKING_ARN)
            if not mlflow_tracking_arn:
                raise ValueError(
                    f"{MLFLOW_TRACKING_ARN} is not provided in ModelMetadata or through set_tracking_arn "
                    f"but MLflow model path was provided."
                )

            if not importlib.util.find_spec("sagemaker_mlflow"):
                raise ImportError(
                    "Unable to import sagemaker_mlflow, check if sagemaker_mlflow is installed"
                )

            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_arn)
            
            if is_run_id_type:
                _, run_id, model_path = mlflow_model_path.split("/", 2)
                artifact_uri = mlflow.get_run(run_id).info.artifact_uri
                if not artifact_uri.endswith("/"):
                    artifact_uri += "/"
                return artifact_uri + model_path

            # Registry path handling
            mlflow_client = mlflow.MlflowClient()
            if not mlflow_model_path.endswith("/"):
                mlflow_model_path += "/"

            if "@" in mlflow_model_path:
                _, model_name_and_alias, artifact_uri = mlflow_model_path.split("/", 2)
                model_name, model_alias = model_name_and_alias.split("@")
                model_metadata = mlflow_client.get_model_version_by_alias(model_name, model_alias)
            else:
                _, model_name, model_version, artifact_uri = mlflow_model_path.split("/", 3)
                model_metadata = mlflow_client.get_model_version(model_name, model_version)

            source = model_metadata.source
            if not source.endswith("/"):
                source += "/"
            return source + artifact_uri

        # Handle model package ARN
        if re.match(MODEL_PACKAGE_ARN_REGEX, mlflow_model_path):
            sagemaker_session = getattr(self, 'sagemaker_session', None)
            if sagemaker_session:
                model_package = sagemaker_session.sagemaker_client.describe_model_package(
                    ModelPackageName=mlflow_model_path
                )
                return model_package["SourceUri"]

        # Direct path
        return mlflow_model_path

    def _mlflow_metadata_exists(self, path: str) -> bool:
        """Check whether MLmodel metadata file exists in the given directory.

        Args:
            path: Directory path to check (local or S3).
            
        Returns:
            True if MLmodel file exists, False otherwise.
        """
        if path.startswith("s3://"):
            s3_downloader = S3Downloader()
            if not path.endswith("/"):
                path += "/"
            s3_uri_to_mlmodel_file = f"{path}{MLFLOW_METADATA_FILE}"
            sagemaker_session = getattr(self, 'sagemaker_session', None)
            if not sagemaker_session:
                return False
            response = s3_downloader.list(s3_uri_to_mlmodel_file, sagemaker_session)
            return len(response) > 0

        file_path = os.path.join(path, MLFLOW_METADATA_FILE)
        return os.path.isfile(file_path)

    def _initialize_for_mlflow(self, artifact_path: str) -> None:
        """Initialize MLflow model artifacts, image URI and model server.

        Downloads artifacts, extracts metadata, and configures model server
        and container image for MLflow model deployment.

        Args:
            artifact_path: Path to the MLflow artifact store.
            
        Raises:
            ValueError: If artifact path is invalid.
        """
        model_path = getattr(self, 'model_path', None)
        sagemaker_session = getattr(self, 'sagemaker_session', None)
        
        if artifact_path.startswith("s3://"):
            _download_s3_artifacts(artifact_path, model_path, sagemaker_session)
        elif os.path.exists(artifact_path):
            _copy_directory_contents(artifact_path, model_path)
        else:
            raise ValueError(f"Invalid path: {artifact_path}")
            
        mlflow_model_metadata_path = _generate_mlflow_artifact_path(
            model_path, MLFLOW_METADATA_FILE
        )
        mlflow_model_dependency_path = _generate_mlflow_artifact_path(
            model_path, MLFLOW_PIP_DEPENDENCY_FILE
        )
        
        flavor_metadata = _get_all_flavor_metadata(mlflow_model_metadata_path)
        deployment_flavor = _get_deployment_flavor(flavor_metadata)

        current_model_server = getattr(self, 'model_server', None)
        self.model_server = current_model_server or _get_default_model_server_for_mlflow(
            deployment_flavor
        )
        
        current_image_uri = getattr(self, 'image_uri', None)
        if not current_image_uri:
            self.image_uri = _select_container_for_mlflow_model(
                mlflow_model_src_path=model_path,
                deployment_flavor=deployment_flavor,
                region=sagemaker_session.boto_region_name if sagemaker_session else None,
                instance_type=getattr(self, 'instance_type', None),
            )
            
        env_vars = getattr(self, 'env_vars', {})
        env_vars.update({"MLFLOW_MODEL_FLAVOR": f"{deployment_flavor}"})
        
        dependencies = getattr(self, 'dependencies', {})
        dependencies.update({"requirements": mlflow_model_dependency_path})



    # ========================================
    # Optimize Utils
    # ========================================

    def _is_inferentia_or_trainium(self, instance_type: Optional[str]) -> bool:
        """Checks whether an instance is compatible with Inferentia.

        Args:
            instance_type (str): The instance type used for the compilation job.

        Returns:
            bool: Whether the given instance type is Inferentia or Trainium.
        """
        if isinstance(instance_type, str):
            match = re.match(r"^ml[\._]([a-z\d\-]+)\.?\w*$", instance_type)
            if match:
                if match[1].startswith("inf") or match[1].startswith("trn"):
                    return True
        return False


    def _is_image_compatible_with_optimization_job(self, image_uri: Optional[str]) -> bool:
        """Checks whether an instance is compatible with an optimization job.

        Args:
            image_uri (str): The image URI of the optimization job.

        Returns:
            bool: Whether the given instance type is compatible with an optimization job.
        """
        if image_uri is None:
            return True
        return "djl-inference:" in image_uri and ("-lmi" in image_uri or "-neuronx-" in image_uri)


    def _deployment_config_contains_draft_model(self, deployment_config: Optional[Dict]) -> bool:
        """Checks whether a deployment config contains a speculative decoding draft model.

        Args:
            deployment_config (Dict): The deployment config to check.

        Returns:
            bool: Whether the deployment config contains a draft model or not.
        """
        if deployment_config is None:
            return False
        deployment_args = deployment_config.get("DeploymentArgs", {})
        additional_data_sources = deployment_args.get("AdditionalDataSources")

        return "speculative_decoding" in additional_data_sources if additional_data_sources else False


    def _is_draft_model_jumpstart_provided(self, deployment_config: Optional[Dict]) -> bool:
        """Checks whether a deployment config's draft model is provided by JumpStart.

        Args:
            deployment_config (Dict): The deployment config to check.

        Returns:
            bool: Whether the draft model is provided by JumpStart or not.
        """
        if deployment_config is None:
            return False

        additional_model_data_sources = deployment_config.get("DeploymentArgs", {}).get(
            "AdditionalDataSources"
        )
        for source in additional_model_data_sources.get("speculative_decoding", []):
            if source["channel_name"] == "draft_model":
                if source.get("provider", {}).get("name") == "JumpStart":
                    return True
                continue
        return False


    def _generate_optimized_model(self, optimization_response: dict):
        """Generates a new optimization model.

        Args:
            pysdk_model (Model): A PySDK model.
            optimization_response (dict): The optimization response.

        Returns:
            Model: A deployable optimized model.
        """
        recommended_image_uri = optimization_response.get("OptimizationOutput", {}).get(
            "RecommendedInferenceImage"
        )
        s3_uri = optimization_response.get("OutputConfig", {}).get("S3OutputLocation")
        deployment_instance_type = optimization_response.get("DeploymentInstanceType")

        if recommended_image_uri:
            self.image_uri = recommended_image_uri
        if s3_uri:
            self.s3_upload_path["S3DataSource"]["S3Uri"] = s3_uri
        if deployment_instance_type:
            self.instance_type = deployment_instance_type

        self.add_tags(
            {"Key": Tag.OPTIMIZATION_JOB_NAME, "Value": optimization_response["OptimizationJobName"]}
        )


    def _is_optimized(self) -> bool:
        """Checks whether an optimization model is optimized.

        Args:
            pysdk_model (Model): A PySDK model.

        Return:
            bool: Whether the given model type is optimized.
        """
        optimized_tags = [Tag.OPTIMIZATION_JOB_NAME, Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER]
        if hasattr(self, "_tags") and self._tags:
            if isinstance(self._tags, dict):
                return self._tags.get("Key") in optimized_tags
            for tag in self._tags:
                if tag.get("Key") in optimized_tags:
                    return True
        return False


    def _generate_model_source(
        self, model_data: Optional[Union[Dict[str, Any], str]], accept_eula: Optional[bool]
    ) -> Optional[Dict[str, Any]]:
        """Extracts model source from model data.

        Args:
            model_data (Optional[Union[Dict[str, Any], str]]): A model data.

        Returns:
            Optional[Dict[str, Any]]: Model source data.
        """
        if model_data is None:
            raise ValueError("Model Optimization Job only supports model with S3 data source.")

        s3_uri = model_data
        if isinstance(s3_uri, dict):
            s3_uri = s3_uri.get("S3DataSource").get("S3Uri")

        model_source = {"S3": {"S3Uri": s3_uri}}
        if accept_eula:
            model_source["S3"]["ModelAccessConfig"] = {"AcceptEula": True}
        return model_source


    def _update_environment_variables(
        self, env: Optional[Dict[str, str]], new_env: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """Updates environment variables based on environment variables.

        Args:
            env (Optional[Dict[str, str]]): The environment variables.
            new_env (Optional[Dict[str, str]]): The new environment variables.

        Returns:
            Optional[Dict[str, str]]: The updated environment variables.
        """
        if new_env:
            if env:
                env.update(new_env)
            else:
                env = new_env
        return env


    def _extract_speculative_draft_model_provider(
        self, speculative_decoding_config: Optional[Dict] = None,
    ) -> Optional[str]:
        """Extracts speculative draft model provider from speculative decoding config.

        Args:
            speculative_decoding_config (Optional[Dict]): A speculative decoding config.

        Returns:
            Optional[str]: The speculative draft model provider.
        """
        if speculative_decoding_config is None:
            return None

        model_provider = speculative_decoding_config.get("ModelProvider", "").lower()

        if model_provider == "jumpstart":
            return "jumpstart"

        if model_provider == "custom" or speculative_decoding_config.get("ModelSource"):
            return "custom"

        if model_provider == "sagemaker":
            return "sagemaker"

        return "auto"


    def _extract_additional_model_data_source_s3_uri(
        self, additional_model_data_source: Optional[Dict] = None,
    ) -> Optional[str]:
        """Extracts model data source s3 uri from a model data source in Pascal case.

        Args:
            additional_model_data_source (Optional[Dict]): A model data source.

        Returns:
            str: S3 uri of the model resources.
        """
        if (
            additional_model_data_source is None
            or additional_model_data_source.get("S3DataSource", None) is None
        ):
            return None

        return additional_model_data_source.get("S3DataSource").get("S3Uri")


    def _extract_deployment_config_additional_model_data_source_s3_uri(
        self, additional_model_data_source: Optional[Dict] = None,
    ) -> Optional[str]:
        """Extracts model data source s3 uri from a model data source in snake case.

        Args:
            additional_model_data_source (Optional[Dict]): A model data source.

        Returns:
            str: S3 uri of the model resources.
        """
        if (
            additional_model_data_source is None
            or additional_model_data_source.get("s3_data_source", None) is None
        ):
            return None

        return additional_model_data_source.get("s3_data_source").get("s3_uri", None)


    def _is_draft_model_gated(
        self, draft_model_config: Optional[Dict] = None,
    ) -> bool:
        """Extracts model gated-ness from draft model data source.

        Args:
            draft_model_config (Optional[Dict]): A model data source.

        Returns:
            bool: Whether the draft model is gated or not.
        """
        return "hosting_eula_key" in draft_model_config if draft_model_config else False


    def _extracts_and_validates_speculative_model_source(
        self, speculative_decoding_config: Dict,
    ) -> str:
        """Extracts model source from speculative decoding config.

        Args:
            speculative_decoding_config (Optional[Dict]): A speculative decoding config.

        Returns:
            str: Model source.

        Raises:
            ValueError: If model source is none.
        """
        model_source: str = speculative_decoding_config.get("ModelSource")

        if not model_source:
            raise ValueError("ModelSource must be provided in speculative decoding config.")
        return model_source


    def _generate_channel_name(self, additional_model_data_sources: Optional[List[Dict]]) -> str:
        """Generates a channel name.

        Args:
            additional_model_data_sources (Optional[List[Dict]]): The additional model data sources.

        Returns:
            str: The channel name.
        """
        channel_name = "draft_model"
        if additional_model_data_sources and len(additional_model_data_sources) > 0:
            channel_name = additional_model_data_sources[0].get("ChannelName", channel_name)

        return channel_name


    def _generate_additional_model_data_sources(
        self, 
        model_source: str,
        channel_name: str,
        accept_eula: bool = False,
        s3_data_type: Optional[str] = "S3Prefix",
        compression_type: Optional[str] = "None",
    ) -> List[Dict]:
        """Generates additional model data sources.

        Args:
            model_source (Optional[str]): The model source.
            channel_name (Optional[str]): The channel name.
            accept_eula (Optional[bool]): Whether to accept eula or not.
            s3_data_type (Optional[str]): The S3 data type, defaults to 'S3Prefix'.
            compression_type (Optional[str]): The compression type, defaults to None.

        Returns:
            List[Dict]: The additional model data sources.
        """

        additional_model_data_source = {
            "ChannelName": channel_name,
            "S3DataSource": {
                "S3Uri": model_source,
                "S3DataType": s3_data_type,
                "CompressionType": compression_type,
            },
        }
        if accept_eula:
            additional_model_data_source["S3DataSource"]["ModelAccessConfig"] = {"AcceptEula": True}

        return [additional_model_data_source]


    def _is_s3_uri(self, s3_uri: Optional[str]) -> bool:
        """Checks whether an S3 URI is valid.

        Args:
            s3_uri (Optional[str]): The S3 URI.

        Returns:
            bool: Whether the S3 URI is valid.
        """
        if s3_uri is None:
            return False

        return re.match("^s3://([^/]+)/?(.*)$", s3_uri) is not None


    def _extract_optimization_config_and_env(
        self,
        quantization_config: Optional[Dict] = None,
        compilation_config: Optional[Dict] = None,
        sharding_config: Optional[Dict] = None,
    ) -> Optional[Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]]:
        """Extracts optimization config and environment variables.

        Args:
            quantization_config (Optional[Dict]): The quantization config.
            compilation_config (Optional[Dict]): The compilation config.
            sharding_config (Optional[Dict]): The sharding config.

        Returns:
            Optional[Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]]:
                The optimization config and environment variables.
        """
        optimization_config = {}
        quantization_override_env = (
            quantization_config.get("OverrideEnvironment") if quantization_config else None
        )
        compilation_override_env = (
            compilation_config.get("OverrideEnvironment") if compilation_config else None
        )
        sharding_override_env = sharding_config.get("OverrideEnvironment") if sharding_config else None

        if quantization_config is not None:
            optimization_config["ModelQuantizationConfig"] = quantization_config

        if compilation_config is not None:
            optimization_config["ModelCompilationConfig"] = compilation_config

        if sharding_config is not None:
            optimization_config["ModelShardingConfig"] = sharding_config

        # Return optimization config dict and environment variables if either is present
        if optimization_config:
            return (
                optimization_config,
                quantization_override_env,
                compilation_override_env,
                sharding_override_env,
            )

        return None, None, None, None


    def _custom_speculative_decoding(
        self,
        speculative_decoding_config: Optional[Dict],
        accept_eula: Optional[bool] = False,
    ):
        """Modifies the given model for speculative decoding config with custom provider.

        Args:
            model (Model): The model.
            speculative_decoding_config (Optional[Dict]): The speculative decoding config.
            accept_eula (Optional[bool]): Whether to accept eula or not.
        """
        if speculative_decoding_config:
            additional_model_source = self._extracts_and_validates_speculative_model_source(
                speculative_decoding_config
            )

            accept_eula = speculative_decoding_config.get("AcceptEula", accept_eula)

            if self._is_s3_uri(additional_model_source):
                channel_name = self._generate_channel_name(self.additional_model_data_sources)
                speculative_draft_model = f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}"

                self.additional_model_data_sources = self._generate_additional_model_data_sources(
                    additional_model_source, channel_name, accept_eula
                )
            else:
                speculative_draft_model = additional_model_source

            self.env_vars.update({"OPTION_SPECULATIVE_DRAFT_MODEL": speculative_draft_model})
            self.add_tags(
                {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "custom"},
            )

    def _get_cached_model_specs(self, model_id, version, region, sagemaker_session):
        """Get cached JumpStart model specs to avoid repeated fetches"""
        if not hasattr(self, '_cached_js_model_specs'):
            self._cached_js_model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
                model_id=model_id,
                version=version,
                region=region,
                sagemaker_session=sagemaker_session,
            )
        return self._cached_js_model_specs


    def _jumpstart_speculative_decoding(
        self,
        speculative_decoding_config: Optional[Dict[str, Any]] = None,
        sagemaker_session: Optional[Session] = None,
    ):
        """Modifies the given model for speculative decoding config with JumpStart provider.

        Args:
            model (Model): The model.
            speculative_decoding_config (Optional[Dict]): The speculative decoding config.
            sagemaker_session (Optional[Session]): Sagemaker session for execution.
        """
        if speculative_decoding_config:
            js_id = speculative_decoding_config.get("ModelID")
            if not js_id:
                raise ValueError(
                    "`ModelID` is a required field in `speculative_decoding_config` when "
                    "using JumpStart as draft model provider."
                )
            
            model_version = speculative_decoding_config.get("ModelVersion", "*")
            accept_eula = speculative_decoding_config.get("AcceptEula", False)
            channel_name = self._generate_channel_name(self.additional_model_data_sources)

            model_specs = self._get_cached_model_specs(
                model_id=js_id,
                version=model_version,
                region=sagemaker_session.boto_region_name,
                sagemaker_session=sagemaker_session,

            )
            
            model_spec_json = model_specs.to_json()

            js_bucket = accessors.JumpStartModelsAccessor.get_jumpstart_content_bucket(self.region)

            if model_spec_json.get("gated_bucket", False):
                if not accept_eula:
                    eula_message = get_eula_message(
                        model_specs=model_specs, region=sagemaker_session.boto_region_name
                    )
                    raise ValueError(
                        f"{eula_message} Set `AcceptEula`=True in "
                        f"speculative_decoding_config once acknowledged."
                    )
                js_bucket = accessors.JumpStartModelsAccessor.get_jumpstart_gated_content_bucket(self.region)

            key_prefix = model_spec_json.get("hosting_prepacked_artifact_key")
            self.additional_model_data_sources = self. _generate_additional_model_data_sources(
                f"s3://{js_bucket}/{key_prefix}",
                channel_name,
                accept_eula,
            )

            self.env_vars.update(
                {"OPTION_SPECULATIVE_DRAFT_MODEL": f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}/"}
            )
            self.add_tags(
                {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "jumpstart"},
            )


    def _optimize_for_hf(
        self,
        output_path: str,
        tags: Optional[Tags] = None,
        job_name: Optional[str] = None,
        quantization_config: Optional[Dict] = None,
        compilation_config: Optional[Dict] = None,
        speculative_decoding_config: Optional[Dict] = None,
        sharding_config: Optional[Dict] = None,
        env_vars: Optional[Dict] = None,
        vpc_config: Optional[Dict] = None,
        kms_key: Optional[str] = None,
        max_runtime_in_sec: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Runs a model optimization job.

        Args:
            output_path (str): Specifies where to store the compiled/quantized model.
            tags (Optional[Tags]): Tags for labeling a model optimization job. Defaults to ``None``.
            job_name (Optional[str]): The name of the model optimization job. Defaults to ``None``.
            quantization_config (Optional[Dict]): Quantization configuration. Defaults to ``None``.
            compilation_config (Optional[Dict]): Compilation configuration. Defaults to ``None``.
            speculative_decoding_config (Optional[Dict]): Speculative decoding configuration.
                Defaults to ``None``
            sharding_config (Optional[Dict]): Model sharding configuration.
                Defaults to ``None``
            env_vars (Optional[Dict]): Additional environment variables to run the optimization
                container. Defaults to ``None``.
            vpc_config (Optional[Dict]): The VpcConfig set on the model. Defaults to ``None``.
            kms_key (Optional[str]): KMS key ARN used to encrypt the model artifacts when uploading
                to S3. Defaults to ``None``.
            max_runtime_in_sec (Optional[int]): Maximum job execution time in seconds. Defaults to
                ``None``.

        Returns:
            Optional[Dict[str, Any]]: Model optimization job input arguments.
        """
        if speculative_decoding_config:
            if speculative_decoding_config.get("ModelProvider", "").lower() == "jumpstart":
                self._jumpstart_speculative_decoding(
                    speculative_decoding_config=speculative_decoding_config,
                    sagemaker_session=self.sagemaker_session,
                )
            else:
                self._custom_speculative_decoding(
                    speculative_decoding_config, False
                )

        if quantization_config or compilation_config or sharding_config:
            create_optimization_job_args = {
                "OptimizationJobName": job_name,
                "DeploymentInstanceType": self.instance_type,
                "RoleArn": self.role_arn,
            }

            if env_vars:
                self.env_vars.update(env_vars)
                create_optimization_job_args["OptimizationEnvironment"] = env_vars

            self._optimize_prepare_for_hf()
            model_source = self._generate_model_source(self.s3_upload_path, False)
            create_optimization_job_args["ModelSource"] = model_source

            (
                optimization_config,
                quantization_override_env,
                compilation_override_env,
                sharding_override_env,
            ) = self._extract_optimization_config_and_env(
                quantization_config, compilation_config, sharding_config
            )
            create_optimization_job_args["OptimizationConfigs"] = [
                {k: v} for k, v in optimization_config.items()
            ]
            self.env_vars.update(
                {
                    **(quantization_override_env or {}),
                    **(compilation_override_env or {}),
                    **(sharding_override_env or {}),
                }
            )

            output_config = {"S3OutputLocation": output_path}
            if kms_key:
                output_config["KmsKeyId"] = kms_key
            create_optimization_job_args["OutputConfig"] = output_config

            if max_runtime_in_sec:
                create_optimization_job_args["StoppingCondition"] = {
                    "MaxRuntimeInSeconds": max_runtime_in_sec
                }
            if tags:
                create_optimization_job_args["Tags"] = tags
            if vpc_config:
                create_optimization_job_args["VpcConfig"] = vpc_config

            if "HF_MODEL_ID" in self.env_vars:
                del self.env_vars["HF_MODEL_ID"]

            return create_optimization_job_args
        return None

    def _optimize_prepare_for_hf(self):
        """Prepare huggingface model data for optimization."""
        custom_model_path: str = (
            self.model_metadata.get("CUSTOM_MODEL_PATH") if self.model_metadata else None
        )
        if self._is_s3_uri(custom_model_path):
            custom_model_path = (
                custom_model_path[:-1] if custom_model_path.endswith("/") else custom_model_path
            )
        else:
            if not custom_model_path:
                custom_model_path = f"/tmp/sagemaker/model-builder/{self.model}"
                self.download_huggingface_model_metadata(
                    self.model,
                    os.path.join(custom_model_path, "code"),
                    self.env_vars.get("HUGGING_FACE_HUB_TOKEN"),
                )

        self.s3_upload_path, env = self._prepare_for_mode(
            model_path=custom_model_path,
            should_upload_artifacts=True,
        )
        self.env_vars.update(env)


    def _is_gated_model(self) -> bool:
        """Determine if ``this`` Model is Gated

        Args:
            model (Model): Jumpstart Model
        Returns:
            bool: ``True`` if ``this`` Model is Gated
        """
        s3_uri = self.s3_upload_path
        if isinstance(s3_uri, dict):
            s3_uri = s3_uri.get("S3DataSource").get("S3Uri")

        if s3_uri is None:
            return False
        return "private" in s3_uri
    
    def set_js_deployment_config(self, config_name: str, instance_type: str) -> None:
        """Sets the deployment config to apply to the model.

        Args:
            config_name (str):
                The name of the deployment config to apply to the model.
                Call list_deployment_configs to see the list of config names.
            instance_type (str):
                The instance_type that the model will use after setting
                the config.
        """
        self.set_deployment_config(config_name, instance_type)
        self.deployment_config_name = config_name

        self.instance_type = instance_type

        if self.additional_model_data_sources:
            self.speculative_decoding_draft_model_source = "sagemaker"
            self.add_tags(
                {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "sagemaker"},
            )
            self.remove_tag_with_key(Tag.OPTIMIZATION_JOB_NAME)
            self.remove_tag_with_key(Tag.FINE_TUNING_MODEL_PATH)
            self.remove_tag_with_key(Tag.FINE_TUNING_JOB_NAME)


    def _set_additional_model_source(
        self, speculative_decoding_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set Additional Model Source to ``this`` model.

        Args:
            speculative_decoding_config (Optional[Dict[str, Any]]): Speculative decoding config.
            accept_eula (Optional[bool]): For models that require a Model Access Config.
        """
        if speculative_decoding_config:
            model_provider = self._extract_speculative_draft_model_provider(speculative_decoding_config)

            channel_name = self._generate_channel_name(self.additional_model_data_sources)

            if model_provider in ["sagemaker", "auto"]:
                additional_model_data_sources = (
                    self._deployment_config.get("DeploymentArgs", {}).get(
                        "AdditionalDataSources"
                    )
                    if self._deployment_config
                    else None
                )
                if additional_model_data_sources is None:
                    deployment_config = self._find_compatible_deployment_config(
                        speculative_decoding_config
                    )
                    if deployment_config:
                        if model_provider == "sagemaker" and self._is_draft_model_jumpstart_provided(
                            deployment_config
                        ):
                            raise ValueError(
                                "No `Sagemaker` provided draft model was found for "
                                f"{self.model}. Try setting `ModelProvider` "
                                "to `Auto` instead."
                            )

                        try:
                            self.set_js_deployment_config(
                                config_name=deployment_config.get("DeploymentConfigName"),
                                instance_type=deployment_config.get("InstanceType"),
                            )
                        except ValueError as e:
                            raise ValueError(
                                f"{e} If using speculative_decoding_config, "
                                "accept the EULA by setting `AcceptEula`=True."
                            )
                    else:
                        raise ValueError(
                            "Cannot find deployment config compatible for optimization job."
                        )
                else:
                    if model_provider == "sagemaker" and self._is_draft_model_jumpstart_provided(
                        self._deployment_config
                    ):
                        raise ValueError(
                            "No `Sagemaker` provided draft model was found for "
                            f"{self.model}. Try setting `ModelProvider` "
                            "to `Auto` instead."
                        )

                self.env_vars.update(
                    {"OPTION_SPECULATIVE_DRAFT_MODEL": f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}/"}
                )
                self.add_tags(
                    {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": model_provider},
                )
            elif model_provider == "jumpstart":
                self._jumpstart_speculative_decoding(
                    speculative_decoding_config=speculative_decoding_config,
                    sagemaker_session=self.sagemaker_session,
                )
            else:
                self._custom_speculative_decoding(
                    speculative_decoding_config,
                    speculative_decoding_config.get("AcceptEula", False),
                )

    def _find_compatible_deployment_config(
        self, speculative_decoding_config: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Finds compatible model deployment config for optimization job.

        Args:
            speculative_decoding_config (Optional[Dict]): Speculative decoding config.

        Returns:
            Optional[Dict[str, Any]]: A compatible model deployment config for optimization job.
        """
        self._ensure_metadata_configs()
        model_provider = self._extract_speculative_draft_model_provider(speculative_decoding_config)
        for deployment_config in self.list_deployment_configs():
            image_uri = deployment_config.get("deployment_config", {}).get("ImageUri")

            if self._is_image_compatible_with_optimization_job(
                image_uri
            ) and self._deployment_config_contains_draft_model(deployment_config):
                if (
                    model_provider in ["sagemaker", "auto"]
                    and deployment_config.get("DeploymentArgs", {}).get("AdditionalDataSources")
                ) or model_provider == "custom":
                    return deployment_config

        if model_provider in ["sagemaker", "auto"]:
            return None

        return self._deployment_config

    def _get_neuron_model_env_vars(
        self, instance_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Gets Neuron model env vars.

        Args:
            instance_type (Optional[str]): Instance type.

        Returns:
            Optional[Dict[str, Any]]: Neuron Model environment variables.
        """
        metadata_configs = self._metadata_configs
        if metadata_configs:
            metadata_config = metadata_configs.get(self.config_name)
            resolve_config = metadata_config.resolved_config if metadata_config else None
            if resolve_config and instance_type not in resolve_config.get(
                "supported_inference_instance_types", []
            ):
                neuro_model_id = resolve_config.get("hosting_neuron_model_id")
                neuro_model_version = resolve_config.get("hosting_neuron_model_version", "*")
                if neuro_model_id:
                    model_specs = self._get_cached_model_specs(
                        model_id=neuro_model_id,
                        version=neuro_model_version,
                        region=self.region,
                        sagemaker_session=self.sagemaker_session,

                    )
                    
                    model_spec_json = model_specs.to_json()
                    return model_spec_json.get("hosting_env_vars", {})
                    
        return None


    def _set_optimization_image_default(
        self, create_optimization_job_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Defaults the optimization image to the JumpStart deployment config default

        Args:
            create_optimization_job_args (Dict[str, Any]): create optimization job request

        Returns:
            Dict[str, Any]: create optimization job request with image uri default
        """
        init_kwargs = get_init_kwargs(
            config_name=self.config_name,
            model_id=self.model,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
            image_uri=self.image_uri,
            region=self.region,
            model_version=self.model_version,
            hub_arn=self.hub_arn,
            tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
            tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
        )
        default_image = self._get_default_vllm_image(init_kwargs.image_uri)

        for optimization_config in create_optimization_job_args.get("OptimizationConfigs"):
            if optimization_config.get("ModelQuantizationConfig"):
                model_quantization_config = optimization_config.get("ModelQuantizationConfig")
                provided_image = model_quantization_config.get("Image")
                if provided_image and self._get_latest_lmi_version_from_list(
                    default_image, provided_image
                ):
                    default_image = provided_image
            if optimization_config.get("ModelShardingConfig"):
                model_sharding_config = optimization_config.get("ModelShardingConfig")
                provided_image = model_sharding_config.get("Image")
                if provided_image and self._get_latest_lmi_version_from_list(
                    default_image, provided_image
                ):
                    default_image = provided_image
        for optimization_config in create_optimization_job_args.get("OptimizationConfigs"):
            if optimization_config.get("ModelQuantizationConfig") is not None:
                optimization_config.get("ModelQuantizationConfig")["Image"] = default_image
            if optimization_config.get("ModelShardingConfig") is not None:
                optimization_config.get("ModelShardingConfig")["Image"] = default_image

        logger.debug(f"Defaulting to {default_image} image for optimization job")

        return create_optimization_job_args

    def _get_default_vllm_image(self, image: str) -> bool:
        """Ensures the minimum working image version for vLLM enabled optimization techniques

        Args:
            image (str): JumpStart provided default image

        Returns:
            str: minimum working image version
        """
        dlc_name, _ = image.split(":")
        major_version_number, _, _ = self._parse_lmi_version(image)

        if major_version_number < self._parse_lmi_version(_JS_MINIMUM_VERSION_IMAGE)[0]:
            minimum_version_default = _JS_MINIMUM_VERSION_IMAGE.format(dlc_name)
            return minimum_version_default
        return image

    def _get_latest_lmi_version_from_list(self, version: str, version_to_compare: str) -> bool:
        """LMI version comparator

        Args:
            version (str): current version
            version_to_compare (str): version to compare to

        Returns:
            bool: if version_to_compare larger or equal to version
        """
        parse_lmi_version = self._parse_lmi_version(version)
        parse_lmi_version_to_compare = self._parse_lmi_version(version_to_compare)

        if parse_lmi_version_to_compare[0] > parse_lmi_version[0]:
            return True
        if parse_lmi_version_to_compare[0] == parse_lmi_version[0]:
            if parse_lmi_version_to_compare[1] > parse_lmi_version[1]:
                return True
            if parse_lmi_version_to_compare[1] == parse_lmi_version[1]:
                if parse_lmi_version_to_compare[2] >= parse_lmi_version[2]:
                    return True
                return False
            return False
        return False

    def _parse_lmi_version(self, image: str) -> Tuple[int, int, int]:
        """Parse out LMI version

        Args:
            image (str): image to parse version out of

        Returns:
            Tuple[int, int, int]: LMI version split into major, minor, patch
            
        Raises:
            ValueError: If the image format cannot be parsed
        """
        _, dlc_tag = image.split(":")
        parts = dlc_tag.split("-")
        
        lmi_version = None
        for part in parts:
            if "." in part and part[0].isdigit():
                lmi_version = part
                break
        
        if not lmi_version:
            raise ValueError(f"Could not find version in image: {image}")
            
        version_parts = lmi_version.split(".")
        if len(version_parts) < 3:
            raise ValueError(f"Invalid version format: {lmi_version} in image: {image}")
            
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1])
        patch_version = int(version_parts[2])
        
        return (major_version, minor_version, patch_version)



    def _optimize_for_jumpstart(
        self,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
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
        max_runtime_in_sec: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Runs a model optimization job.

        Args:
            output_path (Optional[str]): Specifies where to store the compiled/quantized model.
            instance_type (str): Target deployment instance type that the model is optimized for.
            tags (Optional[Tags]): Tags for labeling a model optimization job. Defaults to ``None``.
            job_name (Optional[str]): The name of the model optimization job. Defaults to ``None``.
            accept_eula (bool): For models that require a Model Access Config, specify True or
                False to indicate whether model terms of use have been accepted.
                The `accept_eula` value must be explicitly defined as `True` in order to
                accept the end-user license agreement (EULA) that some
                models require. (Default: None).
            quantization_config (Optional[Dict]): Quantization configuration. Defaults to ``None``.
            compilation_config (Optional[Dict]): Compilation configuration. Defaults to ``None``.
            speculative_decoding_config (Optional[Dict]): Speculative decoding configuration.
                Defaults to ``None``
            sharding_config (Optional[Dict]): Model sharding configuration.
                Defaults to ``None``
            env_vars (Optional[Dict]): Additional environment variables to run the optimization
                container. Defaults to ``None``.
            vpc_config (Optional[Dict]): The VpcConfig set on the model. Defaults to ``None``.
            kms_key (Optional[str]): KMS key ARN used to encrypt the model artifacts when uploading
                to S3. Defaults to ``None``.
            max_runtime_in_sec (Optional[int]): Maximum job execution time in seconds. Defaults to
                ``None``.

        Returns:
            Dict[str, Any]: Model optimization job input arguments.
        """
        if self._is_gated_model() and accept_eula is not True:
            raise ValueError(
                f"Model '{self.model}' requires accepting end-user license agreement (EULA)."
            )

        is_compilation = (compilation_config is not None) or self._is_inferentia_or_trainium(
            instance_type
        )

        env_vars = dict()
        if is_compilation:
            env_vars = self._get_neuron_model_env_vars(instance_type)
        (
            optimization_config,
            quantization_override_env,
            compilation_override_env,
            sharding_override_env,
        ) = self._extract_optimization_config_and_env(
            quantization_config, compilation_config, sharding_config
        )

        if not optimization_config:
            optimization_config = {}

        if not optimization_config.get("ModelCompilationConfig") and is_compilation:
            if not compilation_override_env:
                compilation_override_env = env_vars

            override_compilation_config = (
                {"OverrideEnvironment": compilation_override_env}
                if compilation_override_env
                else {}
            )
            optimization_config["ModelCompilationConfig"] = override_compilation_config

        if speculative_decoding_config:
            self._set_additional_model_source(speculative_decoding_config)
        else:
            deployment_config = self._find_compatible_deployment_config(None)
            if deployment_config:
                self.set_js_deployment_config(
                    config_name=deployment_config.get("DeploymentConfigName"),
                    instance_type=deployment_config.get("InstanceType"),
                )
                env_vars = self.env_vars

        model_source = self._generate_model_source(self.s3_upload_path, accept_eula)
        optimization_env_vars = self._update_environment_variables(env_vars, env_vars)

        output_config = {"S3OutputLocation": output_path}
        if kms_key:
            output_config["KmsKeyId"] = kms_key

        deployment_config_instance_type = (
            self._deployment_config.get("DeploymentArgs", {}).get("InstanceType")
            if self._deployment_config
            else None
        )
        self.instance_type = instance_type or deployment_config_instance_type or self._get_nb_instance()

        create_optimization_job_args = {
            "OptimizationJobName": job_name,
            "ModelSource": model_source,
            "DeploymentInstanceType": self.instance_type,
            "OptimizationConfigs": [{k: v} for k, v in optimization_config.items()],
            "OutputConfig": output_config,
            "RoleArn": self.role_arn,
        }

        if optimization_env_vars:
            create_optimization_job_args["OptimizationEnvironment"] = optimization_env_vars
        if max_runtime_in_sec:
            create_optimization_job_args["StoppingCondition"] = {
                "MaxRuntimeInSeconds": max_runtime_in_sec
            }
        if tags:
            create_optimization_job_args["Tags"] = tags
        if vpc_config:
            create_optimization_job_args["VpcConfig"] = vpc_config

        if accept_eula:
            self.accept_eula = accept_eula
            if isinstance(self.s3_upload_path, dict):
                self.s3_upload_path["S3DataSource"]["ModelAccessConfig"] = {
                    "AcceptEula": True
                }

        optimization_env_vars = self._update_environment_variables(
            optimization_env_vars,
            {
                **(quantization_override_env or {}),
                **(compilation_override_env or {}),
                **(sharding_override_env or {}),
            },
        )
        if optimization_env_vars:
            self.env_vars.update(optimization_env_vars)

        if sharding_config and self._enable_network_isolation:
            logger.warning(
                "EnableNetworkIsolation cannot be set to True since SageMaker Fast Model "
                "Loading of model requires network access. Setting it to False."
            )
            self._enable_network_isolation = False

        if quantization_config or sharding_config or is_compilation:
            return (
                create_optimization_job_args
                if is_compilation
                else self._set_optimization_image_default(create_optimization_job_args)
            )
        return None


    def _generate_optimized_core_model(self, optimization_response: dict) -> Model:
        """Generate optimized CoreModel from optimization job response."""
        
        recommended_image_uri = optimization_response.get("OptimizationOutput", {}).get("RecommendedInferenceImage")
        s3_uri = optimization_response.get("OutputConfig", {}).get("S3OutputLocation")
        deployment_instance_type = optimization_response.get("DeploymentInstanceType")
        if recommended_image_uri:
            self.image_uri = recommended_image_uri
        if s3_uri:
            if isinstance(self.s3_upload_path, dict):
                self.s3_upload_path["S3DataSource"]["S3Uri"] = s3_uri
            else:
                self.s3_upload_path = s3_uri
        if deployment_instance_type:
            self.instance_type = deployment_instance_type

        self.add_tags({"Key": "OptimizationJobName", "Value": optimization_response["OptimizationJobName"]})
        
        self._optimizing = False
        optimized_core_model = self._create_model()
        self.built_model = optimized_core_model
        
        return optimized_core_model



    def deployment_config_response_data(
        self,
        deployment_configs: Optional[List[DeploymentConfigMetadata]],
    ) -> List[Dict[str, Any]]:
        """Deployment config api response data.

        Args:
            deployment_configs (Optional[List[DeploymentConfigMetadata]]):
            List of deployment configs metadata.
        Returns:
            List[Dict[str, Any]]: List of deployment config api response data.
        """
        configs = []
        if not deployment_configs:
            return configs

        for deployment_config in deployment_configs:
            deployment_config_json = deployment_config.to_json()
            benchmark_metrics = deployment_config_json.get("BenchmarkMetrics")
            if benchmark_metrics and deployment_config.deployment_args:
                deployment_config_json["BenchmarkMetrics"] = {
                    deployment_config.deployment_args.instance_type: benchmark_metrics.get(
                        deployment_config.deployment_args.instance_type
                    )
                }

            configs.append(deployment_config_json)
        return configs
    
    # @_deployment_config_lru_cache
    def _get_deployment_configs_benchmarks_data(self) -> Dict[str, Any]:
        """Deployment configs benchmark metrics.

        Returns:
            Dict[str, List[str]]: Deployment config benchmark data.
        """
        return get_metrics_from_deployment_configs(
            self._get_deployment_configs(None, None),
        )

    # @_deployment_config_lru_cache
    def _get_deployment_configs(
        self, selected_config_name: Optional[str], selected_instance_type: Optional[str]
    ) -> List[DeploymentConfigMetadata]:
        """Retrieve deployment configs metadata.

        Args:
            selected_config_name (Optional[str]): The name of the selected deployment config.
            selected_instance_type (Optional[str]): The selected instance type.
        """
        deployment_configs = []
        if not self._metadata_configs:
            return deployment_configs

        err = None
        for config_name, metadata_config in self._metadata_configs.items():
            if selected_config_name == config_name:
                instance_type_to_use = selected_instance_type
            else:
                instance_type_to_use = metadata_config.resolved_config.get(
                    "default_inference_instance_type"
                )

            if metadata_config.benchmark_metrics:
                (
                    err,
                    metadata_config.benchmark_metrics,
                ) = add_instance_rate_stats_to_benchmark_metrics(
                    self.region, metadata_config.benchmark_metrics
                )

            config_components = metadata_config.config_components.get(config_name)
            image_uri = (
                (
                    config_components.hosting_instance_type_variants.get("regional_aliases", {})
                    .get(self.region, {})
                    .get("alias_ecr_uri_1")
                )
                if config_components
                else self.image_uri
            )

            init_kwargs = get_init_kwargs(
                config_name=config_name,
                model_id=self.model,
                instance_type=instance_type_to_use,
                sagemaker_session=self.sagemaker_session,
                image_uri=image_uri,
                region=self.region,
                model_version=getattr(self, 'model_version', None) or "*",
                hub_arn=self.hub_arn,
                tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
                tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
            )
            deploy_kwargs = get_deploy_kwargs(
                model_id=self.model,
                instance_type=instance_type_to_use,
                sagemaker_session=self.sagemaker_session,
                region=self.region,
                model_version=getattr(self, 'model_version', None) or "*",
                hub_arn=self.hub_arn,
                tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
                tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
            )

            deployment_config_metadata = DeploymentConfigMetadata(
                config_name,
                metadata_config,
                init_kwargs,
                deploy_kwargs,
            )
            deployment_configs.append(deployment_config_metadata)

        if err and err["Code"] == "AccessDeniedException":
            error_message = "Instance rate metrics will be omitted. Reason: %s"
            logger.warning(error_message, err["Message"])

        return deployment_configs



    # ========================================
    # General Utils
    # ========================================

    def add_tags(self, tags: Tags) -> None:
        """Add tags to this model.

        Args:
            tags: Tags to add to the model.
        """
        current_tags = getattr(self, '_tags', None)
        self._tags = _validate_new_tags(tags, current_tags)

    def remove_tag_with_key(self, key: str) -> None:
        """Remove a tag with the given key from the list of tags.

        Args:
            key: The key of the tag to remove.
        """
        current_tags = getattr(self, '_tags', None)
        self._tags = remove_tag_with_key(key, current_tags)

    def _get_model_uri(self) -> Optional[str]:
        """Extract model URI from s3_model_data_url.
        
        Returns:
            Model URI string, or None if not available.
        """
        s3_model_data_url = getattr(self, 's3_model_data_url', None)
        if not s3_model_data_url:
            return None
            
        if isinstance(s3_model_data_url, (str, PipelineVariable)):
            return s3_model_data_url
        elif isinstance(s3_model_data_url, dict):
            return s3_model_data_url.get("S3DataSource", {}).get("S3Uri", None)
        return None

    def _ensure_base_name_if_needed(self, image_uri: str, script_uri: Optional[str], model_uri: Optional[str]) -> None:
        """Create base name from image URI if no model name provided.

        Uses JumpStart base name if available, otherwise derives from image URI.
        
        Args:
            image_uri: Container image URI
            script_uri: Optional script URI for JumpStart models
            model_uri: Optional model URI for JumpStart models
        """
        model_name = getattr(self, 'model_name', None)
        if model_name is None:
            base_name = getattr(self, '_base_name', None)
            self._base_name = (
                base_name
                or get_jumpstart_base_name_if_jumpstart_model(script_uri, model_uri)
                or base_name_from_image(image_uri, default_base_name="ModelBuilder")
            )
    

    def _ensure_metadata_configs(self) -> None:
        """Lazy load JumpStart metadata configs when needed."""
        metadata_configs = getattr(self, '_metadata_configs', None)
        model = getattr(self, 'model', None)
        
        if metadata_configs is None and isinstance(model, str):
            from sagemaker.core.jumpstart.utils import get_jumpstart_configs
            
            self._metadata_configs = get_jumpstart_configs(
                region=self.region,
                model_id=model,
                model_version=getattr(self, 'model_version', None) or "*",
                sagemaker_session=getattr(self, 'sagemaker_session', None),
            )
            
    def _user_agent_decorator(self, func):
        """Decorator to add ModelBuilder to user agent string.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function that appends ModelBuilder to user agent.
        """
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            if "ModelBuilder" in result:
                return result
            return result + " ModelBuilder"
        return wrapper

    def _get_serve_setting(self) -> _ServeSettings:
        """Get serve settings for model deployment.
        
        Creates or uses existing S3 model data URL and constructs
        serve settings with deployment configuration.
        
        Returns:
            ServeSettings object with deployment configuration.
        """
        s3_model_data_url = getattr(self, 's3_model_data_url', None)
        if not s3_model_data_url:
            sagemaker_session = getattr(self, 'sagemaker_session', None)
            if sagemaker_session:
                bucket = sagemaker_session.default_bucket()
                model_name = getattr(self, 'model_name', None)
                prefix = f"model-builder/{model_name or 'model'}/{uuid.uuid4().hex}"
                self.s3_model_data_url = f"s3://{bucket}/{prefix}/"
 
        return _ServeSettings(
            role_arn=getattr(self, 'role_arn', None),
            s3_model_data_url=getattr(self, 's3_model_data_url', None),
            instance_type=getattr(self, 'instance_type', None),
            env_vars=getattr(self, 'env_vars', None),
            sagemaker_session=getattr(self, 'sagemaker_session', None),
        )


    def _is_jumpstart_model_id(self) -> bool:
        """Check if model is a JumpStart model ID."""
        if not hasattr(self, '_cached_is_jumpstart'):
            if self.model is None:
                self._cached_is_jumpstart = False
                return self._cached_is_jumpstart

            try:
                model_uris.retrieve(model_id=self.model, model_version="*", model_scope=_JS_SCOPE)
            except KeyError:
                logger.debug(_NO_JS_MODEL_EX)
                self._cached_is_jumpstart = False
                return self._cached_is_jumpstart

            logger.debug("JumpStart Model ID detected.")
            self._cached_is_jumpstart = True
            return self._cached_is_jumpstart

        return self._cached_is_jumpstart
    

    def _has_nvidia_gpu(self) -> bool:
        try:
            _get_available_gpus()
            return True
        except Exception:
            # for nvidia-smi to run, a cuda driver must be present
            logger.debug("CUDA not found, launching Triton in CPU mode.")
            return False
    
    def _is_gpu_instance(self, instance_type: str) -> bool:
        instance_family = instance_type.rsplit(".", 1)[0]
        return instance_family in GPU_INSTANCE_FAMILIES

    def _save_inference_spec(self) -> None:
        """Save inference specification to pickle file."""
        if self.inference_spec:
            pkl_path = Path(self.model_path).joinpath("model_repository").joinpath("model")
            save_pkl(pkl_path, (self.inference_spec, self.schema_builder))
    
    def _hmac_signing(self):
        """Perform HMAC signing on picke file for integrity check"""
        secret_key = generate_secret_key()
        pkl_path = Path(self.model_path).joinpath("model_repository").joinpath("model")

        with open(str(pkl_path.joinpath("serve.pkl")), "rb") as f:
            buffer = f.read()
        hash_value = compute_hash(buffer=buffer, secret_key=secret_key)

        with open(str(pkl_path.joinpath("metadata.json")), "wb") as metadata:
            metadata.write(_MetaData(hash_value).to_json())

        self.secret_key = secret_key

    def _generate_config_pbtxt(self, pkl_path: Path):
        """Generate Triton config.pbtxt file."""
        config_path = pkl_path.joinpath("config.pbtxt")

        input_shape = list(self.schema_builder._sample_input_ndarray.shape)
        output_shape = list(self.schema_builder._sample_output_ndarray.shape)
        input_shape[0] = -1
        output_shape[0] = -1

        config_content = CONFIG_TEMPLATE.format(
            input_name=INPUT_NAME,
            input_shape=str(input_shape),
            input_dtype=self.schema_builder._input_triton_dtype,
            output_name=OUTPUT_NAME,
            output_dtype=self.schema_builder._output_triton_dtype,
            output_shape=str(output_shape),
            hardware_type="KIND_CPU" if "-cpu" in self.image_uri else "KIND_GPU",
        )

        with open(str(config_path), "w") as f:
            f.write(config_content)

    def _pack_conda_env(self, pkl_path: Path):
        """Pack conda environment for Triton deployment."""
        try:
            import conda_pack
            conda_pack.__version__
        except ModuleNotFoundError:
            raise ImportError(
                "Launching Triton with ModelBuilder requires conda_pack library "
                "but it was not found in your environment. "
                "Checkout the instructions on the installation page of its repo: "
                "https://conda.github.io/conda-pack/ "
                "And follow the ones that match your environment. "
                "Please note that you may need to restart your runtime after installation."
            )

        script_path = Path(__file__).parent.joinpath("pack_conda_env.sh")
        env_tar_path = pkl_path.joinpath("triton_env.tar.gz")
        conda_env_name = os.getenv("CONDA_DEFAULT_ENV")

        subprocess.run(["bash", str(script_path), conda_env_name, str(env_tar_path)])

    def _export_tf_to_onnx(self, export_path: str, model: object, schema_builder: SchemaBuilder):
        try:
            import tensorflow as tf
            import tf2onnx

            tf2onnx.convert.from_keras(
                model=model,
                input_signature=[
                    tf.TensorSpec(shape=schema_builder.sample_input.shape, name=INPUT_NAME)
                ],
                output_path=str(export_path.joinpath("model.onnx")),
            )

        except ModuleNotFoundError:
            raise ImportError(
                "Launching Triton with ModelBuilder for a Tensorflow model requires tf2onnx module "
                "but it was not found in your environment. "
                "Checkout the instructions on the installation page of its repo: "
                "https://onnxruntime.ai/docs/install/ "
                "And follow the ones that match your environment. "
                "Please note that you may need to restart your runtime after installation."
            )

    def _export_pytorch_to_onnx(
        self, model: object, export_path: Path, schema_builder: SchemaBuilder
    ):
        """Export PyTorch model object into ONNX format."""
        logger.debug("Converting PyTorch model into ONNX format")
        try:
            from torch.onnx import export

            export(
                model=model,
                args=schema_builder.sample_input,
                f=str(export_path.joinpath("model.onnx")),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                opset_version=17,
                verbose=False,
            )

        except ModuleNotFoundError:
            raise ImportError(
                "Launching Triton with ModelBuilder for a PyTorch model requires onnx module "
                "but it was not found in your environment. "
                "Checkout the instructions on the installation page of its repo: "
                "https://onnxruntime.ai/docs/install/ "
                "And follow the ones that match your environment. "
                "Please note that you may need to restart your runtime after installation."
            )
        
    def _validate_for_triton(self):
        """Validation for Triton deployment."""
        try:
            import tritonclient.http as httpClient
            httpClient.__class__
        except ModuleNotFoundError:
            raise ImportError(
                "Launching Triton with ModelBuilder requires tritonClient[http] module "
                "but it was not found in your environment. "
                "Checkout the instructions on the installation page of its repo: "
                "https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples "
                "And follow the ones that match your environment. "
                "Please note that you may need to restart your runtime after installation."
            )

        if (
            self.mode == Mode.LOCAL_CONTAINER
            and not self._has_nvidia_gpu()
            and self.image_uri
            and "cpu" not in self.image_uri
        ):
            raise ValueError(
                "Your device does not have a Nvidia GPU. "
                "Unable to launch Triton container in GPU mode in your local machine. "
                "Please provide a CPU version triton image to serve your model in LOCAL_CONTAINER mode."
            )

        if self.mode not in SUPPORTED_TRITON_MODE:
            raise ValueError("%s mode is not supported with Triton model server." % self.mode)

        model_path = Path(self.model_path)
        if not model_path.exists():
            model_path.mkdir(parents=True)
        elif not model_path.is_dir():
            raise Exception(f"model_path: {self.model_path} is not a valid directory")

        self.schema_builder._update_serializer_deserializer_for_triton()
        self.schema_builder._detect_dtype_for_triton()
        if not platform.python_version().startswith("3.8"):
            logger.warning(
                f"SageMaker Triton image uses python 3.8, your python version: {platform.python_version()}. "
                "It is recommended to use the same python version to avoid incompatibility."
            )

        if self.model:
            fw, self.framework_version = _detect_framework_and_version(
                str(_get_model_base(self.model))
            )
            if fw == "pytorch":
                self.framework = Framework.PYTORCH
            elif fw == "tensorflow":
                self.framework = Framework.TENSORFLOW

            if self.framework not in SUPPORTED_TRITON_FRAMEWORK:
                raise ValueError("%s is not supported with Triton model server" % self.framework)

        if self.inference_spec:
            if "conda" not in sys.executable.lower():
                raise ValueError(
                    f"Invalid python environment {sys.executable}, please use anaconda "
                    "or miniconda to manage your python environment "
                    "as it is required by Triton to capture "
                    "and pack your python dependencies."
                )

    def _prepare_for_triton(self):
        """Prepare model artifacts for Triton deployment."""
        model_path = Path(self.model_path)
        pkl_path = model_path.joinpath("model_repository").joinpath("model")
        if not pkl_path.exists():
            pkl_path.mkdir(parents=True)

        for root, _, files in os.walk(self.model_path):
            for f in files:
                path_file = os.path.join(root, f)
                if "model_repository" not in path_file:
                    shutil.copy2(path_file, str(pkl_path.joinpath(f)))

        export_path = model_path.joinpath("model_repository").joinpath("model").joinpath("1")
        if not export_path.exists():
            export_path.mkdir(parents=True)

        if self.model:
            self.secret_key = "dummy secret key for onnx backend"

            if self.framework == Framework.PYTORCH:
                self._export_pytorch_to_onnx(
                    export_path=export_path, model=self.model, schema_builder=self.schema_builder
                )
                return

            if self.framework == Framework.TENSORFLOW:
                self._export_tf_to_onnx(
                    export_path=export_path, model=self.model, schema_builder=self.schema_builder
                )
                return

            raise ValueError("%s is not supported" % self.framework)

        if self.inference_spec:
            triton_model_path = Path(__file__).parent.joinpath("model.py")
            shutil.copy2(str(triton_model_path), str(export_path))

            self._generate_config_pbtxt(pkl_path=pkl_path)

            self._pack_conda_env(pkl_path=pkl_path)

            self._hmac_signing()

            return

        raise ValueError("Either model or inference_spec should be provided to ModelBuilder.")
    

    def _auto_detect_image_for_triton(self):
        """Detect image of triton given framework, version and region.

        If InferenceSpec is provided, then default to latest version.
        """
        if self.image_uri:
            logger.debug("Skipping auto detection as the image uri is provided %s", self.image_uri)
            return

        logger.debug(
            "Auto detect container url for the provided model and on instance %s",
            self.instance_type,
        )

        region = self.sagemaker_session.boto_region_name

        if region not in TRITON_IMAGE_ACCOUNT_ID_MAP.keys():
            raise ValueError(
                f"{region} is not supported for triton image. "
                f"Please switch to the following region: {list(TRITON_IMAGE_ACCOUNT_ID_MAP.keys())}"
            )

        base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"

        if (
            not self.inference_spec
            and self.framework == "tensorflow"
            and self.version.startswith("1")
        ):
            self.image_uri = TRITON_IMAGE_BASE.format(
                account_id=TRITON_IMAGE_ACCOUNT_ID_MAP.get(region),
                region=region,
                base=base,
                version=VERSION_FOR_TF1,
            )
        else:
            self.image_uri = TRITON_IMAGE_BASE.format(
                account_id=TRITON_IMAGE_ACCOUNT_ID_MAP.get(region),
                region=region,
                base=base,
                version=LATEST_VERSION,
            )

        if not self._is_gpu_instance(self.instance_type):
            self.image_uri += "-cpu"

        logger.debug(f"Autodetected image: {self.image_uri}. Proceeding with the deployment.")
    

    def _validate_djl_serving_sample_data(self):
        """Validate sample data format for DJL serving."""
        sample_input = self.schema_builder.sample_input
        sample_output = self.schema_builder.sample_output

        if (
            not isinstance(sample_input, dict)
            or "inputs" not in sample_input
            or "parameters" not in sample_input
            or not isinstance(sample_output, list)
            or not isinstance(sample_output[0], dict)
            or "generated_text" not in sample_output[0]
        ):
            raise ValueError(_INVALID_DJL_SAMPLE_DATA_EX)
    
    def _validate_tgi_serving_sample_data(self):
        """Validate sample data format for TGI serving."""
        sample_input = self.schema_builder.sample_input
        sample_output = self.schema_builder.sample_output

        if (
            not isinstance(sample_input, dict)
            or "inputs" not in sample_input
            or "parameters" not in sample_input
            or not isinstance(sample_output, list)
            or not isinstance(sample_output[0], dict)
            or "generated_text" not in sample_output[0]
        ):
            raise ValueError(_INVALID_TGI_SAMPLE_DATA_EX)
        
    def _create_conda_env(self):
        """Create conda environment by running commands."""
        try:
            RequirementsManager().capture_and_install_dependencies
        except subprocess.CalledProcessError:
            logger.error("Failed to create and activate conda environment.")

    
    def _extract_framework_from_model_trainer(self, model_trainer: ModelTrainer) -> Optional[Framework]:
        """Extract framework from ModelTrainer training image."""
        training_image = model_trainer.training_image
        if not training_image:
            training_image = model_trainer._latest_training_job.algorithm_specification.training_image
        
        if "pytorch" in training_image.lower():
            return Framework.PYTORCH
        elif "tensorflow" in training_image.lower():
            return Framework.TENSORFLOW
        elif "huggingface" in training_image.lower():
            return Framework.HUGGINGFACE
        elif "xgboost" in training_image.lower():
            return Framework.XGBOOST
        
        return None


    def _infer_model_server_from_training(self, model_trainer: ModelTrainer) -> Optional[ModelServer]:
        """Infer the best model server based on training configuration."""
        training_image = model_trainer.training_image
        framework = self._extract_framework_from_model_trainer(model_trainer)
        
        if "huggingface" in training_image.lower():
            hyperparams = model_trainer.hyperparameters or {}
            if any(key in hyperparams for key in ["max_new_tokens", "do_sample", "temperature"]):
                logger.info("Auto-detected model server: TGI (HuggingFace text generation)")
                return ModelServer.TGI
            else:
                logger.info("Auto-detected model server: MMS (HuggingFace)")
                return ModelServer.MMS
        
        if framework == Framework.PYTORCH:
            logger.info("Auto-detected model server: TORCHSERVE (PyTorch framework)")
            return ModelServer.TORCHSERVE
        
        if framework == Framework.TENSORFLOW:
            logger.info("Auto-detected model server: TENSORFLOW_SERVING (TensorFlow framework)")
            return ModelServer.TENSORFLOW_SERVING
        
        logger.warning(
            f"Could not auto-detect model server for framework: {framework}. "
            "Defaulting to TORCHSERVE. Consider explicitly setting model_server parameter."
        )
        return ModelServer.TORCHSERVE
    

    def _extract_inference_spec_from_training_code(self, model_trainer: ModelTrainer) -> Optional[str]:
        """Check if training source code contains inference.py."""
        if not model_trainer.source_code or not model_trainer.source_code.source_dir:
            return None
        
        source_dir = model_trainer.source_code.source_dir
        
        # Check for inference.py in source directory
        if source_dir.startswith("s3://"):
            pass
        else:
            inference_path = os.path.join(source_dir, "inference.py")
            if os.path.exists(inference_path):
                return inference_path
        
        return None
    

    def _inherit_training_environment(self, model_trainer: ModelTrainer) -> Dict[str, str]:
        """Inherit relevant environment variables from training."""
        from sagemaker.core.utils.utils import Unassigned
        
        training_env = model_trainer.environment or {}
        if isinstance(training_env, Unassigned):
            training_env = {}
            
        training_job_env = model_trainer._latest_training_job.environment
        if isinstance(training_job_env, Unassigned) or training_job_env is None:
            training_job_env = {}
        
        inherited_env = {**training_env, **training_job_env}
        inference_relevant_keys = [
            "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", 
            "MODEL_CLASS_NAME", "TRANSFORMERS_CACHE",
            "PYTORCH_TRANSFORMERS_CACHE", "HF_HOME"
        ]
        
        return {k: v for k, v in inherited_env.items() 
                if k in inference_relevant_keys or k.startswith("SAGEMAKER_")}
    

    def _extract_version_from_training_image(self, training_image: str) -> Optional[str]:
        """Extract framework version from training image URI."""
        import re
        
        version_match = re.search(r':(\d+\.\d+(?:\.\d+)?)', training_image)
        if version_match:
            return version_match.group(1)
        
        return None


    def _detect_inference_image_from_training(self) -> None:
        """Detect inference image based on ModelTrainer's training image."""
        from sagemaker.core import image_uris
        training_image = self.model.training_image
        
        if "pytorch-training" in training_image:
            self.image_uri = training_image.replace("pytorch-training", "pytorch-inference")
        elif "tensorflow-training" in training_image:
            self.image_uri = training_image.replace("tensorflow-training", "tensorflow-inference")
        elif "huggingface-pytorch-training" in training_image:
            self.image_uri = training_image.replace("huggingface-pytorch-training", "huggingface-pytorch-inference")
        else:
            framework = self._extract_framework_from_model_trainer(self.model)
            fw = framework.value.lower() if framework else "pytorch"
            
            fw_version = self._extract_version_from_training_image(training_image)
            py_tuple = platform.python_version_tuple()
            casted_versions = _cast_to_compatible_version(fw, fw_version) if fw_version else (None,)
            dlc = None

            for casted_version in filter(None, casted_versions):
                try:
                    dlc = image_uris.retrieve(
                        framework=fw,
                        region=self.region,
                        version=casted_version,
                        image_scope="inference",
                        py_version=f"py{py_tuple[0]}{py_tuple[1]}",
                        instance_type=self.instance_type,
                    )
                    break
                except ValueError:
                    pass
            
            if dlc:
                self.image_uri = dlc
            else:
                raise ValueError(f"Could not detect inference image for training image: {training_image}")
    

    def _extract_speculative_draft_model_provider(
        self,
        speculative_decoding_config: Optional[Dict] = None,
    ) -> Optional[str]:
        """Extracts speculative draft model provider from speculative decoding config.

        Args:
            speculative_decoding_config (Optional[Dict]): A speculative decoding config.

        Returns:
            Optional[str]: The speculative draft model provider.
        """
        if speculative_decoding_config is None:
            return None

        model_provider = speculative_decoding_config.get("ModelProvider", "").lower()

        if model_provider == "jumpstart":
            return "jumpstart"

        if model_provider == "custom" or speculative_decoding_config.get("ModelSource"):
            return "custom"

        if model_provider == "sagemaker":
            return "sagemaker"

        return "auto"
    

    def get_huggingface_model_metadata(self, model_id: str, hf_hub_token: Optional[str] = None) -> dict:
        """Retrieves the json metadata of the HuggingFace Model via HuggingFace API.

        Args:
            model_id (str): The HuggingFace Model ID
            hf_hub_token (str): The HuggingFace Hub Token needed for Private/Gated HuggingFace Models

        Returns:
            dict: The model metadata retrieved with the HuggingFace API
        """
        import urllib.request
        from urllib.error import HTTPError, URLError
        import json
        from json import JSONDecodeError

        if not model_id:
            raise ValueError("Model ID is empty. Please provide a valid Model ID.")
        hf_model_metadata_url = f"https://huggingface.co/api/models/{model_id}"
        hf_model_metadata_json = None
        try:
            if hf_hub_token:
                hf_model_metadata_url = urllib.request.Request(
                    hf_model_metadata_url, None, {"Authorization": "Bearer " + hf_hub_token}
                )
            with urllib.request.urlopen(hf_model_metadata_url) as response:
                hf_model_metadata_json = json.load(response)
        except (HTTPError, URLError, TimeoutError, JSONDecodeError) as e:
            if "HTTP Error 401: Unauthorized" in str(e):
                raise ValueError(
                    "Trying to access a gated/private HuggingFace model without valid credentials. "
                    "Please provide a HUGGING_FACE_HUB_TOKEN in env_vars"
                )
            logger.warning(
                "Exception encountered while trying to retrieve HuggingFace model metadata %s. "
                "Details: %s",
                hf_model_metadata_url,
                e,
            )
        if not hf_model_metadata_json:
            raise ValueError(
                "Did not find model metadata for the following HuggingFace Model ID %s" % model_id
            )
        return hf_model_metadata_json


    def download_huggingface_model_metadata(
        self, model_id: str, model_local_path: str, hf_hub_token: Optional[str] = None
    ) -> None:
        """Downloads the HuggingFace Model snapshot via HuggingFace API.

        Args:
            model_id (str): The HuggingFace Model ID
            model_local_path (str): The local path to save the HuggingFace Model snapshot.
            hf_hub_token (str): The HuggingFace Hub Token

        Raises:
            ImportError: If huggingface_hub is not installed.
        """
        if not importlib.util.find_spec("huggingface_hub"):
            raise ImportError("Unable to import huggingface_hub, check if huggingface_hub is installed")

        from huggingface_hub import snapshot_download

        os.makedirs(model_local_path, exist_ok=True)
        logger.info("Downloading model %s from Hugging Face Hub to %s", model_id, model_local_path)
        snapshot_download(repo_id=model_id, local_dir=model_local_path, token=hf_hub_token)

