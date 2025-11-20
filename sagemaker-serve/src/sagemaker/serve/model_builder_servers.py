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

This module provides the ModelBuilder class, which offers a unified interface for building
and deploying machine learning models across different model servers and deployment modes.
It supports various frameworks including PyTorch, TensorFlow, HuggingFace, XGBoost, and more.
"""
from __future__ import absolute_import, annotations

import os
from pathlib import Path
from typing import Union

# SageMaker core imports
from sagemaker.core.resources import Model, Endpoint
from sagemaker.core.utils.utils import logger


# SageMaker serve imports
from sagemaker.serve.local_resources import LocalEndpoint
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.local_hardware import (
    _get_nb_instance,
    _get_gpu_info,
    _get_gpu_info_fallback,
)
from sagemaker.serve.utils.hf_utils import _get_model_config_properties_from_hf
from sagemaker.serve.detector.image_detector import (
     _get_model_base, _detect_framework_and_version
)
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.utils.types import ModelServer

# Model server builders
from sagemaker.serve.model_server.djl_serving.utils import (
    _get_default_djl_configurations,
    _get_default_tensor_parallel_degree,
)
from sagemaker.serve.model_server.tgi.utils import _get_default_tgi_configurations
from sagemaker.serve.model_server.tgi.prepare import prepare_tgi_js_resources
from sagemaker.serve.model_server.djl_serving.prepare import prepare_djl_js_resources
from sagemaker.serve.model_server.multi_model_server.prepare import prepare_mms_js_resources

# Model server preparation
from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
from sagemaker.serve.model_server.multi_model_server.prepare import prepare_for_mms

# MLflow imports
from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
from sagemaker.serve.constants import LOCAL_MODES, SUPPORTED_MODEL_SERVERS

SCRIPT_PARAM_NAME = "sagemaker_program"
DIR_PARAM_NAME = "sagemaker_submit_directory"
CONTAINER_LOG_LEVEL_PARAM_NAME = "sagemaker_container_log_level"
JOB_NAME_PARAM_NAME = "sagemaker_job_name"
MODEL_SERVER_WORKERS_PARAM_NAME = "sagemaker_model_server_workers"
SAGEMAKER_REGION_PARAM_NAME = "sagemaker_region"
SAGEMAKER_OUTPUT_LOCATION = "sagemaker_s3_output"


class _ModelBuilderServers(object):

    def _build_for_model_server(self) -> Model:
        """Build model using explicit model server configuration.
        
        This method routes to the appropriate model server builder based on
        the specified model_server parameter. It validates that the model server
        is supported and that required parameters are provided.
        
        Returns:
            Model: A deployable model object configured for the specified model server.
            
        Raises:
            ValueError: If the model server is not supported or required parameters are missing.
        """
        if self.model_server not in SUPPORTED_MODEL_SERVERS:
            raise ValueError(
                f"{self.model_server} is not supported yet! Supported model servers: {SUPPORTED_MODEL_SERVERS}"
            )

        mlflow_path = None
        if self.model_metadata:
            mlflow_path = self.model_metadata.get(MLFLOW_MODEL_PATH)

        if not self.model and not mlflow_path and not self.inference_spec:
            raise ValueError("Missing required parameter: model, MLflow path, or inference_spec")

        # Route to appropriate model server builder
        if self.model_server == ModelServer.TORCHSERVE:
            return self._build_for_torchserve()
        elif self.model_server == ModelServer.TRITON:
            return self._build_for_triton()
        elif self.model_server == ModelServer.TENSORFLOW_SERVING:
            return self._build_for_tensorflow_serving()
        elif self.model_server == ModelServer.DJL_SERVING:
            return self._build_for_djl()
        elif self.model_server == ModelServer.TEI:
            return self._build_for_tei()
        elif self.model_server == ModelServer.TGI:
            return self._build_for_tgi()
        elif self.model_server == ModelServer.MMS:
            return self._build_for_transformers()
        elif self.model_server == ModelServer.SMD:
            return self._build_for_smd()
        else:
            raise ValueError(f"Unsupported model server: {self.model_server}")


    def _build_for_torchserve(self) -> Model:
        """Build model for TorchServe deployment.
        
        Configures the model for deployment using TorchServe model server.
        Handles both local model objects and HuggingFace model IDs.
        
        Returns:
            Model: Configured model ready for TorchServe deployment.
        """
        self.secret_key = ""
        
        # Save inference spec if we have local artifacts
        self._save_model_inference_spec()

        if isinstance(self.model, str):
            # Configure HuggingFace model support
            if not self._is_jumpstart_model_id():
                self.env_vars.update({"HF_MODEL_ID": self.model})
                
                # Add HuggingFace token if available
                if self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
                    self.env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                
                # HF models download directly from hub
                self.s3_upload_path = None

        if self.mode != Mode.IN_PROCESS:
            self._auto_detect_image_uri()

            # Prepare TorchServe artifacts for local container mode
            if self.mode == Mode.LOCAL_CONTAINER and self.model_path:
                self.secret_key = prepare_for_torchserve(
                    model_path=self.model_path,
                    shared_libs=self.shared_libs,
                    dependencies=self.dependencies,
                    session=self.sagemaker_session,
                    image_uri=self.image_uri,
                    inference_spec=self.inference_spec,
                )
            if self.mode == Mode.SAGEMAKER_ENDPOINT and self.model_path:
                self.secret_key = prepare_for_torchserve(
                    model_path=self.model_path,
                    shared_libs=self.shared_libs,
                    dependencies=self.dependencies,
                    session=self.sagemaker_session,
                    image_uri=self.image_uri,
                    inference_spec=self.inference_spec,
                )

        # Prepare deployment artifacts
        if self.mode in LOCAL_MODES:
            self._prepare_for_mode()
        else:
            self._prepare_for_mode(should_upload_artifacts=True)

        return self._create_model()


    def _build_for_tgi(self) -> Model:
        """Build model for Text Generation Inference (TGI) deployment.
        
        Configures the model for deployment using Hugging Face's TGI server,
        optimized for large language model inference with features like
        tensor parallelism and continuous batching.
        
        Returns:
            Model: Configured model ready for TGI deployment.
        """
        self.secret_key = ""
        
        # Initialize TGI-specific configuration
        if self.model_server != ModelServer.TGI:
            messaging = (
                "HuggingFace Model ID support on model server: "
                f"{self.model_server} is not currently supported. "
                f"Defaulting to {ModelServer.TGI}"
            )
            logger.warning(messaging)
            self.model_server = ModelServer.TGI

        self._validate_tgi_serving_sample_data()
        
        # Use notebook instance type if available
        nb_instance = _get_nb_instance()
        if nb_instance and not self._user_provided_instance_type:
            self.instance_type = nb_instance
            logger.debug(f"Using detected notebook instance type: {nb_instance}")

        from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if isinstance(self.model, str) and not self._is_jumpstart_model_id():
            # Configure HuggingFace model for TGI
            self.env_vars.update({"HF_MODEL_ID": self.model})
            
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )
            
            # Apply TGI-specific configurations
            default_tgi_configurations, _default_max_new_tokens = _get_default_tgi_configurations(
                self.model, self.hf_model_config, self.schema_builder
            )
            # Filter out None values to avoid pydantic validation errors
            filtered_tgi_config = {k: v for k, v in default_tgi_configurations.items() if v is not None}
            self.env_vars.update(filtered_tgi_config)
            
            # Configure schema builder for text generation
            if "parameters" not in self.schema_builder.sample_input:
                self.schema_builder.sample_input["parameters"] = {}
            self.schema_builder.sample_input["parameters"]["max_new_tokens"] = _default_max_new_tokens
            
            # Set TGI sharding defaults
            self.env_vars.setdefault("SHARDED", "false")
            self.env_vars.setdefault("NUM_SHARD", "1")
                
            # Configure HuggingFace cache and authentication
            tgi_env_vars = {
                "HF_HOME": "/tmp",
                "HUGGINGFACE_HUB_CACHE": "/tmp"
            }
            
            if self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
                tgi_env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                
            self.env_vars.update(tgi_env_vars)
            
            # TGI downloads models directly from HuggingFace Hub
            self.s3_upload_path = None

        self._auto_detect_image_uri()
        
        if not self._optimizing:
            if self.mode in LOCAL_MODES:
                self._prepare_for_mode(should_upload_artifacts=True)
            else:
                self.s3_model_data_url, _ = self._prepare_for_mode()

        # Cache management based on mode
        if self.mode in LOCAL_MODES:
            self.env_vars.update({"HF_HUB_OFFLINE": "1"})
        else:
            self.env_vars["HF_HOME"] = "/tmp"
            self.env_vars["HUGGINGFACE_HUB_CACHE"] = "/tmp"

        # GPU-based sharding logic for SAGEMAKER_ENDPOINT mode
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            if not self.instance_type:
                raise ValueError(
                    "Instance type must be provided when building for SageMaker Endpoint mode."
                )
            
            try:
                tot_gpus = _get_gpu_info(self.instance_type, self.sagemaker_session)
            except Exception:  # pylint: disable=W0703
                tot_gpus = _get_gpu_info_fallback(self.instance_type)
            
            default_num_shard = _get_default_tensor_parallel_degree(self.hf_model_config, tot_gpus)
            self.env_vars.update({
                "NUM_SHARD": str(default_num_shard),
                "SHARDED": "true" if default_num_shard > 1 else "false",
            })
            
        model = self._create_model()
    
        if "HF_HUB_OFFLINE" in self.env_vars:
            self.env_vars.update({"HF_HUB_OFFLINE": "0"})
        
        return model

    def _build_for_djl(self) -> Model:
        """Build model for DJL Serving deployment.
        
        Configures the model for deployment using Amazon's Deep Java Library (DJL)
        Serving, which provides high-performance inference with support for
        tensor parallelism and model partitioning.
        
        Returns:
            Model: Configured model ready for DJL Serving deployment.
        """
        self.secret_key = ""
        self.model_server = ModelServer.DJL_SERVING
        
        # Set MODEL_LOADING_TIMEOUT from instance variable
        if self.model_data_download_timeout:
            self.env_vars.update({"MODEL_LOADING_TIMEOUT": str(self.model_data_download_timeout)})
        
        self._validate_djl_serving_sample_data()

        # Create DJL directory structure for local artifacts
        from sagemaker.serve.model_server.djl_serving.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)
        
        # Use notebook instance type if available
        nb_instance = _get_nb_instance()
        if nb_instance and not self._user_provided_instance_type:
            self.instance_type = nb_instance
            logger.debug(f"Using detected notebook instance type: {nb_instance}")

        if isinstance(self.model, str) and not self._is_jumpstart_model_id():
            # Configure HuggingFace model for DJL
            self.env_vars.update({"HF_MODEL_ID": self.model})
            
            # Get model configuration for DJL optimization
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )
            
            # Apply DJL-specific configurations
            default_djl_configurations, _default_max_new_tokens = _get_default_djl_configurations(
                self.model, self.hf_model_config, self.schema_builder
            )
            self.env_vars.update(default_djl_configurations)
            
            # Configure schema builder for text generation
            if "parameters" not in self.schema_builder.sample_input:
                self.schema_builder.sample_input["parameters"] = {}
            self.schema_builder.sample_input["parameters"]["max_new_tokens"] = _default_max_new_tokens
            
            # Set DJL serving defaults
            djl_env_vars = {
                "OPTION_ENGINE": "Python",
                "SERVING_MIN_WORKERS": "1",
                "SERVING_MAX_WORKERS": "1", 
                "OPTION_MODEL_LOADING_TIMEOUT": "240",
                "OPTION_PREDICT_TIMEOUT": "60",
                "TENSOR_PARALLEL_DEGREE": "1"  # Default, will be overridden below
            }
            
            # Add HuggingFace authentication
            if self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
                djl_env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            
            # Update with defaults only if not already set
            for key, value in djl_env_vars.items():
                self.env_vars.setdefault(key, value)
            
            # DJL downloads models directly from HuggingFace Hub
            self.s3_upload_path = None

        self._auto_detect_image_uri()
        
        if not self._optimizing:
            if self.mode in LOCAL_MODES:
                self._prepare_for_mode(should_upload_artifacts=True)
            else:
                self.s3_model_data_url, _ = self._prepare_for_mode()

        # Cache management based on mode
        if self.mode in LOCAL_MODES:
            self.env_vars.update({"HF_HUB_OFFLINE": "1"})

        # GPU-based tensor parallel calculation for SAGEMAKER_ENDPOINT mode
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            if not self.instance_type:
                raise ValueError(
                    "Instance type must be provided when building for SageMaker Endpoint mode."
                )
            
            try:
                tot_gpus = _get_gpu_info(self.instance_type, self.sagemaker_session)
            except Exception:  # pylint: disable=W0703
                tot_gpus = _get_gpu_info_fallback(self.instance_type)
            
            default_tensor_parallel_degree = _get_default_tensor_parallel_degree(
                self.hf_model_config, tot_gpus
            )
            self.env_vars.update({
                "TENSOR_PARALLEL_DEGREE": str(default_tensor_parallel_degree)
            })
        
        model = self._create_model()
        
        if "HF_HUB_OFFLINE" in self.env_vars:
            self.env_vars.update({"TRANSFORMERS_OFFLINE": "0"})
        
        return model


    def _build_for_triton(self) -> Model:
        """Build model for NVIDIA Triton Inference Server deployment.
        
        Configures the model for deployment using NVIDIA Triton Inference Server,
        which provides high-performance inference with support for multiple
        frameworks and advanced features like dynamic batching.
        
        Returns:
            Model: Configured model ready for Triton deployment.
        """
        self.secret_key = ""
        self._validate_for_triton()

        if isinstance(self.model, str):
            self.framework = None
            self.framework_version = None
            
            # Configure HuggingFace model for Triton
            if not self._is_jumpstart_model_id():
                # Get model metadata for task detection
                hf_model_md = self.get_huggingface_model_metadata(
                    self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )
                model_task = hf_model_md.get("pipeline_tag")
                if model_task:
                    self.env_vars.update({"HF_TASK": model_task})
                
                # Configure HuggingFace authentication
                self.env_vars.update({"HF_MODEL_ID": self.model})
                if self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
                    self.env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                
                # Triton downloads models directly from HuggingFace Hub
                self.s3_upload_path = None
        elif self.model:
            fw, self.framework_version = _detect_framework_and_version(str(_get_model_base(self.model)))
            self.framework = self._normalize_framework_to_enum(fw)
        
        # Auto-detect Triton image if not provided
        if not self.image_uri:
            self._auto_detect_image_for_triton()
            
        # Prepare Triton-specific artifacts
        self._save_inference_spec()
        self._prepare_for_triton()
        
        # Prepare deployment artifacts
        if self.mode in LOCAL_MODES:
            self._prepare_for_mode()
        else:
            # self._prepare_for_mode(should_upload_artifacts=True)
            self.s3_model_data_url, _ = self._prepare_for_mode(should_upload_artifacts=True)
            
        return self._create_model()



    def _build_for_tensorflow_serving(self) -> Model:
        """Build model for TensorFlow Serving deployment.
        
        Configures the model for deployment using TensorFlow Serving,
        Google's high-performance serving system for TensorFlow models.
        
        Returns:
            Model: Configured model ready for TensorFlow Serving deployment.
            
        Raises:
            ValueError: If image_uri is not provided for TensorFlow Serving.
        """
        self.secret_key = ""
        if not getattr(self, "_is_mlflow_model", False):
            raise ValueError("Tensorflow Serving is currently only supported for mlflow models.")
        
        # Save Schema Builder
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        code_path = Path(self.model_path).joinpath("code")
        save_pkl(code_path, self.schema_builder)

        if not self.image_uri:
            raise ValueError("image_uri is required for TensorFlow Serving deployment")

        # Prepare TensorFlow Serving artifacts for local container mode
        self.secret_key = prepare_for_tf_serving(
            model_path=self.model_path,
            shared_libs=self.shared_libs,
            dependencies=self.dependencies,
        )

        # Prepare deployment artifacts
        if self.mode in LOCAL_MODES:
            self._prepare_for_mode()
        else:
            self.s3_model_data_url, _ = self._prepare_for_mode(should_upload_artifacts=True)

        return self._create_model()


    def _build_for_tei(self) -> Model:
        """Build model for Text Embeddings Inference (TEI) deployment.
        
        Configures the model for deployment using Hugging Face's TEI server,
        optimized for embedding model inference with features like
        pooling strategies and efficient batching.
        
        Returns:
            Model: Configured model ready for TEI deployment.
        """
        self.secret_key = ""
        
        # Set MODEL_LOADING_TIMEOUT from instance variable
        if self.model_data_download_timeout:
            self.env_vars.update({"MODEL_LOADING_TIMEOUT": str(self.model_data_download_timeout)})
        
        if self.model_server != ModelServer.TEI:
            messaging = (
                "HuggingFace Model ID support on model server: "
                f"{self.model_server} is not currently supported. "
                f"Defaulting to {ModelServer.TEI}"
            )
            logger.warning(messaging)
            self.model_server = ModelServer.TEI

        # Use notebook instance type if available
        nb_instance = _get_nb_instance()
        if nb_instance and not self._user_provided_instance_type:
            self.instance_type = nb_instance
            logger.debug(f"Using detected notebook instance type: {nb_instance}")

        from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if isinstance(self.model, str) and not self._is_jumpstart_model_id():
            # Configure HuggingFace model for TEI
            self.env_vars.update({"HF_MODEL_ID": self.model})
            
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )
            
            # Configure TEI-specific environment variables
            tei_env_vars = {
                "HF_HOME": "/tmp",
                "HUGGINGFACE_HUB_CACHE": "/tmp",
                "MODEL_LOADING_TIMEOUT": "240",
            }
            
            if self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
                tei_env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                
            self.env_vars.update(tei_env_vars)
            
            # TEI downloads models directly from HuggingFace Hub
            self.s3_upload_path = None

        self._auto_detect_image_uri()
        
        if not self._optimizing:
            if self.mode in LOCAL_MODES:
                self._prepare_for_mode(should_upload_artifacts=True)
            else:
                self.s3_model_data_url, _ = self._prepare_for_mode()

        # Cache management based on mode
        if self.mode in LOCAL_MODES:
            self.env_vars.update({"HF_HUB_OFFLINE": "1"})
        else:
            self.env_vars["HF_HOME"] = "/tmp"
            self.env_vars["HUGGINGFACE_HUB_CACHE"] = "/tmp"

        # Instance type validation for SAGEMAKER_ENDPOINT mode
        if self.mode == Mode.SAGEMAKER_ENDPOINT and not self.instance_type:
            raise ValueError(
                "Instance type must be provided when building for SageMaker Endpoint mode."
            )
        
        model = self._create_model()
        
        if "HF_HUB_OFFLINE" in self.env_vars:
            self.env_vars.update({"HF_HUB_OFFLINE": "0"})
        
        return model


    def _build_for_smd(self) -> Model:
        """Build model for SageMaker Distribution (SMD) deployment.
        
        Configures the model for deployment using SageMaker's distribution
        container, which provides a comprehensive ML environment with
        pre-installed frameworks and optimizations.
        
        Returns:
            Model: Configured model ready for SMD deployment.
        """
        self.secret_key = ""
        
        self._save_model_inference_spec()

        if self.mode != Mode.IN_PROCESS:
            # Auto-detect SMD image based on compute requirements
            if not self.image_uri:
                cpu_or_gpu = self._get_processing_unit()
                self.image_uri = self._get_smd_image_uri(processing_unit=cpu_or_gpu)

            self.secret_key = prepare_for_smd(
                model_path=self.model_path,
                shared_libs=self.shared_libs,
                dependencies=self.dependencies,
                inference_spec=self.inference_spec,
            )

        # Prepare deployment artifacts
        if self.mode in LOCAL_MODES:
            self._prepare_for_mode()
        else:
            self.s3_model_data_url, _ = self._prepare_for_mode(should_upload_artifacts=True)

        return self._create_model()

    def _build_for_transformers(self) -> Model:
        """Build model for HuggingFace Transformers deployment.
        
        Configures the model for deployment using Multi-Model Server (MMS)
        with HuggingFace Transformers backend for general-purpose model inference.
        
        Returns:
            Model: Configured model ready for Transformers deployment.
        """
        self.secret_key = ""
        self.model_server = ModelServer.MMS
        
        # Set MODEL_LOADING_TIMEOUT from instance variable
        if self.model_data_download_timeout:
            self.env_vars.update({"MODEL_LOADING_TIMEOUT": str(self.model_data_download_timeout)})

        if self.mode != Mode.IN_PROCESS:
            self._auto_detect_image_uri()

        if self.inference_spec:
            os.makedirs(self.model_path, exist_ok=True)
            code_path = Path(self.model_path).joinpath("code")

            save_pkl(code_path, (self.inference_spec, self.schema_builder))

            if self.mode == Mode.IN_PROCESS:
                self._create_conda_env()

            if self.mode in [Mode.LOCAL_CONTAINER] and self.model_path:
                self.secret_key = prepare_for_mms(
                    model_path=self.model_path,
                    shared_libs=self.shared_libs,
                    dependencies=self.dependencies,
                    session=self.sagemaker_session,
                    image_uri=self.image_uri,
                    inference_spec=self.inference_spec,
                )
            if self.mode == Mode.SAGEMAKER_ENDPOINT and self.model_path:
                self.secret_key = prepare_for_mms(
                    model_path=self.model_path,
                    shared_libs=self.shared_libs,
                    dependencies=self.dependencies,
                    session=self.sagemaker_session,
                    image_uri=self.image_uri,
                    inference_spec=self.inference_spec,
                )

        nb_instance = _get_nb_instance()
        if nb_instance and not self._user_provided_instance_type:
            self.instance_type = nb_instance
            logger.debug(f"Using detected notebook instance type: {nb_instance}")

        from sagemaker.serve.model_server.multi_model_server.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if not isinstance(self.model, str) or not self._is_jumpstart_model_id():
            if self.inference_spec is not None:
                hf_model_id = self.inference_spec.get_model()
                if isinstance(hf_model_id, str):  # Only if it's a valid HF model ID
                    self.env_vars.update({"HF_MODEL_ID": hf_model_id})
                    # Get HF config only for string model IDs
                    if hasattr(self.env_vars, "HF_API_TOKEN"):
                        self.hf_model_config = _get_model_config_properties_from_hf(
                            hf_model_id, self.env_vars.get("HF_API_TOKEN")
                        )
                    else:
                        self.hf_model_config = _get_model_config_properties_from_hf(
                            hf_model_id, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                        )
            elif isinstance(self.model, str):  # Only set HF_MODEL_ID if model is a string
                self.env_vars.update({"HF_MODEL_ID": self.model})
                # Get HF config for string model IDs
                if hasattr(self.env_vars, "HF_API_TOKEN"):
                    self.hf_model_config = _get_model_config_properties_from_hf(
                        self.model, self.env_vars.get("HF_API_TOKEN")
                    )
                else:
                    self.hf_model_config = _get_model_config_properties_from_hf(
                        self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                    )

        if not self._optimizing:
            if self.mode in LOCAL_MODES:
                self._prepare_for_mode(should_upload_artifacts=True)
            else:
                self.s3_model_data_url, _ = self._prepare_for_mode()


        # Clean up empty secret key
        if (
            "SAGEMAKER_SERVE_SECRET_KEY" in self.env_vars
            and not self.env_vars["SAGEMAKER_SERVE_SECRET_KEY"]
        ):
            del self.env_vars["SAGEMAKER_SERVE_SECRET_KEY"]

        # Instance type validation for SAGEMAKER_ENDPOINT mode
        if self.mode == Mode.SAGEMAKER_ENDPOINT and not self.instance_type:
            raise ValueError(
                "Instance type must be provided when building for SageMaker Endpoint mode."
            )

        return self._create_model()



    def _build_for_djl_jumpstart(self, init_kwargs) -> Model:
        """Build DJL model using JumpStart artifacts.
        
        Args:
            init_kwargs: JumpStart initialization parameters.
            
        Returns:
            Model: Configured DJL model for JumpStart deployment.
        """
        self.secret_key = ""
        self.model_server = ModelServer.DJL_SERVING

        from sagemaker.serve.model_server.djl_serving.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if self.mode in LOCAL_MODES:
            # Prepare DJL resources for local deployment
            (
                self.js_model_config,
                self.prepared_for_djl
            ) = prepare_djl_js_resources(
                model_path=self.model_path,
                js_id=self.model,
                dependencies=self.dependencies,
                model_data=self.s3_model_data_url,
            )
            self._prepare_for_mode()
        else:
            # Use pre-packaged JumpStart S3 artifacts
            if not hasattr(self, "s3_upload_path") or not self.s3_upload_path:
                self.s3_upload_path = init_kwargs.model_data

        self.prepared_for_djl = True
        return self._create_model()


    def _build_for_tgi_jumpstart(self, init_kwargs) -> Model:
        """Build TGI model using JumpStart artifacts.
        
        Args:
            init_kwargs: JumpStart initialization parameters.
            
        Returns:
            Model: Configured TGI model for JumpStart deployment.
        """
        self.secret_key = ""
        self.model_server = ModelServer.TGI

        from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if self.mode in LOCAL_MODES:
            # Prepare TGI resources for local deployment
            (
                self.js_model_config,
                _,
            ) = prepare_tgi_js_resources(
                model_path=self.model_path,
                js_id=self.model,
                dependencies=self.dependencies,
                model_data=self.s3_model_data_url,
            )
            self._prepare_for_mode()
        else:
            # Use pre-packaged JumpStart S3 artifacts
            if not hasattr(self, "s3_upload_path") or not self.s3_upload_path:
                self.s3_upload_path = init_kwargs.model_data

        self.prepared_for_tgi = True  
        return self._create_model()


    def _build_for_mms_jumpstart(self, init_kwargs) -> Model:
        """Build MMS model using JumpStart artifacts.
        
        Args:
            init_kwargs: JumpStart initialization parameters.
            
        Returns:
            Model: Configured MMS model for JumpStart deployment.
        """
        self.secret_key = ""
        self.model_server = ModelServer.MMS

        from sagemaker.serve.model_server.multi_model_server.prepare import _create_dir_structure
        _create_dir_structure(self.model_path)

        if self.mode in LOCAL_MODES:
            # Prepare MMS resources for local deployment
            (
                self.js_model_config,
                _,
            ) = prepare_mms_js_resources(
                model_path=self.model_path,
                js_id=self.model,
                dependencies=self.dependencies,
                model_data=self.s3_model_data_url,
            )
            self._prepare_for_mode()
        else:
            # Use pre-packaged JumpStart S3 artifacts
            if not hasattr(self, "s3_upload_path") or not self.s3_upload_path:
                self.s3_upload_path = init_kwargs.model_data

        self.prepared_for_mms = True
        return self._create_model()



    def _build_for_jumpstart(self) -> Model:
        """Build model for JumpStart deployment.
        
        Configures the model for deployment using SageMaker JumpStart,
        which provides pre-trained models with optimized configurations
        and deployment settings.
        
        Returns:
            Model: Configured model ready for JumpStart deployment.
            
        Raises:
            ValueError: If the JumpStart image URI is not supported.
        """
        from sagemaker.core.jumpstart.factory.utils import get_init_kwargs

        self.secret_key = ""
        
        # Get JumpStart model configuration
        init_kwargs = get_init_kwargs(
            model_id=self.model,
            model_version=self.model_version or "*", 
            region=self.region,
            instance_type=self.instance_type,
            tolerate_vulnerable_model=getattr(self, 'tolerate_vulnerable_model', None),
            tolerate_deprecated_model=getattr(self, 'tolerate_deprecated_model', None)
        )
        
        # Configure image URI and environment variables
        self.image_uri = self.image_uri or init_kwargs.image_uri
        
        if hasattr(init_kwargs, 'env') and init_kwargs.env:
            self.env_vars.update(init_kwargs.env)
        
        # Handle model artifacts for fine-tuned models
        if hasattr(init_kwargs, 'model_data') and init_kwargs.model_data:
            if isinstance(init_kwargs.model_data, dict) and 'S3DataSource' in init_kwargs.model_data:
                self.s3_model_data_url = init_kwargs.model_data['S3DataSource']['S3Uri']
            elif isinstance(init_kwargs.model_data, str):
                self.s3_model_data_url = init_kwargs.model_data
        
        # Prepare resources based on mode and model server
        if self.mode == Mode.LOCAL_CONTAINER:
            # Route to appropriate model server and prepare resources
            if "djl-inference" in self.image_uri:
                self.model_server = ModelServer.DJL_SERVING
                if not hasattr(self, "prepared_for_djl"):
                    (
                        self.js_model_config,
                        self.prepared_for_djl,
                    ) = prepare_djl_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.s3_model_data_url,
                    )
                return self._build_for_djl_jumpstart(init_kwargs)
                
            elif "tgi-inference" in self.image_uri:
                self.model_server = ModelServer.TGI
                if not hasattr(self, "prepared_for_tgi"):
                    self.js_model_config, self.prepared_for_tgi = prepare_tgi_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.s3_model_data_url,
                    )
                return self._build_for_tgi_jumpstart(init_kwargs)
                
            elif "huggingface-pytorch-inference" in self.image_uri:
                self.model_server = ModelServer.MMS
                if not hasattr(self, "prepared_for_mms"):
                    self.js_model_config, self.prepared_for_mms = prepare_mms_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.s3_model_data_url,
                    )
                return self._build_for_mms_jumpstart(init_kwargs)
            else:
                raise ValueError(f"Unsupported JumpStart image URI: {self.image_uri}")
        
        else:
            # SAGEMAKER_ENDPOINT mode - prepare artifacts if needed
            if not self._optimizing:
                self._prepare_for_mode()
            
            # Route to appropriate model server based on image URI
            if "djl-inference" in self.image_uri:
                self.model_server = ModelServer.DJL_SERVING
                return self._build_for_djl_jumpstart(init_kwargs)
            elif "tgi-inference" in self.image_uri:
                self.model_server = ModelServer.TGI
                return self._build_for_tgi_jumpstart(init_kwargs)
            elif "huggingface-pytorch-inference" in self.image_uri:
                self.model_server = ModelServer.MMS
                return self._build_for_mms_jumpstart(init_kwargs)
            else:
                raise ValueError(f"Unsupported JumpStart image URI: {self.image_uri}")



    def _djl_model_builder_deploy_wrapper(self, *args, **kwargs) -> Union[Endpoint, LocalEndpoint]:
        """Returns predictor depending on local mode or endpoint mode"""
        timeout = kwargs.get("model_data_download_timeout")
        if timeout:
            self.env_vars.update({"MODEL_LOADING_TIMEOUT": str(timeout)})

        # Handle local deployment modes
        if self.mode == Mode.IN_PROCESS:
            return self._deploy_local_endpoint(**kwargs)
        
        if self.mode == Mode.LOCAL_CONTAINER:
            return self._deploy_local_endpoint(**kwargs)
        
        # Remove mode/role from kwargs if present
        kwargs.pop("mode", None)
        if "role" in kwargs:
            self.role_arn = kwargs.pop("role")
        
        # Set default values
        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        
        if "initial_instance_count" not in kwargs:
            kwargs["initial_instance_count"] = 1
        
        # Deploy to SageMaker endpoint
        return self._deploy_core_endpoint(*args, **kwargs)
    

    def _tgi_model_builder_deploy_wrapper(self, *args, **kwargs) -> Union[Endpoint, LocalEndpoint]:
        """Simplified TGI deploy wrapper - env vars already set during build."""
        
        # Handle mode overrides for local deployment
        if self.mode == Mode.IN_PROCESS:
            return self._deploy_local_endpoint(**kwargs)
        
        if self.mode == Mode.LOCAL_CONTAINER:
            return self._deploy_local_endpoint(**kwargs)
        
        # Remove mode/role from kwargs if present
        kwargs.pop("mode", None)
        if "role" in kwargs:
            self.role_arn = kwargs.pop("role")
        
        # Set default values
        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        
        if "initial_instance_count" not in kwargs:
            kwargs["initial_instance_count"] = 1
        
        # Deploy to SageMaker endpoint
        return self._deploy_core_endpoint(*args, **kwargs)

    
    def _tei_model_builder_deploy_wrapper(self, *args, **kwargs) -> Union[Endpoint, LocalEndpoint]:
        """Simplified TEI deploy wrapper - env vars already set during build."""
        
        # Handle local deployment modes
        if self.mode == Mode.IN_PROCESS:
            return self._deploy_local_endpoint(**kwargs)
        
        if self.mode == Mode.LOCAL_CONTAINER:
            return self._deploy_local_endpoint(**kwargs)
        
        # Remove mode/role from kwargs if present
        kwargs.pop("mode", None)
        if "role" in kwargs:
            self.role_arn = kwargs.pop("role")
        
        # Set default values
        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        
        if "initial_instance_count" not in kwargs:
            kwargs["initial_instance_count"] = 1
        
        # Deploy to SageMaker endpoint
        return self._deploy_core_endpoint(**kwargs)

    

    def _js_builder_deploy_wrapper(self, *args, **kwargs) -> Union[Endpoint, LocalEndpoint]:
        """Simplified JumpStart deploy wrapper - resource prep already done during build."""
        
        # Handle local deployment
        if self.mode == Mode.LOCAL_CONTAINER:
            return self._deploy_local_endpoint(**kwargs)
        
        # Remove mode/role from kwargs if present
        kwargs.pop("mode", None)
        if "role" in kwargs:
            self.role_arn = kwargs.pop("role")
        
        # Set default values
        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        
        if hasattr(self, "instance_type"):
            kwargs.update({"instance_type": self.instance_type})
        
        # Deploy to SageMaker endpoint
        return self._deploy_core_endpoint(**kwargs)

    def _transformers_model_builder_deploy_wrapper(self, *args, **kwargs) -> Union[Endpoint, LocalEndpoint]:
        """Simplified Transformers deploy wrapper - env vars already set during build."""
        
        # Handle local deployment modes
        if self.mode == Mode.LOCAL_CONTAINER:
            return self._deploy_local_endpoint(**kwargs)
        
        if self.mode == Mode.IN_PROCESS:
            return self._deploy_local_endpoint(**kwargs)
        
        # Remove mode/role from kwargs if present
        kwargs.pop("mode", None)
        if "role" in kwargs:
            self.role_arn = kwargs.pop("role")
        
        # Set default values
        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        
        if "initial_instance_count" not in kwargs:
            kwargs["initial_instance_count"] = 1
        
        # Deploy to SageMaker endpoint
        return self._deploy_core_endpoint(**kwargs)
