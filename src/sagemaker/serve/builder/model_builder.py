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
"""Holds the ModelBuilder class and the ModelServer enum."""
from __future__ import absolute_import, annotations

import importlib.util
import json
import uuid
from typing import Any, Type, List, Dict, Optional, Union
from dataclasses import dataclass, field
import logging
import os
import re

from pathlib import Path

from botocore.exceptions import ClientError
from sagemaker_core.main.resources import TrainingJob

from sagemaker.transformer import Transformer
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig
from sagemaker.compute_resource_requirements import ResourceRequirements
from sagemaker.enums import Tag, EndpointType
from sagemaker.estimator import Estimator
from sagemaker.jumpstart.accessors import JumpStartS3PayloadAccessor
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.s3 import S3Downloader
from sagemaker import Session
from sagemaker.model import Model
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.base_predictor import PredictorBase
from sagemaker.serializers import NumpySerializer, TorchTensorSerializer
from sagemaker.deserializers import JSONDeserializer, TorchTensorDeserializer
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.builder.tf_serving_builder import TensorflowServing
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.mode.sagemaker_endpoint_mode import SageMakerEndpointMode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.mode.in_process_mode import InProcessMode
from sagemaker.serve.detector.pickler import save_pkl, save_xgboost
from sagemaker.serve.builder.serve_settings import _ServeSettings
from sagemaker.serve.builder.djl_builder import DJL
from sagemaker.serve.builder.tei_builder import TEI
from sagemaker.serve.builder.tgi_builder import TGI
from sagemaker.serve.builder.jumpstart_builder import JumpStart
from sagemaker.serve.builder.transformers_builder import Transformers
from sagemaker.predictor import Predictor
from sagemaker.serve.model_format.mlflow.constants import (
    MLFLOW_MODEL_PATH,
    MLFLOW_TRACKING_ARN,
    MLFLOW_RUN_ID_REGEX,
    MLFLOW_REGISTRY_PATH_REGEX,
    MODEL_PACKAGE_ARN_REGEX,
    MLFLOW_METADATA_FILE,
    MLFLOW_PIP_DEPENDENCY_FILE,
)
from sagemaker.serve.model_format.mlflow.utils import (
    _get_default_model_server_for_mlflow,
    _download_s3_artifacts,
    _select_container_for_mlflow_model,
    _generate_mlflow_artifact_path,
    _get_all_flavor_metadata,
    _get_deployment_flavor,
    _validate_input_for_mlflow,
    _copy_directory_contents,
)
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import Metadata
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.spec.inference_base import CustomOrchestrator, AsyncCustomOrchestrator
from sagemaker.serve.utils import task
from sagemaker.serve.utils.exceptions import TaskNotFoundException
from sagemaker.serve.utils.lineage_utils import _maintain_lineage_tracking_for_mlflow_model
from sagemaker.serve.utils.optimize_utils import (
    _generate_optimized_model,
    _generate_model_source,
    _extract_optimization_config_and_env,
    _is_s3_uri,
    _custom_speculative_decoding,
    _extract_speculative_draft_model_provider,
    _jumpstart_speculative_decoding,
)
from sagemaker.serve.utils.predictors import (
    _get_local_mode_predictor,
    _get_in_process_mode_predictor,
)
from sagemaker.serve.utils.hardware_detector import (
    _get_gpu_info,
    _get_gpu_info_fallback,
    _total_inference_model_size_mib,
)
from sagemaker.serve.detector.image_detector import (
    auto_detect_container,
    _detect_framework_and_version,
    _get_model_base,
)
from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve
from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
from sagemaker.serve.model_server.triton.triton_builder import Triton
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.serve.utils.types import ModelServer, ModelHub
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.serve.save_retrive.version_1_0_0.save.save_handler import SaveHandler
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import get_metadata
from sagemaker.serve.validations.check_image_and_hardware_type import (
    validate_image_uri_and_hardware,
)
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import Tags
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.huggingface.llm_utils import (
    get_huggingface_model_metadata,
    download_huggingface_model_metadata,
)
from sagemaker.serve.validations.optimization import _validate_optimization_configuration
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules import logger

# Any new server type should be added here
supported_model_servers = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
    ModelServer.TENSORFLOW_SERVING,
    ModelServer.MMS,
    ModelServer.TGI,
    ModelServer.TEI,
    ModelServer.SMD,
}


# pylint: disable=attribute-defined-outside-init, disable=E1101, disable=R0901, disable=R1705
@dataclass
class ModelBuilder(Triton, DJL, JumpStart, TGI, Transformers, TensorflowServing, TEI):
    """Class that builds a deployable model.

    Args:
        role_arn (Optional[str]): The role for the endpoint.
        model_path (Optional[str]): The path of the model directory.
        sagemaker_session (Optional[sagemaker.session.Session]): The
            SageMaker session to use for the execution.
        name (Optional[str]): The model name.
        mode (Optional[Mode]): The mode of operation. The following
            modes are supported:

            * ``Mode.SAGEMAKER_ENDPOINT``: Launch on a SageMaker endpoint
            * ``Mode.LOCAL_CONTAINER``: Launch locally with a container
            * ``Mode.IN_PROCESS``: Launch locally to a FastAPI server instead of using a container.
        shared_libs (List[str]): Any shared libraries you want to bring into
            the model packaging.
        dependencies (Optional[Dict[str, Any]): The dependencies of the model
            or container. Takes a dict as an input where you can specify
            autocapture as ``True`` or ``False``, a requirements file, or custom
            dependencies as a list. A sample ``dependencies`` dict:

            .. code:: python

                {
                    "auto": False,
                    "requirements": "/path/to/requirements.txt",
                    "custom": ["custom_module==1.2.3",
                      "other_module@http://some/website.whl"],
                }

        env_vars (Optional[Dict[str, str]): The environment variables for the runtime
            execution.
        log_level (Optional[int]): The log level. Possible values are ``CRITICAL``,
            ``ERROR``, ``WARNING``, ``INFO``, ``DEBUG``, and ``NOTSET``.
        content_type (Optional[str]): The content type of the endpoint input data. This value
            is automatically derived from the input sample, but you can override it.
        accept_type (Optional[str]): The content type of the data accepted from the endpoint.
            This value is automatically derived from the output sample, but you can override
            the value.
        s3_model_data_url (Optional[str]): The S3 location where you want to upload the model
            package. Defaults to a S3 bucket prefixed with the account ID.
        instance_type (Optional[str]): The instance type of the endpoint. Defaults to the CPU
            instance type to help narrow the container type based on the instance family.
        schema_builder (Optional[SchemaBuilder]): The input/output schema of the model.
            The schema builder translates the input into bytes and converts the response
            into a stream. All translations between the server and the client are handled
            automatically with the specified input and output.
            The schema builder can be omitted for HuggingFace models with task types TextGeneration,
            TextClassification, and QuestionAnswering. Omitting SchemaBuilder is in
            beta for FillMask, and AutomaticSpeechRecognition use-cases.
        model (Optional[Union[object, str, ModelTrainer, TrainingJob, Estimator]]):
            Define object from which training artifacts can be extracted.
            Either ``model`` or ``inference_spec``
            is required for the model builder to build the artifact.
        inference_spec (InferenceSpec): The inference spec file with your customized
            ``invoke`` and ``load`` functions.
        image_uri (Optional[str]): The container image uri (which is derived from a
            SageMaker-based container).
        image_config (dict[str, str] or dict[str, PipelineVariable]): Specifies
            whether the image of model container is pulled from ECR, or private
            registry in your VPC. By default it is set to pull model container
            image from ECR. (default: None).
        vpc_config ( Optional[Dict[str, List[Union[str, PipelineVariable]]]]):
                The VpcConfig set on the model (default: None)
                * 'Subnets' (List[Union[str, PipelineVariable]]): List of subnet ids.
                * 'SecurityGroupIds' (List[Union[str, PipelineVariable]]]): List of security group
                ids.
        model_server (Optional[ModelServer]): The model server to which to deploy.
            You need to provide this argument when you specify an ``image_uri``
            in order for model builder to build the artifacts correctly (according
            to the model server). Possible values for this argument are
            ``TORCHSERVE``, ``MMS``, ``TENSORFLOW_SERVING``, ``DJL_SERVING``,
            ``TRITON``, ``TGI``, and ``TEI``.
        model_metadata (Optional[Dict[str, Any]): Dictionary used to override model metadata.
            Currently, ``HF_TASK`` is overridable for HuggingFace model. HF_TASK should be set for
            new models without task metadata in the Hub, adding unsupported task types will throw
            an exception. ``MLFLOW_MODEL_PATH`` is available for providing local path or s3 path
            to MLflow artifacts. However, ``MLFLOW_MODEL_PATH`` is experimental and is not
            intended for production use at this moment. ``CUSTOM_MODEL_PATH`` is available for
            providing local path or s3 path to model artifacts. ``FINE_TUNING_MODEL_PATH`` is
            available for providing s3 path to fine-tuned model artifacts. ``FINE_TUNING_JOB_NAME``
            is available for providing fine-tuned job name. Both ``FINE_TUNING_MODEL_PATH`` and
            ``FINE_TUNING_JOB_NAME`` are mutually exclusive.
        inference_component_name (Optional[str]): The name for an inference component
            created from this ModelBuilder instance. This or ``resource_requirements`` must be set
            to denote that this instance refers to an inference component.
        modelbuilder_list: Optional[List[ModelBuilder]] = List of ModelBuilder objects which
            can be built in bulk and subsequently deployed in bulk. Currently only supports
            deployments for inference components.
        resource_requirements: Optional[ResourceRequirements] = Defines the compute resources
            allocated to run the model assigned to the inference component. This or
            ``inference_component_name`` must be set to denote that this instance refers
            to an inference component. If ``inference_component_name`` is set but this is not and a
            JumpStart model ID is specified, pre-benchmarked deployment configs will attempt to be
            retrieved for the model.
    """

    model_path: Optional[str] = field(
        default="/tmp/sagemaker/model-builder/" + uuid.uuid1().hex,
        metadata={"help": "Define the path of model directory"},
    )
    role_arn: Optional[str] = field(
        default=None, metadata={"help": "Define the role for the endpoint"}
    )
    sagemaker_session: Optional[Session] = field(
        default=None, metadata={"help": "Define sagemaker session for execution"}
    )
    name: Optional[str] = field(
        default_factory=lambda: "model-name-" + uuid.uuid1().hex,
        metadata={"help": "Define the model name"},
    )
    mode: Optional[Mode] = field(
        default=Mode.SAGEMAKER_ENDPOINT,
        metadata={
            "help": "Define the mode of operation"
            "Model Builder supports three modes "
            "1/ SageMaker Endpoint"
            "2/ Local launch with container"
            "3/ Local launch in process"
        },
    )
    shared_libs: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Define any shared lib you want to bring into the model " "packaging"},
    )
    dependencies: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"auto": False},
        metadata={"help": "Define the dependencies of the model/container"},
    )
    env_vars: Optional[Dict[str, str]] = field(
        default_factory=lambda: {},
        metadata={"help": "Define the environment variables"},
    )
    log_level: Optional[int] = field(
        default=logging.DEBUG, metadata={"help": "Define the log level"}
    )
    content_type: Optional[str] = field(
        default=None,
        metadata={"help": "Define the content type of the input data to the endpoint"},
    )
    accept_type: Optional[str] = field(
        default=None,
        metadata={"help": "Define the accept type of the output data from the endpoint"},
    )
    s3_model_data_url: Optional[str] = field(
        default=None,
        metadata={"help": "Define the s3 location where you want to upload the model package"},
    )
    instance_type: Optional[str] = field(
        default="ml.c5.xlarge",
        metadata={"help": "Define the instance_type of the endpoint"},
    )
    schema_builder: Optional[SchemaBuilder] = field(
        default=None, metadata={"help": "Defines the i/o schema of the model"}
    )
    model: Optional[Union[object, str, ModelTrainer, TrainingJob, Estimator]] = field(
        default=None,
        metadata={"help": "Define object from which training artifacts can be extracted"},
    )
    inference_spec: InferenceSpec = field(
        default=None,
        metadata={"help": "Define the inference spec file for all customizations"},
    )
    image_uri: Optional[str] = field(
        default=None, metadata={"help": "Define the container image uri"}
    )
    image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = field(
        default=None,
        metadata={
            "help": "Specifies whether the image of model container is pulled from ECR,"
            " or private registry in your VPC. By default it is set to pull model "
            "container image from ECR. (default: None)."
        },
    )
    vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = field(
        default=None,
        metadata={
            "help": "The VpcConfig set on the model (default: None)."
            "* 'Subnets' (List[Union[str, PipelineVariable]]): List of subnet ids."
            "* ''SecurityGroupIds'' (List[Union[str, PipelineVariable]]): List of"
            " security group ids."
        },
    )
    model_server: Optional[ModelServer] = field(
        default=None, metadata={"help": "Define the model server to deploy to."}
    )
    model_metadata: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Define the model metadata to override, currently supports `HF_TASK`, "
            "`MLFLOW_MODEL_PATH`, `FINE_TUNING_MODEL_PATH`, `FINE_TUNING_JOB_NAME`, and "
            "`CUSTOM_MODEL_PATH`. HF_TASK should be set for new models without task metadata "
            "in the Hub, Adding unsupported task types will throw an exception."
        },
    )
    inference_component_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defines the name for an Inference Component created from this ModelBuilder."
        },
    )
    modelbuilder_list: Optional[List[ModelBuilder]] = field(
        default=None,
        metadata={"help": "Defines a list of ModelBuilder objects."},
    )
    resource_requirements: Optional[ResourceRequirements] = field(
        default=None,
        metadata={
            "help": "Defines the compute resources allocated to run the model assigned"
            " to the inference component."
        },
    )

    def _save_model_inference_spec(self):
        """Placeholder docstring"""
        # check if path exists and create if not
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        code_path = Path(self.model_path).joinpath("code")
        # save the model or inference spec in cloud pickle format
        if self.inference_spec:
            save_pkl(code_path, (self.inference_spec, self.schema_builder))
        elif self.model:
            self._framework, _ = _detect_framework_and_version(str(_get_model_base(self.model)))
            self.env_vars.update(
                {
                    "MODEL_CLASS_NAME": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
                }
            )

            # TODO: use framework built-in method to save and load the model for all frameworks
            if self._framework == "xgboost":
                save_xgboost(code_path, self.model)
                save_pkl(code_path, (self._framework, self.schema_builder))
            else:
                save_pkl(code_path, (self.model, self.schema_builder))
        elif self._is_mlflow_model:
            save_pkl(code_path, self.schema_builder)
        else:
            raise ValueError("Cannot detect required model or inference spec")

    def _auto_detect_container(self):
        """Placeholder docstring"""
        # Auto detect the container image uri
        if self.image_uri:
            logger.info(
                "Skipping auto detection as the image uri is provided %s",
                self.image_uri,
            )
            return

        if self.model:
            logger.info(
                "Auto detect container url for the provided model and on instance %s",
                self.instance_type,
            )
            self.image_uri = auto_detect_container(
                self.model, self.sagemaker_session.boto_region_name, self.instance_type
            )

        elif self.inference_spec:
            # TODO: this won't work for larger image.
            # Fail and let the customer include the image uri
            logger.warning(
                "model_path provided with no image_uri. Attempting to autodetect the image\
                    by loading the model using inference_spec.load()..."
            )
            self.image_uri = auto_detect_container(
                self.inference_spec.load(self.model_path),
                self.sagemaker_session.boto_region_name,
                self.instance_type,
            )
        else:
            raise ValueError("Cannot detect required model or inference spec")

    def _get_serve_setting(self):
        """Placeholder docstring"""
        return _ServeSettings(
            role_arn=self.role_arn,
            s3_model_data_url=self.s3_model_data_url,
            instance_type=self.instance_type,
            env_vars=self.env_vars,
            sagemaker_session=self.sagemaker_session,
        )

    def _prepare_for_mode(
        self, model_path: Optional[str] = None, should_upload_artifacts: Optional[bool] = False
    ):
        """Prepare this `Model` for serving.

        Args:
            model_path (Optional[str]): Model path
            should_upload_artifacts (Optional[bool]): Whether to upload artifacts to S3.
        """
        # TODO: move mode specific prepare steps under _model_builder_deploy_wrapper
        self.s3_upload_path = None
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            # init the SageMakerEndpointMode object
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
            self.env_vars.update(env_vars_sagemaker)
            return self.s3_upload_path, env_vars_sagemaker
        elif self.mode == Mode.LOCAL_CONTAINER:
            # init the LocalContainerMode object
            self.modes[str(Mode.LOCAL_CONTAINER)] = LocalContainerMode(
                inference_spec=self.inference_spec,
                schema_builder=self.schema_builder,
                session=self.sagemaker_session,
                model_path=self.model_path,
                env_vars=self.env_vars,
                model_server=self.model_server,
            )
            self.modes[str(Mode.LOCAL_CONTAINER)].prepare()
            return None
        elif self.mode == Mode.IN_PROCESS:
            # init the InProcessMode object
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
            "Please specify mode in: %s, %s, %s"
            % (Mode.LOCAL_CONTAINER, Mode.SAGEMAKER_ENDPOINT, Mode.IN_PROCESS)
        )

    def _get_client_translators(self):
        """Placeholder docstring"""
        serializer = None
        if self.content_type == "application/x-npy":
            serializer = NumpySerializer()
        elif self.content_type == "tensor/pt":
            serializer = TorchTensorSerializer()
        elif self.schema_builder and hasattr(self.schema_builder, "custom_input_translator"):
            serializer = self.schema_builder.custom_input_translator
        elif self.schema_builder:
            serializer = self.schema_builder.input_serializer
        else:
            raise Exception("Cannot serialize. Try providing a SchemaBuilder if not present.")

        deserializer = None
        if self.accept_type == "application/json":
            deserializer = JSONDeserializer()
        elif self.accept_type == "tensor/pt":
            deserializer = TorchTensorDeserializer()
        elif self.schema_builder and hasattr(self.schema_builder, "custom_output_translator"):
            deserializer = self.schema_builder.custom_output_translator
        elif self.schema_builder:
            deserializer = self.schema_builder.output_deserializer
        else:
            raise Exception("Cannot deserialize. Try providing a SchemaBuilder if not present.")

        return serializer, deserializer

    def _get_predictor(
        self, endpoint_name: str, sagemaker_session: Session, component_name: Optional[str] = None
    ) -> Predictor:
        """Placeholder docstring"""
        serializer, deserializer = self._get_client_translators()

        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
            component_name=component_name,
        )

    def _create_model(self):
        """Placeholder docstring"""
        # TODO: we should create model as per the framework
        self.pysdk_model = Model(
            image_uri=self.image_uri,
            image_config=self.image_config,
            vpc_config=self.vpc_config,
            model_data=self.s3_upload_path,
            role=self.serve_settings.role_arn,
            env=self.env_vars,
            sagemaker_session=self.sagemaker_session,
            predictor_cls=self._get_predictor,
            name=self.name,
        )

        # store the modes in the model so that we may
        # reference the configurations for local deploy() & predict()
        self.pysdk_model.mode = self.mode
        self.pysdk_model.modes = self.modes
        self.pysdk_model.serve_settings = self.serve_settings
        if self.role_arn:
            self.pysdk_model.role = self.role_arn
        if self.sagemaker_session:
            self.pysdk_model.sagemaker_session = self.sagemaker_session

        # dynamically generate a method to direct model.deploy() logic based on mode
        # unique method to models created via ModelBuilder()
        self._original_deploy = self.pysdk_model.deploy
        self.pysdk_model.deploy = self._model_builder_deploy_wrapper
        self._original_register = self.pysdk_model.register
        self.pysdk_model.register = self._model_builder_register_wrapper
        self.model_package = None
        return self.pysdk_model

    @_capture_telemetry("register")
    def _model_builder_register_wrapper(self, *args, **kwargs):
        """Placeholder docstring"""
        serializer, deserializer = self._get_client_translators()
        if "content_types" not in kwargs:
            self.pysdk_model.content_types = serializer.CONTENT_TYPE.split()
        if "response_types" not in kwargs:
            self.pysdk_model.response_types = deserializer.ACCEPT.split()
        new_model_package = self._original_register(*args, **kwargs)
        self.pysdk_model.model_package_arn = new_model_package.model_package_arn
        new_model_package.deploy = self._model_builder_deploy_model_package_wrapper
        self.model_package = new_model_package
        if getattr(self, "_is_mlflow_model", False) and self.mode == Mode.SAGEMAKER_ENDPOINT:
            _maintain_lineage_tracking_for_mlflow_model(
                mlflow_model_path=self.model_metadata[MLFLOW_MODEL_PATH],
                s3_upload_path=self.s3_upload_path,
                sagemaker_session=self.sagemaker_session,
                tracking_server_arn=self.model_metadata.get(MLFLOW_TRACKING_ARN),
            )
        return new_model_package

    def _model_builder_deploy_model_package_wrapper(self, *args, **kwargs):
        """Placeholder docstring"""
        if self.pysdk_model.model_package_arn is not None:
            return self._model_builder_deploy_wrapper(*args, **kwargs)

        # need to set the model_package_arn
        # so that the model is created using the model_package's configs
        self.pysdk_model.model_package_arn = self.model_package.model_package_arn
        predictor = self._model_builder_deploy_wrapper(*args, **kwargs)
        self.pysdk_model.model_package_arn = None
        return predictor

    def _deploy_for_ic(
        self,
        *args,
        ic_data: Dict[str, Any],
        container_timeout_in_seconds: int = 300,
        model_data_download_timeout: int = 3600,
        instance_type: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        endpoint_name: Optional[str] = None,
        **kwargs,
    ) -> Predictor:
        """Creates an Inference Component from a ModelBuilder."""
        ic_name = ic_data.get("Name", None)
        model = ic_data.get("Model", None)
        resource_requirements = ic_data.get("ResourceRequirements", {})

        # Ensure resource requirements are set for non-JumpStart models
        if not resource_requirements:
            raise ValueError(
                f"Cannot create/update inference component {ic_name} without resource requirements."
            )

        # Check if the Inference Component exists
        if ic_name and self._does_ic_exist(ic_name=ic_name):
            logger.info("Updating Inference Component %s as it already exists.", ic_name)

            # Create spec for updating the IC
            startup_parameters = {}
            if model_data_download_timeout is not None:
                startup_parameters["ModelDataDownloadTimeoutInSeconds"] = (
                    model_data_download_timeout
                )
            if container_timeout_in_seconds is not None:
                startup_parameters["ContainerStartupHealthCheckTimeoutInSeconds"] = (
                    container_timeout_in_seconds
                )
            compute_rr = resource_requirements.get_compute_resource_requirements()
            inference_component_spec = {
                "ModelName": self.name,
                "StartupParameters": startup_parameters,
                "ComputeResourceRequirements": compute_rr,
            }
            runtime_config = {"CopyCount": resource_requirements.copy_count}
            response = self.sagemaker_session.update_inference_component(
                inference_component_name=ic_name,
                specification=inference_component_spec,
                runtime_config=runtime_config,
            )
            return Predictor(endpoint_name=response.get("EndpointName"), component_name=ic_name)
        else:
            kwargs.update(
                {
                    "resources": resource_requirements,
                    "endpoint_type": EndpointType.INFERENCE_COMPONENT_BASED,
                    "inference_component_name": ic_name,
                    "endpoint_logging": False,
                }
            )
            return model.deploy(
                *args,
                container_startup_health_check_timeout=container_timeout_in_seconds,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                mode=Mode.SAGEMAKER_ENDPOINT,
                endpoint_name=endpoint_name,
                **kwargs,
            )

    def _does_ic_exist(self, ic_name: str) -> bool:
        """Returns true if an Inference Component exists with the given name."""
        try:
            self.sagemaker_session.describe_inference_component(inference_component_name=ic_name)
            return True
        except ClientError as e:
            msg = e.response["Error"]["Message"]
            return "Could not find inference component" not in msg

    @_capture_telemetry("torchserve.deploy")
    def _model_builder_deploy_wrapper(
        self,
        *args,
        container_timeout_in_second: int = 300,
        instance_type: str = None,
        initial_instance_count: int = None,
        mode: str = None,
        **kwargs,
    ) -> Type[PredictorBase]:
        """Placeholder docstring"""
        if mode and mode != self.mode:
            self._overwrite_mode_in_deploy(overwrite_mode=mode)

        if self.mode == Mode.IN_PROCESS:
            serializer, deserializer = self._get_client_translators()

            predictor = _get_in_process_mode_predictor(
                self.modes[str(Mode.IN_PROCESS)], serializer, deserializer
            )

            self.modes[str(Mode.IN_PROCESS)].create_server(
                predictor,
            )
            return predictor

        if self.mode == Mode.LOCAL_CONTAINER:
            serializer, deserializer = self._get_client_translators()
            predictor = _get_local_mode_predictor(
                mode_obj=self.modes[str(Mode.LOCAL_CONTAINER)],
                model_server=self.model_server,
                serializer=serializer,
                deserializer=deserializer,
            )

            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri, container_timeout_in_second, self.secret_key, predictor
            )
            return predictor

        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            # Validate parameters
            # Instance type and instance count parameter validation is done based on deployment type
            # and will be done inside Model.deploy()
            if is_1p_image_uri(image_uri=self.image_uri):
                validate_image_uri_and_hardware(
                    image_uri=self.image_uri,
                    instance_type=instance_type,
                    model_server=self.model_server,
                )

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True

        if "inference_component_name" not in kwargs and self.inference_component_name:
            kwargs["inference_component_name"] = self.inference_component_name

        if "resources" not in kwargs and self.resource_requirements:
            kwargs["resources"] = self.resource_requirements

        kwargs.pop("mode", None)
        self.pysdk_model.role = kwargs.pop("role", self.pysdk_model.role)
        predictor = self._original_deploy(
            *args,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            **kwargs,
        )
        if getattr(self, "_is_mlflow_model", False) and self.mode == Mode.SAGEMAKER_ENDPOINT:
            _maintain_lineage_tracking_for_mlflow_model(
                mlflow_model_path=self.model_metadata[MLFLOW_MODEL_PATH],
                s3_upload_path=self.s3_upload_path,
                sagemaker_session=self.sagemaker_session,
                tracking_server_arn=self.model_metadata.get(MLFLOW_TRACKING_ARN),
            )
        return predictor

    def _overwrite_mode_in_deploy(self, overwrite_mode: str):
        """Mode overwritten by customer during model.deploy()"""
        logger.warning(
            "Deploying in %s Mode, overriding existing configurations set for %s mode",
            overwrite_mode,
            self.mode,
        )
        if overwrite_mode == Mode.SAGEMAKER_ENDPOINT:
            self.mode = self.pysdk_model.mode = Mode.SAGEMAKER_ENDPOINT
            s3_upload_path, env_vars_sagemaker = self._prepare_for_mode()
            self.pysdk_model.model_data = s3_upload_path
            self.pysdk_model.env.update(env_vars_sagemaker)
        elif overwrite_mode == Mode.LOCAL_CONTAINER:
            self.mode = self.pysdk_model.mode = Mode.LOCAL_CONTAINER
            self._prepare_for_mode()
        elif overwrite_mode == Mode.IN_PROCESS:
            self.mode = self.pysdk_model.mode = Mode.IN_PROCESS
            self._prepare_for_mode()
        else:
            raise ValueError("Mode %s is not supported!" % overwrite_mode)

    def _build_for_torchserve(self) -> Type[Model]:
        """Build the model for torchserve"""
        self._save_model_inference_spec()

        if self.mode != Mode.IN_PROCESS:
            self._auto_detect_container()

            self.secret_key = prepare_for_torchserve(
                model_path=self.model_path,
                shared_libs=self.shared_libs,
                dependencies=self.dependencies,
                session=self.sagemaker_session,
                image_uri=self.image_uri,
                inference_spec=self.inference_spec,
            )

        self._prepare_for_mode()
        self.model = self._create_model()
        return self.model

    def _build_for_smd(self) -> Type[Model]:
        """Build the model for SageMaker Distribution"""
        self._save_model_inference_spec()

        if self.mode != Mode.IN_PROCESS:
            self._auto_detect_container()

            self.secret_key = prepare_for_smd(
                model_path=self.model_path,
                shared_libs=self.shared_libs,
                dependencies=self.dependencies,
                inference_spec=self.inference_spec,
            )

        self._prepare_for_mode()
        self.model = self._create_model()
        return self.model

    def _user_agent_decorator(self, func):
        """Placeholder docstring"""

        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            if "ModelBuilder" in result:
                return result
            return result + " ModelBuilder"

        return wrapper

    def _handle_mlflow_input(self):
        """Check whether an MLflow model is present and handle accordingly"""
        self._is_mlflow_model = self._has_mlflow_arguments()
        if not self._is_mlflow_model:
            return

        mlflow_model_path = self.model_metadata.get(MLFLOW_MODEL_PATH)
        artifact_path = self._get_artifact_path(mlflow_model_path)
        if not self._mlflow_metadata_exists(artifact_path):
            return

        self._initialize_for_mlflow(artifact_path)
        _validate_input_for_mlflow(self.model_server, self.env_vars.get("MLFLOW_MODEL_FLAVOR"))

    def _has_mlflow_arguments(self) -> bool:
        """Check whether MLflow model arguments are present

        Returns:
            bool: True if MLflow arguments are present, False otherwise.
        """
        if self.inference_spec or self.model:
            logger.info(
                "Either inference spec or model is provided. "
                "ModelBuilder is not handling MLflow model input"
            )
            return False

        if not self.model_metadata:
            logger.info(
                "No ModelMetadata provided. ModelBuilder is not handling MLflow model input"
            )
            return False

        mlflow_model_path = self.model_metadata.get(MLFLOW_MODEL_PATH)
        if not mlflow_model_path:
            logger.info(
                "%s is not provided in ModelMetadata. ModelBuilder is not handling MLflow model "
                "input",
                MLFLOW_MODEL_PATH,
            )
            return False

        return True

    def _get_artifact_path(self, mlflow_model_path: str) -> str:
        """Retrieves the model artifact location given the Mlflow model input.

        Args:
            mlflow_model_path (str): The MLflow model path input.

        Returns:
            str: The path to the model artifact.
        """
        if (is_run_id_type := re.match(MLFLOW_RUN_ID_REGEX, mlflow_model_path)) or re.match(
            MLFLOW_REGISTRY_PATH_REGEX, mlflow_model_path
        ):
            mlflow_tracking_arn = self.model_metadata.get(MLFLOW_TRACKING_ARN)
            if not mlflow_tracking_arn:
                raise ValueError(
                    "%s is not provided in ModelMetadata or through set_tracking_arn "
                    "but MLflow model path was provided." % MLFLOW_TRACKING_ARN,
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

        if re.match(MODEL_PACKAGE_ARN_REGEX, mlflow_model_path):
            model_package = self.sagemaker_session.sagemaker_client.describe_model_package(
                ModelPackageName=mlflow_model_path
            )
            return model_package["SourceUri"]

        return mlflow_model_path

    def _mlflow_metadata_exists(self, path: str) -> bool:
        """Checks whether an MLmodel file exists in the given directory.

        Returns:
            bool: True if the MLmodel file exists, False otherwise.
        """
        if path.startswith("s3://"):
            s3_downloader = S3Downloader()
            if not path.endswith("/"):
                path += "/"
            s3_uri_to_mlmodel_file = f"{path}{MLFLOW_METADATA_FILE}"
            response = s3_downloader.list(s3_uri_to_mlmodel_file, self.sagemaker_session)
            return len(response) > 0

        file_path = os.path.join(path, MLFLOW_METADATA_FILE)
        return os.path.isfile(file_path)

    def _initialize_for_mlflow(self, artifact_path: str) -> None:
        """Initialize mlflow model artifacts, image uri and model server.

        Args:
            artifact_path (str): The path to the artifact store.
        """
        if artifact_path.startswith("s3://"):
            _download_s3_artifacts(artifact_path, self.model_path, self.sagemaker_session)
        elif os.path.exists(artifact_path):
            _copy_directory_contents(artifact_path, self.model_path)
        else:
            raise ValueError("Invalid path: %s" % artifact_path)
        mlflow_model_metadata_path = _generate_mlflow_artifact_path(
            self.model_path, MLFLOW_METADATA_FILE
        )
        # TODO: add validation on MLmodel file
        mlflow_model_dependency_path = _generate_mlflow_artifact_path(
            self.model_path, MLFLOW_PIP_DEPENDENCY_FILE
        )
        flavor_metadata = _get_all_flavor_metadata(mlflow_model_metadata_path)
        deployment_flavor = _get_deployment_flavor(flavor_metadata)

        self.model_server = self.model_server or _get_default_model_server_for_mlflow(
            deployment_flavor
        )
        self.image_uri = self.image_uri or _select_container_for_mlflow_model(
            mlflow_model_src_path=self.model_path,
            deployment_flavor=deployment_flavor,
            region=self.sagemaker_session.boto_region_name,
            instance_type=self.instance_type,
        )
        self.env_vars.update({"MLFLOW_MODEL_FLAVOR": f"{deployment_flavor}"})
        self.dependencies.update({"requirements": mlflow_model_dependency_path})

    @_capture_telemetry("ModelBuilder.build_training_job")
    def _collect_training_job_model_telemetry(self):
        """Dummy method to collect telemetry for training job handshake"""
        return

    @_capture_telemetry("ModelBuilder.build_model_trainer")
    def _collect_model_trainer_model_telemetry(self):
        """Dummy method to collect telemetry for model trainer handshake"""
        return

    @_capture_telemetry("ModelBuilder.build_estimator")
    def _collect_estimator_model_telemetry(self):
        """Dummy method to collect telemetry for estimator handshake"""
        return

    def build(
        self,
        mode: Type[Mode] = None,
        role_arn: str = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Union[ModelBuilder, Type[Model]]:
        """Creates deployable ``Model`` instances with all provided ``ModelBuilder`` objects.

        Args:
            mode (Type[Mode], optional): The mode. Defaults to ``None``.
            role_arn (str, optional): The IAM role arn. Defaults to ``None``.
            sagemaker_session (Optional[Session]): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Union[ModelBuilder, Type[Model]]: A deployable ``ModelBuilder`` object if multiple
            ``ModelBuilders`` were built, or a deployable ``Model`` object.
        """
        if role_arn:
            self.role_arn = role_arn
        self.sagemaker_session = sagemaker_session or self.sagemaker_session or Session()

        deployables = {}

        if not self.modelbuilder_list and not isinstance(
            self.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)
        ):
            self.serve_settings = self._get_serve_setting()
            return self._build_single_modelbuilder(
                mode=mode,
                role_arn=self.role_arn,
                sagemaker_session=sagemaker_session,
            )

        # Multi-ModelBuilder case: deploy
        built_ic_models = []
        if self.modelbuilder_list:
            logger.info("Detected ModelBuilders in modelbuilder_list.")
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
                # Custom orchestrator definition found in inference_spec
                mb.serve_settings = mb._get_serve_setting()
                # Build for Inference Component
                logger.info("Building ModelBuilder %s.", mb.name)
                # Get JS deployment configs if ResourceRequirements not set

                mb = mb._get_ic_resource_requirements(mb=mb)

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
                logger.info(
                    "=====================Build for %s complete.===================",
                    mb.model,
                )
            deployables["InferenceComponents"] = built_ic_models

        if isinstance(self.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)):
            logger.info("Building custom orchestrator.")
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
                # Network isolation of ICs on an endpoint must be consistent
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

            logger.info(
                "=====================Custom orchestrator build complete.===================",
            )

        self._deployables = deployables
        return self

    def _get_processing_unit(self):
        """Detects if the resource requirements are intended for a CPU or GPU instance."""
        # Assume custom orchestrator will be deployed as an endpoint to a CPU instance
        if not self.resource_requirements or not self.resource_requirements.num_accelerators:
            return "cpu"
        for ic in self.modelbuilder_list or []:
            if ic.resource_requirements.num_accelerators > 0:
                return "gpu"
        if self.resource_requirements.num_accelerators > 0:
            return "gpu"

        return "cpu"

    def _get_ic_resource_requirements(self, mb: ModelBuilder = None) -> ModelBuilder:
        """Attempts fetching pre-benchmarked resource requirements for the MB from JumpStart."""
        if mb._is_jumpstart_model_id() and not mb.resource_requirements:
            js_model = JumpStartModel(model_id=mb.model)
            deployment_configs = js_model.list_deployment_configs()
            if not deployment_configs:
                raise ValueError(
                    "No resource requirements were provided for Inference Component "
                    f"{mb.inference_component_name} and no default deployment"
                    " configs were found in JumpStart."
                )
            compute_requirements = (
                deployment_configs[0].get("DeploymentArgs").get("ComputeResourceRequirements")
            )
            logger.info("Retrieved pre-benchmarked deployment configurations from JumpStart.")
            mb.resource_requirements = ResourceRequirements(
                requests={
                    "memory": compute_requirements["MinMemoryRequiredInMb"],
                    "num_accelerators": compute_requirements.get(
                        "NumberOfAcceleratorDevicesRequired", None
                    ),
                    "copies": 1,
                    "num_cpus": compute_requirements.get("NumberOfCpuCoresRequired", None),
                },
                limits={"memory": compute_requirements.get("MaxMemoryRequiredInMb", None)},
            )

        return mb

    @_capture_telemetry("build_custom_orchestrator")
    def _get_smd_image_uri(self, processing_unit: str = None) -> str:
        """Gets the SMD Inference Image URI.

        Returns:
            str: SMD Inference Image URI.
        """
        from sagemaker import image_uris
        import sys

        self.sagemaker_session = self.sagemaker_session or Session()
        from packaging.version import Version

        formatted_py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        if Version(f"{sys.version_info.major}{sys.version_info.minor}") < Version("3.12"):
            raise ValueError(
                f"Found Python version {formatted_py_version} but"
                f"Custom orchestrator deployment requires Python version >= 3.12."
            )

        INSTANCE_TYPES = {"cpu": "ml.c5.xlarge", "gpu": "ml.g5.4xlarge"}

        logger.info("Finding SMD inference image URI for a %s instance.", processing_unit)

        smd_uri = image_uris.retrieve(
            framework="sagemaker-distribution",
            image_scope="inference",
            instance_type=INSTANCE_TYPES[processing_unit],
            region=self.sagemaker_session.boto_region_name,
        )
        logger.info("Found compatible image %s", smd_uri)
        return smd_uri

    # Model Builder is a class to build the model for deployment.
    # It supports three modes of deployment
    # 1/ SageMaker Endpoint
    # 2/ Local launch with container
    # 3/ In process mode with Transformers server in beta release
    @_capture_telemetry("ModelBuilder.build")
    def _build_single_modelbuilder(  # pylint: disable=R0911
        self,
        mode: Type[Mode] = None,
        role_arn: str = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Type[Model]:
        """Create a deployable ``Model`` instance with ``ModelBuilder``.

        Args:
            mode (Type[Mode], optional): The mode. Defaults to ``None``.
            role_arn (str, optional): The IAM role arn. Defaults to ``None``.
            sagemaker_session (Optional[Session]): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Type[Model]: A deployable ``Model`` object.
        """

        self.modes = dict()

        if mode:
            self.mode = mode
        if role_arn:
            self.role_arn = role_arn

        self.serve_settings = self._get_serve_setting()

        if isinstance(self.model, TrainingJob):
            self.model_path = self.model.model_artifacts.s3_model_artifacts
            self.model = None
            self._collect_training_job_model_telemetry()
        elif isinstance(self.model, ModelTrainer):
            self.model_path = self.model._latest_training_job.model_artifacts.s3_model_artifacts
            self.model = None
            self._collect_model_trainer_model_telemetry()
        elif isinstance(self.model, Estimator):
            self.model_path = self.model.output_path
            self.model = None
            self._collect_estimator_model_telemetry()

        self.sagemaker_session = sagemaker_session or self.sagemaker_session or Session()

        self.sagemaker_session.settings._local_download_dir = self.model_path

        # DJL expects `HF_TOKEN` key. This allows backward compatibility
        # until we deprecate HUGGING_FACE_HUB_TOKEN.
        if self.env_vars.get("HUGGING_FACE_HUB_TOKEN") and not self.env_vars.get("HF_TOKEN"):
            self.env_vars["HF_TOKEN"] = self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
        elif self.env_vars.get("HF_TOKEN") and not self.env_vars.get("HUGGING_FACE_HUB_TOKEN"):
            self.env_vars["HUGGING_FACE_HUB_TOKEN"] = self.env_vars.get("HF_TOKEN")

        self.sagemaker_session.settings._local_download_dir = self.model_path

        # https://github.com/boto/botocore/blob/develop/botocore/useragent.py#L258
        # decorate to_string() due to
        # https://github.com/boto/botocore/blob/develop/botocore/client.py#L1014-L1015
        client = self.sagemaker_session.sagemaker_client
        client._user_agent_creator.to_string = self._user_agent_decorator(
            self.sagemaker_session.sagemaker_client._user_agent_creator.to_string
        )

        self._is_custom_image_uri = self.image_uri is not None

        self._handle_mlflow_input()

        self._build_validations()

        if (
            not (isinstance(self.model, str) and self._is_jumpstart_model_id())
        ) and self.model_server:
            self.built_model = self._build_for_model_server()
            return self.built_model

        if isinstance(self.model, str):
            model_task = None

            if self._is_jumpstart_model_id():
                if self.mode == Mode.IN_PROCESS:
                    raise ValueError(
                        f"{self.mode} is not supported for Jumpstart models. "
                        "Please use LOCAL_CONTAINER mode to deploy a Jumpstart model"
                        " on your local machine."
                    )
                self.model_hub = ModelHub.JUMPSTART
                logger.debug("Building for Jumpstart model Id...")
                self.built_model = self._build_for_jumpstart()
                return self.built_model

            if self.mode != Mode.IN_PROCESS:
                if self._use_jumpstart_equivalent():
                    self.model_hub = ModelHub.JUMPSTART
                    logger.debug("Building for Jumpstart equiavalent model Id...")
                    self.built_model = self._build_for_jumpstart()
                    return self.built_model
            self.model_hub = ModelHub.HUGGINGFACE

            if self.model_metadata:
                model_task = self.model_metadata.get("HF_TASK")

            if self._is_djl():
                return self._build_for_djl()
            else:
                hf_model_md = get_huggingface_model_metadata(
                    self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )

                if model_task is None:
                    model_task = hf_model_md.get("pipeline_tag")
                if self.schema_builder is None and model_task is not None:
                    self._hf_schema_builder_init(model_task)
                if model_task == "text-generation":
                    self.built_model = self._build_for_tgi()
                    return self.built_model
                if model_task in ["sentence-similarity", "feature-extraction"]:
                    self.built_model = self._build_for_tei()
                    return self.built_model
                elif self._can_fit_on_single_gpu():
                    self.built_model = self._build_for_transformers()
                    return self.built_model
                else:
                    self.built_model = self._build_for_transformers()
                    return self.built_model

        # Set TorchServe as default model server
        if not self.model_server:
            self.model_server = ModelServer.TORCHSERVE
            self.built_model = self._build_for_torchserve()
            return self.built_model

        raise ValueError("%s model server is not supported" % self.model_server)

    def _build_validations(self):
        """Validations needed for model server overrides, or auto-detection or fallback"""
        if self.inference_spec and self.model:
            raise ValueError("Can only set one of the following: model, inference_spec.")

        if self.image_uri and not is_1p_image_uri(self.image_uri) and self.model_server is None:
            raise ValueError(
                "Model_server must be set when non-first-party image_uri is set. "
                + "Supported model servers: %s" % supported_model_servers
            )

    def _build_for_model_server(self):  # pylint: disable=R0911, R1710
        """Model server overrides"""
        if self.model_server not in supported_model_servers:
            raise ValueError(
                "%s is not supported yet! Supported model servers: %s"
                % (self.model_server, supported_model_servers)
            )

        mlflow_path = None
        if self.model_metadata:
            mlflow_path = self.model_metadata.get(MLFLOW_MODEL_PATH)

        if not self.model and not mlflow_path and not self.inference_spec:
            raise ValueError("Missing required parameter `model` or 'ml_flow' path or inf_spec")

        if self.model_server == ModelServer.TORCHSERVE:
            return self._build_for_torchserve()

        if self.model_server == ModelServer.TRITON:
            return self._build_for_triton()

        if self.model_server == ModelServer.TENSORFLOW_SERVING:
            return self._build_for_tensorflow_serving()

        if self.model_server == ModelServer.DJL_SERVING:
            return self._build_for_djl()

        if self.model_server == ModelServer.TEI:
            return self._build_for_tei()

        if self.model_server == ModelServer.TGI:
            return self._build_for_tgi()

        if self.model_server == ModelServer.MMS:
            return self._build_for_transformers()

        if self.model_server == ModelServer.SMD:
            return self._build_for_smd()

    @_capture_telemetry("ModelBuilder.save")
    def save(
        self,
        save_path: Optional[str] = None,
        s3_path: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        role_arn: Optional[str] = None,
    ) -> Type[Model]:
        """WARNING: This function is expremental and not intended for production use.

        This function is available for models served by DJL serving.

        Args:
            save_path (Optional[str]): The path where you want to save resources. Defaults to
                ``None``.
            s3_path (Optional[str]): The path where you want to upload resources. Defaults to
                ``None``.
            sagemaker_session (Optional[Session]): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain. Defaults to
                ``None``.
            role_arn (Optional[str]): The IAM role arn. Defaults to ``None``.
        """
        self.sagemaker_session = sagemaker_session or Session()

        if role_arn:
            self.role_arn = role_arn

        save_handler = SaveHandler(
            model=self.model,
            schema_builder=self.schema_builder,
            model_loader_path=self.model_path,
            inference_spec=self.inference_spec,
            save_path=save_path,
            s3_path=s3_path,
            sagemaker_session=self.sagemaker_session,
            role_arn=self.role_arn,
            metadata=Metadata(),
        )

        return save_handler.save()

    def validate(self, model_dir: str) -> Type[bool]:
        """WARNING: This function is expremental and not intended for production use.

        The intention of this function is to validate the model in a format that can be
        used across the model servers. This function is not intended to be used for
        production use now and is subject to change.
        """

        return get_metadata(model_dir)

    def set_tracking_arn(self, arn: str):
        """Set tracking server ARN"""
        # TODO: support native MLflow URIs
        if importlib.util.find_spec("sagemaker_mlflow"):
            import mlflow

            mlflow.set_tracking_uri(arn)
            self.model_metadata[MLFLOW_TRACKING_ARN] = arn
        else:
            raise ImportError(
                "Unable to import sagemaker_mlflow, check if sagemaker_mlflow is installed"
            )

    def _hf_schema_builder_init(self, model_task: str):
        """Initialize the schema builder for the given HF_TASK

        Args:
            model_task (str): Required, the task name

        Raises:
            TaskNotFoundException: If the I/O schema for the given task is not found.
        """
        try:
            try:
                sample_inputs, sample_outputs = task.retrieve_local_schemas(model_task)
            except ValueError:
                # samples could not be loaded locally, try to fetch remote hf schema
                from sagemaker_schema_inference_artifacts.huggingface import remote_schema_retriever

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
        except ValueError:
            raise TaskNotFoundException(
                f"HuggingFace Schema builder samples for {model_task} could not be found "
                f"locally or via remote."
            )

    def _can_fit_on_single_gpu(self) -> Type[bool]:
        """Check if model can fit on a single GPU

        If the size of the model is <= single gpu memory size, returns True else False
        """
        try:
            single_gpu_size_mib = self._try_fetch_gpu_info()
            if (
                _total_inference_model_size_mib(self.model, self.env_vars.get("dtypes", "float32"))
                <= single_gpu_size_mib
            ):
                logger.info(
                    "Total inference model size MIB %s, single GPU size for instance MIB %s",
                    _total_inference_model_size_mib(
                        self.model, self.env_vars.get("dtypes", "float32")
                    ),
                    single_gpu_size_mib,
                )
                return True
            return False
        except ValueError:
            logger.info("Unable to determine single GPU size for instance %s", self.instance_type)
            return False

    def _try_fetch_gpu_info(self):
        """Get GPU info

        This function gets the GPU info or fallback to set the size of a single GPU
        """
        try:
            gpu_info = _get_gpu_info(self.instance_type, self.sagemaker_session)
            logger.info("GPU info %s for instance %s", gpu_info, self.instance_type)
            return gpu_info[1] / gpu_info[0]
        except ValueError:
            pass
        try:
            gpu_fallback = _get_gpu_info_fallback(
                self.instance_type, self.sagemaker_session.boto_region_name
            )
            logger.info("GPU fallback picked up %s", gpu_fallback)
            return gpu_fallback[1] / gpu_fallback[0]
        except ValueError:
            raise ValueError(
                f"Unable to determine single GPU size for instance: [{self.instance_type}]"
            )

    def optimize(
        self,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        role_arn: Optional[str] = None,
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
        max_runtime_in_sec: Optional[int] = 36000,
        sagemaker_session: Optional[Session] = None,
    ) -> Model:
        """Create an optimized deployable ``Model`` instance with ``ModelBuilder``.

        Args:
            output_path (str): Specifies where to store the compiled/quantized model.
            instance_type (str): Target deployment instance type that the model is optimized for.
            role_arn (Optional[str]): Execution role arn. Defaults to ``None``.
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
                36000 seconds.
            sagemaker_session (Optional[Session]): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Model: A deployable ``Model`` object.
        """

        # need to get telemetry_opt_out info before telemetry decorator is called
        self.serve_settings = self._get_serve_setting()

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
            max_runtime_in_sec=max_runtime_in_sec,
            sagemaker_session=sagemaker_session,
        )

    @_capture_telemetry("optimize")
    def _model_builder_optimize_wrapper(
        self,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        role_arn: Optional[str] = None,
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
        max_runtime_in_sec: Optional[int] = 36000,
        sagemaker_session: Optional[Session] = None,
    ) -> Model:
        """Runs a model optimization job.

        Args:
            output_path (str): Specifies where to store the compiled/quantized model.
            instance_type (str): Target deployment instance type that the model is optimized for.
            role_arn (Optional[str]): Execution role arn. Defaults to ``None``.
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
                36000 seconds.
            sagemaker_session (Optional[Session]): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Model: A deployable ``Model`` object.
        """
        if (
            hasattr(self, "enable_network_isolation")
            and self.enable_network_isolation
            and sharding_config
        ):
            raise ValueError(
                "EnableNetworkIsolation cannot be set to True since SageMaker Fast Model "
                "Loading of model requires network access."
            )

        # TODO: ideally these dictionaries need to be sagemaker_core shapes
        # TODO: for organization, abstract all validation behind this fn
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
        self.speculative_decoding_draft_model_source = _extract_speculative_draft_model_provider(
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

        self.sagemaker_session = sagemaker_session or self.sagemaker_session or Session()
        self.instance_type = instance_type or self.instance_type
        self.role_arn = role_arn or self.role_arn

        job_name = job_name or f"modelbuilderjob-{uuid.uuid4().hex}"
        if self._is_jumpstart_model_id():
            self.build(mode=self.mode, sagemaker_session=self.sagemaker_session)
            if self.pysdk_model:
                self.pysdk_model.set_deployment_config(
                    instance_type=instance_type, config_name="lmi"
                )
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

            self.build(mode=self.mode, sagemaker_session=self.sagemaker_session)
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
            self.pysdk_model._is_sharded_model = True

        if input_args:
            optimization_instance_type = input_args["DeploymentInstanceType"]

            # Compilation using TRTLLM and Llama-3.1 is currently not supported.
            # TRTLLM is used by Neo if the following are provided:
            #  1) a GPU instance type
            #  2) compilation config
            gpu_instance_families = ["g5", "g6", "p4d", "p4de", "p5"]
            is_gpu_instance = optimization_instance_type and any(
                gpu_instance_family in optimization_instance_type
                for gpu_instance_family in gpu_instance_families
            )

            # HF Model ID format = "meta-llama/Meta-Llama-3.1-8B"
            # JS Model ID format = "meta-textgeneration-llama-3-1-8b"
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

            self.sagemaker_session.sagemaker_client.create_optimization_job(**input_args)
            job_status = self.sagemaker_session.wait_for_optimization_job(job_name)
            return _generate_optimized_model(self.pysdk_model, job_status)

        self.pysdk_model.remove_tag_with_key(Tag.OPTIMIZATION_JOB_NAME)
        if not speculative_decoding_config:
            self.pysdk_model.remove_tag_with_key(Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER)

        return self.pysdk_model

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
                _jumpstart_speculative_decoding(
                    model=self.pysdk_model,
                    speculative_decoding_config=speculative_decoding_config,
                    sagemaker_session=self.sagemaker_session,
                )
            else:
                self.pysdk_model = _custom_speculative_decoding(
                    self.pysdk_model, speculative_decoding_config, False
                )

        if quantization_config or compilation_config or sharding_config:
            create_optimization_job_args = {
                "OptimizationJobName": job_name,
                "DeploymentInstanceType": self.instance_type,
                "RoleArn": self.role_arn,
            }

            if env_vars:
                self.pysdk_model.env.update(env_vars)
                create_optimization_job_args["OptimizationEnvironment"] = env_vars

            self._optimize_prepare_for_hf()
            model_source = _generate_model_source(self.pysdk_model.model_data, False)
            create_optimization_job_args["ModelSource"] = model_source

            (
                optimization_config,
                quantization_override_env,
                compilation_override_env,
                sharding_override_env,
            ) = _extract_optimization_config_and_env(
                quantization_config, compilation_config, sharding_config
            )
            create_optimization_job_args["OptimizationConfigs"] = [
                {k: v} for k, v in optimization_config.items()
            ]
            self.pysdk_model.env.update(
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

            # HF_MODEL_ID needs not to be present, otherwise,
            # HF model artifacts will be re-downloaded during deployment
            if "HF_MODEL_ID" in self.pysdk_model.env:
                del self.pysdk_model.env["HF_MODEL_ID"]

            return create_optimization_job_args
        return None

    def _optimize_prepare_for_hf(self):
        """Prepare huggingface model data for optimization."""
        custom_model_path: str = (
            self.model_metadata.get("CUSTOM_MODEL_PATH") if self.model_metadata else None
        )
        if _is_s3_uri(custom_model_path):
            # Remove slash by the end of s3 uri, as it may lead to / subfolder during upload.
            custom_model_path = (
                custom_model_path[:-1] if custom_model_path.endswith("/") else custom_model_path
            )
        else:
            if not custom_model_path:
                custom_model_path = f"/tmp/sagemaker/model-builder/{self.model}"
                download_huggingface_model_metadata(
                    self.model,
                    os.path.join(custom_model_path, "code"),
                    self.env_vars.get("HUGGING_FACE_HUB_TOKEN"),
                )

        self.pysdk_model.model_data, env = self._prepare_for_mode(
            model_path=custom_model_path,
            should_upload_artifacts=True,
        )
        self.pysdk_model.env.update(env)

    @_capture_telemetry("ModelBuilder.deploy")
    def deploy(
        self,
        endpoint_name: str = None,
        container_timeout_in_second: int = 300,
        instance_type: str = None,
        initial_instance_count: Optional[int] = 1,
        inference_config: Optional[
            Union[
                ServerlessInferenceConfig,
                AsyncInferenceConfig,
                BatchTransformInferenceConfig,
                ResourceRequirements,
            ]
        ] = None,
        update_endpoint: Optional[bool] = False,
        custom_orchestrator_instance_type: str = None,
        custom_orchestrator_initial_instance_count: int = None,
        **kwargs,
    ) -> Union[Predictor, Transformer, List[Predictor]]:
        """Deploys the built Model.

        Depending on the type of config provided, this function will call deployment accordingly.
        Args:
            endpoint_name (str): Name of the endpoint to deploy.
             The supplied base name is used as a prefix and
             a unique ID is appended to guarantee uniqueness.
            initial_instance_count (int): Number of instances to deploy.
            inference_config (Optional[Union[ServerlessInferenceConfig,
               AsyncInferenceConfig, BatchTransformInferenceConfig, ResourceRequirements]]) :
                Additional Config for different deployment types such as
                serverless, async, batch and multi-model/container
            update_endpoint (Optional[bool]):
                Flag to update the model in an existing Amazon SageMaker endpoint.
                If True, this will deploy a new EndpointConfig to an already existing endpoint
                and delete resources corresponding to the previous EndpointConfig. Default: False
                Note: Currently this is supported for single model endpoints
        Returns:
            Transformer for Batch Deployments
            Predictors for all others
        """
        if not hasattr(self, "built_model") and not hasattr(self, "_deployables"):
            raise ValueError("Model needs to be built before deploying")

        if not hasattr(self, "_deployables"):
            if not inference_config:  # Real-time Deployment
                return self.built_model.deploy(
                    instance_type=self.instance_type,
                    initial_instance_count=initial_instance_count,
                    endpoint_name=endpoint_name,
                    update_endpoint=update_endpoint,
                )

            if isinstance(inference_config, ServerlessInferenceConfig):
                return self.built_model.deploy(
                    serverless_inference_config=inference_config,
                    endpoint_name=endpoint_name,
                    update_endpoint=update_endpoint,
                )

            if isinstance(inference_config, AsyncInferenceConfig):
                return self.built_model.deploy(
                    instance_type=self.instance_type,
                    initial_instance_count=initial_instance_count,
                    async_inference_config=inference_config,
                    endpoint_name=endpoint_name,
                    update_endpoint=update_endpoint,
                )

            if isinstance(inference_config, BatchTransformInferenceConfig):
                transformer = self.built_model.transformer(
                    instance_type=inference_config.instance_type,
                    output_path=inference_config.output_path,
                    instance_count=inference_config.instance_count,
                )
                return transformer

        if isinstance(inference_config, ResourceRequirements):
            if update_endpoint:
                raise ValueError(
                    "Currently update_endpoint is supported for single model endpoints"
                )
            # Multi Model and MultiContainer endpoints with Inference Component
            return self.built_model.deploy(
                instance_type=self.instance_type,
                mode=Mode.SAGEMAKER_ENDPOINT,
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=inference_config,
                initial_instance_count=initial_instance_count,
                role=self.role_arn,
                update_endpoint=update_endpoint,
            )

            raise ValueError("Deployment Options not supported")

        # Iterate through deployables for a custom orchestrator deployment.
        # Create all Inference Components first before deploying custom orchestrator if present.
        predictors = []
        for inference_component in self._deployables.get("InferenceComponents", []):
            predictors.append(
                self._deploy_for_ic(
                    ic_data=inference_component,
                    container_timeout_in_seconds=container_timeout_in_second,
                    instance_type=instance_type,
                    initial_instance_count=initial_instance_count,
                    endpoint_name=endpoint_name,
                    **kwargs,
                )
            )
        if self._deployables.get("CustomOrchestrator", None):
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
                predictors.append(
                    custom_orchestrator["Model"].deploy(
                        instance_type=custom_orchestrator_instance_type,
                        initial_instance_count=custom_orchestrator_initial_instance_count,
                        **kwargs,
                    )
                )
            elif custom_orchestrator["Mode"] == "InferenceComponent":
                logger.info(
                    "Deploying custom orchestrator as an inference component "
                    f"to endpoint {endpoint_name}"
                )
                predictors.append(
                    self._deploy_for_ic(
                        ic_data=custom_orchestrator,
                        container_timeout_in_seconds=container_timeout_in_second,
                        instance_type=custom_orchestrator_instance_type or instance_type,
                        initial_instance_count=custom_orchestrator_initial_instance_count
                        or initial_instance_count,
                        endpoint_name=endpoint_name,
                        **kwargs,
                    )
                )

        return predictors

    def display_benchmark_metrics(self, **kwargs):
        """Display Markdown Benchmark Metrics for deployment configs."""
        if not isinstance(self.model, str):
            raise ValueError("Benchmarking is only supported for JumpStart or HuggingFace models")
        if self._is_jumpstart_model_id() or self._use_jumpstart_equivalent():
            return super().display_benchmark_metrics(**kwargs)
        else:
            raise ValueError("This model does not have benchmark metrics yet")

    def get_deployment_config(self) -> Optional[Dict[str, Any]]:
        """Gets the deployment config to apply to the model.

        Returns:
            Optional[Dict[str, Any]]: Deployment config to apply to this model.
        """
        if not isinstance(self.model, str):
            raise ValueError(
                "Deployment config is only supported for JumpStart or HuggingFace models"
            )
        if self._is_jumpstart_model_id() or self._use_jumpstart_equivalent():
            return super().get_deployment_config()
        else:
            raise ValueError("This model does not have any deployment config yet")

    def list_deployment_configs(self) -> List[Dict[str, Any]]:
        """List deployment configs for the model in the current region.

        Returns:
            List[Dict[str, Any]]: A list of deployment configs.
        """
        if not isinstance(self.model, str):
            raise ValueError(
                "Deployment config is only supported for JumpStart or HuggingFace models"
            )
        if self._is_jumpstart_model_id() or self._use_jumpstart_equivalent():
            return super().list_deployment_configs()
        else:
            raise ValueError("This model does not have any deployment config yet")

    def set_deployment_config(self, config_name: str, instance_type: str) -> None:
        """Sets the deployment config to apply to the model.

        Args:
            config_name (str):
                The name of the deployment config to apply to the model.
                Call list_deployment_configs to see the list of config names.
            instance_type (str):
                The instance_type that the model will use after setting
                the config.
        """
        if not isinstance(self.model, str):
            raise ValueError(
                "Deployment config is only supported for JumpStart or HuggingFace models"
            )
        if self._is_jumpstart_model_id() or self._use_jumpstart_equivalent():
            logger.warning(
                "If there are existing deployment configurations, "
                "they will be overwritten by the config %s",
                config_name,
            )
            return super().set_deployment_config(config_name, instance_type)
        else:
            raise ValueError(f"The deployment config {config_name} cannot be set on this model")

    def _use_jumpstart_equivalent(self):
        """Check if the HuggingFace model has a JumpStart equivalent.

        Replace it with the equivalent if there's one
        """
        # Do not use the equivalent JS model if image_uri or env_vars is provided
        if self.image_uri or self.env_vars:
            return False
        if not hasattr(self, "_has_jumpstart_equivalent"):
            self._jumpstart_mapping = self._retrieve_hugging_face_model_mapping()
            self._has_jumpstart_equivalent = self.model in self._jumpstart_mapping
        if self._has_jumpstart_equivalent:
            # Use schema builder from HF model metadata
            if not self.schema_builder:
                model_task = None
                if self.model_metadata:
                    model_task = self.model_metadata.get("HF_TASK")
                hf_model_md = get_huggingface_model_metadata(self.model)
                if not model_task:
                    model_task = hf_model_md.get("pipeline_tag")
                if model_task:
                    self._hf_schema_builder_init(model_task)

            huggingface_model_id = self.model
            jumpstart_model_id = self._jumpstart_mapping[huggingface_model_id]["jumpstart-model-id"]
            self.model = jumpstart_model_id
            merged_date = self._jumpstart_mapping[huggingface_model_id].get("merged-at")
            self._build_for_jumpstart()
            compare_model_diff_message = (
                "If you want to identify the differences between the two, "
                "please use model_uris.retrieve() to retrieve the model "
                "artifact S3 URI and compare them."
            )
            logger.warning(  # pylint: disable=logging-fstring-interpolation
                "Please note that for this model we are using the JumpStart's "
                f'local copy "{jumpstart_model_id}" '
                f'of the HuggingFace model "{huggingface_model_id}" you chose. '
                "We strive to keep our local copy synced with the HF model hub closely. "
                "This model was synced "
                f"{f'on {merged_date}' if merged_date else 'before 11/04/2024'}. "
                f"{compare_model_diff_message if not self._is_gated_model() else ''}"
            )
            return True
        return False

    def _retrieve_hugging_face_model_mapping(self):
        """Retrieve the HuggingFace/JumpStart model mapping and preprocess it."""
        converted_mapping = {}
        region = self.sagemaker_session.boto_region_name
        try:
            mapping_json_object = JumpStartS3PayloadAccessor.get_object_cached(
                bucket=get_jumpstart_content_bucket(region),
                key="hf_model_id_map_cache.json",
                region=region,
                s3_client=self.sagemaker_session.s3_client,
            )
            mapping = json.loads(mapping_json_object)
        except Exception:  # pylint: disable=broad-except
            return converted_mapping

        for k, v in mapping.items():
            converted_mapping[v["hf-model-id"]] = {
                "jumpstart-model-id": k,
                "jumpstart-model-version": v["jumpstart-model-version"],
                "merged-at": v.get("merged-at"),
                "hf-model-repo-sha": v.get("hf-model-repo-sha"),
            }
        return converted_mapping
