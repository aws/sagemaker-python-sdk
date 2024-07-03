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
from __future__ import absolute_import

import importlib.util
import uuid
from typing import Any, Type, List, Dict, Optional, Union
from dataclasses import dataclass, field
import logging
import os
import re

from pathlib import Path

from sagemaker.s3 import S3Downloader

from sagemaker import Session
from sagemaker.model import Model
from sagemaker.base_predictor import PredictorBase
from sagemaker.serializers import NumpySerializer, TorchTensorSerializer
from sagemaker.deserializers import JSONDeserializer, TorchTensorDeserializer
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.builder.tf_serving_builder import TensorflowServing
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.mode.sagemaker_endpoint_mode import SageMakerEndpointMode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
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
from sagemaker.serve.utils import task
from sagemaker.serve.utils.exceptions import TaskNotFoundException
from sagemaker.serve.utils.lineage_utils import _maintain_lineage_tracking_for_mlflow_model
from sagemaker.serve.utils.predictors import _get_local_mode_predictor
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
from sagemaker.serve.model_server.triton.triton_builder import Triton
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.serve.save_retrive.version_1_0_0.save.save_handler import SaveHandler
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import get_metadata
from sagemaker.serve.validations.check_image_and_hardware_type import (
    validate_image_uri_and_hardware,
)
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.huggingface.llm_utils import get_huggingface_model_metadata

logger = logging.getLogger(__name__)

# Any new server type should be added here
supported_model_servers = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
    ModelServer.TENSORFLOW_SERVING,
    ModelServer.MMS,
    ModelServer.TGI,
    ModelServer.TEI,
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
        model (Optional[Union[object, str]): Model object (with ``predict`` method to perform
            inference) or a HuggingFace/JumpStart Model ID. Either ``model`` or ``inference_spec``
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
            intended for production use at this moment.
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
        default="model-name-" + uuid.uuid1().hex,
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
    model: Optional[Union[object, str]] = field(
        default=None,
        metadata={
            "help": (
                'Model object with "predict" method to perform inference '
                "or HuggingFace/JumpStart Model ID"
            )
        },
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
            "`MLFLOW_MODEL_PATH`, and `MLFLOW_TRACKING_ARN`. HF_TASK should be set for new "
            "models without task metadata in the Hub, Adding unsupported task types will "
            "throw an exception"
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

    def _prepare_for_mode(self):
        """Placeholder docstring"""
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
                self.model_path,
                self.secret_key,
                self.serve_settings.s3_model_data_url,
                self.sagemaker_session,
                self.image_uri,
                self.jumpstart if hasattr(self, "jumpstart") else False,
            )
            self.env_vars.update(env_vars_sagemaker)
            return self.s3_upload_path, env_vars_sagemaker
        if self.mode == Mode.LOCAL_CONTAINER:
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

        raise ValueError(
            "Please specify mode in: %s, %s" % (Mode.LOCAL_CONTAINER, Mode.SAGEMAKER_ENDPOINT)
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
            raise Exception("Cannot serialize")

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
            raise Exception("Cannot deserialize")

        return serializer, deserializer

    def _get_predictor(self, endpoint_name: str, sagemaker_session: Session) -> Predictor:
        """Placeholder docstring"""
        serializer, deserializer = self._get_client_translators()

        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
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
        )

        # store the modes in the model so that we may
        # reference the configurations for local deploy() & predict()
        self.pysdk_model.mode = self.mode
        self.pysdk_model.modes = self.modes
        self.pysdk_model.serve_settings = self.serve_settings

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
            if not instance_type:
                raise ValueError("Missing required parameter `instance_type`")

            if not initial_instance_count:
                raise ValueError("Missing required parameter `initial_instance_count`")

            if is_1p_image_uri(image_uri=self.image_uri):
                validate_image_uri_and_hardware(
                    image_uri=self.image_uri,
                    instance_type=instance_type,
                    model_server=self.model_server,
                )

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
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
        else:
            raise ValueError("Mode %s is not supported!" % overwrite_mode)

    def _build_for_torchserve(self) -> Type[Model]:
        """Build the model for torchserve"""
        self._save_model_inference_spec()

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

        return self._create_model()

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
            logger.info(
                "MLflow model metadata not detected in %s. ModelBuilder is not "
                "handling MLflow model input",
                mlflow_model_path,
            )
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

    # Model Builder is a class to build the model for deployment.
    # It supports two modes of deployment
    # 1/ SageMaker Endpoint
    # 2/ Local launch with container
    def build(  # pylint: disable=R0911
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
        self.sagemaker_session = sagemaker_session or Session()

        self.sagemaker_session.settings._local_download_dir = self.model_path

        # https://github.com/boto/botocore/blob/develop/botocore/useragent.py#L258
        # decorate to_string() due to
        # https://github.com/boto/botocore/blob/develop/botocore/client.py#L1014-L1015
        client = self.sagemaker_session.sagemaker_client
        client._user_agent_creator.to_string = self._user_agent_decorator(
            self.sagemaker_session.sagemaker_client._user_agent_creator.to_string
        )

        self.serve_settings = self._get_serve_setting()

        self._is_custom_image_uri = self.image_uri is not None

        self._handle_mlflow_input()

        self._build_validations()

        if self.model_server:
            return self._build_for_model_server()

        if isinstance(self.model, str):
            model_task = None
            if self.model_metadata:
                model_task = self.model_metadata.get("HF_TASK")
            if self._is_jumpstart_model_id():
                return self._build_for_jumpstart()
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
                    return self._build_for_tgi()
                if model_task == "sentence-similarity":
                    return self._build_for_tei()
                elif self._can_fit_on_single_gpu():
                    return self._build_for_transformers()
                else:
                    return self._build_for_transformers()

        # Set TorchServe as default model server
        if not self.model_server:
            self.model_server = ModelServer.TORCHSERVE
            return self._build_for_torchserve()

        raise ValueError("%s model server is not supported" % self.model_server)

    def _build_validations(self):
        """Validations needed for model server overrides, or auto-detection or fallback"""
        if self.mode == Mode.IN_PROCESS:
            raise ValueError("IN_PROCESS mode is not supported yet!")

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
            save_path (Optional[str]): The path where you want to save resources.
            s3_path (Optional[str]): The path where you want to upload resources.
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
