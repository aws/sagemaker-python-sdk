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
import uuid
from typing import Any, Type, List, Dict, Optional, Union
from dataclasses import dataclass, field
import logging
import os

from pathlib import Path

from sagemaker import Session
from sagemaker.model import Model
from sagemaker.base_predictor import PredictorBase
from sagemaker.serializers import NumpySerializer, TorchTensorSerializer
from sagemaker.deserializers import JSONDeserializer, TorchTensorDeserializer
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.mode.sagemaker_endpoint_mode import SageMakerEndpointMode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.detector.pickler import save_pkl, save_xgboost
from sagemaker.serve.builder.serve_settings import _ServeSettings
from sagemaker.serve.builder.djl_builder import DJL
from sagemaker.serve.builder.tgi_builder import TGI
from sagemaker.serve.builder.jumpstart_builder import JumpStart
from sagemaker.predictor import Predictor
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import Metadata
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.utils.predictors import _get_local_mode_predictor
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

logger = logging.getLogger(__name__)

supported_model_server = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
}


# pylint: disable=attribute-defined-outside-init
@dataclass
class ModelBuilder(Triton, DJL, JumpStart, TGI):
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
        model (Optional[Union[object, str]): Model object (with ``predict`` method to perform
            inference) or a HuggingFace/JumpStart Model ID. Either ``model`` or
            ``inference_spec`` is required for the model builder to build the artifact.
        inference_spec (InferenceSpec): The inference spec file with your customized
            ``invoke`` and ``load`` functions.
        image_uri (Optional[str]): The container image uri (which is derived from a
            SageMaker-based container).
        model_server (Optional[ModelServer]): The model server to which to deploy.
            You need to provide this argument when you specify an ``image_uri``
            in order for model builder to build the artifacts correctly (according
            to the model server). Possible values for this argument are
            ``TORCHSERVE``, ``MMS``, ``TENSORFLOW_SERVING``, ``DJL_SERVING``,
            ``TRITON``, and ``TGI``.

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
    model_server: Optional[ModelServer] = field(
        default=None, metadata={"help": "Define the model server to deploy to."}
    )

    def _build_validations(self):
        """Placeholder docstring"""
        # TODO: Beta validations - remove after the launch
        if self.mode == Mode.IN_PROCESS:
            raise ValueError("IN_PROCESS mode is not supported yet!")

        if self.inference_spec and self.model:
            raise ValueError("Cannot have both the Model and Inference spec in the builder")

        if self.image_uri and not is_1p_image_uri(self.image_uri) and self.model_server is None:
            raise ValueError(
                "Model_server must be set when non-first-party image_uri is set. "
                + "Supported model servers: %s" % supported_model_server
            )

        # Set TorchServe as default model server
        if not self.model_server:
            self.model_server = ModelServer.TORCHSERVE

        if self.model_server not in supported_model_server:
            raise ValueError(
                "%s is not supported yet! Supported model servers: %s"
                % (self.model_server, supported_model_server)
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
        return self._original_deploy(
            *args,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            **kwargs,
        )

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
            inference_spec=self.inference_spec,
        )

        self._prepare_for_mode()

        return self._create_model()

    def _user_agent_decorator(self, func):
        """Placeholder docstring"""

        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            return result + " ModelBuilder"

        return wrapper

    # Model Builder is a class to build the model for deployment.
    # It supports three modes of deployment
    # 1/ SageMaker Endpoint
    # 2/ Local launch with container
    def build(
        self,
        mode: Type[Mode] = None,
        role_arn: str = None,
        sagemaker_session: str = None,
    ) -> Type[Model]:
        """Create a deployable ``Model`` instance with ``ModelBuilder``.

        Args:
            mode (Type[Mode], optional): The mode. Defaults to ``None``.
            role_arn (str, optional): The IAM role arn. Defaults to ``None``.
            sagemaker_session (str, optional): The SageMaker session to use
              for the execution. Defaults to ``None``.

        Returns:
            Type[Model]: A deployable ``Model`` object.
        """
        self.modes = dict()

        if mode:
            self.mode = mode
        if role_arn:
            self.role_arn = role_arn
        if sagemaker_session:
            self.sagemaker_session = sagemaker_session
        elif not self.sagemaker_session:
            self.sagemaker_session = Session()

        self.sagemaker_session.settings._local_download_dir = self.model_path

        # https://github.com/boto/botocore/blob/develop/botocore/useragent.py#L258
        # decorate to_string() due to
        # https://github.com/boto/botocore/blob/develop/botocore/client.py#L1014-L1015
        client = self.sagemaker_session.sagemaker_client
        client._user_agent_creator.to_string = self._user_agent_decorator(
            self.sagemaker_session.sagemaker_client._user_agent_creator.to_string
        )

        self.serve_settings = self._get_serve_setting()
        if isinstance(self.model, str):
            if self._is_jumpstart_model_id():
                return self._build_for_jumpstart()
            if self._is_djl():
                return self._build_for_djl()
            return self._build_for_tgi()

        self._build_validations()

        if self.model_server == ModelServer.TORCHSERVE:
            return self._build_for_torchserve()

        if self.model_server == ModelServer.TRITON:
            return self._build_for_triton()

        raise ValueError("%s model server is not supported" % self.model_server)

    def save(
        self,
        save_path: Optional[str] = None,
        s3_path: Optional[str] = None,
        sagemaker_session: Optional[str] = None,
        role_arn: Optional[str] = None,
    ) -> Type[Model]:
        """WARNING: This function is expremental and not intended for production use.

        This function is available for models served by DJL serving.

        Args:
            save_path (Optional[str]): The path where you want to save resources.
            s3_path (Optional[str]): The path where you want to upload resources.
        """
        self.sagemaker_session = sagemaker_session if sagemaker_session else Session()

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
