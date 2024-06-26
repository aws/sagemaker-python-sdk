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
"""Transformers build logic with model builder"""
from __future__ import absolute_import
import logging
from abc import ABC, abstractmethod
from typing import Type
from packaging.version import Version

from sagemaker.model import Model
from sagemaker import image_uris
from sagemaker.serve.utils.local_hardware import (
    _get_nb_instance,
)
from sagemaker.serve.utils.hf_utils import _get_model_config_properties_from_hf
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serve.model_server.multi_model_server.prepare import (
    _create_dir_structure,
)
from sagemaker.serve.utils.predictors import TransformersLocalModePredictor
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.base_predictor import PredictorBase
from sagemaker.huggingface.llm_utils import get_huggingface_model_metadata

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 1800


"""Retrieves images for different libraries - Pytorch, TensorFlow from HuggingFace hub
"""


# pylint: disable=W0108
class Transformers(ABC):
    """Transformers build logic with ModelBuilder()"""

    def __init__(self):
        self.model = None
        self.serve_settings = None
        self.sagemaker_session = None
        self.model_path = None
        self.dependencies = None
        self.modes = None
        self.mode = None
        self.model_server = None
        self.image_uri = None
        self._is_custom_image_uri = False
        self.vpc_config = None
        self._original_deploy = None
        self.hf_model_config = None
        self._default_data_type = None
        self.pysdk_model = None
        self.env_vars = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.secret_key = None
        self.role_arn = None
        self.py_version = None
        self.tensorflow_version = None
        self.pytorch_version = None
        self.instance_type = None
        self.schema_builder = None

    @abstractmethod
    def _prepare_for_mode(self):
        """Abstract method"""

    def _create_transformers_model(self) -> Type[Model]:
        """Initializes HF model with or without image_uri"""
        if self.image_uri is None:
            pysdk_model = self._get_hf_metadata_create_model()
        else:
            pysdk_model = HuggingFaceModel(
                image_uri=self.image_uri,
                vpc_config=self.vpc_config,
                env=self.env_vars,
                role=self.role_arn,
                sagemaker_session=self.sagemaker_session,
            )

        logger.info("Detected %s. Proceeding with the the deployment.", self.image_uri)

        self._original_deploy = pysdk_model.deploy
        pysdk_model.deploy = self._transformers_model_builder_deploy_wrapper
        return pysdk_model

    def _get_hf_metadata_create_model(self) -> Type[Model]:
        """Initializes the model after fetching image

        1. Get the metadata for deciding framework
        2. Get the supported hugging face versions
        3. Create model
        4. Fetch image

        Returns:
            pysdk_model: Corresponding model instance
        """

        hf_model_md = get_huggingface_model_metadata(
            self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
        )
        hf_config = image_uris.config_for_framework("huggingface").get("inference")
        config = hf_config["versions"]
        base_hf_version = sorted(config.keys(), key=lambda v: Version(v), reverse=True)[0]

        if hf_model_md is None:
            raise ValueError("Could not fetch HF metadata")

        if "pytorch" in hf_model_md.get("tags"):
            self.pytorch_version = self._get_supported_version(
                hf_config, base_hf_version, "pytorch"
            )
            self.py_version = config[base_hf_version]["pytorch" + self.pytorch_version].get(
                "py_versions"
            )[-1]
            pysdk_model = HuggingFaceModel(
                env=self.env_vars,
                role=self.role_arn,
                sagemaker_session=self.sagemaker_session,
                py_version=self.py_version,
                transformers_version=base_hf_version,
                pytorch_version=self.pytorch_version,
                vpc_config=self.vpc_config,
            )
        elif "keras" in hf_model_md.get("tags") or "tensorflow" in hf_model_md.get("tags"):
            self.tensorflow_version = self._get_supported_version(
                hf_config, base_hf_version, "tensorflow"
            )
            self.py_version = config[base_hf_version]["tensorflow" + self.tensorflow_version].get(
                "py_versions"
            )[-1]
            pysdk_model = HuggingFaceModel(
                env=self.env_vars,
                role=self.role_arn,
                sagemaker_session=self.sagemaker_session,
                py_version=self.py_version,
                transformers_version=base_hf_version,
                tensorflow_version=self.tensorflow_version,
                vpc_config=self.vpc_config,
            )

        if self.mode == Mode.LOCAL_CONTAINER:
            self.image_uri = pysdk_model.serving_image_uri(
                self.sagemaker_session.boto_region_name, "local"
            )
        else:
            self.image_uri = pysdk_model.serving_image_uri(
                self.sagemaker_session.boto_region_name, self.instance_type
            )

        if pysdk_model is None or self.image_uri is None:
            raise ValueError("PySDK model unable to be created, try overriding image_uri")

        if not pysdk_model.image_uri:
            pysdk_model.image_uri = self.image_uri

        return pysdk_model

    @_capture_telemetry("transformers.deploy")
    def _transformers_model_builder_deploy_wrapper(self, *args, **kwargs) -> Type[PredictorBase]:
        """Returns predictor depending on local or sagemaker endpoint mode

        Returns:
            TransformersLocalModePredictor: During local mode deployment
        """
        timeout = kwargs.get("model_data_download_timeout")
        if timeout:
            self.env_vars.update({"MODEL_LOADING_TIMEOUT": str(timeout)})

        if "mode" in kwargs and kwargs.get("mode") != self.mode:
            overwrite_mode = kwargs.get("mode")
            # mode overwritten by customer during model.deploy()
            logger.warning(
                "Deploying in %s Mode, overriding existing configurations set for %s mode",
                overwrite_mode,
                self.mode,
            )

            if overwrite_mode == Mode.SAGEMAKER_ENDPOINT:
                self.mode = self.pysdk_model.mode = Mode.SAGEMAKER_ENDPOINT
            elif overwrite_mode == Mode.LOCAL_CONTAINER:
                self._prepare_for_mode()
                self.mode = self.pysdk_model.mode = Mode.LOCAL_CONTAINER
            else:
                raise ValueError("Mode %s is not supported!" % overwrite_mode)

        self._set_instance()

        serializer = self.schema_builder.input_serializer
        deserializer = self.schema_builder._output_deserializer
        if self.mode == Mode.LOCAL_CONTAINER:
            timeout = kwargs.get("model_data_download_timeout")

            predictor = TransformersLocalModePredictor(
                self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
            )

            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri,
                timeout if timeout else DEFAULT_TIMEOUT,
                None,
                predictor,
                self.pysdk_model.env,
                jumpstart=False,
            )
            return predictor

        if "mode" in kwargs:
            del kwargs["mode"]
        if "role" in kwargs:
            self.pysdk_model.role = kwargs.get("role")
            del kwargs["role"]

        # set model_data to uncompressed s3 dict
        self.pysdk_model.model_data, env_vars = self._prepare_for_mode()
        self.env_vars.update(env_vars)
        self.pysdk_model.env.update(self.env_vars)

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True

        if "initial_instance_count" not in kwargs:
            kwargs.update({"initial_instance_count": 1})

        predictor = self._original_deploy(*args, **kwargs)

        predictor.serializer = serializer
        predictor.deserializer = deserializer
        return predictor

    def _build_transformers_env(self):
        """Build model for hugging face deployment using"""
        self.nb_instance_type = _get_nb_instance()

        _create_dir_structure(self.model_path)
        if not hasattr(self, "pysdk_model"):
            self.env_vars.update({"HF_MODEL_ID": self.model})

            logger.info(self.env_vars)

            # TODO: Move to a helper function
            if hasattr(self.env_vars, "HF_API_TOKEN"):
                self.hf_model_config = _get_model_config_properties_from_hf(
                    self.model, self.env_vars.get("HF_API_TOKEN")
                )
            else:
                self.hf_model_config = _get_model_config_properties_from_hf(
                    self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )

        self.pysdk_model = self._create_transformers_model()

        if self.mode == Mode.LOCAL_CONTAINER:
            self._prepare_for_mode()

        return self.pysdk_model

    def _set_instance(self, **kwargs):
        """Set the instance : Given the detected notebook type or provided instance type"""
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            if self.nb_instance_type and "instance_type" not in kwargs:
                kwargs.update({"instance_type": self.nb_instance_type})
                logger.info("Setting instance type to %s", self.nb_instance_type)
            elif self.instance_type and "instance_type" not in kwargs:
                kwargs.update({"instance_type": self.instance_type})
                logger.info("Setting instance type to %s", self.instance_type)
            else:
                raise ValueError(
                    "Instance type must be provided when deploying to SageMaker Endpoint mode."
                )

    def _get_supported_version(self, hf_config, hugging_face_version, base_fw):
        """Uses the hugging face json config to pick supported versions"""
        version_config = hf_config.get("versions").get(hugging_face_version)
        versions_to_return = list()
        for key in list(version_config.keys()):
            if key.startswith(base_fw):
                base_fw_version = key[len(base_fw) :]
                if len(hugging_face_version.split(".")) == 2:
                    base_fw_version = ".".join(base_fw_version.split(".")[:-1])
                versions_to_return.append(base_fw_version)
        return sorted(versions_to_return, reverse=True)[0]

    def _build_for_transformers(self):
        """Method that triggers model build

        Returns:PySDK model
        """
        self.secret_key = None
        self.model_server = ModelServer.MMS

        self._build_transformers_env()

        return self.pysdk_model
