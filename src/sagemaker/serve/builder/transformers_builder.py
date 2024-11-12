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
import os
from abc import ABC, abstractmethod
from typing import Type
from pathlib import Path
import subprocess
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
    prepare_for_mms,
)
from sagemaker.serve.detector.image_detector import (
    auto_detect_container,
)
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.utils.optimize_utils import _is_optimized
from sagemaker.serve.utils.predictors import (
    TransformersLocalModePredictor,
    TransformersInProcessModePredictor,
)
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.base_predictor import PredictorBase
from sagemaker.huggingface.llm_utils import get_huggingface_model_metadata
from sagemaker.serve.builder.requirements_manager import RequirementsManager


logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 1800
LOCAL_MODES = [Mode.LOCAL_CONTAINER, Mode.IN_PROCESS]


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
        self.inference_spec = None
        self.shared_libs = None
        self.name = None

    @abstractmethod
    def _prepare_for_mode(self, *args, **kwargs):
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
                name=self.name,
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
            self.env_vars.get("HF_MODEL_ID"), self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
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

        if not self.image_uri and self.mode == Mode.LOCAL_CONTAINER:
            self.image_uri = pysdk_model.serving_image_uri(
                self.sagemaker_session.boto_region_name, "local"
            )
        elif not self.image_uri:
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

        if self.mode == Mode.IN_PROCESS:
            timeout = kwargs.get("model_data_download_timeout")

            predictor = TransformersInProcessModePredictor(
                self.modes[str(Mode.IN_PROCESS)], serializer, deserializer
            )

            self.modes[str(Mode.IN_PROCESS)].create_server(
                predictor,
            )
            return predictor

        self._set_instance(kwargs)

        if "mode" in kwargs:
            del kwargs["mode"]
        if "role" in kwargs:
            self.pysdk_model.role = kwargs.get("role")
            del kwargs["role"]

        if not _is_optimized(self.pysdk_model):
            env_vars = {}
            if str(Mode.LOCAL_CONTAINER) in self.modes:
                # upload model artifacts to S3 if LOCAL_CONTAINER -> SAGEMAKER_ENDPOINT
                self.pysdk_model.model_data, env_vars = self._prepare_for_mode(
                    model_path=self.model_path, should_upload_artifacts=True
                )
            else:
                _, env_vars = self._prepare_for_mode()

            self.env_vars.update(env_vars)
            self.pysdk_model.env.update(self.env_vars)

        if (
            "SAGEMAKER_SERVE_SECRET_KEY" in self.pysdk_model.env
            and not self.pysdk_model.env["SAGEMAKER_SERVE_SECRET_KEY"]
        ):
            del self.pysdk_model.env["SAGEMAKER_SERVE_SECRET_KEY"]

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

            if self.inference_spec is not None:
                self.env_vars.update({"HF_MODEL_ID": self.inference_spec.get_model()})
            else:
                self.env_vars.update({"HF_MODEL_ID": self.model})

            logger.info(self.env_vars)

            # TODO: Move to a helper function
            if hasattr(self.env_vars, "HF_API_TOKEN"):
                self.hf_model_config = _get_model_config_properties_from_hf(
                    self.env_vars.get("HF_MODEL_ID"), self.env_vars.get("HF_API_TOKEN")
                )
            else:
                self.hf_model_config = _get_model_config_properties_from_hf(
                    self.env_vars.get("HF_MODEL_ID"), self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
                )

        self.pysdk_model = self._create_transformers_model()

        if self.mode in LOCAL_MODES:
            self._prepare_for_mode()

        return self.pysdk_model

    def _set_instance(self, kwargs):
        """Set the instance : Given the detected notebook type or provided instance type"""
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            if "instance_type" in kwargs:
                return
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

    def _auto_detect_container(self):
        """Set image_uri by detecting container via model name or inference spec"""
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
            raise ValueError(
                "Cannot detect and set image_uri. Please pass model or inference spec."
            )

    def _build_for_transformers(self):
        """Method that triggers model build

        Returns:PySDK model
        """
        self.secret_key = None
        self.model_server = ModelServer.MMS

        if self.inference_spec:

            os.makedirs(self.model_path, exist_ok=True)

            code_path = Path(self.model_path).joinpath("code")

            save_pkl(code_path, (self.inference_spec, self.schema_builder))
            logger.info("PKL file saved to file: %s", code_path)

            if self.mode == Mode.IN_PROCESS:
                self._create_conda_env()

            self._auto_detect_container()

            self.secret_key = prepare_for_mms(
                model_path=self.model_path,
                shared_libs=self.shared_libs,
                dependencies=self.dependencies,
                session=self.sagemaker_session,
                image_uri=self.image_uri,
                inference_spec=self.inference_spec,
            )

        self._build_transformers_env()

        if self.role_arn:
            self.pysdk_model.role = self.role_arn
        if self.sagemaker_session:
            self.pysdk_model.sagemaker_session = self.sagemaker_session
        return self.pysdk_model

    def _create_conda_env(self):
        """Creating conda environment by running commands"""

        try:
            RequirementsManager().capture_and_install_dependencies
        except subprocess.CalledProcessError:
            print("Failed to create and activate conda environment.")
