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
"""Holds mixin logic to support deployment of Model ID"""
from __future__ import absolute_import
import logging
from typing import Type
from abc import ABC, abstractmethod

from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.serve.utils.hf_utils import _get_model_config_properties_from_hf

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serve.utils.local_hardware import (
    _get_nb_instance,
)
from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
from sagemaker.serve.utils.optimize_utils import _is_optimized
from sagemaker.serve.utils.predictors import TeiLocalModePredictor
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.base_predictor import PredictorBase

logger = logging.getLogger(__name__)

_CODE_FOLDER = "code"


class TEI(ABC):
    """TEI build logic for ModelBuilder()"""

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
        self.image_config = None
        self.vpc_config = None
        self._original_deploy = None
        self.hf_model_config = None
        self._default_tensor_parallel_degree = None
        self._default_data_type = None
        self._default_max_tokens = None
        self.pysdk_model = None
        self.schema_builder = None
        self.env_vars = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.secret_key = None
        self.role_arn = None
        self.name = None

    @abstractmethod
    def _prepare_for_mode(self, *args, **kwargs):
        """Placeholder docstring"""

    @abstractmethod
    def _get_client_translators(self):
        """Placeholder docstring"""

    def _set_to_tei(self):
        """Placeholder docstring"""
        if self.model_server != ModelServer.TEI:
            messaging = (
                "HuggingFace Model ID support on model server: "
                f"{self.model_server} is not currently supported. "
                f"Defaulting to {ModelServer.TEI}"
            )
            logger.warning(messaging)
            self.model_server = ModelServer.TEI

    def _create_tei_model(self, **kwargs) -> Type[Model]:
        """Placeholder docstring"""
        if self.nb_instance_type and "instance_type" not in kwargs:
            kwargs.update({"instance_type": self.nb_instance_type})

        if not self.image_uri:
            self.image_uri = image_uris.retrieve(
                "huggingface-tei",
                image_scope="inference",
                instance_type=kwargs.get("instance_type"),
                region=self.sagemaker_session.boto_region_name,
            )

        pysdk_model = HuggingFaceModel(
            image_uri=self.image_uri,
            image_config=self.image_config,
            vpc_config=self.vpc_config,
            env=self.env_vars,
            role=self.role_arn,
            sagemaker_session=self.sagemaker_session,
            name=self.name,
        )

        logger.info("Detected %s. Proceeding with the the deployment.", self.image_uri)

        self._original_deploy = pysdk_model.deploy
        pysdk_model.deploy = self._tei_model_builder_deploy_wrapper
        return pysdk_model

    @_capture_telemetry("tei.deploy")
    def _tei_model_builder_deploy_wrapper(self, *args, **kwargs) -> Type[PredictorBase]:
        """Placeholder docstring"""
        timeout = kwargs.get("model_data_download_timeout")
        if timeout:
            self.pysdk_model.env.update({"MODEL_LOADING_TIMEOUT": str(timeout)})

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

            predictor = TeiLocalModePredictor(
                self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
            )

            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri,
                timeout if timeout else 1800,
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

        # if the weights have been cached via local container mode -> set to offline
        if str(Mode.LOCAL_CONTAINER) in self.modes:
            self.pysdk_model.env.update({"HF_HUB_OFFLINE": "1"})
        else:
            # if has not been built for local container we must use cache
            # that hosting has write access to.
            self.pysdk_model.env["HF_HOME"] = "/tmp"
            self.pysdk_model.env["HUGGINGFACE_HUB_CACHE"] = "/tmp"

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True

        if self.nb_instance_type and "instance_type" not in kwargs:
            kwargs.update({"instance_type": self.nb_instance_type})
        elif not self.nb_instance_type and "instance_type" not in kwargs:
            raise ValueError(
                "Instance type must be provided when deploying " "to SageMaker Endpoint mode."
            )

        if "initial_instance_count" not in kwargs:
            kwargs.update({"initial_instance_count": 1})

        predictor = self._original_deploy(*args, **kwargs)

        if "HF_HUB_OFFLINE" in self.pysdk_model.env:
            self.pysdk_model.env.update({"HF_HUB_OFFLINE": "0"})

        predictor.serializer = serializer
        predictor.deserializer = deserializer
        return predictor

    def _build_for_hf_tei(self):
        """Placeholder docstring"""
        self.nb_instance_type = _get_nb_instance()

        _create_dir_structure(self.model_path)
        if not hasattr(self, "pysdk_model"):
            self.env_vars.update({"HF_MODEL_ID": self.model})
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )

        self.pysdk_model = self._create_tei_model()

        if self.mode == Mode.LOCAL_CONTAINER:
            self._prepare_for_mode()

        return self.pysdk_model

    def _build_for_tei(self):
        """Placeholder docstring"""
        self.secret_key = None

        self._set_to_tei()

        self.pysdk_model = self._build_for_hf_tei()
        if self.role_arn:
            self.pysdk_model.role = self.role_arn
        if self.sagemaker_session:
            self.pysdk_model.sagemaker_session = self.sagemaker_session
        return self.pysdk_model
