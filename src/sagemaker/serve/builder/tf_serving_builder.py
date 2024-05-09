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
import os
from pathlib import Path
from abc import ABC, abstractmethod

from sagemaker import Session
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving
from sagemaker.tensorflow import TensorFlowModel, TensorFlowPredictor

logger = logging.getLogger(__name__)

_TF_SERVING_MODEL_BUILDER_ENTRY_POINT = "inference.py"
_CODE_FOLDER = "code"


# pylint: disable=attribute-defined-outside-init, disable=E1101
class TensorflowServing(ABC):
    """TensorflowServing build logic for ModelBuilder()"""

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
        self.secret_key = None
        self.engine = None
        self.pysdk_model = None
        self.schema_builder = None
        self.env_vars = None

    @abstractmethod
    def _prepare_for_mode(self):
        """Prepare model artifacts based on mode."""

    @abstractmethod
    def _get_client_translators(self):
        """Set up client marshaller based on schema builder."""

    def _save_schema_builder(self):
        """Save schema builder for tensorflow serving."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        code_path = Path(self.model_path).joinpath("code")
        save_pkl(code_path, self.schema_builder)

    def _get_tensorflow_predictor(
        self, endpoint_name: str, sagemaker_session: Session
    ) -> TensorFlowPredictor:
        """Creates a TensorFlowPredictor object"""
        serializer, deserializer = self._get_client_translators()

        return TensorFlowPredictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )

    def _validate_for_tensorflow_serving(self):
        """Validate for tensorflow serving"""
        if not getattr(self, "_is_mlflow_model", False):
            raise ValueError("Tensorflow Serving is currently only supported for mlflow models.")

    def _create_tensorflow_model(self):
        """Creates a TensorFlow model object"""
        self.pysdk_model = TensorFlowModel(
            image_uri=self.image_uri,
            image_config=self.image_config,
            vpc_config=self.vpc_config,
            model_data=self.s3_upload_path,
            role=self.serve_settings.role_arn,
            env=self.env_vars,
            sagemaker_session=self.sagemaker_session,
            predictor_cls=self._get_tensorflow_predictor,
        )

        self.pysdk_model.mode = self.mode
        self.pysdk_model.modes = self.modes
        self.pysdk_model.serve_settings = self.serve_settings

        self._original_deploy = self.pysdk_model.deploy
        self.pysdk_model.deploy = self._model_builder_deploy_wrapper
        self._original_register = self.pysdk_model.register
        self.pysdk_model.register = self._model_builder_register_wrapper
        self.model_package = None
        return self.pysdk_model

    def _build_for_tensorflow_serving(self):
        """Build the model for Tensorflow Serving"""
        self._validate_for_tensorflow_serving()
        self._save_schema_builder()

        if not self.image_uri:
            raise ValueError("image_uri is not set for tensorflow serving")

        self.secret_key = prepare_for_tf_serving(
            model_path=self.model_path,
            shared_libs=self.shared_libs,
            dependencies=self.dependencies,
        )

        self._prepare_for_mode()

        return self._create_tensorflow_model()
