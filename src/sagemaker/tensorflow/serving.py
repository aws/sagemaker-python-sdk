# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import logging

import sagemaker
from sagemaker.content_types import CONTENT_TYPE_JSON
from sagemaker.fw_utils import create_image_uri
from sagemaker.predictor import json_serializer, json_deserializer
from sagemaker.tensorflow.defaults import TF_VERSION


class Predictor(sagemaker.RealTimePredictor):
    """A ``RealTimePredictor`` implementation for inference against TensorFlow
    Serving endpoints.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=json_serializer,
        deserializer=json_deserializer,
        content_type=None,
        model_name=None,
        model_version=None,
    ):
        """Initialize a ``TFSPredictor``. See ``sagemaker.RealTimePredictor``
        for more info about parameters.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (callable): Optional. Default serializes input data to
                json. Handles dicts, lists, and numpy arrays.
            deserializer (callable): Optional. Default parses the response using
                ``json.load(...)``.
            content_type (str): Optional. The "ContentType" for invocation
                requests. If specified, overrides the ``content_type`` from the
                serializer (default: None).
            model_name (str): Optional. The name of the SavedModel model that
                should handle the request. If not specified, the endpoint's
                default model will handle the request.
            model_version (str): Optional. The version of the SavedModel model
                that should handle the request. If not specified, the latest
                version of the model will be used.
        """
        super(Predictor, self).__init__(
            endpoint_name, sagemaker_session, serializer, deserializer, content_type
        )

        attributes = []
        if model_name:
            attributes.append("tfs-model-name={}".format(model_name))
        if model_version:
            attributes.append("tfs-model-version={}".format(model_version))
        self._model_attributes = ",".join(attributes) if attributes else None

    def classify(self, data):
        """
        Args:
            data:
        """
        return self._classify_or_regress(data, "classify")

    def regress(self, data):
        """
        Args:
            data:
        """
        return self._classify_or_regress(data, "regress")

    def _classify_or_regress(self, data, method):
        """
        Args:
            data:
            method:
        """
        if method not in ["classify", "regress"]:
            raise ValueError("invalid TensorFlow Serving method: {}".format(method))

        if self.content_type != CONTENT_TYPE_JSON:
            raise ValueError("The {} api requires json requests.".format(method))

        args = {"CustomAttributes": "tfs-method={}".format(method)}

        return self.predict(data, args)

    def predict(self, data, initial_args=None):
        """
        Args:
            data:
            initial_args:
        """
        args = dict(initial_args) if initial_args else {}
        if self._model_attributes:
            if "CustomAttributes" in args:
                args["CustomAttributes"] += "," + self._model_attributes
            else:
                args["CustomAttributes"] = self._model_attributes

        return super(Predictor, self).predict(data, args)


class Model(sagemaker.model.FrameworkModel):
    """Placeholder docstring"""

    FRAMEWORK_NAME = "tensorflow-serving"
    LOG_LEVEL_PARAM_NAME = "SAGEMAKER_TFS_NGINX_LOGLEVEL"
    LOG_LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "crit",
    }
    LATEST_EIA_VERSION = [2, 0]

    def __init__(
        self,
        model_data,
        role,
        entry_point=None,
        image=None,
        framework_version=TF_VERSION,
        container_log_level=None,
        predictor_cls=Predictor,
        **kwargs
    ):
        """Initialize a Model.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker APIs that create Amazon SageMaker endpoints use this
                role to access model artifacts.
            entry_point:
            image (str): A Docker image URI (default: None). If not specified, a
                default image for TensorFlow Serving will be used.
            framework_version (str): Optional. TensorFlow Serving version you
                want to use.
            container_log_level (int): Log level to use within the container
                (default: logging.ERROR). Valid values are defined in the Python
                logging module.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            **kwargs: Keyword arguments passed to the ``Model`` initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        super(Model, self).__init__(
            model_data=model_data,
            role=role,
            image=image,
            predictor_cls=predictor_cls,
            entry_point=entry_point,
            **kwargs
        )
        self._framework_version = framework_version
        self._container_log_level = container_log_level

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        endpoint_name=None,
        update_endpoint=False,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
    ):

        if accelerator_type and not self._eia_supported():
            msg = "The TensorFlow version %s doesn't support EIA." % self._framework_version

            raise AttributeError(msg)
        return super(Model, self).deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
        )

    def _eia_supported(self):
        """Return true if TF version is EIA enabled"""
        return [int(s) for s in self._framework_version.split(".")][:2] <= self.LATEST_EIA_VERSION

    def prepare_container_def(self, instance_type, accelerator_type=None):
        """
        Args:
            instance_type:
            accelerator_type:
        """
        image = self._get_image_uri(instance_type, accelerator_type)
        env = self._get_container_env()

        if self.entry_point:
            key_prefix = sagemaker.fw_utils.model_code_key_prefix(self.key_prefix, self.name, image)

            bucket = self.bucket or self.sagemaker_session.default_bucket()
            model_data = "s3://{}/{}/model.tar.gz".format(bucket, key_prefix)

            sagemaker.utils.repack_model(
                self.entry_point,
                self.source_dir,
                self.dependencies,
                self.model_data,
                model_data,
                self.sagemaker_session,
                kms_key=self.model_kms_key,
            )
        else:
            model_data = self.model_data

        return sagemaker.container_def(image, model_data, env)

    def _get_container_env(self):
        """Placeholder docstring"""
        if not self._container_log_level:
            return self.env

        if self._container_log_level not in Model.LOG_LEVEL_MAP:
            logging.warning("ignoring invalid container log level: %s", self._container_log_level)
            return self.env

        env = dict(self.env)
        env[Model.LOG_LEVEL_PARAM_NAME] = Model.LOG_LEVEL_MAP[self._container_log_level]
        return env

    def _get_image_uri(self, instance_type, accelerator_type=None):
        """
        Args:
            instance_type:
            accelerator_type:
        """
        if self.image:
            return self.image

        region_name = self.sagemaker_session.boto_region_name
        return create_image_uri(
            region_name,
            Model.FRAMEWORK_NAME,
            instance_type,
            self._framework_version,
            accelerator_type=accelerator_type,
        )

    def serving_image_uri(
        self, region_name, instance_type, accelerator_type=None
    ):  # pylint: disable=unused-argument
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model (default: None). For example, 'ml.eia1.medium'.

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        return self._get_image_uri(instance_type=instance_type, accelerator_type=accelerator_type)
