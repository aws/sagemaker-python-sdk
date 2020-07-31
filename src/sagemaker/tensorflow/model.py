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
"""Classes for using TensorFlow on Amazon SageMaker for inference."""
from __future__ import absolute_import

import logging

import sagemaker
from sagemaker import image_uris
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer


class TensorFlowPredictor(Predictor):
    """A ``Predictor`` implementation for inference against TensorFlow
    Serving endpoints.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        model_name=None,
        model_version=None,
    ):
        """Initialize a ``TensorFlowPredictor``. See :class:`~sagemaker.predictor.Predictor`
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
            model_name (str): Optional. The name of the SavedModel model that
                should handle the request. If not specified, the endpoint's
                default model will handle the request.
            model_version (str): Optional. The version of the SavedModel model
                that should handle the request. If not specified, the latest
                version of the model will be used.
        """
        super(TensorFlowPredictor, self).__init__(
            endpoint_name, sagemaker_session, serializer, deserializer,
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

        if self.content_type != "application/json":
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

        return super(TensorFlowPredictor, self).predict(data, args)


class TensorFlowModel(sagemaker.model.FrameworkModel):
    """A ``FrameworkModel`` implementation for inference with TensorFlow Serving."""

    _framework_name = "tensorflow"
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
        image_uri=None,
        framework_version=None,
        container_log_level=None,
        predictor_cls=TensorFlowPredictor,
        **kwargs
    ):
        """Initialize a Model.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            image_uri (str): A Docker image URI (default: None). If not specified, a
                default image for TensorFlow Serving will be used. If
                ``framework_version`` is ``None``, then ``image_uri`` is required.
                If also ``None``, then a ``ValueError`` will be raised.
            framework_version (str): Optional. TensorFlow Serving version you
                want to use. Defaults to ``None``. Required unless ``image_uri`` is
                provided.
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
        if framework_version is None and image_uri is None:
            raise ValueError(
                "Both framework_version and image_uri were None. "
                "Either specify framework_version or specify image_uri."
            )
        self.framework_version = framework_version

        super(TensorFlowModel, self).__init__(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            predictor_cls=predictor_cls,
            entry_point=entry_point,
            **kwargs
        )
        self._container_log_level = container_log_level

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=None,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
    ):

        if accelerator_type and not self._eia_supported():
            msg = "The TensorFlow version %s doesn't support EIA." % self.framework_version
            raise AttributeError(msg)

        return super(TensorFlowModel, self).deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
        )

    def _eia_supported(self):
        """Return true if TF version is EIA enabled"""
        return [int(s) for s in self.framework_version.split(".")][:2] <= self.LATEST_EIA_VERSION

    def prepare_container_def(self, instance_type=None, accelerator_type=None):
        """
        Args:
            instance_type:
            accelerator_type:
        """
        if self.image_uri is None and instance_type is None:
            raise ValueError(
                "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
            )

        image_uri = self._get_image_uri(instance_type, accelerator_type)
        env = self._get_container_env()

        if self.entry_point:
            key_prefix = sagemaker.fw_utils.model_code_key_prefix(
                self.key_prefix, self.name, image_uri
            )

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

        return sagemaker.container_def(image_uri, model_data, env)

    def _get_container_env(self):
        """Placeholder docstring"""
        if not self._container_log_level:
            return self.env

        if self._container_log_level not in self.LOG_LEVEL_MAP:
            logging.warning("ignoring invalid container log level: %s", self._container_log_level)
            return self.env

        env = dict(self.env)
        env[self.LOG_LEVEL_PARAM_NAME] = self.LOG_LEVEL_MAP[self._container_log_level]
        return env

    def _get_image_uri(self, instance_type, accelerator_type=None):
        """
        Args:
            instance_type:
            accelerator_type:
        """
        if self.image_uri:
            return self.image_uri

        return image_uris.retrieve(
            self._framework_name,
            self.sagemaker_session.boto_region_name,
            version=self.framework_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
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
