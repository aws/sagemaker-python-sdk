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
from sagemaker import image_uris
from sagemaker.deserializers import CSVDeserializer
from sagemaker.fw_utils import model_code_key_prefix
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import Predictor
from sagemaker.serializers import LibSVMSerializer
from sagemaker.xgboost.defaults import XGBOOST_NAME
from sagemaker.xgboost.utils import validate_py_version, validate_framework_version

logger = logging.getLogger("sagemaker")


class XGBoostPredictor(Predictor):
    """A Predictor for inference against XGBoost Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to xgb.DMatrix
    for XGBoost inference.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=LibSVMSerializer(),
        deserializer=CSVDeserializer(),
    ):
        """Initialize an ``XGBoostPredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference on.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                interactions with Amazon SageMaker APIs and any other AWS services needed.
                If not specified, the estimator creates one using the default AWS configuration
                chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to LibSVM format
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from text/csv to a Python list.
        """
        super(XGBoostPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class XGBoostModel(FrameworkModel):
    """An XGBoost SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = XGBOOST_NAME

    def __init__(
        self,
        model_data,
        role,
        entry_point,
        framework_version,
        image_uri=None,
        py_version="py3",
        predictor_cls=XGBoostPredictor,
        model_server_workers=None,
        **kwargs
    ):
        """Initialize an XGBoostModel.

        Args:
            model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created, the inference
                code might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source file which should
                be executed  as the entry point to model hosting. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
            image_uri (str): A Docker image URI (default: None). If not specified, a default image
                for XGBoost is be used.
            py_version (str): Python version you want to use for executing your model training code
                (default: 'py3').
            framework_version (str): XGBoost version you want to use for executing your model
                training code.
            predictor_cls (callable[str, sagemaker.session.Session]): A function to call to create
                a predictor with an endpoint name and SageMaker ``Session``.
                If specified, ``deploy()`` returns the result of invoking this function on the
                created endpoint name.
            model_server_workers (int): Optional. The number of worker processes used by the
                inference server. If None, server will use one worker per vCPU.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        super(XGBoostModel, self).__init__(
            model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )

        self.py_version = py_version
        self.framework_version = framework_version
        self.model_server_workers = model_server_workers

        validate_py_version(py_version)
        validate_framework_version(framework_version)

    def prepare_container_def(self, instance_type=None, accelerator_type=None):
        """Return a container definition with framework configuration.

        The framework configuration is set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
            accelerator_type (str): The Elastic Inference accelerator type to deploy to the
            instance for loading and making inferences to the model. This parameter is
                unused because accelerator types are not supported by XGBoostModel.

        Returns:
            dict[str, str]: A container definition object usable with the CreateModel API.
        """
        deploy_image = self.image_uri
        if not deploy_image:
            deploy_image = self.serving_image_uri(
                self.sagemaker_session.boto_region_name, instance_type
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)
        return sagemaker.container_def(deploy_image, self.model_data, deploy_env)

    def serving_image_uri(self, region_name, instance_type):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Must be a CPU instance type.

        Returns:
            str: The appropriate image URI based on the given parameters.
        """
        return image_uris.retrieve(
            self._framework_name,
            region_name,
            version=self.framework_version,
            instance_type=instance_type,
        )
