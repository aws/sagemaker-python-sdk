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

from sagemaker import fw_utils

import sagemaker
from sagemaker.fw_utils import (
    create_image_uri,
    model_code_key_prefix,
    python_deprecation_warning,
    empty_framework_version_warning,
)
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.chainer import defaults
from sagemaker.predictor import RealTimePredictor, npy_serializer, numpy_deserializer

logger = logging.getLogger("sagemaker")


class ChainerPredictor(RealTimePredictor):
    """A RealTimePredictor for inference against Chainer Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for Chainer inference.
    """

    def __init__(self, endpoint_name, sagemaker_session=None):
        """Initialize an ``ChainerPredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
        """
        super(ChainerPredictor, self).__init__(
            endpoint_name, sagemaker_session, npy_serializer, numpy_deserializer
        )


class ChainerModel(FrameworkModel):
    """An Chainer SageMaker ``Model`` that can be deployed to a SageMaker
    ``Endpoint``.
    """

    __framework_name__ = "chainer"

    def __init__(
        self,
        model_data,
        role,
        entry_point,
        image=None,
        py_version="py3",
        framework_version=None,
        predictor_cls=ChainerPredictor,
        model_server_workers=None,
        **kwargs
    ):
        """Initialize an ChainerModel.

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
            image (str): A Docker image URI (default: None). If not specified, a
                default image for Chainer will be used.
            py_version (str): Python version you want to use for executing your
                model training code (default: 'py2').
            framework_version (str): Chainer version you want to use for
                executing your model training code.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the
                :class:`~sagemaker.model.FrameworkModel` initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        super(ChainerModel, self).__init__(
            model_data, image, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self.__framework_name__, defaults.LATEST_PY2_VERSION)
            )

        if framework_version is None:
            logger.warning(
                empty_framework_version_warning(defaults.CHAINER_VERSION, defaults.LATEST_VERSION)
            )

        self.py_version = py_version
        self.framework_version = framework_version or defaults.CHAINER_VERSION
        self.model_server_workers = model_server_workers

    def prepare_container_def(self, instance_type, accelerator_type=None):
        """Return a container definition with framework configuration set in
        model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        deploy_image = self.image
        if not deploy_image:
            region_name = self.sagemaker_session.boto_session.region_name
            deploy_image = create_image_uri(
                region_name,
                self.__framework_name__,
                instance_type,
                self.framework_version,
                self.py_version,
                accelerator_type=accelerator_type,
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
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        return fw_utils.create_image_uri(
            region_name,
            self.__framework_name__,
            instance_type,
            self.framework_version,
            self.py_version,
        )
