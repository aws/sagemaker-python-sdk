# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import sagemaker
from sagemaker.fw_utils import create_image_uri
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.mxnet.defaults import MXNET_VERSION
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
from sagemaker.utils import name_from_image


class MXNetPredictor(RealTimePredictor):
    """A RealTimePredictor for inference against MXNet Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to multidimensional tensors for MXNet
    inference."""

    def __init__(self, endpoint_name, sagemaker_session=None):
        """Initialize an ``MXNetPredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference on.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
        """
        super(MXNetPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)


class MXNetModel(FrameworkModel):
    """An MXNet SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    __framework_name__ = 'mxnet'

    def __init__(self, model_data, role, entry_point, image=None, py_version='py2', framework_version=MXNET_VERSION,
                 predictor_cls=MXNetPredictor, model_server_workers=None, **kwargs):
        """Initialize an MXNetModel.

        Args:
            model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to model hosting. This should be compatible with either Python 2.7 or Python 3.5.
            image (str): A Docker image URI (default: None). If not specified, a default image for MXNet will be used.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
            framework_version (str): MXNet version you want to use for executing your model training code.
            predictor_cls (callable[str, sagemaker.session.Session]): A function to call to create a predictor
                with an endpoint name and SageMaker ``Session``. If specified, ``deploy()`` returns the result of
                invoking this function on the created endpoint name.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.
            **kwargs: Keyword arguments passed to the ``FrameworkModel`` initializer.
        """
        super(MXNetModel, self).__init__(model_data, image, role, entry_point, predictor_cls=predictor_cls,
                                         **kwargs)
        self.py_version = py_version
        self.framework_version = framework_version
        self.model_server_workers = model_server_workers

    def prepare_container_def(self, instance_type):
        """Return a container definition with framework configuration set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.

        Returns:
            dict[str, str]: A container definition object usable with the CreateModel API.
        """
        deploy_image = self.image
        if not deploy_image:
            region_name = self.sagemaker_session.boto_session.region_name
            deploy_image = create_image_uri(region_name, self.__framework_name__, instance_type,
                                            self.framework_version, self.py_version)
        deploy_key_prefix = self.key_prefix or self.name or name_from_image(deploy_image)
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)
        return sagemaker.container_def(deploy_image, self.model_data, deploy_env)
