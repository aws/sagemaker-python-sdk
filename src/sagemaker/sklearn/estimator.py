# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging

from sagemaker.estimator import Framework
from sagemaker.fw_registry import default_framework_uri
from sagemaker.fw_utils import framework_name_from_image, empty_framework_version_warning
from sagemaker.sklearn.defaults import SKLEARN_VERSION, SKLEARN_NAME
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger('sagemaker')


class SKLearn(Framework):
    """Handle end-to-end training and deployment of custom Scikit-learn code."""

    __framework_name__ = SKLEARN_NAME

    def __init__(self, entry_point, framework_version=SKLEARN_VERSION, source_dir=None, hyperparameters=None,
                 py_version='py3', image_name=None, **kwargs):
        """
        This ``Estimator`` executes an Scikit-learn script in a managed Scikit-learn execution environment, within a
        SageMaker Training Job. The managed Scikit-learn environment is an Amazon-built Docker container that executes
        functions defined in the supplied ``entry_point`` Python script.

        Training is started by calling :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a
        hosted SageMaker endpoint and returns an :class:`~sagemaker.amazon.sklearn.model.SKLearnPredictor` instance
        that can be used to perform inference against the hosted model.

        Technical documentation on preparing Scikit-learn scripts for SageMaker training and using the Scikit-learn
        Estimator is available on the project home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to training. This should be compatible with either Python 2.7 or Python 3.5.
            source_dir (str): Path (absolute or relative) to a directory with any other training
                source code dependencies aside from tne entry point file (default: None). Structure within this
                directory are preserved when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for training (default: None).
                The hyperparameters are made accessible as a dict[str, str] to the training code on SageMaker.
                For convenience, this accepts other types for keys and values, but ``str()`` will be called
                to convert them before training.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
                              One of 'py2' or 'py3'.
            framework_version (str): Scikit-learn version you want to use for executing your model training code.
                List of supported versions https://github.com/aws/sagemaker-python-sdk#sklearn-sagemaker-estimators
            image_name (str): If specified, the estimator will use this image for training and hosting, instead of
                selecting the appropriate SageMaker official image based on framework_version and py_version. It can
                be an ECR url or dockerhub image and tag.
                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
        """
        # SciKit-Learn does not support distributed training or training on GPU instance types. Fail fast.
        train_instance_type = kwargs.get('train_instance_type')
        _validate_not_gpu_instance_type(train_instance_type)

        train_instance_count = kwargs.get('train_instance_count')
        if train_instance_count:
            if train_instance_count != 1:
                raise AttributeError("Scikit-Learn does not support distributed training. "
                                     "Please remove the 'train_instance_count' argument or set "
                                     "'train_instance_count=1' when initializing SKLearn.")
        super(SKLearn, self).__init__(entry_point, source_dir, hyperparameters, image_name=image_name,
                                      **dict(kwargs, train_instance_count=1))

        self.py_version = py_version

        if framework_version is None:
            logger.warning(empty_framework_version_warning(SKLEARN_VERSION, SKLEARN_VERSION))
        self.framework_version = framework_version or SKLEARN_VERSION

        if image_name is None:
            image_tag = "{}-{}-{}".format(framework_version, "cpu", py_version)
            self.image_name = default_framework_uri(
                SKLearn.__framework_name__,
                self.sagemaker_session.boto_region_name,
                image_tag)

    def create_model(self, model_server_workers=None, role=None,
                     vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Create a SageMaker ``SKLearnModel`` object that can be deployed to an ``Endpoint``.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Passed to initialization of ``SKLearnModel``.

        Returns:
            sagemaker.sklearn.model.SKLearnModel: A SageMaker ``SKLearnModel`` object.
                See :func:`~sagemaker.sklearn.model.SKLearnModel` for full details.
        """
        role = role or self.role
        return SKLearnModel(self.model_data, role, self.entry_point, source_dir=self._model_source_dir(),
                            enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                            container_log_level=self.container_log_level, code_location=self.code_location,
                            py_version=self.py_version, framework_version=self.framework_version,
                            model_server_workers=model_server_workers, image=self.image_name,
                            sagemaker_session=self.sagemaker_session,
                            vpc_config=self.get_vpc_config(vpc_config_override),
                            **kwargs)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(SKLearn, cls)._prepare_init_params_from_job_description(job_details)

        image_name = init_params.pop('image')
        framework, py_version, _, _ = framework_name_from_image(image_name)
        init_params['py_version'] = py_version

        if framework and framework != cls.__framework_name__:
            training_job_name = init_params['base_job_name']
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))
        elif not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params['image_name'] = image_name
        return init_params


def _validate_not_gpu_instance_type(training_instance_type):
    gpu_instance_types = ['ml.p2.xlarge', 'ml.p2.8xlarge', 'ml.p2.16xlarge',
                          'ml.p3.xlarge', 'ml.p3.8xlarge', 'ml.p3.16xlarge']

    if training_instance_type in gpu_instance_types:
        raise ValueError("GPU training in not supported for Scikit-Learn. "
                         "Please pick a different instance type from here: "
                         "https://aws.amazon.com/ec2/instance-types/")
