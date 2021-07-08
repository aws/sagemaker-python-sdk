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

from packaging.version import Version

from sagemaker.deprecations import renamed_kwargs
from sagemaker.estimator import Framework
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    python_deprecation_warning,
    validate_version_or_image_args,
    warn_if_parameter_server_with_multi_gpu,
    validate_smdistributed,
)
from sagemaker.pytorch import defaults
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger("sagemaker")


class PyTorch(Framework):
    """Handle end-to-end training and deployment of custom PyTorch code."""

    _framework_name = "pytorch"

    def __init__(
        self,
        entry_point,
        framework_version=None,
        py_version=None,
        source_dir=None,
        hyperparameters=None,
        image_uri=None,
        distribution=None,
        **kwargs
    ):
        """This ``Estimator`` executes a PyTorch script in a managed PyTorch execution environment.

        The managed PyTorch environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script within a SageMaker Training Job.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~sagemaker.amazon.pytorch.model.PyTorchPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing PyTorch scripts for SageMaker
        training and using the PyTorch Estimator is available on the project
        home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            framework_version (str): PyTorch version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. List of supported versions:
                https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
            py_version (str): Python version you want to use for executing your
                model training code. One of 'py2' or 'py3'. Defaults to ``None``. Required
                unless ``image_uri`` is provided.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for
                training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            image_uri (str): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.
                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            distribution (dict): A dictionary with information on how to run distributed training
                (default: None).  Currently, the following are supported:
                distributed training with parameter servers, SageMaker Distributed (SMD) Data
                and Model Parallelism, and MPI. SMD Model Parallelism can only be used with MPI.
                To enable parameter server use the following setup:

                .. code:: python

                    {
                        "parameter_server": {
                            "enabled": True
                        }
                    }

                To enable MPI:

                .. code:: python

                    {
                        "mpi": {
                            "enabled": True
                        }
                    }

                To enable SMDistributed Data Parallel or Model Parallel:

                .. code:: python

                    {
                        "smdistributed": {
                            "dataparallel": {
                                "enabled": True
                            },
                            "modelparallel": {
                                "enabled": True,
                                "parameters": {}
                            }
                        }
                    }

            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework`
                constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version

        if distribution is not None:
            instance_type = renamed_kwargs(
                "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
            )

            validate_smdistributed(
                instance_type=instance_type,
                framework_name=self._framework_name,
                framework_version=framework_version,
                py_version=py_version,
                distribution=distribution,
                image_uri=image_uri,
            )

            warn_if_parameter_server_with_multi_gpu(
                training_instance_type=instance_type, distribution=distribution
            )

        if "enable_sagemaker_metrics" not in kwargs:
            # enable sagemaker metrics for PT v1.3 or greater:
            if self.framework_version and Version(self.framework_version) >= Version("1.3"):
                kwargs["enable_sagemaker_metrics"] = True

        super(PyTorch, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )
        self.distribution = distribution or {}

    def hyperparameters(self):
        """Return hyperparameters used by your custom PyTorch code during model training."""
        hyperparameters = super(PyTorch, self).hyperparameters()
        additional_hyperparameters = self._distribution_configuration(
            distribution=self.distribution
        )
        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs
    ):
        """Create a SageMaker ``PyTorchModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                If not specified, the training entry point is used.
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file.
                If not specified, the model source directory from training is used.
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container.
                If not specified, the dependencies from training are used.
                This is not supported with "local code" in Local Mode.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.pytorch.model.PyTorchModel`
                constructor.

        Returns:
            sagemaker.pytorch.model.PyTorchModel: A SageMaker ``PyTorchModel``
            object. See :func:`~sagemaker.pytorch.model.PyTorchModel` for full details.
        """
        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        return PyTorchModel(
            self.model_data,
            role or self.role,
            entry_point or self._model_entry_point(),
            framework_version=self.framework_version,
            py_version=self.py_version,
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(PyTorch, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)

        if tag is None:
            framework_version = None
        else:
            framework_version = framework_version_from_tag(tag)
        init_params["framework_version"] = framework_version
        init_params["py_version"] = py_version

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_uri"] = image_uri
            return init_params

        if framework != cls._framework_name:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        return init_params
