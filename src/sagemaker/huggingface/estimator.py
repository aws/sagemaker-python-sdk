# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import re

from sagemaker.deprecations import renamed_kwargs
from sagemaker.estimator import Framework
from sagemaker.fw_utils import (
    framework_name_from_image,
    warn_if_parameter_server_with_multi_gpu,
    validate_smdistributed,
)
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger("sagemaker")


class HuggingFace(Framework):
    """Handle training of custom HuggingFace code."""

    _framework_name = "huggingface"

    def __init__(
        self,
        py_version,
        entry_point,
        transformers_version=None,
        tensorflow_version=None,
        pytorch_version=None,
        source_dir=None,
        hyperparameters=None,
        image_uri=None,
        distribution=None,
        **kwargs
    ):
        """This ``Estimator`` executes a HuggingFace script in a managed execution environment.

        The managed HuggingFace environment is an Amazon-built Docker container that executes
        functions defined in the supplied ``entry_point`` Python script within a SageMaker
        Training Job.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.

        Args:
            py_version (str): Python version you want to use for executing your model training
                code. Defaults to ``None``. Required unless ``image_uri`` is provided.  If
                using PyTorch, the current supported version is ``py36``. If using TensorFlow,
                the current supported version is ``py37``.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            transformers_version (str): Transformers version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. The current supported version is ``4.4.2``.
            tensorflow_version (str): TensorFlow version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``pytorch_version`` is provided. The current supported version is ``1.6.0``.
            pytorch_version (str): PyTorch version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``tensorflow_version`` is provided. The current supported version is ``2.4.1``.
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
        self.framework_version = transformers_version
        self.py_version = py_version
        self.tensorflow_version = tensorflow_version
        self.pytorch_version = pytorch_version

        self._validate_args(image_uri=image_uri)

        if distribution is not None:
            instance_type = renamed_kwargs(
                "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
            )

            base_framework_name = "tensorflow" if tensorflow_version is not None else "pytorch"
            base_framework_version = (
                tensorflow_version if tensorflow_version is not None else pytorch_version
            )

            validate_smdistributed(
                instance_type=instance_type,
                framework_name=base_framework_name,
                framework_version=base_framework_version,
                py_version=self.py_version,
                distribution=distribution,
                image_uri=image_uri,
            )

            warn_if_parameter_server_with_multi_gpu(
                training_instance_type=instance_type, distribution=distribution
            )

        if "enable_sagemaker_metrics" not in kwargs:
            kwargs["enable_sagemaker_metrics"] = True

        super(HuggingFace, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )
        self.distribution = distribution or {}

    def _validate_args(self, image_uri):
        """Placeholder docstring"""
        if image_uri is not None:
            return

        if self.framework_version is None and image_uri is None:
            raise ValueError(
                "transformers_version, and image_uri are both None. "
                "Specify either transformers_version or image_uri"
            )
        if self.tensorflow_version is not None and self.pytorch_version is not None:
            raise ValueError(
                "tensorflow_version and pytorch_version are both not None. "
                "Specify only tensorflow_version or pytorch_version."
            )
        if self.tensorflow_version is None and self.pytorch_version is None:
            raise ValueError(
                "tensorflow_version and pytorch_version are both None. "
                "Specify either tensorflow_version or pytorch_version."
            )
        base_framework_version_len = (
            len(self.tensorflow_version.split("."))
            if self.tensorflow_version is not None
            else len(self.pytorch_version.split("."))
        )
        transformers_version_len = len(self.framework_version.split("."))
        if transformers_version_len != base_framework_version_len:
            raise ValueError(
                "Please use either full version or shortened version for both "
                "transformers_version, tensorflow_version and pytorch_version."
            )

    def hyperparameters(self):
        """Return hyperparameters used by your custom PyTorch code during model training."""
        hyperparameters = super(HuggingFace, self).hyperparameters()
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
        """Placeholder docstring"""
        raise NotImplementedError("Creating model with HuggingFace training job is not supported.")

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: The returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(HuggingFace, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)

        if tag is None:
            framework_version = None
        else:
            framework, pt_or_tf = framework.split("-")
            tag_pattern = re.compile("^(.*)-transformers(.*)-(cpu|gpu)-(py2|py3[67]?)$")
            tag_match = tag_pattern.match(tag)
            pt_or_tf_version = tag_match.group(1)
            framework_version = tag_match.group(2)
            if pt_or_tf == "pytorch":
                init_params["pytorch_version"] = pt_or_tf_version
            else:
                init_params["tensorflow_version"] = pt_or_tf_version

        init_params["transformers_version"] = framework_version
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
