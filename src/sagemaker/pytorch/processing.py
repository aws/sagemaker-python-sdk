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
"""This module contains code related to PyTorch Processors which are used for Processing jobs.

These jobs let customers perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation and interpretation on SageMaker.
"""
from __future__ import absolute_import

from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch.estimator import PyTorch


class PyTorchProcessor(FrameworkProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using PyTorch containers."""

    estimator_cls = PyTorch

    def __init__(
        self,
        framework_version,  # New arg
        role,
        instance_count,
        instance_type,
        py_version="py3",  # New kwarg
        image_uri=None,
        command=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        code_location=None,  # New arg
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """This processor executes a Python script in a PyTorch execution environment.

        Unless ``image_uri`` is specified, the PyTorch environment is an
        Amazon-built Docker container that executes functions defined in the supplied
        ``code`` Python script.

        The arguments have the exact same meaning as in ``FrameworkProcessor``.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.processing.FrameworkProcessor`.
        """
        super().__init__(
            self.estimator_cls,
            framework_version,
            role,
            instance_count,
            instance_type,
            py_version,
            image_uri,
            command,
            volume_size_in_gb,
            volume_kms_key,
            output_kms_key,
            code_location,
            max_runtime_in_seconds,
            base_job_name,
            sagemaker_session,
            env,
            tags,
            network_config,
        )
