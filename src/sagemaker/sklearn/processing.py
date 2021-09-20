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
"""This module contains code related to SKLearn Processors which are used for Processing jobs.

These jobs let customers perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation and interpretation on SageMaker.
"""
from __future__ import absolute_import

from sagemaker.processing import FrameworkProcessor
from sagemaker.sklearn.estimator import SKLearn


class SKLearnProcessor(FrameworkProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using scikit-learn.

    This processor executes a Python script in a scikit-learn execution environment.
    """

    estimator_cls = SKLearn

    def __init__(
        self,
        framework_version,
        role,
        instance_type,
        instance_count,
        py_version="py3",
        image_uri=None,
        command=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        code_location=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initialize an ``SKLearnProcessor`` instance.

        Unless ``image_uri`` is specified, the SKLearn environment is an
        Amazon-built Docker container that executes functions defined in the supplied
        ``code`` Python script.

        The arguments have the exact same meaning as in ``FrameworkProcessor``.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.processing.FrameworkProcessor`.
        """
        super(SKLearnProcessor, self).__init__(
            estimator_cls=self.estimator_cls,
            framework_version=framework_version,
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            py_version=py_version,
            command=command,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            code_location=code_location,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
