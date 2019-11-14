# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains code related to SKLearn Processors, which are used
for Processing jobs. These jobs let customers perform data pre-processing,
post-processing, feature engineering, data validation, and model evaluation
and interpretation on SageMaker.
"""
from __future__ import absolute_import

from sagemaker import Session
from sagemaker.processing import ScriptProcessor
from sagemaker.fw_utils import create_image_uri


class SKLearnProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using scikit-learn."""

    def __init__(
        self,
        framework_version,
        role,
        instance_type,
        py_version="py3",
        volume_size_in_gb=30,
        volume_kms_key=None,
        max_runtime_in_seconds=24 * 60 * 60,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initialize an ``SKLearnProcessor`` instance. The SKLearnProcessor
        handles Amazon SageMaker processing tasks for jobs using scikit-learn.

        Args:
            framework_version (str): The version of scikit-learn.
            role (str): An AWS IAM role. The Amazon SageMaker training jobs
                and APIs that create Amazon SageMaker endpoints use this role
                to access training data and model artifacts. After the endpoint
                is created, the inference code might use the IAM role, if it
                needs to access an AWS resource.
            instance_type (str): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            py_version (str): The python version to use, for example, 'py3'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume.
            max_runtime_in_seconds (int): Timeout in seconds
                (default: 24 * 60 * 60). After this amount of time Amazon
                SageMaker terminates the job regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the processor creates one
                using the default AWS configuration chain.
            env (dict): Environment variables to be passed to the processing job.
            tags ([dict]): List of tags to be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        session = sagemaker_session or Session()
        region = session.boto_region_name
        image_uri = create_image_uri(
            region=region,
            framework="scikit-learn",
            instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
        )

        super(SKLearnProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
            instance_count=1,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
