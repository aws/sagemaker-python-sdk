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
"""Configuration classes for SageMaker inference endpoints.

This module provides configuration classes for different types of SageMaker
inference endpoints including async, serverless, and resource requirements.
"""
from __future__ import print_function, absolute_import
from typing import Optional


class AsyncInferenceConfig(object):
    """Configuration object for async inference endpoints.

    This object specifies configuration related to async endpoint. Use this configuration
    when trying to create async endpoint and make async inference.
    """

    def __init__(
        self,
        output_path=None,
        max_concurrent_invocations_per_instance=None,
        kms_key_id=None,
        notification_config=None,
        failure_path=None,
    ):
        """Initialize an AsyncInferenceConfig object for async inference configuration.

        Args:
            output_path (str): Optional. The Amazon S3 location that endpoints upload
                inference responses to. If no value is provided, Amazon SageMaker will
                use default Amazon S3 Async Inference output path. (Default: None)
            max_concurrent_invocations_per_instance (int): Optional. The maximum number of
                concurrent requests sent by the SageMaker client to the model container. If
                no value is provided, Amazon SageMaker will choose an optimal value for you.
                (Default: None)
            kms_key_id (str): Optional. The Amazon Web Services Key Management Service
                (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt the
                asynchronous inference output in Amazon S3. (Default: None)
            failure_path (str): Optional. The Amazon S3 location that endpoints upload model
                responses for failed requests. If no value is provided, Amazon SageMaker will
                use default Amazon S3 Async Inference failure path. (Default: None)
            notification_config (dict): Optional. Specifies the configuration for notifications
                of inference results for asynchronous inference. Only one notification is generated
                per invocation request (Default: None):
                * success_topic (str): Amazon SNS topic to post a notification to when inference
                completes successfully. If no topic is provided, no notification is sent on success.
                The key in notification_config is 'SuccessTopic'.
                * error_topic (str): Amazon SNS topic to post a notification to when inference
                fails. If no topic is provided, no notification is sent on failure.
                The key in notification_config is 'ErrorTopic'.
                * include_inference_response_in (list): Optional. When provided the inference
                response will be included in the notification topics. If not provided,
                a notification will still be generated on success/error, but will not
                contain the inference response.
                Valid options are SUCCESS_NOTIFICATION_TOPIC, ERROR_NOTIFICATION_TOPIC
        """
        self.output_path = output_path
        self.max_concurrent_invocations_per_instance = max_concurrent_invocations_per_instance
        self.kms_key_id = kms_key_id
        self.notification_config = notification_config
        self.failure_path = failure_path

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "OutputConfig": {
                "S3OutputPath": self.output_path,
                "S3FailurePath": self.failure_path,
            },
        }

        if self.max_concurrent_invocations_per_instance:
            request_dict["ClientConfig"] = {
                "MaxConcurrentInvocationsPerInstance": self.max_concurrent_invocations_per_instance
            }

        if self.kms_key_id:
            request_dict["OutputConfig"]["KmsKeyId"] = self.kms_key_id

        if self.notification_config:
            request_dict["OutputConfig"]["NotificationConfig"] = self.notification_config

        return request_dict


class ServerlessInferenceConfig(object):
    """Configuration object for serverless inference endpoints.

    This object specifies configuration related to serverless endpoint. Use this configuration
    when trying to create serverless endpoint and make serverless inference.
    """

    def __init__(
        self,
        memory_size_in_mb: int = 2048,
        max_concurrency: int = 5,
        provisioned_concurrency: Optional[int] = None,
    ):
        """Initialize a ServerlessInferenceConfig object for serverless inference configuration.

        Args:
            memory_size_in_mb (int): Optional. The memory size of your serverless endpoint.
                Valid values are in 1 GB increments: 1024 MB, 2048 MB, 3072 MB, 4096 MB,
                5120 MB, or 6144 MB. If no value is provided, Amazon SageMaker will choose
                the default value for you. (Default: 2048)
            max_concurrency (int): Optional. The maximum number of concurrent invocations
                your serverless endpoint can process. If no value is provided, Amazon
                SageMaker will choose the default value for you. (Default: 5)
            provisioned_concurrency (int): Optional. The provisioned concurrency of your
                serverless endpoint. If no value is provided, Amazon SageMaker will not
                apply provisioned concurrency to your Serverless endpoint. (Default: None)
        """
        self.memory_size_in_mb = memory_size_in_mb
        self.max_concurrency = max_concurrency
        self.provisioned_concurrency = provisioned_concurrency

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "MemorySizeInMB": self.memory_size_in_mb,
            "MaxConcurrency": self.max_concurrency,
        }

        if self.provisioned_concurrency is not None:
            request_dict["ProvisionedConcurrency"] = self.provisioned_concurrency

        return request_dict


# Re-export ResourceRequirements from its existing location in core
from sagemaker.core.resource_requirements import ResourceRequirements  # noqa: F401, E402

__all__ = ["AsyncInferenceConfig", "ServerlessInferenceConfig", "ResourceRequirements"]
