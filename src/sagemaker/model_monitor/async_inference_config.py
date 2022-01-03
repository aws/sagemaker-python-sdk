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
"""This module contains code related to the AsyncInferenceConfig class.

Codes are used for configuring async inference endpoint. Use it when deploying
the model to the endpoints.
"""
from __future__ import print_function, absolute_import


class AsyncInferenceConfig(object):
    """Configuration object passed in when deploying models to Amazon SageMaker Endpoints.

    This object specifies configuration related to async endpoint. Use this configuration
    when trying to create async endpoint and make async inference
    """

    def __init__(
        self,
        s3_output_path,
        max_concurrent_invocations_per_instance=None,
        kms_key_id=None,
        notification_config=None,
    ):
        """Initialize a AsyncInferenceConfig object for async inference related configuration.

        Args:
            s3_output_path (str): Required. The Amazon S3 location to upload inference responses to.
            max_concurrent_invocations_per_instance (int): Optional. The maximum number of
                concurrent requests sent by the SageMaker client to the model container. If
                no value is provided, Amazon SageMaker will choose an optimal value for you.
            kms_key_id (str): Optional. The Amazon Web Services Key Management Service
                (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt the
                asynchronous inference output in Amazon S3.
            notification_config (dict): Optional. Specifies the configuration for notifications
                of inference results for asynchronous inference:
                * success_topic (str): Amazon SNS topic to post a notification to when inference
                completes successfully. If no topic is provided, no notification is sent on success.
                The key in notification_config is 'SuccessTopic'.
                * error_topic (str): Amazon SNS topic to post a notification to when inference
                fails. If no topic is provided, no notification is sent on failure.
                The key in notification_config is 'ErrorTopic'.
        """
        self.s3_output_path = s3_output_path
        self.max_concurrent_invocations_per_instance = max_concurrent_invocations_per_instance
        self.kms_key_id = kms_key_id
        self.notification_config = notification_config

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "OutputConfig": {
                "S3OutputPath": self.s3_output_path,
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
