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
from __future__ import absolute_import

from sagemaker.async_inference import AsyncInferenceConfig

S3_OUTPUT_PATH = "s3://some-output-path"
S3_FAILURE_PATH = "s3://some-failure-path"
DEFAULT_KMS_KEY_ID = None
DEFAULT_MAX_CONCURRENT_INVOCATIONS = None
DEFAULT_NOTIFICATION_CONFIG = None
DEFAULT_ASYNC_INFERENCE_DICT = {
    "OutputConfig": {
        "S3OutputPath": S3_OUTPUT_PATH,
        "S3FailurePath": S3_FAILURE_PATH,
    },
}

OPTIONAL_KMS_KEY_ID = "some-kms-key-id"
OPTIONAL_MAX_CONCURRENT_INVOCATIONS = 2
OPTIONAL_NOTIFICATION_CONFIG = {
    "SuccessTopic": "some-sunccess-topic",
    "ErrorTopic": "some-error-topic",
    "IncludeInferenceResponseIn": ["SUCCESS_NOTIFICATION_TOPIC", "ERROR_NOTIFICATION_TOPIC"],
}
ASYNC_INFERENCE_DICT_WITH_OPTIONAL = {
    "OutputConfig": {
        "S3OutputPath": S3_OUTPUT_PATH,
        "S3FailurePath": S3_FAILURE_PATH,
        "KmsKeyId": OPTIONAL_KMS_KEY_ID,
        "NotificationConfig": OPTIONAL_NOTIFICATION_CONFIG,
    },
    "ClientConfig": {"MaxConcurrentInvocationsPerInstance": OPTIONAL_MAX_CONCURRENT_INVOCATIONS},
}


def test_init_without_optional():
    async_inference_config = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PATH, failure_path=S3_FAILURE_PATH
    )

    assert async_inference_config.output_path == S3_OUTPUT_PATH
    assert async_inference_config.failure_path == S3_FAILURE_PATH
    assert async_inference_config.kms_key_id == DEFAULT_KMS_KEY_ID
    assert (
        async_inference_config.max_concurrent_invocations_per_instance
        == DEFAULT_MAX_CONCURRENT_INVOCATIONS
    )
    assert async_inference_config.notification_config == DEFAULT_NOTIFICATION_CONFIG


def test_init_with_optional():
    async_inference_config = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PATH,
        failure_path=S3_FAILURE_PATH,
        max_concurrent_invocations_per_instance=OPTIONAL_MAX_CONCURRENT_INVOCATIONS,
        kms_key_id=OPTIONAL_KMS_KEY_ID,
        notification_config=OPTIONAL_NOTIFICATION_CONFIG,
    )

    assert async_inference_config.output_path == S3_OUTPUT_PATH
    assert async_inference_config.kms_key_id == OPTIONAL_KMS_KEY_ID
    assert async_inference_config.failure_path == S3_FAILURE_PATH
    assert (
        async_inference_config.max_concurrent_invocations_per_instance
        == OPTIONAL_MAX_CONCURRENT_INVOCATIONS
    )
    assert async_inference_config.notification_config == OPTIONAL_NOTIFICATION_CONFIG


def test_to_request_dict():
    async_inference_config = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PATH, failure_path=S3_FAILURE_PATH
    )
    assert async_inference_config._to_request_dict() == DEFAULT_ASYNC_INFERENCE_DICT

    async_inference_config_with_optional = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PATH,
        failure_path=S3_FAILURE_PATH,
        max_concurrent_invocations_per_instance=OPTIONAL_MAX_CONCURRENT_INVOCATIONS,
        kms_key_id=OPTIONAL_KMS_KEY_ID,
        notification_config=OPTIONAL_NOTIFICATION_CONFIG,
    )

    assert (
        async_inference_config_with_optional._to_request_dict()
        == ASYNC_INFERENCE_DICT_WITH_OPTIONAL
    )
