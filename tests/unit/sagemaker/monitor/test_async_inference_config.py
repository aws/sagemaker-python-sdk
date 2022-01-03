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

from sagemaker.model_monitor import AsyncInferenceConfig

S3_OUTPUT_PATH = "s3://some-output-path"
DEFAULT_KMS_KEY_ID = None
DEFAULT_MAX_CONCURRENT_INVOCATIONS = None
DEFAULT_NOTIFICATION_CONFIG = None

OPTIONAL_KMS_KEY_ID = "some-kms-key-id"
OPTIONAL_MAX_CONCURRENT_INVOCATIONS = 2
OPTIONAL_NOTIFICATION_CONFIG = {
    "SuccessTopic": "some-sunccess-topic",
    "ErrorTopic": "some-error-topic",
}


def test_init_without_optional():
    async_inference_config = AsyncInferenceConfig(
        s3_output_path=S3_OUTPUT_PATH,
    )

    assert async_inference_config.s3_output_path == S3_OUTPUT_PATH
    assert async_inference_config.kms_key_id == DEFAULT_KMS_KEY_ID
    assert (
        async_inference_config.max_concurrent_invocations_per_instance
        == DEFAULT_MAX_CONCURRENT_INVOCATIONS
    )
    assert async_inference_config.notification_config == DEFAULT_NOTIFICATION_CONFIG


def test_init_with_optional():
    async_inference_config = AsyncInferenceConfig(
        s3_output_path=S3_OUTPUT_PATH,
        kms_key_id=OPTIONAL_KMS_KEY_ID,
        max_concurrent_invocations_per_instance=OPTIONAL_MAX_CONCURRENT_INVOCATIONS,
        notification_config=OPTIONAL_NOTIFICATION_CONFIG,
    )

    assert async_inference_config.s3_output_path == S3_OUTPUT_PATH
    assert async_inference_config.kms_key_id == OPTIONAL_KMS_KEY_ID
    assert (
        async_inference_config.max_concurrent_invocations_per_instance
        == OPTIONAL_MAX_CONCURRENT_INVOCATIONS
    )
    assert async_inference_config.notification_config == OPTIONAL_NOTIFICATION_CONFIG
