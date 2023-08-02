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

from sagemaker.serverless import ServerlessInferenceConfig

DEFAULT_MEMORY_SIZE_IN_MB = 2048
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_PROVISIONED_CONCURRENCY = 5

DEFAULT_REQUEST_DICT = {
    "MemorySizeInMB": DEFAULT_MEMORY_SIZE_IN_MB,
    "MaxConcurrency": DEFAULT_MAX_CONCURRENCY,
}

PROVISIONED_CONCURRENCY_REQUEST_DICT = {
    "MemorySizeInMB": DEFAULT_MEMORY_SIZE_IN_MB,
    "MaxConcurrency": DEFAULT_MAX_CONCURRENCY,
    "ProvisionedConcurrency": DEFAULT_PROVISIONED_CONCURRENCY,
}


def test_init():
    serverless_inference_config = ServerlessInferenceConfig()

    assert serverless_inference_config.memory_size_in_mb == DEFAULT_MEMORY_SIZE_IN_MB
    assert serverless_inference_config.max_concurrency == DEFAULT_MAX_CONCURRENCY

    serverless_provisioned_concurrency_inference_config = ServerlessInferenceConfig(
        provisioned_concurrency=DEFAULT_PROVISIONED_CONCURRENCY
    )

    assert (
        serverless_provisioned_concurrency_inference_config.memory_size_in_mb
        == DEFAULT_MEMORY_SIZE_IN_MB
    )
    assert (
        serverless_provisioned_concurrency_inference_config.max_concurrency
        == DEFAULT_MAX_CONCURRENCY
    )
    assert (
        serverless_provisioned_concurrency_inference_config.provisioned_concurrency
        == DEFAULT_PROVISIONED_CONCURRENCY
    )


def test_to_request_dict():
    serverless_inference_config_dict = ServerlessInferenceConfig()._to_request_dict()

    assert serverless_inference_config_dict == DEFAULT_REQUEST_DICT

    serverless_provisioned_concurrency_inference_config_dict = ServerlessInferenceConfig(
        provisioned_concurrency=DEFAULT_PROVISIONED_CONCURRENCY
    )._to_request_dict()

    assert (
        serverless_provisioned_concurrency_inference_config_dict
        == PROVISIONED_CONCURRENCY_REQUEST_DICT
    )
