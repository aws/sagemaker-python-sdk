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

from sagemaker.model_monitor import ServerlessInferenceConfig

MEMORY_SIZE_IN_MB = 2048
MAX_CONCURRENCY = 2


def test_init():
    serverless_inference_config = ServerlessInferenceConfig(
        memory_size_in_mb=MEMORY_SIZE_IN_MB,
        max_concurrency=MAX_CONCURRENCY,
    )

    assert serverless_inference_config.memory_size_in_mb == MEMORY_SIZE_IN_MB
    assert serverless_inference_config.max_concurrency == MAX_CONCURRENCY
