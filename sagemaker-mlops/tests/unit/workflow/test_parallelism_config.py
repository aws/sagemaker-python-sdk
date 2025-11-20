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
"""Unit tests for workflow parallelism_config."""
from __future__ import absolute_import

from sagemaker.mlops.workflow.parallelism_config import ParallelismConfiguration


def test_parallelism_configuration_init():
    config = ParallelismConfiguration(max_parallel_execution_steps=5)
    assert config.max_parallel_execution_steps == 5


def test_parallelism_configuration_to_request():
    config = ParallelismConfiguration(max_parallel_execution_steps=10)
    request = config.to_request()
    assert request == {"MaxParallelExecutionSteps": 10}
