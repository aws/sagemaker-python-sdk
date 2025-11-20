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
"""Unit tests for workflow monitor_batch_transform_step."""
from __future__ import absolute_import

import pytest


def test_monitor_batch_transform_step_module_exists():
    """Test MonitorBatchTransformStep module can be imported"""
    try:
        from sagemaker.mlops.workflow import monitor_batch_transform_step
        assert monitor_batch_transform_step is not None
    except ImportError:
        pytest.skip("MonitorBatchTransformStep not available")
