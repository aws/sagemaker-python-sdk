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
"""Unit tests for workflow function_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock


def test_delayed_return_to_json_get():
    """Test DelayedReturn _to_json_get method"""
    from sagemaker.mlops.workflow.function_step import DelayedReturn
    
    delayed = DelayedReturn(function_step=Mock())
    json_get = delayed._to_json_get()
    assert json_get is not None
