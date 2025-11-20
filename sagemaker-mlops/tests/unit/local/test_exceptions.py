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
"""Unit tests for local exceptions."""
from __future__ import absolute_import

import pytest

from sagemaker.mlops.local.exceptions import StepExecutionException


def test_step_execution_exception_init():
    exception = StepExecutionException("test-step", "Test error message")
    assert exception.step_name == "test-step"
    assert exception.message == "Test error message"
    assert "test-step" in str(exception)
    assert "Test error message" in str(exception)


def test_step_execution_exception_raise():
    with pytest.raises(StepExecutionException) as exc_info:
        raise StepExecutionException("test-step", "Test error")
    
    assert exc_info.value.step_name == "test-step"
    assert exc_info.value.message == "Test error"
