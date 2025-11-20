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
"""Unit tests for workflow model_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch


def test_model_step_properties():
    """Test ModelStep has properties"""
    from sagemaker.mlops.workflow.model_step import ModelStep
    
    step_args = {"ModelName": "test-model"}
    
    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        step = ModelStep(name="model-step", step_args=step_args)
        assert step.name == "model-step"
        assert hasattr(step, "properties")
