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
"""Unit tests for workflow tuning_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.steps import TuningStep


def test_tuning_step_requires_step_args():
    with pytest.raises(ValueError, match="step_args is required"):
        TuningStep(name="tuning-step", step_args=None)


def test_tuning_step_properties():
    from unittest.mock import patch
    
    step_args = Mock()
    step_args.caller_name = "tune"
    step_args.func_args = [Mock()]
    step_args.func_args[0].sagemaker_session = Mock()
    step_args.func_args[0].sagemaker_session.context = Mock()
    
    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        step = TuningStep(name="tuning-step", step_args=step_args)
        assert hasattr(step, "properties")
