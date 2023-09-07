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

import pytest

from sagemaker.workflow.execution_variables import ExecutionVariables


def test_execution_variable():
    var = ExecutionVariables.START_DATETIME
    assert var.expr == {"Get": "Execution.StartDateTime"}


def test_to_string():
    var = ExecutionVariables.START_DATETIME

    assert var.to_string() == var


def test_implicit_value():
    var = ExecutionVariables.START_DATETIME

    with pytest.raises(TypeError) as error:
        str(var)
    assert "Pipeline variables do not support __str__ operation." in str(error.value)

    with pytest.raises(TypeError) as error:
        int(var)
    assert str(error.value) == "Pipeline variables do not support __int__ operation."

    with pytest.raises(TypeError) as error:
        float(var)
    assert str(error.value) == "Pipeline variables do not support __float__ operation."


def test_add_func():
    var_start_datetime = ExecutionVariables.START_DATETIME
    var_current_datetime = ExecutionVariables.CURRENT_DATETIME

    with pytest.raises(TypeError) as error:
        var_start_datetime + var_current_datetime

    assert str(error.value) == "Pipeline variables do not support concatenation."
