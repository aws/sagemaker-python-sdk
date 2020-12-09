# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.workflow.execution_variables import ExecutionVariables


def test_execution_variable():
    var = ExecutionVariables.START_DATETIME
    assert var.to_request() == {"Get": "Execution.StartDateTime"}
    assert var.expr == {"Get": "Execution.StartDateTime"}
    assert isinstance(var, str)
