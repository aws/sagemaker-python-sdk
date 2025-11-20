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
"""Exceptions for local pipeline execution."""
from __future__ import absolute_import


class StepExecutionException(Exception):
    """Exception raised when a pipeline step execution fails in local mode."""

    def __init__(self, step_name, message):
        """Initialize StepExecutionException.

        Args:
            step_name (str): Name of the step that failed
            message (str): Failure message
        """
        self.step_name = step_name
        self.message = message
        super().__init__(f"Step '{step_name}' failed: {message}")
