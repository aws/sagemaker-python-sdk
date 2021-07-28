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
"""Custom exception classes for Sagemaker SDK"""
from __future__ import absolute_import


class UnexpectedStatusException(ValueError):
    """Raised when resource status is not expected and thus not allowed for further execution"""

    def __init__(self, message, allowed_statuses, actual_status):
        self.allowed_statuses = allowed_statuses
        self.actual_status = actual_status
        super(UnexpectedStatusException, self).__init__(message)
