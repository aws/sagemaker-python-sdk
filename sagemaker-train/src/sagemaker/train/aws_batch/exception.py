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
"""The file Defines customized exception for Batch queueing"""
from __future__ import absolute_import


class NoTrainingJob(Exception):
    """Define NoTrainingJob Exception.

    It means no Training job has been created by AWS Batch service.
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        """Convert Exception to string.

        Returns: a String containing exception error messages.

        """
        return repr(self.value)


class MissingRequiredArgument(Exception):
    """Define MissingRequiredArgument exception.

    It means some required arguments are missing.
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        """Convert Exception to string.

        Returns: a String containing exception error messages.

        """
        return repr(self.value)
