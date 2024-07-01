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
"""Constants used in SageMaker Python SDK telemetry."""

from __future__ import absolute_import
from enum import Enum

# Default AWS region used by SageMaker
DEFAULT_AWS_REGION = "us-west-2"


class Feature(Enum):
    """Enumeration of feature names used in telemetry."""

    SDK_DEFAULTS = 1
    LOCAL_MODE = 2
    REMOTE_FUNCTION = 3

    def __str__(self):  # pylint: disable=E0307
        """Return the feature name."""
        return self.name


class Status(Enum):
    """Enumeration of status values used in telemetry."""

    SUCCESS = 1
    FAILURE = 0

    def __str__(self):  # pylint: disable=E0307
        """Return the status name."""
        return self.name
