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
    MODEL_TRAINER = 4
    ESTIMATOR = 5
    HYPERPOD = 6  # Added to support telemetry in sagemaker-hyperpod-cli
    HYPERPOD_CLI = 7  # Added to support telemetry in sagemaker-hyperpod-cli
    MODEL_CUSTOMIZATION = 8

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


class Region(str, Enum):
    """Telemetry: List of all supported AWS regions."""

    # Classic
    US_EAST_1 = "us-east-1"  # IAD
    US_EAST_2 = "us-east-2"  # CMH
    US_WEST_1 = "us-west-1"  # SFO
    US_WEST_2 = "us-west-2"  # PDX
    AP_NORTHEAST_1 = "ap-northeast-1"  # NRT
    AP_NORTHEAST_2 = "ap-northeast-2"  # ICN
    AP_NORTHEAST_3 = "ap-northeast-3"  # KIX
    AP_SOUTH_1 = "ap-south-1"  # BOM
    AP_SOUTHEAST_1 = "ap-southeast-1"  # SIN
    AP_SOUTHEAST_2 = "ap-southeast-2"  # SYD
    CA_CENTRAL_1 = "ca-central-1"  # YUL
    EU_CENTRAL_1 = "eu-central-1"  # FRA
    EU_NORTH_1 = "eu-north-1"  # ARN
    EU_WEST_1 = "eu-west-1"  # DUB
    EU_WEST_2 = "eu-west-2"  # LHR
    EU_WEST_3 = "eu-west-3"  # CDG
    SA_EAST_1 = "sa-east-1"  # GRU
    # Opt-in
    AP_EAST_1 = "ap-east-1"  # HKG
    AP_SOUTHEAST_3 = "ap-southeast-3"  # CGK
    AF_SOUTH_1 = "af-south-1"  # CPT
    EU_SOUTH_1 = "eu-south-1"  # MXP
    ME_SOUTH_1 = "me-south-1"  # BAH
    MX_CENTRAL_1 = "mx-central-1"  # QRO
    AP_SOUTHEAST_7 = "ap-southeast-7"  # BKK
    AP_SOUTH_2 = "ap-south-2"  # HYD
    AP_SOUTHEAST_4 = "ap-southeast-4"  # MEL
    EU_CENTRAL_2 = "eu-central-2"  # ZRH
    EU_SOUTH_2 = "eu-south-2"  # ZAZ
    IL_CENTRAL_1 = "il-central-1"  # TLV
    ME_CENTRAL_1 = "me-central-1"  # DXB
