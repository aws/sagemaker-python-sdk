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
"""Unit tests for region_validation module."""
from __future__ import absolute_import

import pytest

from sagemaker.core.region_validation import (
    InvalidRegionError,
    validate_region,
    validate_endpoint_url,
)

# All known AWS regions as of 2026. This list ensures the regex pattern
# does not accidentally reject any legitimate region string.
ALL_AWS_REGIONS = [
    # US East
    "us-east-1",
    "us-east-2",
    # US West
    "us-west-1",
    "us-west-2",
    # Africa
    "af-south-1",
    # Asia Pacific
    "ap-east-1",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    # Canada
    "ca-central-1",
    "ca-west-1",
    # Europe
    "eu-central-1",
    "eu-central-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "eu-south-2",
    "eu-north-1",
    # Israel
    "il-central-1",
    # Middle East
    "me-south-1",
    "me-central-1",
    # South America
    "sa-east-1",
    # China
    "cn-north-1",
    "cn-northwest-1",
    # GovCloud
    "us-gov-west-1",
    "us-gov-east-1",
    # ISO / ISOB partitions
    "us-iso-east-1",
    "us-iso-west-1",
    "us-isob-east-1",
    # Mexico
    "mx-central-1",
    # Asia Pacific (Malaysia / Thailand)
    "ap-southeast-7",
]


class TestValidateRegionAcceptsAllAwsRegions:
    """Ensure validate_region passes for every known AWS region."""

    @pytest.mark.parametrize("region", ALL_AWS_REGIONS)
    def test_valid_region(self, region):
        assert validate_region(region) == region


class TestValidateRegionRejectsInvalidInputs:
    """Ensure validate_region rejects malicious or malformed region strings."""

    @pytest.mark.parametrize(
        "invalid_region",
        [
            # SSRF payloads
            "x@attacker.com:443/#",
            "us-east-1.attacker.com",
            "us-east-1\n.attacker.com",
            # Empty / whitespace
            "",
            " ",
            # Missing components
            "useast1",
            "us-east",
            "us-1",
            # Uppercase
            "US-EAST-1",
            "Us-East-1",
            # Special characters
            "us-east-1; rm -rf /",
            "us-east-1/../../etc/passwd",
            # Non-string types
            None,
            123,
            ["us-east-1"],
            # Trailing/leading whitespace
            " us-east-1",
            "us-east-1 ",
            # Newline injection
            "us-east-1\n",
            "us-east-1\r\n",
            # URL-like
            "https://us-east-1",
            # Simple fake region (no digit suffix)
            "testregion",
        ],
    )
    def test_invalid_region(self, invalid_region):
        with pytest.raises(InvalidRegionError):
            validate_region(invalid_region)


class TestValidateEndpointUrl:
    """Ensure validate_endpoint_url accepts AWS domains and rejects others."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://sagemaker.us-east-1.amazonaws.com",
            "https://api.sagemaker.us-west-2.amazonaws.com",
            "https://runtime.sagemaker.eu-west-1.amazonaws.com",
            "https://sagemaker.cn-north-1.amazonaws.com.cn",
            "https://domain.studio.us-west-2.sagemaker.aws",
        ],
    )
    def test_valid_endpoint(self, url):
        assert validate_endpoint_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://attacker.com",
            "https://sagemaker.us-east-1.attacker.com",
            "https://amazonaws.com.attacker.com",
        ],
    )
    def test_invalid_endpoint(self, url):
        with pytest.raises(InvalidRegionError):
            validate_endpoint_url(url)
