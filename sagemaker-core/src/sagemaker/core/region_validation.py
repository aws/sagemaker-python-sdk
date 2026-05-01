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
"""Region validation utilities to prevent SSRF via malicious region strings.

This module provides validation for AWS region parameters before they are
interpolated into endpoint URLs. Without validation, a crafted region value
(e.g., ``x@attacker.com:443/#``) could redirect SDK API calls — including
SigV4-signed requests — to non-AWS hosts.

See: CVE-2026-22611 (AWS SDK for .NET, same vulnerability class).
"""
from __future__ import absolute_import

import re
from urllib.parse import urlparse

# Regex for valid AWS region names (e.g., us-east-1, eu-west-2, cn-north-1, us-gov-west-1).
# Uses \A and \Z anchors to prevent newline injection bypass that $ allows.
_VALID_REGION_PATTERN = re.compile(r"\A[a-z]{2}(-[a-z]+)+-\d+\Z")

# Trusted AWS domain suffixes for endpoint URL validation (defense-in-depth).
_AWS_DOMAINS = (
    ".amazonaws.com",
    ".amazonaws.com.cn",
    ".api.aws",
    ".sagemaker.aws",
)


class InvalidRegionError(ValueError):
    """Raised when an invalid AWS region string is provided.

    This prevents SSRF attacks where a crafted region value
    (e.g., ``x@attacker.com:443/#``) could redirect SDK API calls
    to non-AWS hosts.
    """


def validate_region(region: str) -> str:
    """Validate that a region string is a well-formed AWS region name.

    Args:
        region: The region string to validate.

    Returns:
        The validated region string (unchanged).

    Raises:
        InvalidRegionError: If the region does not match the expected pattern.
    """
    if not isinstance(region, str) or not _VALID_REGION_PATTERN.match(region):
        raise InvalidRegionError(
            f"Invalid AWS region: {region!r}. "
            "Region must match pattern like 'us-east-1', 'eu-west-2', 'cn-north-1'."
        )
    return region


def validate_endpoint_url(url: str) -> str:
    """Validate that a constructed endpoint URL resolves to an AWS host.

    This is a defense-in-depth check that catches URL manipulation even if
    the region regex is somehow bypassed.

    Args:
        url: The constructed endpoint URL.

    Returns:
        The validated URL (unchanged).

    Raises:
        InvalidRegionError: If the URL hostname does not end with a trusted AWS domain.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if not any(hostname.endswith(d) for d in _AWS_DOMAINS):
        raise InvalidRegionError(
            f"Constructed endpoint resolves to non-AWS host: {hostname!r}"
        )
    return url
