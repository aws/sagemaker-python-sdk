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
"""Resource creation module for tracking ARNs of resources created via SDK calls."""
from __future__ import absolute_import

# Maps class name (string) to the attribute name holding the resource ARN.
# String-based keys avoid cross-package imports and circular dependencies.
_RESOURCE_ARN_ATTRIBUTES = {
    "TrainingJob": "training_job_arn",
}


def get_resource_arn(response):
    """Extract the ARN from a SDK response object if available.

    Uses string-based type name lookup to avoid cross-package imports.

    Args:
        response: The return value of a _telemetry_emitter-decorated function.

    Returns:
        str: The ARN string if available, otherwise None.
    """
    if response is None:
        return None

    arn_attr = _RESOURCE_ARN_ATTRIBUTES.get(type(response).__name__)
    if not arn_attr:
        return None

    arn = getattr(response, arn_attr, None)

    # Guard against Unassigned sentinel used in resources.py
    if not arn or type(arn).__name__ == "Unassigned":
        return None

    return str(arn)
