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
"""Legacy lineage module - compatibility shim.

This module provides backward compatibility for code using the old
`sagemaker.lineage` import path. All functionality has been moved to
`sagemaker.core.lineage`.

DEPRECATED: This module is deprecated. Use `sagemaker.core.lineage` instead.
"""
from __future__ import absolute_import

import warnings

# Show deprecation warning
warnings.warn(
    "The 'sagemaker.lineage' module is deprecated. " "Please use 'sagemaker.core.lineage' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from core.lineage for backward compatibility
from sagemaker.core.lineage import *  # noqa: F401, F403
