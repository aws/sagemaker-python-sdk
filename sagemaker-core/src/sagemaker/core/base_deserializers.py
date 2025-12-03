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
"""Backward compatibility module for base deserializers.

This module provides backward compatibility for code importing from
sagemaker.core.base_deserializers. The actual implementation is in
sagemaker.core.deserializers.

.. deprecated:: 3.0.0
    Use :mod:`sagemaker.core.deserializers` instead.
"""
from __future__ import absolute_import

import warnings

# Re-export all deserializers from the correct location
from sagemaker.core.deserializers import *  # noqa: F401, F403

# Issue deprecation warning
warnings.warn(
    "Importing from sagemaker.core.base_deserializers is deprecated. "
    "Use sagemaker.core.deserializers instead.",
    DeprecationWarning,
    stacklevel=2,
)
