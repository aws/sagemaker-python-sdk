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
"""
DEPRECATED: This module has been moved to sagemaker.core.training.configs

This is a backward compatibility shim. Please update your imports to:
    from sagemaker.core.training.configs import ...
"""
from __future__ import absolute_import

import warnings

# Backward compatibility: re-export from core
from sagemaker.core.training.configs import *  # noqa: F401, F403

warnings.warn(
    "sagemaker.train.configs has been moved to sagemaker.core.training.configs. "
    "Please update your imports. This shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)
