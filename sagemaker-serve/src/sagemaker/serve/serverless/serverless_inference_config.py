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
Backward compatibility shim for sagemaker.serve.serverless.serverless_inference_config

This module has been moved to sagemaker.core.inference_config.
This file provides backward compatibility by re-exporting the class from its new location.

DEPRECATED: Import from sagemaker.core.inference_config instead.
"""
from __future__ import absolute_import

import warnings

# Import from new Core location
from sagemaker.core.inference_config import ServerlessInferenceConfig

# Emit deprecation warning when this module is imported
warnings.warn(
    "Importing from sagemaker.serve.serverless.serverless_inference_config is deprecated. "
    "The ServerlessInferenceConfig class has been moved to sagemaker.core.inference_config. "
    "Please update your imports to:\n"
    "  from sagemaker.core.inference_config import ServerlessInferenceConfig\n"
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['ServerlessInferenceConfig']
