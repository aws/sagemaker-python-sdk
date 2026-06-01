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
"""SageMaker resource classes.

This module re-exports all resource classes from the generated resources
module for backward compatibility. It also ensures the sagemaker_session
property is applied to the Model class.

The Model class exported from this module is guaranteed to have the
sagemaker_session property applied. This is the recommended import path
for Model when used with ModelStep or other pipeline components.
"""
from __future__ import absolute_import

# Re-export all generated resource classes for backward compatibility.
from sagemaker.core.generated.resources import *  # noqa: F401,F403

# Import the patched Model explicitly from model_resource. This import:
# 1. Triggers the patch on the Model class object (in-place modification)
# 2. Explicitly re-exports Model so that `from sagemaker.core.resources import Model`
#    is guaranteed to return the patched version.
# Since model_resource patches the same class object that was exported by
# the wildcard import above, both references point to the same (now patched) class.
from sagemaker.core.model_resource import Model  # noqa: F811
