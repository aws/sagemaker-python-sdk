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
Local SageMaker Serve development package.

This __init__.py file is intentionally minimal to avoid conflicts with the existing
sagemaker.serve namespace from the V2 PyPI library that we still depend on.

For imports within this local package, use relative imports:
- from .model import ModelBase, _Model, FrameworkModel, ModelPackage
- from .model_builder import ModelBuilder

Do NOT import these classes in this __init__.py to avoid namespace conflicts.
"""

from __future__ import absolute_import

# Intentionally empty - no imports to avoid conflicts with existing sagemaker.serve
# Use relative imports within the package instead
