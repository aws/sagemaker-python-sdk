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

This __init__.py file imports key modules used by inference scripts to prevent
Python module resolution conflicts with external serve.py files.

The imports below "prime" the module cache so that sagemaker.serve is recognized
as a package, preventing conflicts when inference scripts import from submodules.
"""

from __future__ import absolute_import

# Strategic imports to prime module cache and prevent serve.py conflicts
# Match V2's imports to ensure same priming behavior
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.model_builder import ModelBuilder

__all__ = ["InferenceSpec", "ModelServer", "ModelBuilder"]
