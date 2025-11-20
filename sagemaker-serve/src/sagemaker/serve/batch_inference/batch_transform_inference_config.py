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
"""Config Classes for taking in parameters for Batch Inference"""

from __future__ import absolute_import
from pydantic import BaseModel


class BatchTransformInferenceConfig(BaseModel):
    """Config class for Batch Transform Inference

    * Can be used to deploy from ModelBuilder
    """

    instance_count: int
    instance_type: str
    output_path: str
