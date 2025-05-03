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
"""This module contains utilites for JumpStart model metadata."""
from __future__ import absolute_import

from pydantic import BaseModel, ConfigDict
from typing import Optional


class BaseConfig(BaseModel):
    """BaseConfig"""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )


class JumpStartConfig(BaseConfig):
    """Configuration Class for JumpStart."""

    model_id: str
    model_version: Optional[str] = None
    hub_name: Optional[str] = None
    accept_eula: Optional[bool] = None
    tolerate_vulnerable_model: Optional[bool] = None
    tolerate_deprecated_model: Optional[bool] = None
    training_config_name: Optional[str] = None
    inference_config_name: Optional[str] = None
