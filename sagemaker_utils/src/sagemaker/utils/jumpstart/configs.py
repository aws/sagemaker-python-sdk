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
        frozen=True,
    )


class JumpStartConfig(BaseConfig):
    """Configuration Class for JumpStart.

    Attributes:
        model_id (str): The model ID of the JumpStart model.
        model_version (Optional[str]): The version of the JumpStart model.
            Defaults to None.
        hub_name (Optional[str]): The name of the JumpStart hub. Defaults to None.
        accept_eula (Optional[bool]): Whether to accept the EULA. Defaults to None.
        training_config_name (Optional[str]): The name of the training configuration.
            Defaults to None.
        inference_config_name (Optional[str]): The name of the inference configuration.
            Defaults to None.
    """

    model_id: str
    model_version: Optional[str] = None
    hub_name: Optional[str] = None
    accept_eula: Optional[bool] = False
    training_config_name: Optional[str] = None
    inference_config_name: Optional[str] = None
