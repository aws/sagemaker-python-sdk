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
"""This module contains logic for setting defaults in ModelBuilder."""
from __future__ import absolute_import

from typing import Optional, Dict, List, Union
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from dataclasses import dataclass


@dataclass
class Network:
    """Network configuration for model deployment."""
    subnets: Optional[List[str]] = None
    security_group_ids: Optional[List[str]] = None
    enable_network_isolation: bool = False
    vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None


@dataclass
class Compute:
    """Compute configuration for model deployment."""
    instance_type: Optional[str]
    instance_count: Optional[int] = 1