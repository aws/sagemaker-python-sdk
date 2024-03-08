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
"""Accessors to retrieve instance types GPU info."""
from __future__ import absolute_import

import json
import os
from typing import Dict


def retrieve(region: str) -> Dict[str, Dict[str, int]]:
    """Retrieves instance types GPU info of the given region.

    Args:
        region (str): The AWS region.

    Returns:
        dict[str, dict[str, int]]: A dictionary that contains instance types as keys
                                   and GPU info as values or empty dictionary if the
                                   config for the given region is not found.

    Raises:
        ValueError: If no config found.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "image_uri_config", "instance_gpu_info.json"
    )
    try:
        with open(config_path) as f:
            instance_types_gpu_info_config = json.load(f)
        return instance_types_gpu_info_config.get(region, {})
    except FileNotFoundError:
        raise ValueError("Could not find instance types gpu info.")
