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
"""Accessors to retrieve task fallback input/output schema"""
from __future__ import absolute_import

import json
import os
from typing import Any, Tuple


def retrieve_local_schemas(task: str) -> Tuple[Any, Any]:
    """Retrieves task sample inputs and outputs locally.

    Args:
        task (str): Required, the task name

    Returns:
        Tuple[Any, Any]: A tuple that contains the sample input,
        at index 0, and output schema, at index 1.

    Raises:
        ValueError: If no tasks config found or the task does not exist in the local config.
    """
    config_dir = os.path.dirname(os.path.dirname(__file__))
    task_io_config_path = os.path.join(config_dir, "schema", "task.json")
    try:
        with open(task_io_config_path) as f:
            task_io_config = json.load(f)
            task_io_schemas = task_io_config.get(task, None)

            if task_io_schemas is None:
                raise ValueError(f"Could not find {task} I/O schema.")

            sample_schema = (
                task_io_schemas["sample_inputs"]["properties"],
                task_io_schemas["sample_outputs"]["properties"],
            )
        return sample_schema
    except FileNotFoundError:
        raise ValueError("Could not find tasks config file.")
