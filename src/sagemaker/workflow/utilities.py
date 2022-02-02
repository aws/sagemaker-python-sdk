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
"""Utilities to support workflow."""
from __future__ import absolute_import

from typing import List, Sequence, Union
import hashlib

from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)
from sagemaker.workflow.step_collections import StepCollection


def list_to_request(entities: Sequence[Union[Entity, StepCollection]]) -> List[RequestType]:
    """Get the request structure for list of entities.

    Args:
        entities (Sequence[Entity]): A list of entities.
    Returns:
        list: A request structure for a workflow service call.
    """
    request_dicts = []
    for entity in entities:
        if isinstance(entity, Entity):
            request_dicts.append(entity.to_request())
        elif isinstance(entity, StepCollection):
            request_dicts.extend(entity.request_dicts())
    return request_dicts


def hash_file(path: str) -> str:
    """Get the MD5 hash of a file.

    Args:
        path (str): The local path for the file.
    Returns:
        str: The MD5 hash of the file.
    """
    BUF_SIZE = 65536  # read in 64KiB chunks
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()
