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

from pathlib import Path
from typing import List, Sequence, Union, Set
import hashlib
from urllib.parse import unquote, urlparse
from _hashlib import HASH as Hash

from sagemaker.workflow.pipeline_context import _StepArguments
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)

BUF_SIZE = 65536  # 64KiB


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
    return _hash_file(path, hashlib.md5()).hexdigest()


def hash_files_or_dirs(paths: List[str]) -> str:
    """Get the MD5 hash of the contents of a list of files or directories.

    Hash is changed if:
       * input list is changed
       * new nested directories/files are added to any directory in the input list
       * nested directory/file names are changed for any of the inputted directories
       * content of files is edited

    Args:
        paths: List of file or directory paths
    Returns:
        str: The MD5 hash of the list of files or directories.
    """
    md5 = hashlib.md5()
    for path in sorted(paths):
        md5 = _hash_file_or_dir(path, md5)
    return md5.hexdigest()


def _hash_file_or_dir(path: str, md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        path: path of file or directory
    Returns:
        str: The MD5 hash of the file or directory
    """
    if isinstance(path, str) and path.lower().startswith("file://"):
        path = unquote(urlparse(path).path)
    md5.update(path.encode())
    if Path(path).is_dir():
        md5 = _hash_dir(path, md5)
    elif Path(path).is_file():
        md5 = _hash_file(path, md5)
    return md5


def _hash_dir(directory: Union[str, Path], md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        directory: path of the directory
    Returns:
        str: The MD5 hash of the directory
    """
    if not Path(directory).is_dir():
        raise ValueError(str(directory) + " is not a valid directory")
    for path in sorted(Path(directory).iterdir()):
        md5.update(path.name.encode())
        if path.is_file():
            md5 = _hash_file(path, md5)
        elif path.is_dir():
            md5 = _hash_dir(path, md5)
    return md5


def _hash_file(file: Union[str, Path], md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        file: path of the file
    Returns:
        str: The MD5 hash of the file
    """
    if isinstance(file, str) and file.lower().startswith("file://"):
        file = unquote(urlparse(file).path)
    if not Path(file).is_file():
        raise ValueError(str(file) + " is not a valid file")
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5


def validate_step_args_input(
    step_args: _StepArguments, expected_caller: Set[str], error_message: str
):
    """Validate the `_StepArguments` object which is passed into a pipeline step

    Args:
        step_args (_StepArguments): A `_StepArguments` object to be used for composing
            a pipeline step.
        expected_caller (Set[str]): The expected name of the caller function which is
            intercepted by the PipelineSession to get the step arguments.
        error_message (str): The error message to be thrown if the validation fails.
    """
    if not isinstance(step_args, _StepArguments):
        raise TypeError(error_message)
    if step_args.caller_name not in expected_caller:
        raise ValueError(error_message)
