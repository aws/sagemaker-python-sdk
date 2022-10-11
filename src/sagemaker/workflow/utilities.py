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

import inspect
import logging
from functools import wraps
from pathlib import Path
from typing import List, Sequence, Union, Set, TYPE_CHECKING
import hashlib
from urllib.parse import unquote, urlparse
from _hashlib import HASH as Hash

from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.pipeline_context import _StepArguments
from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sagemaker.workflow.step_collections import StepCollection

BUF_SIZE = 65536  # 64KiB


def list_to_request(entities: Sequence[Union[Entity, "StepCollection"]]) -> List[RequestType]:
    """Get the request structure for list of entities.

    Args:
        entities (Sequence[Entity]): A list of entities.
    Returns:
        list: A request structure for a workflow service call.
    """
    from sagemaker.workflow.step_collections import StepCollection

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


def override_pipeline_parameter_var(func):
    """A decorator to override pipeline Parameters passed into a function

    This is a temporary decorator to override pipeline Parameter objects with their default value
    and display warning information to instruct users to update their code.

    This decorator can help to give a grace period for users to update their code when
    we make changes to explicitly prevent passing any pipeline variables to a function.

    We should remove this decorator after the grace period.
    """
    warning_msg_template = (
        "The input argument %s of function (%s) is a pipeline variable (%s), which is not allowed. "
        "The default_value of this Parameter object will be used to override it. "
        "Please make sure the default_value is valid."
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = "{}.{}".format(func.__module__, func.__name__)
        params = inspect.signature(func).parameters
        args = list(args)
        for i, (arg_name, _) in enumerate(params.items()):
            if i >= len(args):
                break
            if isinstance(args[i], Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(args[i]))
                args[i] = args[i].default_value
        args = tuple(args)

        for arg_name, value in kwargs.items():
            if isinstance(value, Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(value))
                kwargs[arg_name] = value.default_value
        return func(*args, **kwargs)

    return wrapper
