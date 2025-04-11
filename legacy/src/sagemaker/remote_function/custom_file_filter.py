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
"""SageMaker remote function client."""
from __future__ import absolute_import

import fnmatch
import os
import shutil
from typing import List, Optional, Callable, Union

from sagemaker.utils import resolve_value_from_config
from sagemaker.config.config_schema import REMOTE_FUNCTION_PATH, CUSTOM_FILE_FILTER


class CustomFileFilter:
    """Configuration that specifies how the local working directory should be packaged."""

    def __init__(self, *, ignore_name_patterns: List[str] = None):
        """Initialize a CustomFileFilter.

        Args:
            ignore_name_patterns (List[str]): ignore files or directories with names
              that match one of the glob-style patterns. Defaults to None.
        """

        if ignore_name_patterns is None:
            ignore_name_patterns = []

        self._workdir = os.getcwd()
        self._ignore_name_patterns = ignore_name_patterns

    @property
    def ignore_name_patterns(self):
        """Get the ignore name patterns."""
        return self._ignore_name_patterns

    @property
    def workdir(self):
        """Get the working directory."""
        return self._workdir


def resolve_custom_file_filter_from_config_file(
    direct_input: Union[Callable[[str, List], List], CustomFileFilter] = None,
    sagemaker_session=None,
) -> Union[Callable[[str, List], List], CustomFileFilter, None]:
    """Resolve the CustomFileFilter configuration from the config file.

    Args:
        direct_input (Callable[[str, List], List], CustomFileFilter): direct input from the user.
        sagemaker_session (sagemaker.session.Session): sagemaker session.
    Returns:
        CustomFileFilter: configuration that specifies how the local
            working directory should be packaged.
    """
    if direct_input is not None:
        return direct_input
    ignore_name_patterns = resolve_value_from_config(
        direct_input=None,
        config_path=".".join([REMOTE_FUNCTION_PATH, CUSTOM_FILE_FILTER, "IgnoreNamePatterns"]),
        default_value=None,
        sagemaker_session=sagemaker_session,
    )
    if ignore_name_patterns is not None:
        return CustomFileFilter(ignore_name_patterns=ignore_name_patterns)
    return None


def copy_workdir(
    dst: str,
    custom_file_filter: Optional[Union[Callable[[str, List], List], CustomFileFilter]] = None,
):
    """Copy the local working directory to the destination.

    Args:
        dst (str): destination path.
        custom_file_filter (Union[Callable[[str, List], List], CustomFileFilter): configuration that
            specifies how the local working directory should be packaged.
    """

    def _ignore_patterns(path: str, names: List):  # pylint: disable=unused-argument
        ignored_names = set()
        if custom_file_filter.ignore_name_patterns is not None:
            for pattern in custom_file_filter.ignore_name_patterns:
                ignored_names.update(fnmatch.filter(names, pattern))
        return ignored_names

    def _filter_non_python_files(path: str, names: List) -> List:
        """Ignore function for filtering out non python files."""
        to_ignore = []
        for name in names:
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                if not name.endswith(".py"):
                    to_ignore.append(name)
            elif os.path.isdir(full_path):
                if name == "__pycache__":
                    to_ignore.append(name)
            else:
                to_ignore.append(name)

        return to_ignore

    _ignore = None
    _src = os.getcwd()
    if not custom_file_filter:
        _ignore = _filter_non_python_files
    elif callable(custom_file_filter):
        _ignore = custom_file_filter
    elif isinstance(custom_file_filter, CustomFileFilter):
        _ignore = _ignore_patterns
        _src = custom_file_filter.workdir

    shutil.copytree(
        _src,
        dst,
        ignore=_ignore,
    )
