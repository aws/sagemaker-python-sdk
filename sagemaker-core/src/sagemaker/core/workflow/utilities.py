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
"""Workflow utilities for SageMaker pipelines."""
from __future__ import absolute_import

import hashlib
import os
from typing import List, Optional


def hash_files_or_dirs(paths: List[str]) -> str:
    """Compute a hash of the contents of files or directories.

    Args:
        paths (List[str]): List of file or directory paths to hash.

    Returns:
        str: A hex digest of the hash of the contents.
    """
    md5_hash = hashlib.md5()
    for path in sorted(paths):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in sorted(files):
                    file_path = os.path.join(root, f)
                    with open(file_path, "rb") as fh:
                        for chunk in iter(lambda: fh.read(4096), b""):
                            md5_hash.update(chunk)
        elif os.path.isfile(path):
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(4096), b""):
                    md5_hash.update(chunk)
    return md5_hash.hexdigest()


def hash_source_dir_and_dependencies(
    source_dir: str, dependencies: Optional[List[str]] = None
) -> str:
    """Compute a hash of the source directory and dependencies.

    Args:
        source_dir (str): Path to the source directory.
        dependencies (Optional[List[str]]): List of dependency paths.
            Defaults to an empty list if None.

    Returns:
        str: A hex digest of the hash of the contents.
    """
    # Default dependencies to an empty list if None to avoid TypeError
    # when concatenating with [source_dir]
    dependencies = dependencies or []
    return hash_files_or_dirs([source_dir] + dependencies)
