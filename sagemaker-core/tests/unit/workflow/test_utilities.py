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
"""Tests for workflow utilities."""
from __future__ import absolute_import

import os
import tempfile

import pytest

from sagemaker.core.workflow.utilities import (
    hash_files_or_dirs,
    hash_source_dir_and_dependencies,
)


def test_hash_source_dir_and_dependencies_with_none_dependencies():
    """Test that hash_source_dir_and_dependencies does not raise TypeError when dependencies is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple file in the source directory
        source_file = os.path.join(tmpdir, "script.py")
        with open(source_file, "w") as f:
            f.write("print('hello')")

        # This should not raise a TypeError
        result = hash_source_dir_and_dependencies(source_dir=tmpdir, dependencies=None)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


def test_hash_source_dir_and_dependencies_with_empty_dependencies():
    """Test that hash_source_dir_and_dependencies works with an empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "script.py")
        with open(source_file, "w") as f:
            f.write("print('hello')")

        result = hash_source_dir_and_dependencies(source_dir=tmpdir, dependencies=[])
        assert result is not None
        assert isinstance(result, str)


def test_hash_source_dir_and_dependencies_with_dependencies():
    """Test that hash_source_dir_and_dependencies works with actual dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir)
        source_file = os.path.join(source_dir, "script.py")
        with open(source_file, "w") as f:
            f.write("print('hello')")

        dep_file = os.path.join(tmpdir, "dep.py")
        with open(dep_file, "w") as f:
            f.write("import os")

        result = hash_source_dir_and_dependencies(
            source_dir=source_dir, dependencies=[dep_file]
        )
        assert result is not None
        assert isinstance(result, str)


def test_hash_files_or_dirs_with_file():
    """Test hash_files_or_dirs with a single file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = hash_files_or_dirs([test_file])
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest length
