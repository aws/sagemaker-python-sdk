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
# language governing permissions and limitations under the License.
from __future__ import absolute_import
import os
from pathlib import Path
from sagemaker._studio import (
    _append_project_tags,
    _find_config,
    _load_config,
    _parse_tags,
)


def test_find_config_cross_platform(tmpdir):
    """Test _find_config works correctly across different platforms."""
    # Create a completely separate directory for isolated tests
    import tempfile

    with tempfile.TemporaryDirectory() as isolated_root:
        # Setup test directory structure for positive tests
        config = tmpdir.join(".sagemaker-code-config")
        config.write('{"sagemakerProjectId": "proj-1234"}')

        # Test 1: Direct parent directory
        working_dir = tmpdir.mkdir("sub")
        found_path = _find_config(working_dir)
        assert found_path == config

        # Test 2: Deeply nested directories
        nested_dir = tmpdir.mkdir("deep").mkdir("nested").mkdir("path")
        found_path = _find_config(nested_dir)
        assert found_path == config

        # Test 3: Start from root directory
        import os

        root_dir = os.path.abspath(os.sep)
        found_path = _find_config(root_dir)
        assert found_path is None

        # Test 4: No config file in path - using truly isolated directory
        isolated_path = Path(isolated_root) / "nested" / "path"
        isolated_path.mkdir(parents=True)
        found_path = _find_config(isolated_path)
        assert found_path is None


def test_find_config_path_separators(tmpdir):
    """Test _find_config handles different path separator styles.

    Tests:
    1. Forward slashes
    2. Backslashes
    3. Mixed separators
    """
    # Setup
    config = tmpdir.join(".sagemaker-code-config")
    config.write('{"sagemakerProjectId": "proj-1234"}')
    base_path = str(tmpdir)

    # Test different path separator styles
    paths = [
        os.path.join(base_path, "dir1", "dir2"),  # OS native
        "/".join([base_path, "dir1", "dir2"]),  # Forward slashes
        "\\".join([base_path, "dir1", "dir2"]),  # Backslashes
        base_path + "/dir1\\dir2",  # Mixed
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)
        found_path = _find_config(path)
        assert found_path == config


def test_find_config(tmpdir):
    path = tmpdir.join(".sagemaker-code-config")
    path.write('{"sagemakerProjectId": "proj-1234"}')
    working_dir = tmpdir.mkdir("sub")

    found_path = _find_config(working_dir)
    assert found_path == path


def test_find_config_missing(tmpdir):
    working_dir = tmpdir.mkdir("sub")

    found_path = _find_config(working_dir)
    assert found_path is None


def test_load_config(tmpdir):
    path = tmpdir.join(".sagemaker-code-config")
    path.write('{"sagemakerProjectId": "proj-1234"}')

    config = _load_config(path)
    assert isinstance(config, dict)


def test_load_config_malformed(tmpdir):
    path = tmpdir.join(".sagemaker-code-config")
    path.write('{"proj')

    config = _load_config(path)
    assert config is None


def test_parse_tags():
    tags = _parse_tags(
        {
            "sagemakerProjectId": "proj-1234",
            "sagemakerProjectName": "proj-name",
            "foo": "abc",
        }
    )
    assert tags == [
        {"Key": "sagemaker:project-id", "Value": "proj-1234"},
        {"Key": "sagemaker:project-name", "Value": "proj-name"},
    ]


def test_parse_tags_missing():
    tags = _parse_tags(
        {
            "sagemakerProjectId": "proj-1234",
            "foo": "abc",
        }
    )
    assert tags is None


def test_append_project_tags(tmpdir):
    config = tmpdir.join(".sagemaker-code-config")
    config.write('{"sagemakerProjectId": "proj-1234", "sagemakerProjectName": "proj-name"}')
    working_dir = tmpdir.mkdir("sub")

    tags = _append_project_tags(None, working_dir)
    assert tags == [
        {"Key": "sagemaker:project-id", "Value": "proj-1234"},
        {"Key": "sagemaker:project-name", "Value": "proj-name"},
    ]

    tags = _append_project_tags([{"Key": "a", "Value": "b"}], working_dir)
    assert tags == [
        {"Key": "a", "Value": "b"},
        {"Key": "sagemaker:project-id", "Value": "proj-1234"},
        {"Key": "sagemaker:project-name", "Value": "proj-name"},
    ]

    tags = _append_project_tags(
        [{"Key": "sagemaker:project-id", "Value": "proj-1234"}], working_dir
    )
    assert tags == [
        {"Key": "sagemaker:project-id", "Value": "proj-1234"},
        {"Key": "sagemaker:project-name", "Value": "proj-name"},
    ]
