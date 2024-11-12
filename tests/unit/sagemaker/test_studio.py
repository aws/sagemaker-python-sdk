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

from sagemaker._studio import (
    _append_project_tags,
    _find_config,
    _load_config,
    _parse_tags,
)


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
