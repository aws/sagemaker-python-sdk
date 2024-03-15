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
from __future__ import absolute_import

import os
import pathlib
import pytest

from mock import Mock, patch
from sagemaker.utils import _tmpdir
from sagemaker.remote_function.custom_file_filter import (
    CustomFileFilter,
    resolve_custom_file_filter_from_config_file,
    copy_workdir,
)
from sagemaker.config import load_sagemaker_config

from tests.unit import DATA_DIR

WORK_DIR = os.path.join(
    os.path.dirname(__file__), "../../../", "data", "remote_function", "workdir"
)


def test_workdir_config_default_value():
    workdir_config = CustomFileFilter()
    assert workdir_config.ignore_name_patterns == []
    assert workdir_config.workdir is not None


def test_resolve_custom_file_filter():
    sagemaker_session = Mock()
    sagemaker_session.sagemaker_config = load_sagemaker_config(
        additional_config_paths=[os.path.join(DATA_DIR, "remote_function")]
    )
    workdir_config = resolve_custom_file_filter_from_config_file(
        None, sagemaker_session=sagemaker_session
    )
    assert ["data", "test"] == workdir_config.ignore_name_patterns


@pytest.mark.parametrize(
    "ignore_name_patterns, expected",
    [
        (
            None,
            [
                "data",
                "data/data.csv",
                "module",
                "module/train.py",
                "test",
                "test/test.py",
            ],
        ),
        (["*"], []),
        (
            ["data", "test"],
            [
                "module",
                "module/train.py",
            ],
        ),
    ],
)
def test_copy_workdir(expected, ignore_name_patterns):
    workdir_config = CustomFileFilter(ignore_name_patterns=ignore_name_patterns)
    workdir_config._workdir = WORK_DIR

    with _tmpdir() as tmp_dir:
        target_dir = os.path.join(tmp_dir, "workdir")
        copy_workdir(target_dir, workdir_config)
        actual = sorted(str(path) for path in list(pathlib.Path(target_dir).rglob("*")))
        expected = sorted(os.path.join(target_dir, path) for path in expected)
        assert actual == sorted(expected)


@patch("os.getcwd", return_value=WORK_DIR)
def test_copy_workdir_default_behavior(mock_os_getcwd):
    expected = [
        "data",
        "module",
        "module/train.py",
        "test",
        "test/test.py",
    ]
    with _tmpdir() as tmp_dir:
        target_dir = os.path.join(tmp_dir, "workdir")
        copy_workdir(target_dir, None)
        actual = sorted(str(path) for path in list(pathlib.Path(target_dir).rglob("*")))
        expected = sorted(os.path.join(target_dir, path) for path in expected)
        assert actual == sorted(expected)
