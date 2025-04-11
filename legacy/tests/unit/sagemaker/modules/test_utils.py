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
"""Utils Tests."""
from __future__ import absolute_import

import pytest

from tests.unit import DATA_DIR
from sagemaker.modules.utils import (
    _is_valid_s3_uri,
    _is_valid_path,
    _get_unique_name,
    _get_repo_name_from_image,
)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "path": "s3://bucket/key",
            "path_type": "Any",
            "expected": True,
        },
        {
            "path": "s3://bucket/key",
            "path_type": "File",
            "expected": True,
        },
        {
            "path": "s3://bucket/key/",
            "path_type": "Directory",
            "expected": True,
        },
        {
            "path": "s3://bucket/key/",
            "path_type": "File",
            "expected": False,
        },
        {
            "path": "s3://bucket/key",
            "path_type": "Directory",
            "expected": False,
        },
        {
            "path": "/bucket/key",
            "path_type": "Any",
            "expected": False,
        },
    ],
)
def test_is_valid_s3_uri(test_case):
    assert _is_valid_s3_uri(test_case["path"], test_case["path_type"]) == test_case["expected"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "path": DATA_DIR,
            "path_type": "Any",
            "expected": True,
        },
        {
            "path": DATA_DIR,
            "path_type": "Directory",
            "expected": True,
        },
        {
            "path": f"{DATA_DIR}/dummy_input.txt",
            "path_type": "File",
            "expected": True,
        },
        {
            "path": f"{DATA_DIR}/dummy_input.txt",
            "path_type": "Directory",
            "expected": False,
        },
        {
            "path": f"{DATA_DIR}/non_existent",
            "path_type": "Any",
            "expected": False,
        },
    ],
)
def test_is_valid_path(test_case):
    assert _is_valid_path(test_case["path"], test_case["path_type"]) == test_case["expected"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "base": "test",
            "max_length": 5,
        },
        {
            "base": "1111111111" * 7,
            "max_length": None,
        },
    ],
)
def test_get_unique_name(test_case):
    assert (
        len(_get_unique_name(test_case["base"], test_case.get("max_length")))
        <= test_case["max_length"]
        if test_case.get("max_length")
        else 63
    )


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest",
            "expected": "my-custom-image",
        },
        {
            "image": "my-custom-image:latest",
            "expected": "my-custom-image",
        },
        {
            "image": "public.ecr.aws/docker/library/my-custom-image:latest",
            "expected": "my-custom-image",
        },
    ],
)
def test_get_repo_name_from_image(test_case):
    assert _get_repo_name_from_image(test_case["image"]) == test_case["expected"]
