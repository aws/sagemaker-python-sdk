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

from unittest.mock import patch

import pytest

from sagemaker.serve.utils import task

EXPECTED_INPUTS = {"inputs": "Paris is the [MASK] of France.", "parameters": {}}
EXPECTED_OUTPUTS = [{"sequence": "Paris is the capital of France.", "score": 0.7}]
HF_INVALID_TASK = "not-present-task"


def test_retrieve_local_schemas_success():
    inputs, outputs = task.retrieve_local_schemas("fill-mask")

    assert inputs == EXPECTED_INPUTS
    assert outputs == EXPECTED_OUTPUTS


def test_retrieve_local_schemas_text_generation_success():
    inputs, outputs = task.retrieve_local_schemas("text-generation")

    assert inputs is not None
    assert outputs is not None


def test_retrieve_local_schemas_throws():
    with pytest.raises(ValueError, match=f"Could not find {HF_INVALID_TASK} I/O schema."):
        task.retrieve_local_schemas(HF_INVALID_TASK)


@patch("builtins.open")
def test_retrieve_local_schemas_file_not_found(mock_open):
    mock_open.side_effect = FileNotFoundError
    with pytest.raises(ValueError, match="Could not find tasks config file."):
        task.retrieve_local_schemas(HF_INVALID_TASK)
