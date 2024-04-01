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

HF_INVALID_TASK = "not-present-task"


def test_retrieve_local_schemas_text_generation_success():
    inputs, outputs = task.retrieve_local_schemas("text-generation")

    assert inputs == {"inputs": "Hello, I'm a language model", "parameters": {}}
    assert outputs == [
        {
            "generated_text": "Hello, I'm a language modeler. So while writing this, when I went out to "
            "meet my wife or come home she told me that my"
        }
    ]


def test_retrieve_local_schemas_text_classification_success():
    inputs, outputs = task.retrieve_local_schemas("text-classification")

    assert inputs == {
        "inputs": "Where is the capital of France?, Paris is the capital of France.",
        "parameters": {},
    }
    assert outputs == [{"label": "entailment", "score": 0.997}]


def test_retrieve_local_schemas_throws():
    with pytest.raises(ValueError, match=f"Could not find {HF_INVALID_TASK} I/O schema."):
        task.retrieve_local_schemas(HF_INVALID_TASK)


@patch("builtins.open")
def test_retrieve_local_schemas_file_not_found(mock_open):
    mock_open.side_effect = FileNotFoundError
    with pytest.raises(ValueError, match="Could not find tasks config file."):
        task.retrieve_local_schemas(HF_INVALID_TASK)
