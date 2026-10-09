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
"""Tests for the ``JobConfigDocument`` encoder."""
from __future__ import absolute_import

import json

import pytest

from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.job_config_document import convert_job_config_document_to_string
from sagemaker.core.workflow.parameters import ParameterString


def _resolve(join, values):
    """Splice concrete values into a Join, standing in for the pipeline service."""
    return "".join(
        values[piece.name] if isinstance(piece, ParameterString) else piece for piece in join.values
    )


def test_config_without_variables_is_plain_json():
    config = {"A": "x", "B": {"C": [1, 2]}, "D": None}
    document = convert_job_config_document_to_string(config)
    assert isinstance(document, str)
    assert json.loads(document) == config


def test_none_survives_as_null():
    document = convert_job_config_document_to_string({"A": None})
    assert document == '{"A": null}'


def test_single_variable_becomes_a_join_over_the_json_text():
    path = ParameterString(name="OutputPath")
    document = convert_job_config_document_to_string({"Out": {"S3OutputPath": path}, "N": 1})
    assert isinstance(document, Join)
    assert document.on == ""
    assert document.values == ['{"Out": {"S3OutputPath": "', path, '"}, "N": 1}']
    assert json.loads(_resolve(document, {"OutputPath": "s3://b/p"})) == {
        "Out": {"S3OutputPath": "s3://b/p"},
        "N": 1,
    }


def test_multiple_variables_keep_document_order():
    role = ParameterString(name="Role")
    path = ParameterString(name="OutputPath")
    document = convert_job_config_document_to_string({"Role": role, "Out": path})
    assert [v for v in document.values if isinstance(v, ParameterString)] == [role, path]
    assert json.loads(_resolve(document, {"Role": "r", "OutputPath": "p"})) == {
        "Role": "r",
        "Out": "p",
    }


def test_adjacent_variables_emit_no_empty_literal_between_them():
    first = ParameterString(name="First")
    second = ParameterString(name="Second")
    document = convert_job_config_document_to_string({"L": [first, second]})
    assert document.values == ['{"L": ["', first, '", "', second, '"]}']
    assert "" not in document.values


def test_same_variable_used_twice_is_spliced_twice():
    path = ParameterString(name="OutputPath")
    document = convert_job_config_document_to_string({"A": path, "B": path})
    assert document.values.count(path) == 2
    assert json.loads(_resolve(document, {"OutputPath": "p"})) == {"A": "p", "B": "p"}


def test_variable_lands_in_a_string_position():
    """A variable standing in for a number reaches the service quoted."""
    count = ParameterString(name="Count")
    document = convert_job_config_document_to_string({"N": count})
    assert json.loads(_resolve(document, {"Count": "3"})) == {"N": "3"}


def test_quote_in_a_resolved_value_malforms_the_document():
    """The documented limitation: values are spliced as raw text."""
    name = ParameterString(name="Name")
    document = convert_job_config_document_to_string({"Name": name})
    with pytest.raises(json.JSONDecodeError):
        json.loads(_resolve(document, {"Name": 'a"b'}))


def test_unserialisable_non_variable_still_raises_type_error():
    with pytest.raises(TypeError, match="not JSON serializable"):
        convert_job_config_document_to_string({"A": object()})
