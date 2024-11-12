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

import pasta
import pytest

from sagemaker.cli.compatibility.v2.modifiers import training_input
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call, ast_import


@pytest.fixture
def constructors():
    return (
        "sagemaker.session.s3_input(s3_data='s3://a')",
        "sagemaker.inputs.s3_input(s3_data='s3://a')",
        "sagemaker.s3_input(s3_data='s3://a')",
        "session.s3_input(s3_data='s3://a')",
        "inputs.s3_input(s3_data='s3://a')",
        "s3_input(s3_data='s3://a')",
    )


@pytest.fixture
def import_statements():
    return (
        "from sagemaker.session import s3_input",
        "from sagemaker.inputs import s3_input",
        "from sagemaker import s3_input",
    )


def test_constructor_node_should_be_modified(constructors):
    modifier = training_input.TrainingInputConstructorRefactor()
    for constructor in constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node)


def test_constructor_node_should_be_modified_random_call():
    modifier = training_input.TrainingInputConstructorRefactor()
    node = ast_call("FileSystemInput()")
    assert not modifier.node_should_be_modified(node)


def test_constructor_modify_node():
    modifier = training_input.TrainingInputConstructorRefactor()

    node = ast_call("s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "TrainingInput(s3_data='s3://a')" == pasta.dump(node)

    node = ast_call("sagemaker.s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "sagemaker.TrainingInput(s3_data='s3://a')" == pasta.dump(node)

    node = ast_call("session.s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "inputs.TrainingInput(s3_data='s3://a')" == pasta.dump(node)

    node = ast_call("inputs.s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "inputs.TrainingInput(s3_data='s3://a')" == pasta.dump(node)

    node = ast_call("sagemaker.inputs.s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "sagemaker.inputs.TrainingInput(s3_data='s3://a')" == pasta.dump(node)

    node = ast_call("sagemaker.session.s3_input(s3_data='s3://a')")
    modifier.modify_node(node)
    assert "sagemaker.inputs.TrainingInput(s3_data='s3://a')" == pasta.dump(node)


def test_import_from_node_should_be_modified_training_input(import_statements):
    modifier = training_input.TrainingInputImportFromRenamer()
    for statement in import_statements:
        node = ast_import(statement)
        assert modifier.node_should_be_modified(node)


def test_import_from_node_should_be_modified_random_import():
    modifier = training_input.TrainingInputImportFromRenamer()
    node = ast_import("from sagemaker.session import Session")
    assert not modifier.node_should_be_modified(node)


def test_import_from_modify_node():
    modifier = training_input.TrainingInputImportFromRenamer()

    node = ast_import("from sagemaker import s3_input")
    modifier.modify_node(node)
    expected_result = "from sagemaker import TrainingInput"
    assert expected_result == pasta.dump(node)

    node = ast_import("from sagemaker.inputs import s3_input as training_input")
    modifier.modify_node(node)
    expected_result = "from sagemaker.inputs import TrainingInput as training_input"
    assert expected_result == pasta.dump(node)

    node = ast_import("from sagemaker.session import s3_input as training_input")
    modifier.modify_node(node)
    expected_result = "from sagemaker.inputs import TrainingInput as training_input"
    assert expected_result == pasta.dump(node)
