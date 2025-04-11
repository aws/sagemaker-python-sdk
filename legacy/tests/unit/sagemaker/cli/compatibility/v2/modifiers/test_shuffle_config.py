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
        "sagemaker.session.ShuffleConfig(seed)",
        "session.ShuffleConfig(seed)",
    )


@pytest.fixture
def modified_constructors(constructors):
    return [c.replace("session", "inputs") for c in constructors]


def test_constructor_node_should_be_modified(constructors):
    modifier = training_input.ShuffleConfigModuleRenamer()
    for constructor in constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node)


def test_constructor_node_should_be_modified_random_call():
    modifier = training_input.ShuffleConfigModuleRenamer()
    node = ast_call("FileSystemInput()")
    assert not modifier.node_should_be_modified(node)


def test_constructor_modify_node(constructors, modified_constructors):
    modifier = training_input.ShuffleConfigModuleRenamer()

    for before, expected in zip(constructors, modified_constructors):
        node = ast_call(before)
        modifier.modify_node(node)
        assert expected == pasta.dump(node)


def test_import_from_node_should_be_modified_training_input():
    modifier = training_input.ShuffleConfigImportFromRenamer()
    node = ast_import("from sagemaker.session import ShuffleConfig")
    assert modifier.node_should_be_modified(node)


def test_import_from_node_should_be_modified_random_import():
    modifier = training_input.ShuffleConfigImportFromRenamer()
    node = ast_import("from sagemaker.session import Session")
    assert not modifier.node_should_be_modified(node)


def test_import_from_modify_node():
    modifier = training_input.ShuffleConfigImportFromRenamer()
    node = ast_import("from sagemaker.session import ShuffleConfig")

    modifier.modify_node(node)
    assert "from sagemaker.inputs import ShuffleConfig" == pasta.dump(node)
