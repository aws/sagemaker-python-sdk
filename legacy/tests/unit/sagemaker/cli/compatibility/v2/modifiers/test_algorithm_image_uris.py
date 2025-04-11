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

from sagemaker.cli.compatibility.v2.modifiers import image_uris
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call, ast_import


@pytest.fixture
def methods():
    return (
        "get_image_uri('us-west-2', 'sagemaker-xgboost')",
        "sagemaker.get_image_uri(repo_region='us-west-2', repo_name='sagemaker-xgboost')",
        "sagemaker.amazon_estimator.get_image_uri('us-west-2', repo_name='sagemaker-xgboost')",
        "sagemaker.amazon.amazon_estimator.get_image_uri('us-west-2', 'sagemaker-xgboost', repo_version='1')",
    )


@pytest.fixture
def import_statements():
    return (
        "from sagemaker import get_image_uri",
        "from sagemaker.amazon_estimator import get_image_uri",
        "from sagemaker.amazon.amazon_estimator import get_image_uri",
    )


def test_method_node_should_be_modified(methods):
    modifier = image_uris.ImageURIRetrieveRefactor()
    for method in methods:
        node = ast_call(method)
        assert modifier.node_should_be_modified(node)


def test_methodnode_should_be_modified_random_call():
    modifier = image_uris.ImageURIRetrieveRefactor()
    node = ast_call("create_image_uri()")
    assert not modifier.node_should_be_modified(node)


def test_method_modify_node(methods, caplog):
    modifier = image_uris.ImageURIRetrieveRefactor()

    method = "get_image_uri('us-west-2', 'xgboost')"
    node = ast_call(method)
    modifier.modify_node(node)
    assert "image_uris.retrieve('xgboost', 'us-west-2')" == pasta.dump(node)

    method = "amazon_estimator.get_image_uri('us-west-2', 'xgboost')"
    node = ast_call(method)
    modifier.modify_node(node)
    assert "image_uris.retrieve('xgboost', 'us-west-2')" == pasta.dump(node)

    method = "sagemaker.get_image_uri(repo_region='us-west-2', repo_name='xgboost')"
    node = ast_call(method)
    modifier.modify_node(node)
    assert "sagemaker.image_uris.retrieve('xgboost', 'us-west-2')" == pasta.dump(node)

    method = "sagemaker.amazon_estimator.get_image_uri('us-west-2', repo_name='xgboost')"
    node = ast_call(method)
    modifier.modify_node(node)
    assert "sagemaker.image_uris.retrieve('xgboost', 'us-west-2')" == pasta.dump(node)

    method = (
        "sagemaker.amazon.amazon_estimator.get_image_uri('us-west-2', 'xgboost', repo_version='1')"
    )
    node = ast_call(method)
    modifier.modify_node(node)
    assert "sagemaker.image_uris.retrieve('xgboost', 'us-west-2', '1')" == pasta.dump(node)


def test_import_from_node_should_be_modified_image_uris_input(import_statements):
    modifier = image_uris.ImageURIRetrieveImportFromRenamer()

    statement = "from sagemaker import get_image_uri"
    node = ast_import(statement)
    assert modifier.node_should_be_modified(node)

    statement = "from sagemaker.amazon_estimator import get_image_uri"
    node = ast_import(statement)
    assert modifier.node_should_be_modified(node)

    statement = "from sagemaker.amazon.amazon_estimator import get_image_uri"
    node = ast_import(statement)
    assert modifier.node_should_be_modified(node)


def test_import_from_node_should_be_modified_random_import():
    modifier = image_uris.ImageURIRetrieveImportFromRenamer()
    node = ast_import("from sagemaker.amazon_estimator import registry")
    assert not modifier.node_should_be_modified(node)


def test_import_from_modify_node(import_statements):
    modifier = image_uris.ImageURIRetrieveImportFromRenamer()
    expected_result = "from sagemaker import image_uris"

    for import_statement in import_statements:
        node = ast_import(import_statement)
        modifier.modify_node(node)
        assert expected_result == pasta.dump(node)
