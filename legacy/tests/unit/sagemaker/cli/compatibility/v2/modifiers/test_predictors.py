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

from sagemaker.cli.compatibility.v2.modifiers import predictors
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call, ast_import


@pytest.fixture
def base_constructors():
    return (
        "sagemaker.predictor.RealTimePredictor(endpoint='a')",
        "sagemaker.RealTimePredictor(endpoint='b')",
        "RealTimePredictor(endpoint='c')",
    )


@pytest.fixture
def sparkml_constructors():
    return (
        "sagemaker.sparkml.model.SparkMLPredictor(endpoint='a')",
        "sagemaker.sparkml.SparkMLPredictor(endpoint='b')",
        "SparkMLPredictor(endpoint='c')",
    )


@pytest.fixture
def other_constructors():
    return (
        "sagemaker.amazon.knn.KNNPredictor(endpoint='a')",
        "sagemaker.KNNPredictor(endpoint='b')",
        "KNNPredictor(endpoint='c')",
    )


@pytest.fixture
def import_statements():
    return (
        "from sagemaker.predictor import RealTimePredictor",
        "from sagemaker import RealTimePredictor",
    )


def test_constructor_node_should_be_modified_base(base_constructors):
    modifier = predictors.PredictorConstructorRefactor()
    for constructor in base_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node)


def test_constructor_node_should_be_modified_sparkml(sparkml_constructors):
    modifier = predictors.PredictorConstructorRefactor()
    for constructor in sparkml_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node)


def test_constructor_node_should_be_modified_other(other_constructors):
    modifier = predictors.PredictorConstructorRefactor()
    for constructor in other_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node)


def test_constructor_node_should_be_modified_random_call():
    modifier = predictors.PredictorConstructorRefactor()
    node = ast_call("Model()")
    assert not modifier.node_should_be_modified(node)


def test_constructor_modify_node():
    modifier = predictors.PredictorConstructorRefactor()

    node = ast_call("sagemaker.RealTimePredictor(endpoint='a')")
    modifier.modify_node(node)
    assert "sagemaker.Predictor(endpoint_name='a')" == pasta.dump(node)

    node = ast_call("RealTimePredictor(endpoint='a')")
    modifier.modify_node(node)
    assert "Predictor(endpoint_name='a')" == pasta.dump(node)

    node = ast_call("sagemaker.amazon.kmeans.KMeansPredictor(endpoint='a')")
    modifier.modify_node(node)
    assert "sagemaker.amazon.kmeans.KMeansPredictor(endpoint_name='a')" == pasta.dump(node)

    node = ast_call("KMeansPredictor(endpoint='a')")
    modifier.modify_node(node)
    assert "KMeansPredictor(endpoint_name='a')" == pasta.dump(node)


def test_import_from_node_should_be_modified_predictor_module(import_statements):
    modifier = predictors.PredictorImportFromRenamer()
    for statement in import_statements:
        node = ast_import(statement)
        assert modifier.node_should_be_modified(node)


def test_import_from_node_should_be_modified_random_import():
    modifier = predictors.PredictorImportFromRenamer()
    node = ast_import("from sagemaker import Session")
    assert not modifier.node_should_be_modified(node)


def test_import_from_modify_node():
    modifier = predictors.PredictorImportFromRenamer()

    node = ast_import(
        "from sagemaker.predictor import ClassThatHasntBeenRenamed, RealTimePredictor"
    )
    modifier.modify_node(node)
    expected_result = "from sagemaker.predictor import ClassThatHasntBeenRenamed, Predictor"
    assert expected_result == pasta.dump(node)

    node = ast_import("from sagemaker.predictor import RealTimePredictor as RTP")
    modifier.modify_node(node)
    expected_result = "from sagemaker.predictor import Predictor as RTP"
    assert expected_result == pasta.dump(node)
