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

from sagemaker.cli.compatibility.v2.modifiers import tfs
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call, ast_import


def test_constructor_node_should_be_modified_tfs_constructor():
    tfs_constructors = (
        "sagemaker.tensorflow.serving.Model()",
        "sagemaker.tensorflow.serving.Predictor()",
        "Predictor()",
    )

    modifier = tfs.TensorFlowServingConstructorRenamer()

    for constructor in tfs_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_constructor_node_should_be_modified_random_function_call():
    modifier = tfs.TensorFlowServingConstructorRenamer()
    node = ast_call("Model()")
    assert modifier.node_should_be_modified(node) is False


def test_constructor_modify_node():
    modifier = tfs.TensorFlowServingConstructorRenamer()

    node = ast_call("sagemaker.tensorflow.serving.Model()")
    modifier.modify_node(node)
    assert "sagemaker.tensorflow.TensorFlowModel()" == pasta.dump(node)

    node = ast_call("sagemaker.tensorflow.serving.Predictor()")
    modifier.modify_node(node)
    assert "sagemaker.tensorflow.TensorFlowPredictor()" == pasta.dump(node)

    node = ast_call("Predictor()")
    modifier.modify_node(node)
    assert "TensorFlowPredictor()" == pasta.dump(node)


def test_import_from_node_should_be_modified_tfs_module():
    import_statements = (
        "from sagemaker.tensorflow.serving import Model, Predictor",
        "from sagemaker.tensorflow.serving import Predictor",
        "from sagemaker.tensorflow.serving import Model as tfsModel",
    )

    modifier = tfs.TensorFlowServingImportFromRenamer()

    for import_from in import_statements:
        node = ast_import(import_from)
        assert modifier.node_should_be_modified(node) is True


def test_import_from_node_should_be_modified_random_import():
    modifier = tfs.TensorFlowServingImportFromRenamer()
    node = ast_import("from sagemaker import Session")
    assert modifier.node_should_be_modified(node) is False


def test_import_from_modify_node():
    modifier = tfs.TensorFlowServingImportFromRenamer()

    node = ast_import("from sagemaker.tensorflow.serving import Model, Predictor")
    modifier.modify_node(node)
    expected_result = "from sagemaker.tensorflow import TensorFlowModel, TensorFlowPredictor"
    assert expected_result == pasta.dump(node)

    node = ast_import("from sagemaker.tensorflow.serving import Predictor")
    modifier.modify_node(node)
    assert "from sagemaker.tensorflow import TensorFlowPredictor" == pasta.dump(node)

    node = ast_import("from sagemaker.tensorflow.serving import Model as tfsModel")
    modifier.modify_node(node)
    assert "from sagemaker.tensorflow import TensorFlowModel as tfsModel" == pasta.dump(node)


def test_import_check_and_modify_node_tfs_import():
    modifier = tfs.TensorFlowServingImportRenamer()
    node = ast_import("import sagemaker.tensorflow.serving")
    modifier.check_and_modify_node(node)
    assert "import sagemaker.tensorflow" == pasta.dump(node)


def test_import_check_and_modify_node_random_import():
    modifier = tfs.TensorFlowServingImportRenamer()

    import_statement = "import random"
    node = ast_import(import_statement)
    modifier.check_and_modify_node(node)
    assert import_statement == pasta.dump(node)
