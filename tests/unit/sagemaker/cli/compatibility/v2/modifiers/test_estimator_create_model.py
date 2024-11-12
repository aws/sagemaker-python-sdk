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

from sagemaker.cli.compatibility.v2.modifiers import renamed_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

ESTIMATORS = (
    "estimator",
    "chainer",
    "mxnet",
    "mx",
    "pytorch",
    "rl",
    "sklearn",
    "tensorflow",
    "tf",
    "xgboost",
    "xgb",
)


def test_node_should_be_modified():
    modifier = renamed_params.EstimatorCreateModelImageURIRenamer()

    for estimator in ESTIMATORS:
        call = "{}.create_model(image='my-image:latest')".format(estimator)
        assert modifier.node_should_be_modified(ast_call(call))


def test_node_should_be_modified_no_distribution():
    modifier = renamed_params.EstimatorCreateModelImageURIRenamer()

    for estimator in ESTIMATORS:
        call = "{}.create_model()".format(estimator)
        assert not modifier.node_should_be_modified(ast_call(call))


def test_node_should_be_modified_random_function_call():
    modifier = renamed_params.EstimatorCreateModelImageURIRenamer()
    assert not modifier.node_should_be_modified(ast_call("create_model()"))


def test_modify_node():
    node = ast_call("estimator.create_model(image=my_image)")
    modifier = renamed_params.EstimatorCreateModelImageURIRenamer()
    modifier.modify_node(node)

    expected = "estimator.create_model(image_uri=my_image)"
    assert expected == pasta.dump(node)
