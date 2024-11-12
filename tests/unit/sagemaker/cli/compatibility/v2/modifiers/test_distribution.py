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


def test_node_should_be_modified():
    constructors = (
        "TensorFlow(distributions={})",
        "sagemaker.tensorflow.TensorFlow(distributions={})",
        "sagemaker.tensorflow.estimator.TensorFlow(distributions={})",
        "MXNet(distributions={})",
        "sagemaker.mxnet.MXNet(distributions={})",
        "sagemaker.mxnet.estimator.MXNet(distributions={})",
    )

    modifier = renamed_params.DistributionParameterRenamer()

    for call in constructors:
        assert modifier.node_should_be_modified(ast_call(call))


def test_node_should_be_modified_no_distribution():
    constructors = (
        "TensorFlow()",
        "sagemaker.tensorflow.TensorFlow()",
        "sagemaker.tensorflow.estimator.TensorFlow()",
        "MXNet()",
        "sagemaker.mxnet.MXNet()",
        "sagemaker.mxnet.estimator.MXNet()",
    )

    modifier = renamed_params.DistributionParameterRenamer()

    for call in constructors:
        assert not modifier.node_should_be_modified(ast_call(call))


def test_node_should_be_modified_random_function_call():
    modifier = renamed_params.DistributionParameterRenamer()
    assert not modifier.node_should_be_modified(ast_call("Session()"))


def test_modify_node():
    node = ast_call("TensorFlow(distributions={'parameter_server': {'enabled': True}})")
    modifier = renamed_params.DistributionParameterRenamer()
    modifier.modify_node(node)

    expected = "TensorFlow(distribution={'parameter_server': {'enabled': True}})"
    assert expected == pasta.dump(node)
