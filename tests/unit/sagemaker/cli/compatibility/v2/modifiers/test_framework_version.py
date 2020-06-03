# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys

import pasta
import pytest

from sagemaker.cli.compatibility.v2.modifiers import framework_version
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


@pytest.fixture(autouse=True)
def skip_if_py2():
    # Remove once https://github.com/aws/sagemaker-python-sdk/issues/1461 is addressed.
    if sys.version_info.major < 3:
        pytest.skip("v2 migration script doesn't support Python 2.")


def test_node_should_be_modified_fw_constructor_no_fw_version():
    fw_constructors = (
        "TensorFlow()",
        "sagemaker.tensorflow.TensorFlow()",
        "TensorFlowModel()",
        "sagemaker.tensorflow.TensorFlowModel()",
        "MXNet()",
        "sagemaker.mxnet.MXNet()",
        "MXNetModel()",
        "sagemaker.mxnet.MXNetModel()",
        "Chainer()",
        "sagemaker.chainer.Chainer()",
        "ChainerModel()",
        "sagemaker.chainer.ChainerModel()",
        "PyTorch()",
        "sagemaker.pytorch.PyTorch()",
        "PyTorchModel()",
        "sagemaker.pytorch.PyTorchModel()",
        "SKLearn()",
        "sagemaker.sklearn.SKLearn()",
        "SKLearnModel()",
        "sagemaker.sklearn.SKLearnModel()",
    )

    modifier = framework_version.FrameworkVersionEnforcer()

    for constructor in fw_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_fw_constructor_with_fw_version():
    fw_constructors = (
        "TensorFlow(framework_version='2.2')",
        "sagemaker.tensorflow.TensorFlow(framework_version='2.2')",
        "TensorFlowModel(framework_version='1.10')",
        "sagemaker.tensorflow.TensorFlowModel(framework_version='1.10')",
        "MXNet(framework_version='1.6')",
        "sagemaker.mxnet.MXNet(framework_version='1.6')",
        "MXNetModel(framework_version='1.6')",
        "sagemaker.mxnet.MXNetModel(framework_version='1.6')",
        "PyTorch(framework_version='1.4')",
        "sagemaker.pytorch.PyTorch(framework_version='1.4')",
        "PyTorchModel(framework_version='1.4')",
        "sagemaker.pytorch.PyTorchModel(framework_version='1.4')",
        "Chainer(framework_version='5.0')",
        "sagemaker.chainer.Chainer(framework_version='5.0')",
        "ChainerModel(framework_version='5.0')",
        "sagemaker.chainer.ChainerModel(framework_version='5.0')",
        "SKLearn(framework_version='0.20.0')",
        "sagemaker.sklearn.SKLearn(framework_version='0.20.0')",
        "SKLearnModel(framework_version='0.20.0')",
        "sagemaker.sklearn.SKLearnModel(framework_version='0.20.0')",
    )

    modifier = framework_version.FrameworkVersionEnforcer()

    for constructor in fw_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is False


def test_node_should_be_modified_random_function_call():
    node = ast_call("sagemaker.session.Session()")
    modifier = framework_version.FrameworkVersionEnforcer()
    assert modifier.node_should_be_modified(node) is False


def test_modify_node_tf():
    classes = (
        "TensorFlow" "sagemaker.tensorflow.TensorFlow",
        "TensorFlowModel",
        "sagemaker.tensorflow.TensorFlowModel",
    )
    _test_modify_node(classes, "1.11.0")


def test_modify_node_mx():
    classes = ("MXNet", "sagemaker.mxnet.MXNet", "MXNetModel", "sagemaker.mxnet.MXNetModel")
    _test_modify_node(classes, "1.2.0")


def test_modify_node_chainer():
    classes = (
        "Chainer",
        "sagemaker.chainer.Chainer",
        "ChainerModel",
        "sagemaker.chainer.ChainerModel",
    )
    _test_modify_node(classes, "4.1.0")


def test_modify_node_pt():
    classes = (
        "PyTorch",
        "sagemaker.pytorch.PyTorch",
        "PyTorchModel",
        "sagemaker.pytorch.PyTorchModel",
    )
    _test_modify_node(classes, "0.4.0")


def test_modify_node_sklearn():
    classes = (
        "SKLearn",
        "sagemaker.sklearn.SKLearn",
        "SKLearnModel",
        "sagemaker.sklearn.SKLearnModel",
    )
    _test_modify_node(classes, "0.20.0")


def _test_modify_node(classes, default_version):
    modifier = framework_version.FrameworkVersionEnforcer()
    for cls in classes:
        node = ast_call("{}()".format(cls))
        modifier.modify_node(node)

        expected_result = "{}(framework_version='{}')".format(cls, default_version)
        assert expected_result == pasta.dump(node)
