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
        "sagemaker.tensorflow.estimator.TensorFlow()",
        "TensorFlowModel()",
        "sagemaker.tensorflow.TensorFlowModel()",
        "sagemaker.tensorflow.model.TensorFlowModel()",
        "MXNet()",
        "sagemaker.mxnet.MXNet()",
        "sagemaker.mxnet.estimator.MXNet()",
        "MXNetModel()",
        "sagemaker.mxnet.MXNetModel()",
        "sagemaker.mxnet.model.MXNetModel()",
        "Chainer()",
        "sagemaker.chainer.Chainer()",
        "sagemaker.chainer.estimator.Chainer()",
        "ChainerModel()",
        "sagemaker.chainer.ChainerModel()",
        "sagemaker.chainer.model.ChainerModel()",
        "PyTorch()",
        "sagemaker.pytorch.PyTorch()",
        "sagemaker.pytorch.estimator.PyTorch()",
        "PyTorchModel()",
        "sagemaker.pytorch.PyTorchModel()",
        "sagemaker.pytorch.model.PyTorchModel()",
        "SKLearn()",
        "sagemaker.sklearn.SKLearn()",
        "sagemaker.sklearn.estimator.SKLearn()",
        "SKLearnModel()",
        "sagemaker.sklearn.SKLearnModel()",
        "sagemaker.sklearn.model.SKLearnModel()",
    )

    modifier = framework_version.FrameworkVersionEnforcer()

    for constructor in fw_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_fw_constructor_with_fw_version():
    fw_constructors = (
        "TensorFlow(framework_version='2.2')",
        "sagemaker.tensorflow.TensorFlow(framework_version='2.2')",
        "sagemaker.tensorflow.estimator.TensorFlow(framework_version='2.2')",
        "TensorFlowModel(framework_version='1.10')",
        "sagemaker.tensorflow.TensorFlowModel(framework_version='1.10')",
        "sagemaker.tensorflow.model.TensorFlowModel(framework_version='1.10')",
        "MXNet(framework_version='1.6')",
        "sagemaker.mxnet.MXNet(framework_version='1.6')",
        "sagemaker.mxnet.estimator.MXNet(framework_version='1.6')",
        "MXNetModel(framework_version='1.6')",
        "sagemaker.mxnet.MXNetModel(framework_version='1.6')",
        "sagemaker.mxnet.model.MXNetModel(framework_version='1.6')",
        "PyTorch(framework_version='1.4')",
        "sagemaker.pytorch.PyTorch(framework_version='1.4')",
        "sagemaker.pytorch.estimator.PyTorch(framework_version='1.4')",
        "PyTorchModel(framework_version='1.4')",
        "sagemaker.pytorch.PyTorchModel(framework_version='1.4')",
        "sagemaker.pytorch.model.PyTorchModel(framework_version='1.4')",
        "Chainer(framework_version='5.0')",
        "sagemaker.chainer.Chainer(framework_version='5.0')",
        "sagemaker.chainer.estimator.Chainer(framework_version='5.0')",
        "ChainerModel(framework_version='5.0')",
        "sagemaker.chainer.ChainerModel(framework_version='5.0')",
        "sagemaker.chainer.model.ChainerModel(framework_version='5.0')",
        "SKLearn(framework_version='0.20.0')",
        "sagemaker.sklearn.SKLearn(framework_version='0.20.0')",
        "sagemaker.sklearn.estimator.SKLearn(framework_version='0.20.0')",
        "SKLearnModel(framework_version='0.20.0')",
        "sagemaker.sklearn.SKLearnModel(framework_version='0.20.0')",
        "sagemaker.sklearn.model.SKLearnModel(framework_version='0.20.0')",
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
    _test_modify_node("TensorFlow", "1.11.0")


def test_modify_node_mx():
    _test_modify_node("MXNet", "1.2.0")


def test_modify_node_chainer():
    _test_modify_node("Chainer", "4.1.0")


def test_modify_node_pt():
    _test_modify_node("PyTorch", "0.4.0")


def test_modify_node_sklearn():
    _test_modify_node("SKLearn", "0.20.0")


def _test_modify_node(framework, default_version):
    modifier = framework_version.FrameworkVersionEnforcer()

    classes = (
        "{}".format(framework),
        "sagemaker.{}.{}".format(framework.lower(), framework),
        "sagemaker.{}.estimator.{}".format(framework.lower(), framework),
        "{}Model".format(framework),
        "sagemaker.{}.{}Model".format(framework.lower(), framework),
        "sagemaker.{}.model.{}Model".format(framework.lower(), framework),
    )
    for cls in classes:
        node = ast_call("{}()".format(cls))
        modifier.modify_node(node)

        expected_result = "{}(framework_version='{}')".format(cls, default_version)
        assert expected_result == pasta.dump(node)
