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

from sagemaker.cli.compatibility.v2.modifiers import py_version
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


@pytest.fixture(autouse=True)
def skip_if_py2():
    # Remove once https://github.com/aws/sagemaker-python-sdk/issues/1461 is addressed.
    if sys.version_info.major < 3:
        pytest.skip("v2 migration script doesn't support Python 2.")


@pytest.fixture
def constructor_framework_templates():
    return (
        "TensorFlow({})",
        "sagemaker.tensorflow.TensorFlow({})",
        "sagemaker.tensorflow.estimator.TensorFlow({})",
        "MXNet({})",
        "sagemaker.mxnet.MXNet({})",
        "sagemaker.mxnet.estimator.MXNet({})",
        "Chainer({})",
        "sagemaker.chainer.Chainer({})",
        "sagemaker.chainer.estimator.Chainer({})",
        "PyTorch({})",
        "sagemaker.pytorch.PyTorch({})",
        "sagemaker.pytorch.estimator.PyTorch({})",
        "SKLearn({})",
        "sagemaker.sklearn.SKLearn({})",
        "sagemaker.sklearn.estimator.SKLearn({})",
    )


@pytest.fixture
def constructor_model_templates():
    return (
        "MXNetModel({})",
        "sagemaker.mxnet.MXNetModel({})",
        "sagemaker.mxnet.model.MXNetModel({})",
        "ChainerModel({})",
        "sagemaker.chainer.ChainerModel({})",
        "sagemaker.chainer.model.ChainerModel({})",
        "PyTorchModel({})",
        "sagemaker.pytorch.PyTorchModel({})",
        "sagemaker.pytorch.model.PyTorchModel({})",
    )


@pytest.fixture
def constructor_templates(constructor_framework_templates, constructor_model_templates):
    return tuple(list(constructor_framework_templates) + list(constructor_model_templates))


@pytest.fixture
def constructors_no_version(constructor_templates):
    return (ctr.format("") for ctr in constructor_templates)


@pytest.fixture
def constructors_with_version(constructor_templates):
    return (ctr.format("py_version='py3'") for ctr in constructor_templates)


@pytest.fixture
def constructors_with_image_name(constructor_framework_templates):
    return (ctr.format("image_name='my:image'") for ctr in constructor_framework_templates)


@pytest.fixture
def constructors_with_image(constructor_model_templates):
    return (ctr.format("image='my:image'") for ctr in constructor_model_templates)


@pytest.fixture
def constructors_version_not_needed():
    return (
        "TensorFlowModel()",
        "sagemaker.tensorflow.TensorFlowModel()",
        "sagemaker.tensorflow.model.TensorFlowModel()",
        "SKLearnModel()",
        "sagemaker.sklearn.SKLearnModel()",
        "sagemaker.sklearn.model.SKLearnModel()",
    )


def _test_modified(constructors, should_be):
    modifier = py_version.PyVersionEnforcer()
    for constructor in constructors:
        node = ast_call(constructor)
        if should_be:
            assert modifier.node_should_be_modified(node)
        else:
            assert not modifier.node_should_be_modified(node)


def test_node_should_be_modified_fw_constructor_no_version(constructors_no_version):
    _test_modified(constructors_no_version, should_be=True)


def test_node_should_be_modified_fw_constructor_with_version(constructors_with_version):
    _test_modified(constructors_with_version, should_be=False)


def test_node_should_be_modified_fw_constructor_with_image_name(constructors_with_image_name):
    _test_modified(constructors_with_image_name, should_be=False)


def test_node_should_be_modified_fw_constructor_with_image(constructors_with_image):
    _test_modified(constructors_with_image, should_be=False)


def test_node_should_be_modified_fw_constructor_not_needed(constructors_version_not_needed):
    _test_modified(constructors_version_not_needed, should_be=False)


def test_node_should_be_modified_random_function_call():
    _test_modified(["sagemaker.session.Session()"], should_be=False)


def test_modify_node(constructor_templates):
    modifier = py_version.PyVersionEnforcer()
    for template in constructor_templates:
        no_version, with_version = template.format(""), template.format("py_version='py3'")
        node = ast_call(no_version)
        modifier.modify_node(node)

        assert with_version == pasta.dump(node)
