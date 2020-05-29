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

import pasta

from sagemaker.cli.compatibility.v2.modifiers import tf_legacy_mode


def test_node_should_be_modified_tf_constructor_legacy_mode():
    tf_legacy_mode_constructors = (
        "TensorFlow(script_mode=False)",
        "TensorFlow(script_mode=None)",
        "TensorFlow(py_version='py2')",
        "TensorFlow()",
        "sagemaker.tensorflow.TensorFlow(script_mode=False)",
        "sagemaker.tensorflow.TensorFlow(script_mode=None)",
        "sagemaker.tensorflow.TensorFlow(py_version='py2')",
        "sagemaker.tensorflow.TensorFlow()",
    )

    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_legacy_mode_constructors:
        node = _ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_tf_constructor_script_mode():
    tf_script_mode_constructors = (
        "TensorFlow(script_mode=True)",
        "TensorFlow(py_version='py3')",
        "TensorFlow(py_version='py37')",
        "TensorFlow(py_version='py3', script_mode=False)",
        "sagemaker.tensorflow.TensorFlow(script_mode=True)",
        "sagemaker.tensorflow.TensorFlow(py_version='py3')",
        "sagemaker.tensorflow.TensorFlow(py_version='py37')",
        "sagemaker.tensorflow.TensorFlow(py_version='py3', script_mode=False)",
    )

    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_script_mode_constructors:
        node = _ast_call(constructor)
        assert modifier.node_should_be_modified(node) is False


def test_node_should_be_modified_random_function_call():
    node = _ast_call("MXNet(py_version='py3')")
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    assert modifier.node_should_be_modified(node) is False


def test_modify_node_set_script_mode_false():
    tf_constructors = (
        "TensorFlow()",
        "TensorFlow(script_mode=False)",
        "TensorFlow(script_mode=None)",
    )
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_constructors:
        node = _ast_call(constructor)
        modifier.modify_node(node)
        assert "TensorFlow(script_mode=False)" == pasta.dump(node)


def test_modify_node_set_hyperparameters():
    tf_constructor = """TensorFlow(
        checkpoint_path='s3://foo/bar',
        training_steps=100,
        evaluation_steps=10,
        requirements_file='source/requirements.txt',
    )"""

    node = _ast_call(tf_constructor)
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    modifier.modify_node(node)

    expected_hyperparameters = {
        "checkpoint_path": "s3://foo/bar",
        "evaluation_steps": 10,
        "sagemaker_requirements": "source/requirements.txt",
        "training_steps": 100,
    }

    assert expected_hyperparameters == _hyperparameters_from_node(node)


def test_modify_node_preserve_other_hyperparameters():
    tf_constructor = """sagemaker.tensorflow.TensorFlow(
        training_steps=100,
        evaluation_steps=10,
        requirements_file='source/requirements.txt',
        hyperparameters={'optimizer': 'sgd', 'lr': 0.1, 'checkpoint_path': 's3://foo/bar'},
    )"""

    node = _ast_call(tf_constructor)
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    modifier.modify_node(node)

    expected_hyperparameters = {
        "optimizer": "sgd",
        "lr": 0.1,
        "checkpoint_path": "s3://foo/bar",
        "evaluation_steps": 10,
        "sagemaker_requirements": "source/requirements.txt",
        "training_steps": 100,
    }

    assert expected_hyperparameters == _hyperparameters_from_node(node)


def test_modify_node_prefer_param_over_hyperparameter():
    tf_constructor = """sagemaker.tensorflow.TensorFlow(
        training_steps=100,
        requirements_file='source/requirements.txt',
        hyperparameters={'training_steps': 10, 'sagemaker_requirements': 'foo.txt'},
    )"""

    node = _ast_call(tf_constructor)
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    modifier.modify_node(node)

    expected_hyperparameters = {
        "sagemaker_requirements": "source/requirements.txt",
        "training_steps": 100,
    }

    assert expected_hyperparameters == _hyperparameters_from_node(node)


def _hyperparameters_from_node(node):
    for kw in node.keywords:
        if kw.arg == "hyperparameters":
            keys = [k.s for k in kw.value.keys]
            values = [getattr(v, v._fields[0]) for v in kw.value.values]
            return dict(zip(keys, values))


def _ast_call(code):
    return pasta.parse(code).body[0].value
