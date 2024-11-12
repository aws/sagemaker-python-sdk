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
from mock import MagicMock, patch

from sagemaker.cli.compatibility.v2.modifiers import tf_legacy_mode
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

IMAGE_URI = "sagemaker-tensorflow:latest"
REGION_NAME = "us-west-2"


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
        "sagemaker.tensorflow.estimator.TensorFlow(script_mode=False)",
        "sagemaker.tensorflow.estimator.TensorFlow(script_mode=None)",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version='py2')",
        "sagemaker.tensorflow.estimator.TensorFlow()",
    )

    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_legacy_mode_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_tf_constructor_script_mode():
    tf_script_mode_constructors = (
        "TensorFlow(script_mode=True)",
        "TensorFlow(py_version='py3')",
        "TensorFlow(py_version='py37')",
        "TensorFlow(py_version='py3', script_mode=False)",
        "TensorFlow(py_version=py_version, script_mode=False)",
        "TensorFlow(py_version='py3', script_mode=script_mode)",
        "sagemaker.tensorflow.TensorFlow(script_mode=True)",
        "sagemaker.tensorflow.TensorFlow(py_version='py3')",
        "sagemaker.tensorflow.TensorFlow(py_version='py37')",
        "sagemaker.tensorflow.TensorFlow(py_version='py3', script_mode=False)",
        "sagemaker.tensorflow.TensorFlow(py_version=py_version, script_mode=False)",
        "sagemaker.tensorflow.TensorFlow(py_version='py3', script_mode=script_mode)",
        "sagemaker.tensorflow.estimator.TensorFlow(script_mode=True)",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version='py3')",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version='py37')",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version='py3', script_mode=False)",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version=py_version, script_mode=False)",
        "sagemaker.tensorflow.estimator.TensorFlow(py_version='py3', script_mode=script_mode)",
    )

    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_script_mode_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is False


def test_node_should_be_modified_random_function_call():
    node = ast_call("MXNet(py_version='py3')")
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    assert modifier.node_should_be_modified(node) is False


@patch("boto3.Session")
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
def test_modify_node_set_model_dir_and_image_name(retrieve_image_uri, boto_session):
    boto_session.return_value.region_name = REGION_NAME

    tf_constructors = (
        "TensorFlow()",
        "TensorFlow(script_mode=False)",
        "TensorFlow(model_dir='s3//bucket/model')",
    )
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()

    for constructor in tf_constructors:
        node = ast_call(constructor)
        modifier.modify_node(node)

        assert "TensorFlow(image_uri='{}', model_dir=False)".format(IMAGE_URI) == pasta.dump(node)
        retrieve_image_uri.assert_called_with(
            "tensorflow",
            REGION_NAME,
            instance_type="ml.m4.xlarge",
            version="1.11.0",
            py_version="py2",
            image_scope="training",
        )


@patch("boto3.Session")
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
def test_modify_node_set_image_name_from_args(retrieve_image_uri, boto_session):
    boto_session.return_value.region_name = REGION_NAME

    tf_constructor = "TensorFlow(train_instance_type='ml.p2.xlarge', framework_version='1.4.0')"

    node = ast_call(tf_constructor)
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    modifier.modify_node(node)

    retrieve_image_uri.assert_called_with(
        "tensorflow",
        REGION_NAME,
        instance_type="ml.p2.xlarge",
        version="1.4.0",
        py_version="py2",
        image_scope="training",
    )

    expected_string = (
        "TensorFlow(train_instance_type='ml.p2.xlarge', framework_version='1.4.0', "
        "image_uri='{}', model_dir=False)".format(IMAGE_URI)
    )
    assert expected_string == pasta.dump(node)


@patch("boto3.Session", MagicMock())
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
def test_modify_node_set_hyperparameters(retrieve_image_uri):
    tf_constructor = """TensorFlow(
        checkpoint_path='s3://foo/bar',
        training_steps=100,
        evaluation_steps=10,
        requirements_file='source/requirements.txt',
    )"""

    node = ast_call(tf_constructor)
    modifier = tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader()
    modifier.modify_node(node)

    expected_hyperparameters = {
        "checkpoint_path": "s3://foo/bar",
        "evaluation_steps": 10,
        "sagemaker_requirements": "source/requirements.txt",
        "training_steps": 100,
    }

    assert expected_hyperparameters == _hyperparameters_from_node(node)


@patch("boto3.Session", MagicMock())
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
def test_modify_node_preserve_other_hyperparameters(retrieve_image_uri):
    tf_constructor = """sagemaker.tensorflow.TensorFlow(
        training_steps=100,
        evaluation_steps=10,
        requirements_file='source/requirements.txt',
        hyperparameters={'optimizer': 'sgd', 'lr': 0.1, 'checkpoint_path': 's3://foo/bar'},
    )"""

    node = ast_call(tf_constructor)
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


@patch("boto3.Session", MagicMock())
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
def test_modify_node_prefer_param_over_hyperparameter(retrieve_image_uri):
    tf_constructor = """sagemaker.tensorflow.TensorFlow(
        training_steps=100,
        requirements_file='source/requirements.txt',
        hyperparameters={'training_steps': 10, 'sagemaker_requirements': 'foo.txt'},
    )"""

    node = ast_call(tf_constructor)
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
