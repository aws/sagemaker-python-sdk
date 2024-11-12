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

from sagemaker.cli.compatibility.v2.modifiers import tf_legacy_mode
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


def test_node_should_be_modified_fit_with_tensorboard():
    fit_calls = (
        "estimator.fit(run_tensorboard_locally=True)",
        "tensorflow.fit(run_tensorboard_locally=False)",
    )

    modifier = tf_legacy_mode.TensorBoardParameterRemover()

    for call in fit_calls:
        node = ast_call(call)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_fit_without_tensorboard():
    fit_calls = ("estimator.fit()", "tensorflow.fit()")

    modifier = tf_legacy_mode.TensorBoardParameterRemover()

    for call in fit_calls:
        node = ast_call(call)
        assert modifier.node_should_be_modified(node) is False


def test_node_should_be_modified_random_function_call():
    node = ast_call("estimator.deploy(1, 'local')")
    modifier = tf_legacy_mode.TensorBoardParameterRemover()
    assert modifier.node_should_be_modified(node) is False


def test_modify_node():
    fit_calls = (
        "estimator.fit(run_tensorboard_locally=True)",
        "estimator.fit(run_tensorboard_locally=False)",
    )
    modifier = tf_legacy_mode.TensorBoardParameterRemover()

    for call in fit_calls:
        node = ast_call(call)
        modifier.modify_node(node)
        assert "estimator.fit()" == pasta.dump(node)
