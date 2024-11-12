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

from sagemaker.cli.compatibility.v2.modifiers import deprecated_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


def test_node_should_be_modified_tf_constructor_script_mode():
    tf_script_mode_constructors = (
        "TensorFlow(script_mode=True)",
        "sagemaker.tensorflow.TensorFlow(script_mode=True)",
    )

    modifier = deprecated_params.TensorFlowScriptModeParameterRemover()

    for constructor in tf_script_mode_constructors:
        node = ast_call(constructor)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_not_tf_script_mode():
    modifier = deprecated_params.TensorFlowScriptModeParameterRemover()

    for call in ("TensorFlow()", "random()"):
        node = ast_call(call)
        assert modifier.node_should_be_modified(node) is False


def test_modify_node():
    node = ast_call("TensorFlow(script_mode=True)")
    modifier = deprecated_params.TensorFlowScriptModeParameterRemover()
    modifier.modify_node(node)

    assert "TensorFlow()" == pasta.dump(node)
