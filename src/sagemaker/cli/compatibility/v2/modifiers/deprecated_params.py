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
"""Classes to remove deprecated parameters."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier


class TensorFlowScriptModeParamRemover(Modifier):
    """A class to remove ``script_mode`` from TensorFlow estimators (because it's the only mode)."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a TensorFlow estimator with
        ``script_mode`` set.

        This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a TensorFlow estimator with ``script_mode``.
        """
        return self._is_tf_constructor(node) and self._has_script_mode_param(node)

    def _is_tf_constructor(self, node):
        """Checks if the ``ast.Call`` node represents a call of the form
        ``TensorFlow`` or ``sagemaker.tensorflow.TensorFlow``.
        """
        # Check for TensorFlow()
        if isinstance(node.func, ast.Name):
            return node.func.id == "TensorFlow"

        # Check for sagemaker.tensorflow.TensorFlow()
        ends_with_tensorflow_constructor = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "TensorFlow"
        )

        is_in_tensorflow_module = (
            isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "tensorflow"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sagemaker"
        )

        return ends_with_tensorflow_constructor and is_in_tensorflow_module

    def _has_script_mode_param(self, node):
        """Checks if the ``ast.Call`` node's keywords include ``script_mode``."""
        for kw in node.keywords:
            if kw.arg == "script_mode":
                return True

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to remove ``script_mode``.

        Args:
            node (ast.Call): a node that represents a TensorFlow constructor.
        """
        for kw in node.keywords:
            if kw.arg == "script_mode":
                node.keywords.remove(kw)
