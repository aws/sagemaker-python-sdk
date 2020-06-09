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
"""A class to handle argument changes for Airflow functions."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier


class ModelConfigArgModifier(Modifier):
    """A class to handle argument changes for Airflow model config functions."""

    FUNCTION_NAMES = ("model_config", "model_config_from_estimator")

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node creates an Airflow model config and
        contains positional arguments.

        This looks for the following formats:

        - ``model_config``
        - ``airflow.model_config``
        - ``workflow.airflow.model_config``
        - ``sagemaker.workflow.airflow.model_config``

        where ``model_config`` is either ``model_config`` or ``model_config_from_estimator``.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is either a ``model_config`` call or
                a ``model_config_from_estimator`` call and has positional arguments.
        """
        return self._is_model_config_call(node) and len(node.args) > 0

    def _is_model_config_call(self, node):
        """Checks if the node is a ``model_config`` or  ``model_config_from_estimator`` call."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.FUNCTION_NAMES

        if not (isinstance(node.func, ast.Attribute) and node.func.attr in self.FUNCTION_NAMES):
            return False

        return self._is_in_module(node.func, "sagemaker.workflow.airflow".split("."))

    def _is_in_module(self, node, module):
        """Checks if the node is in the module, including partial matches to the module path."""
        if isinstance(node.value, ast.Name):
            return node.value.id == module[-1]

        if isinstance(node.value, ast.Attribute) and node.value.attr == module[-1]:
            return self._is_in_module(node.value, module[:-1])

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's arguments.

        The first argument, the instance type, is turned into a keyword arg,
        leaving the second argument, the model, to be the first argument.

        Args:
            node (ast.Call): a node that represents either a ``model_config`` call or
                a ``model_config_from_estimator`` call.
        """
        instance_type = node.args.pop(0)
        node.keywords.append(ast.keyword(arg="instance_type", value=instance_type))
