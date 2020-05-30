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
"""A class to ensure that ``framework_version`` is defined when constructing framework classes."""
from __future__ import absolute_import

import ast

from tools.compatibility.v2.modifiers.modifier import Modifier

FRAMEWORK_DEFAULTS = {
    "Chainer": "4.1.0",
    "MXNet": "1.2.0",
    "PyTorch": "0.4.0",
    "SKLearn": "0.20.0",
    "TensorFlow": "1.11.0",
}

FRAMEWORKS = list(FRAMEWORK_DEFAULTS.keys())
# TODO: check for sagemaker.tensorflow.serving.Model
FRAMEWORK_CLASSES = FRAMEWORKS + ["{}Model".format(fw) for fw in FRAMEWORKS]
FRAMEWORK_MODULES = [fw.lower() for fw in FRAMEWORKS]


class FrameworkVersionEnforcer(Modifier):
    """A class to ensure that ``framework_version`` is defined when
    instantiating a framework estimator or model.
    """

    def node_should_be_modified(self, node):
        """Checks if the ast.Call node instantiates a framework estimator or model,
        but doesn't specify the ``framework_version`` parameter.

        This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``

        where "TensorFlow" can be Chainer, MXNet, PyTorch, SKLearn, or TensorFlow.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a framework class that
                should specify ``framework_version``, but doesn't.
        """
        if self._is_framework_constructor(node):
            return not self._fw_version_in_keywords(node)

        return False

    def _is_framework_constructor(self, node):
        """Checks if the ``ast.Call`` node represents a call of the form
        <Framework> or sagemaker.<framework>.<Framework>.
        """
        # Check for <Framework> call
        if isinstance(node.func, ast.Name):
            if node.func.id in FRAMEWORK_CLASSES:
                return True

        # Check for sagemaker.<framework>.<Framework> call
        ends_with_framework_constructor = (
            isinstance(node.func, ast.Attribute) and node.func.attr in FRAMEWORK_CLASSES
        )

        is_in_framework_module = (
            isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in FRAMEWORK_MODULES
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sagemaker"
        )

        return ends_with_framework_constructor and is_in_framework_module

    def _fw_version_in_keywords(self, node):
        """Checks if the ``ast.Call`` node's keywords contain ``framework_version``."""
        for kw in node.keywords:
            if kw.arg == "framework_version" and kw.value:
                return True
        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to include ``framework_version``.

        The ``framework_version`` value is determined by the framework:

        - Chainer: "4.1.0"
        - MXNet: "1.2.0"
        - PyTorch: "0.4.0"
        - SKLearn: "0.20.0"
        - TensorFlow: "1.11.0"

        Args:
            node (ast.Call): a node that represents the constructor of a framework class.
        """
        framework = self._framework_name_from_node(node)
        node.keywords.append(
            ast.keyword(arg="framework_version", value=ast.Str(s=FRAMEWORK_DEFAULTS[framework]))
        )

    def _framework_name_from_node(self, node):
        """Retrieves the framework name based on the function call.

        Args:
            node (ast.Call): a node that represents the constructor of a framework class.
                This can represent either <Framework> or sagemaker.<framework>.<Framework>.

        Returns:
            str: the (capitalized) framework name.
        """
        if isinstance(node.func, ast.Name):
            framework = node.func.id
        elif isinstance(node.func, ast.Attribute):
            framework = node.func.attr

        if framework.endswith("Model"):
            framework = framework[: framework.find("Model")]

        return framework
