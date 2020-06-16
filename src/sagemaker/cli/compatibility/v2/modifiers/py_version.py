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
"""A class to ensure that ``py_version`` is defined when constructing framework classes."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

PY_VERSION_ARG = "py_version"
PY_VERSION_DEFAULT = "py3"

FRAMEWORK_MODEL_REQUIRES_PY_VERSION = {
    "Chainer": True,
    "MXNet": True,
    "PyTorch": True,
    "SKLearn": False,
    "TensorFlow": False,
}

FRAMEWORK_CLASSES = list(FRAMEWORK_MODEL_REQUIRES_PY_VERSION.keys())

MODEL_CLASSES = [
    "{}Model".format(fw) for fw, required in FRAMEWORK_MODEL_REQUIRES_PY_VERSION.items() if required
]
FRAMEWORK_MODULES = [fw.lower() for fw in FRAMEWORK_CLASSES]
FRAMEWORK_SUBMODULES = ("model", "estimator")


class PyVersionEnforcer(Modifier):
    """A class to ensure that ``py_version`` is defined when
    instantiating a framework estimator or model, where appropriate.
    """

    def node_should_be_modified(self, node):
        """Checks if the ast.Call node should be modified to include ``py_version``.

        If the ast.Call node instantiates a framework estimator or model, but doesn't
        specify the ``py_version`` parameter when required, then the node should be
        modified. However, if ``image_name`` for a framework estimator or ``image``
        for a model is supplied to the call, then ``py_version`` is not required.

        This looks for the following formats:

        - ``PyTorch``
        - ``sagemaker.pytorch.PyTorch``

        where "PyTorch" can be Chainer, MXNet, PyTorch, SKLearn, or TensorFlow.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a framework class that
                should specify ``py_version``, but doesn't.
        """
        if _is_named_constructor(node, FRAMEWORK_CLASSES):
            return _version_arg_needed(node, "image_name", PY_VERSION_ARG)

        if _is_named_constructor(node, MODEL_CLASSES):
            return _version_arg_needed(node, "image", PY_VERSION_ARG)

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to include ``py_version``.

        Args:
            node (ast.Call): a node that represents the constructor of a framework class.
        """
        node.keywords.append(ast.keyword(arg=PY_VERSION_ARG, value=ast.Str(s=PY_VERSION_DEFAULT)))


def _is_named_constructor(node, names):
    """Checks if the ``ast.Call`` node represents a call to particular named constructors.

    Forms that qualify are either <Framework> or sagemaker.<framework>.<Framework>
    where <Framework> belongs to the list of names passed in.
    """
    # Check for call from particular names of constructors
    if isinstance(node.func, ast.Name):
        return node.func.id in names

    # Check for something.that.ends.with.<framework>.<Framework> call for Framework in names
    if not (isinstance(node.func, ast.Attribute) and node.func.attr in names):
        return False

    # Check for sagemaker.<frameworks>.<estimator/model>.<Framework> call
    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr in FRAMEWORK_SUBMODULES:
        return _is_in_framework_module(node.func.value)

    # Check for sagemaker.<framework>.<Framework> call
    return _is_in_framework_module(node.func)


def _is_in_framework_module(node):
    """Checks if node is an ``ast.Attribute`` representing a ``sagemaker.<framework>`` module."""
    return (
        isinstance(node.value, ast.Attribute)
        and node.value.attr in FRAMEWORK_MODULES
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sagemaker"
    )


def _version_arg_needed(node, image_arg, version_arg):
    """Determines if image_arg or version_arg was supplied"""
    return not (_arg_supplied(node, image_arg) or _arg_supplied(node, version_arg))


def _arg_supplied(node, arg):
    """Checks if the ``ast.Call`` node's keywords contain ``arg``."""
    return any(kw.arg == arg and kw.value for kw in node.keywords)
