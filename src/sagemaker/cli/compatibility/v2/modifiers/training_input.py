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
"""Classes to modify Predictor code to be compatible
with version 2.0 and later of the SageMaker Python SDK.
"""
from __future__ import absolute_import

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

BASE_S3_INPUT = "s3_input"
SESSION = "session"
S3_INPUT = {"s3_input": ("sagemaker", "sagemaker.session")}


class TrainingInputConstructorRefactor(Modifier):
    """A class to refactor *s3_input class."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a class of interest.

        This looks for the following calls:

        - ``sagemaker.s3_input``
        - ``sagemaker.session.s3_input``
        - ``s3_input``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a class of interest.
        """
        return matching.matches_any(node, S3_INPUT)

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to call ``TrainingInput`` instead.

        Args:
            node (ast.Call): a node that represents a *TrainingInput constructor.
        """
        _rename_class(node)


def _rename_class(node):
    """Renames the s3_input class to TrainingInput"""
    if matching.matches_name(node, BASE_S3_INPUT):
        node.func.id = "TrainingInput"
    elif matching.matches_attr(node, BASE_S3_INPUT):
        node.func.attr = "TrainingInput"


class TrainingInputImportFromRenamer(Modifier):
    """A class to update import statements of ``s3_input``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports ``RealTimePredictor`` from the correct module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the import statement imports ``RealTimePredictor`` from the correct module.
        """
        return node.module in S3_INPUT[BASE_S3_INPUT] and any(
            name.name == BASE_S3_INPUT for name in node.names
        )

    def modify_node(self, node):
        """Changes the ``ast.ImportFrom`` node's name from ``s3_input`` to ``TrainingInput``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.
        """
        for name in node.names:
            if name.name == BASE_S3_INPUT:
                name.name = "TrainingInput"
            elif name.name == "session":
                name.name = "inputs"
