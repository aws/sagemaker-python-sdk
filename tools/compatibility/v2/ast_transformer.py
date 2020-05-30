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
"""An ast.NodeTransformer subclass for updating SageMaker Python SDK code."""
from __future__ import absolute_import

import ast

from tools.compatibility.v2.modifiers import framework_version

FUNCTION_CALL_MODIFIERS = [framework_version.FrameworkVersionEnforcer()]


class ASTTransformer(ast.NodeTransformer):
    """An ``ast.NodeTransformer`` subclass that walks the abstract syntax tree and
    modifies nodes to upgrade the given SageMaker Python SDK code.
    """

    def visit_Call(self, node):
        """Visits an ``ast.Call`` node and returns a modified node, if needed.
        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.Call): a node that represents a function call.

        Returns:
            ast.Call: a node that represents a function call, which has
                potentially been modified from the original input.
        """
        for function_checker in FUNCTION_CALL_MODIFIERS:
            function_checker.check_and_modify_node(node)
        return node
