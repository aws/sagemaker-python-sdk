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

from sagemaker.cli.compatibility.v2 import modifiers

FUNCTION_CALL_MODIFIERS = [
    modifiers.renamed_params.EstimatorImageURIRenamer(),
    modifiers.renamed_params.ModelImageURIRenamer(),
    modifiers.framework_version.FrameworkVersionEnforcer(),
    modifiers.tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader(),
    modifiers.tf_legacy_mode.TensorBoardParameterRemover(),
    modifiers.deprecated_params.TensorFlowScriptModeParameterRemover(),
    modifiers.tfs.TensorFlowServingConstructorRenamer(),
    modifiers.predictors.PredictorConstructorRefactor(),
    modifiers.airflow.ModelConfigArgModifier(),
    modifiers.airflow.ModelConfigImageURIRenamer(),
    modifiers.renamed_params.DistributionParameterRenamer(),
    modifiers.renamed_params.S3SessionRenamer(),
]

IMPORT_MODIFIERS = [modifiers.tfs.TensorFlowServingImportRenamer()]

IMPORT_FROM_MODIFIERS = [
    modifiers.predictors.PredictorImportFromRenamer(),
    modifiers.tfs.TensorFlowServingImportFromRenamer(),
]


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

        ast.fix_missing_locations(node)
        return node

    def visit_Import(self, node):
        """Visits an ``ast.Import`` node and returns a modified node, if needed.
        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.Import): a node that represents an import statement.

        Returns:
            ast.Import: a node that represents an import statement, which has
                potentially been modified from the original input.
        """
        for import_checker in IMPORT_MODIFIERS:
            import_checker.check_and_modify_node(node)

        ast.fix_missing_locations(node)
        return node

    def visit_ImportFrom(self, node):
        """Visits an ``ast.ImportFrom`` node and returns a modified node, if needed.
        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.ImportFrom): a node that represents an import statement.

        Returns:
            ast.ImportFrom: a node that represents an import statement, which has
                potentially been modified from the original input.
        """
        for import_checker in IMPORT_FROM_MODIFIERS:
            import_checker.check_and_modify_node(node)

        ast.fix_missing_locations(node)
        return node
