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
"""Classes for updating code in files."""
from __future__ import absolute_import

import os
import logging

import pasta

from ast_transformer import ASTTransformer

LOGGER = logging.getLogger(__name__)


class PyFileUpdater(object):
    """A class for updating Python (``*.py``) files."""

    def __init__(self, input_path, output_path):
        """Creates a ``PyFileUpdater`` for updating a Python file so that
        it is compatible with v2 of the SageMaker Python SDK.

        Args:
            input_path (str): Location of the input file.
            output_path (str): Desired location for the output file.
                If the directories don't already exist, then they are created.
                If a file exists at ``output_path``, then it is overwritten.
        """
        self.input_path = input_path
        self.output_path = output_path

    def update(self):
        """Reads the input Python file, updates the code so that it is
        compatible with v2 of the SageMaker Python SDK, and writes the
        updated code to an output file.
        """
        output = self._update_ast(self._read_input_file())
        self._write_output_file(output)

    def _update_ast(self, input_ast):
        """Updates an abstract syntax tree (AST) so that it is compatible
        with v2 of the SageMaker Python SDK.

        Args:
            input_ast (ast.Module): AST to be updated for use with Python SDK v2.

        Returns:
            ast.Module: Updated AST that is compatible with Python SDK v2.
        """
        return ASTTransformer().visit(input_ast)

    def _read_input_file(self):
        """Reads input file and parse as an abstract syntax tree (AST).

        Returns:
            ast.Module: AST representation of the input file.
        """
        with open(self.input_path) as input_file:
            return pasta.parse(input_file.read())

    def _write_output_file(self, output):
        """Writes abstract syntax tree (AST) to output file.
        Creates the directories for the output path, if needed.

        Args:
            output (ast.Module): AST to save as the output file.
        """
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(self.output_path):
            LOGGER.warning("Overwriting file {}".format(self.output_path))

        with open(self.output_path, "w") as output_file:
            output_file.write(pasta.dump(output))
