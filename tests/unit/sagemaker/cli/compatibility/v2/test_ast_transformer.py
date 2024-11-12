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

import textwrap

import pasta
import pytest

from sagemaker.cli.compatibility.v2.ast_transformer import ASTTransformer


@pytest.fixture
def input_code():
    return textwrap.dedent(
        """
        from sagemaker.predictor import csv_serializer

        csv_serializer.__doc__
        """
    )


@pytest.fixture
def output_code():
    return textwrap.dedent(
        """
        from sagemaker import serializers

        serializers.CSVSerializer().__doc__
        """
    )


def test_simple_script(input_code, output_code):
    input_ast = pasta.parse(input_code)
    output_ast = ASTTransformer().visit(input_ast)
    assert pasta.dump(output_ast) == output_code
