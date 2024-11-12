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

import json
import os

from mock import call, Mock, mock_open, patch

from sagemaker.cli.compatibility.v2 import files


def test_init():
    input_file = "input.py"
    output_file = "output.py"

    updater = files.FileUpdater(input_file, output_file)
    assert input_file == updater.input_path
    assert output_file == updater.output_path


@patch("six.moves.builtins.open", mock_open())
@patch("os.makedirs")
def test_make_output_dirs_if_needed_make_path(makedirs):
    output_dir = "dir"
    output_path = os.path.join(output_dir, "output.py")

    updater = files.FileUpdater("input.py", output_path)
    updater._make_output_dirs_if_needed()

    makedirs.assert_called_with(output_dir)


@patch("six.moves.builtins.open", mock_open())
@patch("os.path.exists", return_value=True)
def test_make_output_dirs_if_needed_overwrite_with_warning(os_path_exists, caplog):
    output_file = "output.py"

    updater = files.FileUpdater("input.py", output_file)
    updater._make_output_dirs_if_needed()

    assert "Overwriting file {}".format(output_file) in caplog.text


@patch("pasta.dump")
@patch("pasta.parse")
@patch("sagemaker.cli.compatibility.v2.files.ASTTransformer")
def test_py_file_update(ast_transformer, pasta_parse, pasta_dump):
    input_ast = Mock()
    pasta_parse.return_value = input_ast

    output_ast = Mock(_fields=[])
    ast_transformer.return_value.visit.return_value = output_ast
    output_code = "print('goodbye')"
    pasta_dump.return_value = output_code

    input_file = "input.py"
    output_file = "output.py"

    input_code = "print('hello, world!')"
    open_mock = mock_open(read_data=input_code)
    with patch("six.moves.builtins.open", open_mock):
        updater = files.PyFileUpdater(input_file, output_file)
        updater.update()

    pasta_parse.assert_called_with(input_code)
    ast_transformer.return_value.visit.assert_called_with(input_ast)

    assert call(input_file) in open_mock.mock_calls
    assert call(output_file, "w") in open_mock.mock_calls

    open_mock().write.assert_called_with(output_code)
    pasta_dump.assert_called_with(output_ast)


@patch("json.dump")
@patch("pasta.dump")
@patch("pasta.parse")
@patch("sagemaker.cli.compatibility.v2.files.ASTTransformer")
def test_update(ast_transformer, pasta_parse, pasta_dump, json_dump):
    notebook_template = """{
        "cells": [
         {
          "cell_type": "code",
          "execution_count": 1,
          "metadata": {},
          "outputs": [],
          "source": [
           "!echo ignore this"
          ]
         },
         {
          "cell_type": "code",
          "execution_count": 2,
          "metadata": {},
          "outputs": [],
          "source": [
           "%%%%bash\\n",
           "echo ignore this too"
          ]
         },
         {
          "cell_type": "code",
          "execution_count": 3,
          "metadata": {},
          "outputs": [],
          "source": [
           "%%cd\\n",
           "echo ignore this too"
          ]
         },
         {
          "cell_type": "code",
          "execution_count": 4,
          "metadata": {},
          "outputs": [],
          "source": [
           "# code to be modified\\n",
           "%s"
          ]
         }
        ],
        "metadata": {
         "kernelspec": {
          "display_name": "Python 3",
          "language": "python",
          "name": "python3"
         },
         "language_info": {
          "codemirror_mode": {
           "name": "ipython",
           "version": 3
          },
          "file_extension": ".py",
          "mimetype": "text/x-python",
          "name": "python",
          "nbconvert_exporter": "python",
          "pygments_lexer": "ipython3",
          "version": "3.6.8"
         }
        },
        "nbformat": 4,
        "nbformat_minor": 2
       }
    """
    input_code = "print('hello, world!')"
    input_notebook = notebook_template % input_code

    input_ast = Mock()
    pasta_parse.return_value = input_ast

    output_ast = Mock(_fields=[])
    ast_transformer.return_value.visit.return_value = output_ast
    output_code = "print('goodbye')"
    pasta_dump.return_value = "# code to be modified\n{}".format(output_code)

    input_file = "input.py"
    output_file = "output.py"

    open_mock = mock_open(read_data=input_notebook)
    with patch("six.moves.builtins.open", open_mock):
        updater = files.JupyterNotebookFileUpdater(input_file, output_file)
        updater.update()

    pasta_parse.assert_called_with("# code to be modified\n{}".format(input_code))
    ast_transformer.return_value.visit.assert_called_with(input_ast)
    pasta_dump.assert_called_with(output_ast)

    assert call(input_file) in open_mock.mock_calls
    assert call(output_file, "w") in open_mock.mock_calls

    json_dump.assert_called_with(json.loads(notebook_template % output_code), open_mock(), indent=1)
    open_mock().write.assert_called_with("\n")
