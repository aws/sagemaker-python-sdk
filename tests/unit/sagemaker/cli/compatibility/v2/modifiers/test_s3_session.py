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

import pasta

from sagemaker.cli.compatibility.v2.modifiers import renamed_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

NAMESPACES = ("", "s3.", "sagemaker.s3.")
FUNCTIONS = (
    "S3Downloader.download",
    "S3Downloader.list",
    "S3Downloader.read_file",
    "S3Uploader.upload",
    "S3Uploader.upload_string_as_file_body",
)


def test_node_should_be_modified():
    modifier = renamed_params.S3SessionRenamer()

    for func in FUNCTIONS:
        for namespace in NAMESPACES:
            call = ast_call("{}{}(session=sess)".format(namespace, func))
            assert modifier.node_should_be_modified(call)


def test_node_should_be_modified_no_session():
    modifier = renamed_params.S3SessionRenamer()

    for func in FUNCTIONS:
        for namespace in NAMESPACES:
            call = ast_call("{}{}()".format(namespace, func))
            assert not modifier.node_should_be_modified(call)


def test_node_should_be_modified_random_function_call():
    modifier = renamed_params.S3SessionRenamer()

    generic_function_calls = (
        "download()",
        "list()",
        "read_file()",
        "upload()",
    )

    for call in generic_function_calls:
        assert not modifier.node_should_be_modified(ast_call(call))


def test_modify_node():
    node = ast_call("S3Downloader.download(session=sess)")
    modifier = renamed_params.S3SessionRenamer()
    modifier.modify_node(node)

    expected = "S3Downloader.download(sagemaker_session=sess)"
    assert expected == pasta.dump(node)
