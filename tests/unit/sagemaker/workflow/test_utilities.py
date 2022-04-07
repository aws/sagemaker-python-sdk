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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import tempfile
from sagemaker.workflow.utilities import hash_file


def test_hash_file():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(tmp.name)
        assert hash == "d41d8cd98f00b204e9800998ecf8427e"


def test_hash_file_uri():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(f"file:///{tmp.name}")
        assert hash == "d41d8cd98f00b204e9800998ecf8427e"
