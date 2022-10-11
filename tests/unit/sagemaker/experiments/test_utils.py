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

from src.sagemaker.experiments._utils import resolve_artifact_name, guess_media_type


def test_resolve_artifact_name():
    file_names = {
        "a": "a",
        "a.txt": "a.txt",
        "b.": "b.",
        ".c": ".c",
        "/x/a/a.txt": "a.txt",
        "/a/b/c.": "c.",
        "./.a": ".a",
        "../b.txt": "b.txt",
        "~/a.txt": "a.txt",
        "c/d.txt": "d.txt",
    }
    for file_name, artifact_name in file_names.items():
        assert artifact_name == resolve_artifact_name(file_name)


def test_guess_media_type():
    assert "text/plain" == guess_media_type("foo.txt")
