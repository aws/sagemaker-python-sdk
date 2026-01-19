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
import pytest
from sagemaker.core import image_uris
from . import expected_uris


@pytest.mark.parametrize("load_config", ["sagemaker-base-python.json"], indirect=True)
@pytest.mark.parametrize("py_version", ["310", "38"])
def test_get_base_python_image_uri(py_version, load_config):
    REGISTRIES = load_config["versions"]["1.0"]["registries"]
    for region in REGISTRIES.keys():
        uri = image_uris.get_base_python_image_uri(
            region=region,
            py_version=py_version,
        )

        repo = "sagemaker-base-python-" + py_version
        expected = expected_uris.base_python_uri(
            repo=repo, account=REGISTRIES[region], region=region
        )
        assert expected == uri
