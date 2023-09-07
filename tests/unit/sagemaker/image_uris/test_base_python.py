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
from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

REGISTRIES = {
    "us-east-2": "429704687514",
    "me-south-1": "117516905037",
    "us-west-2": "236514542706",
    "ca-central-1": "310906938811",
    "ap-east-1": "493642496378",
    "us-east-1": "081325390199",
    "ap-northeast-2": "806072073708",
    "eu-west-2": "712779665605",
    "ap-southeast-2": "52832661640",
    "cn-northwest-1": "390780980154",
    "eu-north-1": "243637512696",
    "cn-north-1": "390048526115",
    "ap-south-1": "394103062818",
    "eu-west-3": "615547856133",
    "ap-southeast-3": "276181064229",
    "af-south-1": "559312083959",
    "eu-west-1": "470317259841",
    "eu-central-1": "936697816551",
    "sa-east-1": "782484402741",
    "ap-northeast-3": "792733760839",
    "eu-south-1": "592751261982",
    "ap-northeast-1": "102112518831",
    "us-west-1": "742091327244",
    "ap-southeast-1": "492261229750",
    "me-central-1": "103105715889",
    "us-gov-east-1": "107072934176",
    "us-gov-west-1": "107173498710",
}


@pytest.mark.parametrize("py_version", ["310", "38"])
def test_get_base_python_image_uri(py_version):
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
