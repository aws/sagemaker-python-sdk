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

from sagemaker.stabilityai import get_stabilityai_image_uri
from tests.unit.sagemaker.image_uris import expected_uris

ACCOUNTS = {
    "af-south-1": "626614931356",
    "il-central-1": "780543022126",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-southeast-3": "907027046896",
    "ca-central-1": "763104351884",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "eu-south-1": "692866216735",
    "me-south-1": "217643126080",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}
SAI_VERSIONS = ["0.1.0"]
SAI_VERSIONS_MAPPING = {"0.1.0": "2.0.1-sgm0.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker"}


@pytest.mark.parametrize("version", SAI_VERSIONS)
def test_stabilityai_image_uris(version):
    for region in ACCOUNTS.keys():
        result = get_stabilityai_image_uri(region=region, version=version)
        expected = expected_uris.stabilityai_framework_uri(
            "stabilityai-pytorch-inference",
            ACCOUNTS[region],
            SAI_VERSIONS_MAPPING[version],
            region=region,
        )
        assert expected == result
