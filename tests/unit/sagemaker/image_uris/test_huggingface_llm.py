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

from sagemaker.huggingface import get_huggingface_llm_image_uri
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
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
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
HF_VERSIONS = ["0.6.0", "0.8.2", "0.9.3", "1.0.3", "1.1.0"]
LMI_VERSIONS = ["0.24.0"]
HF_VERSIONS_MAPPING = {
    "0.6.0": "2.0.0-tgi0.6.0-gpu-py39-cu118-ubuntu20.04",
    "0.8.2": "2.0.0-tgi0.8.2-gpu-py39-cu118-ubuntu20.04",
    "0.9.3": "2.0.1-tgi0.9.3-gpu-py39-cu118-ubuntu20.04",
    "1.0.3": "2.0.1-tgi1.0.3-gpu-py39-cu118-ubuntu20.04",
    "1.1.0": "2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04",
}
LMI_VERSIONS_MAPPING = {"0.24.0": "deepspeed0.10.0-cu118"}


@pytest.mark.parametrize("version", HF_VERSIONS)
def test_huggingface(version):
    for region in ACCOUNTS.keys():
        uri = get_huggingface_llm_image_uri("huggingface", region=region, version=version)

        expected = expected_uris.huggingface_llm_framework_uri(
            "huggingface-pytorch-tgi-inference",
            ACCOUNTS[region],
            version,
            HF_VERSIONS_MAPPING[version],
            region=region,
        )
        assert expected == uri


@pytest.mark.parametrize("version", LMI_VERSIONS)
def test_lmi(version):
    for region in ACCOUNTS.keys():
        uri = get_huggingface_llm_image_uri("lmi", region=region, version=version)

        expected = expected_uris.djl_framework_uri(
            "djl-inference", ACCOUNTS[region], version, LMI_VERSIONS_MAPPING[version], region=region
        )
        assert expected == uri
