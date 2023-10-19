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
DJL_DEEPSPEED_VERSIONS = ["0.24.0", "0.23.0", "0.22.1", "0.21.0", "0.20.0", "0.19.0"]
DJL_FASTERTRANSFORMER_VERSIONS = ["0.24.0", "0.23.0", "0.22.1", "0.21.0"]
DJL_NEURONX_VERSIONS = ["0.24.0", "0.23.0", "0.22.1"]
DJL_VERSIONS_TO_FRAMEWORK = {
    "0.19.0": {"djl-deepspeed": "deepspeed0.7.3-cu113"},
    "0.20.0": {"djl-deepspeed": "deepspeed0.7.5-cu116"},
    "0.21.0": {
        "djl-deepspeed": "deepspeed0.8.3-cu117",
        "djl-fastertransformer": "fastertransformer5.3.0-cu117",
    },
    "0.22.1": {
        "djl-deepspeed": "deepspeed0.9.2-cu118",
        "djl-fastertransformer": "fastertransformer5.3.0-cu118",
        "djl-neuronx": "neuronx-sdk2.10.0",
    },
    "0.23.0": {
        "djl-deepspeed": "deepspeed0.9.5-cu118",
        "djl-fastertransformer": "fastertransformer5.3.0-cu118",
        "djl-neuronx": "neuronx-sdk2.12.0",
    },
    "0.24.0": {
        "djl-deepspeed": "deepspeed0.10.0-cu118",
        "djl-fastertransformer": "fastertransformer5.3.0-cu118",
        "djl-neuronx": "neuronx-sdk2.14.1",
    },
}


@pytest.mark.parametrize("region", ACCOUNTS.keys())
@pytest.mark.parametrize("version", DJL_DEEPSPEED_VERSIONS)
def test_djl_deepspeed(region, version):
    _test_djl_uris(region, version, "djl-deepspeed")


@pytest.mark.parametrize("region", ACCOUNTS.keys())
@pytest.mark.parametrize("version", DJL_FASTERTRANSFORMER_VERSIONS)
def test_djl_fastertransformer(region, version):
    _test_djl_uris(region, version, "djl-fastertransformer")


@pytest.mark.parametrize("region", ACCOUNTS.keys())
@pytest.mark.parametrize("version", DJL_NEURONX_VERSIONS)
def test_djl_neuronx(region, version):
    _test_djl_uris(region, version, "djl-neuronx")


def _test_djl_uris(region, version, djl_framework):
    uri = image_uris.retrieve(framework=djl_framework, region=region, version=version)
    expected = expected_uris.djl_framework_uri(
        "djl-inference",
        ACCOUNTS[region],
        version,
        DJL_VERSIONS_TO_FRAMEWORK[version][djl_framework],
        region,
    )
    assert expected == uri
