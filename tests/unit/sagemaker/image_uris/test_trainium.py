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

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

import pytest

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
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-isob-east-1": "094389454867",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}

TRAINIUM_REGIONS = ACCOUNTS.keys()
TRAINIUM_ALLOWED_FRAMEWORKS = "pytorch"


def _expected_trainium_framework_uri(
    framework, version, region="us-west-2", inference_tool="neuron"
):
    return expected_uris.neuron_framework_uri(
        "{}-neuron".format(framework),
        fw_version=version,
        py_version="py38",
        account=ACCOUNTS[region],
        region=region,
        inference_tool=inference_tool,
    )


def _test_trainium_framework_uris(framework, version):
    for region in TRAINIUM_REGIONS:
        uri = image_uris.retrieve(
            framework, region, instance_type="ml.trn1.xlarge", version=version
        )
        expected = _expected_trainium_framework_uri(
            "{}-training".format(framework), version, region=region, inference_tool="neuron"
        )
        assert expected == uri


def test_trainium_pytorch(pytorch_neuron_version):
    _test_trainium_framework_uris("pytorch", pytorch_neuron_version)


def _test_trainium_unsupported_framework(framework, framework_version):
    for region in TRAINIUM_REGIONS:
        with pytest.raises(ValueError) as error:
            image_uris.retrieve(
                framework, region, version=framework_version, instance_type="ml.trn1.xlarge"
            )
        expectedErr = (
            f"Unsupported framework: {framework}. Supported framework(s) for Trainium instances: "
            f"{TRAINIUM_ALLOWED_FRAMEWORKS}."
        )
        assert expectedErr in str(error)


def test_trainium_unsupported_framework():
    _test_trainium_unsupported_framework("autogluon", "0.6.1")
