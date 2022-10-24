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

GRAVITON_ALGOS = ("tensorflow", "pytotch")
GRAVITON_INSTANCE_TYPES = [
    "ml.c6g.4xlarge",
    "ml.t4g.2xlarge",
    "ml.r6g.2xlarge",
    "ml.m6g.4xlarge",
]

ACCOUNTS = {
    "af-south-1": "626614931356",
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
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}

GRAVITON_REGIONS = ACCOUNTS.keys()


def _test_graviton_framework_uris(framework, version):
    for region in GRAVITON_REGIONS:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                framework, region, instance_type=instance_type, version=version
            )
            expected = _expected_graviton_framework_uri(framework, version, region=region)
            assert expected == uri


def test_graviton_tensorflow(graviton_tensorflow_version):
    _test_graviton_framework_uris("tensorflow", graviton_tensorflow_version)


def test_graviton_pytorch(graviton_pytorch_version):
    _test_graviton_framework_uris("pytorch", graviton_pytorch_version)


def _expected_graviton_framework_uri(framework, version, region):
    return expected_uris.graviton_framework_uri(
        "{}-inference-graviton".format(framework),
        fw_version=version,
        py_version="py38",
        account=ACCOUNTS[region],
        region=region,
    )
