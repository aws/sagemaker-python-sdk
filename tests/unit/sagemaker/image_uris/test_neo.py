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

NEO_ALGOS = ("image-classification-neo", "xgboost-neo")

ACCOUNTS = {
    "af-south-1": "774647643957",
    "ap-east-1": "110948597952",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-northeast-3": "925152966179",
    "ap-south-1": "763008648453",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "ca-central-1": "464438896020",
    "cn-north-1": "472730292857",
    "cn-northwest-1": "474822919863",
    "eu-central-1": "746233611703",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "me-south-1": "836785723513",
    "sa-east-1": "756306329178",
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-gov-west-1": "263933020539",
    "us-iso-east-1": "167761179201",
    "us-isob-east-1": "406031935815",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",,
    "il-central-1": "275950707576"
}

INFERENTIA_REGIONS = ACCOUNTS.keys()


@pytest.mark.parametrize("algo", NEO_ALGOS)
def test_algo_uris(algo):
    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve(algo, region)
        expected = expected_uris.algo_uri(algo, ACCOUNTS[region], region, version="latest")
        assert expected == uri


def _test_neo_framework_uris(framework, version):
    framework_in_config = f"neo-{framework}"
    framework_in_uri = f"inference-{framework}"

    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve(
            framework_in_config, region, instance_type="ml_c5", version=version
        )
        assert _expected_framework_uri(framework_in_uri, version, region=region) == uri

    uri = image_uris.retrieve(
        framework_in_config, "us-west-2", instance_type="ml_p2", version=version
    )
    assert _expected_framework_uri(framework_in_uri, version, processor="gpu") == uri


def test_neo_mxnet(neo_mxnet_version):
    _test_neo_framework_uris("mxnet", neo_mxnet_version)


def test_neo_tf(neo_tensorflow_version):
    _test_neo_framework_uris("tensorflow", neo_tensorflow_version)


def test_neo_pytorch(neo_pytorch_version):
    _test_neo_framework_uris("pytorch", neo_pytorch_version)


def _test_inferentia_framework_uris(framework, version):
    for region in INFERENTIA_REGIONS:
        uri = image_uris.retrieve(
            "inferentia-{}".format(framework), region, instance_type="ml_inf1", version=version
        )
        expected = _expected_framework_uri(
            "neo-{}".format(framework), version, region=region, processor="inf"
        )
        assert expected == uri


def test_inferentia_mxnet(inferentia_mxnet_version):
    _test_inferentia_framework_uris("mxnet", inferentia_mxnet_version)


def test_inferentia_tensorflow(inferentia_tensorflow_version):
    _test_inferentia_framework_uris("tensorflow", inferentia_tensorflow_version)


def test_inferentia_pytorch(inferentia_pytorch_version):
    _test_inferentia_framework_uris("pytorch", inferentia_pytorch_version)


def _expected_framework_uri(framework, version, region="us-west-2", processor="cpu"):
    return expected_uris.framework_uri(
        "sagemaker-{}".format(framework),
        fw_version=version,
        py_version="py3",
        account=ACCOUNTS[region],
        region=region,
        processor=processor,
    )
