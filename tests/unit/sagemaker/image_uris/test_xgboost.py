# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from tests.unit.sagemaker.image_uris import expected_uris, regions

ALGO_REGISTRIES = {
    "ap-east-1": "286214385809",
    "ap-northeast-1": "501404015308",
    "ap-northeast-2": "306986355934",
    "ap-south-1": "991648021394",
    "ap-southeast-1": "475088953585",
    "ap-southeast-2": "544295431143",
    "ca-central-1": "469771592824",
    "cn-north-1": "390948362332",
    "cn-northwest-1": "387376663083",
    "eu-central-1": "813361260812",
    "eu-north-1": "669576153137",
    "eu-west-1": "685385470294",
    "eu-west-2": "644912444149",
    "eu-west-3": "749696950732",
    "me-south-1": "249704162688",
    "sa-east-1": "855470959533",
    "us-east-1": "811284229777",
    "us-east-2": "825641698319",
    "us-gov-west-1": "226302683700",
    "us-iso-east-1": "490574956308",
    "us-west-1": "632365934929",
    "us-west-2": "433757028032",
}
ALGO_VERSIONS = ("1", "latest")

FRAMEWORK_REGISTRIES = {
    "ap-east-1": "651117190479",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698",
    "ap-south-1": "720646828776",
    "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285",
    "ca-central-1": "341280168497",
    "cn-north-1": "450853457545",
    "cn-northwest-1": "451049120500",
    "eu-central-1": "492215442770",
    "eu-north-1": "662702820516",
    "eu-west-1": "141502667606",
    "eu-west-2": "764974769150",
    "eu-west-3": "659782779980",
    "me-south-1": "801668240914",
    "sa-east-1": "737474898029",
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-gov-west-1": "414596584902",
    "us-iso-east-1": "833128469047",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
}


def test_xgboost_framework(xgboost_framework_version):
    for region in regions.regions():
        uri = image_uris.retrieve(
            framework="xgboost", region=region, version=xgboost_framework_version, py_version="py3",
        )

        expected = expected_uris.framework_uri(
            "sagemaker-xgboost",
            xgboost_framework_version,
            FRAMEWORK_REGISTRIES[region],
            py_version="py3",
            region=region,
        )
        assert expected == uri


@pytest.mark.parametrize("xgboost_algo_version", ("1", "latest"))
def test_xgboost_algo(xgboost_algo_version):
    for region in regions.regions():
        uri = image_uris.retrieve(framework="xgboost", region=region, version=xgboost_algo_version)

        expected = expected_uris.algo_uri(
            "xgboost", ALGO_REGISTRIES[region], region, version=xgboost_algo_version
        )
        assert expected == uri
