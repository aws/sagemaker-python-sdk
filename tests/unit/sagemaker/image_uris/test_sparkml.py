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
    "af-south-1": "510948584623",
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
    "eu-south-1": "978288397137",
    "me-south-1": "801668240914",
    "sa-east-1": "737474898029",
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-gov-west-1": "414596584902",
    "us-iso-east-1": "833128469047",
    "us-isob-east-1": "281123927165",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
}
VERSIONS = ["2.2", "2.4", "3.3"]


@pytest.mark.parametrize("version", VERSIONS)
def test_sparkml(version):
    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve("sparkml-serving", region=region, version=version)

        expected = expected_uris.algo_uri(
            "sagemaker-sparkml-serving", ACCOUNTS[region], region, version=version
        )
        assert expected == uri
