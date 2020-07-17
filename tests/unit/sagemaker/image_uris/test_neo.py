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
from tests.unit import NEO_REGION_LIST
from tests.unit.sagemaker.image_uris import expected_uris, regions


ALGO_NAMES = ("image-classification-neo", "xgboost-neo")

ACCOUNTS = {
    "ap-east-1": "110948597952",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-south-1": "763008648453",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "ca-central-1": "464438896020",
    "cn-north-1": "472730292857",
    "cn-northwest-1": "474822919863",
    "eu-central-1": "746233611703",
    "eu-north-1": "601324751636",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "me-south-1": "836785723513",
    "sa-east-1": "756306329178",
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-gov-west-1": "263933020539",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
}


@pytest.mark.parametrize("algo", ALGO_NAMES)
def test_algo_uris(algo):
    for region in regions.regions():
        if region in NEO_REGION_LIST:
            uri = image_uris.retrieve(algo, region)
            expected = expected_uris.algo_uri(algo, ACCOUNTS[region], region, version="latest")
            assert expected == uri
        else:
            with pytest.raises(ValueError) as e:
                image_uris.retrieve(algo, region)
            assert "Unsupported region: {}.".format(region) in str(e.value)
