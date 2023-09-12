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

DATA_WRANGLER_ACCOUNTS = {
    "af-south-1": "143210264188",
    "ap-east-1": "707077482487",
    "ap-northeast-1": "649008135260",
    "ap-northeast-2": "131546521161",
    "ap-northeast-3": "913387583493",
    "ap-south-1": "089933028263",
    "ap-southeast-1": "119527597002",
    "ap-southeast-2": "422173101802",
    "ca-central-1": "557239378090",
    "eu-central-1": "024640144536",
    "eu-north-1": "054986407534",
    "eu-south-1": "488287956546",
    "eu-west-1": "245179582081",
    "eu-west-2": "894491911112",
    "eu-west-3": "807237891255",
    "me-south-1": "376037874950",
    "sa-east-1": "424196993095",
    "us-east-1": "663277389841",
    "us-east-2": "415577184552",
    "us-west-1": "926135532090",
    "us-west-2": "174368400705",
    "cn-north-1": "245909111842",
    "cn-northwest-1": "249157047649",
}

# Accounts only supported in DW 3.x and beyond
DATA_WRANGLER_3X_ACCOUNTS = {
    "il-central-1": "406833011540",
}

VERSIONS = ["1.x", "2.x", "3.x"]


def _test_ecr_uri(account, region, version):
    actual_uri = image_uris.retrieve("data-wrangler", region=region, version=version)
    expected_uri = expected_uris.algo_uri(
        "sagemaker-data-wrangler-container",
        account,
        region,
        version=version,
    )
    return expected_uri == actual_uri


def test_data_wrangler_ecr_uri():
    for version in VERSIONS:
        for region in DATA_WRANGLER_ACCOUNTS.keys():
            assert _test_ecr_uri(
                account=DATA_WRANGLER_ACCOUNTS[region], region=region, version=version
            )


def test_data_wrangler_ecr_uri_3x():
    for region in DATA_WRANGLER_3X_ACCOUNTS.keys():
        assert _test_ecr_uri(
            account=DATA_WRANGLER_3X_ACCOUNTS[region], region=region, version="3.x"
        )


def test_data_wrangler_ecr_uri_none():
    region = "us-west-2"
    actual_uri = image_uris.retrieve("data-wrangler", region=region)
    expected_uri = expected_uris.algo_uri(
        "sagemaker-data-wrangler-container",
        DATA_WRANGLER_ACCOUNTS[region],
        region,
        version=VERSIONS[-1],
    )
    assert expected_uri == actual_uri
