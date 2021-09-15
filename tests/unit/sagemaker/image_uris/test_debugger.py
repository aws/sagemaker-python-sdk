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

ACCOUNTS = {
    "af-south-1": "314341159256",
    "ap-east-1": "199566480951",
    "ap-northeast-1": "430734990657",
    "ap-northeast-2": "578805364391",
    "ap-northeast-3": "479947661362",
    "ap-south-1": "904829902805",
    "ap-southeast-1": "972752614525",
    "ap-southeast-2": "184798709955",
    "ca-central-1": "519511493484",
    "cn-north-1": "618459771430",
    "cn-northwest-1": "658757709296",
    "eu-central-1": "482524230118",
    "eu-north-1": "314864569078",
    "eu-south-1": "563282790590",
    "eu-west-1": "929884845733",
    "eu-west-2": "250201462417",
    "eu-west-3": "447278800020",
    "me-south-1": "986000313247",
    "sa-east-1": "818342061345",
    "us-east-1": "503895931360",
    "us-east-2": "915447279597",
    "us-gov-west-1": "515509971035",
    "us-west-1": "685455198987",
    "us-west-2": "895741380848",
}


def test_debugger():
    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve("debugger", region=region)
        expected = expected_uris.algo_uri(
            "sagemaker-debugger-rules", ACCOUNTS[region], region, version="latest"
        )
        assert expected == uri
