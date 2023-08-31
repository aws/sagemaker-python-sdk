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
    "af-south-1": "875698925577",
    "ap-east-1": "001633400207",
    "ap-northeast-1": "574779866223",
    "ap-northeast-2": "709848358524",
    "ap-northeast-3": "990339680094",
    "ap-south-1": "126357580389",
    "ap-southeast-1": "245545462676",
    "ap-southeast-2": "563025443158",
    "ap-southeast-3": "669540362728",
    "ca-central-1": "536280801234",
    "cn-north-1": "453000072557",
    "cn-northwest-1": "453252182341",
    "eu-central-1": "048819808253",
    "eu-north-1": "895015795356",
    "eu-south-1": "933208885752",
    "eu-west-1": "468650794304",
    "eu-west-2": "749857270468",
    "eu-west-3": "680080141114",
    "me-south-1": "607024016150",
    "sa-east-1": "539772159869",
    "us-east-1": "156813124566",
    "us-east-2": "777275614652",
    "us-west-1": "890145073186",
    "us-west-2": "159807026194",
    "il-central-1": "843974653677",
}


def test_model_monitor():
    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve("model-monitor", region=region)

        expected = expected_uris.monitor_uri(ACCOUNTS[region], region)
        assert expected == uri
