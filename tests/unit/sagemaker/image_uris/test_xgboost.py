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

ALGO_REGISTRIES = {
    "af-south-1": "455444449433",
    "ap-east-1": "286214385809",
    "ap-northeast-1": "501404015308",
    "ap-northeast-2": "306986355934",
    "ap-northeast-3": "867004704886",
    "ap-south-1": "991648021394",
    "ap-south-2": "628508329040",
    "ap-southeast-1": "475088953585",
    "ap-southeast-2": "544295431143",
    "ap-southeast-3": "951798379941",
    "ap-southeast-4": "106583098589",
    "ca-central-1": "469771592824",
    "cn-north-1": "390948362332",
    "cn-northwest-1": "387376663083",
    "eu-central-1": "813361260812",
    "eu-central-2": "680994064768",
    "eu-north-1": "669576153137",
    "eu-west-1": "685385470294",
    "eu-west-2": "644912444149",
    "eu-west-3": "749696950732",
    "eu-south-1": "257386234256",
    "eu-south-2": "104374241257",
    "il-central-1": "898809789911",
    "me-south-1": "249704162688",
    "me-central-1": "272398656194",
    "sa-east-1": "855470959533",
    "us-east-1": "811284229777",
    "us-east-2": "825641698319",
    "us-gov-west-1": "226302683700",
    "us-gov-east-1": "237065988967",
    "us-iso-east-1": "490574956308",
    "us-isob-east-1": "765400339828",
    "us-west-1": "632365934929",
    "us-west-2": "433757028032",
}
ALGO_VERSIONS = ("1", "latest")
XGBOOST_FRAMEWORK_CPU_ONLY_VERSIONS = ("0.90-2", "0.90-1", "1.0-1")
XGBOOST_FRAMEWORK_CPU_GPU_VERSIONS = ("1.2-1", "1.2-2", "1.3-1", "1.5-1", "1.7-1")

FRAMEWORK_REGISTRIES = {
    "af-south-1": "510948584623",
    "ap-east-1": "651117190479",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698",
    "ap-northeast-3": "867004704886",
    "ap-south-1": "720646828776",
    "ap-south-2": "628508329040",
    "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285",
    "ap-southeast-3": "951798379941",
    "ap-southeast-4": "106583098589",
    "ca-central-1": "341280168497",
    "cn-north-1": "450853457545",
    "cn-northwest-1": "451049120500",
    "eu-central-1": "492215442770",
    "eu-central-2": "680994064768",
    "eu-north-1": "662702820516",
    "eu-west-1": "141502667606",
    "eu-west-2": "764974769150",
    "eu-west-3": "659782779980",
    "eu-south-1": "978288397137",
    "eu-south-2": "104374241257",
    "il-central-1": "898809789911",
    "me-south-1": "801668240914",
    "me-central-1": "272398656194",
    "sa-east-1": "737474898029",
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-gov-west-1": "414596584902",
    "us-gov-east-1": "237065988967",
    "us-iso-east-1": "833128469047",
    "us-isob-east-1": "281123927165",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
}


@pytest.mark.parametrize("xgboost_framework_version", XGBOOST_FRAMEWORK_CPU_GPU_VERSIONS)
def test_xgboost_framework(xgboost_framework_version):
    for region in FRAMEWORK_REGISTRIES.keys():
        uri = image_uris.retrieve(
            framework="xgboost",
            region=region,
            version=xgboost_framework_version,
        )

        expected = expected_uris.framework_uri(
            "sagemaker-xgboost",
            xgboost_framework_version,
            FRAMEWORK_REGISTRIES[region],
            py_version=None,
            processor=None,
            region=region,
        )
        assert expected == uri


@pytest.mark.parametrize("xgboost_framework_version", XGBOOST_FRAMEWORK_CPU_ONLY_VERSIONS)
def test_xgboost_framework_cpu_only(xgboost_framework_version):
    for region in FRAMEWORK_REGISTRIES.keys():
        if not (xgboost_framework_version in ["0.90-2", "0.90-1"] and region == "il-central-1"):
            uri = image_uris.retrieve(
                framework="xgboost",
                region=region,
                version=xgboost_framework_version,
            )

            expected = expected_uris.framework_uri(
                "sagemaker-xgboost",
                xgboost_framework_version,
                FRAMEWORK_REGISTRIES[region],
                region=region,
                py_version="py3",
                processor="cpu",
            )
            assert expected == uri


@pytest.mark.parametrize("xgboost_algo_version", ALGO_VERSIONS)
def test_xgboost_algo(xgboost_algo_version):
    for region in ALGO_REGISTRIES.keys():
        if region != "il-central-1":
            uri = image_uris.retrieve(
                framework="xgboost", region=region, version=xgboost_algo_version
            )

            expected = expected_uris.algo_uri(
                "xgboost", ALGO_REGISTRIES[region], region, version=xgboost_algo_version
            )
            assert expected == uri
