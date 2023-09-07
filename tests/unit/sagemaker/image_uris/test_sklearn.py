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


def test_valid_uris(sklearn_version):
    for region in ACCOUNTS.keys():
        uri = image_uris.retrieve(
            "sklearn",
            region=region,
            version=sklearn_version,
            py_version="py3",
            instance_type="ml.c4.xlarge",
        )

        expected = expected_uris.framework_uri(
            "sagemaker-scikit-learn",
            sklearn_version,
            ACCOUNTS[region],
            py_version="py3",
            region=region,
        )
        assert expected == uri


def test_py2_error(sklearn_version):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            "sklearn",
            region="us-west-2",
            version=sklearn_version,
            py_version="py2",
            instance_type="ml.c4.xlarge",
        )

    assert "Unsupported Python version: py2." in str(e.value)


def test_gpu_error(sklearn_version):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            "sklearn",
            region="us-west-2",
            version=sklearn_version,
            instance_type="ml.p2.xlarge",
        )

    assert "Unsupported processor: gpu." in str(e.value)
