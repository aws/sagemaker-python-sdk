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

ALGO_NAMES = (
    "blazingtext",
    "factorization-machines",
    "forecasting-deepar",
    "image-classification",
    "ipinsights",
    "kmeans",
    "knn",
    "linear-learner",
    "ntm",
    "object-detection",
    "object2vec",
    "pca",
    "randomcutforest",
    "semantic-segmentation",
    "seq2seq",
    "lda",
)
ALGO_REGIONS_AND_ACCOUNTS = (
    {
        "algorithms": (
            "pca",
            "kmeans",
            "linear-learner",
            "factorization-machines",
            "ntm",
            "randomcutforest",
            "knn",
            "object2vec",
            "ipinsights",
        ),
        "accounts": {
            "af-south-1": "455444449433",
            "ap-east-1": "286214385809",
            "ap-northeast-1": "351501993468",
            "ap-northeast-2": "835164637446",
            "ap-northeast-3": "867004704886",
            "ap-south-1": "991648021394",
            "ap-south-2": "628508329040",
            "ap-southeast-1": "475088953585",
            "ap-southeast-2": "712309505854",
            "ap-southeast-3": "951798379941",
            "ap-southeast-4": "106583098589",
            "ca-central-1": "469771592824",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
            "eu-central-1": "664544806723",
            "eu-central-2": "680994064768",
            "eu-north-1": "669576153137",
            "eu-west-1": "438346466558",
            "eu-west-2": "644912444149",
            "eu-west-3": "749696950732",
            "eu-south-1": "257386234256",
            "eu-south-2": "104374241257",
            "il-central-1": "898809789911",
            "me-south-1": "249704162688",
            "me-central-1": "272398656194",
            "sa-east-1": "855470959533",
            "us-east-1": "382416733822",
            "us-east-2": "404615174143",
            "us-gov-west-1": "226302683700",
            "us-gov-east-1": "237065988967",
            "us-iso-east-1": "490574956308",
            "us-isob-east-1": "765400339828",
            "us-west-1": "632365934929",
            "us-west-2": "174872318107",
        },
    },
    {
        "algorithms": ("lda",),
        "accounts": {
            "ap-northeast-1": "258307448986",
            "ap-northeast-2": "293181348795",
            "ap-south-1": "991648021394",
            "ap-southeast-1": "475088953585",
            "ap-southeast-2": "297031611018",
            "ca-central-1": "469771592824",
            "eu-central-1": "353608530281",
            "eu-west-1": "999678624901",
            "eu-west-2": "644912444149",
            "us-east-1": "766337827248",
            "us-east-2": "999911452149",
            "us-gov-west-1": "226302683700",
            "us-iso-east-1": "490574956308",
            "us-isob-east-1": "765400339828",
            "us-west-1": "632365934929",
            "us-west-2": "266724342769",
        },
    },
    {
        "algorithms": ("forecasting-deepar",),
        "accounts": {
            "af-south-1": "455444449433",
            "ap-east-1": "286214385809",
            "ap-northeast-1": "633353088612",
            "ap-northeast-2": "204372634319",
            "ap-northeast-3": "867004704886",
            "ap-south-1": "991648021394",
            "ap-southeast-1": "475088953585",
            "ap-southeast-2": "514117268639",
            "ca-central-1": "469771592824",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
            "eu-central-1": "495149712605",
            "eu-north-1": "669576153137",
            "eu-west-1": "224300973850",
            "eu-west-2": "644912444149",
            "eu-west-3": "749696950732",
            "eu-south-1": "257386234256",
            "me-south-1": "249704162688",
            "sa-east-1": "855470959533",
            "us-east-1": "522234722520",
            "us-east-2": "566113047672",
            "us-gov-west-1": "226302683700",
            "us-iso-east-1": "490574956308",
            "us-isob-east-1": "765400339828",
            "us-west-1": "632365934929",
            "us-west-2": "156387875391",
        },
    },
    {
        "algorithms": (
            "seq2seq",
            "image-classification",
            "blazingtext",
            "object-detection",
            "semantic-segmentation",
        ),
        "accounts": {
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
        },
    },
)

IMAGE_URI_FORMAT = "{}.dkr.ecr.{}.{}/{}:1"


def _accounts_for_algo(algo):
    for algo_account_dict in ALGO_REGIONS_AND_ACCOUNTS:
        if algo in algo_account_dict["algorithms"]:
            return algo_account_dict["accounts"]

    return {}


@pytest.mark.parametrize("algo", ALGO_NAMES)
def test_algo_uris(algo):
    accounts = _accounts_for_algo(algo)
    for region in accounts:
        uri = image_uris.retrieve(algo, region)
        assert expected_uris.algo_uri(algo, accounts[region], region) == uri
