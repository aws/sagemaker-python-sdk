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

import boto3

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

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
            "ap-east-1": "286214385809",
            "ap-northeast-1": "351501993468",
            "ap-northeast-2": "835164637446",
            "ap-south-1": "991648021394",
            "ap-southeast-1": "475088953585",
            "ap-southeast-2": "712309505854",
            "ca-central-1": "469771592824",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
            "eu-central-1": "664544806723",
            "eu-north-1": "669576153137",
            "eu-west-1": "438346466558",
            "eu-west-2": "644912444149",
            "eu-west-3": "749696950732",
            "me-south-1": "249704162688",
            "sa-east-1": "855470959533",
            "us-east-1": "382416733822",
            "us-east-2": "404615174143",
            "us-gov-west-1": "226302683700",
            "us-iso-east-1": "490574956308",
            "us-west-1": "632365934929",
            "us-west-2": "174872318107",
        },
    },
)

IMAGE_URI_FORMAT = "{}.dkr.ecr.{}.{}/{}:1"


def _regions():
    boto_session = boto3.Session()
    for partition in boto_session.get_available_partitions():
        for region in boto_session.get_available_regions("sagemaker", partition_name=partition):
            yield region


def _accounts_for_algo(algo):
    for algo_account_dict in ALGO_REGIONS_AND_ACCOUNTS:
        if algo in algo_account_dict["algorithms"]:
            return algo_account_dict["accounts"]

    return {}


def test_factorization_machines():
    algo = "factorization-machines"
    accounts = _accounts_for_algo(algo)

    for region in _regions():
        for scope in ("training", "inference"):
            uri = image_uris.retrieve(algo, region, image_scope=scope)
            assert expected_uris.algo_uri(algo, accounts[region], region) == uri
