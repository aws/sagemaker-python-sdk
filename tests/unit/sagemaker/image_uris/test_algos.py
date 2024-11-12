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


ALGO_NAMES = [
    "blazingtext.json",
    "factorization-machines.json",
    "forecasting-deepar.json",
    "image-classification.json",
    "ipinsights.json",
    "kmeans.json",
    "knn.json",
    "linear-learner.json",
    "ntm.json",
    "object-detection.json",
    "object2vec.json",
    "pca.json",
    "randomcutforest.json",
    "semantic-segmentation.json",
    "seq2seq.json",
    "lda.json",
]


@pytest.mark.parametrize("load_config", ALGO_NAMES, indirect=True)
def test_algo_uris(load_config):
    VERSIONS = load_config["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config["versions"][version]["registries"]
        algo_name = load_config["versions"][version]["repository"]
        for region in ACCOUNTS.keys():
            uri = image_uris.retrieve(algo_name, region)
            assert expected_uris.algo_uri(algo_name, ACCOUNTS[region], region) == uri
