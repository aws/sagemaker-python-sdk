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
from sagemaker.core import image_uris
from . import expected_uris


def _test_ecr_uri(account, region, version):
    actual_uri = image_uris.retrieve("data-wrangler", region=region, version=version)
    expected_uri = expected_uris.algo_uri(
        "sagemaker-data-wrangler-container",
        account,
        region,
        version=version,
    )
    return expected_uri == actual_uri


@pytest.mark.parametrize("load_config", ["data-wrangler.json"], indirect=True)
@pytest.mark.parametrize("extract_versions_for_image_scope", ["processing"], indirect=True)
def test_data_wrangler_ecr_uri(load_config, extract_versions_for_image_scope):
    VERSIONS = extract_versions_for_image_scope
    for version in VERSIONS:
        DATA_WRANGLER_ACCOUNTS = load_config["processing"]["versions"][version]["registries"]
        for region in DATA_WRANGLER_ACCOUNTS.keys():
            assert _test_ecr_uri(
                account=DATA_WRANGLER_ACCOUNTS[region], region=region, version=version
            )


@pytest.mark.parametrize("load_config", ["data-wrangler.json"], indirect=True)
def test_data_wrangler_ecr_uri_none(load_config):
    region = "us-west-2"
    VERSIONS = ["1.x", "2.x", "3.x"]
    DATA_WRANGLER_ACCOUNTS = load_config["processing"]["versions"]["1.x"]["registries"]
    actual_uri = image_uris.retrieve("data-wrangler", region=region)
    expected_uri = expected_uris.algo_uri(
        "sagemaker-data-wrangler-container",
        DATA_WRANGLER_ACCOUNTS[region],
        region,
        version=VERSIONS[-1],
    )
    assert expected_uri == actual_uri
