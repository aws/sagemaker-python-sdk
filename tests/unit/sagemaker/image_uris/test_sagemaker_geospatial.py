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


def _test_ecr_uri(account, region, version, tag):
    actual_uri = image_uris.retrieve("sagemaker-geospatial", region=region, version=version)
    expected_uri = expected_uris.algo_uri_with_tag(
        "sagemaker-geospatial-v1-0",
        account,
        region,
        tag=tag,
    )
    return expected_uri == actual_uri


@pytest.mark.parametrize("load_config", ["sagemaker-geospatial.json"], indirect=True)
@pytest.mark.parametrize("extract_versions_for_image_scope", ["processing"], indirect=True)
def test_sagemaker_geospatial_ecr_uri(load_config, extract_versions_for_image_scope):
    VERSIONS = extract_versions_for_image_scope
    for version in VERSIONS:
        SAGEMAKER_GEOSPATIAL_ACCOUNTS = load_config["processing"]["versions"][version]["registries"]
        for region in SAGEMAKER_GEOSPATIAL_ACCOUNTS.keys():

            assert _test_ecr_uri(
                account=SAGEMAKER_GEOSPATIAL_ACCOUNTS[region],
                region=region,
                version=version,
                tag="latest",
            )


@pytest.mark.parametrize("load_config", ["sagemaker-geospatial.json"], indirect=True)
def test_sagemaker_geospatial_ecr_uri_no_version(load_config):
    region = "us-west-2"
    SAGEMAKER_GEOSPATIAL_ACCOUNTS = load_config["processing"]["versions"]["1.x"]["registries"]
    actual_uri = image_uris.retrieve("sagemaker-geospatial", region=region)
    expected_uri = expected_uris.algo_uri_with_tag(
        "sagemaker-geospatial-v1-0", SAGEMAKER_GEOSPATIAL_ACCOUNTS[region], region, tag="latest"
    )
    assert expected_uri == actual_uri
