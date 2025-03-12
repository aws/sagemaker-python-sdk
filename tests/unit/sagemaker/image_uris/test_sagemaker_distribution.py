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

INSTANCE_TYPES = {"cpu": "ml.c4.xlarge", "gpu": "ml.p2.xlarge"}


def _test_ecr_uri(account, region, version, tag, instance_type, processor):
    actual_uri = image_uris.retrieve(
        "sagemaker-distribution", region=region, instance_type=instance_type, version=version
    )
    expected_uri = expected_uris.sagemaker_distribution_uri(
        "sagemaker-distribution-prod", account, tag, processor, region
    )
    return expected_uri == actual_uri


@pytest.mark.parametrize("load_config", ["sagemaker-distribution.json"], indirect=True)
def test_sagemaker_distribution_ecr_uri(load_config):
    VERSIONS = load_config["versions"]
    processors = load_config["processors"]
    for version in VERSIONS:
        SAGEMAKER_DISTRIBUTION_ACCOUNTS = load_config["versions"][version]["registries"]
        for region in SAGEMAKER_DISTRIBUTION_ACCOUNTS.keys():
            for processor in processors:
                assert _test_ecr_uri(
                    account=SAGEMAKER_DISTRIBUTION_ACCOUNTS[region],
                    region=region,
                    version=version,
                    tag="3.0.0",
                    instance_type=INSTANCE_TYPES[processor],
                    processor=processor,
                )
