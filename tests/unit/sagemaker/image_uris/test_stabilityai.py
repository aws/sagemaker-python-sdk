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

from sagemaker.stabilityai import get_stabilityai_image_uri
from tests.unit.sagemaker.image_uris import expected_uris


SAI_VERSIONS_MAPPING = {"0.1.0": "2.0.1-sgm0.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker"}


@pytest.mark.parametrize("load_config", ["stabilityai.json"], indirect=True)
@pytest.mark.parametrize("scope", ["inference"])
def test_stabilityai_image_uris(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = get_stabilityai_image_uri(region=region, version=version)
            expected = expected_uris.stabilityai_framework_uri(
                "stabilityai-pytorch-inference",
                ACCOUNTS[region],
                SAI_VERSIONS_MAPPING[version],
                region=region,
            )
            assert expected == uri
