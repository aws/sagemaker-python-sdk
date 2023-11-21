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


@pytest.mark.parametrize("load_config", ["sparkml-serving.json"], indirect=True)
def test_sparkml(load_config):
    VERSIONS = load_config["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = image_uris.retrieve("sparkml-serving", region=region, version=version)

            expected = expected_uris.algo_uri(
                "sagemaker-sparkml-serving", ACCOUNTS[region], region, version=version
            )
            assert expected == uri
