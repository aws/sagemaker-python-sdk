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

ALGO_VERSIONS = ("1", "latest")


@pytest.mark.parametrize("load_config", ["xgboost.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_xgboost_uris(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]

        processors = load_config[scope]["versions"][version].get("processors", [None])
        py_versions = load_config[scope]["versions"][version].get("py_versions", [None])
        repo = load_config[scope]["versions"][version]["repository"]
        for processor in processors:
            for py_version in py_versions:
                for region in ACCOUNTS.keys():
                    uri = image_uris.retrieve(
                        framework="xgboost", py_version=py_version, region=region, version=version
                    )

                    if version in ALGO_VERSIONS:
                        expected = expected_uris.algo_uri(
                            "xgboost", ACCOUNTS[region], region, version=version
                        )
                    else:
                        expected = expected_uris.framework_uri(
                            repo,
                            version,
                            ACCOUNTS[region],
                            region=region,
                            py_version=py_version,
                            processor=processor,
                        )

                    assert uri == expected
