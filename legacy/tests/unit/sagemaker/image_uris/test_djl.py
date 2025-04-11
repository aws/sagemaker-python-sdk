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


@pytest.mark.parametrize(
    "load_config_and_file_name",
    ["djl-neuronx.json", "djl-tensorrtllm.json", "djl-lmi.json"],
    indirect=True,
)
def test_djl_uris(load_config_and_file_name):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    VERSIONS = config["versions"]
    for version in VERSIONS:
        ACCOUNTS = config["versions"][version]["registries"]
        tag = config["versions"][version]["tag_prefix"]
        for region in ACCOUNTS.keys():
            _test_djl_uris(ACCOUNTS[region], region, version, tag, framework)


def _test_djl_uris(account, region, version, tag, djl_framework):
    uri = image_uris.retrieve(framework=djl_framework, region=region, version=version)
    expected = expected_uris.djl_framework_uri(
        "djl-inference",
        account,
        tag,
        region,
    )
    assert expected == uri
