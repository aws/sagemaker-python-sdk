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

# Expected regions for DJL LMI based on documentation
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
EXPECTED_DJL_LMI_REGIONS = {
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "af-south-1", "ap-east-1", "ap-east-2", "ap-south-1", "ap-south-2",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4", 
    "ap-southeast-5", "ap-southeast-7", "ap-northeast-1", "ap-northeast-2", 
    "ap-northeast-3", "ca-central-1", "ca-west-1", "eu-central-1", 
    "eu-central-2", "eu-west-1", "eu-west-2", "eu-west-3", "eu-north-1", 
    "eu-south-1", "eu-south-2", "il-central-1", "mx-central-1", 
    "me-south-1", "me-central-1", "sa-east-1", "cn-north-1", "cn-northwest-1"
}


def test_djl_lmi_config_for_framework_has_all_regions():
    """Test that config_for_framework('djl-lmi') returns all expected regions for each version."""
    config = image_uris.config_for_framework("djl-lmi")
    
    # Check that each version has all expected regions
    for version, version_config in config["versions"].items():
        actual_regions = set(version_config["registries"].keys())
        missing_regions = EXPECTED_DJL_LMI_REGIONS - actual_regions
        
        assert not missing_regions, f"Version {version} missing regions: {missing_regions}"

