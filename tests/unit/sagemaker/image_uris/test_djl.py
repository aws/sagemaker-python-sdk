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
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "af-south-1",
    "ap-east-1",
    "ap-east-2",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-7",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "il-central-1",
    "mx-central-1",
    "me-south-1",
    "me-central-1",
    "sa-east-1",
    "cn-north-1",
    "cn-northwest-1",
}

# Known missing framework:version:region combinations that don't exist in ECR
KNOWN_MISSING_COMBINATIONS = {
    "djl-lmi": {
        "0.30.0-lmi12.0.0-cu124": {"ap-east-2"},
        "0.29.0-lmi11.0.0-cu124": {"ap-east-2"},
        "0.28.0-lmi10.0.0-cu124": {"ap-east-2"},
    },
    "djl-neuronx": {
        "0.29.0-neuronx-sdk2.19.1": {
            "ap-east-1",
            "me-central-1",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.28.0-neuronx-sdk2.18.2": {
            "ap-east-1",
            "me-central-1",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.27.0-neuronx-sdk2.18.1": {
            "ap-east-1",
            "me-central-1",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.26.0-neuronx-sdk2.16.0": {
            "ap-east-1",
            "me-central-1",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.25.0-neuronx-sdk2.15.0": {
            "eu-north-1",
            "ap-east-1",
            "me-central-1",
            "eu-west-2",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.24.0-neuronx-sdk2.14.1": {
            "eu-north-1",
            "ap-east-1",
            "me-central-1",
            "eu-west-2",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.23.0-neuronx-sdk2.12.0": {
            "eu-north-1",
            "ap-east-1",
            "me-central-1",
            "eu-west-2",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
        "0.22.1-neuronx-sdk2.10.0": {
            "eu-north-1",
            "ap-east-1",
            "me-central-1",
            "eu-west-2",
            "ap-east-2",
            "ap-southeast-3",
            "eu-south-1",
            "ca-central-1",
            "us-west-1",
            "ap-northeast-3",
            "ap-northeast-2",
            "af-south-1",
            "me-south-1",
        },
    },
    "djl-tensorrtllm": {
        "0.30.0-tensorrtllm0.12.0-cu125": {"ap-east-2"},
        "0.29.0-tensorrtllm0.11.0-cu124": {"ap-east-2"},
        "0.28.0-tensorrtllm0.9.0-cu122": {"ap-east-2"},
        "0.27.0-tensorrtllm0.8.0-cu122": {"ap-east-2"},
        "0.26.0-tensorrtllm0.7.1-cu122": {"ap-east-2"},
        "0.25.0-tensorrtllm0.5.0-cu122": {"ap-east-2"},
    },
    "djl-fastertransformer": {
        "0.24.0-fastertransformer5.3.0-cu118": {"ap-east-2"},
        "0.23.0-fastertransformer5.3.0-cu118": {"ap-east-2"},
        "0.22.1-fastertransformer5.3.0-cu118": {"ap-east-2"},
        "0.21.0-fastertransformer5.3.0-cu117": {"ap-east-2"},
    },
    "djl-deepspeed": {
        "0.27.0-deepspeed0.12.6-cu121": {"ap-east-2"},
        "0.26.0-deepspeed0.12.6-cu121": {"ap-east-2"},
        "0.25.0-deepspeed0.11.0-cu118": {"ap-east-2"},
        "0.24.0-deepspeed0.10.0-cu118": {"ap-east-2"},
        "0.23.0-deepspeed0.9.5-cu118": {"ap-east-2"},
        "0.22.1-deepspeed0.9.2-cu118": {"ap-east-2"},
        "0.21.0-deepspeed0.8.3-cu117": {"ap-east-2"},
        "0.20.0-deepspeed0.7.5-cu116": {"ap-east-2"},
        "0.19.0-deepspeed0.7.3-cu113": {"ap-east-2"},
    },
}


@pytest.mark.parametrize(
    "framework",
    ["djl-deepspeed", "djl-fastertransformer", "djl-lmi", "djl-neuronx", "djl-tensorrtllm"],
)
def test_djl_lmi_config_for_framework_has_all_regions(framework):
    """Test that config_for_framework returns all expected regions for each version."""
    config = image_uris.config_for_framework(framework)

    # Check that each version has all expected regions, excluding known missing combinations
    for version, version_config in config["versions"].items():
        actual_regions = set(version_config["registries"].keys())
        expected_regions_for_version = EXPECTED_DJL_LMI_REGIONS.copy()

        # Use tag_prefix for lookup if available, otherwise fall back to version
        lookup_key = version_config.get("tag_prefix", version)

        # Remove regions that are known to be missing for this framework:version combination
        missing_regions_for_version = KNOWN_MISSING_COMBINATIONS.get(framework, {}).get(
            lookup_key, set()
        )
        expected_regions_for_version -= missing_regions_for_version

        missing_regions = expected_regions_for_version - actual_regions

        assert (
            not missing_regions
        ), f"Framework {framework} version {version} missing regions: {missing_regions}"
