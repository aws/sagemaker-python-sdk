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
from packaging.version import parse

from sagemaker.huggingface import get_huggingface_llm_image_uri
from tests.unit.sagemaker.image_uris import expected_uris

# Mapping of vLLM versions to expected image tags
VLLM_VERSIONS_MAPPING = {
    "inf2": {
        "0.10.2": "0.10.2-neuronx-py310-sdk2.26.0-ubuntu22.04",
    },
}


@pytest.mark.parametrize("load_config", ["huggingface-vllm-neuronx.json"], indirect=True)
def test_vllm_neuronx_uris(load_config):
    """Test that vLLM NeuronX image URIs are correctly generated."""
    VERSIONS = load_config["inference"]["versions"]
    device = load_config["inference"]["processors"][0]

    # Fail if device is not in mapping
    if device not in VLLM_VERSIONS_MAPPING:
        raise ValueError(f"Device {device} not found in VLLM_VERSIONS_MAPPING")

    # Get highest version for the device
    highest_version = max(VLLM_VERSIONS_MAPPING[device].keys(), key=lambda x: parse(x))

    for version in VERSIONS:
        ACCOUNTS = load_config["inference"]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = get_huggingface_llm_image_uri(
                "huggingface-vllm-neuronx",
                region=region,
                version=version,
            )

            # Skip only if test version is higher than highest known version
            if parse(version) > parse(highest_version):
                print(
                    f"Skipping version check for {version} as it is higher than "
                    f"the highest known version {highest_version} in VLLM_VERSIONS_MAPPING."
                )
                continue

            expected = expected_uris.huggingface_llm_framework_uri(
                "huggingface-vllm-inference-neuronx",
                ACCOUNTS[region],
                version,
                VLLM_VERSIONS_MAPPING[device][version],
                region=region,
            )
            assert expected == uri


@pytest.mark.parametrize("load_config", ["huggingface-vllm-neuronx.json"], indirect=True)
def test_vllm_neuronx_version_aliases(load_config):
    """Test that version aliases work correctly."""
    version_aliases = load_config["inference"].get("version_aliases", {})

    for alias, full_version in version_aliases.items():
        uri_alias = get_huggingface_llm_image_uri(
            "huggingface-vllm-neuronx",
            region="us-east-1",
            version=alias,
        )
        uri_full = get_huggingface_llm_image_uri(
            "huggingface-vllm-neuronx",
            region="us-east-1",
            version=full_version,
        )
        # URIs should be identical
        assert uri_alias == uri_full


@pytest.mark.parametrize("load_config", ["huggingface-vllm-neuronx.json"], indirect=True)
def test_vllm_neuronx_all_regions(load_config):
    """Test that all regions have valid registry mappings."""
    version = "0.10.2"
    registries = load_config["inference"]["versions"][version]["registries"]

    for region in registries.keys():
        uri = get_huggingface_llm_image_uri(
            "huggingface-vllm-neuronx",
            region=region,
            version=version,
        )
        # Validate URI format
        assert uri.startswith(f"{registries[region]}.dkr.ecr.{region}")
        assert "huggingface-vllm-inference-neuronx" in uri
        assert "0.10.2" in uri
