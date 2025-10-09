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
from tests.unit.sagemaker.image_uris import expected_uris, conftest

LMI_VERSIONS = ["0.24.0"]
TEI_VERSIONS_MAPPING = {
    "gpu": {
        "1.2.3": "2.0.1-tei1.2.3-gpu-py310-cu122-ubuntu22.04",
        "1.4.0": "2.0.1-tei1.4.0-gpu-py310-cu122-ubuntu22.04",
        "1.6.0": "2.0.1-tei1.6.0-gpu-py310-cu122-ubuntu22.04",
        "1.7.0": "2.0.1-tei1.7.0-gpu-py310-cu122-ubuntu22.04",
        "1.8.0": "2.0.1-tei1.8.0-gpu-py310-cu122-ubuntu22.04",
    },
    "cpu": {
        "1.2.3": "2.0.1-tei1.2.3-cpu-py310-ubuntu22.04",
        "1.4.0": "2.0.1-tei1.4.0-cpu-py310-ubuntu22.04",
        "1.6.0": "2.0.1-tei1.6.0-cpu-py310-ubuntu22.04",
        "1.7.0": "2.0.1-tei1.7.0-cpu-py310-ubuntu22.04",
        "1.8.0": "2.0.1-tei1.8.0-cpu-py310-ubuntu22.04",
    },
}
HF_VERSIONS_MAPPING = {
    "gpu": {
        "0.6.0": "2.0.0-tgi0.6.0-gpu-py39-cu118-ubuntu20.04",
        "0.8.2": "2.0.0-tgi0.8.2-gpu-py39-cu118-ubuntu20.04",
        "0.9.3": "2.0.1-tgi0.9.3-gpu-py39-cu118-ubuntu20.04",
        "1.0.3": "2.0.1-tgi1.0.3-gpu-py39-cu118-ubuntu20.04",
        "1.1.0": "2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04",
        "1.2.0": "2.1.1-tgi1.2.0-gpu-py310-cu121-ubuntu20.04",
        "1.3.1": "2.1.1-tgi1.3.1-gpu-py310-cu121-ubuntu20.04",
        "1.3.3": "2.1.1-tgi1.3.3-gpu-py310-cu121-ubuntu20.04",
        "1.4.0": "2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04",
        "1.4.2": "2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04",
        "1.4.5": "2.1.1-tgi1.4.5-gpu-py310-cu121-ubuntu22.04",
        "2.0.0": "2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04",
        "2.0.1": "2.1.1-tgi2.0.1-gpu-py310-cu121-ubuntu22.04",
        "2.0.2": "2.3.0-tgi2.0.2-gpu-py310-cu121-ubuntu22.04",
        "2.2.0": "2.3.0-tgi2.2.0-gpu-py310-cu121-ubuntu22.04-v2.0",
        "2.3.1": "2.4.0-tgi2.3.1-gpu-py311-cu124-ubuntu22.04",
        "2.4.0": "2.4.0-tgi2.4.0-gpu-py311-cu124-ubuntu22.04-v2.2",
        "3.0.1": "2.4.0-tgi3.0.1-gpu-py311-cu124-ubuntu22.04-v2.1",
        "3.3.4": "2.7.0-tgi3.3.4-gpu-py311-cu124-ubuntu22.04",
        "3.3.6": "2.7.0-tgi3.3.6-gpu-py311-cu124-ubuntu22.04",
    },
    "inf2": {
        "0.0.16": "1.13.1-optimum0.0.16-neuronx-py310-ubuntu22.04",
        "0.0.17": "1.13.1-optimum0.0.17-neuronx-py310-ubuntu22.04",
        "0.0.18": "1.13.1-optimum0.0.18-neuronx-py310-ubuntu22.04",
        "0.0.19": "1.13.1-optimum0.0.19-neuronx-py310-ubuntu22.04",
        "0.0.20": "1.13.1-optimum0.0.20-neuronx-py310-ubuntu22.04",
        "0.0.21": "1.13.1-optimum0.0.21-neuronx-py310-ubuntu22.04",
        "0.0.22": "2.1.2-optimum0.0.22-neuronx-py310-ubuntu22.04",
        "0.0.23": "2.1.2-optimum0.0.23-neuronx-py310-ubuntu22.04",
        "0.0.24": "2.1.2-optimum0.0.24-neuronx-py310-ubuntu22.04",
        "0.0.25": "2.1.2-optimum0.0.25-neuronx-py310-ubuntu22.04",
        "0.0.27": "2.1.2-optimum0.0.27-neuronx-py310-ubuntu22.04",
    },
}


@pytest.mark.parametrize(
    "load_config", ["huggingface-llm.json", "huggingface-llm-neuronx.json"], indirect=True
)
def test_huggingface_uris(load_config):
    VERSIONS = load_config["inference"]["versions"]
    device = load_config["inference"]["processors"][0]
    backend = "huggingface-neuronx" if device == "inf2" else "huggingface"

    # Fail if device is not in mapping
    if device not in HF_VERSIONS_MAPPING:
        raise ValueError(f"Device {device} not found in HF_VERSIONS_MAPPING")

    # Get highest version for the device
    highest_version = max(HF_VERSIONS_MAPPING[device].keys(), key=lambda x: parse(x))

    for version in VERSIONS:
        ACCOUNTS = load_config["inference"]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = get_huggingface_llm_image_uri(backend, region=region, version=version)

            # Skip only if test version is higher than highest known version.
            # There's now automation to add new TGI releases to image_uri_config directory
            # that doesn't involve a human raising a PR.
            if parse(version) > parse(highest_version):
                print(
                    f"Skipping version check for {version} as there is "
                    "automation that now updates the image_uri_config "
                    "without a human raising a PR. Tests will pass for "
                    f"versions higher than {highest_version} that are not in HF_VERSIONS_MAPPING."
                )
                continue

            expected = expected_uris.huggingface_llm_framework_uri(
                "huggingface-pytorch-tgi-inference",
                ACCOUNTS[region],
                version,
                HF_VERSIONS_MAPPING[device][version],
                region=region,
            )
            assert expected == uri


@pytest.mark.parametrize(
    "load_config", ["huggingface-tei.json", "huggingface-tei-cpu.json"], indirect=True
)
def test_huggingface_tei_uris(load_config):
    VERSIONS = load_config["inference"]["versions"]
    device = load_config["inference"]["processors"][0]
    backend = "huggingface-tei" if device == "gpu" else "huggingface-tei-cpu"
    repo = "tei" if device == "gpu" else "tei-cpu"
    for version in VERSIONS:
        ACCOUNTS = load_config["inference"]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = get_huggingface_llm_image_uri(backend, region=region, version=version)
            expected = expected_uris.huggingface_llm_framework_uri(
                repo,
                ACCOUNTS[region],
                version,
                TEI_VERSIONS_MAPPING[device][version],
                region=region,
            )
            assert expected == uri


@pytest.mark.parametrize("load_config", ["huggingface-llm.json"], indirect=True)
def test_lmi_uris(load_config):
    VERSIONS = load_config["inference"]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config["inference"]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            for lmi_version in LMI_VERSIONS:
                djl_deepspeed_config = conftest.get_config("djl-deepspeed.json")
                DJL_DEEPSPEED_REGIONS = djl_deepspeed_config["versions"][lmi_version][
                    "registries"
                ].keys()
                if region not in DJL_DEEPSPEED_REGIONS:
                    continue

                uri = get_huggingface_llm_image_uri("lmi", region=region, version=lmi_version)
                tag = djl_deepspeed_config["versions"][lmi_version]["tag_prefix"]

                expected = expected_uris.djl_framework_uri(
                    "djl-inference", ACCOUNTS[region], tag, region=region
                )
                assert expected == uri
