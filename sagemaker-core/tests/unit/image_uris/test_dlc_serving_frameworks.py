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

# GPU instance; these configs omit "processors", so the processor slot is dropped
# and the whole tag is taken verbatim from tag_prefix (instance type is ignored).
INSTANCE_TYPE = "ml.g5.2xlarge"

# DLC serving-framework configs that use the whole-tag (djl-lmi style) pattern:
# each version's tag_prefix is the full image tag (channel or amzn2023), with no
# processor/py/container_version tokens appended.
SERVING_CONFIG_FILES = [
    "vllm-server.json",
    "vllm-omni.json",
    "sglang-server.json",
    "llama-cpp.json",
    "llama-cpp-arm64.json",
    "ray-serve.json",
    "whisperx.json",
]


@pytest.mark.parametrize("load_config_and_file_name", SERVING_CONFIG_FILES, indirect=True)
def test_serving_framework_uris(load_config_and_file_name):
    """Every (version, region) resolves to the expected repo:tag verbatim."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    for version, version_config in config["versions"].items():
        repo = version_config["repository"]
        tag = version_config["tag_prefix"]
        for region, account in version_config["registries"].items():
            uri = image_uris.retrieve(
                framework=framework,
                region=region,
                version=version,
                image_scope="inference",
                instance_type=INSTANCE_TYPE,
            )
            # account (registry), region, repository and tag are config-controlled;
            # the domain suffix is resolved by botocore.
            assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
            assert uri.endswith(f"/{repo}:{tag}"), uri


@pytest.mark.parametrize("load_config_and_file_name", SERVING_CONFIG_FILES, indirect=True)
def test_serving_framework_latest_alias(load_config_and_file_name):
    """The 'latest' alias resolves to its target version's tag."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    target = config["version_aliases"]["latest"]
    expected = config["versions"][target]
    uri = image_uris.retrieve(
        framework=framework,
        region="us-west-2",
        version="latest",
        image_scope="inference",
        instance_type=INSTANCE_TYPE,
    )
    assert uri.endswith(f"/{expected['repository']}:{expected['tag_prefix']}"), uri
