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
from sagemaker.core.common_utils import ALTERNATE_DOMAINS

# vLLM images are GPU-only; a GPU instance type selects the "gpu" processor.
INSTANCE_TYPE = "ml.g5.2xlarge"
DEFAULT_DOMAIN = "amazonaws.com"

# Regions whose ECR host suffix is stable across botocore versions (commercial,
# China, GovCloud). Exact-host assertions are limited to these; every other region
# is still covered by the account/region/repository/tag checks in test_vllm_uris,
# which avoids depending on botocore endpoint data for newer ISO partitions.
FULL_URI_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "cn-north-1", "us-gov-west-1"]


@pytest.mark.parametrize("load_config", ["vllm.json"], indirect=True)
def test_vllm_uris(load_config):
    """Every (version, region) resolves to the vllm repo with the expected account + tag."""
    config = load_config
    assert config["inference"]["processors"] == ["gpu"]
    versions = config["inference"]["versions"]
    for version, version_config in versions.items():
        py_version = version_config["py_versions"][0]
        container_version = version_config["container_version"]["gpu"]
        expected_tag = f"{version}-gpu-{py_version}-{container_version}"
        for region, account in version_config["registries"].items():
            uri = image_uris.retrieve(
                framework="vllm",
                region=region,
                version=version,
                image_scope="inference",
                instance_type=INSTANCE_TYPE,
            )
            # account (registry), region, repository and tag are config-controlled;
            # the domain suffix is resolved by botocore and asserted separately below.
            assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
            assert uri.endswith(f"/vllm:{expected_tag}"), uri


@pytest.mark.parametrize("load_config", ["vllm.json"], indirect=True)
def test_vllm_full_uri_for_representative_regions(load_config):
    """Exact URI (including domain) for representative commercial/China/GovCloud regions."""
    config = load_config
    versions = config["inference"]["versions"]
    for version, version_config in versions.items():
        py_version = version_config["py_versions"][0]
        container_version = version_config["container_version"]["gpu"]
        for region in FULL_URI_REGIONS:
            if region not in version_config["registries"]:
                continue
            account = version_config["registries"][region]
            domain = ALTERNATE_DOMAINS.get(region, DEFAULT_DOMAIN)
            expected = (
                f"{account}.dkr.ecr.{region}.{domain}"
                f"/vllm:{version}-gpu-{py_version}-{container_version}"
            )
            uri = image_uris.retrieve(
                framework="vllm",
                region=region,
                version=version,
                image_scope="inference",
                instance_type=INSTANCE_TYPE,
            )
            assert uri == expected


@pytest.mark.parametrize("load_config", ["vllm.json"], indirect=True)
def test_vllm_version_aliases_resolve_to_newest_patch(load_config):
    """Each minor alias resolves to its newest patch version."""
    config = load_config
    aliases = config["inference"]["version_aliases"]
    for alias, target_version in aliases.items():
        uri = image_uris.retrieve(
            framework="vllm",
            region="us-west-2",
            version=alias,
            image_scope="inference",
            instance_type=INSTANCE_TYPE,
        )
        assert f"/vllm:{target_version}-gpu-" in uri, uri
