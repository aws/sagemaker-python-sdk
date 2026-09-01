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

GPU_INSTANCE = "ml.g5.2xlarge"
CPU_INSTANCE = "ml.m5.xlarge"
DEFAULT_DOMAIN = "amazonaws.com"

# Regions whose ECR host suffix is stable across botocore versions, used for the
# exact-URI assertions; the remaining regions are checked on account/region/repo/tag
# so the suite does not depend on botocore endpoint data for newer ISO partitions.
FULL_URI_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "cn-north-1", "us-gov-west-1"]

# The newest TensorFlow version per scope, with the Python version baked into its
# tag. These are asserted as the maximum registered version so that adding a newer
# one to tensorflow.json fails here until this test is updated deliberately.
#   tensorflow-inference:2.20.0-{cpu,gpu}-py312
#   tensorflow-training:2.21.0-{cpu,gpu}-py312
LATEST = {
    "inference": ("2.20.0", "py312"),
    "training": ("2.21.0", "py312"),
}


def _expected_repo(scope):
    return "tensorflow-inference" if scope == "inference" else "tensorflow-training"


@pytest.mark.parametrize("scope", ["inference", "training"])
@pytest.mark.parametrize("load_config", ["tensorflow.json"], indirect=True)
def test_tensorflow_latest_version_is_registered(load_config, scope):
    """The newest version in tensorflow.json is the one this file covers."""
    version, py_version = LATEST[scope]
    versions = load_config[scope]["versions"]
    assert version in versions, f"{version} missing from tensorflow.json {scope}"
    assert versions[version]["repository"] == _expected_repo(scope)
    assert versions[version]["py_versions"] == [py_version]
    assert load_config[scope]["processors"] == ["cpu", "gpu"]


@pytest.mark.parametrize("scope", ["inference", "training"])
@pytest.mark.parametrize("load_config", ["tensorflow.json"], indirect=True)
def test_tensorflow_latest_version_uris(load_config, scope):
    """Every (processor, region) for the newest version resolves to the expected tag."""
    version, py_version = LATEST[scope]
    version_config = load_config[scope]["versions"][version]
    repo = _expected_repo(scope)
    for processor, instance_type in (("cpu", CPU_INSTANCE), ("gpu", GPU_INSTANCE)):
        expected_tag = f"{version}-{processor}-{py_version}"
        for region, account in version_config["registries"].items():
            uri = image_uris.retrieve(
                framework="tensorflow",
                region=region,
                version=version,
                image_scope=scope,
                instance_type=instance_type,
            )
            assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
            assert uri.endswith(f"/{repo}:{expected_tag}"), uri


@pytest.mark.parametrize("scope", ["inference", "training"])
@pytest.mark.parametrize("load_config", ["tensorflow.json"], indirect=True)
def test_tensorflow_latest_version_full_uri(load_config, scope):
    """Exact URI (including domain) for representative commercial/China/GovCloud regions."""
    version, py_version = LATEST[scope]
    version_config = load_config[scope]["versions"][version]
    repo = _expected_repo(scope)
    for region in FULL_URI_REGIONS:
        account = version_config["registries"][region]
        domain = ALTERNATE_DOMAINS.get(region, DEFAULT_DOMAIN)
        for processor, instance_type in (("cpu", CPU_INSTANCE), ("gpu", GPU_INSTANCE)):
            uri = image_uris.retrieve(
                framework="tensorflow",
                region=region,
                version=version,
                image_scope=scope,
                instance_type=instance_type,
            )
            expected_tag = f"{version}-{processor}-{py_version}"
            assert uri == f"{account}.dkr.ecr.{region}.{domain}/{repo}:{expected_tag}"


@pytest.mark.parametrize("scope", ["inference", "training"])
@pytest.mark.parametrize("load_config", ["tensorflow.json"], indirect=True)
def test_tensorflow_minor_alias_resolves_to_newest_patch(load_config, scope):
    """The minor alias (2.20 / 2.21) points at its newest patch and keeps the py suffix."""
    version, py_version = LATEST[scope]
    alias = version.rsplit(".", 1)[0]
    assert load_config[scope]["version_aliases"][alias] == version
    for processor, instance_type in (("cpu", CPU_INSTANCE), ("gpu", GPU_INSTANCE)):
        uri = image_uris.retrieve(
            framework="tensorflow",
            region="us-west-2",
            version=alias,
            image_scope=scope,
            instance_type=instance_type,
        )
        # The alias is used verbatim as the tag prefix, matching the published
        # `<minor>-<processor>-py312` tags.
        assert uri.endswith(f"/{_expected_repo(scope)}:{alias}-{processor}-{py_version}"), uri
