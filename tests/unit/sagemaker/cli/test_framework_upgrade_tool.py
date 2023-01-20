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

from sagemaker.cli import framework_upgrade


FRAMEWORK_REGION_REGISTRY = {
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ca-central-1": "763104351884",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "me-south-1": "217643126080",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-isob-east-1": "094389454867",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}


@pytest.fixture
def dlc_content():
    content = {
        "eia": {
            "processors": ["cpu"],
            "version_aliases": {"1.0": "1.0.0"},
            "versions": {
                "1.0.0": {
                    "registries": FRAMEWORK_REGION_REGISTRY,
                    "repository": "tensorflow-inference-eia",
                    "py_versions": ["py2", "py3"],
                }
            },
        }
    }
    return content


@pytest.fixture
def dlc_expected_content():
    new_content = {
        "eia": {
            "processors": ["cpu", "gpu"],
            "version_aliases": {"1.0": "1.0.0", "2.0": "2.0.0"},
            "versions": {
                "2.0.0": {
                    "registries": FRAMEWORK_REGION_REGISTRY,
                    "repository": "tensorflow-inference-eia",
                    "py_versions": ["py37"],
                }
            },
        }
    }
    return new_content


@pytest.fixture
def algo_content():
    content = {
        "processors": ["cpu"],
        "scope": ["training"],
        "versions": {
            "0.10": {
                "py_versions": ["py3"],
                "registries": FRAMEWORK_REGION_REGISTRY,
                "repository": "sagemaker-rl-tensorflow",
                "tag_prefix": "coach0.10",
            }
        },
    }
    return content


@pytest.fixture
def algo_expected_content():
    new_content = {
        "processors": ["cpu", "gpu"],
        "scope": ["training", "inference"],
        "versions": {
            "0.20": {
                "py_versions": ["py3", "py37"],
                "registries": FRAMEWORK_REGION_REGISTRY,
                "repository": "sagemaker-rl-tensorflow",
                "tag_prefix": "coach0.20",
            }
        },
    }
    return new_content


@pytest.fixture
def dlc_region_content():
    region_info = {
        "training": {"versions": {"1.0": {"registries": {"us-west-2": "123456789012"}}}},
        "inference": {"versions": {"1.0": {"registries": {"us-west-2": "123456789012"}}}},
    }
    return region_info


@pytest.fixture
def dlc_expected_region_content():
    region_info = {
        "training": {
            "versions": {
                "1.0": {"registries": {"us-west-2": "123456789012", "us-east-1": "987654321098"}}
            }
        },
        "inference": {
            "versions": {
                "1.0": {"registries": {"us-west-2": "123456789012", "us-east-1": "987654321098"}}
            }
        },
    }
    return region_info


@pytest.fixture
def algo_region_content():
    region_info = {
        "scope": ["training"],
        "versions": {"1.0": {"registries": {"us-west-2": "123456789012"}}},
    }
    return region_info


@pytest.fixture
def algo_expected_region_content():
    region_info = {
        "scope": ["training"],
        "versions": {
            "1.0": {"registries": {"us-west-2": "123456789012", "us-east-1": "987654321098"}}
        },
    }
    return region_info


@pytest.fixture
def dlc_no_optional_content():
    content = {
        "inference": {
            "processors": ["cpu"],
            "version_aliases": {"1.0": "1.0.0"},
            "versions": {
                "1.0.0": {
                    "registries": FRAMEWORK_REGION_REGISTRY,
                    "repository": "tensorflow-inference",
                }
            },
        }
    }
    return content


@pytest.fixture
def algo_no_optional_content():
    content = {
        "processors": ["cpu"],
        "scope": ["training"],
        "versions": {
            "0.10": {
                "registries": FRAMEWORK_REGION_REGISTRY,
                "repository": "sagemaker-rl-tensorflow",
            }
        },
    }
    return content


def test_add_tensorflow_eia(dlc_content, dlc_expected_content):
    latest_repository = "tensorflow-inference-eia"
    processors = ["cpu", "gpu"]
    short_version = "2.0"
    full_version = "2.0.0"
    type = "eia"
    py_versions = ["py37"]
    framework_upgrade.add_dlc_framework_version(
        dlc_content,
        short_version,
        full_version,
        type,
        processors,
        py_versions,
        FRAMEWORK_REGION_REGISTRY,
        latest_repository,
    )
    assert dlc_content["eia"]["processors"] == dlc_expected_content["eia"]["processors"]
    assert dlc_content["eia"]["version_aliases"] == dlc_expected_content["eia"]["version_aliases"]
    assert (
        dlc_content["eia"]["versions"][full_version]
        == dlc_expected_content["eia"]["versions"][full_version]
    )


def test_add_coach_tensorflow(algo_content, algo_expected_content):
    processors = ["cpu", "gpu"]
    full_version = "0.20"
    scopes = ["training", "inference"]
    py_versions = ["py3", "py37"]
    repository = "sagemaker-rl-tensorflow"
    tag_prefix = "coach0.20"
    framework_upgrade.add_algo_version(
        algo_content,
        processors,
        scopes,
        full_version,
        py_versions,
        FRAMEWORK_REGION_REGISTRY,
        repository,
        tag_prefix,
    )
    assert algo_content["processors"] == algo_expected_content["processors"]
    assert algo_content["scope"] == algo_expected_content["scope"]
    assert algo_content["versions"][full_version] == algo_expected_content["versions"][full_version]


def test_dlc_add_region(dlc_region_content, dlc_expected_region_content):
    region = "us-east-1"
    account = "987654321098"
    framework_upgrade.add_region(dlc_region_content, region, account)
    assert dlc_region_content == dlc_expected_region_content


def test_algo_add_region(algo_region_content, algo_expected_region_content):
    region = "us-east-1"
    account = "987654321098"
    framework_upgrade.add_region(algo_region_content, region, account)
    assert algo_region_content == algo_expected_region_content


def test_dlc_get_latest_content(dlc_content):
    latest_version = "1.0.0"
    scope = "eia"
    registries, py_versions, repository = framework_upgrade.get_latest_values(
        dlc_content, scope=scope
    )
    assert registries == dlc_content[scope]["versions"][latest_version]["registries"]
    assert py_versions == dlc_content[scope]["versions"][latest_version]["py_versions"]
    assert repository == dlc_content[scope]["versions"][latest_version]["repository"]


def test_algo_get_latest_content(algo_content):
    latest_version = "0.10"
    registries, py_versions, repository = framework_upgrade.get_latest_values(algo_content)
    assert registries == algo_content["versions"][latest_version]["registries"]
    assert py_versions == algo_content["versions"][latest_version]["py_versions"]
    assert repository == algo_content["versions"][latest_version]["repository"]


def test_dlc_get_latest_content_no_optional(dlc_no_optional_content):
    latest_version = "1.0.0"
    scope = "inference"
    registries, py_versions, repository = framework_upgrade.get_latest_values(
        dlc_no_optional_content, scope=scope
    )
    assert py_versions is None
    assert registries == dlc_no_optional_content[scope]["versions"][latest_version]["registries"]
    assert repository == dlc_no_optional_content[scope]["versions"][latest_version]["repository"]


def test_algo_get_latest_content_no_optional(algo_no_optional_content):
    latest_version = "0.10"
    registries, py_versions, repository = framework_upgrade.get_latest_values(
        algo_no_optional_content
    )
    assert py_versions is None
    assert registries == algo_no_optional_content["versions"][latest_version]["registries"]
    assert repository == algo_no_optional_content["versions"][latest_version]["repository"]
