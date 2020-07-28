# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.image_uri_config import framework_upgrade


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
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}


@pytest.fixture
def content():
    content = {
        "eia": {
            "processors": ["cpu"],
            "version_aliases": {"1.0": "1.0.0"},
            "versions": {
                "1.0.0": {
                    "registries": {
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
                        "us-west-1": "763104351884",
                        "us-west-2": "763104351884",
                    },
                    "repository": "tensorflow-inference-eia",
                    "py_versions": ["py2", "py3"],
                }
            },
        }
    }
    return content


@pytest.fixture
def expected_content():
    new_content = {
        "eia": {
            "processors": ["cpu", "gpu"],
            "version_aliases": {"1.0": "1.0.0", "2.0": "2.0.0"},
            "versions": {
                "2.0.0": {
                    "registries": {
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
                        "us-west-1": "763104351884",
                        "us-west-2": "763104351884",
                    },
                    "repository": "tensorflow-inference-eia",
                    "py_versions": ["py37"],
                }
            },
        }
    }
    return new_content


def test_add_tensorflow_eia(content, expected_content):
    framework = "tensorflow"
    processors = ["cpu", "gpu"]
    short_version = "2.0"
    full_version = "2.0.0"
    type = "eia"
    py_versions = ["py37"]
    framework_upgrade.add_dlc_framework_version(
        content,
        framework,
        short_version,
        full_version,
        type,
        processors,
        py_versions,
        FRAMEWORK_REGION_REGISTRY,
    )
    assert content["eia"]["processors"] == expected_content["eia"]["processors"]
    assert content["eia"]["version_aliases"] == expected_content["eia"]["version_aliases"]
    assert (
        content["eia"]["versions"][full_version]
        == expected_content["eia"]["versions"][full_version]
    )
