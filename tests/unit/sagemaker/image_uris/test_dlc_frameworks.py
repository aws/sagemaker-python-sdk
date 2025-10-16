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

from packaging.version import Version

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

import pytest

COMMON_INSTANCE_TYPES = {"cpu": "ml.c4.xlarge", "gpu": "ml.p2.xlarge"}
RENEWED_INSTANCE_TYPES = {"cpu": "ml.c4.xlarge", "gpu": "ml.g4dn.xlarge"}


@pytest.mark.parametrize("load_config", ["chainer.json"], indirect=True)
def test_chainer_uris(load_config):
    VERSIONS = load_config["versions"]
    for version in VERSIONS:
        ALGO_ACCOUNTS = load_config["versions"][version]["registries"]
        py_versions = load_config["versions"][version]["py_versions"]
        repo = load_config["versions"][version]["repository"]
        processors = load_config["processors"]

        for processor in processors:
            instance_type = COMMON_INSTANCE_TYPES[processor]
            for py_version in py_versions:
                for region in ALGO_ACCOUNTS.keys():
                    uri = image_uris.retrieve(
                        "chainer",
                        region=region,
                        version=version,
                        py_version=py_version,
                        image_scope="training",
                        instance_type=instance_type,
                    )
                    expected = expected_uris.framework_uri(
                        repo=repo,
                        fw_version=version,
                        py_version=py_version,
                        processor=processor,
                        region=region,
                        account=ALGO_ACCOUNTS[region],
                    )
                    assert uri == expected


@pytest.mark.parametrize(
    "load_config_and_file_name", ["tensorflow.json", "mxnet.json", "pytorch.json"], indirect=True
)
@pytest.mark.parametrize("scope", ["training", "inference", "eia"])
def test_dlc_framework_uris(load_config_and_file_name, scope):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    VERSIONS = config[scope]["versions"]

    for version in VERSIONS:
        if not config[scope]["versions"][version].get("registries"):
            continue
        ACCOUNTS = config[scope]["versions"][version]["registries"]
        repo = config[scope]["versions"][version]["repository"]
        py_versions = config[scope]["versions"][version].get("py_versions", [None])
        processors = config[scope].get("processors", ["cpu", "gpu"])

        for processor in processors:
            instance_type = COMMON_INSTANCE_TYPES[processor]
            if (framework == "pytorch" and Version(version) >= Version("1.13")) or (
                framework == "tensorflow" and Version(version) >= Version("2.12")
            ):
                instance_type = RENEWED_INSTANCE_TYPES[processor]
            for py_version in py_versions:
                for region in ACCOUNTS.keys():
                    if scope == "eia":
                        uri = image_uris.retrieve(
                            framework,
                            region=region,
                            version=version,
                            py_version=py_version,
                            image_scope="inference",
                            instance_type=instance_type,
                            accelerator_type="ml.eia1.medium",
                        )
                    else:
                        uri = image_uris.retrieve(
                            framework,
                            region=region,
                            version=version,
                            py_version=py_version,
                            image_scope=scope,
                            instance_type=instance_type,
                        )
                    expected = expected_uris.framework_uri(
                        repo=repo,
                        fw_version=version,
                        py_version=py_version,
                        processor=processor,
                        region=region,
                        account=ACCOUNTS[region],
                    )
                    # Handle special regions
                    domain = expected_uris.get_special_region_domain(region)
                    if domain != ".amazonaws.com":
                        expected = expected.replace(".amazonaws.com", domain)

                    assert uri == expected


@pytest.mark.parametrize(
    "load_config_and_file_name", ["tensorflow.json", "mxnet.json"], indirect=True
)
def test_uncommon_format_dlc_framework_version_uris(load_config_and_file_name):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    py_versions = ["py2", "py3"]

    # These versions are formatted differently than others for their framework
    if framework == "tensorflow":
        SCOPES = ["training"]
        VERSIONS = ["1.13.1"]
    elif framework == "mxnet":
        SCOPES = ["training", "inference"]
        VERSIONS = ["1.4.1"]

    for scope in SCOPES:
        for py_version in py_versions:
            for version in VERSIONS:
                ACCOUNTS = config[scope]["versions"][version][py_version]["registries"]
                processors = config[scope].get("processors", ["cpu", "gpu"])
                repo = config[scope]["versions"][version][py_version]["repository"]
                for processor in processors:
                    instance_type = COMMON_INSTANCE_TYPES[processor]
                    for region in ACCOUNTS.keys():
                        uri = image_uris.retrieve(
                            framework,
                            region=region,
                            version=version,
                            py_version=py_version,
                            image_scope=scope,
                            instance_type=instance_type,
                        )
                        expected = expected_uris.framework_uri(
                            repo=repo,
                            fw_version=version,
                            py_version=py_version,
                            processor=processor,
                            region=region,
                            account=ACCOUNTS[region],
                        )
                        assert uri == expected
