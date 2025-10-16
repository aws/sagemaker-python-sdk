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

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris
from sagemaker.fw_utils import GRAVITON_ALLOWED_FRAMEWORKS

import pytest

GRAVITON_INSTANCE_TYPES = [
    "ml.m6g.4xlarge",
    "ml.m6gd.2xlarge",
    "ml.c6g.2xlarge",
    "ml.c6gd.4xlarge",
    "ml.c6gn.4xlarge",
    "ml.c7g.2xlarge",
    "ml.r6g.2xlarge",
    "ml.r6gd.4xlarge",
]


def _test_graviton_framework_uris(
    framework, version, py_version, account, region, container_version="ubuntu20.04-sagemaker"
):
    for instance_type in GRAVITON_INSTANCE_TYPES:
        uri = image_uris.retrieve(framework, region, instance_type=instance_type, version=version)
        expected = _expected_graviton_framework_uri(
            framework,
            version,
            py_version,
            account,
            region=region,
            container_version=container_version,
        )
        # Handle special regions
        domain = expected_uris.get_special_region_domain(region)
        if domain != ".amazonaws.com":
            expected = expected.replace(".amazonaws.com", domain)
        assert expected == uri


@pytest.mark.parametrize(
    "load_config_and_file_name", ["pytorch.json", "tensorflow.json"], indirect=True
)
@pytest.mark.parametrize("scope", ["inference_graviton"])
def test_graviton_framework_uris(load_config_and_file_name, scope):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    VERSIONS = config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = config[scope]["versions"][version]["registries"]
        py_versions = config[scope]["versions"][version]["py_versions"]
        container_version = (
            config[scope]["versions"][version].get("container_version", {}).get("cpu", None)
        )
        if container_version:
            container_version = container_version + "-sagemaker"
        for py_version in py_versions:
            for region in ACCOUNTS.keys():
                if container_version:
                    _test_graviton_framework_uris(
                        framework, version, py_version, ACCOUNTS[region], region, container_version
                    )
                else:
                    _test_graviton_framework_uris(
                        framework, version, py_version, ACCOUNTS[region], region
                    )


def _test_graviton_unsupported_framework(framework, region, framework_version):
    for instance_type in GRAVITON_INSTANCE_TYPES:
        with pytest.raises(ValueError) as error:
            image_uris.retrieve(
                framework, region, version=framework_version, instance_type=instance_type
            )
        expectedErr = (
            f"Unsupported framework: {framework}. Supported framework(s) for Graviton instances: "
            f"{GRAVITON_ALLOWED_FRAMEWORKS}."
        )
        assert expectedErr in str(error)


@pytest.mark.parametrize("load_config", ["pytorch.json"], indirect=True)
@pytest.mark.parametrize("scope", ["inference_graviton"])
def test_graviton_unsupported_framework(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            _test_graviton_unsupported_framework("autogluon", region, version)


def test_graviton_xgboost_instance_type_specified(graviton_xgboost_versions):
    for xgboost_version in graviton_xgboost_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                "xgboost", "us-west-2", version=xgboost_version, instance_type=instance_type
            )
            expected = (
                "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:"
                f"{xgboost_version}-arm64"
            )
            assert expected == uri


def test_graviton_xgboost_image_scope_specified(graviton_xgboost_versions):
    for xgboost_version in graviton_xgboost_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                "xgboost", "us-west-2", version=xgboost_version, image_scope="inference_graviton"
            )
            expected = (
                "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:"
                f"{xgboost_version}-arm64"
            )
            assert expected == uri


def test_graviton_xgboost_image_scope_specified_x86_instance(graviton_xgboost_versions):
    for xgboost_version in graviton_xgboost_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            with pytest.raises(ValueError) as error:
                image_uris.retrieve(
                    "xgboost",
                    "us-west-2",
                    version=xgboost_version,
                    image_scope="inference_graviton",
                    instance_type="ml.m5.xlarge",
                )
            assert "Unsupported instance type: m5." in str(error)


def test_graviton_xgboost_unsupported_version(graviton_xgboost_unsupported_versions):
    for xgboost_version in graviton_xgboost_unsupported_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            with pytest.raises(ValueError) as error:
                image_uris.retrieve(
                    "xgboost", "us-west-2", version=xgboost_version, instance_type=instance_type
                )
            assert f"Unsupported xgboost version: {xgboost_version}." in str(error)


def test_graviton_sklearn_instance_type_specified(graviton_sklearn_versions):
    for sklearn_version in graviton_sklearn_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                "sklearn", "us-west-2", version=sklearn_version, instance_type=instance_type
            )
            expected = (
                "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:"
                f"{sklearn_version}-arm64-cpu-py3"
            )
            assert expected == uri


def test_graviton_sklearn_image_scope_specified(graviton_sklearn_versions):
    for sklearn_version in graviton_sklearn_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                "sklearn", "us-west-2", version=sklearn_version, image_scope="inference_graviton"
            )
            expected = (
                "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:"
                f"{sklearn_version}-arm64-cpu-py3"
            )
            assert expected == uri


def test_graviton_sklearn_unsupported_version(graviton_sklearn_unsupported_versions):
    for sklearn_version in graviton_sklearn_unsupported_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                "sklearn", "us-west-2", version=sklearn_version, instance_type=instance_type
            )
            # Expected URI for SKLearn instead of ValueError because it only
            # supports one version for Graviton and therefore will always return
            # the default. See: image_uris._validate_version_and_set_if_needed
            expected = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3"
            assert expected == uri


def test_graviton_sklearn_image_scope_specified_x86_instance(graviton_sklearn_unsupported_versions):
    for sklearn_version in graviton_sklearn_unsupported_versions:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            with pytest.raises(ValueError) as error:
                image_uris.retrieve(
                    "sklearn",
                    "us-west-2",
                    version=sklearn_version,
                    image_scope="inference_graviton",
                    instance_type="ml.m5.xlarge",
                )
            assert "Unsupported instance type: m5." in str(error)


def _expected_graviton_framework_uri(
    framework, version, py_version, account, region, container_version
):
    return expected_uris.graviton_framework_uri(
        "{}-inference-graviton".format(framework),
        fw_version=version,
        py_version=py_version,
        account=account,
        region=region,
        container_version=container_version,
    )
