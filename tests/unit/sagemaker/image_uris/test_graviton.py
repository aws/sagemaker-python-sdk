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

GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY = [
    "m6g",
    "m6gd",
    "c6g",
    "c6gd",
    "c6gn",
    "c7g",
    "r6g",
    "r6gd",
]
GRAVITON_ALGOS = ("tensorflow", "pytorch")
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

ACCOUNTS = {
    "af-south-1": "626614931356",
    "il-central-1": "780543022126",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-southeast-3": "907027046896",
    "ca-central-1": "763104351884",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "eu-south-1": "692866216735",
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

GRAVITON_REGIONS = ACCOUNTS.keys()


def _test_graviton_framework_uris(framework, version):
    for region in GRAVITON_REGIONS:
        for instance_type in GRAVITON_INSTANCE_TYPES:
            uri = image_uris.retrieve(
                framework, region, instance_type=instance_type, version=version
            )
            expected = _expected_graviton_framework_uri(framework, version, region=region)
            assert expected == uri


def test_graviton_tensorflow(graviton_tensorflow_version):
    _test_graviton_framework_uris("tensorflow", graviton_tensorflow_version)


def test_graviton_pytorch(graviton_pytorch_version):
    _test_graviton_framework_uris("pytorch", graviton_pytorch_version)


def _test_graviton_unsupported_framework(framework, framework_version):
    for region in GRAVITON_REGIONS:
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


def test_graviton_unsupported_framework():
    _test_graviton_unsupported_framework("autogluon", "0.6.1")


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


def _expected_graviton_framework_uri(framework, version, region):
    return expected_uris.graviton_framework_uri(
        "{}-inference-graviton".format(framework),
        fw_version=version,
        py_version="py38",
        account=ACCOUNTS[region],
        region=region,
    )
