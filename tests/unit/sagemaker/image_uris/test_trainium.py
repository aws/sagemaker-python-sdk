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

import pytest


TRAINIUM_ALLOWED_FRAMEWORKS = "pytorch"


def _expected_trainium_framework_uri(
    framework, version, account, region="us-west-2", inference_tool="neuron"
):
    return expected_uris.neuron_framework_uri(
        "{}-neuron".format(framework),
        fw_version=version,
        py_version="py38",
        account=account,
        region=region,
        inference_tool=inference_tool,
    )


@pytest.mark.parametrize("load_config", ["pytorch-neuron.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training"])
def test_trainium_pytorch_uris(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = image_uris.retrieve(
                "pytorch", region, instance_type="ml.trn1.xlarge", version=version
            )
            expected = _expected_trainium_framework_uri(
                "{}-training".format("pytorch"),
                version,
                ACCOUNTS[region],
                region=region,
                inference_tool="neuron",
            )
            assert expected == uri


@pytest.mark.parametrize("load_config", ["pytorch-neuron.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training"])
def test_trainium_unsupported_framework(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            with pytest.raises(ValueError) as error:
                image_uris.retrieve(
                    "autogluon", region, instance_type="ml.trn1.xlarge", version=version
                )
                expectedErr = (
                    f"Unsupported framework: autogluon. Supported framework(s) for Trainium instances: "
                    f"{TRAINIUM_ALLOWED_FRAMEWORKS}."
                )
                assert expectedErr in str(error)
