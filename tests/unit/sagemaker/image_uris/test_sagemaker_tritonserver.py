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
from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

INSTANCE_TYPES = {"cpu": "ml.c4.xlarge", "gpu": "ml.p2.xlarge"}


@pytest.mark.parametrize(
    "load_config_and_file_name",
    ["sagemaker-tritonserver.json"],
    indirect=True,
)
def test_sagemaker_tritonserver_uris(load_config_and_file_name):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    VERSIONS = config["versions"]
    processors = config["processors"]
    for version in VERSIONS:
        ACCOUNTS = config["versions"][version]["registries"]
        tag = config["versions"][version]["tag_prefix"]
        for processor in processors:
            instance_type = INSTANCE_TYPES[processor]
            for region in ACCOUNTS.keys():
                _test_sagemaker_tritonserver_uris(
                    ACCOUNTS[region], region, version, tag, framework, instance_type, processor
                )


def _test_sagemaker_tritonserver_uris(
    account, region, version, tag, triton_framework, instance_type, processor
):
    uri = image_uris.retrieve(
        framework=triton_framework, region=region, version=version, instance_type=instance_type
    )
    expected = expected_uris.sagemaker_triton_framework_uri(
        "sagemaker-tritonserver",
        account,
        tag,
        processor,
        region,
    )
    assert expected == uri
