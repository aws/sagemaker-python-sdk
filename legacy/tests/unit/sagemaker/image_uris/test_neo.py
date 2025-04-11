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

NEO_FRAMEWORK_INSTANCE_TYPES = {"cpu": "ml_c5", "gpu": "ml_p2"}
INFERNTIA_FRAMEWORK_INSTANCE_TYPES = {"inf": "ml_inf1"}

NEO_ALGOS = ["image-classification-neo.json", "xgboost-neo.json"]
NEO_FRAMEWORKS = ["neo-pytorch.json", "neo-tensorflow.json", "neo-mxnet.json"]
INFERENTIA_FRAMEWORKS = [
    "inferentia-mxnet.json",
    "inferentia-pytorch.json",
    "inferentia-tensorflow.json",
]
NEO_AND_INFERENTIA_FRAMEWORKS = NEO_FRAMEWORKS + INFERENTIA_FRAMEWORKS


@pytest.mark.parametrize("load_config_and_file_name", NEO_ALGOS, indirect=True)
def test_neo_alogos(load_config_and_file_name):
    config, file_name = load_config_and_file_name
    algo = file_name.split(".json")[0]
    VERSIONS = config["versions"]
    for version in VERSIONS:
        ACCOUNTS = config["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = image_uris.retrieve(algo, region)
            expected = expected_uris.algo_uri(algo, ACCOUNTS[region], region, version)
            assert uri == expected


@pytest.mark.parametrize("load_config_and_file_name", NEO_AND_INFERENTIA_FRAMEWORKS, indirect=True)
def test_neo_and_inferentia_frameworks(load_config_and_file_name):
    config, file_name = load_config_and_file_name
    framework = file_name.split(".json")[0]
    VERSIONS = config["versions"]
    processors = config["processors"]
    for version in VERSIONS:
        ACCOUNTS = config["versions"][version]["registries"]
        py_versions = config["versions"][version].get("py_versions", [None])
        repo = config["versions"][version]["repository"]
        for processor in processors:
            if file_name in NEO_FRAMEWORKS:
                instance_type = NEO_FRAMEWORK_INSTANCE_TYPES[processor]
            else:
                instance_type = INFERNTIA_FRAMEWORK_INSTANCE_TYPES[processor]

            for py_version in py_versions:
                for region in ACCOUNTS.keys():
                    uri = image_uris.retrieve(
                        framework, region, instance_type=instance_type, version=version
                    )
                    expected = expected_uris.framework_uri(
                        repo,
                        fw_version=version,
                        py_version=py_version,
                        account=ACCOUNTS[region],
                        region=region,
                        processor=processor,
                    )
                    assert uri == expected
