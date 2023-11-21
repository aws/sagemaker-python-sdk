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
    [
        "coach-tensorflow.json",
        "coach-mxnet.json",
        "ray-tensorflow.json",
        "ray-pytorch.json",
        "vw.json",
    ],
    indirect=True,
)
def test_rl_image_uris(load_config_and_file_name):
    config, filename = load_config_and_file_name
    framework = filename.split(".json")[0]
    VERSIONS = config["versions"]
    processors = config["processors"]
    for version in VERSIONS:
        ACCOUNTS = config["versions"][version]["registries"]
        py_versions = config["versions"][version].get("py_versions", [None])
        repo = config["versions"][version]["repository"]
        tag_prefix = config["versions"][version]["tag_prefix"]
        for processor in processors:
            instance_type = INSTANCE_TYPES[processor]
            for py_version in py_versions:
                for region in ACCOUNTS.keys():
                    uri = image_uris.retrieve(
                        framework, region, version=version, instance_type=instance_type
                    )

                    expected = expected_uris.framework_uri(
                        repo,
                        tag_prefix,
                        ACCOUNTS[region],
                        py_version=py_version,
                        processor=processor,
                        region=region,
                    )

                    assert uri == expected
