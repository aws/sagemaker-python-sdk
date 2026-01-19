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
from . import expected_uris

INSTANCE_TYPES = {"cpu": "ml.c4.xlarge", "gpu": "ml.p2.xlarge"}


@pytest.mark.parametrize("load_config", ["autogluon.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_autogluon_uris(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        py_versions = load_config[scope]["versions"][version]["py_versions"]
        processors = load_config[scope]["versions"][version].get("processors", ["cpu"])

        for processor in processors:
            instance_type = INSTANCE_TYPES[processor]
            for py_version in py_versions:
                for region in ACCOUNTS.keys():
                    uri = image_uris.retrieve(
                        "autogluon",
                        region=region,
                        version=version,
                        py_version=py_version,
                        image_scope=scope,
                        instance_type=instance_type,
                    )
                    expected = expected_uris.framework_uri(
                        f"autogluon-{scope}",
                        version,
                        ACCOUNTS[region],
                        py_version=py_version,
                        region=region,
                        processor=processor,
                    )

                    assert uri == expected


@pytest.mark.parametrize("load_config", ["autogluon.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training"])
def test_py3_error(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        with pytest.raises(ValueError) as e:
            image_uris.retrieve(
                "autogluon",
                region="us-west-2",
                version=version,
                py_version="py3",
                image_scope="training",
                instance_type="ml.c4.xlarge",
            )

    assert "Unsupported Python version: py3." in str(e.value)
