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
    "ca-central-1": "763104351884",
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
    "us-gov-east-1": "446045086412",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-isob-east-1": "094389454867",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}
VERSIONS = [
    "0.3.1",
    "0.3.2",
    "0.4.0",
    "0.4.2",
    "0.4.3",
    "0.3",
    "0.4",
    "0.5.2",
    "0.5",
    "0.6.1",
    "0.6",
    "0.6.2",
    "0.7.0",
    "0.7",
    "0.8",
    "0.8.2"
]

SCOPES = ["training", "inference"]
PROCESSORS = ["cpu", "gpu"]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("processor", PROCESSORS)
def test_valid_uris_training(version, scope, processor):
    instance_type = "ml.c4.xlarge" if processor == "cpu" else "ml.p2.xlarge"
    if version == "0.3.1":
        py_version = "py37"
    elif version < "0.7":
        py_version = "py38"
    else:
        py_version = "py39"
    if (
        scope == "inference"
        and processor == "gpu"
        and version in ["0.3.1", "0.3.2", "0.4.0", "0.3"]
    ):
        with pytest.raises(ValueError) as e:
            image_uris.retrieve(
                "autogluon",
                region="us-west-2",
                version=version,
                py_version=py_version,
                image_scope=scope,
                instance_type=instance_type,
            )

        assert "Unsupported processor: gpu." in str(e.value)
    else:
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


@pytest.mark.parametrize("version", VERSIONS)
def test_py3_error(version):
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
