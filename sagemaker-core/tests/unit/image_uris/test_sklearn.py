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


@pytest.mark.parametrize("load_config", ["sklearn.json"], indirect=True)
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_sklearn_uris(load_config, scope):
    VERSIONS = load_config[scope]["versions"]
    for version in VERSIONS:
        ACCOUNTS = load_config[scope]["versions"][version]["registries"]
        for region in ACCOUNTS.keys():
            uri = image_uris.retrieve(
                "sklearn",
                region=region,
                version=version,
                py_version="py3",
                instance_type="ml.c4.xlarge",
            )

            expected = expected_uris.framework_uri(
                "sagemaker-scikit-learn",
                version,
                ACCOUNTS[region],
                py_version="py3",
                region=region,
            )
            assert expected == uri


def test_py2_error(sklearn_version):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            "sklearn",
            region="us-west-2",
            version=sklearn_version,
            py_version="py2",
            instance_type="ml.c4.xlarge",
        )

    assert "Unsupported Python version: py2." in str(e.value)


def test_gpu_error(sklearn_version):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            "sklearn",
            region="us-west-2",
            version=sklearn_version,
            instance_type="ml.p2.xlarge",
        )

    assert "Unsupported processor: gpu." in str(e.value)
