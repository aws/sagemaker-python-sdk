# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import copy

import pytest
from mock import patch

from sagemaker import image_uris

BASE_CONFIG = {
    "processors": ["cpu", "gpu"],
    "versions": {
        "1.0.0": {
            "registries": {"us-west-2": "123412341234"},
            "repository": "dummy",
            "py_versions": ["py3"],
        },
    },
}


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_framework_all_args(config_for_framework):
    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_aliased_version(config_for_framework):
    version = "1.0.0-build123"

    config = copy.deepcopy(BASE_CONFIG)
    config["version_aliases"] = {version: "1.0.0"}
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version=version,
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:{}-cpu-py3".format(version) == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_version(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            version="1",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
        )

    assert "Unsupported some-framework version: 1." in str(e.value)
    assert "Supported version(s): 1.0.0." in str(e.value)

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
        )

    assert "Unsupported some-framework version: None." in str(e.value)
    assert "Supported version(s): 1.0.0." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_region(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-east-2",
        )

    assert "Unsupported region: us-east-2." in str(e.value)
    assert "Supported region(s): us-west-2." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_ecr_hostname(config_for_framework):
    registries = {
        "cn-north-1": "000000010010",
        "cn-northwest-1": "010000001000",
        "us-iso-east-1": "000111111000",
    }

    config = copy.deepcopy(BASE_CONFIG)
    config["versions"]["1.0.0"]["registries"] = registries
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="cn-north-1",
    )
    assert "000000010010.dkr.ecr.cn-north-1.amazonaws.com.cn/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="cn-northwest-1",
    )
    assert "010000001000.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-iso-east-1",
    )
    assert "000111111000.dkr.ecr.us-iso-east-1.c2s.ic.gov/dummy:1.0.0-cpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_python_version(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            version="1.0.0",
            py_version="py2",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
        )

    assert "Unsupported Python version for some-framework 1.0.0: py2." in str(e.value)
    assert "Supported Python version(s): py3." in str(e.value)

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            version="1.0.0",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
        )

    assert "Unsupported Python version for some-framework 1.0.0: None." in str(e.value)
    assert "Supported Python version(s): py3." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_processor_type(config_for_framework):
    for cpu in ("local", "ml.t2.medium", "ml.m5.xlarge", "ml.r5.large"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=cpu,
            region="us-west-2",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    for gpu in ("local_gpu", "ml.p3.2xlarge", "ml.g4dn.xlarge"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=gpu,
            region="us-west-2",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-gpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_processor_type(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="not-an-instance-type",
            region="us-west-2",
        )

    assert "Invalid SageMaker instance type: not-an-instance-type." in str(e.value)

    config = copy.deepcopy(BASE_CONFIG)
    config["processors"] = ["cpu"]
    config_for_framework.return_value = config

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.p2.xlarge",
            region="us-west-2",
        )

    assert "Unsupported processor type: gpu (for ml.p2.xlarge)." in str(e.value)
    assert "Supported type(s): cpu." in str(e.value)
