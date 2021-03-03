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
import logging

import pytest
from mock import patch

from sagemaker import image_uris

BASE_CONFIG = {
    "processors": ["cpu", "gpu"],
    "scope": ["training", "inference"],
    "versions": {
        "1.0.0": {
            "registries": {"us-west-2": "123412341234"},
            "repository": "dummy",
            "py_versions": ["py3", "py37"],
        },
        "1.1.0": {
            "registries": {"us-west-2": "123412341234"},
            "repository": "dummy",
            "py_versions": ["py3", "py37"],
        },
    },
}


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_framework(config_for_framework):
    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_image_scope(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
            image_scope="invalid-image-scope",
        )
    assert "Unsupported image scope: invalid-image-scope." in str(e.value)
    assert "Supported image scope(s): training, inference." in str(e.value)

    config = copy.deepcopy(BASE_CONFIG)
    config["scope"].append("eia")
    config_for_framework.return_value = config

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
        )
    assert "Unsupported image scope: None." in str(e.value)
    assert "Supported image scope(s): training, inference, eia." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_default_image_scope(config_for_framework, caplog):
    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    config = copy.deepcopy(BASE_CONFIG)
    config["scope"] = ["eia"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="ignorable-scope",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri
    assert "Ignoring image scope: ignorable-scope." in caplog.text


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_eia(config_for_framework, caplog):
    base_config = copy.deepcopy(BASE_CONFIG)
    del base_config["scope"]

    config = {
        "training": base_config,
        "eia": {
            "processors": ["cpu"],
            "versions": {
                "1.0.0": {
                    "registries": {"us-west-2": "123412341234"},
                    "repository": "dummy-eia",
                    "py_versions": ["py3", "py37"],
                },
            },
        },
    }
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        accelerator_type="ml.eia1.medium",
        region="us-west-2",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy-eia:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        accelerator_type="local_sagemaker_notebook",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy-eia:1.0.0-cpu-py3" == uri
    assert "Ignoring image scope: training." in caplog.text


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_invalid_accelerator(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            accelerator_type="fake-accelerator",
            region="us-west-2",
        )
    assert "Invalid SageMaker Elastic Inference accelerator type: fake-accelerator." in str(e.value)


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
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:{}-cpu-py3".format(version) == uri

    del config["versions"]["1.1.0"]
    uri = image_uris.retrieve(
        framework="useless-string",
        version=version,
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:{}-cpu-py3".format(version) == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_default_version_if_possible(config_for_framework, caplog):
    config = copy.deepcopy(BASE_CONFIG)
    del config["versions"]["1.1.0"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="invalid-version",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri
    assert "Ignoring framework/algorithm version: invalid-version." in caplog.text


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_version(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            version="1",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
            image_scope="training",
        )

    assert "Unsupported some-framework version: 1." in str(e.value)
    assert "Supported some-framework version(s): 1.0.0, 1.1.0." in str(e.value)

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="some-framework",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
            image_scope="training",
        )

    assert "Unsupported some-framework version: None." in str(e.value)
    assert "Supported some-framework version(s): 1.0.0, 1.1.0." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_region(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="ml.c4.xlarge",
            region="us-east-2",
            image_scope="training",
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
        image_scope="training",
    )
    assert "000000010010.dkr.ecr.cn-north-1.amazonaws.com.cn/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="cn-northwest-1",
        image_scope="training",
    )
    assert "010000001000.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-iso-east-1",
        image_scope="training",
    )
    assert "000111111000.dkr.ecr.us-iso-east-1.c2s.ic.gov/dummy:1.0.0-cpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_no_python_version(config_for_framework, caplog):
    config = copy.deepcopy(BASE_CONFIG)
    config["versions"]["1.0.0"]["py_versions"] = []
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu" == uri

    caplog.set_level(logging.INFO)
    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu" == uri
    assert "Ignoring unnecessary Python version: py3." in caplog.text


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_different_config_per_python_version(config_for_framework, caplog):
    config = {
        "processors": ["cpu", "gpu"],
        "scope": ["training", "inference"],
        "versions": {
            "1.0.0": {
                "py3": {"registries": {"us-west-2": "123412341234"}, "repository": "foo"},
                "py37": {"registries": {"us-west-2": "012345678901"}, "repository": "bar"},
            },
        },
    }
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/foo:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py37",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "012345678901.dkr.ecr.us-west-2.amazonaws.com/bar:1.0.0-cpu-py37" == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_default_python_version_if_possible(config_for_framework):
    config = copy.deepcopy(BASE_CONFIG)
    config["versions"]["1.0.0"]["py_versions"] = ["py3"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_python_version(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py2",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
            image_scope="training",
        )

    assert "Unsupported Python version: py2." in str(e.value)
    assert "Supported Python version(s): py3, py37." in str(e.value)

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            instance_type="ml.c4.xlarge",
            region="us-west-2",
            image_scope="training",
        )

    assert "Unsupported Python version: None." in str(e.value)
    assert "Supported Python version(s): py3, py37." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_processor_type(config_for_framework):
    for cpu in ("local", "ml.t2.medium", "ml.m5.xlarge", "ml.r5.large"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=cpu,
            region="us-west-2",
            image_scope="training",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    for gpu in ("local_gpu", "ml.p3.2xlarge", "ml.g4dn.xlarge"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=gpu,
            region="us-west-2",
            image_scope="training",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-gpu-py3" == uri


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_processor_type_neo(config_for_framework):
    for cpu in ("ml_m4", "ml_m5", "ml_c4", "ml_c5"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=cpu,
            region="us-west-2",
            image_scope="training",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    for gpu in ("ml_p2", "ml_p3"):
        uri = image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type=gpu,
            region="us-west-2",
            image_scope="training",
        )
        assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-gpu-py3" == uri

    config = copy.deepcopy(BASE_CONFIG)
    config["processors"] = ["inf"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml_inf1",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-inf-py3" == uri

    config = copy.deepcopy(BASE_CONFIG)
    config["processors"] = ["c5"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml_c5",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-c5-py3" == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_processor_type_from_version_specific_processor_config(config_for_framework):
    config = copy.deepcopy(BASE_CONFIG)
    del config["processors"]
    config["versions"]["1.0.0"]["processors"] = ["cpu"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.1.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.1.0-py3" == uri


@patch("sagemaker.image_uris.config_for_framework")
def test_retrieve_default_processor_type_if_possible(config_for_framework):
    config = copy.deepcopy(BASE_CONFIG)
    config["processors"] = ["cpu"]
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:1.0.0-cpu-py3" == uri


def test_retrieve_auto_selected_container_version():
    uri = image_uris.retrieve(
        framework="tensorflow",
        region="us-west-2",
        version="2.3",
        py_version="py37",
        instance_type="ml.p4d.24xlarge",
        image_scope="training",
    )
    assert (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3-gpu-py37-cu110-ubuntu18.04-v3"
        == uri
    )


def test_retrieve_pytorch_container_version():
    uri = image_uris.retrieve(
        framework="pytorch",
        region="us-west-2",
        version="1.6",
        py_version="py3",
        instance_type="ml.p4d.24xlarge",
        image_scope="training",
    )
    assert (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.6-gpu-py3-cu110-ubuntu18.04-v3"
        == uri
    )


@patch("sagemaker.image_uris.config_for_framework", return_value=BASE_CONFIG)
def test_retrieve_unsupported_processor_type(config_for_framework):
    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            instance_type="not-an-instance-type",
            region="us-west-2",
            image_scope="training",
        )

    assert "Invalid SageMaker instance type: not-an-instance-type." in str(e.value)

    with pytest.raises(ValueError) as e:
        image_uris.retrieve(
            framework="useless-string",
            version="1.0.0",
            py_version="py3",
            region="us-west-2",
            image_scope="training",
        )

    assert "Empty SageMaker instance type." in str(e.value)

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
            image_scope="training",
        )

    assert "Unsupported processor: gpu." in str(e.value)
    assert "Supported processor(s): cpu." in str(e.value)


@patch("sagemaker.image_uris.config_for_framework")
def test_tag_prefix(config_for_framework):
    tag_prefix = "1.0.0-build123"

    config = copy.deepcopy(BASE_CONFIG)
    config["versions"]["1.0.0"]["tag_prefix"] = "1.0.0-build123"
    config_for_framework.return_value = config

    uri = image_uris.retrieve(
        framework="useless-string",
        version="1.0.0",
        py_version="py3",
        instance_type="ml.c4.xlarge",
        region="us-west-2",
        image_scope="training",
    )
    assert "123412341234.dkr.ecr.us-west-2.amazonaws.com/dummy:{}-cpu-py3".format(tag_prefix) == uri
