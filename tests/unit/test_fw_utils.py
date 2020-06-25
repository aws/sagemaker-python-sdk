# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import inspect
import os
import tarfile

import pytest
from mock import Mock, patch

from contextlib import contextmanager
from sagemaker import fw_utils
from sagemaker.utils import name_from_image

DATA_DIR = "data_dir"
BUCKET_NAME = "mybucket"
ROLE = "Sagemaker"
REGION = "us-west-2"
SCRIPT_PATH = "script.py"
TIMESTAMP = "2017-10-10-14-14-15"
ECR_PREFIX_FORMAT = "{}.dkr.ecr.mars-south-3.amazonaws.com"

MOCK_ACCOUNT = "520713654638"
MOCK_FRAMEWORK = "mlfw"
MOCK_REGION = "mars-south-3"
MOCK_ACCELERATOR = "eia1.medium"
MOCK_HKG_REGION = "ap-east-1"
MOCK_BAH_REGION = "me-south-1"


ORIGINAL_FW_VERSIONS = {
    "pytorch": ["0.4", "0.4.0", "1.0", "1.0.0"],
    "mxnet": ["0.12", "0.12.1", "1.0", "1.0.0", "1.1", "1.1.0", "1.2", "1.2.1"],
    "tensorflow": [
        "1.4",
        "1.4.1",
        "1.5",
        "1.5.0",
        "1.6",
        "1.6.0",
        "1.7",
        "1.7.0",
        "1.8",
        "1.8.0",
        "1.9",
        "1.9.0",
        "1.10",
        "1.10.0",
    ],
}


SERVING_FW_VERSIONS = {
    "pytorch": ["1.1", "1.1.0"],
    "mxnet": ["1.3", "1.3.0", "1.4.0"],
    "tensorflow": ["1.11", "1.11.0", "1.12", "1.12.0"],
}


def get_account(framework, framework_version, py_version="py3"):
    if (
        framework_version in ORIGINAL_FW_VERSIONS[framework]
        or framework_version in SERVING_FW_VERSIONS[framework]
        or is_mxnet_1_4_py2(
            framework, framework_version, py_version
        )  # except for MXNet 1.4.1 (1.4) py2 Asimov teams owns both py2 and py3
    ):
        return fw_utils.DEFAULT_ACCOUNT
    return fw_utils.ASIMOV_DEFAULT_ACCOUNT


def get_repo_name(framework, framework_version, is_serving=False, py_version="py3"):
    if (
        framework_version in ORIGINAL_FW_VERSIONS[framework]
        or framework_version in SERVING_FW_VERSIONS[framework]
        or is_mxnet_1_4_py2(framework, framework_version, py_version)
    ):  # except for MXNet 1.4.1 (1.4) py2 Asimov teams owns both py2 and py3
        # TODO: check whether sagemaker-{}-serving images actually exist for ORIGINAL_FW_VERSIONS
        if is_serving:
            ecr_repo = "sagemaker-{}-serving"
        else:
            ecr_repo = "sagemaker-{}"
            if framework == "tensorflow" and framework_version in SERVING_FW_VERSIONS[framework]:
                framework = framework + "-scriptmode"
    elif is_serving:
        ecr_repo = "{}-inference"
    else:
        ecr_repo = "{}-training"
    return ecr_repo.format(framework)


def is_mxnet_1_4_py2(framework, framework_version, py_version):
    return framework == "mxnet" and py_version == "py2" and framework_version in ["1.4", "1.4.1"]


@pytest.fixture(
    scope="module",
    params=["0.4", "0.4.0", "1.0", "1.0.0", "1.1", "1.1.0", "1.2", "1.2.0", "1.3", "1.3.1"],
)
def pytorch_version(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        "0.12",
        "0.12.1",
        "1.0",
        "1.0.0",
        "1.1",
        "1.1.0",
        "1.2",
        "1.2.1",
        "1.3",
        "1.3.0",
        "1.4",
        "1.4.0",
        "1.4.1",
        "1.6",
        "1.6.0",
    ],
)
def mxnet_version(request):
    return request.param


@contextmanager
def cd(path):
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session", boto_session=boto_mock, s3_client=None, s3_resource=None
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session_mock.expand_role = Mock(name="expand_role", return_value=ROLE)
    session_mock.sagemaker_client.describe_training_job = Mock(
        return_value={"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    )
    return session_mock


@patch("sagemaker.fw_utils.get_ecr_image_uri_prefix")
def test_create_image_uri_cpu(ecr_prefix):
    ecr_prefix.return_value = ECR_PREFIX_FORMAT.format("23")
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.c4.large", "1.0rc", "py2", "23"
    )
    assert image_uri == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu-py2"

    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "local", "1.0rc", "py2", "23"
    )
    assert image_uri == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu-py2"

    ecr_prefix.return_value = "246785580436.dkr.ecr.us-gov-west-1.amazonaws.com"
    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", MOCK_FRAMEWORK, "ml.c4.large", "1.0rc", "py2"
    )
    assert (
        image_uri == "246785580436.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-mlfw:1.0rc-cpu-py2"
    )

    ecr_prefix.return_value = "744548109606.dkr.ecr.us-iso-east-1.c2s.ic.gov"
    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", MOCK_FRAMEWORK, "ml.c4.large", "1.0rc", "py2"
    )
    assert image_uri == "744548109606.dkr.ecr.us-iso-east-1.c2s.ic.gov/sagemaker-mlfw:1.0rc-cpu-py2"


@patch("sagemaker.fw_utils.get_ecr_image_uri_prefix", return_value=ECR_PREFIX_FORMAT.format("23"))
def test_create_image_uri_no_python(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.c4.large", "1.0rc", account="23"
    )
    assert image_uri == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu"


def test_create_image_uri_bad_python():
    with pytest.raises(ValueError):
        fw_utils.create_image_uri(MOCK_REGION, MOCK_FRAMEWORK, "ml.c4.large", "1.0rc", "py0")


@patch("sagemaker.fw_utils.get_ecr_image_uri_prefix", return_value=ECR_PREFIX_FORMAT.format("23"))
def test_create_image_uri_gpu(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3", "23"
    )
    assert image_uri == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"

    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "local_gpu", "1.0rc", "py3", "23"
    )
    assert image_uri == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"


@patch("sagemaker.fw_utils.get_ecr_image_uri_prefix", return_value=ECR_PREFIX_FORMAT.format("23"))
def test_create_image_uri_accelerator_tfs(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION,
        "tensorflow-serving",
        "ml.c4.large",
        "1.1.0",
        accelerator_type="ml.eia1.large",
        account="23",
    )
    assert (
        image_uri
        == "23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-tensorflow-serving-eia:1.1.0-cpu"
    )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_create_image_uri_default_account(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert (
        image_uri == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    )


def test_create_image_uri_gov_cloud():
    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert (
        image_uri == "246785580436.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    )


def test_create_image_uri_hkg():
    image_uri = fw_utils.create_image_uri(
        MOCK_HKG_REGION, MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert {
        image_uri == "871362719292.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    }


def test_create_image_uri_bah():
    image_uri = fw_utils.create_image_uri(
        MOCK_BAH_REGION, MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert {
        image_uri == "217643126080.dkr.ecr.me-south-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    }


def test_create_image_uri_cn_north_1():
    image_uri = fw_utils.create_image_uri(
        "cn-north-1", MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert {
        image_uri == "727897471807.dkr.ecr.me-south-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    }


def test_create_image_uri_cn_northwest_1():
    image_uri = fw_utils.create_image_uri(
        "cn-northwest-1", MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3"
    )
    assert {
        image_uri == "727897471807.dkr.ecr.me-south-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"
    }


def test_create_image_uri_py37_invalid_framework():
    error_message = "{} does not support Python 3.7 at this time.".format(MOCK_FRAMEWORK)

    with pytest.raises(ValueError) as error:
        fw_utils.create_image_uri(REGION, MOCK_FRAMEWORK, "ml.m4.xlarge", "1.4.0", "py37")
    assert error_message in str(error)


def test_create_image_uri_py37():
    image_uri = fw_utils.create_image_uri(
        REGION, "tensorflow-scriptmode", "ml.m4.xlarge", "1.15.2", "py37"
    )
    assert (
        image_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-cpu-py37"
    )


def test_tf_eia_images():
    image_uri = fw_utils.create_image_uri(
        "us-west-2",
        "tensorflow-serving",
        "ml.m4.xlarge",
        "2.0.0",
        "py3",
        accelerator_type="ml.eia1.medium",
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-eia:2.0.0-cpu".format(
            fw_utils.ASIMOV_PROD_ACCOUNT
        )
    )


def test_mxnet_eia_images():
    image_uri = fw_utils.create_image_uri(
        "us-east-1",
        "mxnet-serving",
        "ml.c4.2xlarge",
        "1.5.1",
        "py3",
        accelerator_type="ml.eia1.large",
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference-eia:1.5.1-cpu-py3".format(
            fw_utils.ASIMOV_PROD_ACCOUNT
        )
    )


def test_pytorch_eia_images():
    image_uri = fw_utils.create_image_uri(
        "us-east-1",
        "pytorch-serving",
        "ml.c4.2xlarge",
        "1.3.1",
        "py3",
        accelerator_type="ml.eia1.large",
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-eia:1.3.1-cpu-py3".format(
            fw_utils.ASIMOV_PROD_ACCOUNT
        )
    )


def test_pytorch_eia_py2_error():
    error_message = "pytorch-serving is not supported with Amazon Elastic Inference in Python 2."
    with pytest.raises(ValueError) as error:
        fw_utils.create_image_uri(
            "us-east-1",
            "pytorch-serving",
            "ml.c4.2xlarge",
            "1.3.1",
            "py2",
            accelerator_type="ml.eia1.large",
        )
    assert error_message in str(error)


def test_create_image_uri_override_account():
    image_uri = fw_utils.create_image_uri(
        "us-west-1", MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3", account="fake"
    )
    assert image_uri == "fake.dkr.ecr.us-west-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"


def test_create_image_uri_gov_cloud_override_account():
    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3", account="fake"
    )
    assert image_uri == "fake.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"


def test_create_image_uri_hkg_override_account():
    image_uri = fw_utils.create_image_uri(
        MOCK_HKG_REGION, MOCK_FRAMEWORK, "ml.p3.2xlarge", "1.0rc", "py3", account="fake"
    )
    assert {image_uri == "fake.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3"}


def test_create_dlc_image_uri():
    image_uri = fw_utils.create_image_uri(
        "us-west-2", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.14", "py3"
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.14-gpu-py3".format(
            fw_utils.ASIMOV_DEFAULT_ACCOUNT
        )
    )

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.13.1", "py3"
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13.1-gpu-py3".format(
            fw_utils.ASIMOV_DEFAULT_ACCOUNT
        )
    )

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "tensorflow-serving", "ml.c4.2xlarge", "1.13.1"
    )
    assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:1.13.1-cpu".format(
        fw_utils.ASIMOV_DEFAULT_ACCOUNT
    )

    image_uri = fw_utils.create_image_uri("us-west-2", "mxnet", "ml.p3.2xlarge", "1.4.1", "py3")
    assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:1.4.1-gpu-py3".format(
        fw_utils.ASIMOV_DEFAULT_ACCOUNT
    )

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "mxnet-serving", "ml.c4.2xlarge", "1.4.1", "py3"
    )
    assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.4.1-cpu-py3".format(
        fw_utils.ASIMOV_DEFAULT_ACCOUNT
    )

    image_uri = fw_utils.create_image_uri(
        "us-west-2",
        "mxnet-serving",
        "ml.c4.2xlarge",
        "1.4.1",
        "py3",
        accelerator_type="ml.eia1.medium",
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference-eia:1.4.1-cpu-py3".format(
            fw_utils.ASIMOV_PROD_ACCOUNT
        )
    )


def test_create_dlc_image_uri_py2():
    image_uri = fw_utils.create_image_uri(
        "us-west-2", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.13.1", "py2"
    )
    assert (
        image_uri
        == "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.13.1-gpu-py2"
    )

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.14", "py2"
    )
    assert (
        image_uri
        == "{}.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.14-gpu-py2".format(
            fw_utils.ASIMOV_DEFAULT_ACCOUNT
        )
    )

    image_uri = fw_utils.create_image_uri("us-west-2", "mxnet", "ml.p3.2xlarge", "1.4.1", "py2")
    assert image_uri == "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.4.1-gpu-py2"

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "mxnet-serving", "ml.c4.2xlarge", "1.3.1", "py2"
    )
    assert (
        image_uri
        == "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-serving:1.3.1-cpu-py2"
    )


def test_create_dlc_image_uri_iso_east_1():
    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", "tensorflow-scriptmode", "ml.m4.xlarge", "1.13.1", "py3"
    )
    assert (
        image_uri
        == "886529160074.dkr.ecr.us-iso-east-1.c2s.ic.gov/tensorflow-training:1.13.1-cpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.14", "py2"
    )
    assert (
        image_uri
        == "886529160074.dkr.ecr.us-iso-east-1.c2s.ic.gov/tensorflow-training:1.14-gpu-py2"
    )

    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", "tensorflow-serving", "ml.m4.xlarge", "1.13.0"
    )
    assert (
        image_uri == "886529160074.dkr.ecr.us-iso-east-1.c2s.ic.gov/tensorflow-inference:1.13.0-cpu"
    )

    image_uri = fw_utils.create_image_uri("us-iso-east-1", "mxnet", "ml.p3.2xlarge", "1.4.1", "py3")
    assert image_uri == "886529160074.dkr.ecr.us-iso-east-1.c2s.ic.gov/mxnet-training:1.4.1-gpu-py3"

    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", "mxnet-serving", "ml.c4.2xlarge", "1.4.1", "py3"
    )
    assert (
        image_uri == "886529160074.dkr.ecr.us-iso-east-1.c2s.ic.gov/mxnet-inference:1.4.1-cpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-iso-east-1", "mxnet-serving", "ml.c4.2xlarge", "1.3.1", "py3"
    )
    assert (
        image_uri
        == "744548109606.dkr.ecr.us-iso-east-1.c2s.ic.gov/sagemaker-mxnet-serving:1.3.1-cpu-py3"
    )


def test_create_dlc_image_uri_gov_west_1():
    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "tensorflow-scriptmode", "ml.m4.xlarge", "1.13.1", "py3"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-training:1.13.1-cpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "tensorflow-scriptmode", "ml.p3.2xlarge", "1.14", "py2"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-training:1.14-gpu-py2"
    )

    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "tensorflow-serving", "ml.m4.xlarge", "1.13.0"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-inference:1.13.0-cpu"
    )

    image_uri = fw_utils.create_image_uri("us-gov-west-1", "mxnet", "ml.p3.2xlarge", "1.4.1", "py3")
    assert (
        image_uri == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-training:1.4.1-gpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "mxnet-serving", "ml.c4.2xlarge", "1.4.1", "py3"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-inference:1.4.1-cpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "pytorch", "ml.p3.2xlarge", "1.2.0", "py3"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.2.0-gpu-py3"
    )

    image_uri = fw_utils.create_image_uri(
        "us-gov-west-1", "pytorch-serving", "ml.c4.2xlarge", "1.2.0", "py3"
    )
    assert (
        image_uri
        == "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.2.0-cpu-py3"
    )


def test_create_image_uri_pytorch(pytorch_version):
    image_uri = fw_utils.create_image_uri(
        "us-west-2", "pytorch", "ml.p3.2xlarge", pytorch_version, "py3"
    )
    assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-gpu-py3".format(
        get_account("pytorch", pytorch_version),
        get_repo_name("pytorch", pytorch_version),
        pytorch_version,
    )

    if pytorch_version not in ORIGINAL_FW_VERSIONS:
        image_uri = fw_utils.create_image_uri(
            "us-west-2", "pytorch-serving", "ml.c4.2xlarge", pytorch_version, "py2"
        )
        assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-cpu-py2".format(
            get_account("pytorch", pytorch_version),
            get_repo_name("pytorch", pytorch_version, True),
            pytorch_version,
        )


def test_create_image_uri_mxnet(mxnet_version):

    image_uri = fw_utils.create_image_uri(
        "us-west-2", "mxnet", "ml.p3.2xlarge", mxnet_version, "py3"
    )
    assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-gpu-py3".format(
        get_account("mxnet", mxnet_version), get_repo_name("mxnet", mxnet_version), mxnet_version
    )

    if mxnet_version not in ORIGINAL_FW_VERSIONS:
        py_version = "py2"
        image_uri = fw_utils.create_image_uri(
            "us-west-2", "mxnet-serving", "ml.c4.2xlarge", mxnet_version, py_version
        )
        assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-cpu-{}".format(
            get_account("mxnet", mxnet_version, py_version),
            get_repo_name("mxnet", mxnet_version, True, py_version),
            mxnet_version,
            py_version,
        )


def test_create_image_uri_tensorflow(tf_version):
    if tf_version in ORIGINAL_FW_VERSIONS["tensorflow"]:
        image_uri = fw_utils.create_image_uri(
            "us-west-2", "tensorflow", "ml.p3.2xlarge", tf_version, "py2"
        )
        assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-gpu-py2".format(
            get_account("tensorflow", tf_version),
            get_repo_name("tensorflow", tf_version),
            tf_version,
        )
    else:
        image_uri = fw_utils.create_image_uri(
            "us-west-2", "tensorflow-scriptmode", "ml.p3.2xlarge", tf_version, "py3"
        )
        assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-gpu-py3".format(
            get_account("tensorflow", tf_version),
            get_repo_name("tensorflow", tf_version),
            tf_version,
        )

        image_uri = fw_utils.create_image_uri(
            "us-west-2", "tensorflow-serving", "ml.c4.2xlarge", tf_version
        )
        assert image_uri == "{}.dkr.ecr.us-west-2.amazonaws.com/{}:{}-cpu".format(
            get_account("tensorflow", tf_version),
            get_repo_name("tensorflow", tf_version, True),
            tf_version,
        )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_create_image_uri_accelerator_tf(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, "tensorflow", "ml.p3.2xlarge", "1.0", "py3", accelerator_type="ml.eia1.medium"
    )
    assert (
        image_uri
        == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-tensorflow-eia:1.0-gpu-py3"
    )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_create_image_uri_accelerator_mxnet_serving(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION,
        "mxnet-serving",
        "ml.p3.2xlarge",
        "1.0",
        "py3",
        accelerator_type="ml.eia1.medium",
    )
    assert (
        image_uri
        == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mxnet-serving-eia:1.0-gpu-py3"
    )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_create_image_uri_local_sagemaker_notebook_accelerator(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION,
        "mxnet",
        "ml.p3.2xlarge",
        "1.0",
        "py3",
        accelerator_type="local_sagemaker_notebook",
    )
    assert (
        image_uri
        == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mxnet-eia:1.0-gpu-py3"
    )


def test_invalid_accelerator():
    error_message = "{} is not a valid SageMaker Elastic Inference accelerator type.".format(
        MOCK_ACCELERATOR
    )
    # accelerator type is missing 'ml.' prefix
    with pytest.raises(ValueError) as error:
        fw_utils.create_image_uri(
            MOCK_REGION,
            "tensorflow",
            "ml.p3.2xlarge",
            "1.0.0",
            "py3",
            accelerator_type=MOCK_ACCELERATOR,
        )

    assert error_message in str(error)


def test_invalid_framework_accelerator():
    error_message = "{} is not supported with Amazon Elastic Inference.".format(MOCK_FRAMEWORK)
    # accelerator was chosen for unsupported framework
    with pytest.raises(ValueError) as error:
        fw_utils.create_image_uri(
            MOCK_REGION,
            MOCK_FRAMEWORK,
            "ml.p3.2xlarge",
            "1.0.0",
            "py3",
            accelerator_type="ml.eia1.medium",
        )

    assert error_message in str(error)


def test_invalid_framework_accelerator_with_neo():
    error_message = "Neo does not support Amazon Elastic Inference."
    # accelerator was chosen for unsupported framework
    with pytest.raises(ValueError) as error:
        fw_utils.create_image_uri(
            MOCK_REGION,
            "tensorflow",
            "ml.p3.2xlarge",
            "1.0.0",
            "py3",
            accelerator_type="ml.eia1.medium",
            optimized_families=["c5", "p3"],
        )

    assert error_message in str(error)


def test_invalid_instance_type():
    # instance type is missing 'ml.' prefix
    with pytest.raises(ValueError):
        fw_utils.create_image_uri(MOCK_REGION, MOCK_FRAMEWORK, "p3.2xlarge", "1.0.0", "py3")


def test_valid_inferentia_image():
    image_uri = fw_utils.create_image_uri(
        REGION,
        "neo-tensorflow",
        "ml.inf1.2xlarge",
        "1.15.0",
        py_version="py3",
        account=MOCK_ACCOUNT,
    )
    assert (
        image_uri
        == "{}.dkr.ecr.{}.amazonaws.com/sagemaker-neo-tensorflow:1.15.0-inf-py3".format(
            MOCK_ACCOUNT, REGION
        )
    )


def test_invalid_inferentia_region():
    with pytest.raises(ValueError) as e:
        fw_utils.create_image_uri(
            "ap-south-1",
            "neo-tensorflow",
            "ml.inf1.2xlarge",
            "1.15.0",
            py_version="py3",
            account=MOCK_ACCOUNT,
        )
    assert "Inferentia is not supported in region ap-south-1." in str(e)


def test_inferentia_invalid_framework():
    with pytest.raises(ValueError) as e:
        fw_utils.create_image_uri(
            REGION,
            "neo-pytorch",
            "ml.inf1.2xlarge",
            "1.4.0",
            py_version="py3",
            account=MOCK_ACCOUNT,
        )
    assert "Inferentia does not support pytorch." in str(e)


def test_invalid_inferentia_framework_version():
    with pytest.raises(ValueError) as e:
        fw_utils.create_image_uri(
            REGION,
            "neo-tensorflow",
            "ml.inf1.2xlarge",
            "1.15.2",
            py_version="py3",
            account=MOCK_ACCOUNT,
        )
    assert "Inferentia is not supported with tensorflow version 1.15.2." in str(e)


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_optimized_family(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION,
        MOCK_FRAMEWORK,
        "ml.p3.2xlarge",
        "1.0.0",
        "py3",
        optimized_families=["c5", "p3"],
    )
    assert (
        image_uri == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-p3-py3"
    )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_unoptimized_cpu_family(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.m4.xlarge", "1.0.0", "py3", optimized_families=["c5", "p3"]
    )
    assert (
        image_uri == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-cpu-py3"
    )


@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value=ECR_PREFIX_FORMAT.format(MOCK_ACCOUNT),
)
def test_unoptimized_gpu_family(ecr_prefix):
    image_uri = fw_utils.create_image_uri(
        MOCK_REGION, MOCK_FRAMEWORK, "ml.p2.xlarge", "1.0.0", "py3", optimized_families=["c5", "p3"]
    )
    assert (
        image_uri == "520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-gpu-py3"
    )


def test_tar_and_upload_dir_s3(sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = "mnist.py"
    directory = "s3://m"
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, directory
    )

    assert result == fw_utils.UploadedCode("s3://m", "mnist.py")


@patch("sagemaker.utils")
def test_tar_and_upload_dir_s3_with_kms(utils, sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = "mnist.py"
    kms_key = "kms-key"
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, kms_key=kms_key
    )

    assert result == fw_utils.UploadedCode(
        "s3://{}/{}/sourcedir.tar.gz".format(bucket, s3_key_prefix), script
    )

    extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
    obj = sagemaker_session.resource("s3").Object("", "")
    obj.upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=extra_args)


def test_validate_source_dir_does_not_exits(sagemaker_session):
    script = "mnist.py"
    directory = " !@#$%^&*()path probably in not there.!@#$%^&*()"
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_validate_source_dir_is_not_directory(sagemaker_session):
    script = "mnist.py"
    directory = inspect.getfile(inspect.currentframe())
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_validate_source_dir_file_not_in_dir():
    script = " !@#$%^&*() .myscript. !@#$%^&*() "
    directory = "."
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_tar_and_upload_dir_not_s3(sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = os.path.basename(__file__)
    directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, directory
    )
    assert result == fw_utils.UploadedCode(
        "s3://{}/{}/sourcedir.tar.gz".format(bucket, s3_key_prefix), script
    )


def file_tree(tmpdir, files=None, folders=None):
    files = files or []
    folders = folders or []
    for file in files:
        tmpdir.join(file).ensure(file=True)

    for folder in folders:
        tmpdir.join(folder).ensure(dir=True)

    return str(tmpdir)


def test_tar_and_upload_dir_no_directory(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py"])
    entrypoint = os.path.join(source_dir, "train.py")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", entrypoint, None
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_no_directory_only_entrypoint(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py", "not_me.py"])
    entrypoint = os.path.join(source_dir, "train.py")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", entrypoint, None
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_no_directory_bare_filename(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py"])
    entrypoint = "train.py"

    with patch("shutil.rmtree"):
        with cd(source_dir):
            result = fw_utils.tar_and_upload_dir(
                sagemaker_session, "bucket", "prefix", entrypoint, None
            )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_directory(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/train.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_subdirectory(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/sub/train.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/sub/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_directory_and_files(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/train.py", "src-dir/laucher", "src-dir/module/__init__.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/laucher", "/module/__init__.py", "/train.py"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_tar_and_upload_dir_with_directories_and_files(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/a/b", "src-dir/a/b2", "src-dir/x/y", "src-dir/x/y2", "src-dir/z"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "a/b", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="a/b"
    )

    assert {"/a/b", "/a/b2", "/x/y", "/x/y2", "/z"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_tar_and_upload_dir_with_many_folders(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/a/b", "src-dir/a/b2", "common/x/y", "common/x/y2", "t/y/z"])
    source_dir = os.path.join(str(tmpdir), "src-dir")
    dependencies = [os.path.join(str(tmpdir), "common"), os.path.join(str(tmpdir), "t", "y", "z")]

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "pipeline.py", source_dir, dependencies
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="pipeline.py"
    )

    assert {"/a/b", "/a/b2", "/common/x/y", "/common/x/y2", "/z"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_test_tar_and_upload_dir_with_subfolders(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["a/b/c", "a/b/c2"])
    root = file_tree(tmpdir, ["x/y/z", "x/y/z2"])

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session,
            "bucket",
            "prefix",
            "b/c",
            os.path.join(root, "a"),
            [os.path.join(root, "x")],
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="b/c"
    )

    assert {"/b/c", "/b/c2", "/x/y/z", "/x/y/z2"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def list_source_dir_files(sagemaker_session, tmpdir):
    source_dir_tar = sagemaker_session.resource("s3").Object().upload_file.call_args[0][0]

    source_dir_files = list_tar_files("/opt/ml/code/", source_dir_tar, tmpdir)
    return source_dir_files


def list_tar_files(folder, tar_ball, tmpdir):
    startpath = str(tmpdir.ensure(folder, dir=True))

    with tarfile.open(name=tar_ball, mode="r:gz") as t:
        t.extractall(path=startpath)

    def walk():
        for root, dirs, files in os.walk(startpath):
            path = root.replace(startpath, "")
            for f in files:
                yield "%s/%s" % (path, f)

    result = set(walk())
    return result if result else {}


def test_framework_name_from_image_mxnet():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.1-gpu-py3"
    assert ("mxnet", "py3", "1.1-gpu-py3", None) == fw_utils.framework_name_from_image(image_name)


def test_framework_name_from_image_mxnet_in_gov():
    image_name = "123.dkr.ecr.region-name.c2s.ic.gov/sagemaker-mxnet:1.1-gpu-py3"
    assert ("mxnet", "py3", "1.1-gpu-py3", None) == fw_utils.framework_name_from_image(image_name)


def test_framework_name_from_image_tf():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.6-cpu-py2"
    assert ("tensorflow", "py2", "1.6-cpu-py2", None) == fw_utils.framework_name_from_image(
        image_name
    )


def test_framework_name_from_image_tf_scriptmode():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12-cpu-py3"
    assert (
        "tensorflow",
        "py3",
        "1.12-cpu-py3",
        "scriptmode",
    ) == fw_utils.framework_name_from_image(image_name)

    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-cpu-py3"
    assert ("tensorflow", "py3", "1.13-cpu-py3", "training") == fw_utils.framework_name_from_image(
        image_name
    )


def test_framework_name_from_image_rl():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-mxnet:toolkit1.1-gpu-py3"
    assert ("mxnet", "py3", "toolkit1.1-gpu-py3", None) == fw_utils.framework_name_from_image(
        image_name
    )


def test_legacy_name_from_framework_image():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2"
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(image_name)
    assert framework == "mxnet"
    assert py_ver == "py3"
    assert tag == "2.5.6-gpu-py2"


def test_legacy_name_from_wrong_framework():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py2-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_python():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_device():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_image_any_tag():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:any-tag"
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(image_name)
    assert framework == "tensorflow"
    assert py_ver == "py2"
    assert tag == "any-tag"


def test_framework_version_from_tag():
    version = fw_utils.framework_version_from_tag("1.5rc-keras-gpu-py2")
    assert version == "1.5rc-keras"


def test_framework_version_from_tag_other():
    version = fw_utils.framework_version_from_tag("weird-tag-py2")
    assert version is None


def test_parse_s3_url():
    bucket, key_prefix = fw_utils.parse_s3_url("s3://bucket/code_location")
    assert "bucket" == bucket
    assert "code_location" == key_prefix


def test_parse_s3_url_fail():
    with pytest.raises(ValueError) as error:
        fw_utils.parse_s3_url("t3://code_location")
    assert "Expecting 's3' scheme" in str(error)


def test_model_code_key_prefix_with_all_values_present():
    key_prefix = fw_utils.model_code_key_prefix("prefix", "model_name", "image_name")
    assert key_prefix == "prefix/model_name"


def test_model_code_key_prefix_with_no_prefix_and_all_other_values_present():
    key_prefix = fw_utils.model_code_key_prefix(None, "model_name", "image_name")
    assert key_prefix == "model_name"


@patch("time.strftime", return_value=TIMESTAMP)
def test_model_code_key_prefix_with_only_image_present(time):
    key_prefix = fw_utils.model_code_key_prefix(None, None, "image_name")
    assert key_prefix == name_from_image("image_name")


@patch("time.strftime", return_value=TIMESTAMP)
def test_model_code_key_prefix_and_image_present(time):
    key_prefix = fw_utils.model_code_key_prefix("prefix", None, "image_name")
    assert key_prefix == "prefix/" + name_from_image("image_name")


def test_model_code_key_prefix_with_prefix_present_and_others_none_fail():
    with pytest.raises(TypeError) as error:
        fw_utils.model_code_key_prefix("prefix", None, None)
    assert "expected string" in str(error)


def test_model_code_key_prefix_with_all_none_fail():
    with pytest.raises(TypeError) as error:
        fw_utils.model_code_key_prefix(None, None, None)
    assert "expected string" in str(error)


def test_region_supports_debugger_feature_returns_true_for_supported_regions():
    assert fw_utils._region_supports_debugger("us-west-2") is True
    assert fw_utils._region_supports_debugger("us-east-2") is True


def test_region_supports_debugger_feature_returns_false_for_unsupported_regions():
    assert fw_utils._region_supports_debugger("us-gov-west-1") is False
    assert fw_utils._region_supports_debugger("us-iso-east-1") is False


def test_warn_if_parameter_server_with_multi_gpu(caplog):
    train_instance_type = "ml.p2.8xlarge"
    distributions = {"parameter_server": {"enabled": True}}

    fw_utils.warn_if_parameter_server_with_multi_gpu(
        training_instance_type=train_instance_type, distributions=distributions
    )
    assert fw_utils.PARAMETER_SERVER_MULTI_GPU_WARNING in caplog.text


def test_warn_if_parameter_server_with_local_multi_gpu(caplog):
    train_instance_type = "local_gpu"
    distributions = {"parameter_server": {"enabled": True}}

    fw_utils.warn_if_parameter_server_with_multi_gpu(
        training_instance_type=train_instance_type, distributions=distributions
    )
    assert fw_utils.PARAMETER_SERVER_MULTI_GPU_WARNING in caplog.text


def test_validate_version_or_image_args_not_raises():
    good_args = [("1.0", "py3", None), (None, "py3", "my:uri"), ("1.0", None, "my:uri")]
    for framework_version, py_version, image_name in good_args:
        fw_utils.validate_version_or_image_args(framework_version, py_version, image_name)


def test_validate_version_or_image_args_raises():
    bad_args = [(None, None, None), (None, "py3", None), ("1.0", None, None)]
    for framework_version, py_version, image_name in bad_args:
        with pytest.raises(ValueError):
            fw_utils.validate_version_or_image_args(framework_version, py_version, image_name)
