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

from packaging.version import Version

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

import pytest

INSTANCE_TYPES_AND_PROCESSORS = (("ml.c4.xlarge", "cpu"), ("ml.p2.xlarge", "gpu"))
RENEWED_PYTORCH_INSTANCE_TYPES_AND_PROCESSORS = (("ml.c4.xlarge", "cpu"), ("ml.g4dn.xlarge", "gpu"))
REGION = "us-west-2"

DLC_ACCOUNT = "763104351884"
DLC_ALTERNATE_REGION_ACCOUNTS = {
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-south-1": "692866216735",
    "il-central-1": "780543022126",
    "me-south-1": "217643126080",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-isob-east-1": "094389454867",
}
SAGEMAKER_ACCOUNT = "520713654638"
SAGEMAKER_ALTERNATE_REGION_ACCOUNTS = {
    "af-south-1": "313743910680",
    "ap-east-1": "057415533634",
    "cn-north-1": "422961961927",
    "cn-northwest-1": "423003514399",
    "eu-south-1": "048378556238",
    "me-south-1": "724002660598",
    "us-gov-west-1": "246785580436",
    "us-iso-east-1": "744548109606",
    "us-isob-east-1": "453391408702",
}
ELASTIC_INFERENCE_REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "eu-west-1",
    "us-east-1",
    "us-east-2",
    "us-west-2",
]


def _test_image_uris(
    framework,
    fw_version,
    py_version,
    scope,
    expected_fn,
    expected_fn_args,
    base_framework_version=None,
):
    base_args = {
        "framework": framework,
        "version": fw_version,
        "py_version": py_version,
        "image_scope": scope,
    }

    TYPES_AND_PROCESSORS = INSTANCE_TYPES_AND_PROCESSORS
    if (framework == "pytorch" and Version(fw_version) >= Version("1.13")) or (
        framework == "tensorflow" and Version(fw_version) >= Version("2.12")
    ):
        """Handle P2 deprecation"""
        TYPES_AND_PROCESSORS = RENEWED_PYTORCH_INSTANCE_TYPES_AND_PROCESSORS

    for instance_type, processor in TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(region=REGION, instance_type=instance_type, **base_args)

        expected = expected_fn(processor=processor, **expected_fn_args)
        assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        if (
            scope == "training"
            and framework == "tensorflow"
            and Version(fw_version) == Version("2.12")
        ):
            if region in ["cn-north-1", "cn-northwest-1", "us-iso-east-1", "us-isob-east-1"]:
                pytest.skip(f"TF 2.12 SM DLC is not available in {region} region")

        uri = image_uris.retrieve(region=region, instance_type="ml.c4.xlarge", **base_args)

        expected = expected_fn(region=region, **expected_fn_args)
        assert expected == uri


def test_chainer(chainer_version, chainer_py_version):
    expected_fn_args = {
        "chainer_version": chainer_version,
        "py_version": chainer_py_version,
    }

    _test_image_uris(
        "chainer",
        chainer_version,
        chainer_py_version,
        "training",
        _expected_chainer_uri,
        expected_fn_args,
    )


def _expected_chainer_uri(chainer_version, py_version, processor="cpu", region=REGION):
    account = SAGEMAKER_ACCOUNT if region == REGION else SAGEMAKER_ALTERNATE_REGION_ACCOUNTS[region]
    return expected_uris.framework_uri(
        repo="sagemaker-chainer",
        fw_version=chainer_version,
        py_version=py_version,
        processor=processor,
        region=region,
        account=account,
    )


def test_tensorflow_training(tensorflow_training_version, tensorflow_training_py_version):
    expected_fn_args = {
        "tf_training_version": tensorflow_training_version,
        "py_version": tensorflow_training_py_version,
    }

    _test_image_uris(
        "tensorflow",
        tensorflow_training_version,
        tensorflow_training_py_version,
        "training",
        _expected_tf_training_uri,
        expected_fn_args,
    )


def _expected_tf_training_uri(tf_training_version, py_version, processor="cpu", region=REGION):
    version = Version(tf_training_version)
    if version < Version("1.11"):
        repo = "sagemaker-tensorflow"
    elif version < Version("1.13"):
        repo = "sagemaker-tensorflow-scriptmode"
    elif version >= Version("1.14"):
        repo = "tensorflow-training"
    else:
        repo = "sagemaker-tensorflow-scriptmode" if py_version == "py2" else "tensorflow-training"

    return expected_uris.framework_uri(
        repo,
        tf_training_version,
        _sagemaker_or_dlc_account(repo, region),
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_tensorflow_inference(tensorflow_inference_version, tensorflow_inference_py_version):
    _test_image_uris(
        "tensorflow",
        tensorflow_inference_version,
        tensorflow_inference_py_version,
        "inference",
        _expected_tf_inference_uri,
        {"tf_inference_version": tensorflow_inference_version},
    )


def test_tensorflow_eia(tensorflow_eia_version):
    base_args = {
        "framework": "tensorflow",
        "version": tensorflow_eia_version,
        "py_version": "py2",
        "instance_type": "ml.c4.xlarge",
        "accelerator_type": "ml.eia1.medium",
        "image_scope": "inference",
    }

    uri = image_uris.retrieve(region=REGION, **base_args)

    expected = _expected_tf_inference_uri(tensorflow_eia_version, eia=True)
    assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        uri = image_uris.retrieve(region=region, **base_args)

        expected = _expected_tf_inference_uri(tensorflow_eia_version, region=region, eia=True)
        assert expected == uri


def _expected_tf_inference_uri(tf_inference_version, processor="cpu", region=REGION, eia=False):
    version = Version(tf_inference_version)
    repo = _expected_tf_inference_repo(version, eia)
    py_version = "py2" if version < Version("1.11") else None

    account = _sagemaker_or_dlc_account(repo, region)
    return expected_uris.framework_uri(
        repo,
        tf_inference_version,
        account,
        py_version,
        processor=processor,
        region=region,
    )


def _expected_tf_inference_repo(version, eia):
    if version < Version("1.11"):
        repo = "sagemaker-tensorflow"
    elif version < Version("1.13") or version == Version("1.13") and eia:
        repo = "sagemaker-tensorflow-serving"
    else:
        repo = "tensorflow-inference"

    if eia:
        repo = "-".join((repo, "eia"))

    return repo


def test_mxnet_training(mxnet_training_version, mxnet_training_py_version):
    expected_fn_args = {
        "mxnet_version": mxnet_training_version,
        "py_version": mxnet_training_py_version,
    }

    _test_image_uris(
        "mxnet",
        mxnet_training_version,
        mxnet_training_py_version,
        "training",
        _expected_mxnet_training_uri,
        expected_fn_args,
    )


def _expected_mxnet_training_uri(mxnet_version, py_version, processor="cpu", region=REGION):
    version = Version(mxnet_version)
    if version < Version("1.4") or mxnet_version == "1.4.0":
        repo = "sagemaker-mxnet"
    elif version >= Version("1.6.0"):
        repo = "mxnet-training"
    else:
        repo = "sagemaker-mxnet" if py_version == "py2" else "mxnet-training"

    return expected_uris.framework_uri(
        repo,
        mxnet_version,
        _sagemaker_or_dlc_account(repo, region),
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_mxnet_inference(mxnet_inference_version, mxnet_inference_py_version):
    expected_fn_args = {
        "mxnet_version": mxnet_inference_version,
        "py_version": mxnet_inference_py_version,
    }

    _test_image_uris(
        "mxnet",
        mxnet_inference_version,
        mxnet_inference_py_version,
        "inference",
        _expected_mxnet_inference_uri,
        expected_fn_args,
    )


def test_mxnet_eia(mxnet_eia_version, mxnet_eia_py_version):
    base_args = {
        "framework": "mxnet",
        "version": mxnet_eia_version,
        "py_version": mxnet_eia_py_version,
        "image_scope": "inference",
        "instance_type": "ml.c4.xlarge",
        "accelerator_type": "ml.eia1.medium",
    }

    uri = image_uris.retrieve(region=REGION, **base_args)

    expected = _expected_mxnet_inference_uri(mxnet_eia_version, mxnet_eia_py_version, eia=True)
    assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        uri = image_uris.retrieve(region=region, **base_args)

        expected = _expected_mxnet_inference_uri(
            mxnet_eia_version, mxnet_eia_py_version, region=region, eia=True
        )
        assert expected == uri


def _expected_mxnet_inference_uri(
    mxnet_version, py_version, processor="cpu", region=REGION, eia=False
):
    version = Version(mxnet_version)
    if version < Version("1.4"):
        repo = "sagemaker-mxnet"
    elif mxnet_version == "1.4.0":
        repo = "sagemaker-mxnet-serving"
    elif version >= Version("1.5"):
        repo = "mxnet-inference"
    else:
        repo = "sagemaker-mxnet-serving" if py_version == "py2" and not eia else "mxnet-inference"

    if eia:
        repo = "-".join((repo, "eia"))

    return expected_uris.framework_uri(
        repo,
        mxnet_version,
        _sagemaker_or_dlc_account(repo, region),
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_pytorch_training(pytorch_training_version, pytorch_training_py_version):
    _test_image_uris(
        "pytorch",
        pytorch_training_version,
        pytorch_training_py_version,
        "training",
        _expected_pytorch_training_uri,
        {
            "pytorch_version": pytorch_training_version,
            "py_version": pytorch_training_py_version,
        },
    )


def _expected_pytorch_training_uri(pytorch_version, py_version, processor="cpu", region=REGION):
    version = Version(pytorch_version)
    if version < Version("1.2"):
        repo = "sagemaker-pytorch"
    else:
        repo = "pytorch-training"

    return expected_uris.framework_uri(
        repo,
        pytorch_version,
        _sagemaker_or_dlc_account(repo, region),
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_pytorch_inference(pytorch_inference_version, pytorch_inference_py_version):
    _test_image_uris(
        "pytorch",
        pytorch_inference_version,
        pytorch_inference_py_version,
        "inference",
        _expected_pytorch_inference_uri,
        {
            "pytorch_version": pytorch_inference_version,
            "py_version": pytorch_inference_py_version,
        },
    )


def _expected_pytorch_inference_uri(pytorch_version, py_version, processor="cpu", region=REGION):
    version = Version(pytorch_version)
    if version < Version("1.2"):
        repo = "sagemaker-pytorch"
    else:
        repo = "pytorch-inference"

    return expected_uris.framework_uri(
        repo,
        pytorch_version,
        _sagemaker_or_dlc_account(repo, region),
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_pytorch_eia(pytorch_eia_version, pytorch_eia_py_version):
    base_args = {
        "framework": "pytorch",
        "version": pytorch_eia_version,
        "py_version": pytorch_eia_py_version,
        "image_scope": "inference",
        "instance_type": "ml.c4.xlarge",
        "accelerator_type": "ml.eia1.medium",
    }

    uri = image_uris.retrieve(region=REGION, **base_args)

    expected = expected_uris.framework_uri(
        "pytorch-inference-eia",
        pytorch_eia_version,
        DLC_ACCOUNT,
        py_version=pytorch_eia_py_version,
        region=REGION,
    )
    assert expected == uri

    for region in ELASTIC_INFERENCE_REGIONS:
        uri = image_uris.retrieve(region=region, **base_args)
        account = DLC_ALTERNATE_REGION_ACCOUNTS.get(region, DLC_ACCOUNT)

        expected = expected_uris.framework_uri(
            "pytorch-inference-eia",
            pytorch_eia_version,
            account,
            py_version=pytorch_eia_py_version,
            region=region,
        )
        assert expected == uri


def _sagemaker_or_dlc_account(repo, region):
    if repo.startswith("sagemaker"):
        return (
            SAGEMAKER_ACCOUNT if region == REGION else SAGEMAKER_ALTERNATE_REGION_ACCOUNTS[region]
        )
    else:
        return DLC_ACCOUNT if region == REGION else DLC_ALTERNATE_REGION_ACCOUNTS[region]
