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

from packaging.version import Version

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

INSTANCE_TYPES_AND_PROCESSORS = (("ml.c4.xlarge", "cpu"), ("ml.p2.xlarge", "gpu"))
REGION = "us-west-2"

DLC_ACCOUNT = "763104351884"
DLC_ALTERNATE_REGION_ACCOUNTS = {
    "ap-east-1": "871362719292",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "me-south-1": "217643126080",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
}
SAGEMAKER_ACCOUNT = "520713654638"
SAGEMAKER_ALTERNATE_REGION_ACCOUNTS = {
    "ap-east-1": "057415533634",
    "cn-north-1": "422961961927",
    "cn-northwest-1": "423003514399",
    "me-south-1": "724002660598",
    "us-gov-west-1": "246785580436",
    "us-iso-east-1": "744548109606",
}


def test_chainer(chainer_version, chainer_py_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        for scope in ("training", "inference"):
            uri = image_uris.retrieve(
                framework="chainer",
                region=REGION,
                version=chainer_version,
                py_version=chainer_py_version,
                instance_type=instance_type,
                image_scope=scope,
            )
            expected = expected_uris.framework_uri(
                repo="sagemaker-chainer",
                fw_version=chainer_version,
                py_version=chainer_py_version,
                account=SAGEMAKER_ACCOUNT,
                processor=processor,
            )
            assert expected == uri

    for region, account in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.items():
        uri = image_uris.retrieve(
            framework="chainer",
            region=region,
            version=chainer_version,
            py_version=chainer_py_version,
            instance_type="ml.c4.xlarge",
            image_scope="training",
        )
        expected = expected_uris.framework_uri(
            repo="sagemaker-chainer",
            fw_version=chainer_version,
            py_version=chainer_py_version,
            account=account,
            region=region,
        )
        assert expected == uri


def test_tensorflow_training(tensorflow_training_version, tensorflow_training_py_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            framework="tensorflow",
            region=REGION,
            version=tensorflow_training_version,
            py_version=tensorflow_training_py_version,
            instance_type=instance_type,
            image_scope="training",
        )

        expected = _expected_tf_training_uri(
            tensorflow_training_version, tensorflow_training_py_version, processor=processor
        )
        assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        uri = image_uris.retrieve(
            framework="tensorflow",
            region=region,
            version=tensorflow_training_version,
            py_version=tensorflow_training_py_version,
            instance_type="ml.c4.xlarge",
            image_scope="training",
        )

        expected = _expected_tf_training_uri(
            tensorflow_training_version, tensorflow_training_py_version, region=region
        )
        assert expected == uri


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

    if repo.startswith("sagemaker"):
        account = (
            SAGEMAKER_ACCOUNT if region == REGION else SAGEMAKER_ALTERNATE_REGION_ACCOUNTS[region]
        )
    else:
        account = DLC_ACCOUNT if region == REGION else DLC_ALTERNATE_REGION_ACCOUNTS[region]

    return expected_uris.framework_uri(
        repo,
        tf_training_version,
        account,
        py_version=py_version,
        processor=processor,
        region=region,
    )


def test_tensorflow_inference(tensorflow_inference_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            framework="tensorflow",
            region=REGION,
            version=tensorflow_inference_version,
            py_version="py2",
            instance_type=instance_type,
            image_scope="inference",
        )

        expected = _expected_tf_inference_uri(tensorflow_inference_version, processor=processor)
        assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        uri = image_uris.retrieve(
            framework="tensorflow",
            region=region,
            version=tensorflow_inference_version,
            py_version="py2",
            instance_type="ml.c4.xlarge",
            image_scope="inference",
        )

        expected = _expected_tf_inference_uri(tensorflow_inference_version, region=region)
        assert expected == uri


def test_tensorflow_eia(tensorflow_eia_version):
    uri = image_uris.retrieve(
        framework="tensorflow",
        region=REGION,
        version=tensorflow_eia_version,
        py_version="py2",
        instance_type="ml.c4.xlarge",
        accelerator_type="ml.eia1.medium",
        image_scope="inference",
    )

    expected = _expected_tf_inference_uri(tensorflow_eia_version, eia=True)
    assert expected == uri

    for region in SAGEMAKER_ALTERNATE_REGION_ACCOUNTS.keys():
        uri = image_uris.retrieve(
            framework="tensorflow",
            region=region,
            version=tensorflow_eia_version,
            py_version="py2",
            instance_type="ml.c4.xlarge",
            accelerator_type="ml.eia1.medium",
            image_scope="inference",
        )

        expected = _expected_tf_inference_uri(tensorflow_eia_version, region=region, eia=True)
        assert expected == uri


def _expected_tf_inference_uri(tf_inference_version, processor="cpu", region=REGION, eia=False):
    version = Version(tf_inference_version)
    repo = _expected_tf_inference_repo(version, eia)
    py_version = "py2" if version < Version("1.11") else None

    if repo.startswith("sagemaker"):
        account = (
            SAGEMAKER_ACCOUNT if region == REGION else SAGEMAKER_ALTERNATE_REGION_ACCOUNTS[region]
        )
    else:
        account = DLC_ACCOUNT if region == REGION else DLC_ALTERNATE_REGION_ACCOUNTS[region]

    return expected_uris.framework_uri(
        repo, tf_inference_version, account, py_version, processor=processor, region=region,
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
