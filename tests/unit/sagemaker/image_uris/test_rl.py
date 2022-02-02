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

INSTANCE_TYPES_AND_PROCESSORS = (("ml.c4.xlarge", "cpu"), ("ml.p2.xlarge", "gpu"))
REGION = "us-west-2"

RL_ACCOUNT = "462105765813"
SAGEMAKER_ACCOUNT = "520713654638"


def _version_for_tag(toolkit, toolkit_version, framework, framework_in_tag=False):
    if framework_in_tag:
        return "-".join((toolkit, toolkit_version, framework))
    else:
        return "{}{}".format(toolkit, toolkit_version)


def test_coach_tf(coach_tensorflow_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            "coach-tensorflow",
            REGION,
            version=coach_tensorflow_version,
            instance_type=instance_type,
        )
        assert _expected_coach_tf_uri(coach_tensorflow_version, processor) == uri


def _expected_coach_tf_uri(coach_tf_version, processor):
    if Version(coach_tf_version) > Version("0.11.1"):
        return expected_uris.framework_uri(
            "sagemaker-rl-coach-container",
            _version_for_tag("coach", coach_tf_version, "tf", True),
            RL_ACCOUNT,
            py_version="py3",
            processor=processor,
        )
    else:
        return expected_uris.framework_uri(
            "sagemaker-rl-tensorflow",
            _version_for_tag("coach", coach_tf_version, "tf"),
            SAGEMAKER_ACCOUNT,
            py_version="py3",
            processor=processor,
        )


def test_coach_mxnet(coach_mxnet_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            "coach-mxnet", REGION, version=coach_mxnet_version, instance_type=instance_type
        )

        expected = expected_uris.framework_uri(
            "sagemaker-rl-mxnet",
            "coach{}".format(coach_mxnet_version),
            SAGEMAKER_ACCOUNT,
            py_version="py3",
            processor=processor,
        )
        assert expected == uri


def test_ray_tf(ray_tensorflow_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            "ray-tensorflow", REGION, version=ray_tensorflow_version, instance_type=instance_type
        )
        assert _expected_ray_tf_uri(ray_tensorflow_version, processor) == uri


def _expected_ray_tf_uri(ray_tf_version, processor):
    if Version(ray_tf_version) > Version("1.0.0"):
        return expected_uris.framework_uri(
            "sagemaker-rl-ray-container",
            _version_for_tag("ray", ray_tf_version, "tf", True),
            RL_ACCOUNT,
            py_version="py37",
            processor=processor,
        )
    elif Version(ray_tf_version) > Version("0.6.5"):
        return expected_uris.framework_uri(
            "sagemaker-rl-ray-container",
            _version_for_tag("ray", ray_tf_version, "tf", True),
            RL_ACCOUNT,
            py_version="py36",
            processor=processor,
        )
    else:
        return expected_uris.framework_uri(
            "sagemaker-rl-tensorflow",
            _version_for_tag("ray", ray_tf_version, "tf"),
            SAGEMAKER_ACCOUNT,
            py_version="py3",
            processor=processor,
        )


def test_ray_pytorch(ray_pytorch_version):
    for instance_type, processor in INSTANCE_TYPES_AND_PROCESSORS:
        uri = image_uris.retrieve(
            "ray-pytorch", REGION, version=ray_pytorch_version, instance_type=instance_type
        )

        expected = expected_uris.framework_uri(
            "sagemaker-rl-ray-container",
            "ray-{}-torch".format(ray_pytorch_version),
            RL_ACCOUNT,
            py_version="py36",
            processor=processor,
        )

        assert expected == uri


def test_vw(vw_version):
    version = "vw-{}".format(vw_version)
    uri = image_uris.retrieve("vw", REGION, version=version, instance_type="ml.c4.xlarge")

    expected = expected_uris.framework_uri("sagemaker-rl-vw-container", version, RL_ACCOUNT)
    assert expected == uri
