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

from mock import Mock, patch
from packaging import version
import pytest

from sagemaker.tensorflow import TensorFlow

REGION = "us-west-2"

ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}


@pytest.fixture()
def sagemaker_session():
    session_mock = Mock(
        name="sagemaker_session",
        boto_region_name=REGION,
        default_bucket_prefix=None,
    )
    session_mock.sagemaker_config = {}
    return session_mock


def _build_tf(sagemaker_session, **kwargs):
    return TensorFlow(
        sagemaker_session=sagemaker_session,
        entry_point="dummy.py",
        role="dummy-role",
        instance_count=1,
        instance_type="ml.c4.xlarge",
        **kwargs,
    )


@patch("sagemaker.fw_utils.python_deprecation_warning")
def test_estimator_py2_deprecation_warning(warning, sagemaker_session):
    estimator = _build_tf(sagemaker_session, framework_version="2.1.1", py_version="py2")

    assert estimator.py_version == "py2"
    warning.assert_called_with("tensorflow", "2.1.1")


def test_py2_version_deprecated(sagemaker_session):
    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session, framework_version="2.1.2", py_version="py2")

    msg = (
        "Python 2 containers are only available with 2.1.1 and lower versions. "
        "Please use a Python 3 container."
    )
    assert msg in str(e.value)


def test_py2_version_is_not_deprecated(sagemaker_session):
    estimator = _build_tf(sagemaker_session, framework_version="1.15.0", py_version="py2")
    assert estimator.py_version == "py2"
    estimator = _build_tf(sagemaker_session, framework_version="2.0.0", py_version="py2")
    assert estimator.py_version == "py2"


def test_framework_name(sagemaker_session):
    tf = _build_tf(sagemaker_session, framework_version="1.15.2", py_version="py3")
    assert tf._framework_name == "tensorflow"


def test_tf_add_environment_variables(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="1.15.2",
        py_version="py3",
        environment=ENV_INPUT,
    )
    assert tf.environment == ENV_INPUT


def test_tf_miss_environment_variables(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="1.15.2",
        py_version="py3",
        environment=None,
    )
    assert not tf.environment


def test_enable_sm_metrics(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="1.15.2",
        py_version="py3",
        enable_sagemaker_metrics=True,
    )
    assert tf.enable_sagemaker_metrics


def test_disable_sm_metrics(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="1.15.2",
        py_version="py3",
        enable_sagemaker_metrics=False,
    )
    assert not tf.enable_sagemaker_metrics


def test_disable_sm_metrics_if_fw_ver_is_less_than_1_15(
    sagemaker_session, tensorflow_training_version, tensorflow_training_py_version
):
    if version.Version(tensorflow_training_version) > version.Version("1.14"):
        pytest.skip("This test is for TF 1.14 and lower.")

    tf = _build_tf(
        sagemaker_session,
        framework_version=tensorflow_training_version,
        py_version=tensorflow_training_py_version,
        image_uri="old-image",
    )
    assert tf.enable_sagemaker_metrics is None


def test_enable_sm_metrics_if_fw_ver_is_at_least_1_15(
    sagemaker_session, tensorflow_training_version, tensorflow_training_py_version
):
    if version.Version(tensorflow_training_version) < version.Version("1.15"):
        pytest.skip("This test is for TF 1.15 and higher.")

    tf = _build_tf(
        sagemaker_session,
        framework_version=tensorflow_training_version,
        py_version=tensorflow_training_py_version,
    )
    assert tf.enable_sagemaker_metrics


def test_require_image_uri_if_fw_ver_is_less_than_1_11(
    sagemaker_session, tensorflow_training_version, tensorflow_training_py_version
):
    if version.Version(tensorflow_training_version) > version.Version("1.10"):
        pytest.skip("This test is for TF 1.10 and lower.")

    with pytest.raises(ValueError) as e:
        _build_tf(
            sagemaker_session,
            framework_version=tensorflow_training_version,
            py_version=tensorflow_training_py_version,
        )

    expected_msg = (
        "TF {version} supports only legacy mode. Please supply the image URI directly with "
        "'image_uri=520713654638.dkr.ecr.{region}.amazonaws.com/"
        "sagemaker-tensorflow:{version}-cpu-py2' and set 'model_dir=False'. If you are using any "
        "legacy parameters (training_steps, evaluation_steps, checkpoint_path, requirements_file), "
        "make sure to pass them directly as hyperparameters instead."
    ).format(version=tensorflow_training_version, region=REGION)

    assert expected_msg in str(e.value)
