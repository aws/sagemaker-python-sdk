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

import os

from mock import Mock
import pytest

from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris, regions

from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.tensorflow import TensorFlow
from sagemaker.pytorch import PyTorch
from tests.unit import DATA_DIR


ACCOUNTS = {
    "af-south-1": "314341159256",
    "ap-east-1": "199566480951",
    "ap-northeast-1": "430734990657",
    "ap-northeast-2": "578805364391",
    "ap-south-1": "904829902805",
    "ap-southeast-1": "972752614525",
    "ap-southeast-2": "184798709955",
    "ca-central-1": "519511493484",
    "cn-north-1": "618459771430",
    "cn-northwest-1": "658757709296",
    "eu-central-1": "482524230118",
    "eu-north-1": "314864569078",
    "eu-south-1": "563282790590",
    "eu-west-1": "929884845733",
    "eu-west-2": "250201462417",
    "eu-west-3": "447278800020",
    "me-south-1": "986000313247",
    "sa-east-1": "818342061345",
    "us-east-1": "503895931360",
    "us-east-2": "915447279597",
    "us-west-1": "685455198987",
    "us-west-2": "895741380848",
}

SCRIPT_FILE = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_FILE)
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ROLE = "Dummy"
REGION = "us-west-2"

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    return session


def _build_tf(
    sagemaker_session,
    framework_version=None,
    py_version=None,
    instance_type=None,
    base_job_name=None,
    **kwargs
):
    return TensorFlow(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        py_version=py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=instance_type if instance_type else INSTANCE_TYPE,
        base_job_name=base_job_name,
        **kwargs
    )


def _build_pt(
    sagemaker_session,
    framework_version=None,
    py_version=None,
    instance_type=None,
    base_job_name=None,
    **kwargs
):
    return PyTorch(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        py_version=py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=instance_type if instance_type else INSTANCE_TYPE,
        base_job_name=base_job_name,
        **kwargs
    )


def test_debugger():
    for region in regions.regions():
        if region in ACCOUNTS.keys():
            uri = image_uris.retrieve("debugger", region=region)

            expected = expected_uris.algo_uri(
                "sagemaker-debugger-rules", ACCOUNTS[region], region, version="latest"
            )
            assert expected == uri


def test_tensorflow_default_profiling_image_uri(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="2.3.1",
        py_version="py37",
        instance_type="ml.p3.8xlarge",
    )

    assert tf.has_custom_profiler_config is False and "cu110" not in tf.training_image_uri()


def test_tensorflow_custom_profiling_image_uri(sagemaker_session):
    tf = _build_tf(
        sagemaker_session,
        framework_version="2.3.1",
        py_version="py37",
        instance_type="ml.p3.8xlarge",
        profiler_config=ProfilerConfig(framework_profile_params=FrameworkProfile()),
    )

    assert tf.has_custom_profiler_config is True and "cu110" in tf.training_image_uri()


def test_pytorch_default_profiling_image_uri(sagemaker_session):
    pt = _build_pt(
        sagemaker_session,
        framework_version="1.6.0",
        py_version="py36",
        instance_type="ml.p3.8xlarge",
    )

    assert pt.has_custom_profiler_config is False and "cu110" not in pt.training_image_uri()


def test_pytorch_custom_profiling_image_uri(sagemaker_session):
    pt = _build_pt(
        sagemaker_session,
        framework_version="1.6.0",
        py_version="py36",
        instance_type="ml.p3.8xlarge",
        profiler_config=ProfilerConfig(framework_profile_params=FrameworkProfile()),
    )

    assert pt.has_custom_profiler_config is True and "cu110" in pt.training_image_uri()
