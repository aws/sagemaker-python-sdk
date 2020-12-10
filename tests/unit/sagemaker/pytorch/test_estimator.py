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

import os

from mock import Mock
import pytest

from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.pytorch import PyTorch
from tests.unit import DATA_DIR

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


def test_default_profiling_image_uri(sagemaker_session):
    pt = _build_pt(
        sagemaker_session,
        framework_version="1.6.0",
        py_version="py36",
        instance_type="ml.p3.8xlarge",
    )

    assert pt.has_custom_profiler_config is False and "cu110" not in pt.training_image_uri()


def test_custom_profiling_image_uri(sagemaker_session):
    pt = _build_pt(
        sagemaker_session,
        framework_version="1.6.0",
        py_version="py36",
        instance_type="ml.p3.8xlarge",
        profiler_config=ProfilerConfig(framework_profile_params=FrameworkProfile()),
    )

    assert pt.has_custom_profiler_config is True and "cu110" in pt.training_image_uri()
