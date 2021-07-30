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
from mock import Mock, ANY
from sagemaker.tensorflow import TensorFlow


SCRIPT = "resnet_cifar_10.py"
TIMESTAMP = "2017-11-06-14:14:15.673"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE_GPU = "ml.p2.xlarge"
INSTANCE_TYPE_CPU = "ml.m4.xlarge"
REPOSITORY = "tensorflow-inference"
PROCESSOR = "cpu"
REGION = "us-west-2"
IMAGE_URI_FORMAT_STRING = "763104351884.dkr.ecr.{}.amazonaws.com/{}:{}-{}"
REGION = "us-west-2"
ROLE = "SagemakerRole"
SOURCE_DIR = "s3://fefergerger"
ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}
ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    ims = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
    )
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.expand_role = Mock(name="expand_role", return_value=ROLE)
    ims.sagemaker_client.describe_training_job = Mock(
        return_value={"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    )
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    return ims


def test_model_dir_false(sagemaker_session):
    estimator = TensorFlow(
        entry_point=SCRIPT,
        source_dir=SOURCE_DIR,
        role=ROLE,
        framework_version="2.3.0",
        py_version="py37",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        model_dir=False,
    )
    estimator.hyperparameters()
    assert estimator.model_dir is False


# Test that we pass all necessary fields from estimator to the session when we call deploy
def test_deploy(sagemaker_session):
    estimator = TensorFlow(
        entry_point=SCRIPT,
        source_dir=SOURCE_DIR,
        role=ROLE,
        framework_version="2.3.0",
        py_version="py37",
        instance_count=2,
        instance_type=INSTANCE_TYPE_CPU,
        sagemaker_session=sagemaker_session,
        base_job_name="test-cifar",
    )

    estimator.fit("s3://mybucket/train")

    estimator.deploy(initial_instance_count=1, instance_type=INSTANCE_TYPE_CPU)
    image = IMAGE_URI_FORMAT_STRING.format(REGION, REPOSITORY, "2.3.0", PROCESSOR)
    sagemaker_session.create_model.assert_called_with(
        ANY,
        ROLE,
        {
            "Image": image,
            "Environment": {"SAGEMAKER_TFS_NGINX_LOGLEVEL": "info"},
            "ModelDataUrl": "s3://m/m.tar.gz",
        },
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )
