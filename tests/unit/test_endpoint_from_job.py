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
from mock import MagicMock, Mock

import sagemaker

JOB_NAME = "myjob"
INITIAL_INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE = "myimage"
S3_MODEL_ARTIFACTS = "s3://mybucket/mymodel"
S3_MODEL_SRC_COMPRESSED = {
    "S3DataSource": {
        "S3Uri": S3_MODEL_ARTIFACTS,
        "S3DataType": "S3Object",
        "CompressionType": "Gzip",
    }
}
TRAIN_ROLE = "mytrainrole"
VPC_CONFIG = {"Subnets": ["subnet-foo"], "SecurityGroupIds": ["sg-foo"]}
TRAINING_JOB_RESPONSE = {
    "AlgorithmSpecification": {"TrainingImage": IMAGE},
    "ModelArtifacts": {"S3ModelArtifacts": S3_MODEL_ARTIFACTS},
    "RoleArn": TRAIN_ROLE,
    "VpcConfig": VPC_CONFIG,
}
FULL_CONTAINER_DEF = {"Environment": {}, "Image": IMAGE, "ModelDataUrl": S3_MODEL_ARTIFACTS}
DEPLOY_IMAGE = "mydeployimage"
DEPLOY_ROLE = "mydeployrole"
NEW_ENTITY_NAME = "mynewendpoint"
ENV_VARS = {"PYTHONUNBUFFERED": "TRUE", "some": "nonsense"}
ENDPOINT_FROM_MODEL_RETURNED_NAME = "endpointfrommodelname"
REGION = "us-west-2"


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    ims = sagemaker.Session(
        sagemaker_client=MagicMock(name="sagemaker_client"), boto_session=boto_mock
    )
    ims.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=TRAINING_JOB_RESPONSE
    )

    ims.endpoint_from_model_data = Mock(
        "endpoint_from_model_data", return_value=ENDPOINT_FROM_MODEL_RETURNED_NAME
    )
    return ims


def test_all_defaults_no_existing_entities(sagemaker_session):
    original_args = {
        "job_name": JOB_NAME,
        "initial_instance_count": INITIAL_INSTANCE_COUNT,
        "instance_type": INSTANCE_TYPE,
        "wait": False,
    }

    returned_name = sagemaker_session.endpoint_from_job(**original_args)

    expected_args = original_args.copy()
    expected_args.pop("job_name")
    expected_args["model_s3_location"] = S3_MODEL_SRC_COMPRESSED
    expected_args["image_uri"] = IMAGE
    expected_args["role"] = TRAIN_ROLE
    expected_args["name"] = JOB_NAME
    expected_args["model_environment_vars"] = None
    expected_args["model_vpc_config"] = VPC_CONFIG
    expected_args["accelerator_type"] = None
    expected_args["data_capture_config"] = None
    sagemaker_session.endpoint_from_model_data.assert_called_once_with(**expected_args)
    assert returned_name == ENDPOINT_FROM_MODEL_RETURNED_NAME


def test_no_defaults_no_existing_entities(sagemaker_session):
    vpc_config_override = {"Subnets": ["foo", "bar"], "SecurityGroupIds": ["baz"]}

    original_args = {
        "job_name": JOB_NAME,
        "initial_instance_count": INITIAL_INSTANCE_COUNT,
        "instance_type": INSTANCE_TYPE,
        "image_uri": DEPLOY_IMAGE,
        "role": DEPLOY_ROLE,
        "name": NEW_ENTITY_NAME,
        "model_environment_vars": ENV_VARS,
        "vpc_config_override": vpc_config_override,
        "accelerator_type": ACCELERATOR_TYPE,
        "wait": False,
    }

    returned_name = sagemaker_session.endpoint_from_job(**original_args)

    expected_args = original_args.copy()
    expected_args.pop("job_name")
    expected_args["model_s3_location"] = S3_MODEL_SRC_COMPRESSED
    expected_args["model_vpc_config"] = expected_args.pop("vpc_config_override")
    expected_args["data_capture_config"] = None
    sagemaker_session.endpoint_from_model_data.assert_called_once_with(**expected_args)
    assert returned_name == ENDPOINT_FROM_MODEL_RETURNED_NAME
