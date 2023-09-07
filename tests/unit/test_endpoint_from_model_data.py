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
from botocore.exceptions import ClientError
from mock import MagicMock, Mock
from mock import patch

from .common import (
    _raise_unexpected_client_error,
    _raise_does_already_exists_client_error,
    _raise_does_not_exist_client_error,
)
import sagemaker

ENDPOINT_NAME = "myendpoint"
INITIAL_INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
S3_MODEL_ARTIFACTS = "s3://mybucket/mymodel"
DEPLOY_IMAGE = "mydeployimage"
CONTAINER_DEF = {"Environment": {}, "Image": DEPLOY_IMAGE, "ModelDataUrl": S3_MODEL_ARTIFACTS}
VPC_CONFIG = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
DEPLOY_ROLE = "mydeployrole"
ENV_VARS = {"PYTHONUNBUFFERED": "TRUE", "some": "nonsense"}
NAME_FROM_IMAGE = "namefromimage"
REGION = "us-west-2"


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    ims = sagemaker.Session(
        sagemaker_client=MagicMock(name="sagemaker_client"), boto_session=boto_mock
    )
    ims.create_model = Mock(name="create_model")
    ims.create_endpoint_config = Mock(name="create_endpoint_config")
    ims.create_endpoint = Mock(name="create_endpoint")
    return ims


@patch("sagemaker.session.name_from_image", return_value=NAME_FROM_IMAGE)
def test_all_defaults_no_existing_entities(name_from_image_mock, sagemaker_session):
    returned_name = sagemaker_session.endpoint_from_model_data(
        model_s3_location=S3_MODEL_ARTIFACTS,
        image_uri=DEPLOY_IMAGE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        role=DEPLOY_ROLE,
        wait=False,
    )
    sagemaker_session.create_model.assert_called_once_with(
        name=NAME_FROM_IMAGE, role=DEPLOY_ROLE, container_defs=CONTAINER_DEF, vpc_config=None
    )
    sagemaker_session.create_endpoint_config.assert_called_once_with(
        name=NAME_FROM_IMAGE,
        model_name=NAME_FROM_IMAGE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        data_capture_config_dict=None,
        tags=None,
    )
    sagemaker_session.create_endpoint.assert_called_once_with(
        endpoint_name=NAME_FROM_IMAGE, config_name=NAME_FROM_IMAGE, wait=False, tags=None
    )
    assert returned_name == NAME_FROM_IMAGE


@patch("sagemaker.session.name_from_image", return_value=NAME_FROM_IMAGE)
def test_no_defaults_no_existing_entities(name_from_image_mock, sagemaker_session):
    container_def_with_env = CONTAINER_DEF.copy()
    container_def_with_env.update({"Environment": ENV_VARS})

    returned_name = sagemaker_session.endpoint_from_model_data(
        model_s3_location=S3_MODEL_ARTIFACTS,
        image_uri=DEPLOY_IMAGE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        role=DEPLOY_ROLE,
        wait=False,
        name=ENDPOINT_NAME,
        model_environment_vars=ENV_VARS,
        model_vpc_config=VPC_CONFIG,
        accelerator_type=ACCELERATOR_TYPE,
    )
    sagemaker_session.create_model.assert_called_once_with(
        name=ENDPOINT_NAME,
        role=DEPLOY_ROLE,
        container_defs=container_def_with_env,
        vpc_config=VPC_CONFIG,
    )
    sagemaker_session.create_endpoint_config.assert_called_once_with(
        name=ENDPOINT_NAME,
        model_name=ENDPOINT_NAME,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        data_capture_config_dict=None,
        tags=None,
    )
    sagemaker_session.create_endpoint.assert_called_once_with(
        endpoint_name=ENDPOINT_NAME, config_name=ENDPOINT_NAME, wait=False, tags=None
    )
    assert returned_name == ENDPOINT_NAME


@patch("sagemaker.session.name_from_image", return_value=NAME_FROM_IMAGE)
def test_model_and_endpoint_config_exist(name_from_image_mock, sagemaker_session):
    container_def_with_env = CONTAINER_DEF.copy()

    sagemaker_session.create_endpoint_config = Mock(
        name="create_endpoint_config", side_effect=_raise_does_already_exists_client_error
    )

    try:
        sagemaker_session.endpoint_from_model_data(
            model_s3_location=S3_MODEL_ARTIFACTS,
            image_uri=DEPLOY_IMAGE,
            initial_instance_count=INITIAL_INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            wait=False,
        )
    except ClientError:
        assert False, "Unexpected ClientError raised for resource already exists scenario"

    sagemaker_session.create_model.assert_called_once_with(
        name=NAME_FROM_IMAGE,
        role=None,
        container_defs=container_def_with_env,
        vpc_config=None,
    )
    sagemaker_session.create_endpoint_config.assert_called_once_with(
        name=NAME_FROM_IMAGE,
        model_name=NAME_FROM_IMAGE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        data_capture_config_dict=None,
        tags=None,
    )
    sagemaker_session.create_endpoint.assert_called_once_with(
        endpoint_name=NAME_FROM_IMAGE, config_name=NAME_FROM_IMAGE, wait=False, tags=None
    )


@patch("sagemaker.session.name_from_image", return_value=NAME_FROM_IMAGE)
def test_model_and_endpoint_config_raises_unexpected_error(name_from_image_mock, sagemaker_session):
    container_def_with_env = CONTAINER_DEF.copy()

    sagemaker_session.create_endpoint_config = Mock(
        name="create_endpoint_config", side_effect=_raise_unexpected_client_error
    )

    with pytest.raises(ClientError):
        sagemaker_session.endpoint_from_model_data(
            model_s3_location=S3_MODEL_ARTIFACTS,
            image_uri=DEPLOY_IMAGE,
            initial_instance_count=INITIAL_INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            wait=False,
        )

    sagemaker_session.create_model.assert_called_once_with(
        name=NAME_FROM_IMAGE,
        role=None,
        container_defs=container_def_with_env,
        vpc_config=None,
    )
    sagemaker_session.create_endpoint_config.assert_called_once_with(
        name=NAME_FROM_IMAGE,
        model_name=NAME_FROM_IMAGE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        data_capture_config_dict=None,
        tags=None,
    )
    sagemaker_session.create_endpoint.assert_not_called()


def test_create_resource_entity_exists():
    # _create_resource returns False
    assert not sagemaker.session._create_resource(_raise_does_already_exists_client_error)


def test_create_resource_unexpected_error():
    # _create_resource returns ClientError
    with pytest.raises(ClientError):
        sagemaker.session._create_resource(_raise_unexpected_client_error)


def test_create_resource_entity_doesnt_exist():
    # _create_resource returns True
    assert sagemaker.session._create_resource(lambda: None)


def test_deployment_entity_exists():
    assert sagemaker.session._deployment_entity_exists(lambda: None)


def test_entity_doesnt_exist():
    assert not sagemaker.session._deployment_entity_exists(_raise_does_not_exist_client_error)


def test_describe_failure():
    with pytest.raises(ClientError):
        sagemaker.session._deployment_entity_exists(_raise_unexpected_client_error)
