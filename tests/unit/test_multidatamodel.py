# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from six.moves.urllib import parse

import pytest
from mock import MagicMock, Mock, call, patch

from sagemaker.multidatamodel import MULTI_MODEL_CONTAINER_MODE
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.mxnet import MXNetModel, MXNetPredictor

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}
ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}]}

ENTRY_POINT = "mock.py"
MXNET_MODEL_DATA = "s3://mybucket/mxnet_path/model.tar.gz"
MXNET_MODEL_NAME = "dummy-mxnet-model"
MXNET_ROLE = "DummyMXNetRole"

IMAGE = "123456789012.dkr.ecr.dummyregion.amazonaws.com/dummyimage:latest"
REGION = "us-west-2"
ROLE = "DummyRole"
MODEL_NAME = "dummy-model"
VALID_MULTI_MODEL_DATA_PREFIX = "s3://mybucket/path/"
INVALID_S3_URL = "https://my-training-bucket.s3.myregion.amazonaws.com/output/model.tar.gz"
VALID_S3_URL = "s3://my-training-bucket/output/model.tar.gz"
S3_URL_SOURCE_BUCKET = "my-training-bucket"
S3_URL_SOURCE_PREFIX = "output/model.tar.gz"
DST_BUCKET = "mybucket"

MULTI_MODEL_ENDPOINT_NAME = "multimodel-endpoint"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    s3_mock = Mock()
    boto_mock.client("s3").return_value = s3_mock
    boto_mock.client("s3").get_paginator("list_objects_v2").paginate.return_value = Mock()
    s3_mock.reset_mock()
    return session


@pytest.fixture()
def multi_data_model(sagemaker_session):
    return MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture()
def mxnet_model(sagemaker_session):
    return MXNetModel(
        MXNET_MODEL_DATA,
        role=MXNET_ROLE,
        entry_point=ENTRY_POINT,
        sagemaker_session=sagemaker_session,
        name=MXNET_MODEL_NAME,
    )


def test_multi_data_model_create_with_invalid_model_data_prefix():
    invalid_model_data_prefix = "https://mybucket/path/"
    with pytest.raises(ValueError) as ex:
        MultiDataModel(
            model_name=MODEL_NAME,
            model_data_prefix=invalid_model_data_prefix,
            image=IMAGE,
            role=ROLE,
        )
    err_msg = (
        'ValueError: Expecting S3 model prefix beginning with "s3://" and ending in "/". '
        'Received: "{}"'.format(invalid_model_data_prefix)
    )
    assert err_msg in str(ex)


def test_multi_data_model_create(sagemaker_session):
    model = MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )

    url = parse.urlparse(VALID_MULTI_MODEL_DATA_PREFIX)
    bucket, model_data_path = url.netloc, url.path.lstrip("/")
    calls = [call(Bucket=bucket, Key=os.path.join(model_data_path, "/"))]

    model.s3_client.put_object.assert_has_calls(calls)

    assert model.sagemaker_session == sagemaker_session
    assert model.model_name == MODEL_NAME
    assert model.model_data_prefix == VALID_MULTI_MODEL_DATA_PREFIX
    assert model.role == ROLE
    assert model.image == IMAGE
    assert model.vpc_config is None


def test_multi_data_model_create_with_model_arg(sagemaker_session, mxnet_model):
    model = MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
    )

    url = parse.urlparse(VALID_MULTI_MODEL_DATA_PREFIX)
    bucket, model_data_path = url.netloc, url.path.lstrip("/")
    calls = [call(Bucket=bucket, Key=os.path.join(model_data_path, "/"))]

    model.s3_client.put_object.assert_has_calls(calls)

    assert model.sagemaker_session == sagemaker_session
    assert model.model_name == MODEL_NAME
    assert model.model_data_prefix == VALID_MULTI_MODEL_DATA_PREFIX
    assert model.role == ROLE
    assert model.image == IMAGE
    assert model.vpc_config is None


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_prepare_container_def_mxnet(sagemaker_session, mxnet_model):
    expected_container_env_keys = [
        "SAGEMAKER_CONTAINER_LOG_LEVEL",
        "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS",
        "SAGEMAKER_PROGRAM",
        "SAGEMAKER_REGION",
        "SAGEMAKER_SUBMIT_DIRECTORY",
        "EXTRA_ENV_MOCK",
    ]
    model = MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
        env={"EXTRA_ENV_MOCK": "MockValue"},
    )

    container_def = model.prepare_container_def(INSTANCE_TYPE)

    assert container_def["Image"] == IMAGE
    assert container_def["ModelDataUrl"] == VALID_MULTI_MODEL_DATA_PREFIX
    assert container_def["Mode"] == MULTI_MODEL_CONTAINER_MODE
    # Check if the environment variables for both MultiDataModel
    # and MXNetModel exist as part of the container definition
    assert set(container_def["Environment"].keys()) == set(expected_container_env_keys)


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy(sagemaker_session, mxnet_model):
    model = MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
        env={"EXTRA_ENV_MOCK": "MockValue"},
    )

    predictor = model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=MULTI_MODEL_ENDPOINT_NAME,
    )

    assert isinstance(predictor, MXNetPredictor)


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_update(sagemaker_session, mxnet_model):
    model = MultiDataModel(
        model_name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
    )

    model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=MULTI_MODEL_ENDPOINT_NAME,
        update_endpoint=True,
    )

    sagemaker_session.create_model.assert_called()
    sagemaker_session.create_endpoint_config.assert_called_with(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
    )

    config_name = sagemaker_session.create_endpoint_config(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
    )
    sagemaker_session.update_endpoint.assert_called_with(MULTI_MODEL_ENDPOINT_NAME, config_name)
    sagemaker_session.create_endpoint.assert_not_called()


def test_add_model_with_invalid_model_uri(multi_data_model):
    with pytest.raises(ValueError) as ex:
        multi_data_model.add_model(INVALID_S3_URL)

    assert 'ValueError: Expecting S3 model path beginning with "s3://". Received: "{}"'.format(
        INVALID_S3_URL
    ) in str(ex)


def test_add_model(multi_data_model):
    multi_data_model.add_model(VALID_S3_URL)

    multi_data_model.s3_client.copy.assert_called()
    calls = [
        call(
            {"Bucket": S3_URL_SOURCE_BUCKET, "Key": S3_URL_SOURCE_PREFIX},
            DST_BUCKET,
            "path/output/model.tar.gz",
        )
    ]
    multi_data_model.s3_client.copy.assert_has_calls(calls)


def test_add_model_with_dst_path(multi_data_model):
    multi_data_model.add_model(VALID_S3_URL, "customer-a/model.tar.gz")

    multi_data_model.s3_client.copy.assert_called()
    calls = [
        call(
            {"Bucket": S3_URL_SOURCE_BUCKET, "Key": S3_URL_SOURCE_PREFIX},
            DST_BUCKET,
            "path/customer-a/model.tar.gz",
        )
    ]
    multi_data_model.s3_client.copy.assert_has_calls(calls)


def test_list_models(multi_data_model):
    multi_data_model.list_models()

    multi_data_model.s3_client.get_paginator.assert_called_with("list_objects_v2")
    assert multi_data_model.s3_client.get_paginator("list_objects_v2").paginate.called_with(
        Bucket=S3_URL_SOURCE_BUCKET, Prefix="path/"
    )
