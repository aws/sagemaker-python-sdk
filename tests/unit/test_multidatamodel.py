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

import os
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
MXNET_FRAMEWORK_VERSION = "1.2"
MXNET_PY_VERSION = "py2"
MXNET_IMAGE = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:{}-cpu-{}".format(
    MXNET_FRAMEWORK_VERSION, MXNET_PY_VERSION
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
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
EXPECTED_PROD_VARIANT = [
    {
        "InitialVariantWeight": 1,
        "InitialInstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "ModelName": MODEL_NAME,
        "VariantName": "AllTraffic",
    }
]


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
        default_bucket_prefix=None,
    )
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.list_s3_files(
        bucket=S3_URL_SOURCE_BUCKET, key_prefix=S3_URL_SOURCE_PREFIX
    ).return_value = Mock()
    session.upload_data = Mock(
        name="upload_data",
        return_value=os.path.join(VALID_MULTI_MODEL_DATA_PREFIX, "mleap_model.tar.gz"),
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}

    s3_mock = Mock()
    boto_mock.client("s3").return_value = s3_mock
    boto_mock.client("s3").get_paginator("list_objects_v2").paginate.return_value = Mock()
    s3_mock.reset_mock()

    return session


@pytest.fixture()
def multi_data_model(sagemaker_session):
    return MultiDataModel(
        name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image_uri=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture()
def mxnet_model(sagemaker_session):
    return MXNetModel(
        MXNET_MODEL_DATA,
        entry_point=ENTRY_POINT,
        framework_version=MXNET_FRAMEWORK_VERSION,
        py_version=MXNET_PY_VERSION,
        role=MXNET_ROLE,
        sagemaker_session=sagemaker_session,
        name=MXNET_MODEL_NAME,
        enable_network_isolation=True,
    )


def test_multi_data_model_create_with_invalid_model_data_prefix():
    invalid_model_data_prefix = "https://mybucket/path/"
    with pytest.raises(ValueError) as ex:
        MultiDataModel(
            name=MODEL_NAME, model_data_prefix=invalid_model_data_prefix, image_uri=IMAGE, role=ROLE
        )
    err_msg = 'Expecting S3 model prefix beginning with "s3://". Received: "{}"'.format(
        invalid_model_data_prefix
    )
    assert err_msg in str(ex.value)


def test_multi_data_model_create_with_invalid_arguments(sagemaker_session, mxnet_model):
    with pytest.raises(ValueError) as ex:
        MultiDataModel(
            name=MODEL_NAME,
            model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
            image_uri=IMAGE,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model=mxnet_model,
        )
    assert (
        "Parameters image_uri, role, and kwargs are not permitted when model parameter is passed."
        in str(ex)
    )


def test_multi_data_model_create(sagemaker_session):
    model = MultiDataModel(
        name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image_uri=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )

    assert model.sagemaker_session == sagemaker_session
    assert model.name == MODEL_NAME
    assert model.model_data_prefix == VALID_MULTI_MODEL_DATA_PREFIX
    assert model.role == ROLE
    assert model.image_uri == IMAGE
    assert model.vpc_config is None


@patch("sagemaker.multidatamodel.Session", MagicMock())
def test_multi_data_model_create_with_model_arg_only(mxnet_model):
    model = MultiDataModel(
        name=MODEL_NAME, model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX, model=mxnet_model
    )

    assert model.model_data_prefix == VALID_MULTI_MODEL_DATA_PREFIX
    assert model.model == mxnet_model
    assert hasattr(model, "role") is False
    assert hasattr(model, "image_uri") is False


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_prepare_container_def_mxnet(sagemaker_session, mxnet_model):
    expected_container_env_keys = [
        "SAGEMAKER_CONTAINER_LOG_LEVEL",
        "SAGEMAKER_PROGRAM",
        "SAGEMAKER_REGION",
        "SAGEMAKER_SUBMIT_DIRECTORY",
    ]
    model = MultiDataModel(
        name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
    )

    container_def = model.prepare_container_def(INSTANCE_TYPE)

    assert container_def["Image"] == MXNET_IMAGE
    assert container_def["ModelDataUrl"] == VALID_MULTI_MODEL_DATA_PREFIX
    assert container_def["Mode"] == MULTI_MODEL_CONTAINER_MODE
    # Check if the environment variables defined only for MXNetModel
    # are part of the MultiDataModel container definition
    assert set(container_def["Environment"].keys()) == set(expected_container_env_keys)


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_multi_data_model(sagemaker_session):
    model = MultiDataModel(
        name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        image_uri=IMAGE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        env={"EXTRA_ENV_MOCK": "MockValue"},
    )
    model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=MULTI_MODEL_ENDPOINT_NAME,
    )

    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME,
        ROLE,
        model.prepare_container_def(INSTANCE_TYPE),
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MULTI_MODEL_ENDPOINT_NAME,
        wait=True,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        production_variants=EXPECTED_PROD_VARIANT,
    )


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_multi_data_framework_model(sagemaker_session, mxnet_model):
    model = MultiDataModel(
        name=MODEL_NAME,
        model_data_prefix=VALID_MULTI_MODEL_DATA_PREFIX,
        sagemaker_session=sagemaker_session,
        model=mxnet_model,
    )

    predictor = model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=MULTI_MODEL_ENDPOINT_NAME,
    )

    # Assert if this is called with mxnet_model parameters
    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME,
        MXNET_ROLE,
        model.prepare_container_def(INSTANCE_TYPE),
        vpc_config=None,
        enable_network_isolation=True,
        tags=None,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MULTI_MODEL_ENDPOINT_NAME,
        wait=True,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        production_variants=EXPECTED_PROD_VARIANT,
    )
    sagemaker_session.create_endpoint_config.assert_not_called()
    assert isinstance(predictor, MXNetPredictor)


def test_add_model_local_file_path(multi_data_model):
    valid_local_model_artifact_path = os.path.join(DATA_DIR, "sparkml_model", "mleap_model.tar.gz")
    uploaded_s3_path = multi_data_model.add_model(valid_local_model_artifact_path)

    assert uploaded_s3_path == os.path.join(VALID_MULTI_MODEL_DATA_PREFIX, "mleap_model.tar.gz")


def test_add_model_s3_path(multi_data_model):
    uploaded_s3_path = multi_data_model.add_model(VALID_S3_URL)

    assert uploaded_s3_path == os.path.join(VALID_MULTI_MODEL_DATA_PREFIX, "output/model.tar.gz")
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
    uploaded_s3_path = multi_data_model.add_model(VALID_S3_URL, "customer-a/model.tar.gz")

    assert uploaded_s3_path == os.path.join(
        VALID_MULTI_MODEL_DATA_PREFIX, "customer-a/model.tar.gz"
    )
    multi_data_model.s3_client.copy.assert_called()
    calls = [
        call(
            {"Bucket": S3_URL_SOURCE_BUCKET, "Key": S3_URL_SOURCE_PREFIX},
            DST_BUCKET,
            "path/customer-a/model.tar.gz",
        )
    ]
    multi_data_model.s3_client.copy.assert_has_calls(calls)


def test_add_model_with_invalid_model_uri(multi_data_model):
    with pytest.raises(ValueError) as ex:
        multi_data_model.add_model(INVALID_S3_URL)

    assert 'model_source must either be a valid local file path or s3 uri. Received: "{}"'.format(
        INVALID_S3_URL
    ) in str(ex.value)


def test_list_models(multi_data_model):
    multi_data_model.list_models()

    multi_data_model.sagemaker_session.list_s3_files.assert_called()
    assert multi_data_model.sagemaker_session.list_s3_files.called_with(
        Bucket=S3_URL_SOURCE_BUCKET, Prefix=S3_URL_SOURCE_PREFIX
    )
