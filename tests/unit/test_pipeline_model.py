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

import copy

import pytest
from botocore.utils import merge_dicts
from mock import Mock, patch
from mock.mock import ANY

from sagemaker.model import FrameworkModel
from sagemaker.pipeline import PipelineModel
from sagemaker.predictor import Predictor
from sagemaker.session_settings import SessionSettings
from sagemaker.sparkml import SparkMLModel
from tests.unit import SAGEMAKER_CONFIG_MODEL, SAGEMAKER_CONFIG_ENDPOINT_CONFIG

ENTRY_POINT = "blah.py"
MODEL_DATA_1 = "s3://bucket/model_1.tar.gz"
MODEL_DATA_2 = "s3://bucket/model_2.tar.gz"
MODEL_IMAGE_1 = "mi-1"
MODEL_IMAGE_2 = "mi-2"
INSTANCE_TYPE = "ml.m4.xlarge"
ROLE = "some-role"
ENV_1 = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json"}
ENV_2 = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"}
ENDPOINT = "some-ep"


TIMESTAMP = "2017-10-10-14-14-15"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
IMAGE_NAME = "fakeimage"
REGION = "us-west-2"


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            MODEL_DATA_1,
            MODEL_IMAGE_1,
            ROLE,
            ENTRY_POINT,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )

    def create_predictor(self, endpoint_name):
        return Predictor(endpoint_name, self.sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_client=None,
        s3_resource=None,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}

    return sms


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_prepare_container_def(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        env={"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    assert model.pipeline_container_def(INSTANCE_TYPE) == [
        {
            "Environment": {
                "SAGEMAKER_PROGRAM": "blah.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/mi-1-2017-10-10-14-14-15/sourcedir.tar.gz",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": "us-west-2",
            },
            "Image": "mi-1",
            "ModelDataUrl": "s3://bucket/model_1.tar.gz",
        },
        {
            "Environment": {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
            "Image": "246618743249.dkr.ecr.us-west-2.amazonaws.com"
            + "/sagemaker-sparkml-serving:3.3",
            "ModelDataUrl": "s3://bucket/model_2.tar.gz",
        },
    ]


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    kms_key = "pipeline-model-deploy-kms-key"
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, kms_key=kms_key)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name="mi-1-2017-10-10-14-14-15",
        production_variants=[
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags=None,
        kms_key=kms_key,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy_endpoint_name(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name="mi-1-2017-10-10-14-14-15",
        production_variants=[
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy_update_endpoint(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    endpoint_name = "endpoint-name"
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=True,
    )

    sagemaker_session.create_endpoint_config.assert_called_with(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )
    config_name = sagemaker_session.create_endpoint_config(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, config_name, wait=True)
    sagemaker_session.create_endpoint.assert_not_called()


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_transformer(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model_name = "ModelName"
    model = PipelineModel(
        models=[framework_model, sparkml_model],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        name=model_name,
    )

    instance_count = 55
    strategy = "MultiRecord"
    assemble_with = "Line"
    output_path = "s3://output/path"
    output_kms_key = "output:kms:key"
    accept = "application/jsonlines"
    env = {"my_key": "my_value"}
    max_concurrent_transforms = 20
    max_payload = 5
    tags = [{"my_tag": "my_value"}]
    volume_kms_key = "volume:kms:key"
    transformer = model.transformer(
        instance_type=INSTANCE_TYPE,
        instance_count=instance_count,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=output_kms_key,
        accept=accept,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        volume_kms_key=volume_kms_key,
    )
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.instance_count == instance_count
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == output_path
    assert transformer.output_kms_key == output_kms_key
    assert transformer.accept == accept
    assert transformer.env == env
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.tags == tags
    assert transformer.volume_kms_key == volume_kms_key
    assert transformer.model_name == model_name


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy_tags(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    tags = [{"ModelName": "TestModel"}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, tags=tags)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name="mi-1-2017-10-10-14-14-15",
        production_variants=[
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags=tags,
        wait=True,
        kms_key=None,
        data_capture_config_dict=None,
    )


def test_pipeline_model_without_role(sagemaker_session):
    with pytest.raises(ValueError):
        PipelineModel([], sagemaker_session=sagemaker_session)


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_pipeline_model_with_config_injection(tfo, time, sagemaker_session):
    combined_config = copy.deepcopy(SAGEMAKER_CONFIG_MODEL)
    endpoint_config = copy.deepcopy(SAGEMAKER_CONFIG_ENDPOINT_CONFIG)
    merge_dicts(combined_config, endpoint_config)
    sagemaker_session.sagemaker_config = combined_config

    sagemaker_session.create_model = Mock()
    sagemaker_session.endpoint_from_production_variants = Mock()

    expected_role_arn = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"]["ExecutionRoleArn"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"][
        "EnableNetworkIsolation"
    ]
    expected_vpc_config = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"]["VpcConfig"]
    expected_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]

    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    pipeline_model = PipelineModel(
        [framework_model, sparkml_model], sagemaker_session=sagemaker_session
    )
    assert pipeline_model.role == expected_role_arn
    assert pipeline_model.vpc_config == expected_vpc_config
    assert pipeline_model.enable_network_isolation == expected_enable_network_isolation

    pipeline_model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)

    sagemaker_session.create_model.assert_called_with(
        ANY,
        expected_role_arn,
        ANY,
        vpc_config=expected_vpc_config,
        enable_network_isolation=expected_enable_network_isolation,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name="mi-1-2017-10-10-14-14-15",
        production_variants=[
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags=None,
        kms_key=expected_kms_key_id,
        wait=True,
        data_capture_config_dict=None,
    )


def test_delete_model_without_deploy(sagemaker_session):
    pipeline_model = PipelineModel([], role=ROLE, sagemaker_session=sagemaker_session)

    expected_error_message = "The SageMaker model must be created before attempting to delete."
    with pytest.raises(ValueError, match=expected_error_message):
        pipeline_model.delete_model()


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_delete_model(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    pipeline_model = PipelineModel(
        [framework_model], role=ROLE, sagemaker_session=sagemaker_session
    )
    pipeline_model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)

    pipeline_model.delete_model()
    sagemaker_session.delete_model.assert_called_with(pipeline_model.name)


@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_network_isolation(tfo, time, sagemaker_session):
    framework_model = DummyFrameworkModel(sagemaker_session)
    sparkml_model = SparkMLModel(
        model_data=MODEL_DATA_2, role=ROLE, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[framework_model, sparkml_model],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=True,
    )
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)

    sagemaker_session.create_model.assert_called_with(
        model.name,
        ROLE,
        [
            {
                "Image": "mi-1",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "blah.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/mi-1-2017-10-10-14-14-15/sourcedir.tar.gz",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": "us-west-2",
                },
                "ModelDataUrl": "s3://bucket/model_1.tar.gz",
            },
            {
                "Image": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-sparkml-serving:3.3",
                "Environment": {},
                "ModelDataUrl": "s3://bucket/model_2.tar.gz",
            },
        ],
        vpc_config=None,
        enable_network_isolation=True,
    )
