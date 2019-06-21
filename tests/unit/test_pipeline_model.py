# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mock import Mock, patch

from sagemaker.model import FrameworkModel
from sagemaker.pipeline import PipelineModel
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import ModelContainer
from sagemaker.sparkml import SparkMLModel

ENTRY_POINT = "blah.py"
MODEL_DATA_1 = "s3://bucket/model_1.tar.gz"
MODEL_DATA_2 = "s3://bucket/model_2.tar.gz"
MODEL_IMAGE_1 = "mi-1"
MODEL_IMAGE_2 = "mi-2"
INSTANCE_TYPE = "ml.m4.xlarge"
ROLE = "some-role"
ENV_1 = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json"}
ENV_2 = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"}
MODEL_CONTAINER_1 = ModelContainer(image=MODEL_IMAGE_1, model_data=MODEL_DATA_1, env=ENV_1)
MODEL_CONTAINER_2 = ModelContainer(image=MODEL_IMAGE_2, model_data=MODEL_DATA_2, env=ENV_2)
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
            **kwargs
        )

    def create_predictor(self, endpoint_name):
        return RealTimePredictor(endpoint_name, self.sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
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
                "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
            },
            "Image": "mi-1",
            "ModelDataUrl": "s3://bucket/model_1.tar.gz",
        },
        {
            "Environment": {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
            "Image": "246618743249.dkr.ecr.us-west-2.amazonaws.com"
            + "/sagemaker-sparkml-serving:2.2",
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
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        "mi-1-2017-10-10-14-14-15",
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        None,
        wait=True,
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
        "mi-1-2017-10-10-14-14-15",
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        None,
        wait=True,
    )


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
        "mi-1-2017-10-10-14-14-15",
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": "mi-1-2017-10-10-14-14-15",
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags,
        wait=True,
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
