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
from mock import Mock

from sagemaker.fw_registry import registry
from sagemaker.sparkml import SparkMLModel, SparkMLPredictor

MODEL_DATA = "s3://bucket/model.tar.gz"
ROLE = "myrole"
TRAIN_INSTANCE_TYPE = "ml.c4.xlarge"

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
ENDPOINT = "some-endpoint"

ENDPOINT_DESC = {"EndpointConfigName": ENDPOINT}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        region_name=REGION,
        config=None,
        local_mode=False,
    )
    sms.boto_region_name = REGION
    sms.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    sms.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    return sms


def test_sparkml_model(sagemaker_session):
    sparkml = SparkMLModel(sagemaker_session=sagemaker_session, model_data=MODEL_DATA, role=ROLE)
    assert sparkml.image == registry(REGION, "sparkml-serving") + "/sagemaker-sparkml-serving:2.2"


def test_predictor_type(sagemaker_session):
    sparkml = SparkMLModel(sagemaker_session=sagemaker_session, model_data=MODEL_DATA, role=ROLE)
    predictor = sparkml.deploy(1, TRAIN_INSTANCE_TYPE)

    assert isinstance(predictor, SparkMLPredictor)
