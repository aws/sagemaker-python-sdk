# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor
import os
import pytest
from mock import Mock, patch

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"
INSTANCE_TYPE = "p2.xlarge"
ROLE = "some-role"

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_NAME = 'dummy_script.py'
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = '2017-10-10-14-14-15'
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'c4.4xlarge'
IMAGE_NAME = 'fakeimage'
REGION = 'us-west-2'


class DummyFrameworkModel(FrameworkModel):

    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(MODEL_DATA, MODEL_IMAGE, ROLE, ENTRY_POINT,
                                                  sagemaker_session=sagemaker_session, **kwargs)

    def create_predictor(self, endpoint_name):
        return RealTimePredictor(endpoint_name, self.sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    return ims


@patch('tarfile.open')
@patch('time.strftime', return_value=TIMESTAMP)
def test_prepare_container_def(tfopen, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    assert model.prepare_container_def(INSTANCE_TYPE) == {
        'Environment': {'SAGEMAKER_PROGRAM': 'blah.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': 's3://mybucket/mi-2017-10-10-14-14-15/sourcedir.tar.gz',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': 'us-west-2',
                        'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false'},
        'Image': 'mi',
        'ModelDataUrl': 's3://bucket/model.tar.gz'}


@patch('tarfile.open')
@patch('os.path.exists', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.listdir', return_value=['blah.py'])
@patch('time.strftime', return_value=TIMESTAMP)
def test_create_no_defaults(tfopen, exists, isdir, listdir, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session, source_dir="sd", env={"a": "a"}, name="name",
                                enable_cloudwatch_metrics=True, container_log_level=55,
                                code_location="s3://cb/cp")

    assert model.prepare_container_def(INSTANCE_TYPE) == {
        'Environment': {'SAGEMAKER_PROGRAM': 'blah.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': 's3://cb/cp/sourcedir.tar.gz',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '55',
                        'SAGEMAKER_REGION': 'us-west-2',
                        'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'true',
                        'a': 'a'},
        'Image': 'mi',
        'ModelDataUrl': 's3://bucket/model.tar.gz'}


@patch('tarfile.open')
@patch('time.strftime', return_value=TIMESTAMP)
def test_deploy(tfo, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        'mi-2017-10-10-14-14-15',
        [{'InitialVariantWeight': 1,
          'ModelName': 'mi-2017-10-10-14-14-15',
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 1,
          'VariantName': 'AllTraffic'}])


@patch('tarfile.open')
@patch('time.strftime', return_value=TIMESTAMP)
def test_deploy_endpoint_name(tfo, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    model.deploy(endpoint_name='blah', instance_type=INSTANCE_TYPE, initial_instance_count=55)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        'blah',
        [{'InitialVariantWeight': 1,
          'ModelName': 'mi-2017-10-10-14-14-15',
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 55,
          'VariantName': 'AllTraffic'}])
