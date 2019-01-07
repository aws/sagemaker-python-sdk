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

import copy
import os

import sagemaker
from sagemaker.model import FrameworkModel, ModelPackage
from sagemaker.predictor import RealTimePredictor

import pytest
from mock import MagicMock, Mock, patch

MODEL_DATA = 's3://bucket/model.tar.gz'
MODEL_IMAGE = 'mi'
ENTRY_POINT = 'blah.py'
INSTANCE_TYPE = 'p2.xlarge'
ROLE = 'some-role'

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_NAME = 'dummy_script.py'
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = '2017-10-10-14-14-15'
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'c4.4xlarge'
ACCELERATOR_TYPE = 'ml.eia.medium'
IMAGE_NAME = 'fakeimage'
REGION = 'us-west-2'
MODEL_NAME = '{}-{}'.format(MODEL_IMAGE, TIMESTAMP)


DESCRIBE_MODEL_PACKAGE_RESPONSE = {
    'InferenceSpecification': {
        'SupportedResponseMIMETypes': [
            'text'
        ],
        'SupportedContentTypes': [
            'text/csv'
        ],
        'SupportedTransformInstanceTypes': [
            'ml.m4.xlarge',
            'ml.m4.2xlarge'
        ],
        'Containers': [
            {
                'Image': '1.dkr.ecr.us-east-2.amazonaws.com/decision-trees-sample:latest',
                'ImageDigest': 'sha256:1234556789',
                'ModelDataUrl': 's3://bucket/output/model.tar.gz'
            }
        ],
        'SupportedRealtimeInferenceInstanceTypes': [
            'ml.m4.xlarge',
            'ml.m4.2xlarge',

        ]
    },
    'ModelPackageDescription': 'Model Package created from training with '
                               'arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees',
    'CreationTime': 1542752036.687,
    'ModelPackageArn': 'arn:aws:sagemaker:us-east-2:123:model-package/mp-scikit-decision-trees',
    'ModelPackageStatusDetails': {
        'ValidationStatuses': [],
        'ImageScanStatuses': []
    },
    'SourceAlgorithmSpecification': {
        'SourceAlgorithms': [
            {
                'ModelDataUrl': 's3://bucket/output/model.tar.gz',
                'AlgorithmName': 'arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees'
            }
        ]
    },

    'ModelPackageStatus': 'Completed',
    'ModelPackageName': 'mp-scikit-decision-trees-1542410022-2018-11-20-22-13-56-502',
    'CertifyForMarketplace': False
}


class DummyFrameworkModel(FrameworkModel):

    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(MODEL_DATA, MODEL_IMAGE, ROLE, ENTRY_POINT,
                                                  sagemaker_session=sagemaker_session, **kwargs)

    def create_predictor(self, endpoint_name):
        return RealTimePredictor(endpoint_name, self.sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock,
               boto_region_name=REGION, config=None, local_mode=False)
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    return sms


@patch('shutil.rmtree', MagicMock())
@patch('tarfile.open', MagicMock())
@patch('os.listdir', MagicMock(return_value=['blah.py']))
@patch('time.strftime', return_value=TIMESTAMP)
def test_prepare_container_def(time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    assert model.prepare_container_def(INSTANCE_TYPE) == {
        'Environment': {'SAGEMAKER_PROGRAM': ENTRY_POINT,
                        'SAGEMAKER_SUBMIT_DIRECTORY': 's3://mybucket/mi-2017-10-10-14-14-15/sourcedir.tar.gz',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': REGION,
                        'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false'},
        'Image': MODEL_IMAGE,
        'ModelDataUrl': MODEL_DATA}


@patch('shutil.rmtree', MagicMock())
@patch('tarfile.open', MagicMock())
@patch('os.path.exists', MagicMock(return_value=True))
@patch('os.path.isdir', MagicMock(return_value=True))
@patch('os.listdir', MagicMock(return_value=['blah.py']))
@patch('time.strftime', MagicMock(return_value=TIMESTAMP))
def test_create_no_defaults(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir='sd', env={"a": "a"}, name="name",
                                enable_cloudwatch_metrics=True, container_log_level=55,
                                code_location="s3://cb/cp")

    assert model.prepare_container_def(INSTANCE_TYPE) == {
        'Environment': {'SAGEMAKER_PROGRAM': ENTRY_POINT,
                        'SAGEMAKER_SUBMIT_DIRECTORY': 's3://cb/cp/name/sourcedir.tar.gz',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '55',
                        'SAGEMAKER_REGION': REGION,
                        'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'true',
                        'a': 'a'},
        'Image': MODEL_IMAGE,
        'ModelDataUrl': MODEL_DATA}


@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
@patch('time.strftime', MagicMock(return_value=TIMESTAMP))
def test_deploy(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [{'InitialVariantWeight': 1,
          'ModelName': MODEL_NAME,
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 1,
          'VariantName': 'AllTraffic'}],
        None)


@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
@patch('time.strftime', MagicMock(return_value=TIMESTAMP))
def test_deploy_endpoint_name(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.deploy(endpoint_name='blah', instance_type=INSTANCE_TYPE, initial_instance_count=55)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        'blah',
        [{'InitialVariantWeight': 1,
          'ModelName': MODEL_NAME,
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 55,
          'VariantName': 'AllTraffic'}],
        None)


@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
@patch('time.strftime', MagicMock(return_value=TIMESTAMP))
def test_deploy_tags(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    tags = [{'ModelName': 'TestModel'}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, tags=tags)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [{'InitialVariantWeight': 1,
          'ModelName': MODEL_NAME,
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 1,
          'VariantName': 'AllTraffic'}],
        tags)


@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
@patch('tarfile.open')
@patch('time.strftime', return_value=TIMESTAMP)
def test_deploy_accelerator_type(tfo, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, accelerator_type=ACCELERATOR_TYPE)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [{'InitialVariantWeight': 1,
          'ModelName': MODEL_NAME,
          'InstanceType': INSTANCE_TYPE,
          'InitialInstanceCount': 1,
          'VariantName': 'AllTraffic',
          'AcceleratorType': ACCELERATOR_TYPE}],
        None)


@patch('sagemaker.session.Session')
@patch('sagemaker.local.LocalSession')
@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
def test_deploy_creates_correct_session(local_session, session, tmpdir):
    # We expect a LocalSession when deploying to instance_type = 'local'
    model = DummyFrameworkModel(sagemaker_session=None, source_dir=str(tmpdir))
    model.deploy(endpoint_name='blah', instance_type='local', initial_instance_count=1)
    assert model.sagemaker_session == local_session.return_value

    # We expect a real Session when deploying to instance_type != local/local_gpu
    model = DummyFrameworkModel(sagemaker_session=None, source_dir=str(tmpdir))
    model.deploy(endpoint_name='remote_endpoint', instance_type='ml.m4.4xlarge', initial_instance_count=2)
    assert model.sagemaker_session == session.return_value


def test_model_enable_network_isolation(sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session=sagemaker_session)
    assert model.enable_network_isolation() is False


@patch('sagemaker.model.Model._create_sagemaker_model', Mock())
def test_model_create_transformer(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE)

    model = DummyFrameworkModel(sagemaker_session=sagemaker_session)
    model.name = 'auto-generated-model'
    transformer = model.transformer(instance_count=1, instance_type='ml.m4.xlarge',
                                    env={'test': True})
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == 'auto-generated-model'
    assert transformer.instance_type == 'ml.m4.xlarge'
    assert transformer.env == {'test': True}


def test_model_package_enable_network_isolation_with_no_product_id(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE)

    model_package = ModelPackage(role='role', model_package_arn='my-model-package',
                                 sagemaker_session=sagemaker_session)
    assert model_package.enable_network_isolation() is False


def test_model_package_enable_network_isolation_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response['InferenceSpecification']['Containers'].append(
        {
            'Image': '1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest',
            'ModelDataUrl': 's3://bucket/output/model.tar.gz',
            'ProductId': 'some-product-id'
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response)

    model_package = ModelPackage(role='role', model_package_arn='my-model-package',
                                 sagemaker_session=sagemaker_session)
    assert model_package.enable_network_isolation() is True


@patch('sagemaker.model.ModelPackage._create_sagemaker_model', Mock())
def test_model_package_create_transformer(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE)

    model_package = ModelPackage(role='role', model_package_arn='my-model-package',
                                 sagemaker_session=sagemaker_session)
    model_package.name = 'auto-generated-model'
    transformer = model_package.transformer(instance_count=1, instance_type='ml.m4.xlarge',
                                            env={'test': True})
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == 'auto-generated-model'
    assert transformer.instance_type == 'ml.m4.xlarge'
    assert transformer.env == {'test': True}


@patch('sagemaker.model.ModelPackage._create_sagemaker_model', Mock())
def test_model_package_create_transformer_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response['InferenceSpecification']['Containers'].append(
        {
            'Image': '1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest',
            'ModelDataUrl': 's3://bucket/output/model.tar.gz',
            'ProductId': 'some-product-id'
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response)

    model_package = ModelPackage(role='role', model_package_arn='my-model-package',
                                 sagemaker_session=sagemaker_session)
    model_package.name = 'auto-generated-model'
    transformer = model_package.transformer(instance_count=1, instance_type='ml.m4.xlarge',
                                            env={'test': True})
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == 'auto-generated-model'
    assert transformer.instance_type == 'ml.m4.xlarge'
    assert transformer.env is None
