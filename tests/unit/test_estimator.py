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

import logging
import json
import os
from time import sleep

import pytest
from mock import Mock, patch

from sagemaker.estimator import Estimator, Framework, _TrainingJob
from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import s3_input
from sagemaker.transformer import Transformer

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_NAME = 'dummy_script.py'
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = '2017-11-06-14:14:15.671'
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'c4.4xlarge'
ROLE = 'DummyRole'
IMAGE_NAME = 'fakeimage'
REGION = 'us-west-2'
JOB_NAME = '{}-{}'.format(IMAGE_NAME, TIMESTAMP)
TAGS = [{'Name': 'some-tag', 'Value': 'value-for-tag'}]
OUTPUT_PATH = 's3://bucket/prefix'

COMMON_TRAIN_ARGS = {
    'volume_size': 30,
    'hyperparameters': {
        'sagemaker_program': 'dummy_script.py',
        'sagemaker_enable_cloudwatch_metrics': False,
        'sagemaker_container_log_level': logging.INFO,
    },
    'input_mode': 'File',
    'instance_type': 'c4.4xlarge',
    'inputs': 's3://mybucket/train',
    'instance_count': 1,
    'role': 'DummyRole',
    'kms_key_id': None,
    'max_run': 24,
    'wait': True,
}

DESCRIBE_TRAINING_JOB_RESULT = {
    'ModelArtifacts': {
        'S3ModelArtifacts': MODEL_DATA
    }
}

MODEL_CONTAINER_DEF = {
    'Environment': {
        'SAGEMAKER_PROGRAM': ENTRY_POINT,
        'SAGEMAKER_SUBMIT_DIRECTORY': 's3://mybucket/mi-2017-10-10-14-14-15/sourcedir.tar.gz',
        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
        'SAGEMAKER_REGION': REGION,
        'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false'
    },
    'Image': MODEL_IMAGE,
    'ModelDataUrl': MODEL_DATA,
}


class DummyFramework(Framework):
    __framework_name__ = 'dummy'

    def train_image(self):
        return IMAGE_NAME

    def create_model(self, role=None, model_server_workers=None):
        return DummyFrameworkModel(self.sagemaker_session)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        init_params = super(DummyFramework, cls)._prepare_init_params_from_job_description(job_details)
        init_params.pop("image", None)
        return init_params


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(MODEL_DATA, MODEL_IMAGE, INSTANCE_TYPE, ROLE, ENTRY_POINT,
                                                  sagemaker_session=sagemaker_session, **kwargs)

    def create_predictor(self, endpoint_name):
        return None

    def prepare_container_def(self, instance_type):
        return MODEL_CONTAINER_DEF


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock,
               boto_region_name=REGION, config=None, local_mode=False)
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                      return_value=DESCRIBE_TRAINING_JOB_RESULT)
    return sms


def test_sagemaker_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                           train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)
        t.fit('thisdoesntstartwiths3')
    assert 'must be a valid S3 or FILE URI' in str(error)


@patch('time.strftime', return_value=TIMESTAMP)
def test_custom_code_bucket(time, sagemaker_session):
    code_bucket = 'codebucket'
    prefix = 'someprefix'
    code_location = 's3://{}/{}'.format(code_bucket, prefix)
    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                       code_location=code_location)
    t.fit('s3://bucket/mydata')

    expected_key = '{}/{}/source/sourcedir.tar.gz'.format(prefix, JOB_NAME)
    _, s3_args, _ = sagemaker_session.boto_session.resource('s3').Object.mock_calls[0]
    assert s3_args == (code_bucket, expected_key)

    expected_submit_dir = 's3://{}/{}'.format(code_bucket, expected_key)
    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs['hyperparameters']['sagemaker_submit_directory'] == json.dumps(expected_submit_dir)


@patch('time.strftime', return_value=TIMESTAMP)
def test_custom_code_bucket_without_prefix(time, sagemaker_session):
    code_bucket = 'codebucket'
    code_location = 's3://{}'.format(code_bucket)
    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                       code_location=code_location)
    t.fit('s3://bucket/mydata')

    expected_key = '{}/source/sourcedir.tar.gz'.format(JOB_NAME)
    _, s3_args, _ = sagemaker_session.boto_session.resource('s3').Object.mock_calls[0]
    assert s3_args == (code_bucket, expected_key)

    expected_submit_dir = 's3://{}/{}'.format(code_bucket, expected_key)
    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs['hyperparameters']['sagemaker_submit_directory'] == json.dumps(expected_submit_dir)


def test_invalid_custom_code_bucket(sagemaker_session):
    code_location = 'thisllworkright?'
    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                       code_location=code_location)

    with pytest.raises(ValueError) as error:
        t.fit('s3://bucket/mydata')
    assert "Expecting 's3' scheme" in str(error)


BASE_HP = {
    'sagemaker_program': json.dumps(SCRIPT_NAME),
    'sagemaker_submit_directory': json.dumps('s3://mybucket/{}/source/sourcedir.tar.gz'.format(JOB_NAME)),
    'sagemaker_job_name': json.dumps(JOB_NAME)
}


def test_local_code_location():
    config = {
        'local': {
            'local_code': True,
            'region': 'us-west-2'
        }
    }
    sms = Mock(name='sagemaker_session', boto_session=None,
               boto_region_name=REGION, config=config, local_mode=True)
    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sms,
                       train_instance_count=1, train_instance_type='local',
                       base_job_name=IMAGE_NAME, hyperparameters={123: [456], 'learning_rate': 0.1})

    t.fit('file:///data/file')
    assert t.source_dir == DATA_DIR
    assert t.entry_point == 'dummy_script.py'


@patch('time.strftime', return_value=TIMESTAMP)
def test_start_new_convert_hyperparameters_to_str(strftime, sagemaker_session):
    uri = 'bucket/mydata'

    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                       base_job_name=IMAGE_NAME, hyperparameters={123: [456], 'learning_rate': 0.1})
    t.fit('s3://{}'.format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters['sagemaker_enable_cloudwatch_metrics'] = 'false'
    expected_hyperparameters['sagemaker_container_log_level'] = str(logging.INFO)
    expected_hyperparameters['learning_rate'] = json.dumps(0.1)
    expected_hyperparameters['123'] = json.dumps([456])
    expected_hyperparameters['sagemaker_region'] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]['hyperparameters']
    assert actual_hyperparameter == expected_hyperparameters


@patch('time.strftime', return_value=TIMESTAMP)
def test_start_new_wait_called(strftime, sagemaker_session):
    uri = 'bucket/mydata'

    t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    t.fit('s3://{}'.format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters['sagemaker_enable_cloudwatch_metrics'] = 'false'
    expected_hyperparameters['sagemaker_container_log_level'] = str(logging.INFO)
    expected_hyperparameters['sagemaker_region'] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]['hyperparameters']
    assert actual_hyperparameter == expected_hyperparameters
    assert sagemaker_session.wait_for_job.assert_called_once


def test_delete_endpoint(sagemaker_session):
    t = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                       container_log_level=logging.INFO)

    class tj(object):
        @property
        def name(self):
            return 'myjob'

    t.latest_training_job = tj()

    t.delete_endpoint()

    sagemaker_session.delete_endpoint.assert_called_with('myjob')


def test_delete_endpoint_without_endpoint(sagemaker_session):
    t = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    with pytest.raises(ValueError) as error:
        t.delete_endpoint()
    assert 'Endpoint was not created yet' in str(error)


def test_enable_cloudwatch_metrics(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=True)
    fw.fit(inputs=s3_input('s3://mybucket/train'))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs['hyperparameters']['sagemaker_enable_cloudwatch_metrics']


def test_attach_framework(sagemaker_session):
    returned_job_description = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other-py2-cpu:1.0.4',
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
                                 'sagemaker_job_name': '"neo"',
            'training_steps': '100',
        },
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge',
        },
        'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {
            'KmsKeyId': '',
            'S3OutputPath': 's3://place/output/neo',
        },
        'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'},
    }
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    framework_estimator = DummyFramework.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert framework_estimator.latest_training_job.job_name == 'neo'
    assert framework_estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert framework_estimator.train_instance_count == 1
    assert framework_estimator.train_max_run == 24 * 60 * 60
    assert framework_estimator.input_mode == 'File'
    assert framework_estimator.base_job_name == 'neo'
    assert framework_estimator.output_path == 's3://place/output/neo'
    assert framework_estimator.output_kms_key == ''
    assert framework_estimator.hyperparameters()['training_steps'] == '100'
    assert framework_estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert framework_estimator.entry_point == 'iris-dnn-classifier.py'


def test_attach_framework_with_tuning(sagemaker_session):
    returned_job_description = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other-py2-cpu:1.0.4'
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            '_tuning_objective_metric': 'Validation-accuracy',
        },

        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        },
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {
            'KmsKeyId': '',
            'S3OutputPath': 's3://place/output/neo'
        },
        'TrainingJobOutput': {
            'S3TrainingJobOutput': 's3://here/output.tar.gz'
        }
    }

    mock_describe_training_job = Mock(name='describe_training_job',
                                      return_value=returned_job_description)
    sagemaker_session.sagemaker_client.describe_training_job = mock_describe_training_job

    framework_estimator = DummyFramework.attach(training_job_name='neo',
                                                sagemaker_session=sagemaker_session)
    assert framework_estimator.latest_training_job.job_name == 'neo'
    assert framework_estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert framework_estimator.train_instance_count == 1
    assert framework_estimator.train_max_run == 24 * 60 * 60
    assert framework_estimator.input_mode == 'File'
    assert framework_estimator.base_job_name == 'neo'
    assert framework_estimator.output_path == 's3://place/output/neo'
    assert framework_estimator.output_kms_key == ''
    hyper_params = framework_estimator.hyperparameters()
    assert hyper_params['training_steps'] == '100'
    assert hyper_params['_tuning_objective_metric'] == '"Validation-accuracy"'
    assert framework_estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert framework_estimator.entry_point == 'iris-dnn-classifier.py'


@patch('time.strftime', return_value=TIMESTAMP)
def test_fit_verify_job_name(strftime, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=True, tags=TAGS)
    fw.fit(inputs=s3_input('s3://mybucket/train'))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]

    assert train_kwargs['hyperparameters']['sagemaker_enable_cloudwatch_metrics']
    assert train_kwargs['image'] == IMAGE_NAME
    assert train_kwargs['input_mode'] == 'File'
    assert train_kwargs['tags'] == TAGS
    assert train_kwargs['job_name'] == JOB_NAME
    assert fw.latest_training_job.name == JOB_NAME


def test_prepare_for_training_unique_job_name_generation(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=True)
    fw._prepare_for_training()
    first_job_name = fw._current_job_name

    sleep(0.1)
    fw._prepare_for_training()
    second_job_name = fw._current_job_name

    assert first_job_name != second_job_name


def test_prepare_for_training_force_name(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        base_job_name='some', enable_cloudwatch_metrics=True)
    fw._prepare_for_training(job_name='use_it')
    assert 'use_it' == fw._current_job_name


@patch('time.strftime', return_value=TIMESTAMP)
def test_prepare_for_training_force_name_generation(strftime, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        base_job_name='some', enable_cloudwatch_metrics=True)
    fw.base_job_name = None
    fw._prepare_for_training()
    assert JOB_NAME == fw._current_job_name


@patch('time.strftime', return_value=TIMESTAMP)
def test_init_with_source_dir_s3(strftime, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, source_dir='s3://location', role=ROLE,
                        sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=False)
    fw._prepare_for_training()

    expected_hyperparameters = {
        'sagemaker_program': SCRIPT_NAME,
        'sagemaker_job_name': JOB_NAME,
        'sagemaker_enable_cloudwatch_metrics': False,
        'sagemaker_container_log_level': logging.INFO,
        'sagemaker_submit_directory': 's3://location',
        'sagemaker_region': 'us-west-2',
    }
    assert fw._hyperparameters == expected_hyperparameters


@patch('sagemaker.estimator.name_from_image', return_value=MODEL_IMAGE)
def test_framework_transformer_creation(name_from_image, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, train_instance_count=INSTANCE_COUNT,
                        train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session)
    fw.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    transformer = fw.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    name_from_image.assert_called_with(MODEL_IMAGE)
    sagemaker_session.create_model.assert_called_with(MODEL_IMAGE, ROLE, MODEL_CONTAINER_DEF)

    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == MODEL_IMAGE
    assert transformer.tags is None
    assert transformer.env == {}


@patch('sagemaker.estimator.name_from_image', return_value=MODEL_IMAGE)
def test_framework_transformer_creation_with_optional_params(name_from_image, sagemaker_session):
    base_name = 'foo'
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, train_instance_count=INSTANCE_COUNT,
                        train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session,
                        base_job_name=base_name)
    fw.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    strategy = 'MultiRecord'
    assemble_with = 'Line'
    kms_key = 'key'
    accept = 'text/csv'
    max_concurrent_transforms = 1
    max_payload = 6
    env = {'FOO': 'BAR'}
    new_role = 'dummy-model-role'

    transformer = fw.transformer(INSTANCE_COUNT, INSTANCE_TYPE, strategy=strategy, assemble_with=assemble_with,
                                 output_path=OUTPUT_PATH, output_kms_key=kms_key, accept=accept, tags=TAGS,
                                 max_concurrent_transforms=max_concurrent_transforms, max_payload=max_payload,
                                 env=env, role=new_role, model_server_workers=1)

    sagemaker_session.create_model.assert_called_with(MODEL_IMAGE, new_role, MODEL_CONTAINER_DEF)
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == OUTPUT_PATH
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.base_transform_job_name == base_name
    assert transformer.tags == TAGS


def test_ensure_latest_training_job(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, train_instance_count=INSTANCE_COUNT,
                        train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session)
    fw.latest_training_job = Mock(name='training_job')

    fw._ensure_latest_training_job()


def test_ensure_latest_training_job_failure(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, train_instance_count=INSTANCE_COUNT,
                        train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError) as e:
        fw._ensure_latest_training_job()
    assert 'Estimator is not associated with a training job' in str(e)


def test_estimator_transformer_creation(sagemaker_session):
    estimator = Estimator(image_name=IMAGE_NAME, role=ROLE, train_instance_count=INSTANCE_COUNT,
                          train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session)
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    sagemaker_session.create_model_from_job.return_value = JOB_NAME

    transformer = estimator.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.create_model_from_job.assert_called_with(JOB_NAME, role=None)
    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == JOB_NAME
    assert transformer.tags is None


def test_estimator_transformer_creation_with_optional_params(sagemaker_session):
    base_name = 'foo'
    estimator = Estimator(image_name=IMAGE_NAME, role=ROLE, train_instance_count=INSTANCE_COUNT,
                          train_instance_type=INSTANCE_TYPE, sagemaker_session=sagemaker_session,
                          base_job_name=base_name)
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    sagemaker_session.create_model_from_job.return_value = JOB_NAME

    strategy = 'MultiRecord'
    assemble_with = 'Line'
    kms_key = 'key'
    accept = 'text/csv'
    max_concurrent_transforms = 1
    max_payload = 6
    env = {'FOO': 'BAR'}

    transformer = estimator.transformer(INSTANCE_COUNT, INSTANCE_TYPE, strategy=strategy, assemble_with=assemble_with,
                                        output_path=OUTPUT_PATH, output_kms_key=kms_key, accept=accept, tags=TAGS,
                                        max_concurrent_transforms=max_concurrent_transforms, max_payload=max_payload,
                                        env=env, role=ROLE)

    sagemaker_session.create_model_from_job.assert_called_with(JOB_NAME, role=ROLE)
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == OUTPUT_PATH
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.base_transform_job_name == base_name
    assert transformer.tags == TAGS


# _TrainingJob 'utils'
def test_start_new(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    hyperparameters = {'mock': 'hyperparameters'}
    inputs = 's3://mybucket/train'

    estimator = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE,
                          output_path=OUTPUT_PATH, sagemaker_session=sagemaker_session,
                          hyperparameters=hyperparameters)

    started_training_job = training_job.start_new(estimator, inputs)
    called_args = sagemaker_session.train.call_args

    assert started_training_job.sagemaker_session == sagemaker_session
    assert called_args[1]['hyperparameters'] == hyperparameters
    sagemaker_session.train.assert_called_once()


def test_start_new_not_local_mode_error(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    inputs = 'file://mybucket/train'

    estimator = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE,
                          output_path=OUTPUT_PATH, sagemaker_session=sagemaker_session)
    with pytest.raises(ValueError) as error:
        training_job.start_new(estimator, inputs)
        assert 'File URIs are supported in local mode only. Please use a S3 URI instead.' == str(error)


def test_container_log_level(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        container_log_level=logging.DEBUG)
    fw.fit(inputs=s3_input('s3://mybucket/train'))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs['hyperparameters']['sagemaker_container_log_level'] == '10'


def test_wait_without_logs(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    training_job.wait(False)

    sagemaker_session.wait_for_job.assert_called_once()
    assert not sagemaker_session.logs_for_job.called


def test_wait_with_logs(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    training_job.wait()

    sagemaker_session.logs_for_job.assert_called_once()
    assert not sagemaker_session.wait_for_job.called


def test_unsupported_type_in_dict():
    with pytest.raises(ValueError):
        _TrainingJob._format_inputs_to_input_config({'a': 66})


#################################################################################
# Tests for the generic Estimator class

BASE_TRAIN_CALL = {
    'hyperparameters': {},
    'image': IMAGE_NAME,
    'input_config': [{
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://bucket/training-prefix'
            }
        },
        'ChannelName': 'train'
    }],
    'input_mode': 'File',
    'output_config': {'S3OutputPath': OUTPUT_PATH},
    'resource_config': {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
        'VolumeSizeInGB': 30
    },
    'stop_condition': {'MaxRuntimeInSeconds': 86400},
    'tags': None,
    'vpc_config': {'SecurityGroupIds': None, 'Subnets': None}
}

HYPERPARAMS = {'x': 1, 'y': 'hello'}
STRINGIFIED_HYPERPARAMS = dict([(x, str(y)) for x, y in HYPERPARAMS.items()])
HP_TRAIN_CALL = dict(BASE_TRAIN_CALL)
HP_TRAIN_CALL.update({'hyperparameters': STRINGIFIED_HYPERPARAMS})


def test_generic_to_fit_no_hps(sagemaker_session):
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path=OUTPUT_PATH,
                  sagemaker_session=sagemaker_session)

    e.fit({'train': 's3://bucket/training-prefix'})

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args['job_name'].startswith(IMAGE_NAME)

    args.pop('job_name')
    args.pop('role')

    assert args == BASE_TRAIN_CALL


def test_generic_to_fit_with_hps(sagemaker_session):
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path=OUTPUT_PATH,
                  sagemaker_session=sagemaker_session)

    e.set_hyperparameters(**HYPERPARAMS)

    e.fit({'train': 's3://bucket/training-prefix'})

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args['job_name'].startswith(IMAGE_NAME)

    args.pop('job_name')
    args.pop('role')

    assert args == HP_TRAIN_CALL


def test_generic_to_deploy(sagemaker_session):
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path=OUTPUT_PATH,
                  sagemaker_session=sagemaker_session)

    e.set_hyperparameters(**HYPERPARAMS)

    e.fit({'train': 's3://bucket/training-prefix'})

    predictor = e.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args['job_name'].startswith(IMAGE_NAME)

    args.pop('job_name')
    args.pop('role')

    assert args == HP_TRAIN_CALL

    sagemaker_session.create_model.assert_called_once()
    args = sagemaker_session.create_model.call_args[0]
    assert args[0].startswith(IMAGE_NAME)
    assert args[1] == ROLE
    assert args[2]['Image'] == IMAGE_NAME
    assert args[2]['ModelDataUrl'] == MODEL_DATA

    assert isinstance(predictor, RealTimePredictor)
    assert predictor.endpoint.startswith(IMAGE_NAME)
    assert predictor.sagemaker_session == sagemaker_session


def test_generic_training_job_analytics(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job', return_value={
        'TuningJobArn': 'arn:aws:sagemaker:us-west-2:968277160000:hyper-parameter-tuning-job/mock-tuner',
        'TrainingStartTime': 1530562991.299,
    })
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job',
        return_value={
            'TrainingJobDefinition': {
                "AlgorithmSpecification": {
                    "TrainingImage": "some-image-url",
                    "TrainingInputMode": "File",
                    "MetricDefinitions": [
                        {
                            "Name": "train:loss",
                            "Regex": "train_loss=([0-9]+\\.[0-9]+)"
                        },
                        {
                            "Name": "validation:loss",
                            "Regex": "valid_loss=([0-9]+\\.[0-9]+)"
                        }
                    ]
                }
            }
        }
    )

    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path=OUTPUT_PATH,
                  sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError) as err:  # noqa: F841
        # No training job yet
        a = e.training_job_analytics
        assert a is not None  # This line is never reached

    e.set_hyperparameters(**HYPERPARAMS)
    e.fit({'train': 's3://bucket/training-prefix'})
    a = e.training_job_analytics
    assert a is not None


@patch('sagemaker.estimator.LocalSession')
@patch('sagemaker.estimator.Session')
def test_local_mode(session_class, local_session_class):
    local_session = Mock()
    local_session.local_mode = True

    session = Mock()
    session.local_mode = False

    local_session_class.return_value = local_session
    session_class.return_value = session

    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, 'local')
    print(e.sagemaker_session.local_mode)
    assert e.sagemaker_session.local_mode is True

    e2 = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, 'local_gpu')
    assert e2.sagemaker_session.local_mode is True

    e3 = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE)
    assert e3.sagemaker_session.local_mode is False


@patch('sagemaker.estimator.LocalSession')
def test_distributed_gpu_local_mode(LocalSession):
    with pytest.raises(RuntimeError):
        Estimator(IMAGE_NAME, ROLE, 3, 'local_gpu', output_path=OUTPUT_PATH)

#################################################################################
