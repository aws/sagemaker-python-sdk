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
import logging
import json
import os
import pytest
from mock import Mock, patch

from sagemaker.estimator import Estimator, Framework, _TrainingJob
from sagemaker.session import s3_input
from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor

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

COMMON_TRAIN_ARGS = {'volume_size': 30,
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
                     'wait': True}

DESCRIBE_TRAINING_JOB_RESULT = {
    'ModelArtifacts': {
        'S3ModelArtifacts': MODEL_DATA
    }
}


class DummyFramework(Framework):
    __framework_name__ = 'dummy'

    def train_image(self):
        return IMAGE_NAME

    def create_model(self):
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


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                      return_value=DESCRIBE_TRAINING_JOB_RESULT)
    return ims


def test_sagemaker_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                           train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)
        t.fit('thisdoesntstartwiths3')
    assert 'must be a valid S3 URI' in str(error)


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
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other-py2-cpu:1.0.4'},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'checkpoint_path': '"s3://other/1508872349"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
                                     'sagemaker_job_name': '"neo"',
                                     'training_steps': '100'},
                                'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
                                'ResourceConfig':
                                    {'VolumeSizeInGB': 30,
                                     'InstanceCount': 1,
                                     'InstanceType': 'ml.c4.xlarge'},
                                'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
                                'TrainingJobName': 'neo',
                                'TrainingJobStatus': 'Completed',
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
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


def test_fit_then_fit_again(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=True)
    fw.fit(inputs=s3_input('s3://mybucket/train'))
    first_job_name = fw.latest_training_job.name

    fw.fit(inputs=s3_input('s3://mybucket/train2'))
    second_job_name = fw.latest_training_job.name

    assert first_job_name != second_job_name


@patch('time.strftime', return_value=TIMESTAMP)
def test_fit_verify_job_name(strftime, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=True)
    fw.fit(inputs=s3_input('s3://mybucket/train'))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]

    assert train_kwargs['hyperparameters']['sagemaker_enable_cloudwatch_metrics']
    assert train_kwargs['image'] == IMAGE_NAME
    assert train_kwargs['input_mode'] == 'File'
    assert train_kwargs['job_name'] == JOB_NAME
    assert fw.latest_training_job.name == JOB_NAME


def test_fit_force_name(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        base_job_name='some', enable_cloudwatch_metrics=True)
    fw.fit(inputs=s3_input('s3://mybucket/train'), job_name='use_it')
    assert 'use_it' == fw.latest_training_job.name


@patch('time.strftime', return_value=TIMESTAMP)
def test_fit_force_generation(strftime, sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        base_job_name='some', enable_cloudwatch_metrics=True)
    fw.base_job_name = None
    fw.fit(inputs=s3_input('s3://mybucket/train'))
    assert JOB_NAME == fw.latest_training_job.name


@patch('time.strftime', return_value=TIMESTAMP)
def test_init_with_source_dir_s3(strftime, sagemaker_session):
    uri = 'bucket/mydata'

    fw = DummyFramework(entry_point=SCRIPT_PATH, source_dir='s3://location', role=ROLE,
                        sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        enable_cloudwatch_metrics=False)
    fw.fit('s3://{}'.format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters['sagemaker_enable_cloudwatch_metrics'] = 'false'
    expected_hyperparameters['sagemaker_container_log_level'] = str(logging.INFO)
    expected_hyperparameters['sagemaker_submit_directory'] = json.dumps("s3://location")
    expected_hyperparameters['sagemaker_region'] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]['hyperparameters']
    assert actual_hyperparameter == expected_hyperparameters


# _TrainingJob 'utils'
def test_format_input_single_unamed_channel():
    input_dict = _TrainingJob._format_inputs_to_input_config('s3://blah/blah')
    assert input_dict == [{
        'ChannelName': 'training',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://blah/blah'
            }
        }
    }]


def test_container_log_level(sagemaker_session):
    fw = DummyFramework(entry_point=SCRIPT_PATH, role='DummyRole', sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        container_log_level=logging.DEBUG)
    fw.fit(inputs=s3_input('s3://mybucket/train'))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs['hyperparameters']['sagemaker_container_log_level'] == '10'


def test_format_input_multiple_channels():
    input_list = _TrainingJob._format_inputs_to_input_config({'a': 's3://blah/blah', 'b': 's3://foo/bar'})
    expected = [{
        'ChannelName': 'a',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://blah/blah'
            }
        }
    },
        {
            'ChannelName': 'b',
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://foo/bar'
                }
            }
    }]

    # convert back into map for comparison so list order (which is arbitrary) is ignored
    assert {c['ChannelName']: c for c in input_list} == {c['ChannelName']: c for c in expected}


def test_format_input_s3_input():
    input_dict = _TrainingJob._format_inputs_to_input_config(s3_input('s3://foo/bar', distribution='ShardedByS3Key',
                                                                      compression='gzip', content_type='whizz',
                                                                      record_wrapping='bang'))
    assert input_dict == [{
        'CompressionType': 'gzip',
        'ChannelName': 'training',
        'ContentType': 'whizz',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3DataDistributionType': 'ShardedByS3Key',
                'S3Uri': 's3://foo/bar'}},
        'RecordWrapperType': 'bang'}]


def test_dict_of_mixed_input_types():
    input_list = _TrainingJob._format_inputs_to_input_config({
        'a': 's3://foo/bar',
        'b': s3_input('s3://whizz/bang')})

    expected = [
        {'ChannelName': 'a',
         'DataSource': {
             'S3DataSource': {
                 'S3DataDistributionType': 'FullyReplicated',
                 'S3DataType': 'S3Prefix',
                 'S3Uri': 's3://foo/bar'
             }
         }
         },
        {
            'ChannelName': 'b',
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://whizz/bang'
                }
            }
        }]

    # convert back into map for comparison so list order (which is arbitrary) is ignored
    assert {c['ChannelName']: c for c in input_list} == {c['ChannelName']: c for c in expected}


def test_unsupported_type():
    with pytest.raises(ValueError):
        _TrainingJob._format_inputs_to_input_config(55)


def test_unsupported_type_in_dict():
    with pytest.raises(ValueError) as error:
        _TrainingJob._format_inputs_to_input_config({'a': 66})
    assert 'Expecting one of str or s3_input' in str(error)


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
    'output_config': {'S3OutputPath': 's3://bucket/prefix'},
    'resource_config': {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
        'VolumeSizeInGB': 30
    },
    'stop_condition': {'MaxRuntimeInSeconds': 86400}
}


HYPERPARAMS = {'x': 1, 'y': 'hello'}
STRINGIFIED_HYPERPARAMS = dict([(x, str(y)) for x, y in HYPERPARAMS.items()])
HP_TRAIN_CALL = dict(BASE_TRAIN_CALL)
HP_TRAIN_CALL.update({'hyperparameters': STRINGIFIED_HYPERPARAMS})


def test_generic_to_fit_no_hps(sagemaker_session):
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path='s3://bucket/prefix',
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
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path='s3://bucket/prefix',
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
    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path='s3://bucket/prefix',
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

#################################################################################
