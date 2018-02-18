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
import pytest
import io
import six
from mock import Mock, patch, call
import sagemaker
from sagemaker import s3_input, Session, get_execution_role
import datetime

from botocore.exceptions import ClientError

REGION = 'us-west-2'


@pytest.fixture()
def boto_session():
    boto_session = Mock(region_name=REGION)
    return boto_session


def test_get_execution_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = 'arn:aws:iam::369233609183:role/SageMakerRole'

    actual = get_execution_role(session)
    assert actual == 'arn:aws:iam::369233609183:role/SageMakerRole'


def test_get_execution_role_works_with_servie_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = \
        'arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388'

    actual = get_execution_role(session)
    assert actual == 'arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388'


def test_get_execution_role_throws_exception_if_arn_is_not_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = 'arn:aws:iam::369233609183:user/marcos'

    with pytest.raises(ValueError) as error:
        get_execution_role(session)
    assert 'ValueError: The current AWS identity is not a role' in str(error)


def test_get_caller_identity_arn_from_an_user(boto_session):
    sess = Session(boto_session)
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': 'arn:aws:iam::369233609183:user/mia'}

    actual = sess.get_caller_identity_arn()
    assert actual == 'arn:aws:iam::369233609183:user/mia'


def test_get_caller_identity_arn_from_a_role(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:sts::369233609183:assumed-role/SageMakerRole/6d009ef3-5306-49d5-8efc-78db644d8122'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}

    actual = sess.get_caller_identity_arn()
    assert actual == 'arn:aws:iam::369233609183:role/SageMakerRole'


def test_get_caller_identity_arn_from_a_execution_role(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:sts::369233609183:assumed-role/AmazonSageMaker-ExecutionRole-20171129T072388/SageMaker'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}

    actual = sess.get_caller_identity_arn()
    assert actual == 'arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388'


def test_delete_endpoint(boto_session):
    sess = Session(boto_session)
    sess.delete_endpoint('my_endpoint')

    boto_session.client().delete_endpoint.assert_called_with(EndpointName='my_endpoint')


def test_s3_input_all_defaults():
    prefix = 'pre'
    actual = s3_input(s3_data=prefix)
    expected = \
        {'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': prefix
            }
        }
        }
    assert actual.config == expected


def test_s3_input_all_arguments():
    prefix = 'pre'
    distribution = 'FullyReplicated'
    compression = 'Gzip'
    content_type = 'text/csv'
    record_wrapping = 'RecordIO'
    s3_data_type = 'Manifestfile'
    result = s3_input(s3_data=prefix, distribution=distribution, compression=compression,
                      content_type=content_type, record_wrapping=record_wrapping, s3_data_type=s3_data_type)
    expected = \
        {'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': distribution,
                'S3DataType': s3_data_type,
                'S3Uri': prefix,
            }
        },
            'CompressionType': compression,
            'ContentType': content_type,
            'RecordWrapperType': record_wrapping
        }

    assert result.config == expected


IMAGE = 'myimage'
S3_INPUT_URI = 's3://mybucket/data'
S3_OUTPUT = 's3://sagemaker-123/output/jobname'
ROLE = 'SageMakerRole'
EXPANDED_ROLE = 'arn:aws:iam::111111111111:role/ExpandedRole'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.xlarge'
MAX_SIZE = 30
MAX_TIME = 3 * 60 * 60
JOB_NAME = 'jobname'

DEFAULT_EXPECTED_TRAIN_JOB_ARGS = {
    # 'HyperParameters': None,
    'OutputDataConfig': {
        'S3OutputPath': S3_OUTPUT
    },
    'RoleArn': EXPANDED_ROLE,
    'ResourceConfig': {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
        'VolumeSizeInGB': MAX_SIZE
    },
    'InputDataConfig': [
        {
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': S3_INPUT_URI
                }
            },
            'ChannelName': 'training'
        }
    ],
    'AlgorithmSpecification': {
        'TrainingInputMode': 'File',
        'TrainingImage': IMAGE
    },
    'TrainingJobName': JOB_NAME,
    'StoppingCondition': {
        'MaxRuntimeInSeconds': MAX_TIME
    }
}

COMPLETED_DESCRIBE_JOB_RESULT = dict(DEFAULT_EXPECTED_TRAIN_JOB_ARGS)
COMPLETED_DESCRIBE_JOB_RESULT.update({'TrainingJobStatus': 'Completed'})
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {'ModelArtifacts': {
        'S3ModelArtifacts': S3_OUTPUT + '/model/model.tar.gz'
    }})
# TrainingStartTime and TrainingEndTime are for billable seconds calculation
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {'TrainingStartTime': datetime.datetime(2018, 2, 17, 7, 15, 0, 103000)})
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {'TrainingEndTime': datetime.datetime(2018, 2, 17, 7, 19, 34, 953000)})
IN_PROGRESS_DESCRIBE_JOB_RESULT = dict(DEFAULT_EXPECTED_TRAIN_JOB_ARGS)
IN_PROGRESS_DESCRIBE_JOB_RESULT.update({'TrainingJobStatus': 'InProgress'})


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    ims.expand_role = Mock(return_value=EXPANDED_ROLE)
    return ims


def test_train_pack_to_request(sagemaker_session):
    in_config = [{
        'ChannelName': 'training',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': S3_INPUT_URI
            }
        }
    }]

    out_config = {'S3OutputPath': S3_OUTPUT}

    resource_config = {'InstanceCount': INSTANCE_COUNT,
                       'InstanceType': INSTANCE_TYPE,
                       'VolumeSizeInGB': MAX_SIZE}

    stop_cond = {'MaxRuntimeInSeconds': MAX_TIME}

    sagemaker_session.train(image=IMAGE, input_mode='File', input_config=in_config, role=EXPANDED_ROLE,
                            job_name=JOB_NAME, output_config=out_config, resource_config=resource_config,
                            hyperparameters=None, stop_condition=stop_cond)

    assert sagemaker_session.sagemaker_client.method_calls[0] == (
        'create_training_job', (), DEFAULT_EXPECTED_TRAIN_JOB_ARGS)


@patch('sys.stdout', new_callable=io.BytesIO if six.PY2 else io.StringIO)
def test_color_wrap(bio):
    color_wrap = sagemaker.logs.ColorWrap()
    color_wrap(0, 'hi there')
    assert bio.getvalue() == 'hi there\n'


class MockBotoException(ClientError):
    def __init__(self, code):
        self.response = {'Error': {'Code': code}}


DEFAULT_LOG_STREAMS = {'logStreams': [{'logStreamName': JOB_NAME + '/xxxxxxxxx'}]}
LIFECYCLE_LOG_STREAMS = [MockBotoException('ResourceNotFoundException'),
                         DEFAULT_LOG_STREAMS,
                         DEFAULT_LOG_STREAMS,
                         DEFAULT_LOG_STREAMS,
                         DEFAULT_LOG_STREAMS,
                         DEFAULT_LOG_STREAMS,
                         DEFAULT_LOG_STREAMS]

DEFAULT_LOG_EVENTS = [{'nextForwardToken': None, 'events': [{'timestamp': 1, 'message': 'hi there #1'}]},
                      {'nextForwardToken': None, 'events': []}]
STREAM_LOG_EVENTS = [{'nextForwardToken': None, 'events': [{'timestamp': 1, 'message': 'hi there #1'}]},
                     {'nextForwardToken': None, 'events': []},
                     {'nextForwardToken': None, 'events': [{'timestamp': 1, 'message': 'hi there #1'},
                                                           {'timestamp': 2, 'message': 'hi there #2'}]},
                     {'nextForwardToken': None, 'events': []},
                     {'nextForwardToken': None, 'events': [{'timestamp': 2, 'message': 'hi there #2'},
                                                           {'timestamp': 2, 'message': 'hi there #2a'},
                                                           {'timestamp': 3, 'message': 'hi there #3'}]},
                     {'nextForwardToken': None, 'events': []}]


@pytest.fixture()
def sagemaker_session_complete():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('logs').describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client('logs').get_log_events.side_effect = DEFAULT_LOG_EVENTS
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    return ims


@pytest.fixture()
def sagemaker_session_ready_lifecycle():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('logs').describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client('logs').get_log_events.side_effect = STREAM_LOG_EVENTS
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    ims.sagemaker_client.describe_training_job.side_effect = [IN_PROGRESS_DESCRIBE_JOB_RESULT,
                                                              IN_PROGRESS_DESCRIBE_JOB_RESULT,
                                                              COMPLETED_DESCRIBE_JOB_RESULT]
    return ims


@pytest.fixture()
def sagemaker_session_full_lifecycle():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('logs').describe_log_streams.side_effect = LIFECYCLE_LOG_STREAMS
    boto_mock.client('logs').get_log_events.side_effect = STREAM_LOG_EVENTS
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    ims.sagemaker_client.describe_training_job.side_effect = [IN_PROGRESS_DESCRIBE_JOB_RESULT,
                                                              IN_PROGRESS_DESCRIBE_JOB_RESULT,
                                                              COMPLETED_DESCRIBE_JOB_RESULT]
    return ims


@patch('sagemaker.logs.ColorWrap')
def test_logs_for_job_no_wait(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_job(JOB_NAME)
    ims.sagemaker_client.describe_training_job.assert_called_once_with(TrainingJobName=JOB_NAME)
    cw().assert_called_with(0, 'hi there #1')


@patch('sagemaker.logs.ColorWrap')
def test_logs_for_job_wait_on_completed(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [call(TrainingJobName=JOB_NAME,)]
    cw().assert_called_with(0, 'hi there #1')


@patch('sagemaker.logs.ColorWrap')
def test_logs_for_job_no_wait_on_running(cw, sagemaker_session_ready_lifecycle):
    ims = sagemaker_session_ready_lifecycle
    ims.logs_for_job(JOB_NAME)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [call(TrainingJobName=JOB_NAME,)]
    cw().assert_called_with(0, 'hi there #1')


@patch('sagemaker.logs.ColorWrap')
@patch('time.time', side_effect=[0, 30, 60, 90, 120, 150, 180])
def test_logs_for_job_full_lifecycle(time, cw, sagemaker_session_full_lifecycle):
    ims = sagemaker_session_full_lifecycle
    ims.logs_for_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [call(TrainingJobName=JOB_NAME,)] * 3
    assert cw().call_args_list == [call(0, 'hi there #1'), call(0, 'hi there #2'),
                                   call(0, 'hi there #2a'), call(0, 'hi there #3')]


def test_create_model_from_job(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME)

    assert call(TrainingJobName='jobname') in ims.sagemaker_client.describe_training_job.call_args_list
    ims.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn='arn:aws:iam::111111111111:role/ExpandedRole',
        ModelName='jobname',
        PrimaryContainer={
            'Environment': {}, 'ModelDataUrl': 's3://sagemaker-123/output/jobname/model/model.tar.gz',
            'Image': 'myimage'})


def test_create_model_from_job_with_image(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, primary_container_image='some-image')
    [create_model_call] = ims.sagemaker_client.create_model.call_args_list
    assert dict(create_model_call[1]['PrimaryContainer'])['Image'] == 'some-image'


def test_create_model_from_job_with_container_def(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, primary_container_image='some-image', model_data_url='some-data',
                              env={'a': 'b'})
    [create_model_call] = ims.sagemaker_client.create_model.call_args_list
    c_def = create_model_call[1]['PrimaryContainer']
    assert c_def['Image'] == 'some-image'
    assert c_def['ModelDataUrl'] == 'some-data'
    assert c_def['Environment'] == {'a': 'b'}


def test_endpoint_from_production_variants(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={'EndpointStatus': 'InService'})
    pvs = [sagemaker.production_variant('A', 'ml.p2.xlarge'), sagemaker.production_variant('B', 'p299.4096xlarge')]
    ex = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find your thing'}}, 'b')
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    sagemaker_session.endpoint_from_production_variants('some-endpoint', pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(EndpointConfigName='some-endpoint',
                                                                          EndpointName='some-endpoint')
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName='some-endpoint',
        ProductionVariants=[
            {
                'InstanceType': 'ml.p2.xlarge',
                'ModelName': 'A',
                'InitialVariantWeight': 1,
                'InitialInstanceCount': 1,
                'VariantName': 'AllTraffic'
            },
            {
                'InstanceType': 'p299.4096xlarge',
                'ModelName': 'B',
                'InitialVariantWeight': 1,
                'InitialInstanceCount': 1,
                'VariantName': 'AllTraffic'}])
