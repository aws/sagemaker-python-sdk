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

from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.estimator import Estimator
from sagemaker.job import _Job
from sagemaker.session import s3_input

BUCKET_NAME = 's3://mybucket/train'
S3_OUTPUT_PATH = 's3://bucket/prefix'
LOCAL_FILE_NAME = 'file://local/file'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'c4.4xlarge'
VOLUME_SIZE = 1
MAX_RUNTIME = 1
ROLE = 'DummyRole'
IMAGE_NAME = 'fakeimage'
JOB_NAME = 'fakejob'
VOLUME_KMS_KEY = 'volkmskey'
CHANNEL_NAME = 'testChannel'
MODEL_URI = 's3://bucket/prefix/model.tar.gz'
LOCAL_MODEL_NAME = 'file://local/file.tar.gz'


@pytest.fixture()
def estimator(sagemaker_session):
    return Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, train_volume_size=VOLUME_SIZE,
                     train_max_run=MAX_RUNTIME, output_path=S3_OUTPUT_PATH, sagemaker_session=sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session')
    mock_session = Mock(name='sagemaker_session', boto_session=boto_mock)
    mock_session.expand_role = Mock(name='expand_role', return_value=ROLE)

    return mock_session


def test_load_config(estimator):
    inputs = s3_input(BUCKET_NAME)

    config = _Job._load_config(inputs, estimator)

    assert config['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] == BUCKET_NAME
    assert config['role'] == ROLE
    assert config['output_config']['S3OutputPath'] == S3_OUTPUT_PATH
    assert 'KmsKeyId' not in config['output_config']
    assert config['resource_config']['InstanceCount'] == INSTANCE_COUNT
    assert config['resource_config']['InstanceType'] == INSTANCE_TYPE
    assert config['resource_config']['VolumeSizeInGB'] == VOLUME_SIZE
    assert config['stop_condition']['MaxRuntimeInSeconds'] == MAX_RUNTIME


def test_load_config_with_model_channel(estimator):
    inputs = s3_input(BUCKET_NAME)

    estimator.model_uri = MODEL_URI
    estimator.model_channel_name = CHANNEL_NAME

    config = _Job._load_config(inputs, estimator)

    assert config['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] == BUCKET_NAME
    assert config['input_config'][1]['DataSource']['S3DataSource']['S3Uri'] == MODEL_URI
    assert config['input_config'][1]['ChannelName'] == CHANNEL_NAME
    assert config['role'] == ROLE
    assert config['output_config']['S3OutputPath'] == S3_OUTPUT_PATH
    assert 'KmsKeyId' not in config['output_config']
    assert config['resource_config']['InstanceCount'] == INSTANCE_COUNT
    assert config['resource_config']['InstanceType'] == INSTANCE_TYPE
    assert config['resource_config']['VolumeSizeInGB'] == VOLUME_SIZE
    assert config['stop_condition']['MaxRuntimeInSeconds'] == MAX_RUNTIME


def test_load_config_with_model_channel_no_inputs(estimator):
    estimator.model_uri = MODEL_URI
    estimator.model_channel_name = CHANNEL_NAME

    config = _Job._load_config(inputs=None, estimator=estimator)

    assert config['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] == MODEL_URI
    assert config['input_config'][0]['ChannelName'] == CHANNEL_NAME
    assert config['role'] == ROLE
    assert config['output_config']['S3OutputPath'] == S3_OUTPUT_PATH
    assert 'KmsKeyId' not in config['output_config']
    assert config['resource_config']['InstanceCount'] == INSTANCE_COUNT
    assert config['resource_config']['InstanceType'] == INSTANCE_TYPE
    assert config['resource_config']['VolumeSizeInGB'] == VOLUME_SIZE
    assert config['stop_condition']['MaxRuntimeInSeconds'] == MAX_RUNTIME


def test_format_inputs_none():
    channels = _Job._format_inputs_to_input_config(inputs=None)

    assert channels is None


def test_format_inputs_to_input_config_string():
    inputs = BUCKET_NAME

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]['DataSource']['S3DataSource']['S3Uri'] == inputs


def test_format_inputs_to_input_config_s3_input():
    inputs = s3_input(BUCKET_NAME)

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]['DataSource']['S3DataSource']['S3Uri'] == inputs.config['DataSource'][
        'S3DataSource']['S3Uri']


def test_format_inputs_to_input_config_dict():
    inputs = {'train': BUCKET_NAME}

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]['DataSource']['S3DataSource']['S3Uri'] == inputs['train']


def test_format_inputs_to_input_config_record_set():
    inputs = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]['DataSource']['S3DataSource']['S3Uri'] == inputs.s3_data
    assert channels[0]['DataSource']['S3DataSource']['S3DataType'] == inputs.s3_data_type


def test_format_inputs_to_input_config_list():
    records = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [records]

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]['DataSource']['S3DataSource']['S3Uri'] == records.s3_data
    assert channels[0]['DataSource']['S3DataSource']['S3DataType'] == records.s3_data_type


def test_prepare_model_channel():
    model_channel = _Job._prepare_model_channel([], MODEL_URI, CHANNEL_NAME)

    # The model channel should use all the defaults except InputMode
    assert model_channel['DataSource']['S3DataSource']['S3Uri'] == MODEL_URI
    assert model_channel['DataSource']['S3DataSource']['S3DataDistributionType'] == 'FullyReplicated'
    assert model_channel['DataSource']['S3DataSource']['S3DataType'] == 'S3Prefix'
    assert model_channel['InputMode'] == 'File'
    assert model_channel['ChannelName'] == CHANNEL_NAME
    assert 'CompressionType' not in model_channel
    assert model_channel['ContentType'] == 'application/x-sagemaker-model'
    assert 'RecordWrapperType' not in model_channel


def test_prepare_model_channel_duplicate():
    channels = [{
        'ChannelName': CHANNEL_NAME,
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://blah/blah'
            }
        }
    }]

    with pytest.raises(ValueError) as error:
        _Job._prepare_model_channel(channels, MODEL_URI, CHANNEL_NAME)

    assert 'Duplicate channels not allowed.' in str(error)


def test_prepare_model_channel_with_missing_name():
    with pytest.raises(ValueError) as ex:
        _Job._prepare_model_channel([], model_uri=MODEL_URI, model_channel_name=None)

    assert 'Expected a pre-trained model channel name if a model URL is specified.' in str(ex)


def test_prepare_model_channel_with_missing_uri():
    assert _Job._prepare_model_channel([], model_uri=None, model_channel_name=None) is None


def test_format_inputs_to_input_config_list_not_all_records():
    records = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [records, 'mock']

    with pytest.raises(ValueError) as ex:
        _Job._format_inputs_to_input_config(inputs)

    assert 'List compatible only with RecordSets.' in str(ex)


def test_format_inputs_to_input_config_list_duplicate_channel():
    record = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [record, record]

    with pytest.raises(ValueError) as ex:
        _Job._format_inputs_to_input_config(inputs)

    assert 'Duplicate channels not allowed.' in str(ex)


def test_format_input_single_unamed_channel():
    input_dict = _Job._format_inputs_to_input_config('s3://blah/blah')
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


def test_format_input_multiple_channels():
    input_list = _Job._format_inputs_to_input_config({'a': 's3://blah/blah', 'b': 's3://foo/bar'})
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
    input_dict = _Job._format_inputs_to_input_config(s3_input('s3://foo/bar', distribution='ShardedByS3Key',
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
    input_list = _Job._format_inputs_to_input_config({
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


def test_format_inputs_to_input_config_exception():
    inputs = 1

    with pytest.raises(ValueError):
        _Job._format_inputs_to_input_config(inputs)


def test_unsupported_type_in_dict():
    with pytest.raises(ValueError):
        _Job._format_inputs_to_input_config({'a': 66})


def test_format_string_uri_input_string():
    inputs = BUCKET_NAME

    s3_uri_input = _Job._format_string_uri_input(inputs)

    assert s3_uri_input.config['DataSource']['S3DataSource']['S3Uri'] == inputs


def test_format_string_uri_input_string_exception():
    inputs = 'mybucket/train'

    with pytest.raises(ValueError):
        _Job._format_string_uri_input(inputs)


def test_format_string_uri_input_local_file():
    file_uri_input = _Job._format_string_uri_input(LOCAL_FILE_NAME)

    assert file_uri_input.config['DataSource']['FileDataSource']['FileUri'] == LOCAL_FILE_NAME


def test_format_string_uri_input():
    inputs = s3_input(BUCKET_NAME)

    s3_uri_input = _Job._format_string_uri_input(inputs)

    assert s3_uri_input.config['DataSource']['S3DataSource']['S3Uri'] == inputs.config[
        'DataSource']['S3DataSource']['S3Uri']


def test_format_string_uri_input_exception():
    inputs = 1

    with pytest.raises(ValueError):
        _Job._format_string_uri_input(inputs)


def test_format_model_uri_input_string():
    model_uri = MODEL_URI

    model_uri_input = _Job._format_model_uri_input(model_uri)

    assert model_uri_input.config['DataSource']['S3DataSource']['S3Uri'] == model_uri


def test_format_model_uri_input_local_file():
    model_uri_input = _Job._format_model_uri_input(LOCAL_MODEL_NAME)

    assert model_uri_input.config['DataSource']['FileDataSource']['FileUri'] == LOCAL_MODEL_NAME


def test_format_model_uri_input_exception():
    model_uri = 1

    with pytest.raises(ValueError):
        _Job._format_model_uri_input(model_uri)


def test_prepare_output_config():
    kms_key_id = 'kms_key'

    config = _Job._prepare_output_config(BUCKET_NAME, kms_key_id)

    assert config['S3OutputPath'] == BUCKET_NAME
    assert config['KmsKeyId'] == kms_key_id


def test_prepare_output_config_kms_key_none():
    s3_path = BUCKET_NAME
    kms_key_id = None

    config = _Job._prepare_output_config(s3_path, kms_key_id)

    assert config['S3OutputPath'] == s3_path
    assert 'KmsKeyId' not in config


def test_prepare_resource_config():
    resource_config = _Job._prepare_resource_config(INSTANCE_COUNT, INSTANCE_TYPE, VOLUME_SIZE, None)

    assert resource_config == {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
        'VolumeSizeInGB': VOLUME_SIZE
    }


def test_prepare_resource_config_with_volume_kms():
    resource_config = _Job._prepare_resource_config(INSTANCE_COUNT, INSTANCE_TYPE, VOLUME_SIZE, VOLUME_KMS_KEY)

    assert resource_config == {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
        'VolumeSizeInGB': VOLUME_SIZE,
        'VolumeKmsKeyId': VOLUME_KMS_KEY
    }


def test_prepare_stop_condition():
    max_run = 1

    stop_condition = _Job._prepare_stop_condition(max_run)

    assert stop_condition['MaxRuntimeInSeconds'] == max_run


def test_name(sagemaker_session):
    job = _Job(sagemaker_session, JOB_NAME)
    assert job.name == JOB_NAME
