# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import base64
import json
import os

import pytest
import yaml
from mock import call, patch, Mock

import sagemaker
from sagemaker.local.image import _SageMakerContainer

BUCKET_NAME = 'mybucket'
EXPANDED_ROLE = 'arn:aws:iam::111111111111:role/ExpandedRole'
INPUT_DATA_CONFIG = [
    {
        'ChannelName': 'a',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': '/tmp/source1'
            }
        }
    },
    {
        'ChannelName': 'b',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'FullyReplicated',
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://my-own-bucket/prefix'
            }
        }
    }
]
HYPERPARAMETERS = {'a': 1, 'b': "bee"}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}
    boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.return_value = []

    sms = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    sms.expand_role = Mock(return_value=EXPANDED_ROLE)

    return sms


@patch('sagemaker.local.local_session.LocalSession')
def test_write_config_file(LocalSession, tmpdir):

    sagemaker_container = _SageMakerContainer('local', 2, 'my-image')
    sagemaker_container.container_root = str(tmpdir.mkdir('container-root'))
    host = "algo-1"

    sagemaker.local.image._create_config_file_directories(sagemaker_container.container_root, host)

    container_root = sagemaker_container.container_root
    config_file_root = os.path.join(container_root, host, 'input', 'config')

    hyperparameters_file = os.path.join(config_file_root, 'hyperparameters.json')
    resource_config_file = os.path.join(config_file_root, 'resourceconfig.json')
    input_data_config_file = os.path.join(config_file_root, 'inputdataconfig.json')

    # write the config files, and then lets check they exist and have the right content.
    sagemaker_container.write_config_files(host, HYPERPARAMETERS, INPUT_DATA_CONFIG)

    assert os.path.exists(hyperparameters_file)
    assert os.path.exists(resource_config_file)
    assert os.path.exists(input_data_config_file)

    hyperparameters_data = json.load(open(hyperparameters_file))
    resource_config_data = json.load(open(resource_config_file))
    input_data_config_data = json.load(open(input_data_config_file))

    # Validate HyperParameters
    for k, v in HYPERPARAMETERS.items():
        assert k in hyperparameters_data
        assert hyperparameters_data[k] == v

    # Validate Resource Config
    assert resource_config_data['current_host'] == host
    assert resource_config_data['hosts'] == sagemaker_container.hosts

    # Validate Input Data Config
    for channel in INPUT_DATA_CONFIG:
        assert channel['ChannelName'] in input_data_config_data


@patch('sagemaker.local.local_session.LocalSession')
def test_retrieve_artifacts(LocalSession, tmpdir):
    sagemaker_container = _SageMakerContainer('local', 2, 'my-image')
    sagemaker_container.hosts = ['algo-1', 'algo-2']  # avoid any randomness
    sagemaker_container.container_root = str(tmpdir.mkdir('container-root'))

    volume1 = os.path.join(sagemaker_container.container_root, 'algo-1/output/')
    volume2 = os.path.join(sagemaker_container.container_root, 'algo-2/output/')
    os.makedirs(volume1)
    os.makedirs(volume2)

    compose_data = {
        'services': {
            'algo-1': {
                'volumes': ['%s:/opt/ml/model' % volume1]
            },
            'algo-2': {
                'volumes': ['%s:/opt/ml/model' % volume2]
            }
        }
    }

    dirs1 = ['model', 'model/data']
    dirs2 = ['model', 'model/data', 'model/tmp']

    files1 = ['model/data/model.json', 'model/data/variables.csv']
    files2 = ['model/data/model.json', 'model/data/variables2.csv', 'model/tmp/something-else.json']

    expected = ['model', 'model/data/', 'model/data/model.json', 'model/data/variables.csv',
                'model/data/variables2.csv', 'model/tmp/something-else.json']

    for d in dirs1:
        os.mkdir(os.path.join(volume1, d))
    for d in dirs2:
        os.mkdir(os.path.join(volume2, d))

    # create all the files
    for f in files1:
        open(os.path.join(volume1, f), 'a').close()
    for f in files2:
        open(os.path.join(volume2, f), 'a').close()

    s3_model_artifacts = sagemaker_container.retrieve_model_artifacts(compose_data)

    for f in expected:
        assert os.path.exists(os.path.join(s3_model_artifacts, f))


def test_stream_output():

    # it should raise an exception if the command fails
    with pytest.raises(Exception):
        sagemaker.local.image._execute_and_stream_output(['ls', '/some/unknown/path'])

    exit_code = sagemaker.local.image._execute_and_stream_output(['echo', 'hello'])
    assert exit_code == 0

    exit_code = sagemaker.local.image._execute_and_stream_output('echo hello!!!')
    assert exit_code == 0


def test_check_output():

    with pytest.raises(Exception):
        sagemaker.local.image._check_output(['ls', '/some/unknown/path'])

    msg = 'hello!'

    output = sagemaker.local.image._check_output(['echo', msg]).strip()
    assert output == msg

    output = sagemaker.local.image._check_output("echo %s" % msg).strip()
    assert output == msg


@patch('sagemaker.local.local_session.LocalSession')
@patch('sagemaker.local.image._execute_and_stream_output')
@patch('sagemaker.local.image._SageMakerContainer._cleanup')
@patch('sagemaker.local.image._SageMakerContainer._download_folder')
def test_train(_download_folder, _cleanup, _execute_and_stream_output, LocalSession, tmpdir, sagemaker_session):

    directories = [str(tmpdir.mkdir('container-root')), str(tmpdir.mkdir('data'))]
    with patch('sagemaker.local.image._SageMakerContainer._create_tmp_folder',
               side_effect=directories):

        instance_count = 2
        image = 'my-image'
        sagemaker_container = _SageMakerContainer('local', instance_count, image, sagemaker_session=sagemaker_session)
        sagemaker_container.train(INPUT_DATA_CONFIG, HYPERPARAMETERS)

        channel_dir = os.path.join(directories[1], 'b')
        download_folder_calls = [call('my-own-bucket', 'prefix', channel_dir)]
        _download_folder.assert_has_calls(download_folder_calls)

        docker_compose_file = os.path.join(sagemaker_container.container_root, 'docker-compose.yaml')

        call_args = _execute_and_stream_output.call_args[0][0]
        assert call_args is not None

        expected = ['docker-compose', '-f', docker_compose_file, 'up', '--build', '--abort-on-container-exit']
        for i, v in enumerate(expected):
            assert call_args[i] == v

        with open(docker_compose_file, 'r') as f:
            config = yaml.load(f)
            assert len(config['services']) == instance_count
            for h in sagemaker_container.hosts:
                assert config['services'][h]['image'] == image
                assert config['services'][h]['command'] == 'train'


@patch('sagemaker.local.image._HostingContainer.up')
@patch('shutil.copy')
@patch('shutil.copytree')
def test_serve(up, copy, copytree, tmpdir, sagemaker_session):

    with patch('sagemaker.local.image._SageMakerContainer._create_tmp_folder',
               return_value=str(tmpdir.mkdir('container-root'))):

        image = 'my-image'
        sagemaker_container = _SageMakerContainer('local', 1, image, sagemaker_session=sagemaker_session)
        primary_container = {'ModelDataUrl': '/some/model/path', 'Environment': {'env1': 1, 'env2': 'b'}}

        sagemaker_container.serve(primary_container)
        docker_compose_file = os.path.join(sagemaker_container.container_root, 'docker-compose.yaml')

        with open(docker_compose_file, 'r') as f:
            config = yaml.load(f)

            for h in sagemaker_container.hosts:
                assert config['services'][h]['image'] == image
                assert config['services'][h]['command'] == 'serve'


@patch('os.makedirs')
def test_download_folder(makedirs):
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}

    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    train_data = Mock()
    validation_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = '/prefix/train/train_data.csv'
    validation_data.bucket_name.return_value = BUCKET_NAME
    validation_data.key = '/prefix/train/validation_data.csv'

    s3_files = [train_data, validation_data]
    boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    obj_mock = Mock()
    boto_mock.resource('s3').Object.return_value = obj_mock

    sagemaker_container = _SageMakerContainer('local', 2, 'my-image', sagemaker_session=session)
    sagemaker_container._download_folder(BUCKET_NAME, '/prefix', '/tmp')

    obj_mock.download_file.assert_called()
    calls = [call(os.path.join('/tmp', 'train/train_data.csv')),
             call(os.path.join('/tmp', 'train/validation_data.csv'))]
    obj_mock.download_file.assert_has_calls(calls)


def test_ecr_login_non_ecr():
    session_mock = Mock()
    sagemaker.local.image._ecr_login_if_needed(session_mock, 'ubuntu')

    session_mock.assert_not_called()


@patch('sagemaker.local.image._check_output', return_value='123451324')
def test_ecr_login_image_exists(_check_output):
    session_mock = Mock()

    image = '520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-have:1.0'
    sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    session_mock.assert_not_called()
    _check_output.assert_called()


@patch('subprocess.check_output', return_value=''.encode('utf-8'))
def test_ecr_login_needed(check_output):
    session_mock = Mock()

    token = 'very-secure-token'
    token_response = 'AWS:%s' % token
    b64_token = base64.b64encode(token_response.encode('utf-8'))
    response = {
        u'authorizationData':
            [
                {
                    u'authorizationToken': b64_token,
                    u'proxyEndpoint': u'https://520713654638.dkr.ecr.us-east-1.amazonaws.com'
                }
            ],
        'ResponseMetadata':
            {
                'RetryAttempts': 0,
                'HTTPStatusCode': 200,
                'RequestId': '25b2ac63-36bf-11e8-ab6a-e5dc597d2ad9',
            }
    }
    session_mock.client('ecr').get_authorization_token.return_value = response
    image = '520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-need:1.1'
    sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    expected_command = 'docker login -u AWS -p %s https://520713654638.dkr.ecr.us-east-1.amazonaws.com' % token

    check_output.assert_called_with(expected_command, shell=True)
    session_mock.client('ecr').get_authorization_token.assert_called_with(registryIds=['520713654638'])
