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
import json
import os

import pytest
import yaml
from mock import patch, Mock

import sagemaker
from sagemaker.image import SageMakerContainer

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
                'S3Uri': 's3://foo/bar'
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

    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.expand_role = Mock(return_value=EXPANDED_ROLE)

    return ims


@patch('sagemaker.local_session.LocalSession')
def test_write_config_file(LocalSession, tmpdir):

    sagemaker_container = SageMakerContainer('local', 2, 'my-image')
    sagemaker_container.container_root = str(tmpdir.mkdir('container-root'))
    host = "algo-1"

    sagemaker.image._prepare_config_file_directory(sagemaker_container.container_root, host)

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


@patch('sagemaker.local_session.LocalSession')
def test_retrieve_artifacts(LocalSession, tmpdir):
    sagemaker_container = SageMakerContainer('local', 2, 'my-image')
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
        sagemaker.image._stream_output(['ls', '/some/unknown/path'])

    exit_code = sagemaker.image._stream_output(['echo', 'hello'])
    assert exit_code == 0

    exit_code = sagemaker.image._stream_output('echo hello!!!')
    assert exit_code == 0


def test_check_output():

    with pytest.raises(Exception):
        sagemaker.image._check_output(['ls', '/some/unknown/path'])

    msg = 'hello!'

    output = sagemaker.image._check_output(['echo', msg]).strip()
    assert output == msg

    output = sagemaker.image._check_output("echo %s" % msg).strip()
    assert output == msg


@patch('sagemaker.local_session.LocalSession')
@patch('sagemaker.image._stream_output')
@patch('sagemaker.image._cleanup')
def test_train(LocalSession, _stream_output, _cleanup, tmpdir, sagemaker_session):

    with patch('sagemaker.image._create_tmp_folder', return_value=str(tmpdir.mkdir('container-root'))):

        instance_count = 2
        image = 'my-image'
        sagemaker_container = SageMakerContainer('local', instance_count, image, sagemaker_session=sagemaker_session)
        sagemaker_container.train(INPUT_DATA_CONFIG, HYPERPARAMETERS)

        docker_compose_file = os.path.join(sagemaker_container.container_root, 'docker-compose.yaml')

        call_args = _stream_output.call_args[0][0]
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


@patch('sagemaker.image._Container.up')
@patch('shutil.copy')
@patch('shutil.copytree')
def test_serve(up, copy, copytree, tmpdir, sagemaker_session):

    with patch('sagemaker.image._create_tmp_folder', return_value=str(tmpdir.mkdir('container-root'))):

        image = 'my-image'
        sagemaker_container = SageMakerContainer('local', 1, image, sagemaker_session=sagemaker_session)
        primary_container = {'ModelDataUrl': '/some/model/path', 'Environment': {'env1': 1, 'env2': 'b'}}

        sagemaker_container.serve(primary_container)
        docker_compose_file = sagemaker_container.container.compose_file

        with open(docker_compose_file, 'r') as f:
            config = yaml.load(f)

            for h in sagemaker_container.hosts:
                assert config['services'][h]['image'] == image
                assert config['services'][h]['command'] == 'serve'
