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
import pytest
import sys
from distutils.util import strtobool
from mock import MagicMock, Mock
from mock import patch


from sagemaker.chainer import defaults
from sagemaker.chainer import Chainer
from sagemaker.chainer import ChainerPredictor, ChainerModel


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_PATH = os.path.join(DATA_DIR, 'dummy_script.py')
MODEL_DATA = "s3://some/data.tar.gz"
TIMESTAMP = '2017-11-06-14:14:15.672'
TIME = 1507167947
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
ACCELERATOR_TYPE = 'ml.eia.medium'
PYTHON_VERSION = 'py' + str(sys.version_info.major)
IMAGE_NAME = 'sagemaker-chainer'
JOB_NAME = '{}-{}'.format(IMAGE_NAME, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"
ROLE = 'Dummy'
REGION = 'us-west-2'
GPU = 'ml.p2.xlarge'
CPU = 'ml.c4.xlarge'

ENDPOINT_DESC = {
    'EndpointConfigName': 'test-endpoint'
}

ENDPOINT_CONFIG_DESC = {
    'ProductionVariants': [{'ModelName': 'model-1'},
                           {'ModelName': 'model-2'}]
}

LIST_TAGS_RESULT = {
    'Tags': [{'Key': 'TagtestKey', 'Value': 'TagtestValue'}]
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    session = Mock(name='sagemaker_session', boto_session=boto_mock,
                   boto_region_name=REGION, config=None, local_mode=False)

    describe = {'ModelArtifacts': {'S3ModelArtifacts': 's3://m/m.tar.gz'}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    return session


def _get_full_cpu_image_uri(version, py_version=PYTHON_VERSION):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, version, 'cpu', py_version)


def _get_full_gpu_image_uri(version, py_version=PYTHON_VERSION):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, version, 'gpu', py_version)


def _get_full_cpu_image_uri_with_ei(version, py_version=PYTHON_VERSION):
    return _get_full_cpu_image_uri(version, py_version=py_version) + '-eia'


def _chainer_estimator(sagemaker_session, framework_version=defaults.CHAINER_VERSION, train_instance_type=None,
                       base_job_name=None, use_mpi=None, num_processes=None,
                       process_slots_per_host=None, additional_mpi_options=None, **kwargs):
    return Chainer(entry_point=SCRIPT_PATH,
                   framework_version=framework_version,
                   role=ROLE,
                   sagemaker_session=sagemaker_session,
                   train_instance_count=INSTANCE_COUNT,
                   train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
                   base_job_name=base_job_name,
                   use_mpi=use_mpi,
                   num_processes=num_processes,
                   process_slots_per_host=process_slots_per_host,
                   additional_mpi_options=additional_mpi_options,
                   py_version=PYTHON_VERSION,
                   **kwargs)


def _create_train_job(version):
    return {
        'image': _get_full_cpu_image_uri(version),
        'input_mode': 'File',
        'input_config': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix'
                    }
                }
            }
        ],
        'role': ROLE,
        'job_name': JOB_NAME,
        'output_config': {
            'S3OutputPath': 's3://{}/'.format(BUCKET_NAME),
        },
        'resource_config': {
            'InstanceType': 'ml.c4.4xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30,
        },
        'hyperparameters': {
            'sagemaker_program': json.dumps('dummy_script.py'),
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': str(logging.INFO),
            'sagemaker_job_name': json.dumps(JOB_NAME),
            'sagemaker_submit_directory':
                json.dumps('s3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME)),
            'sagemaker_region': '"us-west-2"'
        },
        'stop_condition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        },
        'tags': None,
        'vpc_config': None,
        'metric_definitions': None
    }


def _create_train_job_with_additional_hyperparameters(version):
    return {
        'image': _get_full_cpu_image_uri(version),
        'input_mode': 'File',
        'input_config': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix'
                    }
                }
            }
        ],
        'role': ROLE,
        'job_name': JOB_NAME,
        'output_config': {
            'S3OutputPath': 's3://{}/'.format(BUCKET_NAME),
        },
        'resource_config': {
            'InstanceType': 'ml.c4.4xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30,
        },
        'hyperparameters': {
            'sagemaker_program': json.dumps('dummy_script.py'),
            'sagemaker_container_log_level': str(logging.INFO),
            'sagemaker_job_name': json.dumps(JOB_NAME),
            'sagemaker_submit_directory':
                json.dumps('s3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME)),
            'sagemaker_region': '"us-west-2"',
            'sagemaker_num_processes': '4',
            'sagemaker_additional_mpi_options': '"-x MY_ENVIRONMENT_VARIABLE"',
            'sagemaker_process_slots_per_host': '10',
            'sagemaker_use_mpi': 'true'
        },
        'stop_condition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        }
    }


def test_additional_hyperparameters(sagemaker_session):
    chainer = _chainer_estimator(sagemaker_session, use_mpi=True, num_processes=4, process_slots_per_host=10,
                                 additional_mpi_options="-x MY_ENVIRONMENT_VARIABLE")
    assert bool(strtobool(chainer.hyperparameters()['sagemaker_use_mpi']))
    assert int(chainer.hyperparameters()['sagemaker_num_processes']) == 4
    assert int(chainer.hyperparameters()['sagemaker_process_slots_per_host']) == 10
    assert str(chainer.hyperparameters()['sagemaker_additional_mpi_options']) == '\"-x MY_ENVIRONMENT_VARIABLE\"'


def test_attach_with_additional_hyperparameters(sagemaker_session, chainer_version):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:{}-cpu-{}'.format(chainer_version,
                                                                                            PYTHON_VERSION)
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': training_image},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_s3_uri_training': '"sagemaker-3/integ-test-data/tf_iris"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
                                     'sagemaker_job_name': '"neo"',
                                     'sagemaker_region': '"us-west-2"',
                                     'sagemaker_num_processes': '4',
                                     'sagemaker_additional_mpi_options': '"-x MY_ENVIRONMENT_VARIABLE"',
                                     'sagemaker_process_slots_per_host': '10',
                                     'sagemaker_use_mpi': 'true'
                                     },
                                'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
                                'ResourceConfig':
                                    {'VolumeSizeInGB': 30,
                                     'InstanceCount': 1,
                                     'InstanceType': 'ml.c4.xlarge'},
                                'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
                                'TrainingJobName': 'neo',
                                'TrainingJobStatus': 'Completed',
                                'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:336:training-job/neo',
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    estimator = Chainer.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert bool(estimator.hyperparameters()['sagemaker_use_mpi'])
    assert int(estimator.hyperparameters()['sagemaker_num_processes']) == 4
    assert int(estimator.hyperparameters()['sagemaker_process_slots_per_host']) == 10
    assert str(estimator.hyperparameters()['sagemaker_additional_mpi_options']) == '\"-x MY_ENVIRONMENT_VARIABLE\"'
    assert estimator.use_mpi
    assert estimator.num_processes == 4
    assert estimator.process_slots_per_host == 10
    assert estimator.additional_mpi_options == "-x MY_ENVIRONMENT_VARIABLE"


def test_create_model(sagemaker_session, chainer_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    chainer = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                      framework_version=chainer_version, container_log_level=container_log_level,
                      py_version=PYTHON_VERSION, base_job_name='job', source_dir=source_dir)

    job_name = 'new_name'
    chainer.fit(inputs='s3://mybucket/train', job_name=job_name)
    model = chainer.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == chainer_version
    assert model.py_version == chainer.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.vpc_config is None


def test_create_model_with_optional_params(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    enable_cloudwatch_metrics = 'true'
    chainer = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                      container_log_level=container_log_level, py_version=PYTHON_VERSION, base_job_name='job',
                      source_dir=source_dir, enable_cloudwatch_metrics=enable_cloudwatch_metrics)

    chainer.fit(inputs='s3://mybucket/train', job_name='new_name')

    new_role = 'role'
    model_server_workers = 2
    vpc_config = {'Subnets': ['foo'], 'SecurityGroupIds': ['bar']}
    model = chainer.create_model(role=new_role, model_server_workers=model_server_workers,
                                 vpc_config_override=vpc_config)

    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    custom_image = 'ubuntu:latest'
    chainer = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                      image_name=custom_image, container_log_level=container_log_level,
                      py_version=PYTHON_VERSION, base_job_name='job', source_dir=source_dir)

    chainer.fit(inputs='s3://mybucket/train', job_name='new_name')
    model = chainer.create_model()

    assert model.image == custom_image


@patch('sagemaker.utils.create_tar_file', MagicMock())
@patch('time.strftime', return_value=TIMESTAMP)
def test_chainer(strftime, sagemaker_session, chainer_version):
    chainer = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE, py_version=PYTHON_VERSION,
                      framework_version=chainer_version)

    inputs = 's3://mybucket/train'

    chainer.fit(inputs=inputs)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ['train', 'logs_for_job']
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ['resource']

    expected_train_args = _create_train_job(chainer_version)
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = chainer.create_model()

    expected_image_base = '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:{}-gpu-{}'
    assert {'Environment':
            {'SAGEMAKER_SUBMIT_DIRECTORY':
             's3://mybucket/sagemaker-chainer-{}/source/sourcedir.tar.gz'.format(TIMESTAMP),
             'SAGEMAKER_PROGRAM': 'dummy_script.py',
             'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
             'SAGEMAKER_REGION': 'us-west-2',
             'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'},
            'Image': expected_image_base.format(chainer_version, PYTHON_VERSION),
            'ModelDataUrl': 's3://m/m.tar.gz'} == model.prepare_container_def(GPU)

    assert 'cpu' in model.prepare_container_def(CPU)['Image']
    predictor = chainer.deploy(1, GPU)
    assert isinstance(predictor, ChainerPredictor)


@patch('sagemaker.utils.create_tar_file', MagicMock())
def test_model(sagemaker_session):
    model = ChainerModel("s3://some/data.tar.gz", role=ROLE, entry_point=SCRIPT_PATH,
                         sagemaker_session=sagemaker_session)
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, ChainerPredictor)


@patch('sagemaker.fw_utils.tar_and_upload_dir', MagicMock())
def test_model_prepare_container_def_accelerator_error(sagemaker_session):
    model = ChainerModel(MODEL_DATA, role=ROLE, entry_point=SCRIPT_PATH,
                         sagemaker_session=sagemaker_session)
    with pytest.raises(ValueError):
        model.prepare_container_def(INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE)


def test_train_image_default(sagemaker_session):
    chainer = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE, py_version=PYTHON_VERSION)

    assert _get_full_cpu_image_uri(defaults.CHAINER_VERSION) in chainer.train_image()


def test_train_image_cpu_instances(sagemaker_session, chainer_version):
    chainer = _chainer_estimator(sagemaker_session, chainer_version, train_instance_type='ml.c2.2xlarge')
    assert chainer.train_image() == _get_full_cpu_image_uri(chainer_version)

    chainer = _chainer_estimator(sagemaker_session, chainer_version, train_instance_type='ml.c4.2xlarge')
    assert chainer.train_image() == _get_full_cpu_image_uri(chainer_version)

    chainer = _chainer_estimator(sagemaker_session, chainer_version, train_instance_type='ml.m16')
    assert chainer.train_image() == _get_full_cpu_image_uri(chainer_version)


def test_train_image_gpu_instances(sagemaker_session, chainer_version):
    chainer = _chainer_estimator(sagemaker_session, chainer_version, train_instance_type='ml.g2.2xlarge')
    assert chainer.train_image() == _get_full_gpu_image_uri(chainer_version)

    chainer = _chainer_estimator(sagemaker_session, chainer_version, train_instance_type='ml.p2.2xlarge')
    assert chainer.train_image() == _get_full_gpu_image_uri(chainer_version)


def test_attach(sagemaker_session, chainer_version):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:{}-cpu-{}'.format(chainer_version,
                                                                                            PYTHON_VERSION)
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': training_image},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_s3_uri_training': '"sagemaker-3/integ-test-data/tf_iris"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
                                     'sagemaker_job_name': '"neo"',
                                     'training_steps': '100',
                                     'sagemaker_region': '"us-west-2"'},
                                'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
                                'ResourceConfig':
                                    {'VolumeSizeInGB': 30,
                                     'InstanceCount': 1,
                                     'InstanceType': 'ml.c4.xlarge'},
                                'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
                                'TrainingJobName': 'neo',
                                'TrainingJobStatus': 'Completed',
                                'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:336:training-job/neo',
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    estimator = Chainer.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.py_version == PYTHON_VERSION
    assert estimator.framework_version == chainer_version
    assert estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert estimator.train_instance_count == 1
    assert estimator.train_max_run == 24 * 60 * 60
    assert estimator.input_mode == 'File'
    assert estimator.base_job_name == 'neo'
    assert estimator.output_path == 's3://place/output/neo'
    assert estimator.output_kms_key == ''
    assert estimator.hyperparameters()['training_steps'] == '100'
    assert estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert estimator.entry_point == 'iris-dnn-classifier.py'


def test_attach_wrong_framework(sagemaker_session):
    rjd = {'AlgorithmSpecification':
           {'TrainingInputMode': 'File',
            'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0.4'},
           'HyperParameters':
               {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                'checkpoint_path': '"s3://other/1508872349"',
                'sagemaker_program': '"iris-dnn-classifier.py"',
                'sagemaker_enable_cloudwatch_metrics': 'false',
                'sagemaker_container_log_level': '"logging.INFO"',
                'training_steps': '100',
                'sagemaker_region': '"us-west-2"'},
           'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
           'ResourceConfig':
               {'VolumeSizeInGB': 30,
                'InstanceCount': 1,
                'InstanceType': 'ml.c4.xlarge'},
           'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
           'TrainingJobName': 'neo',
           'TrainingJobStatus': 'Completed',
           'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:336:training-job/neo',
           'OutputDataConfig': {'KmsKeyId': '',
                                'S3OutputPath': 's3://place/output/neo'},
           'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job', return_value=rjd)

    with pytest.raises(ValueError) as error:
        Chainer.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/my_custom_chainer_image:latest'
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': training_image},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_s3_uri_training': '"sagemaker-3/integ-test-data/tf_iris"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
                                     'sagemaker_job_name': '"neo"',
                                     'training_steps': '100',
                                     'sagemaker_region': '"us-west-2"'},
                                'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
                                'ResourceConfig':
                                    {'VolumeSizeInGB': 30,
                                     'InstanceCount': 1,
                                     'InstanceType': 'ml.c4.xlarge'},
                                'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
                                'TrainingJobName': 'neo',
                                'TrainingJobStatus': 'Completed',
                                'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:336:training-job/neo',
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    estimator = Chainer.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.image_name == training_image
    assert estimator.train_image() == training_image


@patch('sagemaker.chainer.estimator.empty_framework_version_warning')
def test_empty_framework_version(warning, sagemaker_session):
    estimator = Chainer(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                        train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                        framework_version=None)

    assert estimator.framework_version == defaults.CHAINER_VERSION
    warning.assert_called_with(defaults.CHAINER_VERSION, Chainer.LATEST_VERSION)
