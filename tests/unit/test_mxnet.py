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
import logging

import json
import os
import pytest
from mock import Mock
from mock import patch

from sagemaker.mxnet import defaults
from sagemaker.mxnet import MXNet
from sagemaker.mxnet import MXNetPredictor, MXNetModel


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_PATH = os.path.join(DATA_DIR, 'dummy_script.py')
TIMESTAMP = '2017-11-06-14:14:15.672'
TIME = 1507167947
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
IMAGE_CPU_NAME = 'sagemaker-mxnet-py2-cpu'
JOB_NAME = '{}-{}'.format(IMAGE_CPU_NAME, TIMESTAMP)
FULL_IMAGE_URI = '520713654638.dkr.ecr.us-west-2.amazonaws.com/{}:{}-cpu-py2'
ROLE = 'Dummy'
REGION = 'us-west-2'
GPU = 'ml.p2.xlarge'
CPU = 'ml.c4.xlarge'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.sagemaker_client.describe_training_job = Mock(return_value={'ModelArtifacts':
                                                                    {'S3ModelArtifacts': 's3://m/m.tar.gz'}})
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.expand_role = Mock(name="expand_role", return_value=ROLE)
    return ims


def _get_full_image_uri(version):
    return FULL_IMAGE_URI.format(IMAGE_CPU_NAME, version)


def _create_train_job(version):
    return {'image': _get_full_image_uri(version),
            'input_mode': 'File',
            'input_config': [{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix'
                    }
                }
            }],
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
            }}


def test_create_model(sagemaker_session, mxnet_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    enable_cloudwatch_metrics = 'true'
    mx = MXNet(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
               train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
               framework_version=mxnet_version, container_log_level=container_log_level,
               base_job_name='job', source_dir=source_dir, enable_cloudwatch_metrics=enable_cloudwatch_metrics)

    job_name = 'new_name'
    mx.fit(inputs='s3://mybucket/train', job_name='new_name')
    model = mx.create_model()
    mx.container_log_level

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == mxnet_version
    assert model.py_version == mx.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.enable_cloudwatch_metrics == enable_cloudwatch_metrics


@patch('time.strftime', return_value=TIMESTAMP)
def test_mxnet(strftime, sagemaker_session, mxnet_version):
    mx = MXNet(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
               train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
               framework_version=mxnet_version)

    inputs = 's3://mybucket/train'

    mx.fit(inputs=inputs)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ['train', 'logs_for_job']
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ['resource']

    expected_train_args = _create_train_job(mxnet_version)
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = mx.create_model()

    expected_image_base = '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-gpu:{}-gpu-py2'
    assert {'Environment':
            {'SAGEMAKER_SUBMIT_DIRECTORY':
             's3://mybucket/sagemaker-mxnet-py2-cpu-{}/sourcedir.tar.gz'.format(TIMESTAMP),
             'SAGEMAKER_PROGRAM': 'dummy_script.py',
             'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
             'SAGEMAKER_REGION': 'us-west-2',
             'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'},
            'Image': expected_image_base.format(mxnet_version),
            'ModelDataUrl': 's3://m/m.tar.gz'} == model.prepare_container_def(GPU)

    assert 'cpu' in model.prepare_container_def(CPU)['Image']
    predictor = mx.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)


def test_model(sagemaker_session):
    model = MXNetModel("s3://some/data.tar.gz", role=ROLE, entry_point=SCRIPT_PATH,
                       sagemaker_session=sagemaker_session)
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)


def test_train_image_default(sagemaker_session):
    mx = MXNet(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
               train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    assert _get_full_image_uri(defaults.MXNET_VERSION) in mx.train_image()


def test_attach(sagemaker_session, mxnet_version):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:{}-cpu-py2'.format(mxnet_version)
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
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    estimator = MXNet.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.py_version == 'py2'
    assert estimator.framework_version == mxnet_version
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


def test_attach_old_container(sagemaker_session):
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0'},
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
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    estimator = MXNet.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.py_version == 'py2'
    assert estimator.framework_version == '0.12'
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
            'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:1.0.4'},
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
           'OutputDataConfig': {'KmsKeyId': '',
                                'S3OutputPath': 's3://place/output/neo'},
           'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job', return_value=rjd)

    with pytest.raises(ValueError) as error:
        MXNet.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)
