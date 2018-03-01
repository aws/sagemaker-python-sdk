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
from mock import Mock, patch
from sagemaker.model import MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.session import s3_input
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow import defaults
from sagemaker.fw_utils import create_image_uri
from sagemaker.tensorflow import TensorFlowPredictor, TensorFlowModel

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_PATH = os.path.join(DATA_DIR, 'dummy_script.py')
TIMESTAMP = '2017-11-06-14:14:15.673'
TIME = 1510006209.073025
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
CPU_IMAGE_NAME = 'sagemaker-tensorflow-py2-cpu'
GPU_IMAGE_NAME = 'sagemaker-tensorflow-py2-gpu'
JOB_NAME = '{}-{}'.format(CPU_IMAGE_NAME, TIMESTAMP)
ROLE = 'Dummy'
REGION = 'us-west-2'
DOCKER_TAG = '1.0'
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.expand_role = Mock(name="expand_role", return_value=ROLE)
    ims.sagemaker_client.describe_training_job = Mock(return_value={'ModelArtifacts':
                                                                    {'S3ModelArtifacts': 's3://m/m.tar.gz'}
                                                                    })
    return ims


def _get_full_cpu_image_uri(version):
    return IMAGE_URI_FORMAT_STRING.format(REGION, CPU_IMAGE_NAME, version, 'cpu', 'py2')


def _get_full_gpu_image_uri(version):
    return IMAGE_URI_FORMAT_STRING.format(REGION, GPU_IMAGE_NAME, version, 'gpu', 'py2')


def _create_train_job(tf_version):
    return {'image': _get_full_cpu_image_uri(tf_version),
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
                'training_steps': '1000',
                'evaluation_steps': '10',
                'sagemaker_program': json.dumps('dummy_script.py'),
                'sagemaker_submit_directory': json.dumps('s3://{}/{}/source/sourcedir.tar.gz'.format(
                    BUCKET_NAME, JOB_NAME)),
                'sagemaker_enable_cloudwatch_metrics': 'false',
                'sagemaker_container_log_level': str(logging.INFO),
                'sagemaker_job_name': json.dumps(JOB_NAME),
                'checkpoint_path': json.dumps('s3://{}/{}/checkpoints'.format(BUCKET_NAME, JOB_NAME)),
                'sagemaker_region': '"us-west-2"'
            },
            'stop_condition': {
                'MaxRuntimeInSeconds': 24 * 60 * 60
            }}


def _build_tf(sagemaker_session, framework_version=defaults.TF_VERSION, train_instance_type=None,
              checkpoint_path=None, enable_cloudwatch_metrics=False, base_job_name=None,
              training_steps=None, evalutation_steps=None, **kwargs):
    return TensorFlow(entry_point=SCRIPT_PATH,
                      training_steps=training_steps,
                      evaluation_steps=evalutation_steps,
                      framework_version=framework_version,
                      role=ROLE,
                      sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT,
                      train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
                      checkpoint_path=checkpoint_path,
                      enable_cloudwatch_metrics=enable_cloudwatch_metrics,
                      base_job_name=base_job_name,
                      **kwargs)


def test_tf_support_cpu_instances(sagemaker_session, tf_version):
    tf = _build_tf(sagemaker_session, tf_version, train_instance_type='ml.c2.2xlarge')

    assert tf.train_image() == _get_full_cpu_image_uri(tf_version)

    tf = _build_tf(sagemaker_session, tf_version, train_instance_type='ml.c4.2xlarge')

    assert tf.train_image() == _get_full_cpu_image_uri(tf_version)

    tf = _build_tf(sagemaker_session, tf_version, train_instance_type='ml.m16')

    assert tf.train_image() == _get_full_cpu_image_uri(tf_version)


def test_tf_support_gpu_instances(sagemaker_session, tf_version):
    tf = _build_tf(sagemaker_session, tf_version, train_instance_type='ml.g2.2xlarge')

    assert tf.train_image() == _get_full_gpu_image_uri(tf_version)

    tf = _build_tf(sagemaker_session, tf_version, train_instance_type='ml.p2.2xlarge')

    assert tf.train_image() == _get_full_gpu_image_uri(tf_version)


def test_tf_deploy_model_server_workers(sagemaker_session):
    tf = _build_tf(sagemaker_session)
    tf.fit(inputs=s3_input('s3://mybucket/train'))

    tf.deploy(initial_instance_count=1, instance_type='ml.c2.2xlarge', model_server_workers=2)

    assert "2" == sagemaker_session.method_calls[3][1][2]['Environment'][
        MODEL_SERVER_WORKERS_PARAM_NAME.upper()]


def test_tf_deploy_model_server_workers_unset(sagemaker_session):
    tf = _build_tf(sagemaker_session)
    tf.fit(inputs=s3_input('s3://mybucket/train'))

    tf.deploy(initial_instance_count=1, instance_type='ml.c2.2xlarge')

    assert MODEL_SERVER_WORKERS_PARAM_NAME.upper() not in sagemaker_session.method_calls[3][1][2]['Environment']


def test_create_model(sagemaker_session, tf_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    enable_cloudwatch_metrics = 'true'
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, framework_version=tf_version,
                    container_log_level=container_log_level, base_job_name='job',
                    source_dir=source_dir, enable_cloudwatch_metrics=enable_cloudwatch_metrics)

    job_name = 'doing something'
    tf.fit(inputs='s3://mybucket/train', job_name=job_name)
    model = tf.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == tf_version
    assert model.py_version == tf.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.enable_cloudwatch_metrics == enable_cloudwatch_metrics


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
def test_tf(time, strftime, sagemaker_session, tf_version):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, framework_version=tf_version)

    inputs = 's3://mybucket/train'

    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ['train', 'logs_for_job']
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ['resource']

    expected_train_args = _create_train_job(tf_version)
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = tf.create_model()

    assert {'Environment':
            {'SAGEMAKER_SUBMIT_DIRECTORY': 's3://{}/{}/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME),
             'SAGEMAKER_PROGRAM': 'dummy_script.py',
             'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
             'SAGEMAKER_REGION': 'us-west-2',
             'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
             },
            'Image': create_image_uri('us-west-2', "tensorflow", GPU_IMAGE_NAME, tf_version, "py2"),
            'ModelDataUrl': 's3://m/m.tar.gz'} == model.prepare_container_def(GPU_IMAGE_NAME)

    assert 'cpu' in model.prepare_container_def(CPU_IMAGE_NAME)['Image']
    predictor = tf.deploy(1, GPU_IMAGE_NAME)
    assert isinstance(predictor, TensorFlowPredictor)


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('subprocess.Popen')
@patch('subprocess.call')
@patch('os.access', return_value=False)
def test_run_tensorboard_locally_without_tensorboard_binary(time, strftime, popen, call, access, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    with pytest.raises(EnvironmentError) as error:
        tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)
    assert str(error.value) == 'TensorBoard is not installed in the system. Please install TensorBoard using the ' \
                               'following command: \n pip install tensorboard'


def test_model(sagemaker_session, tf_version):
    model = TensorFlowModel("s3://some/data.tar.gz", role=ROLE, entry_point=SCRIPT_PATH,
                            sagemaker_session=sagemaker_session)
    predictor = model.deploy(1, GPU_IMAGE_NAME)
    assert isinstance(predictor, TensorFlowPredictor)


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('subprocess.Popen')
@patch('subprocess.call')
@patch('os.access', side_effect=[False, True])
def test_run_tensorboard_locally_without_awscli_binary(time, strftime, popen, call, access, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    with pytest.raises(EnvironmentError) as error:
        tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)
    assert str(error.value) == 'The AWS CLI is not installed in the system. Please install the AWS CLI using the ' \
                               'following command: \n pip install awscli'


@patch('tempfile.mkdtemp', return_value='/my/temp/folder')
@patch('os.access', return_value=True)
@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@pytest.mark.skip(reason="this test fails sometimes and it needs further investigation")
def test_run_tensorboard_locally(time, strftime, popen, call, access, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)

    popen.assert_called_with(['tensorboard', '--logdir', '/my/temp/folder', '--host', 'localhost', '--port', '6006'],
                             stderr=-1,
                             stdout=-1
                             )


@patch('tempfile.mkdtemp', return_value='/my/temp/folder')
@patch('socket.socket')
@patch('os.access', return_value=True)
@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@pytest.mark.skip(reason="this test fails sometimes and it needs further investigation")
def test_run_tensorboard_locally_port_in_use(time, strftime, popen, call, access, socket, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    popen().poll.side_effect = [True, False]

    tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)

    popen.assert_any_call(['tensorboard', '--logdir', '/my/temp/folder', '--host', 'localhost', '--port', '6006'],
                          stderr=-1, stdout=-1)

    popen.assert_any_call(['tensorboard', '--logdir', '/my/temp/folder', '--host', 'localhost', '--port', '6007'],
                          stderr=-1, stdout=-1)


def test_tf_checkpoint_not_set(sagemaker_session):
    job_name = "sagemaker-tensorflow-py2-gpu-2017-10-24-14-12-09"
    tf = _build_tf(sagemaker_session, checkpoint_path=None, base_job_name=job_name,
                   output_path="s3://{}/".format(sagemaker_session.default_bucket()))
    tf.fit(inputs=s3_input('s3://mybucket/train'), job_name=job_name)

    expected_result = '"s3://{}/{}/checkpoints"'.format(sagemaker_session.default_bucket(), job_name)
    assert tf.hyperparameters()['checkpoint_path'] == expected_result


def test_tf_training_and_evaluation_steps_not_set(sagemaker_session):
    job_name = "sagemaker-tensorflow-py2-gpu-2017-10-24-14-12-09"
    output_path = "s3://{}/output/{}/".format(sagemaker_session.default_bucket(), job_name)

    tf = _build_tf(sagemaker_session, training_steps=None, evalutation_steps=None, output_path=output_path)
    tf.fit(inputs=s3_input('s3://mybucket/train'))
    assert tf.hyperparameters()['training_steps'] == 'null'
    assert tf.hyperparameters()['evaluation_steps'] == 'null'


def test_tf_training_and_evaluation_steps(sagemaker_session):
    job_name = "sagemaker-tensorflow-py2-gpu-2017-10-24-14-12-09"
    output_path = "s3://{}/output/{}/".format(sagemaker_session.default_bucket(), job_name)

    tf = _build_tf(sagemaker_session, training_steps=123, evalutation_steps=456, output_path=output_path)
    tf.fit(inputs=s3_input('s3://mybucket/train'))
    assert tf.hyperparameters()['training_steps'] == '123'
    assert tf.hyperparameters()['evaluation_steps'] == '456'


def test_tf_checkpoint_set(sagemaker_session):
    tf = _build_tf(sagemaker_session, checkpoint_path='s3://my_checkpoint_bucket')
    assert tf.hyperparameters()['checkpoint_path'] == json.dumps("s3://my_checkpoint_bucket")


def test_train_image_default(sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH,
                    role=ROLE,
                    sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE)

    assert _get_full_cpu_image_uri(defaults.TF_VERSION) in tf.train_image()


def test_attach(sagemaker_session, tf_version):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:{}-cpu-py2'.format(tf_version)
    rjd = {'AlgorithmSpecification':
           {'TrainingInputMode': 'File',
            'TrainingImage': training_image},
           'HyperParameters':
               {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                'checkpoint_path': '"s3://other/1508872349"',
                'sagemaker_program': '"iris-dnn-classifier.py"',
                'sagemaker_enable_cloudwatch_metrics': 'false',
                'sagemaker_container_log_level': '"logging.INFO"',
                'sagemaker_job_name': '"neo"',
                'training_steps': '100',
                'evaluation_steps': '10'},
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

    estimator = TensorFlow.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.py_version == 'py2'
    assert estimator.framework_version == tf_version
    assert estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert estimator.train_instance_count == 1
    assert estimator.train_max_run == 24 * 60 * 60
    assert estimator.input_mode == 'File'
    assert estimator.training_steps == 100
    assert estimator.evaluation_steps == 10
    assert estimator.input_mode == 'File'
    assert estimator.base_job_name == 'neo'
    assert estimator.output_path == 's3://place/output/neo'
    assert estimator.output_kms_key == ''
    assert estimator.hyperparameters()['training_steps'] == '100'
    assert estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert estimator.entry_point == 'iris-dnn-classifier.py'
    assert estimator.checkpoint_path == 's3://other/1508872349'


def test_attach_old_container(sagemaker_session):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:1.0'
    rjd = {'AlgorithmSpecification':
           {'TrainingInputMode': 'File',
            'TrainingImage': training_image},
           'HyperParameters':
               {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                'checkpoint_path': '"s3://other/1508872349"',
                'sagemaker_program': '"iris-dnn-classifier.py"',
                'sagemaker_enable_cloudwatch_metrics': 'false',
                'sagemaker_container_log_level': '"logging.INFO"',
                'sagemaker_job_name': '"neo"',
                'training_steps': '100',
                'evaluation_steps': '10'},
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

    estimator = TensorFlow.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.py_version == 'py2'
    assert estimator.framework_version == '1.4'
    assert estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert estimator.train_instance_count == 1
    assert estimator.train_max_run == 24 * 60 * 60
    assert estimator.input_mode == 'File'
    assert estimator.training_steps == 100
    assert estimator.evaluation_steps == 10
    assert estimator.input_mode == 'File'
    assert estimator.base_job_name == 'neo'
    assert estimator.output_path == 's3://place/output/neo'
    assert estimator.output_kms_key == ''
    assert estimator.hyperparameters()['training_steps'] == '100'
    assert estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert estimator.entry_point == 'iris-dnn-classifier.py'
    assert estimator.checkpoint_path == 's3://other/1508872349'


def test_attach_wrong_framework(sagemaker_session):
    returned_job_description = {'AlgorithmSpecification':
                                {'TrainingInputMode': 'File',
                                 'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0'},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
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

    with pytest.raises(ValueError) as error:
        TensorFlow.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)
