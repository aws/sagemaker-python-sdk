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

import json
import logging
import os

import pytest
from mock import patch, Mock

from sagemaker.fw_utils import create_image_uri, UploadedCode
from sagemaker.model import MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.session import s3_input
from sagemaker.tensorflow import defaults, TensorFlow, TensorFlowModel, TensorFlowPredictor
import sagemaker.tensorflow.estimator as tfe

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_FILE = 'dummy_script.py'
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_FILE)
REQUIREMENTS_FILE = 'dummy_requirements.txt'
TIMESTAMP = '2017-11-06-14:14:15.673'
TIME = 1510006209.073025
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
IMAGE_REPO_NAME = 'sagemaker-tensorflow'
SM_IMAGE_REPO_NAME = 'sagemaker-tensorflow-scriptmode'
JOB_NAME = '{}-{}'.format(IMAGE_REPO_NAME, TIMESTAMP)
SM_JOB_NAME = '{}-{}'.format(SM_IMAGE_REPO_NAME, TIMESTAMP)
ROLE = 'Dummy'
REGION = 'us-west-2'
DOCKER_TAG = '1.0'
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"
SCRIPT_MODE_REPO_NAME = 'sagemaker-tensorflow-scriptmode'
DISTRIBUTION_ENABLED = {'parameter_server': {'enabled': True}}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    session = Mock(name='sagemaker_session', boto_session=boto_mock,
                   boto_region_name=REGION, config=None, local_mode=False)
    session.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    describe = {'ModelArtifacts': {'S3ModelArtifacts': 's3://m/m.tar.gz'}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    return session


def _get_full_cpu_image_uri(version, repo=IMAGE_REPO_NAME, py_version='py2'):
    return IMAGE_URI_FORMAT_STRING.format(REGION, repo, version, 'cpu', py_version)


def _get_full_gpu_image_uri(version, repo=IMAGE_REPO_NAME, py_version='py2'):
    return IMAGE_URI_FORMAT_STRING.format(REGION, repo, version, 'gpu', py_version)


def _hyperparameters(script_mode=False):
    job_name = SM_JOB_NAME if script_mode else JOB_NAME
    hps = {
        'sagemaker_program': json.dumps('dummy_script.py'),
        'sagemaker_submit_directory': json.dumps('s3://{}/{}/source/sourcedir.tar.gz'.format(
            BUCKET_NAME, job_name)),
        'sagemaker_enable_cloudwatch_metrics': 'false',
        'sagemaker_container_log_level': str(logging.INFO),
        'sagemaker_job_name': json.dumps(job_name),
        'sagemaker_region': json.dumps('us-west-2')
    }
    if script_mode:
        hps['model_dir'] = json.dumps('s3://{}/{}/model'.format(BUCKET_NAME, job_name))
    else:
        hps['checkpoint_path'] = json.dumps('s3://{}/{}/checkpoints'.format(BUCKET_NAME, job_name))
        hps['training_steps'] = '1000'
        hps['evaluation_steps'] = '10'
        hps['sagemaker_requirements'] = '"{}"'.format(REQUIREMENTS_FILE)
    return hps


def _create_train_job(tf_version, script_mode=False, repo_name=IMAGE_REPO_NAME, py_version='py2'):
    return {
        'image': _get_full_cpu_image_uri(tf_version, repo=repo_name, py_version=py_version),
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
        'job_name': '{}-{}'.format(repo_name, TIMESTAMP),
        'output_config': {
            'S3OutputPath': 's3://{}/'.format(BUCKET_NAME),
        },
        'resource_config': {
            'InstanceType': 'ml.c4.4xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30,
        },
        'hyperparameters': _hyperparameters(script_mode),
        'stop_condition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        },
        'tags': None,
        'vpc_config': None,
        'metric_definitions': None
    }


def _build_tf(sagemaker_session, framework_version=defaults.TF_VERSION, train_instance_type=None,
              checkpoint_path=None, base_job_name=None,
              training_steps=None, evaluation_steps=None, **kwargs):
    return TensorFlow(entry_point=SCRIPT_PATH,
                      training_steps=training_steps,
                      evaluation_steps=evaluation_steps,
                      framework_version=framework_version,
                      role=ROLE,
                      sagemaker_session=sagemaker_session,
                      train_instance_count=INSTANCE_COUNT,
                      train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
                      checkpoint_path=checkpoint_path,
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


def test_tf_invalid_requirements_path(sagemaker_session):
    requirements_file = '/foo/bar/requirements.txt'
    with pytest.raises(ValueError) as e:
        _build_tf(sagemaker_session, requirements_file=requirements_file, source_dir=DATA_DIR)
    assert 'Requirements file {} is not a path relative to source_dir.'.format(requirements_file) in str(e.value)


def test_tf_nonexistent_requirements_path(sagemaker_session):
    requirements_file = 'nonexistent_requirements.txt'
    with pytest.raises(ValueError) as e:
        _build_tf(sagemaker_session, requirements_file=requirements_file, source_dir=DATA_DIR)
    assert 'Requirements file {} does not exist.'.format(requirements_file) in str(e.value)


def test_create_model(sagemaker_session, tf_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, framework_version=tf_version,
                    container_log_level=container_log_level, base_job_name='job',
                    source_dir=source_dir)

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
    assert model.vpc_config is None


def test_create_model_with_optional_params(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    enable_cloudwatch_metrics = 'true'
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, container_log_level=container_log_level, base_job_name='job',
                    source_dir=source_dir, enable_cloudwatch_metrics=enable_cloudwatch_metrics)

    job_name = 'doing something'
    tf.fit(inputs='s3://mybucket/train', job_name=job_name)

    new_role = 'role'
    model_server_workers = 2
    vpc_config = {'Subnets': ['foo'], 'SecurityGroupIds': ['bar']}
    model = tf.create_model(role=new_role, model_server_workers=model_server_workers,
                            vpc_config_override=vpc_config)

    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    custom_image = 'tensorflow:1.0'
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, image_name=custom_image,
                    container_log_level=container_log_level, base_job_name='job',
                    source_dir=source_dir)

    job_name = 'doing something'
    tf.fit(inputs='s3://mybucket/train', job_name=job_name)
    model = tf.create_model()

    assert model.image == custom_image


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('sagemaker.estimator.tar_and_upload_dir')
@patch('sagemaker.model.tar_and_upload_dir')
def test_tf(m_tar, e_tar, time, strftime, sagemaker_session, tf_version):
    tf = TensorFlow(entry_point=SCRIPT_FILE, role=ROLE, sagemaker_session=sagemaker_session, training_steps=1000,
                    evaluation_steps=10, train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                    framework_version=tf_version, requirements_file=REQUIREMENTS_FILE, source_dir=DATA_DIR)

    inputs = 's3://mybucket/train'
    s3_prefix = 's3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME)
    e_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    s3_prefix = 's3://{}/{}/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME)
    m_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ['train', 'logs_for_job']

    expected_train_args = _create_train_job(tf_version)
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = tf.create_model()

    environment = {
        'Environment': {
            'SAGEMAKER_SUBMIT_DIRECTORY': 's3://{}/{}/sourcedir.tar.gz'.format(BUCKET_NAME, JOB_NAME),
            'SAGEMAKER_PROGRAM': 'dummy_script.py', 'SAGEMAKER_REQUIREMENTS': 'dummy_requirements.txt',
            'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false', 'SAGEMAKER_REGION': 'us-west-2',
            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
        },
        'Image': create_image_uri('us-west-2', "tensorflow", INSTANCE_TYPE, tf_version, "py2"),
        'ModelDataUrl': 's3://m/m.tar.gz'
    }
    assert environment == model.prepare_container_def(INSTANCE_TYPE)

    assert 'cpu' in model.prepare_container_def(INSTANCE_TYPE)['Image']
    predictor = tf.deploy(1, INSTANCE_TYPE)
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
    predictor = model.deploy(1, INSTANCE_TYPE)
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


@patch('sagemaker.tensorflow.estimator.Tensorboard._sync_directories')
@patch('tempfile.mkdtemp', return_value='/my/temp/folder')
@patch('shutil.rmtree')
@patch('os.access', return_value=True)
@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('time.sleep')
def test_run_tensorboard_locally(sleep, time, strftime, popen, call, access, rmtree, mkdtemp, sync, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    popen().poll.return_value = None

    tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)

    popen.assert_called_with(['tensorboard', '--logdir', '/my/temp/folder', '--host', 'localhost', '--port', '6006'],
                             stderr=-1,
                             stdout=-1)


@patch('sagemaker.tensorflow.estimator.Tensorboard._sync_directories')
@patch('tempfile.mkdtemp', return_value='/my/temp/folder')
@patch('shutil.rmtree')
@patch('socket.socket')
@patch('os.access', return_value=True)
@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('time.sleep')
def test_run_tensorboard_locally_port_in_use(sleep, time, strftime, popen, call, access, socket, rmtree, mkdtemp, sync,
                                             sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)

    popen().poll.side_effect = [-1, None]

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

    tf = _build_tf(sagemaker_session, training_steps=None, evaluation_steps=None, output_path=output_path)
    tf.fit(inputs=s3_input('s3://mybucket/train'))
    assert tf.hyperparameters()['training_steps'] == 'null'
    assert tf.hyperparameters()['evaluation_steps'] == 'null'


def test_tf_training_and_evaluation_steps(sagemaker_session):
    job_name = "sagemaker-tensorflow-py2-gpu-2017-10-24-14-12-09"
    output_path = "s3://{}/output/{}/".format(sagemaker_session.default_bucket(), job_name)

    tf = _build_tf(sagemaker_session, training_steps=123, evaluation_steps=456, output_path=output_path)
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
    rjd = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': training_image
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            'evaluation_steps': '10'
        },
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'
        },
        'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {'KmsKeyId': '', 'S3OutputPath': 's3://place/output/neo'},
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


def test_attach_new_repo_name(sagemaker_session, tf_version):
    training_image = '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:{}-cpu-py2'.format(tf_version)
    rjd = {
        'AlgorithmSpecification': {'TrainingInputMode': 'File', 'TrainingImage': training_image},
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            'evaluation_steps': '10'
        },
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'
        },
        'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {'KmsKeyId': '', 'S3OutputPath': 's3://place/output/neo'},
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
    assert estimator.train_image() == training_image


def test_attach_old_container(sagemaker_session):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:1.0'
    rjd = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': training_image},
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            'evaluation_steps': '10'},
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'},
        'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {'KmsKeyId': '', 'S3OutputPath': 's3://place/output/neo'},
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
    returned_job_description = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0'
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'training_steps': '100'

        },
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig':
            {'VolumeSizeInGB': 30,
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
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=returned_job_description)

    with pytest.raises(ValueError) as error:
        TensorFlow.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/tensorflow_with_custom_binary:1.0'
    rjd = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': training_image},
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            'evaluation_steps': '10'},
        'RoleArn': 'arn:aws:iam::366:role/SageMakerRole',
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'},
        'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {'KmsKeyId': '', 'S3OutputPath': 's3://place/output/neo'},
        'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job', return_value=rjd)

    estimator = TensorFlow.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.image_name == training_image
    assert estimator.train_image() == training_image


@patch('sagemaker.fw_utils.empty_framework_version_warning')
def test_empty_framework_version(warning, sagemaker_session):
    estimator = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                           train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                           framework_version=None)

    assert estimator.framework_version == defaults.TF_VERSION
    warning.assert_called_with(defaults.TF_VERSION, defaults.TF_VERSION)


def _deprecated_args_msg(args):
    return '{} are deprecated in script mode. Please do not set {}.'.format(
        ', '.join(tfe._FRAMEWORK_MODE_ARGS), args)


def test_script_mode_deprecated_args(sagemaker_session):
    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session=sagemaker_session, py_version='py3', checkpoint_path='some_path')
    assert _deprecated_args_msg('checkpoint_path') in str(e.value)

    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session=sagemaker_session, py_version='py3', training_steps=1)
    assert _deprecated_args_msg('training_steps') in str(e.value)

    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session=sagemaker_session, script_mode=True, evaluation_steps=1)
    assert _deprecated_args_msg('evaluation_steps') in str(e.value)

    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session=sagemaker_session, script_mode=True, requirements_file='some_file')
    assert _deprecated_args_msg('requirements_file') in str(e.value)

    with pytest.raises(AttributeError) as e:
        _build_tf(sagemaker_session=sagemaker_session, script_mode=True, checkpoint_path='some_path',
                  requirements_file='some_file', training_steps=1, evaluation_steps=1)
    assert _deprecated_args_msg('training_steps, evaluation_steps, requirements_file, checkpoint_path') in str(e.value)


def test_script_mode_enabled(sagemaker_session):
    tf = _build_tf(sagemaker_session=sagemaker_session, py_version='py3')
    assert tf._script_mode_enabled() is True

    tf = _build_tf(sagemaker_session=sagemaker_session, script_mode=True)
    assert tf._script_mode_enabled() is True

    tf = _build_tf(sagemaker_session=sagemaker_session)
    assert tf._script_mode_enabled() is False


@patch('sagemaker.tensorflow.estimator.TensorFlow._create_tfs_model')
def test_script_mode_create_model(create_tfs_model, sagemaker_session):
    tf = _build_tf(sagemaker_session=sagemaker_session, py_version='py3')
    tf.create_model()
    create_tfs_model.assert_called_once()


@patch('sagemaker.tensorflow.estimator.Tensorboard._sync_directories')
@patch('sagemaker.tensorflow.estimator.Tensorboard.start')
@patch('tempfile.mkdtemp', return_value='/my/temp/folder')
@patch('shutil.rmtree')
@patch('os.access', return_value=True)
@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('time.sleep')
def test_script_mode_tensorboard(sleep, time, strftime, popen, call, access, rmtree, mkdtemp,
                                 start, sync, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                    framework_version='some_version', script_mode=True)
    popen().poll.return_value = None
    tf.fit(inputs='s3://mybucket/train', run_tensorboard_locally=True)
    start.assert_not_called()


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('sagemaker.estimator.tar_and_upload_dir')
@patch('sagemaker.model.tar_and_upload_dir')
def test_tf_script_mode(m_tar, e_tar, time, strftime, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_FILE, role=ROLE, sagemaker_session=sagemaker_session, py_version='py3',
                    train_instance_type=INSTANCE_TYPE, train_instance_count=1, framework_version='1.11',
                    source_dir=DATA_DIR)

    inputs = 's3://mybucket/train'
    s3_prefix = 's3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, SM_JOB_NAME)
    e_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    s3_prefix = 's3://{}/{}/sourcedir.tar.gz'.format(BUCKET_NAME, SM_JOB_NAME)
    m_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ['train', 'logs_for_job']

    expected_train_args = _create_train_job('1.11', script_mode=True, repo_name=SM_IMAGE_REPO_NAME, py_version='py3')
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args


@patch('time.strftime', return_value=TIMESTAMP)
@patch('time.time', return_value=TIME)
@patch('sagemaker.estimator.tar_and_upload_dir')
@patch('sagemaker.model.tar_and_upload_dir')
def test_tf_script_mode_ps(m_tar, e_tar, time, strftime, sagemaker_session):
    tf = TensorFlow(entry_point=SCRIPT_FILE, role=ROLE, sagemaker_session=sagemaker_session, py_version='py3',
                    train_instance_type=INSTANCE_TYPE, train_instance_count=1, framework_version='1.11',
                    source_dir=DATA_DIR, distributions=DISTRIBUTION_ENABLED)

    inputs = 's3://mybucket/train'
    s3_prefix = 's3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, SM_JOB_NAME)
    e_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    s3_prefix = 's3://{}/{}/sourcedir.tar.gz'.format(BUCKET_NAME, SM_JOB_NAME)
    m_tar.return_value = UploadedCode(s3_prefix=s3_prefix, script_name=SCRIPT_FILE)
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ['train', 'logs_for_job']

    expected_train_args = _create_train_job('1.11', script_mode=True, repo_name=SM_IMAGE_REPO_NAME, py_version='py3')
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs
    expected_train_args['hyperparameters'][TensorFlow.LAUNCH_PS_ENV_NAME] = json.dumps(True)

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args
