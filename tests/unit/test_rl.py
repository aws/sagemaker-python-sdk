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
from mock import MagicMock, Mock
from mock import patch

from sagemaker.mxnet import MXNetModel, MXNetPredictor
from sagemaker.rl import RLEstimator, RLFramework, RLToolkit, TOOLKIT_FRAMEWORK_VERSION_MAP
import sagemaker.tensorflow.serving as tfs


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SCRIPT_PATH = os.path.join(DATA_DIR, 'dummy_script.py')
TIMESTAMP = '2017-11-06-14:14:15.672'
TIME = 1507167947
BUCKET_NAME = 'notmybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
IMAGE_NAME = 'sagemaker-rl'
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}-{}:{}{}-{}-py3"
PYTHON_VERSION = 'py3'
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


@pytest.fixture(name='sagemaker_session')
def fixture_sagemaker_session():
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


def _get_full_cpu_image_uri(toolkit, toolkit_version, framework):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, framework,
                                          toolkit, toolkit_version, 'cpu')


def _get_full_gpu_image_uri(toolkit, toolkit_version, framework):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, framework,
                                          toolkit, toolkit_version, 'gpu')


def _rl_estimator(sagemaker_session, toolkit=RLToolkit.COACH,
                  toolkit_version=RLEstimator.COACH_LATEST_VERSION_MXNET, framework=RLFramework.MXNET,
                  train_instance_type=None, base_job_name=None, **kwargs):
    return RLEstimator(entry_point=SCRIPT_PATH,
                       toolkit=toolkit,
                       toolkit_version=toolkit_version,
                       framework=framework,
                       role=ROLE,
                       sagemaker_session=sagemaker_session,
                       train_instance_count=INSTANCE_COUNT,
                       train_instance_type=train_instance_type or INSTANCE_TYPE,
                       base_job_name=base_job_name,
                       **kwargs)


def _create_train_job(toolkit, toolkit_version, framework):
    job_name = '{}-{}-{}'.format(IMAGE_NAME, framework, TIMESTAMP)
    return {
        'image': _get_full_cpu_image_uri(toolkit, toolkit_version, framework),
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
        'job_name': job_name,
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
            'sagemaker_estimator': '"RLEstimator"',
            'sagemaker_container_log_level': str(logging.INFO),
            'sagemaker_job_name': json.dumps(job_name),
            'sagemaker_s3_output': '"s3://{}/"'.format(BUCKET_NAME),
            'sagemaker_submit_directory':
                json.dumps('s3://{}/{}/source/sourcedir.tar.gz'.format(BUCKET_NAME, job_name)),
            'sagemaker_region': '"us-west-2"'
        },
        'stop_condition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        },
        'tags': None,
        'vpc_config': None,
        'metric_definitions': [
            {'Name': 'reward-training',
             'Regex': '^Training>.*Total reward=(.*?),'},
            {'Name': 'reward-testing',
             'Regex': '^Testing>.*Total reward=(.*?),'}
        ]
    }


def test_create_tf_model(sagemaker_session, rl_coach_tf_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    rl = RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                     train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                     toolkit=RLToolkit.COACH, toolkit_version=rl_coach_tf_version,
                     framework=RLFramework.TENSORFLOW, container_log_level=container_log_level,
                     source_dir=source_dir)

    job_name = 'new_name'
    rl.fit(inputs='s3://mybucket/train', job_name='new_name')
    model = rl.create_model()
    supported_versions = TOOLKIT_FRAMEWORK_VERSION_MAP[RLToolkit.COACH.value]
    framework_version = supported_versions[rl_coach_tf_version][RLFramework.TENSORFLOW.value]

    assert isinstance(model, tfs.Model)
    assert model.sagemaker_session == sagemaker_session
    assert model._framework_version == framework_version
    assert model.role == ROLE
    assert model.name == job_name
    assert model._container_log_level == container_log_level
    assert model.vpc_config is None


def test_create_mxnet_model(sagemaker_session, rl_coach_mxnet_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    rl = RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                     train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                     toolkit=RLToolkit.COACH, toolkit_version=rl_coach_mxnet_version,
                     framework=RLFramework.MXNET, container_log_level=container_log_level,
                     source_dir=source_dir)

    job_name = 'new_name'
    rl.fit(inputs='s3://mybucket/train', job_name='new_name')
    model = rl.create_model()
    supported_versions = TOOLKIT_FRAMEWORK_VERSION_MAP[RLToolkit.COACH.value]
    framework_version = supported_versions[rl_coach_mxnet_version][RLFramework.MXNET.value]

    assert isinstance(model, MXNetModel)
    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == framework_version
    assert model.py_version == PYTHON_VERSION
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.vpc_config is None


def test_create_model_with_optional_params(sagemaker_session, rl_coach_mxnet_version):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    rl = RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                     train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                     toolkit=RLToolkit.COACH, toolkit_version=rl_coach_mxnet_version,
                     framework=RLFramework.MXNET, container_log_level=container_log_level,
                     source_dir=source_dir)

    rl.fit(job_name='new_name')

    new_role = 'role'
    new_entry_point = 'deploy_script.py'
    vpc_config = {'Subnets': ['foo'], 'SecurityGroupIds': ['bar']}
    model = rl.create_model(role=new_role, entry_point=new_entry_point,
                            vpc_config_override=vpc_config)

    assert model.role == new_role
    assert model.vpc_config == vpc_config
    assert model.entry_point == new_entry_point


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    image = 'selfdrivingcars:9000'
    rl = RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                     train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                     image_name=image, container_log_level=container_log_level,
                     source_dir=source_dir)

    job_name = 'new_name'
    rl.fit(job_name=job_name)
    new_entry_point = 'deploy_script.py'
    model = rl.create_model(entry_point=new_entry_point)

    assert model.sagemaker_session == sagemaker_session
    assert model.image == image
    assert model.entry_point == new_entry_point
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir


@patch('sagemaker.utils.create_tar_file', MagicMock())
@patch('time.strftime', return_value=TIMESTAMP)
def test_rl(strftime, sagemaker_session, rl_coach_mxnet_version):
    rl = RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                     train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                     toolkit=RLToolkit.COACH, toolkit_version=rl_coach_mxnet_version,
                     framework=RLFramework.MXNET)

    inputs = 's3://mybucket/train'

    rl.fit(inputs=inputs)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ['train', 'logs_for_job']
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ['resource']

    expected_train_args = _create_train_job(RLToolkit.COACH.value, rl_coach_mxnet_version,
                                            RLFramework.MXNET.value)
    expected_train_args['input_config'][0]['DataSource']['S3DataSource']['S3Uri'] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = rl.create_model()
    supported_versions = TOOLKIT_FRAMEWORK_VERSION_MAP[RLToolkit.COACH.value]
    framework_version = supported_versions[rl_coach_mxnet_version][RLFramework.MXNET.value]

    expected_image_base = '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:{}-gpu-py3'
    submit_dir = 's3://notmybucket/sagemaker-rl-mxnet-{}/source/sourcedir.tar.gz'.format(TIMESTAMP)
    assert {'Environment': {'SAGEMAKER_SUBMIT_DIRECTORY': submit_dir,
                            'SAGEMAKER_PROGRAM': 'dummy_script.py',
                            'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
                            'SAGEMAKER_REGION': 'us-west-2',
                            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'},
            'Image': expected_image_base.format(framework_version),
            'ModelDataUrl': 's3://m/m.tar.gz'} == model.prepare_container_def(GPU)

    assert 'cpu' in model.prepare_container_def(CPU)['Image']


@patch('sagemaker.utils.create_tar_file', MagicMock())
def test_deploy_mxnet(sagemaker_session, rl_coach_mxnet_version):
    rl = _rl_estimator(sagemaker_session, RLToolkit.COACH, rl_coach_mxnet_version, RLFramework.MXNET,
                       train_instance_type='ml.g2.2xlarge')
    rl.fit()
    predictor = rl.deploy(1, CPU)
    assert isinstance(predictor, MXNetPredictor)


@patch('sagemaker.utils.create_tar_file', MagicMock())
def test_deploy_tfs(sagemaker_session, rl_coach_tf_version):
    rl = _rl_estimator(sagemaker_session, RLToolkit.COACH, rl_coach_tf_version, RLFramework.TENSORFLOW,
                       train_instance_type='ml.g2.2xlarge')
    rl.fit()
    predictor = rl.deploy(1, GPU)
    assert isinstance(predictor, tfs.Predictor)


@patch('sagemaker.utils.create_tar_file', MagicMock())
def test_deploy_ray(sagemaker_session, rl_ray_version):
    rl = _rl_estimator(sagemaker_session, RLToolkit.RAY, rl_ray_version, RLFramework.TENSORFLOW,
                       train_instance_type='ml.g2.2xlarge')
    rl.fit()
    with pytest.raises(NotImplementedError) as e:
        rl.deploy(1, GPU)
    assert 'deployment of Ray models is not currently available' in str(e.value)


def test_train_image_cpu_instances(sagemaker_session, rl_ray_version):
    toolkit = RLToolkit.RAY
    framework = RLFramework.TENSORFLOW
    rl = _rl_estimator(sagemaker_session, toolkit, rl_ray_version, framework,
                       train_instance_type='ml.c2.2xlarge')
    assert rl.train_image() == _get_full_cpu_image_uri(toolkit.value, rl_ray_version,
                                                       framework.value)

    rl = _rl_estimator(sagemaker_session, toolkit, rl_ray_version, framework,
                       train_instance_type='ml.c4.2xlarge')
    assert rl.train_image() == _get_full_cpu_image_uri(toolkit.value, rl_ray_version,
                                                       framework.value)

    rl = _rl_estimator(sagemaker_session, toolkit, rl_ray_version, framework,
                       train_instance_type='ml.m16')
    assert rl.train_image() == _get_full_cpu_image_uri(toolkit.value, rl_ray_version,
                                                       framework.value)


def test_train_image_gpu_instances(sagemaker_session, rl_coach_mxnet_version):
    toolkit = RLToolkit.COACH
    framework = RLFramework.MXNET
    rl = _rl_estimator(sagemaker_session, toolkit, rl_coach_mxnet_version, framework,
                       train_instance_type='ml.g2.2xlarge')
    assert rl.train_image() == _get_full_gpu_image_uri(toolkit.value, rl_coach_mxnet_version,
                                                       framework.value)

    rl = _rl_estimator(sagemaker_session, toolkit, rl_coach_mxnet_version, framework,
                       train_instance_type='ml.p2.2xlarge')
    assert rl.train_image() == _get_full_gpu_image_uri(toolkit.value, rl_coach_mxnet_version,
                                                       framework.value)


def test_attach(sagemaker_session, rl_coach_mxnet_version):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-{}:{}{}-cpu-py3'\
        .format(RLFramework.MXNET.value, RLToolkit.COACH.value, rl_coach_mxnet_version)
    supported_versions = TOOLKIT_FRAMEWORK_VERSION_MAP[RLToolkit.COACH.value]
    framework_version = supported_versions[rl_coach_mxnet_version][RLFramework.MXNET.value]
    returned_job_description = {'AlgorithmSpecification': {'TrainingInputMode': 'File',
                                                           'TrainingImage': training_image},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'sagemaker_program': '"train_coach.py"',
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
                                'TrainingJobOutput': {
                                    'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = \
        Mock(name='describe_training_job', return_value=returned_job_description)

    estimator = RLEstimator.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.framework == RLFramework.MXNET.value
    assert estimator.toolkit == RLToolkit.COACH.value
    assert estimator.framework_version == framework_version
    assert estimator.toolkit_version == rl_coach_mxnet_version
    assert estimator.role == 'arn:aws:iam::366:role/SageMakerRole'
    assert estimator.train_instance_count == 1
    assert estimator.train_max_run == 24 * 60 * 60
    assert estimator.input_mode == 'File'
    assert estimator.base_job_name == 'neo'
    assert estimator.output_path == 's3://place/output/neo'
    assert estimator.output_kms_key == ''
    assert estimator.hyperparameters()['training_steps'] == '100'
    assert estimator.source_dir == 's3://some/sourcedir.tar.gz'
    assert estimator.entry_point == 'train_coach.py'
    assert estimator.metric_definitions == RLEstimator.default_metric_definitions(RLToolkit.COACH)


def test_attach_wrong_framework(sagemaker_session):
    training_image = '1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0.4'
    rjd = {'AlgorithmSpecification': {'TrainingInputMode': 'File',
                                      'TrainingImage': training_image},
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
    sagemaker_session.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                                    return_value=rjd)

    with pytest.raises(ValueError) as error:
        RLEstimator.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = 'rl:latest'
    returned_job_description = {'AlgorithmSpecification': {'TrainingInputMode': 'File',
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
                                'OutputDataConfig':
                                    {'KmsKeyId': '',
                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput':
                                    {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sagemaker_session.sagemaker_client.describe_training_job = \
        Mock(name='describe_training_job', return_value=returned_job_description)

    estimator = RLEstimator.attach(training_job_name='neo', sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == 'neo'
    assert estimator.image_name == training_image
    assert estimator.train_image() == training_image


def test_wrong_framework_format(sagemaker_session):
    with pytest.raises(ValueError) as e:
        RLEstimator(toolkit=RLToolkit.RAY, framework='TF',
                    toolkit_version=RLEstimator.RAY_LATEST_VERSION,
                    entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                    framework_version=None)

    assert 'Invalid type' in str(e.value)


def test_wrong_toolkit_format(sagemaker_session):
    with pytest.raises(ValueError) as e:
        RLEstimator(toolkit='coach', framework=RLFramework.TENSORFLOW,
                    toolkit_version=RLEstimator.COACH_LATEST_VERSION_TF,
                    entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE,
                    framework_version=None)

    assert 'Invalid type' in str(e.value)


def test_missing_required_parameters(sagemaker_session):
    with pytest.raises(AttributeError) as e:
        RLEstimator(entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)
    assert 'Please provide `toolkit`, `toolkit_version`, `framework`' + \
           ' or `image_name` parameter.' in str(e.value)


def test_wrong_type_parameters(sagemaker_session):
    with pytest.raises(AttributeError) as e:
        RLEstimator(toolkit=RLToolkit.COACH, framework=RLFramework.TENSORFLOW,
                    toolkit_version=RLEstimator.RAY_LATEST_VERSION,
                    entry_point=SCRIPT_PATH, role=ROLE, sagemaker_session=sagemaker_session,
                    train_instance_count=INSTANCE_COUNT, train_instance_type=INSTANCE_TYPE)
    assert 'combination is not supported.' in str(e.value)
