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

import datetime
import io
import logging

import pytest
import six
from botocore.exceptions import ClientError
from mock import ANY, MagicMock, Mock, patch, call, mock_open

import sagemaker
from sagemaker import s3_input, Session, get_execution_role
from sagemaker.session import _tuning_job_status, _transform_job_status, _train_done
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

STATIC_HPs = {"feature_dim": "784", }

SAMPLE_PARAM_RANGES = [{"Name": "mini_batch_size", "MinValue": "10", "MaxValue": "100"}]

REGION = 'us-west-2'


@pytest.fixture()
def boto_session():
    boto_session = Mock(region_name=REGION)

    mock_client = Mock()
    mock_client._client_config.user_agent = \
        'Boto3/1.9.69 Python/3.6.5 Linux/4.14.77-70.82.amzn1.x86_64 Botocore/1.12.69 Resource'

    boto_session.client.return_value = mock_client
    return boto_session


def test_get_execution_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = 'arn:aws:iam::369233609183:role/SageMakerRole'

    actual = get_execution_role(session)
    assert actual == 'arn:aws:iam::369233609183:role/SageMakerRole'


def test_get_execution_role_works_with_service_role():
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


def test_get_execution_role_throws_exception_if_arn_is_not_role_with_role_in_name():
    session = Mock()
    session.get_caller_identity_arn.return_value = 'arn:aws:iam::369233609183:user/marcos-role'

    with pytest.raises(ValueError) as error:
        get_execution_role(session)
    assert 'ValueError: The current AWS identity is not a role' in str(error)


def test_get_caller_identity_arn_from_an_user(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:iam::369233609183:user/mia'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}
    sess.boto_session.client('iam').get_role.return_value = {'Role': {'Arn': arn}}

    actual = sess.get_caller_identity_arn()
    assert actual == 'arn:aws:iam::369233609183:user/mia'


def test_get_caller_identity_arn_from_an_user_without_permissions(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:iam::369233609183:user/mia'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}
    sess.boto_session.client('iam').get_role.side_effect = ClientError({}, {})

    with patch('logging.Logger.warning') as mock_logger:
        actual = sess.get_caller_identity_arn()
        assert actual == 'arn:aws:iam::369233609183:user/mia'
        mock_logger.assert_called_once()


def test_get_caller_identity_arn_from_a_role(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:sts::369233609183:assumed-role/SageMakerRole/6d009ef3-5306-49d5-8efc-78db644d8122'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}

    expected_role = 'arn:aws:iam::369233609183:role/SageMakerRole'
    sess.boto_session.client('iam').get_role.return_value = {'Role': {'Arn': expected_role}}

    actual = sess.get_caller_identity_arn()
    assert actual == expected_role


def test_get_caller_identity_arn_from_a_execution_role(boto_session):
    sess = Session(boto_session)
    arn = 'arn:aws:sts::369233609183:assumed-role/AmazonSageMaker-ExecutionRole-20171129T072388/SageMaker'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': arn}
    sess.boto_session.client('iam').get_role.return_value = {'Role': {'Arn': arn}}

    actual = sess.get_caller_identity_arn()
    assert actual == 'arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388'


def test_get_caller_identity_arn_from_role_with_path(boto_session):
    sess = Session(boto_session)
    arn_prefix = 'arn:aws:iam::369233609183:role'
    role_name = 'name'
    sess.boto_session.client('sts').get_caller_identity.return_value = {'Arn': '/'.join([arn_prefix, role_name])}

    role_path = 'path'
    role_with_path = '/'.join([arn_prefix, role_path, role_name])
    sess.boto_session.client('iam').get_role.return_value = {'Role': {'Arn': role_with_path}}

    actual = sess.get_caller_identity_arn()
    assert actual == role_with_path


def test_delete_endpoint(boto_session):
    sess = Session(boto_session)
    sess.delete_endpoint('my_endpoint')

    boto_session.client().delete_endpoint.assert_called_with(EndpointName='my_endpoint')


def test_delete_endpoint_config(boto_session):
    sess = Session(boto_session)
    sess.delete_endpoint_config('my_endpoint_config')

    boto_session.client().delete_endpoint_config.assert_called_with(EndpointConfigName='my_endpoint_config')


def test_delete_model(boto_session):
    sess = Session(boto_session)

    model_name = 'my_model'
    sess.delete_model(model_name)

    boto_session.client().delete_model.assert_called_with(ModelName=model_name)


def test_user_agent_injected(boto_session):
    assert 'AWS-SageMaker-Python-SDK' not in boto_session.client('sagemaker')._client_config.user_agent

    sess = Session(boto_session)

    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_runtime_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' not in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' not in sess.sagemaker_runtime_client._client_config.user_agent


def test_user_agent_injected_with_nbi(boto_session):
    assert 'AWS-SageMaker-Python-SDK' not in boto_session.client('sagemaker')._client_config.user_agent

    with patch('six.moves.builtins.open', mock_open(read_data='120.0-0')) as mo:
        sess = Session(boto_session)

        mo.assert_called_with('/etc/opt/ml/sagemaker-notebook-instance-version.txt')

    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_runtime_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' in sess.sagemaker_runtime_client._client_config.user_agent


def test_user_agent_injected_with_nbi_ioerror(boto_session):
    assert 'AWS-SageMaker-Python-SDK' not in boto_session.client('sagemaker')._client_config.user_agent

    with patch('six.moves.builtins.open', MagicMock(side_effect=IOError('File not found'))) as mo:
        sess = Session(boto_session)

        mo.assert_called_with('/etc/opt/ml/sagemaker-notebook-instance-version.txt')

    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Python-SDK' in sess.sagemaker_runtime_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' not in sess.sagemaker_client._client_config.user_agent
    assert 'AWS-SageMaker-Notebook-Instance' not in sess.sagemaker_runtime_client._client_config.user_agent


def test_s3_input_all_defaults():
    prefix = 'pre'
    actual = s3_input(s3_data=prefix)
    expected = {
        'DataSource': {
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
    input_mode = 'Pipe'
    result = s3_input(s3_data=prefix, distribution=distribution, compression=compression, input_mode=input_mode,
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
            'RecordWrapperType': record_wrapping,
            'InputMode': input_mode
        }

    assert result.config == expected


IMAGE = 'myimage'
S3_INPUT_URI = 's3://mybucket/data'
S3_OUTPUT = 's3://sagemaker-123/output/jobname'
ROLE = 'SageMakerRole'
EXPANDED_ROLE = 'arn:aws:iam::111111111111:role/ExpandedRole'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.xlarge'
ACCELERATOR_TYPE = 'ml.eia.medium'
MAX_SIZE = 30
MAX_TIME = 3 * 60 * 60
JOB_NAME = 'jobname'
TAGS = [{'Name': 'some-tag', 'Value': 'value-for-tag'}]
VPC_CONFIG = {'Subnets': ['foo'], 'SecurityGroupIds': ['bar']}
METRIC_DEFINITONS = [{'Name': 'validation-rmse', 'Regex': 'validation-rmse=(\\d+)'}]

DEFAULT_EXPECTED_TRAIN_JOB_ARGS = {
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
    },
    'VpcConfig': VPC_CONFIG,
}

COMPLETED_DESCRIBE_JOB_RESULT = dict(DEFAULT_EXPECTED_TRAIN_JOB_ARGS)
COMPLETED_DESCRIBE_JOB_RESULT.update({'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:336:training-job/' + JOB_NAME})
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

STOPPED_DESCRIBE_JOB_RESULT = dict(COMPLETED_DESCRIBE_JOB_RESULT)
STOPPED_DESCRIBE_JOB_RESULT.update({'TrainingJobStatus': 'Stopped'})

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
                            hyperparameters=None, stop_condition=stop_cond, tags=None, vpc_config=VPC_CONFIG,
                            metric_definitions=None)

    assert sagemaker_session.sagemaker_client.method_calls[0] == (
        'create_training_job', (), DEFAULT_EXPECTED_TRAIN_JOB_ARGS)


SAMPLE_STOPPING_CONDITION = {'MaxRuntimeInSeconds': MAX_TIME}

RESOURCE_CONFIG = {'InstanceCount': INSTANCE_COUNT, 'InstanceType': INSTANCE_TYPE, 'VolumeSizeInGB': MAX_SIZE}

SAMPLE_INPUT = [{'DataSource': {
    'S3DataSource': {'S3DataDistributionType': 'FullyReplicated', 'S3DataType': 'S3Prefix', 'S3Uri': S3_INPUT_URI}},
    'ChannelName': 'training'}]

SAMPLE_OUTPUT = {'S3OutputPath': S3_OUTPUT}

SAMPLE_OBJECTIVE = {'Type': "Maximize", 'MetricName': "val-score", }

SAMPLE_METRIC_DEF = [{"Name": "train:progress", "Regex": "regex-1"}]

SAMPLE_TUNING_JOB_REQUEST = {
    'HyperParameterTuningJobName': 'dummy-tuning-1',
    'HyperParameterTuningJobConfig': {
        'Strategy': "Bayesian",
        'HyperParameterTuningJobObjective': SAMPLE_OBJECTIVE,
        'ResourceLimits': {
            'MaxNumberOfTrainingJobs': 100,
            'MaxParallelTrainingJobs': 5,
        },
        'ParameterRanges': SAMPLE_PARAM_RANGES,
        'TrainingJobEarlyStoppingType': 'Off'
    },
    'TrainingJobDefinition': {
        'StaticHyperParameters': STATIC_HPs,
        'AlgorithmSpecification': {
            'TrainingImage': "dummy-image-1",
            'TrainingInputMode': "File",
            'MetricDefinitions': SAMPLE_METRIC_DEF
        },
        'RoleArn': EXPANDED_ROLE,
        'InputDataConfig': SAMPLE_INPUT,
        'OutputDataConfig': SAMPLE_OUTPUT,

        'ResourceConfig': RESOURCE_CONFIG,
        'StoppingCondition': SAMPLE_STOPPING_CONDITION
    }
}


@pytest.mark.parametrize('warm_start_type, parents', [
    ("IdenticalDataAndAlgorithm", {"p1", "p2", "p3"}),
    ("TransferLearning", {"p1", "p2", "p3"}),
])
def test_tune_warm_start(sagemaker_session, warm_start_type, parents):

    def assert_create_tuning_job_request(**kwrags):
        assert kwrags["HyperParameterTuningJobConfig"] == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"] == SAMPLE_TUNING_JOB_REQUEST["TrainingJobDefinition"]
        assert kwrags["WarmStartConfig"] == {
            'WarmStartType': warm_start_type,
            'ParentHyperParameterTuningJobs': [{'HyperParameterTuningJobName': parent} for parent in parents]
        }

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = assert_create_tuning_job_request
    sagemaker_session.tune(job_name="dummy-tuning-1",
                           strategy="Bayesian",
                           objective_type="Maximize",
                           objective_metric_name="val-score",
                           max_jobs=100,
                           max_parallel_jobs=5,
                           parameter_ranges=SAMPLE_PARAM_RANGES,
                           static_hyperparameters=STATIC_HPs,
                           image="dummy-image-1",
                           input_mode="File",
                           metric_definitions=SAMPLE_METRIC_DEF,
                           role=EXPANDED_ROLE,
                           input_config=SAMPLE_INPUT,
                           output_config=SAMPLE_OUTPUT,
                           resource_config=RESOURCE_CONFIG,
                           stop_condition=SAMPLE_STOPPING_CONDITION,
                           tags=None,
                           warm_start_config=WarmStartConfig(warm_start_type=WarmStartTypes(warm_start_type),
                                                             parents=parents).to_input_req())


def test_tune(sagemaker_session):

    def assert_create_tuning_job_request(**kwrags):
        assert kwrags["HyperParameterTuningJobConfig"] == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"] == SAMPLE_TUNING_JOB_REQUEST["TrainingJobDefinition"]
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = assert_create_tuning_job_request
    sagemaker_session.tune(job_name="dummy-tuning-1",
                           strategy="Bayesian",
                           objective_type="Maximize",
                           objective_metric_name="val-score",
                           max_jobs=100,
                           max_parallel_jobs=5,
                           parameter_ranges=SAMPLE_PARAM_RANGES,
                           static_hyperparameters=STATIC_HPs,
                           image="dummy-image-1",
                           input_mode="File",
                           metric_definitions=SAMPLE_METRIC_DEF,
                           role=EXPANDED_ROLE,
                           input_config=SAMPLE_INPUT,
                           output_config=SAMPLE_OUTPUT,
                           resource_config=RESOURCE_CONFIG,
                           stop_condition=SAMPLE_STOPPING_CONDITION,
                           tags=None,
                           warm_start_config=None)


def test_tune_with_encryption_flag(sagemaker_session):

    def assert_create_tuning_job_request(**kwrags):
        assert kwrags["HyperParameterTuningJobConfig"] == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"]["EnableInterContainerTrafficEncryption"] is True
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = assert_create_tuning_job_request
    sagemaker_session.tune(job_name="dummy-tuning-1",
                           strategy="Bayesian",
                           objective_type="Maximize",
                           objective_metric_name="val-score",
                           max_jobs=100,
                           max_parallel_jobs=5,
                           parameter_ranges=SAMPLE_PARAM_RANGES,
                           static_hyperparameters=STATIC_HPs,
                           image="dummy-image-1",
                           input_mode="File",
                           metric_definitions=SAMPLE_METRIC_DEF,
                           role=EXPANDED_ROLE,
                           input_config=SAMPLE_INPUT,
                           output_config=SAMPLE_OUTPUT,
                           resource_config=RESOURCE_CONFIG,
                           stop_condition=SAMPLE_STOPPING_CONDITION,
                           tags=None,
                           warm_start_config=None,
                           encrypt_inter_container_traffic=True)


def test_stop_tuning_job(sagemaker_session):
    sms = sagemaker_session
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(name='stop_hyper_parameter_tuning_job')

    sagemaker_session.stop_tuning_job(JOB_NAME)
    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(HyperParameterTuningJobName=JOB_NAME)


def test_stop_tuning_job_client_error_already_stopped(sagemaker_session):
    sms = sagemaker_session
    exception = ClientError({'Error': {'Code': 'ValidationException'}}, 'Operation')
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(name='stop_hyper_parameter_tuning_job',
                                                                side_effect=exception)
    sagemaker_session.stop_tuning_job(JOB_NAME)

    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(HyperParameterTuningJobName=JOB_NAME)


def test_stop_tuning_job_client_error(sagemaker_session):
    error_response = {'Error': {'Code': 'MockException', 'Message': 'MockMessage'}}
    operation = 'Operation'
    exception = ClientError(error_response, operation)

    sms = sagemaker_session
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(name='stop_hyper_parameter_tuning_job',
                                                                side_effect=exception)

    with pytest.raises(ClientError) as e:
        sagemaker_session.stop_tuning_job(JOB_NAME)

    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(HyperParameterTuningJobName=JOB_NAME)
    assert 'An error occurred (MockException) when calling the Operation operation: MockMessage' in str(e)


def test_train_pack_to_request_with_optional_params(sagemaker_session):
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
    hyperparameters = {'foo': 'bar'}

    sagemaker_session.train(image=IMAGE, input_mode='File', input_config=in_config, role=EXPANDED_ROLE,
                            job_name=JOB_NAME, output_config=out_config, resource_config=resource_config,
                            vpc_config=VPC_CONFIG, hyperparameters=hyperparameters, stop_condition=stop_cond, tags=TAGS,
                            metric_definitions=METRIC_DEFINITONS, encrypt_inter_container_traffic=True)

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    assert actual_train_args['VpcConfig'] == VPC_CONFIG
    assert actual_train_args['HyperParameters'] == hyperparameters
    assert actual_train_args['Tags'] == TAGS
    assert actual_train_args['AlgorithmSpecification']['MetricDefinitions'] == METRIC_DEFINITONS
    assert actual_train_args['EnableInterContainerTrafficEncryption'] is True


def test_transform_pack_to_request(sagemaker_session):
    model_name = 'my-model'

    in_config = {
        'CompressionType': 'None',
        'ContentType': 'text/csv',
        'SplitType': 'None',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': S3_INPUT_URI,
            },
        },
    }

    out_config = {'S3OutputPath': S3_OUTPUT}

    resource_config = {
        'InstanceCount': INSTANCE_COUNT,
        'InstanceType': INSTANCE_TYPE,
    }

    expected_args = {
        'TransformJobName': JOB_NAME,
        'ModelName': model_name,
        'TransformInput': in_config,
        'TransformOutput': out_config,
        'TransformResources': resource_config,
    }

    sagemaker_session.transform(job_name=JOB_NAME, model_name=model_name, strategy=None, max_concurrent_transforms=None,
                                max_payload=None, env=None, input_config=in_config, output_config=out_config,
                                resource_config=resource_config, tags=None, data_processing=None)

    _, _, actual_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_args == expected_args


def test_transform_pack_to_request_with_optional_params(sagemaker_session):
    strategy = 'strategy'
    max_concurrent_transforms = 1
    max_payload = 0
    env = {'FOO': 'BAR'}

    sagemaker_session.transform(job_name=JOB_NAME, model_name='my-model', strategy=strategy,
                                max_concurrent_transforms=max_concurrent_transforms,
                                env=env, max_payload=max_payload, input_config={}, output_config={},
                                resource_config={}, tags=TAGS, data_processing=None)

    _, _, actual_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_args['BatchStrategy'] == strategy
    assert actual_args['MaxConcurrentTransforms'] == max_concurrent_transforms
    assert actual_args['MaxPayloadInMB'] == max_payload
    assert actual_args['Environment'] == env
    assert actual_args['Tags'] == TAGS


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
def sagemaker_session_stopped():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('logs').describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client('logs').get_log_events.side_effect = DEFAULT_LOG_EVENTS
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    ims.sagemaker_client.describe_training_job.return_value = STOPPED_DESCRIBE_JOB_RESULT
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
def test_logs_for_job_no_wait_stopped_job(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
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
def test_logs_for_job_wait_on_stopped(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
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


MODEL_NAME = 'some-model'
PRIMARY_CONTAINER = {
    'Environment': {},
    'Image': IMAGE,
    'ModelDataUrl': 's3://sagemaker-123/output/jobname/model/model.tar.gz',
}


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       PrimaryContainer=PRIMARY_CONTAINER)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_with_tags(expand_container_def, sagemaker_session):
    tags = [{'Key': 'TagtestKey', 'Value': 'TagtestValue'}]
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER, tags=tags)

    assert model == MODEL_NAME
    tags = [{'Value': 'TagtestValue', 'Key': 'TagtestKey'}]
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       PrimaryContainer=PRIMARY_CONTAINER,
                                                                       Tags=tags)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_with_primary_container(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, container_defs=PRIMARY_CONTAINER)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       PrimaryContainer=PRIMARY_CONTAINER)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_with_both(expand_container_def, sagemaker_session):
    with pytest.raises(ValueError):
        sagemaker_session.create_model(MODEL_NAME, ROLE, container_defs=PRIMARY_CONTAINER,
                                       primary_container=PRIMARY_CONTAINER)


CONTAINERS = [
    {
        'Environment': {'SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT': 'application/json'},
        'Image': 'mi-1',
        'ModelDataUrl': 's3://bucket/model_1.tar.gz'
    },
    {
        'Environment': {},
        'Image': 'mi-2',
        'ModelDataUrl': 's3://bucket/model_2.tar.gz'
    }
]


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_pipeline_model(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, container_defs=CONTAINERS)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       Containers=CONTAINERS)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_vpc_config(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER, VPC_CONFIG)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       PrimaryContainer=PRIMARY_CONTAINER,
                                                                       VpcConfig=VPC_CONFIG)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_pipeline_model_vpc_config(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, CONTAINERS, VPC_CONFIG)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                                       ModelName=MODEL_NAME,
                                                                       Containers=CONTAINERS,
                                                                       VpcConfig=VPC_CONFIG)


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_already_exists(expand_container_def, sagemaker_session, caplog):
    error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Cannot create already existing model'}}
    exception = ClientError(error_response, 'Operation')
    sagemaker_session.sagemaker_client.create_model.side_effect = exception

    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)
    assert model == MODEL_NAME

    expected_warning = ('sagemaker', logging.WARNING, 'Using already existing model: {}'.format(MODEL_NAME))
    assert expected_warning in caplog.record_tuples


@patch('sagemaker.session._expand_container_def', return_value=PRIMARY_CONTAINER)
def test_create_model_failure(expand_container_def, sagemaker_session):
    error_message = 'this is expected'
    sagemaker_session.sagemaker_client.create_model.side_effect = RuntimeError(error_message)

    with pytest.raises(RuntimeError) as e:
        sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)

    assert error_message in str(e)


def test_create_model_from_job(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME)

    assert call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    ims.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                         ModelName=JOB_NAME,
                                                         PrimaryContainer=PRIMARY_CONTAINER,
                                                         VpcConfig=VPC_CONFIG)


def test_create_model_from_job_with_tags(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, tags=TAGS)

    assert call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    ims.sagemaker_client.create_model.assert_called_with(ExecutionRoleArn=EXPANDED_ROLE,
                                                         ModelName=JOB_NAME,
                                                         PrimaryContainer=PRIMARY_CONTAINER,
                                                         VpcConfig=VPC_CONFIG,
                                                         Tags=TAGS)


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


def test_create_model_from_job_with_vpc_config_override(sagemaker_session):
    vpc_config_override = {'Subnets': ['foo', 'bar'], 'SecurityGroupIds': ['baz']}

    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, vpc_config_override=vpc_config_override)
    assert ims.sagemaker_client.create_model.call_args[1]['VpcConfig'] == vpc_config_override

    ims.create_model_from_job(JOB_NAME, vpc_config_override=None)
    assert 'VpcConfig' not in ims.sagemaker_client.create_model.call_args[1]


def test_endpoint_from_production_variants(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={'EndpointStatus': 'InService'})
    pvs = [sagemaker.production_variant('A', 'ml.p2.xlarge'), sagemaker.production_variant('B', 'p299.4096xlarge')]
    ex = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find your thing'}}, 'b')
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    sagemaker_session.endpoint_from_production_variants('some-endpoint', pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(EndpointConfigName='some-endpoint',
                                                                          EndpointName='some-endpoint',
                                                                          Tags=[])
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName='some-endpoint',
        ProductionVariants=pvs)


def test_create_endpoint_config_with_tags(sagemaker_session):
    tags = [{'Key': 'TagtestKey', 'Value': 'TagtestValue'}]

    sagemaker_session.create_endpoint_config('endpoint-test', 'simple-model', 1, 'local', tags=tags)

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName='endpoint-test',
        ProductionVariants=ANY,
        Tags=tags)


def test_endpoint_from_production_variants_with_tags(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={'EndpointStatus': 'InService'})
    pvs = [sagemaker.production_variant('A', 'ml.p2.xlarge'), sagemaker.production_variant('B', 'p299.4096xlarge')]
    ex = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find your thing'}}, 'b')
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    tags = [{'ModelName': 'TestModel'}]
    sagemaker_session.endpoint_from_production_variants('some-endpoint', pvs, tags)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(EndpointConfigName='some-endpoint',
                                                                          EndpointName='some-endpoint',
                                                                          Tags=tags)
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName='some-endpoint',
        ProductionVariants=pvs,
        Tags=tags)


def test_endpoint_from_production_variants_with_accelerator_type(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={'EndpointStatus': 'InService'})
    pvs = [sagemaker.production_variant('A', 'ml.p2.xlarge', accelerator_type=ACCELERATOR_TYPE),
           sagemaker.production_variant('B', 'p299.4096xlarge', accelerator_type=ACCELERATOR_TYPE)]
    ex = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find your thing'}}, 'b')
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    tags = [{'ModelName': 'TestModel'}]
    sagemaker_session.endpoint_from_production_variants('some-endpoint', pvs, tags)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(EndpointConfigName='some-endpoint',
                                                                          EndpointName='some-endpoint',
                                                                          Tags=tags)
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName='some-endpoint',
        ProductionVariants=pvs,
        Tags=tags)


def test_update_endpoint_succeed(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(return_value={'EndpointStatus': 'InService'})
    endpoint_name = "some-endpoint"
    endpoint_config = "some-endpoint-config"
    returned_endpoint_name = sagemaker_session.update_endpoint(endpoint_name, endpoint_config)
    assert returned_endpoint_name == endpoint_name


def test_update_endpoint_non_existing_endpoint(sagemaker_session):
    error = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find entity'}}, 'foo')
    expected_error_message = 'Endpoint with name "non-existing-endpoint" does not exist; ' \
                             'please use an existing endpoint name'
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(side_effect=error)
    with pytest.raises(ValueError, match=expected_error_message):
        sagemaker_session.update_endpoint("non-existing-endpoint", "non-existing-config")


@patch('time.sleep')
def test_wait_for_tuning_job(sleep, sagemaker_session):
    hyperparameter_tuning_job_desc = {'HyperParameterTuningJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=hyperparameter_tuning_job_desc)

    result = sagemaker_session.wait_for_tuning_job(JOB_NAME)
    assert result['HyperParameterTuningJobStatus'] == 'Completed'


def test_tune_job_status(sagemaker_session):
    hyperparameter_tuning_job_desc = {'HyperParameterTuningJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=hyperparameter_tuning_job_desc)

    result = _tuning_job_status(sagemaker_session.sagemaker_client, JOB_NAME)

    assert result['HyperParameterTuningJobStatus'] == 'Completed'


def test_tune_job_status_none(sagemaker_session):
    hyperparameter_tuning_job_desc = {'HyperParameterTuningJobStatus': 'InProgress'}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=hyperparameter_tuning_job_desc)

    result = _tuning_job_status(sagemaker_session.sagemaker_client, JOB_NAME)

    assert result is None


@patch('time.sleep')
def test_wait_for_transform_job_completed(sleep, sagemaker_session):
    transform_job_desc = {'TransformJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name='describe_transform_job', return_value=transform_job_desc)

    assert sagemaker_session.wait_for_transform_job(JOB_NAME)['TransformJobStatus'] == 'Completed'


@patch('time.sleep')
def test_wait_for_transform_job_in_progress(sleep, sagemaker_session):
    transform_job_desc_in_progress = {'TransformJobStatus': 'InProgress'}
    transform_job_desc_in_completed = {'TransformJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name='describe_transform_job', side_effect=[transform_job_desc_in_progress,
                                                    transform_job_desc_in_completed])

    assert sagemaker_session.wait_for_transform_job(JOB_NAME, 1)['TransformJobStatus'] == 'Completed'
    assert 2 == sagemaker_session.sagemaker_client.describe_transform_job.call_count


def test_transform_job_status(sagemaker_session):
    transform_job_desc = {'TransformJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name='describe_transform_job', return_value=transform_job_desc)

    result = _transform_job_status(sagemaker_session.sagemaker_client, JOB_NAME)
    assert result['TransformJobStatus'] == 'Completed'


def test_transform_job_status_none(sagemaker_session):
    transform_job_desc = {'TransformJobStatus': 'InProgress'}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name='describe_transform_job', return_value=transform_job_desc)

    result = _transform_job_status(sagemaker_session.sagemaker_client, JOB_NAME)
    assert result is None


def test_train_done_completed(sagemaker_session):
    training_job_desc = {'TrainingJobStatus': 'Completed'}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name='describe_training_job', return_value=training_job_desc)

    actual_job_desc, training_finished = _train_done(sagemaker_session.sagemaker_client, JOB_NAME, None)

    assert actual_job_desc['TrainingJobStatus'] == 'Completed'
    assert training_finished is True


def test_train_done_in_progress(sagemaker_session):
    training_job_desc = {'TrainingJobStatus': 'InProgress'}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name='describe_training_job', return_value=training_job_desc)

    actual_job_desc, training_finished = _train_done(sagemaker_session.sagemaker_client, JOB_NAME, None)

    assert actual_job_desc['TrainingJobStatus'] == 'InProgress'
    assert training_finished is False
