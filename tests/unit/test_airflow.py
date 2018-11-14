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

from mock import Mock, patch

from sagemaker.workflow.airflow import get_training_config
from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.amazon.ntm import NTM


@patch('sagemaker.session.Session.default_bucket', return_value='output')
def test_byo_training_config_required_args(default_bucket):
    byo = Estimator(image_name="byo",
                    role="{{ role }}",
                    train_instance_count="{{ instance_count }}",
                    train_instance_type="ml.c4.2xlarge")

    byo.set_hyperparameters(epochs=32,
                            feature_dim=1024,
                            mini_batch_size=256)

    data = {'train': "{{ training_data }}"}

    training_config = get_training_config(byo, data)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': 'byo',
            'TrainingInputMode': 'File'
        },
        'OutputDataConfig': {
            'S3OutputPath': 's3://output/'
        },
        'TrainingJobName': "byo-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400
        },
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': 30
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': '{{ training_data }}'
                }
            }, 'ChannelName': 'train'
        }],
        'HyperParameters': {
            'epochs': '32',
            'feature_dim': '1024',
            'mini_batch_size': '256'}
    }
    assert training_config == expected_config


def test_byo_training_config_all_args():
    byo = Estimator(image_name="byo",
                    role="{{ role }}",
                    train_instance_count="{{ instance_count }}",
                    train_instance_type="ml.c4.2xlarge",
                    train_volume_size="{{ train_volume_size }}",
                    train_volume_kms_key="{{ train_volume_kms_key }}",
                    train_max_run="{{ train_max_run }}",
                    input_mode='Pipe',
                    output_path="{{ output_path }}",
                    output_kms_key="{{ output_volume_kms_key }}",
                    base_job_name="{{ base_job_name }}",
                    tags=[{"{{ key }}":"{{ value }}"}],
                    subnets=["{{ subnet }}"],
                    security_group_ids=["{{ security_group_ids }}"],
                    model_uri="{{ model_uri }}",
                    model_channel_name="{{ model_chanel }}")

    byo.set_hyperparameters(epochs=32,
                            feature_dim=1024,
                            mini_batch_size=256)

    data = {
        'train': "{{ training_data }}",
        'evaluation': "{{ eval_data }}"
    }

    training_config = get_training_config(byo, data)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': 'byo',
            'TrainingInputMode': 'Pipe'
        },
        'OutputDataConfig': {
            'S3OutputPath': '{{ output_path }}',
            'KmsKeyId': '{{ output_volume_kms_key }}'
        },
        'TrainingJobName': "{{ base_job_name }}-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {
            'MaxRuntimeInSeconds': '{{ train_max_run }}'
        },
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': '{{ train_volume_size }}',
            'VolumeKmsKeyId': '{{ train_volume_kms_key }}'
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [
            {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix',
                        'S3Uri': '{{ training_data }}'
                    }
                },
                'ChannelName': 'train'
            },
            {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix',
                        'S3Uri': '{{ eval_data }}'
                    }
                },
                'ChannelName': 'evaluation'
            },
            {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix',
                        'S3Uri': '{{ model_uri }}'
                    }
                },
                'ContentType': 'application/x-sagemaker-model',
                'InputMode': 'File',
                'ChannelName': '{{ model_chanel }}'
            }
        ],
        'VpcConfig': {
            'Subnets': ['{{ subnet }}'],
            'SecurityGroupIds': ['{{ security_group_ids }}']
        },
        'HyperParameters': {
            'epochs': '32',
            'feature_dim': '1024',
            'mini_batch_size': '256'},
        'Tags': [{'{{ key }}': '{{ value }}'}]
    }
    assert training_config == expected_config


@patch('sagemaker.session.Session.default_bucket', return_value='output')
def test_framework_training_config_required_args(default_bucket):
    tf = TensorFlow(entry_point="{{ entry_point }}",
                    framework_version='1.10.0',
                    training_steps=1000,
                    evaluation_steps=100,
                    role="{{ role }}",
                    train_instance_count="{{ instance_count }}",
                    train_instance_type="ml.c4.2xlarge")

    data = "{{ training_data }}"

    training_config = get_training_config(tf, data)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2',
            'TrainingInputMode': 'File'
        },
        'OutputDataConfig': {
            'S3OutputPath': 's3://output/'
        },
        'TrainingJobName': "sagemaker-tensorflow-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400
        },
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': 30
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': '{{ training_data }}'
                }
            },
            'ChannelName': 'training'
        }],
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://output/sagemaker-tensorflow-'
                                          '{{ execution_date.strftime(\'%Y-%m-%d-%H-%M-%S\') }}'
                                          '/source/sourcedir.tar.gz"',
            'sagemaker_program': '"{{ entry_point }}"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '20',
            'sagemaker_job_name': '"sagemaker-tensorflow-{{ execution_date.strftime(\'%Y-%m-%d-%H-%M-%S\') }}"',
            'sagemaker_region': '"us-west-2"',
            'checkpoint_path': '"s3://output/sagemaker-tensorflow-{{ execution_date.strftime(\'%Y-%m-%d-%H-%M-%S\') }}'
                               '/checkpoints"',
            'training_steps': '1000',
            'evaluation_steps': '100',
            'sagemaker_requirements': '""'},
        'S3Operations': {
            'S3Upload': [{
                'Path': '{{ entry_point }}',
                'Bucket': 'output',
                'Key': "sagemaker-tensorflow-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}"
                       "/source/sourcedir.tar.gz",
                'Tar': True}]
        }
    }
    assert training_config == expected_config


def test_framework_training_config_all_args():
    tf = TensorFlow(entry_point="{{ entry_point }}",
                    source_dir="{{ source_dir }}",
                    enable_cloudwatch_metrics=False,
                    container_log_level="{{ log_level }}",
                    code_location="{{ bucket_name }}",
                    training_steps=1000,
                    evaluation_steps=100,
                    checkpoint_path="{{ checkpoint_path }}",
                    py_version='py2',
                    framework_version='1.10.0',
                    requirements_file="",
                    role="{{ role }}",
                    train_instance_count="{{ instance_count }}",
                    train_instance_type="ml.c4.2xlarge",
                    train_volume_size="{{ train_volume_size }}",
                    train_volume_kms_key="{{ train_volume_kms_key }}",
                    train_max_run="{{ train_max_run }}",
                    input_mode='Pipe',
                    output_path="{{ output_path }}",
                    output_kms_key="{{ output_volume_kms_key }}",
                    base_job_name="{{ base_job_name }}",
                    tags=[{"{{ key }}": "{{ value }}"}],
                    subnets=["{{ subnet }}"],
                    security_group_ids=["{{ security_group_ids }}"])

    data = "{{ training_data }}"

    training_config = get_training_config(tf, data)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2',
            'TrainingInputMode': 'Pipe'
        },
        'OutputDataConfig': {
            'S3OutputPath': '{{ output_path }}',
            'KmsKeyId': '{{ output_volume_kms_key }}'
        },
        'TrainingJobName': "{{ base_job_name }}-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {
            'MaxRuntimeInSeconds': '{{ train_max_run }}'
        },
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': '{{ train_volume_size }}',
            'VolumeKmsKeyId': '{{ train_volume_kms_key }}'
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': '{{ training_data }}'
                }
            },
            'ChannelName': 'training'
        }],
        'VpcConfig': {
            'Subnets': ['{{ subnet }}'],
            'SecurityGroupIds': ['{{ security_group_ids }}']
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://{{ bucket_name }}/{{ base_job_name }}-'
                                          '{{ execution_date.strftime(\'%Y-%m-%d-%H-%M-%S\') }}'
                                          '/source/sourcedir.tar.gz"',
            'sagemaker_program': '"{{ entry_point }}"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"{{ log_level }}"',
            'sagemaker_job_name': '"{{ base_job_name }}-{{ execution_date.strftime(\'%Y-%m-%d-%H-%M-%S\') }}"',
            'sagemaker_region': '"us-west-2"',
            'checkpoint_path': '"{{ checkpoint_path }}"',
            'training_steps': '1000',
            'evaluation_steps': '100',
            'sagemaker_requirements': '""'
        },
        'Tags': [{'{{ key }}': '{{ value }}'}],
        'S3Operations': {
            'S3Upload': [{
                'Path': '{{ source_dir }}',
                'Bucket': '{{ bucket_name }}',
                'Key': "{{ base_job_name }}-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}"
                       "/source/sourcedir.tar.gz",
                'Tar': True}]
        }
    }
    assert training_config == expected_config


@patch('sagemaker.session.Session.default_bucket', return_value='output')
def test_amazon_alg_training_config_required_args(default_bucket):
    ntm = NTM(role="{{ role }}",
              num_topics=10,
              train_instance_count="{{ instance_count }}",
              train_instance_type="ml.c4.2xlarge")

    ntm.epochs = 32
    ntm.mini_batch_size = 256

    record = RecordSet("{{ record }}", 10000, 100, 'S3Prefix')

    training_config = get_training_config(ntm, record)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:1',
            'TrainingInputMode': 'File'
        },
        'OutputDataConfig': {
            'S3OutputPath': 's3://output/'
        },
        'TrainingJobName': "ntm-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {'MaxRuntimeInSeconds': 86400},
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': 30
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'ShardedByS3Key',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': '{{ record }}'
                }
            },
            'ChannelName': 'train'
        }],
        'HyperParameters': {
            'num_topics': '10',
            'epochs': '32',
            'mini_batch_size': '256',
            'feature_dim': '100'
        }
    }
    assert training_config == expected_config


def test_amazon_alg_training_config_all_args():
    ntm = NTM(role="{{ role }}",
              num_topics=10,
              train_instance_count="{{ instance_count }}",
              train_instance_type="ml.c4.2xlarge",
              train_volume_size="{{ train_volume_size }}",
              train_volume_kms_key="{{ train_volume_kms_key }}",
              train_max_run="{{ train_max_run }}",
              input_mode='Pipe',
              output_path="{{ output_path }}",
              output_kms_key="{{ output_volume_kms_key }}",
              base_job_name="{{ base_job_name }}",
              tags=[{"{{ key }}": "{{ value }}"}],
              subnets=["{{ subnet }}"],
              security_group_ids=["{{ security_group_ids }}"])

    ntm.epochs = 32
    ntm.mini_batch_size = 256

    record = RecordSet("{{ record }}", 10000, 100, 'S3Prefix')

    training_config = get_training_config(ntm, record)
    expected_config = {
        'AlgorithmSpecification': {
            'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:1',
            'TrainingInputMode': 'Pipe'
        },
        'OutputDataConfig': {
            'S3OutputPath': '{{ output_path }}',
            'KmsKeyId': '{{ output_volume_kms_key }}'
        },
        'TrainingJobName': "{{ base_job_name }}-{{ execution_date.strftime('%Y-%m-%d-%H-%M-%S') }}",
        'StoppingCondition': {
            'MaxRuntimeInSeconds': '{{ train_max_run }}'
        },
        'ResourceConfig': {
            'InstanceCount': '{{ instance_count }}',
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': '{{ train_volume_size }}',
            'VolumeKmsKeyId': '{{ train_volume_kms_key }}'
        },
        'RoleArn': '{{ role }}',
        'InputDataConfig': [{
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'ShardedByS3Key',
                    'S3DataType': 'S3Prefix',
                    'S3Uri': '{{ record }}'
                }
            },
            'ChannelName': 'train'
        }],
        'VpcConfig': {
            'Subnets': ['{{ subnet }}'],
            'SecurityGroupIds': ['{{ security_group_ids }}']
        },
        'HyperParameters': {
            'num_topics': '10',
            'epochs': '32',
            'mini_batch_size': '256',
            'feature_dim': '100'
        },
        'Tags': [{'{{ key }}': '{{ value }}'}]
    }

    assert training_config == expected_config