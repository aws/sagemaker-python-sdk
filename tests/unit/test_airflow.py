# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mock import Mock, MagicMock, patch

from sagemaker import chainer, estimator, model, mxnet, tensorflow, transformer, tuner
from sagemaker.workflow import airflow
from sagemaker.amazon import amazon_estimator
from sagemaker.amazon import knn, linear_learner, ntm, pca


REGION = "us-west-2"
BUCKET_NAME = "output"
TIME_STAMP = "1111"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session._default_bucket = BUCKET_NAME
    return session


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_byo_training_config_required_args(sagemaker_session):
    byo = estimator.Estimator(
        image_name="byo",
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        sagemaker_session=sagemaker_session,
    )

    byo.set_hyperparameters(epochs=32, feature_dim=1024, mini_batch_size=256)

    data = {"train": "{{ training_data }}"}

    config = airflow.training_config(byo, data)
    expected_config = {
        "AlgorithmSpecification": {"TrainingImage": "byo", "TrainingInputMode": "File"},
        "OutputDataConfig": {"S3OutputPath": "s3://output/"},
        "TrainingJobName": "byo-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 30,
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ training_data }}",
                    }
                },
                "ChannelName": "train",
            }
        ],
        "HyperParameters": {"epochs": "32", "feature_dim": "1024", "mini_batch_size": "256"},
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_byo_training_config_all_args(sagemaker_session):
    byo = estimator.Estimator(
        image_name="byo",
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        train_volume_size="{{ train_volume_size }}",
        train_volume_kms_key="{{ train_volume_kms_key }}",
        train_max_run="{{ train_max_run }}",
        input_mode="Pipe",
        output_path="{{ output_path }}",
        output_kms_key="{{ output_volume_kms_key }}",
        base_job_name="{{ base_job_name }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        subnets=["{{ subnet }}"],
        security_group_ids=["{{ security_group_ids }}"],
        model_uri="{{ model_uri }}",
        model_channel_name="{{ model_chanel }}",
        sagemaker_session=sagemaker_session,
    )

    byo.set_hyperparameters(epochs=32, feature_dim=1024, mini_batch_size=256)

    data = {"train": "{{ training_data }}"}

    config = airflow.training_config(byo, data)
    expected_config = {
        "AlgorithmSpecification": {"TrainingImage": "byo", "TrainingInputMode": "Pipe"},
        "OutputDataConfig": {
            "S3OutputPath": "{{ output_path }}",
            "KmsKeyId": "{{ output_volume_kms_key }}",
        },
        "TrainingJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": "{{ train_max_run }}"},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": "{{ train_volume_size }}",
            "VolumeKmsKeyId": "{{ train_volume_kms_key }}",
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ training_data }}",
                    }
                },
                "ChannelName": "train",
            },
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ model_uri }}",
                    }
                },
                "ContentType": "application/x-sagemaker-model",
                "InputMode": "File",
                "ChannelName": "{{ model_chanel }}",
            },
        ],
        "VpcConfig": {
            "Subnets": ["{{ subnet }}"],
            "SecurityGroupIds": ["{{ security_group_ids }}"],
        },
        "HyperParameters": {"epochs": "32", "feature_dim": "1024", "mini_batch_size": "256"},
        "Tags": [{"{{ key }}": "{{ value }}"}],
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "sagemaker-tensorflow-{}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="520713654638.dkr.ecr.us-west-2.amazonaws.com",
)
def test_framework_training_config_required_args(ecr_prefix, sagemaker_session):
    tf = tensorflow.TensorFlow(
        entry_point="/some/script.py",
        framework_version="1.10.0",
        training_steps=1000,
        evaluation_steps=100,
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        sagemaker_session=sagemaker_session,
    )

    data = "{{ training_data }}"

    config = airflow.training_config(tf, data)
    expected_config = {
        "AlgorithmSpecification": {
            "TrainingImage": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2",
            "TrainingInputMode": "File",
        },
        "OutputDataConfig": {"S3OutputPath": "s3://output/"},
        "TrainingJobName": "sagemaker-tensorflow-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 30,
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ training_data }}",
                    }
                },
                "ChannelName": "training",
            }
        ],
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://output/sagemaker-tensorflow-%s/source/sourcedir.tar.gz"'
            % TIME_STAMP,
            "sagemaker_program": '"script.py"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": "20",
            "sagemaker_job_name": '"sagemaker-tensorflow-%s"' % TIME_STAMP,
            "sagemaker_region": '"us-west-2"',
            "checkpoint_path": '"s3://output/sagemaker-tensorflow-%s/checkpoints"' % TIME_STAMP,
            "training_steps": "1000",
            "evaluation_steps": "100",
            "sagemaker_requirements": '""',
        },
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "/some/script.py",
                    "Bucket": "output",
                    "Key": "sagemaker-tensorflow-%s/source/sourcedir.tar.gz" % TIME_STAMP,
                    "Tar": True,
                }
            ]
        },
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.estimator.parse_s3_url",
    MagicMock(return_value=["{{ output_path }}", "{{ output_path }}"]),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="520713654638.dkr.ecr.us-west-2.amazonaws.com",
)
def test_framework_training_config_all_args(ecr_prefix, sagemaker_session):
    tf = tensorflow.TensorFlow(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        enable_cloudwatch_metrics=False,
        container_log_level="{{ log_level }}",
        code_location="s3://{{ bucket_name }}/{{ prefix }}",
        training_steps=1000,
        evaluation_steps=100,
        checkpoint_path="{{ checkpoint_path }}",
        py_version="py2",
        framework_version="1.10.0",
        requirements_file="",
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        train_volume_size="{{ train_volume_size }}",
        train_volume_kms_key="{{ train_volume_kms_key }}",
        train_max_run="{{ train_max_run }}",
        input_mode="Pipe",
        output_path="{{ output_path }}",
        output_kms_key="{{ output_volume_kms_key }}",
        base_job_name="{{ base_job_name }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        subnets=["{{ subnet }}"],
        security_group_ids=["{{ security_group_ids }}"],
        metric_definitions=[{"Name": "{{ name }}", "Regex": "{{ regex }}"}],
        sagemaker_session=sagemaker_session,
    )

    data = "{{ training_data }}"

    config = airflow.training_config(tf, data)
    expected_config = {
        "AlgorithmSpecification": {
            "TrainingImage": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2",
            "TrainingInputMode": "Pipe",
            "MetricDefinitions": [{"Name": "{{ name }}", "Regex": "{{ regex }}"}],
        },
        "OutputDataConfig": {
            "S3OutputPath": "{{ output_path }}",
            "KmsKeyId": "{{ output_volume_kms_key }}",
        },
        "TrainingJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": "{{ train_max_run }}"},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": "{{ train_volume_size }}",
            "VolumeKmsKeyId": "{{ train_volume_kms_key }}",
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ training_data }}",
                    }
                },
                "ChannelName": "training",
            }
        ],
        "VpcConfig": {
            "Subnets": ["{{ subnet }}"],
            "SecurityGroupIds": ["{{ security_group_ids }}"],
        },
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://{{ bucket_name }}/{{ prefix }}/{{ base_job_name }}-%s/'
            'source/sourcedir.tar.gz"' % TIME_STAMP,
            "sagemaker_program": '"{{ entry_point }}"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": '"{{ log_level }}"',
            "sagemaker_job_name": '"{{ base_job_name }}-%s"' % TIME_STAMP,
            "sagemaker_region": '"us-west-2"',
            "checkpoint_path": '"{{ checkpoint_path }}"',
            "training_steps": "1000",
            "evaluation_steps": "100",
            "sagemaker_requirements": '""',
        },
        "Tags": [{"{{ key }}": "{{ value }}"}],
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "{{ source_dir }}",
                    "Bucket": "{{ bucket_name }}",
                    "Key": "{{ prefix }}/{{ base_job_name }}-%s/source/sourcedir.tar.gz"
                    % TIME_STAMP,
                    "Tar": True,
                }
            ]
        },
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_amazon_alg_training_config_required_args(sagemaker_session):
    ntm_estimator = ntm.NTM(
        role="{{ role }}",
        num_topics=10,
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        sagemaker_session=sagemaker_session,
    )

    ntm_estimator.epochs = 32

    record = amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix")

    config = airflow.training_config(ntm_estimator, record, mini_batch_size=256)
    expected_config = {
        "AlgorithmSpecification": {
            "TrainingImage": "174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:1",
            "TrainingInputMode": "File",
        },
        "OutputDataConfig": {"S3OutputPath": "s3://output/"},
        "TrainingJobName": "ntm-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 30,
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "ShardedByS3Key",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ record }}",
                    }
                },
                "ChannelName": "train",
            }
        ],
        "HyperParameters": {
            "num_topics": "10",
            "epochs": "32",
            "mini_batch_size": "256",
            "feature_dim": "100",
        },
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_amazon_alg_training_config_all_args(sagemaker_session):
    ntm_estimator = ntm.NTM(
        role="{{ role }}",
        num_topics=10,
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.c4.2xlarge",
        train_volume_size="{{ train_volume_size }}",
        train_volume_kms_key="{{ train_volume_kms_key }}",
        train_max_run="{{ train_max_run }}",
        input_mode="Pipe",
        output_path="{{ output_path }}",
        output_kms_key="{{ output_volume_kms_key }}",
        base_job_name="{{ base_job_name }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        subnets=["{{ subnet }}"],
        security_group_ids=["{{ security_group_ids }}"],
        sagemaker_session=sagemaker_session,
    )

    ntm_estimator.epochs = 32

    record = amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix")

    config = airflow.training_config(ntm_estimator, record, mini_batch_size=256)
    expected_config = {
        "AlgorithmSpecification": {
            "TrainingImage": "174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:1",
            "TrainingInputMode": "Pipe",
        },
        "OutputDataConfig": {
            "S3OutputPath": "{{ output_path }}",
            "KmsKeyId": "{{ output_volume_kms_key }}",
        },
        "TrainingJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
        "StoppingCondition": {"MaxRuntimeInSeconds": "{{ train_max_run }}"},
        "ResourceConfig": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": "{{ train_volume_size }}",
            "VolumeKmsKeyId": "{{ train_volume_kms_key }}",
        },
        "RoleArn": "{{ role }}",
        "InputDataConfig": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "ShardedByS3Key",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{{ record }}",
                    }
                },
                "ChannelName": "train",
            }
        ],
        "VpcConfig": {
            "Subnets": ["{{ subnet }}"],
            "SecurityGroupIds": ["{{ security_group_ids }}"],
        },
        "HyperParameters": {
            "num_topics": "10",
            "epochs": "32",
            "mini_batch_size": "256",
            "feature_dim": "100",
        },
        "Tags": [{"{{ key }}": "{{ value }}"}],
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("sagemaker.utils.sagemaker_short_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "{{{{ base_job_name }}}}-{0}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="520713654638.dkr.ecr.us-west-2.amazonaws.com",
)
def test_framework_tuning_config(ecr_prefix, sagemaker_session):
    mxnet_estimator = mxnet.MXNet(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        py_version="py3",
        framework_version="1.3.0",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="{{ base_job_name }}",
        hyperparameters={"batch_size": 100},
    )

    hyperparameter_ranges = {
        "optimizer": tuner.CategoricalParameter(["sgd", "Adam"]),
        "learning_rate": tuner.ContinuousParameter(0.01, 0.2),
        "num_epoch": tuner.IntegerParameter(10, 50),
    }
    objective_metric_name = "Validation-accuracy"
    metric_definitions = [
        {"Name": "Validation-accuracy", "Regex": "Validation-accuracy=([0-9\\.]+)"}
    ]

    mxnet_tuner = tuner.HyperparameterTuner(
        estimator=mxnet_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        strategy="Bayesian",
        objective_type="Maximize",
        max_jobs="{{ max_job }}",
        max_parallel_jobs="{{ max_parallel_job }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        base_tuning_job_name="{{ base_job_name }}",
    )

    data = "{{ training_data }}"

    config = airflow.tuning_config(mxnet_tuner, data)
    expected_config = {
        "HyperParameterTuningJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
        "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {
                "Type": "Maximize",
                "MetricName": "Validation-accuracy",
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": "{{ max_job }}",
                "MaxParallelTrainingJobs": "{{ max_parallel_job }}",
            },
            "ParameterRanges": {
                "ContinuousParameterRanges": [
                    {
                        "Name": "learning_rate",
                        "MinValue": "0.01",
                        "MaxValue": "0.2",
                        "ScalingType": "Auto",
                    }
                ],
                "CategoricalParameterRanges": [
                    {"Name": "optimizer", "Values": ['"sgd"', '"Adam"']}
                ],
                "IntegerParameterRanges": [
                    {"Name": "num_epoch", "MinValue": "10", "MaxValue": "50", "ScalingType": "Auto"}
                ],
            },
            "TrainingJobEarlyStoppingType": "Off",
        },
        "TrainingJobDefinition": {
            "AlgorithmSpecification": {
                "TrainingImage": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.3.0-cpu-py3",
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": "Validation-accuracy", "Regex": "Validation-accuracy=([0-9\\.]+)"}
                ],
            },
            "OutputDataConfig": {"S3OutputPath": "s3://output/"},
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m4.xlarge",
                "VolumeSizeInGB": 30,
            },
            "RoleArn": "{{ role }}",
            "InputDataConfig": [
                {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": "{{ training_data }}",
                        }
                    },
                    "ChannelName": "training",
                }
            ],
            "StaticHyperParameters": {
                "batch_size": "100",
                "sagemaker_submit_directory": '"s3://output/{{ base_job_name }}-%s/source/sourcedir.tar.gz"'
                % TIME_STAMP,
                "sagemaker_program": '"{{ entry_point }}"',
                "sagemaker_enable_cloudwatch_metrics": "false",
                "sagemaker_container_log_level": "20",
                "sagemaker_job_name": '"{{ base_job_name }}-%s"' % TIME_STAMP,
                "sagemaker_region": '"us-west-2"',
                "sagemaker_estimator_module": '"sagemaker.mxnet.estimator"',
                "sagemaker_estimator_class_name": '"MXNet"',
            },
        },
        "Tags": [{"{{ key }}": "{{ value }}"}],
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "{{ source_dir }}",
                    "Bucket": "output",
                    "Key": "{{ base_job_name }}-%s/source/sourcedir.tar.gz" % TIME_STAMP,
                    "Tar": True,
                }
            ]
        },
    }
    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("sagemaker.utils.sagemaker_short_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "{{{{ base_job_name }}}}-{0}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="520713654638.dkr.ecr.us-west-2.amazonaws.com",
)
@patch(
    "sagemaker.amazon.amazon_estimator.get_ecr_image_uri_prefix",
    return_value="174872318107.dkr.ecr.us-west-2.amazonaws.com",
)
def test_multi_estimator_tuning_config(algo_ecr_prefix, fw_ecr_prefix, sagemaker_session):
    estimator_dict = {}
    hyperparameter_ranges_dict = {}
    objective_metric_name_dict = {}
    metric_definitions_dict = {}

    mxnet_estimator_name = "mxnet"
    estimator_dict[mxnet_estimator_name] = mxnet.MXNet(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        py_version="py3",
        framework_version="1.3.0",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="{{ base_job_name }}",
        hyperparameters={"batch_size": 100},
    )
    hyperparameter_ranges_dict[mxnet_estimator_name] = {
        "optimizer": tuner.CategoricalParameter(["sgd", "Adam"]),
        "learning_rate": tuner.ContinuousParameter(0.01, 0.2),
        "num_epoch": tuner.IntegerParameter(10, 50),
    }
    objective_metric_name_dict[mxnet_estimator_name] = "Validation-accuracy"
    metric_definitions_dict[mxnet_estimator_name] = [
        {"Name": "Validation-accuracy", "Regex": "Validation-accuracy=([0-9\\.]+)"}
    ]

    ll_estimator_name = "linear_learner"
    estimator_dict[ll_estimator_name] = linear_learner.LinearLearner(
        predictor_type="binary_classifier",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.c4.2xlarge",
        sagemaker_session=sagemaker_session,
    )
    hyperparameter_ranges_dict[ll_estimator_name] = {
        "learning_rate": tuner.ContinuousParameter(0.2, 0.5),
        "use_bias": tuner.CategoricalParameter([True, False]),
    }
    objective_metric_name_dict[ll_estimator_name] = "validation:binary_classification_accuracy"

    multi_estimator_tuner = tuner.HyperparameterTuner.create(
        estimator_dict=estimator_dict,
        objective_metric_name_dict=objective_metric_name_dict,
        hyperparameter_ranges_dict=hyperparameter_ranges_dict,
        metric_definitions_dict=metric_definitions_dict,
        strategy="Bayesian",
        objective_type="Maximize",
        max_jobs="{{ max_job }}",
        max_parallel_jobs="{{ max_parallel_job }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        base_tuning_job_name="{{ base_job_name }}",
    )

    data = {
        mxnet_estimator_name: "{{ training_data_mxnet }}",
        ll_estimator_name: amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix"),
    }

    config = airflow.tuning_config(multi_estimator_tuner, inputs=data, include_cls_metadata={})

    expected_config = {
        "HyperParameterTuningJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
        "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": "{{ max_job }}",
                "MaxParallelTrainingJobs": "{{ max_parallel_job }}",
            },
            "TrainingJobEarlyStoppingType": "Off",
        },
        "TrainingJobDefinitions": [
            {
                "DefinitionName": "linear_learner",
                "TuningObjective": {
                    "MetricName": "validation:binary_classification_accuracy",
                    "Type": "Maximize",
                },
                "HyperParameterRanges": {
                    "CategoricalParameterRanges": [
                        {"Name": "use_bias", "Values": ["True", "False"]}
                    ],
                    "ContinuousParameterRanges": [
                        {
                            "MaxValue": "0.5",
                            "MinValue": "0.2",
                            "Name": "learning_rate",
                            "ScalingType": "Auto",
                        }
                    ],
                    "IntegerParameterRanges": [],
                },
                "StaticHyperParameters": {
                    "feature_dim": "100",
                    "predictor_type": "binary_classifier",
                },
                "AlgorithmSpecification": {
                    "MetricDefinitions": None,
                    "TrainingImage": "174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1",
                    "TrainingInputMode": "File",
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataDistributionType": "ShardedByS3Key",
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ record }}",
                            }
                        },
                    }
                ],
                "OutputDataConfig": {"S3OutputPath": "s3://output/"},
                "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.c4.2xlarge",
                    "VolumeSizeInGB": 30,
                },
                "RoleArn": "{{ role }}",
                "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            },
            {
                "DefinitionName": "mxnet",
                "TuningObjective": {"MetricName": "Validation-accuracy", "Type": "Maximize"},
                "HyperParameterRanges": {
                    "CategoricalParameterRanges": [
                        {"Name": "optimizer", "Values": ['"sgd"', '"Adam"']}
                    ],
                    "ContinuousParameterRanges": [
                        {
                            "MaxValue": "0.2",
                            "MinValue": "0.01",
                            "Name": "learning_rate",
                            "ScalingType": "Auto",
                        }
                    ],
                    "IntegerParameterRanges": [
                        {
                            "MaxValue": "50",
                            "MinValue": "10",
                            "Name": "num_epoch",
                            "ScalingType": "Auto",
                        }
                    ],
                },
                "StaticHyperParameters": {
                    "batch_size": "100",
                    "sagemaker_container_log_level": "20",
                    "sagemaker_enable_cloudwatch_metrics": "false",
                    "sagemaker_estimator_class_name": '"MXNet"',
                    "sagemaker_estimator_module": '"sagemaker.mxnet.estimator"',
                    "sagemaker_job_name": '"{{ base_job_name }}-%s"' % TIME_STAMP,
                    "sagemaker_program": '"{{ entry_point }}"',
                    "sagemaker_region": '"us-west-2"',
                    "sagemaker_submit_directory": '"s3://output/{{ base_job_name }}-%s/source/sourcedir.tar.gz"'
                    % TIME_STAMP,
                },
                "AlgorithmSpecification": {
                    "MetricDefinitions": [
                        {"Name": "Validation-accuracy", "Regex": "Validation-accuracy=([0-9\\.]+)"}
                    ],
                    "TrainingImage": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.3.0-cpu-py3",
                    "TrainingInputMode": "File",
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataDistributionType": "FullyReplicated",
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ training_data_mxnet }}",
                            }
                        },
                    }
                ],
                "OutputDataConfig": {"S3OutputPath": "s3://output/"},
                "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m4.xlarge",
                    "VolumeSizeInGB": 30,
                },
                "RoleArn": "{{ role }}",
                "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            },
        ],
        "S3Operations": {
            "S3Upload": [
                {
                    "Bucket": "output",
                    "Key": "{{ base_job_name }}-%s/source/sourcedir.tar.gz" % TIME_STAMP,
                    "Path": "{{ source_dir }}",
                    "Tar": True,
                }
            ]
        },
        "Tags": [{"{{ key }}": "{{ value }}"}],
    }

    assert config == expected_config


def test_merge_s3_operations():
    s3_operations_list = [
        {
            "S3Upload": [
                {
                    "Bucket": "output",
                    "Key": "base_job_name-111/source/sourcedir.tar.gz",
                    "Path": "source_dir",
                    "Tar": True,
                }
            ]
        },
        {
            "S3Upload": [
                {
                    "Bucket": "output",
                    "Key": "base_job_name-111/source/sourcedir.tar.gz",
                    "Path": "source_dir",
                    "Tar": True,
                }
            ],
            "S3CreateBucket": [{"Bucket": "output"}],
        },
        {
            "S3Upload": [
                {
                    "Bucket": "output_2",
                    "Key": "base_job_name-111/source/sourcedir_2.tar.gz",
                    "Path": "source_dir_2",
                    "Tar": True,
                }
            ]
        },
        {"S3CreateBucket": [{"Bucket": "output_2"}]},
        {},
    ]

    expected_result = {
        "S3Upload": [
            {
                "Bucket": "output",
                "Key": "base_job_name-111/source/sourcedir.tar.gz",
                "Path": "source_dir",
                "Tar": True,
            },
            {
                "Bucket": "output_2",
                "Key": "base_job_name-111/source/sourcedir_2.tar.gz",
                "Path": "source_dir_2",
                "Tar": True,
            },
        ],
        "S3CreateBucket": [{"Bucket": "output"}, {"Bucket": "output_2"}],
    }

    assert airflow._merge_s3_operations(s3_operations_list) == expected_result


def test_byo_model_config(sagemaker_session):
    byo_model = model.Model(
        model_data="{{ model_data }}",
        image="{{ image }}",
        role="{{ role }}",
        env={"{{ key }}": "{{ value }}"},
        name="model",
        sagemaker_session=sagemaker_session,
    )

    config = airflow.model_config(instance_type="ml.c4.xlarge", model=byo_model)
    expected_config = {
        "ModelName": "model",
        "PrimaryContainer": {
            "Image": "{{ image }}",
            "Environment": {"{{ key }}": "{{ value }}"},
            "ModelDataUrl": "{{ model_data }}",
        },
        "ExecutionRoleArn": "{{ role }}",
    }

    assert config == expected_config


def test_byo_framework_model_config(sagemaker_session):
    byo_model = model.FrameworkModel(
        model_data="{{ model_data }}",
        image="{{ image }}",
        role="{{ role }}",
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        env={"{{ key }}": "{{ value }}"},
        name="model",
        sagemaker_session=sagemaker_session,
    )

    config = airflow.model_config(instance_type="ml.c4.xlarge", model=byo_model)
    expected_config = {
        "ModelName": "model",
        "PrimaryContainer": {
            "Image": "{{ image }}",
            "Environment": {
                "{{ key }}": "{{ value }}",
                "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/model/source/sourcedir.tar.gz",
                "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": "us-west-2",
            },
            "ModelDataUrl": "{{ model_data }}",
        },
        "ExecutionRoleArn": "{{ role }}",
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "{{ source_dir }}",
                    "Bucket": "output",
                    "Key": "model/source/sourcedir.tar.gz",
                    "Tar": True,
                }
            ]
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_framework_model_config(sagemaker_session):
    chainer_model = chainer.ChainerModel(
        model_data="{{ model_data }}",
        role="{{ role }}",
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        image=None,
        py_version="py3",
        framework_version="5.0.0",
        model_server_workers="{{ model_server_worker }}",
        sagemaker_session=sagemaker_session,
    )

    config = airflow.model_config(instance_type="ml.c4.xlarge", model=chainer_model)
    expected_config = {
        "ModelName": "sagemaker-chainer-%s" % TIME_STAMP,
        "PrimaryContainer": {
            "Image": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:5.0.0-cpu-py3",
            "Environment": {
                "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/sagemaker-chainer-%s/source/sourcedir.tar.gz"
                % TIME_STAMP,
                "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": "us-west-2",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "{{ model_server_worker }}",
            },
            "ModelDataUrl": "{{ model_data }}",
        },
        "ExecutionRoleArn": "{{ role }}",
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "{{ source_dir }}",
                    "Bucket": "output",
                    "Key": "sagemaker-chainer-%s/source/sourcedir.tar.gz" % TIME_STAMP,
                    "Tar": True,
                }
            ]
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_amazon_alg_model_config(sagemaker_session):
    pca_model = pca.PCAModel(
        model_data="{{ model_data }}", role="{{ role }}", sagemaker_session=sagemaker_session
    )

    config = airflow.model_config(instance_type="ml.c4.xlarge", model=pca_model)
    expected_config = {
        "ModelName": "pca-%s" % TIME_STAMP,
        "PrimaryContainer": {
            "Image": "174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1",
            "Environment": {},
            "ModelDataUrl": "{{ model_data }}",
        },
        "ExecutionRoleArn": "{{ role }}",
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "{{{{ base_job_name }}}}-{0}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="763104351884.dkr.ecr.us-west-2.amazonaws.com",
)
def test_model_config_from_framework_estimator(ecr_prefix, sagemaker_session):
    mxnet_estimator = mxnet.MXNet(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        py_version="py3",
        framework_version="1.6.0",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="{{ base_job_name }}",
        hyperparameters={"batch_size": 100},
    )

    data = "{{ training_data }}"

    # simulate training
    airflow.training_config(mxnet_estimator, data)

    config = airflow.model_config_from_estimator(
        instance_type="ml.c4.xlarge",
        estimator=mxnet_estimator,
        task_id="task_id",
        task_type="training",
    )
    expected_config = {
        "ModelName": "mxnet-inference-%s" % TIME_STAMP,
        "PrimaryContainer": {
            "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.6.0-cpu-py3",
            "Environment": {
                "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']"
                "['TrainingJobName'] }}/source/sourcedir.tar.gz",
                "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": "us-west-2",
            },
            "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']['TrainingJobName'] }}"
            "/output/model.tar.gz",
        },
        "ExecutionRoleArn": "{{ role }}",
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_model_config_from_amazon_alg_estimator(sagemaker_session):
    knn_estimator = knn.KNN(
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.m4.xlarge",
        k=16,
        sample_size=128,
        predictor_type="regressor",
        sagemaker_session=sagemaker_session,
    )

    record = amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix")

    # simulate training
    airflow.training_config(knn_estimator, record, mini_batch_size=256)

    config = airflow.model_config_from_estimator(
        instance_type="ml.c4.xlarge", estimator=knn_estimator, task_id="task_id", task_type="tuning"
    )
    expected_config = {
        "ModelName": "knn-%s" % TIME_STAMP,
        "PrimaryContainer": {
            "Image": "174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1",
            "Environment": {},
            "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Tuning']['BestTrainingJob']"
            "['TrainingJobName'] }}/output/model.tar.gz",
        },
        "ExecutionRoleArn": "{{ role }}",
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_transform_config(sagemaker_session):
    tf_transformer = transformer.Transformer(
        model_name="tensorflow-model",
        instance_count="{{ instance_count }}",
        instance_type="ml.p2.xlarge",
        strategy="SingleRecord",
        assemble_with="Line",
        output_path="{{ output_path }}",
        output_kms_key="{{ kms_key }}",
        accept="{{ accept }}",
        max_concurrent_transforms="{{ max_parallel_job }}",
        max_payload="{{ max_payload }}",
        tags=[{"{{ key }}": "{{ value }}"}],
        env={"{{ key }}": "{{ value }}"},
        base_transform_job_name="tensorflow-transform",
        sagemaker_session=sagemaker_session,
        volume_kms_key="{{ kms_key }}",
    )

    data = "{{ transform_data }}"

    config = airflow.transform_config(
        tf_transformer,
        data,
        data_type="S3Prefix",
        content_type="{{ content_type }}",
        compression_type="{{ compression_type }}",
        split_type="{{ split_type }}",
    )
    expected_config = {
        "TransformJobName": "tensorflow-transform-%s" % TIME_STAMP,
        "ModelName": "tensorflow-model",
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "{{ transform_data }}"}
            },
            "ContentType": "{{ content_type }}",
            "CompressionType": "{{ compression_type }}",
            "SplitType": "{{ split_type }}",
        },
        "TransformOutput": {
            "S3OutputPath": "{{ output_path }}",
            "KmsKeyId": "{{ kms_key }}",
            "AssembleWith": "Line",
            "Accept": "{{ accept }}",
        },
        "TransformResources": {
            "InstanceCount": "{{ instance_count }}",
            "InstanceType": "ml.p2.xlarge",
            "VolumeKmsKeyId": "{{ kms_key }}",
        },
        "BatchStrategy": "SingleRecord",
        "MaxConcurrentTransforms": "{{ max_parallel_job }}",
        "MaxPayloadInMB": "{{ max_payload }}",
        "Environment": {"{{ key }}": "{{ value }}"},
        "Tags": [{"{{ key }}": "{{ value }}"}],
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "{{{{ base_job_name }}}}-{0}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="763104351884.dkr.ecr.us-west-2.amazonaws.com",
)
def test_transform_config_from_framework_estimator(ecr_prefix, sagemaker_session):
    mxnet_estimator = mxnet.MXNet(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        py_version="py3",
        framework_version="1.6.0",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="{{ base_job_name }}",
        hyperparameters={"batch_size": 100},
    )

    train_data = "{{ train_data }}"
    transform_data = "{{ transform_data }}"

    # simulate training
    airflow.training_config(mxnet_estimator, train_data)

    config = airflow.transform_config_from_estimator(
        estimator=mxnet_estimator,
        task_id="task_id",
        task_type="training",
        instance_count="{{ instance_count }}",
        instance_type="ml.p2.xlarge",
        data=transform_data,
    )
    expected_config = {
        "Model": {
            "ModelName": "mxnet-inference-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.6.0-gpu-py3",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/{{ ti.xcom_pull(task_ids='task_id')"
                    "['Training']['TrainingJobName'] }}"
                    "/source/sourcedir.tar.gz",
                    "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": "us-west-2",
                },
                "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']['TrainingJobName'] }}"
                "/output/model.tar.gz",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "Transform": {
            "TransformJobName": "{{ base_job_name }}-%s" % TIME_STAMP,
            "ModelName": "mxnet-inference-%s" % TIME_STAMP,
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "{{ transform_data }}"}
                }
            },
            "TransformOutput": {"S3OutputPath": "s3://output/{{ base_job_name }}-%s" % TIME_STAMP},
            "TransformResources": {
                "InstanceCount": "{{ instance_count }}",
                "InstanceType": "ml.p2.xlarge",
            },
            "Environment": {},
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_transform_config_from_amazon_alg_estimator(sagemaker_session):
    knn_estimator = knn.KNN(
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.m4.xlarge",
        k=16,
        sample_size=128,
        predictor_type="regressor",
        sagemaker_session=sagemaker_session,
    )

    record = amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix")
    transform_data = "{{ transform_data }}"

    # simulate training
    airflow.training_config(knn_estimator, record, mini_batch_size=256)

    config = airflow.transform_config_from_estimator(
        estimator=knn_estimator,
        task_id="task_id",
        task_type="training",
        instance_count="{{ instance_count }}",
        instance_type="ml.p2.xlarge",
        data=transform_data,
    )
    expected_config = {
        "Model": {
            "ModelName": "knn-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1",
                "Environment": {},
                "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']['TrainingJobName'] }}"
                "/output/model.tar.gz",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "Transform": {
            "TransformJobName": "knn-%s" % TIME_STAMP,
            "ModelName": "knn-%s" % TIME_STAMP,
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "{{ transform_data }}"}
                }
            },
            "TransformOutput": {"S3OutputPath": "s3://output/knn-%s" % TIME_STAMP},
            "TransformResources": {
                "InstanceCount": "{{ instance_count }}",
                "InstanceType": "ml.p2.xlarge",
            },
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_deploy_framework_model_config(sagemaker_session):
    chainer_model = chainer.ChainerModel(
        model_data="{{ model_data }}",
        role="{{ role }}",
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        image=None,
        py_version="py3",
        framework_version="5.0.0",
        model_server_workers="{{ model_server_worker }}",
        sagemaker_session=sagemaker_session,
    )

    config = airflow.deploy_config(
        chainer_model, initial_instance_count="{{ instance_count }}", instance_type="ml.m4.xlarge"
    )
    expected_config = {
        "Model": {
            "ModelName": "sagemaker-chainer-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:5.0.0-cpu-py3",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/sagemaker-chainer-%s/source/sourcedir.tar.gz"
                    % TIME_STAMP,
                    "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": "us-west-2",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "{{ model_server_worker }}",
                },
                "ModelDataUrl": "{{ model_data }}",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "EndpointConfig": {
            "EndpointConfigName": "sagemaker-chainer-%s" % TIME_STAMP,
            "ProductionVariants": [
                {
                    "InstanceType": "ml.m4.xlarge",
                    "InitialInstanceCount": "{{ instance_count }}",
                    "ModelName": "sagemaker-chainer-%s" % TIME_STAMP,
                    "VariantName": "AllTraffic",
                    "InitialVariantWeight": 1,
                }
            ],
        },
        "Endpoint": {
            "EndpointName": "sagemaker-chainer-%s" % TIME_STAMP,
            "EndpointConfigName": "sagemaker-chainer-%s" % TIME_STAMP,
        },
        "S3Operations": {
            "S3Upload": [
                {
                    "Path": "{{ source_dir }}",
                    "Bucket": "output",
                    "Key": "sagemaker-chainer-%s/source/sourcedir.tar.gz" % TIME_STAMP,
                    "Tar": True,
                }
            ]
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_deploy_amazon_alg_model_config(sagemaker_session):
    pca_model = pca.PCAModel(
        model_data="{{ model_data }}", role="{{ role }}", sagemaker_session=sagemaker_session
    )

    config = airflow.deploy_config(
        pca_model, initial_instance_count="{{ instance_count }}", instance_type="ml.c4.xlarge"
    )
    expected_config = {
        "Model": {
            "ModelName": "pca-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1",
                "Environment": {},
                "ModelDataUrl": "{{ model_data }}",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "EndpointConfig": {
            "EndpointConfigName": "pca-%s" % TIME_STAMP,
            "ProductionVariants": [
                {
                    "InstanceType": "ml.c4.xlarge",
                    "InitialInstanceCount": "{{ instance_count }}",
                    "ModelName": "pca-%s" % TIME_STAMP,
                    "VariantName": "AllTraffic",
                    "InitialVariantWeight": 1,
                }
            ],
        },
        "Endpoint": {
            "EndpointName": "pca-%s" % TIME_STAMP,
            "EndpointConfigName": "pca-%s" % TIME_STAMP,
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
@patch("os.path.isfile", MagicMock(return_value=True))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock())
@patch(
    "sagemaker.fw_utils.parse_s3_url",
    MagicMock(
        return_value=[
            "output",
            "{{{{ base_job_name }}}}-{0}/source/sourcedir.tar.gz".format(TIME_STAMP),
        ]
    ),
)
@patch(
    "sagemaker.fw_utils.get_ecr_image_uri_prefix",
    return_value="763104351884.dkr.ecr.us-west-2.amazonaws.com",
)
def test_deploy_config_from_framework_estimator(ecr_prefix, sagemaker_session):
    mxnet_estimator = mxnet.MXNet(
        entry_point="{{ entry_point }}",
        source_dir="{{ source_dir }}",
        py_version="py3",
        framework_version="1.6.0",
        role="{{ role }}",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="{{ base_job_name }}",
        hyperparameters={"batch_size": 100},
    )

    train_data = "{{ train_data }}"

    # simulate training
    airflow.training_config(mxnet_estimator, train_data)

    config = airflow.deploy_config_from_estimator(
        estimator=mxnet_estimator,
        task_id="task_id",
        task_type="training",
        initial_instance_count="{{ instance_count}}",
        instance_type="ml.c4.large",
        endpoint_name="mxnet-endpoint",
    )
    expected_config = {
        "Model": {
            "ModelName": "mxnet-inference-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.6.0-cpu-py3",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "{{ entry_point }}",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']"
                    "['TrainingJobName'] }}/source/sourcedir.tar.gz",
                    "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": "us-west-2",
                },
                "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Training']['TrainingJobName'] }}"
                "/output/model.tar.gz",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "EndpointConfig": {
            "EndpointConfigName": "mxnet-inference-%s" % TIME_STAMP,
            "ProductionVariants": [
                {
                    "InstanceType": "ml.c4.large",
                    "InitialInstanceCount": "{{ instance_count}}",
                    "ModelName": "mxnet-inference-%s" % TIME_STAMP,
                    "VariantName": "AllTraffic",
                    "InitialVariantWeight": 1,
                }
            ],
        },
        "Endpoint": {
            "EndpointName": "mxnet-endpoint",
            "EndpointConfigName": "mxnet-inference-%s" % TIME_STAMP,
        },
    }

    assert config == expected_config


@patch("sagemaker.utils.sagemaker_timestamp", MagicMock(return_value=TIME_STAMP))
def test_deploy_config_from_amazon_alg_estimator(sagemaker_session):
    knn_estimator = knn.KNN(
        role="{{ role }}",
        train_instance_count="{{ instance_count }}",
        train_instance_type="ml.m4.xlarge",
        k=16,
        sample_size=128,
        predictor_type="regressor",
        sagemaker_session=sagemaker_session,
    )

    record = amazon_estimator.RecordSet("{{ record }}", 10000, 100, "S3Prefix")

    # simulate training
    airflow.training_config(knn_estimator, record, mini_batch_size=256)

    config = airflow.deploy_config_from_estimator(
        estimator=knn_estimator,
        task_id="task_id",
        task_type="tuning",
        initial_instance_count="{{ instance_count }}",
        instance_type="ml.p2.xlarge",
    )
    expected_config = {
        "Model": {
            "ModelName": "knn-%s" % TIME_STAMP,
            "PrimaryContainer": {
                "Image": "174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1",
                "Environment": {},
                "ModelDataUrl": "s3://output/{{ ti.xcom_pull(task_ids='task_id')['Tuning']['BestTrainingJob']"
                "['TrainingJobName'] }}/output/model.tar.gz",
            },
            "ExecutionRoleArn": "{{ role }}",
        },
        "EndpointConfig": {
            "EndpointConfigName": "knn-%s" % TIME_STAMP,
            "ProductionVariants": [
                {
                    "InstanceType": "ml.p2.xlarge",
                    "InitialInstanceCount": "{{ instance_count }}",
                    "ModelName": "knn-%s" % TIME_STAMP,
                    "VariantName": "AllTraffic",
                    "InitialVariantWeight": 1,
                }
            ],
        },
        "Endpoint": {
            "EndpointName": "knn-%s" % TIME_STAMP,
            "EndpointConfigName": "knn-%s" % TIME_STAMP,
        },
    }

    assert config == expected_config
