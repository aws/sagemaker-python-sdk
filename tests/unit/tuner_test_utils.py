# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

from mock import Mock
from sagemaker.amazon.pca import PCA
from sagemaker.estimator import Estimator
from sagemaker.parameter import CategoricalParameter, ContinuousParameter, IntegerParameter
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DATA = "s3://bucket/model.tar.gz"

JOB_NAME = "tuning_job"
TRAINING_JOB_NAME = "training_job_neo"
BASE_JOB_NAME = "base_tuning_job"
REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
ROLE = "myrole"
IMAGE_NAME = "image"

INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_COMPONENTS = 5

SCRIPT_NAME = "my_script.py"
FRAMEWORK_VERSION = "1.0.0"
PY_VERSION = "py3"

INPUTS = "s3://mybucket/train"

STRATEGY = ("Bayesian",)
OBJECTIVE_TYPE = "Minimize"
EARLY_STOPPING_TYPE = "Auto"

OBJECTIVE_METRIC_NAME = "mock_metric"
OBJECTIVE_METRIC_NAME_TWO = "mock_metric_two"

HYPERPARAMETER_RANGES = {
    "validated": ContinuousParameter(0, 5),
    "elizabeth": IntegerParameter(0, 5),
    "blank": CategoricalParameter([0, 5]),
}
HYPERPARAMETER_RANGES_TWO = {
    "num_components": IntegerParameter(2, 4),
    "algorithm_mode": CategoricalParameter(["regular", "randomized"]),
}

METRIC_DEFINITIONS = "mock_metric_definitions"

MAX_JOBS = 10
MAX_PARALLEL_JOBS = 5
TAGS = [{"key1": "value1"}]

LIST_TAGS_RESULT = {"Tags": [{"Key": "key1", "Value": "value1"}]}

ESTIMATOR_NAME = "estimator_name"
ESTIMATOR_NAME_TWO = "estimator_name_two"

ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}

SAGEMAKER_SESSION = Mock()
# For tests which doesn't verify config file injection, operate with empty config
SAGEMAKER_SESSION.sagemaker_config = {}
SAGEMAKER_SESSION.default_bucket = Mock(return_value=BUCKET_NAME)
SAGEMAKER_SESSION.default_bucket_prefix = None


ESTIMATOR = Estimator(
    IMAGE_NAME,
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    output_path="s3://bucket/prefix",
    sagemaker_session=SAGEMAKER_SESSION,
    environment=ENV_INPUT,
)
ESTIMATOR_TWO = PCA(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    NUM_COMPONENTS,
    sagemaker_session=SAGEMAKER_SESSION,
    environment=ENV_INPUT,
)

WARM_START_CONFIG = WarmStartConfig(
    warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, parents={"p1", "p2", "p3"}
)

TUNING_JOB_DETAILS = {
    "HyperParameterTuningJobConfig": {
        "ResourceLimits": {
            "MaxParallelTrainingJobs": 1,
            "MaxNumberOfTrainingJobs": 1,
            "MaxRuntimeInSeconds": 1,
        },
        "HyperParameterTuningJobObjective": {
            "MetricName": OBJECTIVE_METRIC_NAME,
            "Type": "Minimize",
        },
        "Strategy": "Bayesian",
        "ParameterRanges": {
            "CategoricalParameterRanges": [],
            "ContinuousParameterRanges": [],
            "IntegerParameterRanges": [
                {
                    "MaxValue": "100",
                    "Name": "mini_batch_size",
                    "MinValue": "10",
                    "ScalingType": "Auto",
                }
            ],
        },
        "TrainingJobEarlyStoppingType": "Off",
        "RandomSeed": 0,
        "TuningJobCompletionCriteria": {
            "BestObjectiveNotImproving": {"MaxNumberOfTrainingJobsNotImproving": 5},
            "ConvergenceDetected": {"CompleteOnConvergence": "Enabled"},
            "TargetObjectiveMetricValue": 0.42,
        },
    },
    "HyperParameterTuningJobName": JOB_NAME,
    "TrainingJobDefinition": {
        "RoleArn": ROLE,
        "StaticHyperParameters": {
            "num_components": "10",
            "_tuning_objective_metric": "train:throughput",
            "feature_dim": "784",
            "sagemaker_estimator_module": '"sagemaker.amazon.pca"',
            "sagemaker_estimator_class_name": '"PCA"',
        },
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceType": "ml.c4.xlarge",
            "InstanceCount": 1,
        },
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE_NAME,
            "TrainingInputMode": "File",
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "ShardedByS3Key",
                        "S3Uri": INPUTS,
                        "S3DataType": "ManifestFile",
                    }
                },
            }
        ],
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "OutputDataConfig": {"S3OutputPath": BUCKET_NAME},
        "Environment": ENV_INPUT,
    },
    "TrainingJobCounters": {
        "ClientError": 0,
        "Completed": 1,
        "InProgress": 0,
        "Fault": 0,
        "Stopped": 0,
    },
    "HyperParameterTuningEndTime": 1526605831.0,
    "CreationTime": 1526605605.0,
    "HyperParameterTuningJobArn": "arn:tuning_job",
}

MULTI_ALGO_TUNING_JOB_DETAILS = {
    "HyperParameterTuningJobConfig": {
        "ResourceLimits": {"MaxParallelTrainingJobs": 2, "MaxNumberOfTrainingJobs": 4},
        "Strategy": "Bayesian",
        "TrainingJobEarlyStoppingType": "Off",
    },
    "HyperParameterTuningJobName": JOB_NAME,
    "TrainingJobDefinitions": [
        {
            "DefinitionName": ESTIMATOR_NAME,
            "TuningObjective": {"MetricName": OBJECTIVE_METRIC_NAME, "Type": "Minimize"},
            "HyperParameterRanges": {
                "CategoricalParameterRanges": [],
                "ContinuousParameterRanges": [],
                "IntegerParameterRanges": [
                    {
                        "MaxValue": "100",
                        "Name": "mini_batch_size",
                        "MinValue": "10",
                        "ScalingType": "Auto",
                    }
                ],
            },
            "RoleArn": ROLE,
            "StaticHyperParameters": {
                "num_components": "1",
                "_tuning_objective_metric": "train:throughput",
                "feature_dim": "784",
                "sagemaker_estimator_module": '"sagemaker.amazon.pca"',
                "sagemaker_estimator_class_name": '"PCA"',
            },
            "ResourceConfig": {
                "VolumeSizeInGB": 30,
                "InstanceType": "ml.c4.xlarge",
                "InstanceCount": 1,
            },
            "AlgorithmSpecification": {"TrainingImage": IMAGE_NAME, "TrainingInputMode": "File"},
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "ShardedByS3Key",
                            "S3Uri": INPUTS,
                            "S3DataType": "ManifestFile",
                        }
                    },
                }
            ],
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "OutputDataConfig": {"S3OutputPath": BUCKET_NAME},
            "Environment": ENV_INPUT,
        },
        {
            "DefinitionName": ESTIMATOR_NAME_TWO,
            "TuningObjective": {"MetricName": OBJECTIVE_METRIC_NAME_TWO, "Type": "Minimize"},
            "HyperParameterRanges": {
                "CategoricalParameterRanges": [{"Name": "kernel", "Values": ["rbf", "sigmoid"]}],
                "ContinuousParameterRanges": [],
                "IntegerParameterRanges": [
                    {"MaxValue": "10", "Name": "tree_count", "MinValue": "1", "ScalingType": "Auto"}
                ],
            },
            "RoleArn": ROLE,
            "StaticHyperParameters": {
                "blank": "1",
                "_tuning_objective_metric": OBJECTIVE_METRIC_NAME_TWO,
            },
            "ResourceConfig": {
                "VolumeSizeInGB": 30,
                "InstanceType": "ml.m4.4xlarge",
                "InstanceCount": 1,
            },
            "AlgorithmSpecification": {
                "TrainingImage": IMAGE_NAME,
                "TrainingInputMode": "File",
                "MetricDefinitions": METRIC_DEFINITIONS,
            },
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "ShardedByS3Key",
                            "S3Uri": INPUTS,
                            "S3DataType": "ManifestFile",
                        }
                    },
                }
            ],
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "OutputDataConfig": {"S3OutputPath": BUCKET_NAME},
            "Environment": ENV_INPUT,
        },
    ],
    "TrainingJobCounters": {
        "ClientError": 0,
        "Completed": 1,
        "InProgress": 0,
        "Fault": 0,
        "Stopped": 0,
    },
    "HyperParameterTuningEndTime": 1526605831.0,
    "CreationTime": 1526605605.0,
    "HyperParameterTuningJobArn": "arn:tuning_job",
}

TRAINING_JOB_DESCRIPTION = {
    "AlgorithmSpecification": {
        "TrainingInputMode": "File",
        "TrainingImage": IMAGE_NAME,
        "MetricDefinitions": METRIC_DEFINITIONS,
    },
    "HyperParameters": {
        "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
        "checkpoint_path": '"s3://other/1508872349"',
        "sagemaker_program": '"iris-dnn-classifier.py"',
        "sagemaker_enable_cloudwatch_metrics": "false",
        "sagemaker_container_log_level": '"logging.INFO"',
        "sagemaker_job_name": '"neo"',
        "training_steps": "100",
        "_tuning_objective_metric": "Validation-accuracy",
    },
    "RoleArn": ROLE,
    "ResourceConfig": {"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.c4.xlarge"},
    "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
    "TrainingJobName": TRAINING_JOB_NAME,
    "TrainingJobStatus": "Completed",
    "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
    "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    "ModelArtifacts": {"S3ModelArtifacts": MODEL_DATA},
    "Environment": ENV_INPUT,
}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}
