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

import os

from sagemaker.estimator import Framework
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase, RecordSet
from sagemaker.job import _Job
from sagemaker.utils import base_name_from_image, airflow_name_from_base
from sagemaker.model import DIR_PARAM_NAME, SCRIPT_PARAM_NAME, CLOUDWATCH_METRICS_PARAM_NAME, \
    CONTAINER_LOG_LEVEL_PARAM_NAME, JOB_NAME_PARAM_NAME, SAGEMAKER_REGION_PARAM_NAME


def prepare_framework(estimator, s3_operations, default_bucket):
    """
    Prepare S3 operations (specify where to upload source_dir) and environment variables
        related to framework.

    Args:
        estimator: The framework estimator to get information from and update.
        s3_operations: The dict to specify s3 operations (upload source_dir).
        default_bucket: The default bucket to use in training.
    """
    bucket = estimator.code_location if estimator.code_location else default_bucket
    key = '{}/source/sourcedir.tar.gz'.format(estimator._current_job_name)
    script = os.path.basename(estimator.entry_point)
    if estimator.source_dir and estimator.source_dir.lower().startswith('s3://'):
        code_dir = estimator.source_dir
    else:
        code_dir = 's3://{}/{}'.format(bucket, key)
        s3_operations['S3Upload'] = [{
            'Path': estimator.source_dir or script,
            'Bucket': bucket,
            'Key': key,
            'Tar': True
        }]
    estimator._hyperparameters[DIR_PARAM_NAME] = code_dir
    estimator._hyperparameters[SCRIPT_PARAM_NAME] = script
    estimator._hyperparameters[CLOUDWATCH_METRICS_PARAM_NAME] = estimator.enable_cloudwatch_metrics
    estimator._hyperparameters[CONTAINER_LOG_LEVEL_PARAM_NAME] = estimator.container_log_level
    estimator._hyperparameters[JOB_NAME_PARAM_NAME] = estimator._current_job_name
    estimator._hyperparameters[SAGEMAKER_REGION_PARAM_NAME] = estimator.sagemaker_session.boto_region_name


def prepare_amazon_algorithm_estimator(estimator, inputs):
    """
    Set up amazon algorithm estimator, adding the required feature_dim hyperparameter from training data.
    Args:
        estimator: The Amazon algorithm estimator to get information from and update.
        inputs: The training data, must be in RecordSet format.
    """
    if isinstance(inputs, list):
        for record in inputs:
            if isinstance(record, RecordSet) and record.channel == 'train':
                estimator.feature_dim = record.feature_dim
                break
    elif isinstance(inputs, RecordSet):
        estimator.feature_dim = inputs.feature_dim
    else:
        raise TypeError('The training data of Amazon Algorithm estimator is not represented by RecordSet.')


def get_training_config(estimator, inputs=None, job_name=None):
    """
    Export Airflow training config from an estimator

    Args:
        estimator: The estimator to export training config from. Can be a BYO estimator,
            Framework estimator or Amazon algorithm estimator.
        inputs: The training data.
        job_name: Specify a training job name if needed.

    Returns:
        A dict of training config that can be directly used by SageMakerTrainingOperator
            in Airflow.
    """
    default_bucket = estimator.sagemaker_session.default_bucket()
    s3_operations = {}

    if job_name is not None:
        estimator._current_job_name = job_name
    else:
        base_name = estimator.base_job_name or base_name_from_image(estimator.train_image())
        estimator._current_job_name = airflow_name_from_base(base_name)

    if estimator.output_path is None:
        estimator.output_path = 's3://{}/'.format(default_bucket)

    if isinstance(estimator, Framework):
        prepare_framework(estimator, s3_operations, default_bucket)

    elif isinstance(estimator, AmazonAlgorithmEstimatorBase):
        prepare_amazon_algorithm_estimator(estimator, inputs)

    job_config = _Job._load_config(inputs, estimator, expand_role=False, validate_uri=False)

    train_config = {
        'AlgorithmSpecification': {
            'TrainingImage': estimator.train_image(),
            'TrainingInputMode': estimator.input_mode
        },
        'OutputDataConfig': job_config['output_config'],
        'TrainingJobName': estimator._current_job_name,
        'StoppingCondition': job_config['stop_condition'],
        'ResourceConfig': job_config['resource_config'],
        'RoleArn': job_config['role'],
    }

    if job_config['input_config'] is not None:
        train_config['InputDataConfig'] = job_config['input_config']

    if job_config['vpc_config'] is not None:
        train_config['VpcConfig'] = job_config['vpc_config']

    if estimator.hyperparameters() is not None:
        hyperparameters = {str(k): str(v) for (k, v) in estimator.hyperparameters().items()}

    if hyperparameters and len(hyperparameters) > 0:
        train_config['HyperParameters'] = hyperparameters

    if estimator.tags is not None:
        train_config['Tags'] = estimator.tags

    if s3_operations:
        train_config['S3Operations'] = s3_operations

    return train_config
