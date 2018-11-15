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
from __future__ import print_function, absolute_import

import os

import sagemaker
from sagemaker import job, utils, model
from sagemaker.amazon import amazon_estimator


def prepare_framework(estimator, s3_operations):
    """Prepare S3 operations (specify where to upload source_dir) and environment variables
    related to framework.

    Args:
        estimator (sagemaker.estimator.Estimator): The framework estimator to get information from and update.
        s3_operations (dict): The dict to specify s3 operations (upload source_dir).
    """
    bucket = estimator.code_location if estimator.code_location else estimator.sagemaker_session._default_bucket
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
    estimator._hyperparameters[model.DIR_PARAM_NAME] = code_dir
    estimator._hyperparameters[model.SCRIPT_PARAM_NAME] = script
    estimator._hyperparameters[model.CLOUDWATCH_METRICS_PARAM_NAME] = estimator.enable_cloudwatch_metrics
    estimator._hyperparameters[model.CONTAINER_LOG_LEVEL_PARAM_NAME] = estimator.container_log_level
    estimator._hyperparameters[model.JOB_NAME_PARAM_NAME] = estimator._current_job_name
    estimator._hyperparameters[model.SAGEMAKER_REGION_PARAM_NAME] = estimator.sagemaker_session.boto_region_name


def prepare_amazon_algorithm_estimator(estimator, inputs):
    """ Set up amazon algorithm estimator, adding the required `feature_dim` hyperparameter from training data.

    Args:
        estimator (sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
            An estimator for a built-in Amazon algorithm to get information from and update.
        inputs (single or list of sagemaker.amazon.amazon_estimator.RecordSet):
            The training data, must be in RecordSet format.
    """
    if isinstance(inputs, list):
        for record in inputs:
            if isinstance(record, amazon_estimator.RecordSet) and record.channel == 'train':
                estimator.feature_dim = record.feature_dim
                break
    elif isinstance(inputs, amazon_estimator.RecordSet):
        estimator.feature_dim = inputs.feature_dim
    else:
        raise TypeError('Training data must be represented in RecordSet or list of RecordSets')


def training_config(estimator, inputs=None, job_name=None):  # noqa: C901 - suppress complexity warning for this method
    """Export Airflow training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatroBase):
            The estimator to export training config from. Can be a BYO estimator,
            Framework estimator or Amazon algorithm estimator.
        inputs (str, dict, single or list of sagemaker.amazon.amazon_estimator.RecordSet):
            The training data.
        job_name (str): Specify a training job name if needed.

    Returns:
        A dict of training config that can be directly used by SageMakerTrainingOperator
            in Airflow.
    """
    default_bucket = estimator.sagemaker_session.default_bucket()
    s3_operations = {}

    if job_name is not None:
        estimator._current_job_name = job_name
    else:
        base_name = estimator.base_job_name or utils.base_name_from_image(estimator.train_image())
        estimator._current_job_name = utils.airflow_name_from_base(base_name)

    if estimator.output_path is None:
        estimator.output_path = 's3://{}/'.format(default_bucket)

    if isinstance(estimator, sagemaker.estimator.Framework):
        prepare_framework(estimator, s3_operations)

    elif isinstance(estimator, amazon_estimator.AmazonAlgorithmEstimatorBase):
        prepare_amazon_algorithm_estimator(estimator, inputs)

    job_config = job._Job._load_config(inputs, estimator, expand_role=False, validate_uri=False)

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
