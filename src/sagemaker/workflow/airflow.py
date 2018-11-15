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


def prepare_amazon_algorithm_estimator(estimator, inputs, mini_batch_size=None):
    """ Set up amazon algorithm estimator, adding the required `feature_dim` hyperparameter from training data.

    Args:
        estimator (sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
            An estimator for a built-in Amazon algorithm to get information from and update.
        inputs: The training data.
            * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                Amazon :class:~`Record` objects serialized and stored in S3.
                For use with an estimator for an Amazon algorithm.
            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects, where each instance is
                a different channel of training data.
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
    estimator.mini_batch_size = mini_batch_size


def training_config(estimator, inputs=None, job_name=None, **kargs):  # noqa: C901
    """Export Airflow training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatroBase):
            The estimator to export training config from. Can be a BYO estimator,
            Framework estimator or Amazon algorithm estimator.
        inputs: Information about the training data. Please refer to the ``fit()`` method of
                the associated estimator, as this can take any of the following forms:

            * (str) - The S3 location where training data is saved.
            * (dict[str, str] or dict[str, sagemaker.session.s3_input]) - If using multiple channels for
                training data, you can specify a dict mapping channel names
                to strings or :func:`~sagemaker.session.s3_input` objects.
            * (sagemaker.session.s3_input) - Channel configuration for S3 data sources that can provide
                additional information about the training dataset. See :func:`sagemaker.session.s3_input`
                for full details.
            * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                Amazon :class:~`Record` objects serialized and stored in S3.
                For use with an estimator for an Amazon algorithm.
            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects, where each instance is
                a different channel of training data.

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
        prepare_amazon_algorithm_estimator(estimator, inputs, **kargs)

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


def tuning_config(tuner, inputs, job_name=None):
    train_config = training_config(tuner.estimator, inputs, job_name)

    train_config.pop('Tags', None)
    train_config.pop('TrainingJobName', None)
    s3_operations = train_config.pop('S3Operations', None)
    hyperparameters = train_config.pop('HyperParameters', None)

    if hyperparameters and len(hyperparameters) > 0:
        tuner.static_hyperparameters = \
            {utils.to_str(k): utils.to_str(v) for (k, v) in hyperparameters.items()}

    if job_name is not None:
        tuner._current_job_name = job_name
    else:
        base_name = tuner.base_tuning_job_name or utils.base_name_from_image(tuner.estimator.train_image())
        tuner._current_job_name = utils.airflow_name_from_base(base_name, tuner.TUNING_JOB_NAME_MAX_LENGTH, True)

    for hyperparameter_name in tuner._hyperparameter_ranges.keys():
        tuner.static_hyperparameters.pop(hyperparameter_name, None)

    train_config['StaticHyperParameters'] = tuner.static_hyperparameters

    tune_config = {
        'HyperParameterTuningJobName': tuner._current_job_name,
        'HyperParameterTuningJobConfig': {
            'Strategy': tuner.strategy,
            'HyperParameterTuningJobObjective': {
                'Type': tuner.objective_type,
                'MetricName': tuner.objective_metric_name,
            },
            'ResourceLimits': {
                'MaxNumberOfTrainingJobs': tuner.max_jobs,
                'MaxParallelTrainingJobs': tuner.max_parallel_jobs,
            },
            'ParameterRanges': tuner.hyperparameter_ranges(),
        },
        'TrainingJobDefinition': train_config
    }

    if tuner.metric_definitions is not None:
        tune_config['TrainingJobDefinition']['AlgorithmSpecification']['MetricDefinitions'] = \
            tuner.metric_definitions

    if tuner.tags is not None:
        tune_config['Tags'] = tuner.tags

    if s3_operations is not None:
        tune_config['S3Operations'] = s3_operations

    return tune_config
