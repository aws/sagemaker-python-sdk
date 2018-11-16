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
from sagemaker import fw_utils, job, utils, session, vpc_utils
from sagemaker.amazon import amazon_estimator


def prepare_framework(estimator, s3_operations):
    """Prepare S3 operations (specify where to upload `source_dir`) and environment variables
    related to framework.

    Args:
        estimator (sagemaker.estimator.Estimator): The framework estimator to get information from and update.
        s3_operations (dict): The dict to specify s3 operations (upload `source_dir`).
    """
    bucket = estimator.code_location if estimator.code_location else estimator.sagemaker_session._default_bucket
    key = '{}/source/sourcedir.tar.gz'.format(estimator._current_job_name)
    script = os.path.basename(estimator.entry_point)
    if estimator.source_dir and estimator.source_dir.lower().startswith('s3://'):
        code_dir = estimator.source_dir
        estimator.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
    else:
        code_dir = 's3://{}/{}'.format(bucket, key)
        estimator.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
        s3_operations['S3Upload'] = [{
            'Path': estimator.source_dir or script,
            'Bucket': bucket,
            'Key': key,
            'Tar': True
        }]
    estimator._hyperparameters[sagemaker.model.DIR_PARAM_NAME] = code_dir
    estimator._hyperparameters[sagemaker.model.SCRIPT_PARAM_NAME] = script
    estimator._hyperparameters[sagemaker.model.CLOUDWATCH_METRICS_PARAM_NAME] = \
        estimator.enable_cloudwatch_metrics
    estimator._hyperparameters[sagemaker.model.CONTAINER_LOG_LEVEL_PARAM_NAME] = estimator.container_log_level
    estimator._hyperparameters[sagemaker.model.JOB_NAME_PARAM_NAME] = estimator._current_job_name
    estimator._hyperparameters[sagemaker.model.SAGEMAKER_REGION_PARAM_NAME] = \
        estimator.sagemaker_session.boto_region_name


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


def training_base_config(estimator, inputs=None, job_name=None, mini_batch_size=None):
    """Export Airflow base training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatorBase):
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
        mini_batch_size (int): Specify this argument only when estimator is a built-in estimator of an
            Amazon algorithm. For other estimators, batch size should be specified in the estimator.

    Returns:
        dict: Training config that can be directly used by SageMakerTrainingOperator in Airflow.
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
        prepare_amazon_algorithm_estimator(estimator, inputs, mini_batch_size)
    job_config = job._Job._load_config(inputs, estimator, expand_role=False, validate_uri=False)

    train_config = {
        'AlgorithmSpecification': {
            'TrainingImage': estimator.train_image(),
            'TrainingInputMode': estimator.input_mode
        },
        'OutputDataConfig': job_config['output_config'],
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

    if s3_operations:
        train_config['S3Operations'] = s3_operations

    return train_config


def training_config(estimator, inputs=None, job_name=None, mini_batch_size=None):
    """Export Airflow training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatorBase):
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
        mini_batch_size (int): Specify this argument only when estimator is a built-in estimator of an
            Amazon algorithm. For other estimators, batch size should be specified in the estimator.

    Returns:
        dict: Training config that can be directly used by SageMakerTrainingOperator in Airflow.
    """

    train_config = training_base_config(estimator, inputs, job_name, mini_batch_size)

    train_config['TrainingJobName'] = estimator._current_job_name

    if estimator.tags is not None:
        train_config['Tags'] = estimator.tags

    return train_config


def tuning_config(tuner, inputs, job_name=None):
    """Export Airflow tuning config from an estimator

    Args:
        tuner (sagemaker.tuner.HyperparameterTuner): The tuner to export tuning config from.
        inputs: Information about the training data. Please refer to the ``fit()`` method of
                the associated estimator in the tuner, as this can take any of the following forms:

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

        job_name (str): Specify a tuning job name if needed.

    Returns:
        dict: Tuning config that can be directly used by SageMakerTuningOperator in Airflow.
    """
    train_config = training_base_config(tuner.estimator, inputs)
    hyperparameters = train_config.pop('HyperParameters', None)
    s3_operations = train_config.pop('S3Operations', None)

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


def prepare_framework_container_def(model, instance_type, s3_operations):
    """Prepare the framework model container information. Specify related S3 operations for Airflow to perform.
    (Upload `source_dir`)

    Args:
        model (sagemaker.model.FrameworkModel): The framework model
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.
        s3_operations (dict): The dict to specify S3 operations (upload `source_dir`).

    Returns:
        dict: The container information of this framework model.
    """
    deploy_image = model.image
    if not deploy_image:
        region_name = model.sagemaker_session.boto_session.region_name
        deploy_image = fw_utils.create_image_uri(
            region_name, model.__framework_name__, instance_type, model.framework_version, model.py_version)

    base_name = utils.base_name_from_image(deploy_image)
    model.name = model.name or utils.airflow_name_from_base(base_name)

    bucket = model.bucket or model.sagemaker_session._default_bucket
    script = os.path.basename(model.entry_point)
    key = '{}/source/sourcedir.tar.gz'.format(model.name)

    if model.source_dir and model.source_dir.lower().startswith('s3://'):
        model.uploaded_code = fw_utils.UploadedCode(s3_prefix=model.source_dir, script_name=script)
    else:
        code_dir = 's3://{}/{}'.format(bucket, key)
        model.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
        s3_operations['S3Upload'] = [{
            'Path': model.source_dir or script,
            'Bucket': bucket,
            'Key': key,
            'Tar': True
        }]

    deploy_env = dict(model.env)
    deploy_env.update(model._framework_env_vars())

    try:
        if model.model_server_workers:
            deploy_env[sagemaker.model.MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(model.model_server_workers)
    except AttributeError:
        # This applies to a FrameworkModel which is not SageMaker Deep Learning Framework Model
        pass

    return sagemaker.container_def(deploy_image, model.model_data, deploy_env)


def model_config(instance_type, model, role=None, image=None):
    """Export Airflow model config from a SageMaker model

    Args:
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'
        model (sagemaker.model.FrameworkModel): The SageMaker model to export Airflow config from
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
        image (str): An container image to use for deploying the model

    Returns:
        dict: Model config that can be directly used by SageMakerModelOperator in Airflow. It can also be part
        of the config used by SageMakerEndpointOperator and SageMakerTransformOperator in Airflow.
    """
    s3_operations = {}
    model.image = image or model.image

    if isinstance(model, sagemaker.model.FrameworkModel):
        container_def = prepare_framework_container_def(model, instance_type, s3_operations)
    else:
        container_def = model.prepare_container_def(instance_type)
        base_name = utils.base_name_from_image(container_def['Image'])
        model.name = model.name or utils.airflow_name_from_base(base_name)

    primary_container = session._expand_container_def(container_def)

    config = {
        'ModelName': model.name,
        'PrimaryContainer': primary_container,
        'ExecutionRoleArn': role or model.role
    }

    if model.vpc_config:
        config['VpcConfig'] = model.vpc_config

    if s3_operations:
        config['S3Operations'] = s3_operations

    return config


def model_config_from_estimator(instance_type, estimator, role=None, image=None, model_server_workers=None,
                                vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
    """Export Airflow model config from a SageMaker estimator

    Args:
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to export Airflow config from.
            It has to be an estimator associated with a training job.
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
        image (str): An container image to use for deploying the model
        model_server_workers (int): The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU. Only effective when estimator is
                SageMaker framework.
        vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on the model.
            Default: use subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.

    Returns:
        dict: Model config that can be directly used by SageMakerModelOperator in Airflow. It can also be part
        of the config used by SageMakerEndpointOperator and SageMakerTransformOperator in Airflow.
    """
    if isinstance(estimator, sagemaker.estimator.Estimator):
        model = estimator.create_model(role=role, image=image, vpc_config_override=vpc_config_override)
    elif isinstance(estimator, sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
        model = estimator.create_model(vpc_config_override=vpc_config_override)
    elif isinstance(estimator, sagemaker.estimator.Framework):
        model = estimator.create_model(model_server_workers=model_server_workers, role=role,
                                       vpc_config_override=vpc_config_override)

    return model_config(instance_type, model, role, image)
