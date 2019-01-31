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
import re

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
    if estimator.code_location is not None:
        bucket, key = fw_utils.parse_s3_url(estimator.code_location)
        key = os.path.join(key, estimator._current_job_name, 'source', 'sourcedir.tar.gz')
    else:
        bucket = estimator.sagemaker_session._default_bucket
        key = os.path.join(estimator._current_job_name, 'source', 'sourcedir.tar.gz')
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
        estimator._current_job_name = utils.name_from_base(base_name)

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
        tuner._current_job_name = utils.name_from_base(base_name, tuner.TUNING_JOB_NAME_MAX_LENGTH, True)

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


def update_submit_s3_uri(estimator, job_name):
    """Updated the S3 URI of the framework source directory in given estimator.

    Args:
        estimator (sagemaker.estimator.Framework): The Framework estimator to update.
        job_name (str): The new job name included in the submit S3 URI

    Returns:
        str: The updated S3 URI of framework source directory
    """
    if estimator.uploaded_code is None:
        return

    pattern = r'(?<=/)[^/]+?(?=/source/sourcedir.tar.gz)'

    # update the S3 URI with the latest training job.
    # s3://path/old_job/source/sourcedir.tar.gz will become s3://path/new_job/source/sourcedir.tar.gz
    submit_uri = estimator.uploaded_code.s3_prefix
    submit_uri = re.sub(pattern, job_name, submit_uri)
    script_name = estimator.uploaded_code.script_name
    estimator.uploaded_code = fw_utils.UploadedCode(submit_uri, script_name)


def update_estimator_from_task(estimator, task_id, task_type):
    """Update training job of the estimator from a task in the DAG

    Args:
        estimator (sagemaker.estimator.EstimatorBase): The estimator to update
        task_id (str): The task id of any airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates training jobs in the DAG.
        task_type (str): Whether the task is from SageMakerTrainingOperator or SageMakerTuningOperator. Values can be
            'training', 'tuning' or None (which means training job is not from any task).
    """
    if task_type is None:
        return
    if task_type.lower() == 'training':
        training_job = "{{ ti.xcom_pull(task_ids='%s')['Training']['TrainingJobName'] }}" % task_id
        job_name = training_job
    elif task_type.lower() == 'tuning':
        training_job = "{{ ti.xcom_pull(task_ids='%s')['Tuning']['BestTrainingJob']['TrainingJobName'] }}" % task_id
        # need to strip the double quotes in json to get the string
        job_name = "{{ ti.xcom_pull(task_ids='%s')['Tuning']['TrainingJobDefinition']['StaticHyperParameters']" \
                   "['sagemaker_job_name'].strip('%s') }}" % (task_id, '"')
    else:
        raise ValueError("task_type must be either 'training', 'tuning' or None.")
    estimator._current_job_name = training_job
    if isinstance(estimator, sagemaker.estimator.Framework):
        update_submit_s3_uri(estimator, job_name)


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
    model.name = model.name or utils.name_from_base(base_name)

    bucket = model.bucket or model.sagemaker_session._default_bucket
    script = os.path.basename(model.entry_point)
    key = '{}/source/sourcedir.tar.gz'.format(model.name)

    if model.source_dir and model.source_dir.lower().startswith('s3://'):
        code_dir = model.source_dir
        model.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
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
        model.name = model.name or utils.name_from_base(base_name)

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


def model_config_from_estimator(instance_type, estimator, task_id, task_type, role=None, image=None, name=None,
                                model_server_workers=None, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
    """Export Airflow model config from a SageMaker estimator

    Args:
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to export Airflow config from.
            It has to be an estimator associated with a training job.
        task_id (str): The task id of any airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates training jobs in the DAG. The model config
            is built based on the training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or SageMakerTuningOperator. Values can be
            'training', 'tuning' or None (which means training job is not from any task).
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
        image (str): An container image to use for deploying the model
        name (str): Name of the model
        model_server_workers (int): The number of worker processes used by the inference server.
            If None, server will use one worker per vCPU. Only effective when estimator is a
            SageMaker framework.
        vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on the model.
            Default: use subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.

    Returns:
        dict: Model config that can be directly used by SageMakerModelOperator in Airflow. It can also be part
            of the config used by SageMakerEndpointOperator in Airflow.
    """
    update_estimator_from_task(estimator, task_id, task_type)
    if isinstance(estimator, sagemaker.estimator.Estimator):
        model = estimator.create_model(role=role, image=image, vpc_config_override=vpc_config_override)
    elif isinstance(estimator, sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
        model = estimator.create_model(vpc_config_override=vpc_config_override)
    elif isinstance(estimator, sagemaker.estimator.Framework):
        model = estimator.create_model(model_server_workers=model_server_workers, role=role,
                                       vpc_config_override=vpc_config_override)
    else:
        raise TypeError('Estimator must be one of sagemaker.estimator.Estimator, sagemaker.estimator.Framework'
                        ' or sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.')
    model.name = name

    return model_config(instance_type, model, role, image)


def transform_config(transformer, data, data_type='S3Prefix', content_type=None, compression_type=None,
                     split_type=None, job_name=None):
    """Export Airflow transform config from a SageMaker transformer

    Args:
        transformer (sagemaker.transformer.Transformer): The SageMaker transformer to export Airflow
            config from.
        data (str): Input data location in S3.
        data_type (str): What the S3 location defines (default: 'S3Prefix'). Valid values:

            * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will be used as
                inputs for the transform job.
            * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object to use as
                an input for the transform job.

        content_type (str): MIME type of the input data (default: None).
        compression_type (str): Compression type of the input data, if compressed (default: None).
            Valid values: 'Gzip', None.
        split_type (str): The record delimiter for the input object (default: 'None').
            Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
        job_name (str): job name (default: None). If not specified, one will be generated.

    Returns:
        dict: Transform config that can be directly used by SageMakerTransformOperator in Airflow.
    """
    if job_name is not None:
        transformer._current_job_name = job_name
    else:
        base_name = transformer.base_transform_job_name
        transformer._current_job_name = utils.name_from_base(base_name) \
            if base_name is not None else transformer.model_name

    if transformer.output_path is None:
        transformer.output_path = 's3://{}/{}'.format(
            transformer.sagemaker_session.default_bucket(), transformer._current_job_name)

    job_config = sagemaker.transformer._TransformJob._load_config(
        data, data_type, content_type, compression_type, split_type, transformer)

    config = {
        'TransformJobName': transformer._current_job_name,
        'ModelName': transformer.model_name,
        'TransformInput': job_config['input_config'],
        'TransformOutput': job_config['output_config'],
        'TransformResources': job_config['resource_config'],
    }

    if transformer.strategy is not None:
        config['BatchStrategy'] = transformer.strategy

    if transformer.max_concurrent_transforms is not None:
        config['MaxConcurrentTransforms'] = transformer.max_concurrent_transforms

    if transformer.max_payload is not None:
        config['MaxPayloadInMB'] = transformer.max_payload

    if transformer.env is not None:
        config['Environment'] = transformer.env

    if transformer.tags is not None:
        config['Tags'] = transformer.tags

    return config


def transform_config_from_estimator(estimator, task_id, task_type, instance_count, instance_type, data,
                                    data_type='S3Prefix', content_type=None, compression_type=None, split_type=None,
                                    job_name=None, model_name=None, strategy=None, assemble_with=None, output_path=None,
                                    output_kms_key=None, accept=None, env=None, max_concurrent_transforms=None,
                                    max_payload=None, tags=None, role=None, volume_kms_key=None,
                                    model_server_workers=None, image=None, vpc_config_override=None):
    """Export Airflow transform config from a SageMaker estimator

    Args:
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to export Airflow config from.
            It has to be an estimator associated with a training job.
        task_id (str): The task id of any airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates training jobs in the DAG. The transform
            config is built based on the training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or SageMakerTuningOperator. Values can be
            'training', 'tuning' or None (which means training job is not from any task).
        instance_count (int): Number of EC2 instances to use.
        instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
        data (str): Input data location in S3.
        data_type (str): What the S3 location defines (default: 'S3Prefix'). Valid values:

            * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will be used as
                inputs for the transform job.
            * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object to use as
                an input for the transform job.

        content_type (str): MIME type of the input data (default: None).
        compression_type (str): Compression type of the input data, if compressed (default: None).
            Valid values: 'Gzip', None.
        split_type (str): The record delimiter for the input object (default: 'None').
            Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
        job_name (str): transform job name (default: None). If not specified, one will be generated.
        model_name (str): model name (default: None). If not specified, one will be generated.
        strategy (str): The strategy used to decide how to batch records in a single request (default: None).
            Valid values: 'MULTI_RECORD' and 'SINGLE_RECORD'.
        assemble_with (str): How the output is assembled (default: None). Valid values: 'Line' or 'None'.
        output_path (str): S3 location for saving the transform result. If not specified, results are stored to
            a default bucket.
        output_kms_key (str): Optional. KMS key ID for encrypting the transform output (default: None).
        accept (str): The content type accepted by the endpoint deployed during the transform job.
        env (dict): Environment variables to be set for use during the transform job (default: None).
        max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
            each individual transform container at one time.
        max_payload (int): Maximum size of the payload in a single HTTP request to the container in MB.
        tags (list[dict]): List of tags for labeling a transform job. If none specified, then the tags used for
            the training job are used for the transform job.
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
            transform jobs. If not specified, the role from the Estimator will be used.
        volume_kms_key (str): Optional. KMS key ID for encrypting the volume attached to the ML
            compute instance (default: None).
        model_server_workers (int): Optional. The number of worker processes used by the inference server.
            If None, server will use one worker per vCPU.
        image (str): An container image to use for deploying the model
        vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on the model.
            Default: use subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.

    Returns:
        dict: Transform config that can be directly used by SageMakerTransformOperator in Airflow.
    """
    model_base_config = model_config_from_estimator(instance_type=instance_type, estimator=estimator, task_id=task_id,
                                                    task_type=task_type, role=role, image=image, name=model_name,
                                                    model_server_workers=model_server_workers,
                                                    vpc_config_override=vpc_config_override)

    if isinstance(estimator, sagemaker.estimator.Framework):
        transformer = estimator.transformer(instance_count, instance_type, strategy, assemble_with, output_path,
                                            output_kms_key, accept, env, max_concurrent_transforms,
                                            max_payload, tags, role, model_server_workers, volume_kms_key)
    else:
        transformer = estimator.transformer(instance_count, instance_type, strategy, assemble_with, output_path,
                                            output_kms_key, accept, env, max_concurrent_transforms,
                                            max_payload, tags, role, volume_kms_key)
    transformer.model_name = model_base_config['ModelName']

    transform_base_config = transform_config(transformer, data, data_type, content_type, compression_type,
                                             split_type, job_name)

    config = {
        'Model': model_base_config,
        'Transform': transform_base_config
    }

    return config


def deploy_config(model, initial_instance_count, instance_type, endpoint_name=None, tags=None):
    """Export Airflow deploy config from a SageMaker model

    Args:
        model (sagemaker.model.Model): The SageMaker model to export the Airflow config from.
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.
        initial_instance_count (int): The initial number of instances to run in the
            ``Endpoint`` created from this ``Model``.
        endpoint_name (str): The name of the endpoint to create (default: None).
            If not specified, a unique endpoint name will be created.
        tags (list[dict]): List of tags for labeling a training job. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

    Returns:
        dict: Deploy config that can be directly used by SageMakerEndpointOperator in Airflow.

    """
    model_base_config = model_config(instance_type, model)

    production_variant = sagemaker.production_variant(model.name, instance_type, initial_instance_count)
    name = model.name
    config_options = {'EndpointConfigName': name, 'ProductionVariants': [production_variant]}
    if tags is not None:
        config_options['Tags'] = tags

    endpoint_name = endpoint_name or name
    endpoint_base_config = {
        'EndpointName': endpoint_name,
        'EndpointConfigName': name
    }

    config = {
        'Model': model_base_config,
        'EndpointConfig': config_options,
        'Endpoint': endpoint_base_config
    }

    # if there is s3 operations needed for model, move it to root level of config
    s3_operations = model_base_config.pop('S3Operations', None)
    if s3_operations is not None:
        config['S3Operations'] = s3_operations

    return config


def deploy_config_from_estimator(estimator, task_id, task_type, initial_instance_count, instance_type,
                                 model_name=None, endpoint_name=None, tags=None, **kwargs):
    """Export Airflow deploy config from a SageMaker estimator

    Args:
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to export Airflow config from.
            It has to be an estimator associated with a training job.
        task_id (str): The task id of any airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates training jobs in the DAG. The endpoint
            config is built based on the training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or SageMakerTuningOperator. Values can be
            'training', 'tuning' or None (which means training job is not from any task).
        initial_instance_count (int): Minimum number of EC2 instances to deploy to an endpoint for prediction.
        instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
            for example, 'ml.c4.xlarge'.
        model_name (str): Name to use for creating an Amazon SageMaker model. If not specified, one will be generated.
        endpoint_name (str): Name to use for creating an Amazon SageMaker endpoint. If not specified, the name of
            the SageMaker model is used.
        tags (list[dict]): List of tags for labeling a training job. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        **kwargs: Passed to invocation of ``create_model()``. Implementations may customize
            ``create_model()`` to accept ``**kwargs`` to customize model creation during deploy.
            For more, see the implementation docs.

    Returns:
        dict: Deploy config that can be directly used by SageMakerEndpointOperator in Airflow.
    """
    update_estimator_from_task(estimator, task_id, task_type)
    model = estimator.create_model(**kwargs)
    model.name = model_name
    config = deploy_config(model, initial_instance_count, instance_type, endpoint_name, tags)
    return config
