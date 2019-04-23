# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import, print_function

import json
import logging
import os
import re
import sys
import time
import warnings

import boto3
import botocore.config
import six
import yaml
from botocore.exceptions import ClientError

import sagemaker.logs
from sagemaker import vpc_utils
from sagemaker.user_agent import prepend_user_agent
from sagemaker.utils import name_from_image, secondary_training_status_changed, secondary_training_status_message

LOGGER = logging.getLogger('sagemaker')

_STATUS_CODE_TABLE = {
    'COMPLETED': 'Completed',
    'INPROGRESS': 'InProgress',
    'FAILED': 'Failed',
    'STOPPED': 'Stopped',
    'STOPPING': 'Stopping',
    'STARTING': 'Starting'
}


class LogState(object):
    STARTING = 1
    WAIT_IN_PROGRESS = 2
    TAILING = 3
    JOB_COMPLETE = 4
    COMPLETE = 5


class Session(object):
    """Manage interactions with the Amazon SageMaker APIs and any other AWS services needed.

    This class provides convenient methods for manipulating entities and resources that Amazon SageMaker uses,
    such as training jobs, endpoints, and input datasets in S3.

    AWS service calls are delegated to an underlying Boto3 session, which by default
    is initialized using the AWS configuration chain. When you make an Amazon SageMaker API call that
    accesses an S3 bucket location and one is not specified, the ``Session`` creates a default bucket based on
    a naming convention which includes the current AWS account ID.
    """

    def __init__(self, boto_session=None, sagemaker_client=None, sagemaker_runtime_client=None):
        """Initialize a SageMaker ``Session``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service calls
                are delegated to (default: None). If not provided, one is created with default AWS configuration chain.
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker service calls other
                than ``InvokeEndpoint`` (default: None). Estimators created using this ``Session`` use this client.
                If not provided, one will be created using this instance's ``boto_session``.
            sagemaker_runtime_client (boto3.SageMakerRuntime.Client): Client which makes ``InvokeEndpoint``
                calls to Amazon SageMaker (default: None). Predictors created using this ``Session`` use this client.
                If not provided, one will be created using this instance's ``boto_session``.
        """
        self._default_bucket = None

        sagemaker_config_file = os.path.join(os.path.expanduser('~'), '.sagemaker', 'config.yaml')
        if os.path.exists(sagemaker_config_file):
            self.config = yaml.load(open(sagemaker_config_file, 'r'))
        else:
            self.config = None

        self._initialize(boto_session, sagemaker_client, sagemaker_runtime_client)

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        """Initialize this SageMaker Session.

        Creates or uses a boto_session, sagemaker_client and sagemaker_runtime_client.
        Sets the region_name.
        """
        self.boto_session = boto_session or boto3.Session()

        self._region_name = self.boto_session.region_name
        if self._region_name is None:
            raise ValueError('Must setup local AWS configuration with a region supported by SageMaker.')

        self.sagemaker_client = sagemaker_client or self.boto_session.client('sagemaker')
        prepend_user_agent(self.sagemaker_client)

        if sagemaker_runtime_client is not None:
            self.sagemaker_runtime_client = sagemaker_runtime_client
        else:
            config = botocore.config.Config(read_timeout=80)
            self.sagemaker_runtime_client = self.boto_session.client('runtime.sagemaker', config=config)

        prepend_user_agent(self.sagemaker_runtime_client)

        self.local_mode = False

    @property
    def boto_region_name(self):
        return self._region_name

    def upload_data(self, path, bucket=None, key_prefix='data'):
        """Upload local file or directory to S3.

        If a single file is specified for upload, the resulting S3 object key is ``{key_prefix}/{filename}``
        (filename does not include the local path, if any specified).

        If a directory is specified for upload, the API uploads all content, recursively,
        preserving relative structure of subdirectories. The resulting object key names are:
        ``{key_prefix}/{relative_subdirectory_path}/filename``.

        Args:
            path (str): Path (absolute or relative) of local file or directory to upload.
            bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
                default bucket of the ``Session`` is used (if default bucket does not exist, the ``Session``
                creates it).
            key_prefix (str): Optional S3 object key name prefix (default: 'data'). S3 uses the prefix to
                create a directory structure for the bucket content that it display in the S3 console.

        Returns:
            str: The S3 URI of the uploaded file(s). If a file is specified in the path argument, the URI format is:
                ``s3://{bucket name}/{key_prefix}/{original_file_name}``.
                If a directory is specified in the path argument, the URI format is ``s3://{bucket name}/{key_prefix}``.
        """
        # Generate a tuple for each file that we want to upload of the form (local_path, s3_key).
        files = []
        key_suffix = None
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for name in filenames:
                    local_path = os.path.join(dirpath, name)
                    s3_relative_prefix = '' if path == dirpath else os.path.relpath(dirpath, start=path) + '/'
                    s3_key = '{}/{}{}'.format(key_prefix, s3_relative_prefix, name)
                    files.append((local_path, s3_key))
        else:
            _, name = os.path.split(path)
            s3_key = '{}/{}'.format(key_prefix, name)
            files.append((path, s3_key))
            key_suffix = name

        bucket = bucket or self.default_bucket()
        s3 = self.boto_session.resource('s3')

        for local_path, s3_key in files:
            s3.Object(bucket, s3_key).upload_file(local_path)

        s3_uri = 's3://{}/{}'.format(bucket, key_prefix)
        # If a specific file was used as input (instead of a directory), we return the full S3 key
        # of the uploaded object. This prevents unintentionally using other files under the same prefix
        # during training.
        if key_suffix:
            s3_uri = '{}/{}'.format(s3_uri, key_suffix)
        return s3_uri

    def default_bucket(self):
        """Return the name of the default bucket to use in relevant Amazon SageMaker interactions.

        Returns:
            str: The name of the default bucket, which is of the form: ``sagemaker-{region}-{AWS account ID}``.
        """
        if self._default_bucket:
            return self._default_bucket

        account = self.boto_session.client('sts').get_caller_identity()['Account']
        region = self.boto_session.region_name
        default_bucket = 'sagemaker-{}-{}'.format(region, account)

        s3 = self.boto_session.resource('s3')
        try:
            # 'us-east-1' cannot be specified because it is the default region:
            # https://github.com/boto/boto3/issues/125
            if region == 'us-east-1':
                s3.create_bucket(Bucket=default_bucket)
            else:
                s3.create_bucket(Bucket=default_bucket, CreateBucketConfiguration={'LocationConstraint': region})

            LOGGER.info('Created S3 bucket: {}'.format(default_bucket))
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']

            if error_code == 'BucketAlreadyOwnedByYou':
                pass
            elif error_code == 'OperationAborted' and 'conflicting conditional operation' in message:
                # If this bucket is already being concurrently created, we don't need to create it again.
                pass
            elif error_code == 'TooManyBuckets':
                # Succeed if the default bucket exists
                s3.meta.client.head_bucket(Bucket=default_bucket)
            else:
                raise

        self._default_bucket = default_bucket

        return self._default_bucket

    def train(self, input_mode, input_config, role, job_name, output_config,  # noqa: C901
              resource_config, vpc_config, hyperparameters, stop_condition, tags, metric_definitions,
              enable_network_isolation=False, image=None, algorithm_arn=None,
              encrypt_inter_container_traffic=False):
        """Create an Amazon SageMaker training job.

        Args:
            input_mode (str): The input mode that the algorithm supports. Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                    a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

            input_config (list): A list of Channel objects. Each channel is a named input source. Please refer to
                 the format details described:
                 https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            job_name (str): Name of the training job being created.
            output_config (dict): The S3 URI where you want to store the training results and optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:

                * instance_count (int): Number of EC2 instances to use for training.
                    The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
                    The key in resource_config is 'InstanceType'.

            vpc_config (dict): Contains values for VpcConfig:

                * subnets (list[str]): List of subnet ids.
                    The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                    The key in vpc_config is 'SecurityGroupIds'.

            hyperparameters (dict): Hyperparameters for model training. The hyperparameters are made accessible as
                a dict[str, str] to the training code on SageMaker. For convenience, this accepts other types for
                keys and values, but ``str()`` will be called to convert them before training.
            stop_condition (dict): Defines when training shall finish. Contains entries that can be understood by the
                service like ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s) used to evaluate the
                training jobs. Each dictionary contains two keys: 'Name' for the name of the metric, and 'Regex' for
                the regular expression used to extract the metric from the logs.
            enable_network_isolation (bool): Whether to request for the training job to run with
                network isolation or not.
            image (str): Docker image containing training code.
            algorithm_arn (str): Algorithm Arn from Marketplace.
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training containers is
                encrypted for the training job (default: ``False``).

        Returns:
            str: ARN of the training job, if it is created.
        """

        train_request = {
            'AlgorithmSpecification': {
                'TrainingInputMode': input_mode
            },
            'OutputDataConfig': output_config,
            'TrainingJobName': job_name,
            'StoppingCondition': stop_condition,
            'ResourceConfig': resource_config,
            'RoleArn': role,
        }

        if image and algorithm_arn:
            raise ValueError('image and algorithm_arn are mutually exclusive.'
                             'Both were provided: image: %s algorithm_arn: %s' % (image, algorithm_arn))

        if image is None and algorithm_arn is None:
            raise ValueError('either image or algorithm_arn is required. None was provided.')

        if image is not None:
            train_request['AlgorithmSpecification']['TrainingImage'] = image

        if algorithm_arn is not None:
            train_request['AlgorithmSpecification']['AlgorithmName'] = algorithm_arn

        if input_config is not None:
            train_request['InputDataConfig'] = input_config

        if metric_definitions is not None:
            train_request['AlgorithmSpecification']['MetricDefinitions'] = metric_definitions

        if hyperparameters and len(hyperparameters) > 0:
            train_request['HyperParameters'] = hyperparameters

        if tags is not None:
            train_request['Tags'] = tags

        if vpc_config is not None:
            train_request['VpcConfig'] = vpc_config

        if enable_network_isolation:
            train_request['EnableNetworkIsolation'] = enable_network_isolation

        if encrypt_inter_container_traffic:
            train_request['EnableInterContainerTrafficEncryption'] = \
                encrypt_inter_container_traffic

        LOGGER.info('Creating training-job with name: {}'.format(job_name))
        LOGGER.debug('train request: {}'.format(json.dumps(train_request, indent=4)))
        self.sagemaker_client.create_training_job(**train_request)

    def compile_model(self, input_model_config, output_model_config, role,
                      job_name, stop_condition, tags):
        """Create an Amazon SageMaker Neo compilation job.

        Args:
            input_model_config (dict): the trained model and the Amazon S3 location where it is stored.
            output_model_config (dict): Identifies the Amazon S3 location where you want Amazon SageMaker Neo to save
                the results of compilation job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker Neo compilation jobs use this
                role to access model artifacts. You must grant sufficient permissions to this role.
            job_name (str): Name of the compilation job being created.
            stop_condition (dict): Defines when compilation job shall finish. Contains entries that can be understood
                by the service like ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling a compile model job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            str: ARN of the compile model job, if it is created.
        """

        compilation_job_request = {
            'InputConfig': input_model_config,
            'OutputConfig': output_model_config,
            'RoleArn': role,
            'StoppingCondition': stop_condition,
            'CompilationJobName': job_name
        }

        if tags is not None:
            compilation_job_request['Tags'] = tags

        LOGGER.info('Creating compilation-job with name: {}'.format(job_name))
        self.sagemaker_client.create_compilation_job(**compilation_job_request)

    def tune(self, job_name, strategy, objective_type, objective_metric_name,
             max_jobs, max_parallel_jobs, parameter_ranges,
             static_hyperparameters, input_mode, metric_definitions,
             role, input_config, output_config, resource_config, stop_condition, tags,
             warm_start_config, enable_network_isolation=False, image=None, algorithm_arn=None,
             early_stopping_type='Off', encrypt_inter_container_traffic=False, vpc_config=None):
        """Create an Amazon SageMaker hyperparameter tuning job

        Args:
            job_name (str): Name of the tuning job being created.
            strategy (str): Strategy to be used for hyperparameter estimations.
            objective_type (str): The type of the objective metric for evaluating training jobs. This value can be
                either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter tuning job.
            max_parallel_jobs (int): Maximum number of parallel training jobs to start.
            parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can be one of three types:
                 Continuous, Integer, or Categorical.
            static_hyperparameters (dict): Hyperparameters for model training. These hyperparameters remain
                unchanged across all of the training jobs for the hyperparameter tuning job. The hyperparameters are
                made accessible as a dictionary for the training code on SageMaker.
            image (str): Docker image containing training code.
            input_mode (str): The input mode that the algorithm supports. Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                    a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s) used to evaluate the
                training jobs. Each dictionary contains two keys: 'Name' for the name of the metric, and 'Regex' for
                the regular expression used to extract the metric from the logs. This should be defined only for
                jobs that don't use an Amazon algorithm.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            input_config (list): A list of Channel objects. Each channel is a named input source. Please refer to
                 the format details described:
                 https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            output_config (dict): The S3 URI where you want to store the training results and optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:

                * instance_count (int): Number of EC2 instances to use for training.
                    The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
                    The key in resource_config is 'InstanceType'.

            stop_condition (dict): When training should finish, e.g. ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling the tuning job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            warm_start_config (dict): Configuration defining the type of warm start and
                other required configurations.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off'. If set to 'Off', early stopping will not be attempted.
                If set to 'Auto', early stopping of some training jobs may happen, but is not guaranteed to.
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training containers
                is encrypted for the training jobs started for this hyperparameter tuning job (default: ``False``).
            vpc_config (dict): Contains values for VpcConfig (default: None):

                * subnets (list[str]): List of subnet ids.
                    The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                    The key in vpc_config is 'SecurityGroupIds'.

        """
        tune_request = {
            'HyperParameterTuningJobName': job_name,
            'HyperParameterTuningJobConfig': {
                'Strategy': strategy,
                'HyperParameterTuningJobObjective': {
                    'Type': objective_type,
                    'MetricName': objective_metric_name,
                },
                'ResourceLimits': {
                    'MaxNumberOfTrainingJobs': max_jobs,
                    'MaxParallelTrainingJobs': max_parallel_jobs,
                },
                'ParameterRanges': parameter_ranges,
                'TrainingJobEarlyStoppingType': early_stopping_type,
            },
            'TrainingJobDefinition': {
                'StaticHyperParameters': static_hyperparameters,
                'RoleArn': role,
                'OutputDataConfig': output_config,
                'ResourceConfig': resource_config,
                'StoppingCondition': stop_condition,
            }
        }

        algorithm_spec = {
            'TrainingInputMode': input_mode
        }
        if algorithm_arn:
            algorithm_spec['AlgorithmName'] = algorithm_arn
        else:
            algorithm_spec['TrainingImage'] = image

        tune_request['TrainingJobDefinition']['AlgorithmSpecification'] = algorithm_spec

        if input_config is not None:
            tune_request['TrainingJobDefinition']['InputDataConfig'] = input_config

        if warm_start_config:
            tune_request['WarmStartConfig'] = warm_start_config

        if metric_definitions is not None:
            tune_request['TrainingJobDefinition']['AlgorithmSpecification']['MetricDefinitions'] = metric_definitions

        if tags is not None:
            tune_request['Tags'] = tags

        if vpc_config is not None:
            tune_request['TrainingJobDefinition']['VpcConfig'] = vpc_config

        if enable_network_isolation:
            tune_request['TrainingJobDefinition']['EnableNetworkIsolation'] = True

        if encrypt_inter_container_traffic:
            tune_request['TrainingJobDefinition']['EnableInterContainerTrafficEncryption'] = True

        LOGGER.info('Creating hyperparameter tuning job with name: {}'.format(job_name))
        LOGGER.debug('tune request: {}'.format(json.dumps(tune_request, indent=4)))
        self.sagemaker_client.create_hyper_parameter_tuning_job(**tune_request)

    def stop_tuning_job(self, name):
        """Stop the Amazon SageMaker hyperparameter tuning job with the specified name.

        Args:
            name (str): Name of the Amazon SageMaker hyperparameter tuning job.

        Raises:
            ClientError: If an error occurs while trying to stop the hyperparameter tuning job.
        """
        try:
            LOGGER.info('Stopping tuning job: {}'.format(name))
            self.sagemaker_client.stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            # allow to pass if the job already stopped
            if error_code == 'ValidationException':
                LOGGER.info('Tuning job: {} is already stopped or not running.'.format(name))
            else:
                LOGGER.error('Error occurred while attempting to stop tuning job: {}. Please try again.'.format(name))
                raise

    def transform(self, job_name, model_name, strategy, max_concurrent_transforms, max_payload, env,
                  input_config, output_config, resource_config, tags):
        """Create an Amazon SageMaker transform job.

        Args:
            job_name (str): Name of the transform job being created.
            model_name (str): Name of the SageMaker model being used for the transform job.
            strategy (str): The strategy used to decide how to batch records in a single request.
                Possible values are 'MULTI_RECORD' and 'SINGLE_RECORD'.
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the container in MB.
            env (dict): Environment variables to be set for use during the transform job.
            input_config (dict): A dictionary describing the input data (and its location) for the job.
            output_config (dict): A dictionary describing the output location for the job.
            resource_config (dict): A dictionary describing the resources to complete the job.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        """
        transform_request = {
            'TransformJobName': job_name,
            'ModelName': model_name,
            'TransformInput': input_config,
            'TransformOutput': output_config,
            'TransformResources': resource_config,
        }

        if strategy is not None:
            transform_request['BatchStrategy'] = strategy

        if max_concurrent_transforms is not None:
            transform_request['MaxConcurrentTransforms'] = max_concurrent_transforms

        if max_payload is not None:
            transform_request['MaxPayloadInMB'] = max_payload

        if env is not None:
            transform_request['Environment'] = env

        if tags is not None:
            transform_request['Tags'] = tags

        LOGGER.info('Creating transform job with name: {}'.format(job_name))
        LOGGER.debug('Transform request: {}'.format(json.dumps(transform_request, indent=4)))
        self.sagemaker_client.create_transform_job(**transform_request)

    def create_model(self, name, role, container_defs, vpc_config=None,
                     enable_network_isolation=False, primary_container=None,
                     tags=None):
        """Create an Amazon SageMaker ``Model``.
        Specify the S3 location of the model artifacts and Docker image containing
        the inference code. Amazon SageMaker uses this information to deploy the
        model in Amazon SageMaker. This method can also be used to create a Model for an Inference Pipeline
        if you pass the list of container definitions through the containers parameter.

        Args:
            name (str): Name of the Amazon SageMaker ``Model`` to create.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            container_defs (list[dict[str, str]] or [dict[str, str]]): A single container definition or a list of
                container definitions which will be invoked sequentially while performing the prediction. If the list
                contains only one container, then it'll be passed to SageMaker Hosting as the ``PrimaryContainer`` and
                otherwise, it'll be passed as ``Containers``.You can also specify the  return value of
                ``sagemaker.get_container_def()`` or ``sagemaker.pipeline_container_def()``, which will used to
                create more advanced container configurations ,including model containers which need artifacts from S3.
            vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            enable_network_isolation (bool): Wether the model requires network isolation or not.
            primary_container (str or dict[str, str]): Docker image which defines the inference code.
                You can also specify the return value of ``sagemaker.container_def()``, which is used to create
                more advanced container configurations, including model containers which need artifacts from S3. This
                field is deprecated, please use container_defs instead.
            tags(List[dict[str, str]]): Optional. The list of tags to add to the model. Example:
                    >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
                    For more information about tags, see https://boto3.amazonaws.com/v1/documentation\
                    /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags


        Returns:
            str: Name of the Amazon SageMaker ``Model`` created.
        """
        if container_defs and primary_container:
            raise ValueError('Both container_defs and primary_container can not be passed as input')

        if primary_container:
            msg = 'primary_container is going to be deprecated in a future release. Please use container_defs instead.'
            warnings.warn(msg, DeprecationWarning)
            container_defs = primary_container

        role = self.expand_role(role)

        if isinstance(container_defs, list):
            container_definition = container_defs
        else:
            container_definition = _expand_container_def(container_defs)

        create_model_request = _create_model_request(name=name,
                                                     role=role,
                                                     container_def=container_definition,
                                                     tags=tags)

        if vpc_config:
            create_model_request['VpcConfig'] = vpc_config

        if enable_network_isolation:
            create_model_request['EnableNetworkIsolation'] = True

        LOGGER.info('Creating model with name: {}'.format(name))
        LOGGER.debug('CreateModel request: {}'.format(json.dumps(create_model_request, indent=4)))

        try:
            self.sagemaker_client.create_model(**create_model_request)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']

            if error_code == 'ValidationException' and 'Cannot create already existing model' in message:
                LOGGER.warning('Using already existing model: {}'.format(name))
            else:
                raise

        return name

    def create_model_from_job(self, training_job_name, name=None, role=None, primary_container_image=None,
                              model_data_url=None, env=None, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
        """Create an Amazon SageMaker ``Model`` from a SageMaker Training Job.

        Args:
            training_job_name (str): The Amazon SageMaker Training Job name.
            name (str): The name of the SageMaker ``Model`` to create (default: None).
                If not specified, the training job name is used.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, specified either by an IAM role name or
                role ARN. If None, the ``RoleArn`` from the SageMaker Training Job will be used.
            primary_container_image (str): The Docker image reference (default: None). If None, it defaults to
                the Training Image in ``training_job_name``.
            model_data_url (str): S3 location of the model data (default: None). If None, defaults to
                the ``ModelS3Artifacts`` of ``training_job_name``.
            env (dict[string,string]): Model environment variables (default: {}).
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use VpcConfig from training job.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

        Returns:
            str: The name of the created ``Model``.
        """
        training_job = self.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        name = name or training_job_name
        role = role or training_job['RoleArn']
        env = env or {}
        primary_container = container_def(
            primary_container_image or training_job['AlgorithmSpecification']['TrainingImage'],
            model_data_url=model_data_url or training_job['ModelArtifacts']['S3ModelArtifacts'],
            env=env)
        vpc_config = _vpc_config_from_training_job(training_job, vpc_config_override)
        return self.create_model(name, role, primary_container, vpc_config=vpc_config)

    def create_model_package_from_algorithm(self, name, description, algorithm_arn, model_data):
        """Create a SageMaker Model Package from the results of training with an Algorithm Package

        Args:
            name (str): ModelPackage name
            description (str): Model Package description
            algorithm_arn (str): arn or name of the algorithm used for training.
            model_data (str): s3 URI to the model artifacts produced by training
        """
        request = {
            'ModelPackageName': name,
            'ModelPackageDescription': description,
            'SourceAlgorithmSpecification': {
                'SourceAlgorithms': [
                    {
                        'AlgorithmName': algorithm_arn,
                        'ModelDataUrl': model_data
                    }
                ]
            }
        }
        try:
            LOGGER.info('Creating model package with name: {}'.format(name))
            self.sagemaker_client.create_model_package(**request)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']

            if (
                    error_code == 'ValidationException'
                    and 'ModelPackage already exists' in message
            ):
                LOGGER.warning('Using already existing model package: {}'.format(name))
            else:
                raise

    def wait_for_model_package(self, model_package_name, poll=5):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """
        desc = _wait_until(lambda: _create_model_package_status(self.sagemaker_client, model_package_name),
                           poll)
        status = desc['ModelPackageStatus']

        if status != 'Completed':
            reason = desc.get('FailureReason', None)
            raise ValueError('Error creating model package {}: {} Reason: {}'.format(
                model_package_name, status, reason))
        return desc

    def create_endpoint_config(self, name, model_name, initial_instance_count, instance_type,
                               accelerator_type=None, tags=None, kms_key=None):
        """Create an Amazon SageMaker endpoint configuration.

        The endpoint configuration identifies the Amazon SageMaker model (created using the
        ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide this
        endpoint configuration to the ``CreateEndpoint`` API, which then launches the hardware and deploys the model.

        Args:
            name (str): Name of the Amazon SageMaker endpoint configuration to create.
            model_name (str): Name of the Amazon SageMaker ``Model``.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to launch, for example, 'ml.c4.xlarge'.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the instance. For example,
                'ml.eia1.medium'. For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            tags(List[dict[str, str]]): Optional. The list of tags to add to the endpoint config. Example:
                    >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
                    For more information about tags, see https://boto3.amazonaws.com/v1/documentation\
                    /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags


        Returns:
            str: Name of the endpoint point configuration created.
        """
        LOGGER.info('Creating endpoint-config with name {}'.format(name))

        tags = tags or []

        request = {
            'EndpointConfigName': name,
            'ProductionVariants': [
                production_variant(model_name, instance_type, initial_instance_count,
                                   accelerator_type=accelerator_type)
            ],
        }

        if tags is not None:
            request['Tags'] = tags

        if kms_key is not None:
            request['KmsKeyId'] = kms_key

        self.sagemaker_client.create_endpoint_config(**request)
        return name

    def create_endpoint(self, endpoint_name, config_name, tags=None, wait=True):
        """Create an Amazon SageMaker ``Endpoint`` according to the endpoint configuration specified in the request.

        Once the ``Endpoint`` is created, client applications can send requests to obtain inferences.
        The endpoint configuration is created using the ``CreateEndpointConfig`` API.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
            config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` created.
        """
        LOGGER.info('Creating endpoint with name {}'.format(endpoint_name))

        tags = tags or []

        self.sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags)
        if wait:
            self.wait_for_endpoint(endpoint_name)
        return endpoint_name

    def update_endpoint(self, endpoint_name, endpoint_config_name):
        """ Update an Amazon SageMaker ``Endpoint`` according to the endpoint configuration specified in the request

        Raise an error if endpoint with endpoint_name does not exist.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to update.
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` being updated.
        """
        if not _deployment_entity_exists(lambda: self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)):
            raise ValueError('Endpoint with name "{}" does not exist; please use an existing endpoint name'
                             .format(endpoint_name))

        self.sagemaker_client.update_endpoint(EndpointName=endpoint_name,
                                              EndpointConfigName=endpoint_config_name)
        return endpoint_name

    def delete_endpoint(self, endpoint_name):
        """Delete an Amazon SageMaker ``Endpoint``.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to delete.
        """
        LOGGER.info('Deleting endpoint with name: {}'.format(endpoint_name))
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    def delete_endpoint_config(self, endpoint_config_name):
        """Delete an Amazon SageMaker endpoint configuration.

        Args:
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to delete.
        """
        LOGGER.info('Deleting endpoint configuration with name: {}'.format(endpoint_config_name))
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    def delete_model(self, model_name):
        """Delete an Amazon SageMaker Model.

        Args:
            model_name (str): Name of the Amazon SageMaker model to delete.

        """
        LOGGER.info('Deleting model with name: {}'.format(model_name))
        self.sagemaker_client.delete_model(ModelName=model_name)

    def wait_for_job(self, job, poll=5):
        """Wait for an Amazon SageMaker training job to complete.

        Args:
            job (str): Name of the training job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTrainingJob`` API.

        Raises:
            ValueError: If the training job fails.
        """
        desc = _wait_until_training_done(lambda last_desc: _train_done(self.sagemaker_client, job, last_desc),
                                         None, poll)
        self._check_job_status(job, desc, 'TrainingJobStatus')
        return desc

    def wait_for_compilation_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Neo compilation job to complete.

        Args:
            job (str): Name of the compilation job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeCompilationJob`` API.

        Raises:
            ValueError: If the compilation job fails.
        """
        desc = _wait_until(lambda: _compilation_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, 'CompilationJobStatus')
        return desc

    def wait_for_tuning_job(self, job, poll=5):
        """Wait for an Amazon SageMaker hyperparameter tuning job to complete.

        Args:
            job (str): Name of the tuning job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeHyperParameterTuningJob`` API.

        Raises:
            ValueError: If the hyperparameter tuning job fails.
        """
        desc = _wait_until(lambda: _tuning_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, 'HyperParameterTuningJobStatus')
        return desc

    def wait_for_transform_job(self, job, poll=5):
        """Wait for an Amazon SageMaker transform job to complete.

        Args:
            job (str): Name of the transform job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTransformJob`` API.

        Raises:
            ValueError: If the transform job fails.
        """
        desc = _wait_until(lambda: _transform_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, 'TransformJobStatus')
        return desc

    def _check_job_status(self, job, desc, status_key_name):
        """Check to see if the job completed successfully and, if not, construct and
        raise a ValueError.

        Args:
            job (str): The name of the job to check.
            desc (dict[str, str]): The result of ``describe_training_job()``.
            status_key_name (str): Status key name to check for.

        Raises:
            ValueError: If the training job fails.
        """
        status = desc[status_key_name]
        # If the status is capital case, then convert it to Camel case
        status = _STATUS_CODE_TABLE.get(status, status)

        if status != 'Completed' and status != 'Stopped':
            reason = desc.get('FailureReason', '(No reason provided)')
            job_type = status_key_name.replace('JobStatus', ' job')
            raise ValueError('Error for {} {}: {} Reason: {}'.format(job_type, job, status, reason))

    def wait_for_endpoint(self, endpoint, poll=5):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """
        desc = _wait_until(lambda: _deploy_done(self.sagemaker_client, endpoint), poll)
        status = desc['EndpointStatus']

        if status != 'InService':
            reason = desc.get('FailureReason', None)
            raise ValueError('Error hosting endpoint {}: {} Reason: {}'.format(endpoint, status, reason))
        return desc

    def endpoint_from_job(self, job_name, initial_instance_count, instance_type,
                          deployment_image=None, name=None, role=None, wait=True,
                          model_environment_vars=None, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
                          accelerator_type=None):
        """Create an ``Endpoint`` using the results of a successful training job.

        Specify the job name, Docker image containing the inference code, and hardware configuration to deploy
        the model. Internally the API, creates an Amazon SageMaker model (that describes the model artifacts and
        the Docker image containing inference code), endpoint configuration (describing the hardware to deploy
        for hosting the model), and creates an ``Endpoint`` (launches the EC2 instances and deploys the model on them).
        In response, the API returns the endpoint name to which you can send requests for inferences.

        Args:
            job_name (str): Name of the training job to deploy the results of.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                for example, 'ml.c4.xlarge'.
            deployment_image (str): The Docker image which defines the inference code to be used as the entry point for
                accepting prediction requests. If not specified, uses the image used for the training job.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses the training job name.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model container
                (default: None).
            vpc_config_override (dict[str, list[str]]): Overrides VpcConfig set on the model.
                Default: use VpcConfig from training job.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the instance. For example,
                'ml.eia1.medium'. For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

        Returns:
            str: Name of the ``Endpoint`` that is created.
        """
        job_desc = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        output_url = job_desc['ModelArtifacts']['S3ModelArtifacts']
        deployment_image = deployment_image or job_desc['AlgorithmSpecification']['TrainingImage']
        role = role or job_desc['RoleArn']
        name = name or job_name
        vpc_config_override = _vpc_config_from_training_job(job_desc, vpc_config_override)

        return self.endpoint_from_model_data(model_s3_location=output_url, deployment_image=deployment_image,
                                             initial_instance_count=initial_instance_count, instance_type=instance_type,
                                             name=name, role=role, wait=wait,
                                             model_environment_vars=model_environment_vars,
                                             model_vpc_config=vpc_config_override, accelerator_type=accelerator_type)

    def endpoint_from_model_data(self, model_s3_location, deployment_image, initial_instance_count, instance_type,
                                 name=None, role=None, wait=True, model_environment_vars=None, model_vpc_config=None,
                                 accelerator_type=None):
        """Create and deploy to an ``Endpoint`` using existing model data stored in S3.

        Args:
            model_s3_location (str): S3 URI of the model artifacts to use for the endpoint.
            deployment_image (str): The Docker image which defines the runtime code to be used as
                the entry point for accepting prediction requests.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction, e.g. 'ml.c4.xlarge'.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses a name generated by
                combining the image name with a timestamp.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model container
                (default: None).
            model_vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the instance. For example,
                'ml.eia1.medium'. For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

        Returns:
            str: Name of the ``Endpoint`` that is created.
        """

        model_environment_vars = model_environment_vars or {}
        name = name or name_from_image(deployment_image)
        model_vpc_config = vpc_utils.sanitize(model_vpc_config)

        if _deployment_entity_exists(lambda: self.sagemaker_client.describe_endpoint(EndpointName=name)):
            raise ValueError('Endpoint with name "{}" already exists; please pick a different name.'.format(name))

        if not _deployment_entity_exists(lambda: self.sagemaker_client.describe_model(ModelName=name)):
            primary_container = container_def(image=deployment_image,
                                              model_data_url=model_s3_location,
                                              env=model_environment_vars)
            self.create_model(name=name,
                              role=role,
                              container_defs=primary_container,
                              vpc_config=model_vpc_config)

        if not _deployment_entity_exists(
                lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)):
            self.create_endpoint_config(name=name,
                                        model_name=name,
                                        initial_instance_count=initial_instance_count,
                                        instance_type=instance_type,
                                        accelerator_type=accelerator_type)

        self.create_endpoint(endpoint_name=name, config_name=name, wait=wait)
        return name

    def endpoint_from_production_variants(self, name, production_variants, tags=None, kms_key=None, wait=True):
        """Create an SageMaker ``Endpoint`` from a list of production variants.

        Args:
            name (str): The name of the ``Endpoint`` to create.
            production_variants (list[dict[str, str]]): The list of production variants to deploy.
            tags (list[dict[str, str]]): A list of key-value pairs for tagging the endpoint (default: None).
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume attached
                to the instance hosting the endpoint.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).

        Returns:
            str: The name of the created ``Endpoint``.
        """

        if not _deployment_entity_exists(
                lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)):
            config_options = {'EndpointConfigName': name, 'ProductionVariants': production_variants}
            if tags:
                config_options['Tags'] = tags
            if kms_key:
                config_options['KmsKeyId'] = kms_key

            self.sagemaker_client.create_endpoint_config(**config_options)
        return self.create_endpoint(endpoint_name=name, config_name=name, tags=tags, wait=wait)

    def expand_role(self, role):
        """Expand an IAM role name into an ARN.

        If the role is already in the form of an ARN, then the role is simply returned. Otherwise we retrieve the full
        ARN and return it.

        Args:
            role (str): An AWS IAM role (either name or full ARN).

        Returns:
            str: The corresponding AWS IAM role ARN.
        """
        if '/' in role:
            return role
        else:
            return self.boto_session.resource('iam').Role(role).arn

    def get_caller_identity_arn(self):
        """Returns the ARN user or role whose credentials are used to call the API.
        Returns:
            (str): The ARN user or role
        """
        assumed_role = self.boto_session.client('sts').get_caller_identity()['Arn']

        if 'AmazonSageMaker-ExecutionRole' in assumed_role:
            role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/service-role/\3', assumed_role)
            return role

        role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/\3', assumed_role)

        # Call IAM to get the role's path
        role_name = role[role.rfind('/') + 1:]
        try:
            role = self.boto_session.client('iam').get_role(RoleName=role_name)['Role']['Arn']
        except ClientError:
            LOGGER.warning("Couldn't call 'get_role' to get Role ARN from role name {} to get Role path."
                           .format(role_name))

        return role

    def logs_for_job(self, job_name, wait=False, poll=10):  # noqa: C901 - suppress complexity warning for this method
        """Display the logs for a given training job, optionally tailing them until the
        job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the training job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes (default: False).
            poll (int): The interval in seconds between polling for new log entries and job completion (default: 5).

        Raises:
            ValueError: If waiting and the training job fails.
        """

        description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        print(secondary_training_status_message(description, None), end='')
        instance_count = description['ResourceConfig']['InstanceCount']
        status = description['TrainingJobStatus']

        stream_names = []  # The list of log streams
        positions = {}     # The current position in each stream, map of stream name -> position

        # Increase retries allowed (from default of 4), as we don't want waiting for a training job
        # to be interrupted by a transient exception.
        config = botocore.config.Config(retries={'max_attempts': 15})
        client = self.boto_session.client('logs', config=config)
        log_group = '/aws/sagemaker/TrainingJobs'

        job_already_completed = True if status == 'Completed' or status == 'Failed' or status == 'Stopped' else False

        state = LogState.TAILING if wait and not job_already_completed else LogState.COMPLETE
        dot = False

        color_wrap = sagemaker.logs.ColorWrap()

        # The loop below implements a state machine that alternates between checking the job status and
        # reading whatever is available in the logs at this point. Note, that if we were called with
        # wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is complete).
        #
        # The state table:
        #
        # STATE               ACTIONS                        CONDITION             NEW STATE
        # ----------------    ----------------               -----------------     ----------------
        # TAILING             Read logs, Pause, Get status   Job complete          JOB_COMPLETE
        #                                                    Else                  TAILING
        # JOB_COMPLETE        Read logs, Pause               Any                   COMPLETE
        # COMPLETE            Read logs, Exit                                      N/A
        #
        # Notes:
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to Cloudwatch after
        #   the job was marked complete.
        last_describe_job_call = time.time()
        last_description = description
        while True:
            if len(stream_names) < instance_count:
                # Log streams are created whenever a container starts writing to stdout/err, so this list
                # may be dynamic until we have a stream for every instance.
                try:
                    streams = client.describe_log_streams(logGroupName=log_group, logStreamNamePrefix=job_name + '/',
                                                          orderBy='LogStreamName', limit=instance_count)
                    stream_names = [s['logStreamName'] for s in streams['logStreams']]
                    positions.update([(s, sagemaker.logs.Position(timestamp=0, skip=0))
                                      for s in stream_names if s not in positions])
                except ClientError as e:
                    # On the very first training job run on an account, there's no log group until
                    # the container starts logging, so ignore any errors thrown about that
                    err = e.response.get('Error', {})
                    if err.get('Code', None) != 'ResourceNotFoundException':
                        raise

            if len(stream_names) > 0:
                if dot:
                    print('')
                    dot = False
                for idx, event in sagemaker.logs.multi_stream_iter(client, log_group, stream_names, positions):
                    color_wrap(idx, event['message'])
                    ts, count = positions[stream_names[idx]]
                    if event['timestamp'] == ts:
                        positions[stream_names[idx]] = sagemaker.logs.Position(timestamp=ts, skip=count + 1)
                    else:
                        positions[stream_names[idx]] = sagemaker.logs.Position(timestamp=event['timestamp'], skip=1)
            else:
                dot = True
                print('.', end='')
                sys.stdout.flush()
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
                last_describe_job_call = time.time()

                if secondary_training_status_changed(description, last_description):
                    print()
                    print(secondary_training_status_message(description, last_description), end='')
                    last_description = description

                status = description['TrainingJobStatus']

                if status == 'Completed' or status == 'Failed' or status == 'Stopped':
                    print()
                    state = LogState.JOB_COMPLETE

        if wait:
            self._check_job_status(job_name, description, 'TrainingJobStatus')
            if dot:
                print()
            # Customers are not billed for hardware provisioning, so billable time is less than total time
            billable_time = (description['TrainingEndTime'] - description['TrainingStartTime']) * instance_count
            print('Billable seconds:', int(billable_time.total_seconds()) + 1)


def container_def(image, model_data_url=None, env=None):
    """Create a definition for executing a container as part of a SageMaker model.

    Args:
        image (str): Docker image to run for this container.
        model_data_url (str): S3 URI of data required by this container,
            e.g. SageMaker training job model artifacts (default: None).
        env (dict[str, str]): Environment variables to set inside the container (default: None).
    Returns:
        dict[str, str]: A complete container definition object usable with the CreateModel API if passed via
        `PrimaryContainers` field.
    """
    if env is None:
        env = {}
    c_def = {'Image': image, 'Environment': env}
    if model_data_url:
        c_def['ModelDataUrl'] = model_data_url
    return c_def


def pipeline_container_def(models, instance_type=None):
    """
    Create a definition for executing a pipeline of containers as part of a SageMaker model.
    Args:
        models (list[sagemaker.Model]): this will be a list of ``sagemaker.Model`` objects in the order the inference
        should be invoked.
        instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge' (default: None).
    Returns:
        list[dict[str, str]]: list of container definition objects usable with with the CreateModel API for inference
        pipelines if passed via `Containers` field.
    """
    c_defs = []  # should contain list of container definitions in the same order customer passed
    for model in models:
        c_defs.append(model.prepare_container_def(instance_type))
    return c_defs


def production_variant(model_name, instance_type, initial_instance_count=1, variant_name='AllTraffic',
                       initial_weight=1, accelerator_type=None):
    """Create a production variant description suitable for use in a ``ProductionVariant`` list as part of a
    ``CreateEndpointConfig`` request.

    Args:
        model_name (str): The name of the SageMaker model this production variant references.
        instance_type (str): The EC2 instance type for this production variant. For example, 'ml.c4.8xlarge'.
        initial_instance_count (int): The initial instance count for this production variant (default: 1).
        variant_name (string): The ``VariantName`` of this production variant (default: 'AllTraffic').
        initial_weight (int): The relative ``InitialVariantWeight`` of this production variant (default: 1).
        accelerator_type (str): Type of Elastic Inference accelerator for this production variant. For example,
            'ml.eia1.medium'. For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

    Returns:
        dict[str, str]: An SageMaker ``ProductionVariant`` description
    """
    production_variant_configuration = {
        'ModelName': model_name,
        'InstanceType': instance_type,
        'InitialInstanceCount': initial_instance_count,
        'VariantName': variant_name,
        'InitialVariantWeight': initial_weight
    }

    if accelerator_type:
        production_variant_configuration['AcceleratorType'] = accelerator_type

    return production_variant_configuration


def get_execution_role(sagemaker_session=None):
    """Return the role ARN whose credentials are used to call the API.
    Throws an exception if
    Args:
        sagemaker_session(Session): Current sagemaker session
    Returns:
        (str): The role ARN
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    arn = sagemaker_session.get_caller_identity_arn()

    if ':role/' in arn:
        return arn
    message = 'The current AWS identity is not a role: {}, therefore it cannot be used as a SageMaker execution role'
    raise ValueError(message.format(arn))


class s3_input(object):
    """Amazon SageMaker channel configurations for S3 data sources.

    Attributes:
        config (dict[str, dict]): A SageMaker ``DataSource`` referencing a SageMaker ``S3DataSource``.
    """

    def __init__(self, s3_data, distribution='FullyReplicated', compression=None,
                 content_type=None, record_wrapping=None, s3_data_type='S3Prefix',
                 input_mode=None, attribute_names=None, shuffle_config=None):
        """Create a definition for input data used by an SageMaker training job.

        See AWS documentation on the ``CreateTrainingJob`` API for more details on the parameters.

        Args:
            s3_data (str): Defines the location of s3 data to train on.
            distribution (str): Valid values: 'FullyReplicated', 'ShardedByS3Key'
                (default: 'FullyReplicated').
            compression (str): Valid values: 'Gzip', None (default: None). This is used only in Pipe input mode.
            content_type (str): MIME type of the input data (default: None).
            record_wrapping (str): Valid values: 'RecordIO' (default: None).
            s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile', 'AugmentedManifestFile'. If 'S3Prefix',
                ``s3_data`` defines a prefix of s3 objects to train on. All objects with s3 keys beginning with
                ``s3_data`` will be used to train. If 'ManifestFile' or 'AugmentedManifestFile', then ``s3_data``
                defines a single s3 manifest file or augmented manifest file (respectively), listing the s3 data to
                train on. Both the ManifestFile and AugmentedManifestFile formats are described in the SageMaker API
                documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/API_S3DataSource.html
            input_mode (str): Optional override for this channel's input mode (default: None). By default, channels will
                use the input mode defined on ``sagemaker.estimator.EstimatorBase.input_mode``, but they will ignore
                that setting if this parameter is set.

                    * None - Amazon SageMaker will use the input mode specified in the ``Estimator``.
                    * 'File' - Amazon SageMaker copies the training dataset from the S3 location to a local directory.
                    * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

            attribute_names (list[str]): A list of one or more attribute names to use that are found in a specified
                AugmentedManifestFile.
            shuffle_config (ShuffleConfig): If specified this configuration enables shuffling on this channel. See the
                SageMaker API documentation for more info:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
        """

        self.config = {
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': distribution,
                    'S3DataType': s3_data_type,
                    'S3Uri': s3_data
                }
            }
        }

        if compression is not None:
            self.config['CompressionType'] = compression
        if content_type is not None:
            self.config['ContentType'] = content_type
        if record_wrapping is not None:
            self.config['RecordWrapperType'] = record_wrapping
        if input_mode is not None:
            self.config['InputMode'] = input_mode
        if attribute_names is not None:
            self.config['DataSource']['S3DataSource']['AttributeNames'] = attribute_names
        if shuffle_config is not None:
            self.config['ShuffleConfig'] = {'Seed': shuffle_config.seed}


class ShuffleConfig(object):
    """
    Used to configure channel shuffling using a seed. See SageMaker
    documentation for more detail: https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
    """
    def __init__(self, seed):
        """
        Create a ShuffleConfig.
        Args:
            seed (long): the long value used to seed the shuffled sequence.
        """
        self.seed = seed


class ModelContainer(object):
    """
    Amazon SageMaker Model configurations for inference pipelines.
    Attributes:
        model_data (str): S3 Model artifact location
        image (str): Docker image URL in ECR
        env (dict[str,str]): Environment variable mapping
    """

    def __init__(self, model_data, image, env=None):
        """
        Create a definition of a model which can be part of an Inference Pipeline
        Args:
            model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
            image (str): A Docker image URI.
            env (dict[str, str]): Environment variables to run with ``image`` when hosted in SageMaker (default: None).
        """
        self.model_data = model_data
        self.image = image
        self.env = env


def _create_model_request(name, role, container_def=None, tags=None):  # pylint: disable=redefined-outer-name
    request = {'ModelName': name, 'ExecutionRoleArn': role}

    if isinstance(container_def, list):
        request['Containers'] = container_def
    else:
        request['PrimaryContainer'] = container_def

    if tags:
        request['Tags'] = tags

    return request


def _deployment_entity_exists(describe_fn):
    try:
        describe_fn()
        return True
    except ClientError as ce:
        error_code = ce.response['Error']['Code']
        if not (error_code == 'ValidationException' and 'Could not find' in ce.response['Error']['Message']):
            raise ce
        return False


def _train_done(sagemaker_client, job_name, last_desc):
    in_progress_statuses = ['InProgress', 'Created']

    desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    status = desc['TrainingJobStatus']

    if secondary_training_status_changed(desc, last_desc):
        print()
        print(secondary_training_status_message(desc, last_desc), end='')
    else:
        print('.', end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return desc, False

    print()
    return desc, True


def _compilation_job_status(sagemaker_client, job_name):
    compile_status_codes = {
        'Completed': '!',
        'InProgress': '.',
        'Failed': '*',
        'Stopped': 's',
        'Stopping': '_'
    }
    in_progress_statuses = ['InProgress', 'Stopping', 'Starting']

    desc = sagemaker_client.describe_compilation_job(CompilationJobName=job_name)
    status = desc['CompilationJobStatus']

    status = _STATUS_CODE_TABLE.get(status, status)
    print(compile_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    return desc


def _tuning_job_status(sagemaker_client, job_name):
    tuning_status_codes = {
        'Completed': '!',
        'InProgress': '.',
        'Failed': '*',
        'Stopped': 's',
        'Stopping': '_'
    }
    in_progress_statuses = ['InProgress', 'Stopping']

    desc = sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
    status = desc['HyperParameterTuningJobStatus']

    print(tuning_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print('')
    return desc


def _transform_job_status(sagemaker_client, job_name):
    transform_job_status_codes = {
        'Completed': '!',
        'InProgress': '.',
        'Failed': '*',
        'Stopped': 's',
        'Stopping': '_'
    }
    in_progress_statuses = ['InProgress', 'Stopping']

    desc = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    status = desc['TransformJobStatus']

    print(transform_job_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print('')
    return desc


def _create_model_package_status(sagemaker_client, model_package_name):
    in_progress_statuses = ['InProgress', 'Pending']

    desc = sagemaker_client.describe_model_package(ModelPackageName=model_package_name)
    status = desc['ModelPackageStatus']
    print('.', end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print('')
    return desc


def _deploy_done(sagemaker_client, endpoint_name):
    hosting_status_codes = {
        "OutOfService": "x",
        "Creating": "-",
        "Updating": "-",
        "InService": "!",
        "RollingBack": "<",
        "Deleting": "o",
        "Failed": "*"
    }
    in_progress_statuses = ['Creating', 'Updating']

    desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = desc['EndpointStatus']

    print(hosting_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    return None if status in in_progress_statuses else desc


def _wait_until_training_done(callable_fn, desc, poll=5):
    job_desc, finished = callable_fn(desc)
    while not finished:
        time.sleep(poll)
        job_desc, finished = callable_fn(job_desc)
    return job_desc


def _wait_until(callable_fn, poll=5):
    result = callable_fn()
    while result is None:
        time.sleep(poll)
        result = callable_fn()
    return result


def _expand_container_def(c_def):
    if isinstance(c_def, six.string_types):
        return container_def(c_def)
    return c_def


def _vpc_config_from_training_job(training_job_desc, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
    if vpc_config_override is vpc_utils.VPC_CONFIG_DEFAULT:
        return training_job_desc.get(vpc_utils.VPC_CONFIG_KEY)
    else:
        return vpc_utils.sanitize(vpc_config_override)
